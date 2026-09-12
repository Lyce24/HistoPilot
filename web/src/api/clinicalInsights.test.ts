import { afterEach, describe, expect, it, vi } from 'vitest';
import { clinicalAnalyses, type ClinicalSelection } from './clinicalUtility';
import { interpretations } from './interpretation';
import { downloadArtifact, fetchArtifactBlob, request } from './client';
vi.mock('./client', () => ({ request: vi.fn(), downloadArtifact: vi.fn(), fetchArtifactBlob: vi.fn() }));
afterEach(() => vi.clearAllMocks());
describe('clinical and interpretation API contracts', () => {
  it('publishes the reviewed clinical selection with the same hash and idempotency operation', async () => {
    const selection: ClinicalSelection = { evaluationId: 'eval', name: 'External utility', unit: 'patient', positiveClass: 'high', threshold: null, bins: 10, thresholdMin: .05, thresholdMax: .5, thresholdSteps: 91 };
    await clinicalAnalyses.save('project/1', selection, 'review-hash', 'operation');
    expect(request).toHaveBeenCalledWith('/projects/project%2F1/clinical-analyses', { method: 'POST', body: JSON.stringify({ ...selection, previewHash: 'review-hash', operationId: 'operation' }) });
  });
  it('requests attention for an exact member and bounded integer viewport with cancellation', async () => {
    const controller = new AbortController();
    await interpretations.attention('p', 'study', 'Slide A', '2', { x: 10, y: 20, width: 300, height: 400 }, 10000, 10000, controller.signal);
    const [url, options] = vi.mocked(request).mock.calls[0];
    expect(url).toContain('/slides/Slide%20A/attention?');
    const query = new URLSearchParams(url.split('?')[1]);
    expect(Object.fromEntries(query)).toEqual({ member: '2', offset: '10000', limit: '10000', x: '10', y: '20', width: '300', height: '400' });
    expect(options?.signal).toBe(controller.signal);
  });
  it('loads slide images through authenticated binary requests and exports full saved maps', async () => {
    await interpretations.thumbnail('p', 'study', 'S1');
    expect(fetchArtifactBlob).toHaveBeenCalledWith('/projects/p/interpretations/study/slides/S1/thumbnail?max_size=1536', undefined);
    await interpretations.download('p', 'study', 'slide-0.json');
    expect(downloadArtifact).toHaveBeenCalledWith('/projects/p/interpretations/study/artifacts/slide-0.json', 'slide-0.json');
  });
  it('scopes gallery search and packed batch execution to the same frozen source', async () => {
    const source = { predictorId: 'ensemble', featureBundleId: 'bundle', packArtifactId: 'pack', slideFolder: '/slides' };
    const controller = new AbortController();
    await interpretations.gallery('p', source, 'case 1', 24, controller.signal);
    expect(request).toHaveBeenCalledWith('/projects/p/interpretations/gallery', { method: 'POST', body: JSON.stringify({ ...source, search: 'case 1', offset: 24, limit: 24 }), signal: controller.signal });
    await interpretations.visualize('p', { ...source, slidePaths: ['/slides/a.svs', '/slides/b.svs'] }, 'unchanged-operation');
    expect(request).toHaveBeenCalledWith('/projects/p/interpretations/visualize', { method: 'POST', body: JSON.stringify({ ...source, slidePaths: ['/slides/a.svs', '/slides/b.svs'], operationId: 'unchanged-operation' }) });
  });
  it('requests global top attention independently of any viewport and crops by verified patch identity', async () => {
    const controller = new AbortController();
    await interpretations.topAttention('p', 'study', 'Slide A', '1', 20, controller.signal);
    expect(request).toHaveBeenCalledWith('/projects/p/interpretations/study/slides/Slide%20A/attention/top?member=1&limit=20', { signal: controller.signal });
    await interpretations.patchImage('p', 'study', 'Slide A', '1', 123456, controller.signal);
    expect(fetchArtifactBlob).toHaveBeenCalledWith('/projects/p/interpretations/study/slides/Slide%20A/patches/123456/image?member=1&max_size=512', controller.signal);
  });
});
