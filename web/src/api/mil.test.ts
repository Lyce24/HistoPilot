import { afterEach, describe, expect, it, vi } from 'vitest';
import type { MILExperimentSpec } from './mil';

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('MIL experiment planning API', () => {
  it('checks the exact frozen inputs and persists them in a revisioned server draft', async () => {
    const spec: MILExperimentSpec = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'mmap', packArtifactId: 'pack' };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ canPlan: true, resolvedLoadingPolicy: 'mmap', packArtifactId: 'pack', bundleId: 'bundle', featureSetId: 'feature', findings: [], executionImplemented: false }))
      .mockResolvedValueOnce(json({ id: 'draft', revision: 3, payload: { type: 'mil-experiment', spec } }));
    vi.stubGlobal('fetch', fetcher);
    const { mil } = await import('./mil');
    const { scientific } = await import('./scientific');
    await expect(mil.preview('project/one', spec)).resolves.toMatchObject({ canPlan: true, executionImplemented: false });
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/mil-experiments/preview');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(spec);
    await scientific.saveDraft('project/one', { kind: 'experiment', name: 'My MIL plan', payload: { type: 'mil-experiment', spec } }, { id: 'draft', revision: 2 });
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/project%2Fone/drafts/draft');
    expect(fetcher.mock.calls[2][1].method).toBe('PATCH');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ expectedRevision: 2, name: 'My MIL plan', payload: { type: 'mil-experiment', spec } });
  });

  it('preserves an explicit native request and surfaces failed server checks', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: 'Bundle source changed; verify a new bundle.' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { mil } = await import('./mil');
    await expect(mil.preview('project', { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native', packArtifactId: null })).rejects.toMatchObject({ status: 409 });
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toMatchObject({ loadingPolicy: 'native', packArtifactId: null });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
