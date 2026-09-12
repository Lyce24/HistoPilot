import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalInterpretation, { GalleryWorkspace } from './LocalInterpretation';
import type { GallerySlide, GallerySource, InterpretationSource, VisualizeSelection } from '../api/interpretation';
import { defaultRepresentation, mergeVisualizationItems, representationCompatible, selectedGallerySlides, sourceCompatible, validInterpretationResources, visualizationRequest } from '../lib/interpretationGallery';
import SlideGalleryCard from '../components/SlideGalleryCard';
import type { Workspace } from '../api/types';
const source: GallerySource = { predictorId: 'refit-1', featureBundleId: 'bundle', packArtifactId: null, slideFolder: '/slides' };
const resources = { maxConcurrentRuns: 1, gpuIds: [], runsPerGpu: 1, cpuThreadsPerRun: 4, dataLoaderWorkers: 0, ramGbPerRun: 8 };
const slide = (id: string, available = true): GallerySlide => ({ slideId: id, name: `${id}.svs`, relativePath: `${id}.svs`, slidePath: `/slides/${id}.svs`, available, reason: available ? null : 'No exact slide ID in this feature bundle.', patchCount: available ? 500 : null });
const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.unstubAllGlobals(); });
function client() { const value = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }); clients.push(value); return value; }
describe('interpretation gallery selection and source identity', () => {
  it('retains explicit selections across search and pages without including unavailable matches', () => {
    const first = slide('first'), nextPage = slide('second'), unavailable = slide('missing', false);
    const selected = selectedGallerySlides(new Map(), first, true);
    const next = selectedGallerySlides(selected, nextPage, true);
    expect([...next.keys()]).toEqual([first.slidePath, nextPage.slidePath]);
    expect([...selected.keys()]).toEqual([first.slidePath]);
    expect(selectedGallerySlides(next, unavailable, true).size).toBe(2);
    expect([...selectedGallerySlides(next, first, false).keys()]).toEqual([nextPage.slidePath]);
  });
  it('checks frozen source encoder, dimensions and integrity before enabling it', () => {
    const item: InterpretationSource = { id: 'b', name: 'Verified features', current: true, encoderId: 'uni', dimensions: 1024, dtype: 'float32', findings: [], packs: [], slideCount: 100, featureSetId: 'f' };
    expect(sourceCompatible(item, 'uni', 1024, 'float32')).toBe(true);
    expect(sourceCompatible(item, 'other', 1024, 'float32')).toBe(false);
    expect(sourceCompatible(item, 'uni', 768, 'float32')).toBe(false);
    expect(sourceCompatible({ ...item, current: false }, 'uni', 1024, 'float32')).toBe(false);
    expect(sourceCompatible({ ...item, findings: [{ severity: 'error', code: 'STALE', message: 'Features changed.' }] }, 'uni', 1024, 'float32')).toBe(false);
  });
  it('chooses an exact compatible pack for FP16 predictors without silently casting FP32 originals', () => {
    const original: InterpretationSource = { id: 'bundle', name: 'Mixed representations', current: true, encoderId: 'uni', dimensions: 1024, dtype: 'float32', findings: [], packs: [], slideCount: 100, featureSetId: 'f' };
    const packed = { ...original, packs: [{ id: 'packed-z', name: 'FP16 features', outputDtype: 'float16' }, { id: 'packed-a', name: 'Other FP16 features', outputDtype: 'float16' }, { id: 'packed32', name: 'FP32 pack', outputDtype: 'float32' }] };
    expect(sourceCompatible(original, 'uni', 1024, 'float16')).toBe(false);
    expect(sourceCompatible(packed, 'uni', 1024, 'float16')).toBe(true);
    expect(defaultRepresentation(original, 'float16')).toBeNull();
    expect(defaultRepresentation(packed, 'float16', 'packed-z')).toBe('packed-z');
    expect(defaultRepresentation(packed, 'float16', 'missing-pack')).toBe('packed-a');
    expect(defaultRepresentation({ ...packed, packs: [...packed.packs].reverse() }, 'float16')).toBe('packed-a');
    expect(defaultRepresentation(packed, 'float32')).toBe('');
    expect(defaultRepresentation(packed, 'float32', 'packed32')).toBe('packed32');
    expect(representationCompatible(['float16', 'float32'], 'float16')).toBe(false);
    expect(representationCompatible('float32', undefined)).toBe(false);
  });
  it('freezes exact native or packed requests for retries and merges outcomes by slide path', () => {
    const selection: VisualizeSelection = { ...source, packArtifactId: 'pack-1', slidePaths: ['/slides/b.svs', '/slides/a.svs', '/slides/a.svs'], resources };
    const request = visualizationRequest(selection, 'stable-operation');
    selection.slidePaths.push('/slides/other.svs'); selection.resources = { ...resources, ramGbPerRun: 32 };
    expect(request).toMatchObject({ operationId: 'stable-operation', selection: { packArtifactId: 'pack-1', slidePaths: ['/slides/a.svs', '/slides/b.svs'], resources: { ramGbPerRun: 8 } } });
    const failed = { slidePath: '/slides/a.svs', slideId: 'a', status: 'failed', reused: false, error: { code: 'RAM', message: 'Insufficient RAM' } };
    const success = { slidePath: '/slides/a.svs', slideId: 'a', status: 'queued', reused: true, interpretationId: 'study' };
    expect(mergeVisualizationItems([failed], [success])).toEqual([success]);
  });
  it('rejects empty, non-finite and out-of-range resources before automatic execution', () => {
    expect(validInterpretationResources(resources)).toBe(true);
    expect(validInterpretationResources({ ...resources, gpuIds: [127], ramGbPerRun: .5 })).toBe(true);
    for (const value of [{ ramGbPerRun: 0 }, { ramGbPerRun: Infinity }, { cpuThreadsPerRun: 0 }, { cpuThreadsPerRun: 2.5 }, { gpuIds: [128] }, { gpuIds: [NaN] }]) expect(validInterpretationResources({ ...resources, ...value })).toBe(false);
  });
});
describe('gallery controls and lazy slide cards', () => {
  it('preselects the predictor FP16 pack and explains why original FP32 features are unavailable', () => {
    vi.stubGlobal('window', { location: { hash: '#interpretation?predictor=fp16-predictor' } });
    const queryClient = client();
    queryClient.setQueryData(['predictors', 'p'], { items: [{ id: 'fp16-predictor', lifecycleState: 'active', manifest: { name: 'FP16 ensemble', method: 'ensemble', experimentId: 'exp', recipe: { model: 'abmil' }, inputs: { features: { bundle: { id: 'bundle' }, encoderId: 'uni', dimensions: 1024, dtype: 'float16' }, loading: { packArtifactId: 'preferred16' } } } }] });
    queryClient.setQueryData(['interpretation-sources', 'p'], { items: [{ id: 'bundle', name: 'Feature bundle', current: true, encoderId: 'uni', dimensions: 1024, dtype: 'float32', findings: [], slideCount: 100, featureSetId: 'f', packs: [{ id: 'preferred16', name: 'Frozen FP16 pack', outputDtype: 'float16' }, { id: 'other32', name: 'FP32 pack', outputDtype: 'float32' }] }] });
    for (const key of ['model-evaluations', 'clinical-analyses', 'interpretations']) queryClient.setQueryData([key, 'p'], { items: [] });
    const html = renderToStaticMarkup(<QueryClientProvider client={queryClient}><LocalInterpretation workspace={{ project: { id: 'p' } } as Workspace} /></QueryClientProvider>);
    expect(html).toContain('<option value="preferred16" selected="">Frozen FP16 pack · float16</option>');
    expect(html).toContain('<option value="" disabled="">Original slide features · float32 · requires float16</option>');
    expect(html).toContain('<option value="other32" disabled="">FP32 pack · float32 · requires float16</option>');
    expect(html).toContain('Predictor requires float16');
  });
  it('keeps search above slide cards, displays exact feature mismatch reasons, and starts with no batch selection', () => {
    const queryClient = client();
    queryClient.setQueryData(['interpretation-gallery', 'p', source, '', 0], { items: [slide('one'), slide('missing', false)], total: 2, offset: 0, limit: 24, hasMore: false, folder: '/slides', source: { featureBundleId: 'bundle', packArtifactId: null, encoderId: 'uni', dimensions: 1024, dtype: 'float32' }, warnings: [] });
    const html = renderToStaticMarkup(<QueryClientProvider client={queryClient}><GalleryWorkspace project="p" source={source} selected={[]} search="" offset={0} locked={false} canContinue onSelection={() => {}} onSearch={() => {}} onPage={() => {}} onContinue={() => {}} /></QueryClientProvider>);
    expect(html.indexOf('Search slides')).toBeLessThan(html.indexOf('Select one.svs'));
    expect(html).toContain('No exact slide ID in this feature bundle.');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Continue with 0 selected/);
    expect(html).not.toContain('checked=""');
    expect(html).not.toContain('Review attention study');
    const thumbnailQueries = queryClient.getQueryCache().findAll({ queryKey: ['interpretation-gallery-thumbnail'] });
    expect(thumbnailQueries).toHaveLength(2);
    expect(thumbnailQueries.every((query) => query.getObserversCount() === 0)).toBe(true);
  });
  it('disables adding an unselected slide when the batch limit is reached', () => {
    const queryClient = client();
    const html = renderToStaticMarkup(<QueryClientProvider client={queryClient}><SlideGalleryCard project="p" slide={slide('one')} checked={false} disabled={false} selectionDisabled onToggle={() => {}} /></QueryClientProvider>);
    expect(html).toMatch(/<input[^>]*disabled=""[^>]*aria-label="Select one.svs for batch visualization"/);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Select one.svs"/);
  });
});
