import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider, type QueryObserverOptions } from '@tanstack/react-query';
import type { GallerySlide, Interpretation, InterpretationExecution, VisualizeSelection } from '../api/interpretation';
import type { Workspace } from '../api/types';
import { adoptSavedInterpretation, replaceWizardContext, initialWizardDraft, persistWizardDraft, restoreWizardDraft, selectedBatchReady, wizardResources, wizardRoute, wizardStage, type InterpretationWizardDraft, type SelectedStudyState } from '../lib/interpretationWizard';
import LocalInterpretation from './LocalInterpretation';

const workspace = { project: { id: 'p' } } as Workspace;
const selectedSlide = (id: string): GallerySlide => ({ slideId: id, slidePath: `/slides/${id}.svs`, name: id, relativePath: `${id}.svs`, available: true, reason: null, patchCount: 2 });
const result = (id: string) => ({ slideId: id, patchCount: 2, probabilities: [.4, .6], attentionArtifact: 'slide-0.json', members: [{ index: 0, checkpointSha256: 'hash', probabilities: [.4, .6] }] });
const execution = (id: string, status: InterpretationExecution['status'] = 'completed'): InterpretationExecution => ({ status, result: status === 'completed' ? { slides: [result(id)] } : null });
function record(id: string, status: InterpretationExecution['status'] = 'completed'): Interpretation {
  return { id: `study-${id}`, createdAt: '', contentHash: id, lifecycleState: 'active', execution: execution(id, status), manifest: { kind: 'model-interpretation', name: `Attention ${id}`, predictorId: 'predictor', experimentId: 'experiment', encoderId: 'uni', method: 'refit', memberCount: 1, slides: [{ ...selectedSlide(id), patchCount: 2, featurePath: `/features/${id}.h5`, featureKey: 'features', coordinatesKey: 'coords', coordinateSpace: 'level0', confirmRowAlignment: true, width: 1000, height: 1000, backend: 'test', levelDownsamples: [1], dimensions: 1024, dtype: 'float32', patchWidthLevel0: 256, patchHeightLevel0: 256, alignment: 'embedded_verified' }] } };
}
function draft(ids: string[] = ['one', 'two']): InterpretationWizardDraft {
  const value = initialWizardDraft(new URLSearchParams('predictor=predictor'));
  const selected = ids.map(selectedSlide);
  const request: VisualizeSelection = { predictorId: 'predictor', featureBundleId: 'bundle', packArtifactId: null, slideFolder: '/slides', slidePaths: selected.map((slide) => slide.slidePath), patchWidthLevel0: 256, patchHeightLevel0: 256 };
  return { ...value, selected, search: 'retained search', offset: 24, batch: { id: 'batch-1', selected, request, pending: null, items: ids.map((id) => ({ slidePath: `/slides/${id}.svs`, slideId: id, interpretationId: `study-${id}`, status: 'completed', reused: true })) } };
}
const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.unstubAllGlobals(); });
function storage(hash: string) {
  const values = new Map<string, string>();
  vi.stubGlobal('window', { location: { hash }, sessionStorage: { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => values.set(key, value) } });
  return values;
}
function render(stage: string, value = draft(), records = [record('one'), record('two')], model = 'abmil') {
  storage(`#interpretation?predictor=predictor&${stage}`); persistWizardDraft('p', value);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }); clients.push(client);
  client.setQueryData(['predictors', 'p'], { items: [{ id: 'predictor', lifecycleState: 'active', manifest: { name: 'Selected refit', method: 'refit', experimentId: 'experiment', recipe: { model }, inputs: { features: { bundle: { id: 'bundle' }, encoderId: 'uni', dimensions: 1024, dtype: 'float32' }, loading: {} } } }] });
  client.setQueryData(['interpretation-sources', 'p'], { items: [{ id: 'bundle', name: 'Frozen bundle', current: true, findings: [], encoderId: 'uni', dimensions: 1024, dtype: 'float32', slideCount: 2, featureSetId: 'features', packs: [], datasetId: 'data', datasetName: 'Imported slides', slideFolder: '/slides' }] });
  for (const key of ['model-evaluations', 'clinical-analyses']) client.setQueryData([key, 'p'], { items: [] });
  client.setQueryData(['interpretations', 'p'], { items: records });
  for (const item of records) { client.setQueryData(['interpretation', 'p', item.id], item); client.setQueryData(['compute-job', 'p', 'interpretation', item.id], item.execution); }
  return { client, html: renderToStaticMarkup(<QueryClientProvider client={client}><LocalInterpretation workspace={workspace} /></QueryClientProvider>) };
}

describe('interpretation stages and resumable context', () => {
  it('opens a linked case search without replacing an uncertain visualization operation', () => {
    storage('#interpretation');
    const parameters = new URLSearchParams('predictor=predictor&search=Slide+A');
    expect(initialWizardDraft(parameters).search).toBe('Slide A');
    const value = draft();
    persistWizardDraft('p', value);
    expect(restoreWizardDraft('p', parameters)).toMatchObject({ search: 'Slide A', offset: 0, selected: value.selected });
    value.batch!.pending = { selection: value.batch!.request!, operationId: 'pending-review' };
    persistWizardDraft('p', value);
    expect(restoreWizardDraft('p', parameters)).toMatchObject({ search: 'retained search', offset: 24, batch: { uncertain: true, pending: { operationId: 'pending-review' } } });
  });
  it('preserves lineage in stage URLs and supports legacy saved-study links', () => {
    const query = new URLSearchParams('experiment=exp&predictor=predictor&evaluation=eval&clinical=report');
    const route = wizardRoute(query, 'viewer', { predictorId: 'predictor', evaluationId: 'eval', clinicalId: 'report' }, 'study 1', 'Slide A');
    const parsed = new URLSearchParams(route.split('?')[1]);
    expect(parsed.get('experiment')).toBe('exp'); expect(parsed.get('stage')).toBe('viewer'); expect(parsed.get('slide')).toBe('Slide A');
    expect(wizardStage(new URLSearchParams('interpretation=old-study'))).toBe('viewer');
    expect(wizardStage(new URLSearchParams('stage=select&interpretation=old-study'))).toBe('select');
    expect(wizardRoute(parsed, 'results', { predictorId: 'predictor', evaluationId: 'eval', clinicalId: 'report' })).not.toContain('interpretation=');
  });
  it('restores selection/search and the exact uncertain operation after reload, without serializing heavy record contents', () => {
    const values = storage('#interpretation'); const value = draft();
    value.batch!.pending = { selection: structuredClone(value.batch!.request!), operationId: 'same-operation' };
    value.batch!.items[0].interpretation = record('one');
    persistWizardDraft('p', value);
    const restored = restoreWizardDraft('p', new URLSearchParams('predictor=another-external-link'));
    expect(restored.search).toBe('retained search'); expect(restored.offset).toBe(24); expect(restored.selected).toHaveLength(2);
    expect(restored.batch?.pending).toEqual(value.batch!.pending); expect(restored.batch?.uncertain).toBe(true);
    expect(restored.predictorId).toBe('predictor'); expect([...values.values()][0]).not.toContain('checkpointSha256');
    expect(restoreWizardDraft('another-project', new URLSearchParams()).batch).toBeNull();
  });
  it('defaults older resource fields and rejects malformed saved operations before they can be retried', () => {
    const values = storage('#interpretation'); const value = draft();
    persistWizardDraft('p', value);
    const key = [...values.keys()][0];
    values.set(key, JSON.stringify({ ...value, resources: { device: 'gpu' } }));
    const restored = restoreWizardDraft('p', new URLSearchParams());
    expect(wizardResources(restored.resources)).toMatchObject({ gpuIds: [0], cpuThreadsPerRun: 4, ramGbPerRun: 8 });
    expect(restored.resources.width).toBe('');
    values.set(key, JSON.stringify({ ...value, batch: { ...value.batch, pending: { operationId: 'lost-operation', selection: { slidePaths: null } } } }));
    expect(restoreWizardDraft('p', new URLSearchParams()).batch).toBeNull();
    values.set(key, JSON.stringify({ ...value, batch: { ...value.batch, selected: [{ slideId: 'broken' }] } }));
    expect(restoreWizardDraft('p', new URLSearchParams()).batch).toBeNull();
  });
  it('adopts a saved study’s exact bundle/pack and common explicit geometry without inheriting another gallery', () => {
    const previous = { ...draft(), bundleId: 'other-bundle', packChoice: 'other-pack', resources: { ...draft().resources, width: '999', height: '999' } };
    const saved = record('saved'); saved.manifest.featureBundleId = 'saved-bundle'; saved.manifest.packArtifactId = 'saved-pack';
    saved.manifest.selection = { slides: [{ patchWidthLevel0: 123.5, patchHeightLevel0: 240 }] };
    const restored = adoptSavedInterpretation(previous, saved);
    expect(restored).toMatchObject({ bundleId: 'saved-bundle', packChoice: 'saved-pack', search: '', offset: 0, resources: { width: '123.5', height: '240', threads: '4' } });
    expect(restored.selected.map((slide) => slide.slidePath)).toEqual(['/slides/saved.svs']);
    expect(adoptSavedInterpretation(restored, saved)).toBe(restored);
    saved.manifest.packArtifactId = null; saved.manifest.selection = { slides: [{}, {}] };
    expect(adoptSavedInterpretation(previous, saved)).toMatchObject({ packChoice: '', resources: { width: '', height: '' } });
    delete saved.manifest.featureBundleId;
    const legacy = adoptSavedInterpretation(previous, saved);
    expect(legacy).toMatchObject({ bundleId: null, packChoice: null, selected: [] });
    expect(legacy.batch?.selected.map((slide) => slide.slidePath)).toEqual(['/slides/saved.svs']);
    previous.batch!.pending = { selection: previous.batch!.request!, operationId: 'uncertain' };
    expect(adoptSavedInterpretation(previous, saved)).toBe(previous);
  });
  it('persists an explicit selection context and replaces its URL before a reload', () => {
    storage('#interpretation?stage=select&predictor=old&evaluation=old-evaluation');
    const history = { state: null, replaceState: vi.fn((_state: unknown, _title: string, route: string) => { window.location.hash = route; }) };
    Object.assign(window, { history, dispatchEvent: vi.fn() });
    const next = { ...draft(), predictorId: 'new-model', evaluationId: 'new-evaluation', clinicalId: '', selected: [], batch: null };
    replaceWizardContext('p', next, new URLSearchParams('stage=select&predictor=old&evaluation=old-evaluation'), 'select');
    const query = new URLSearchParams(window.location.hash.split('?')[1]);
    expect(query.get('predictor')).toBe('new-model'); expect(query.get('evaluation')).toBe('new-evaluation');
    expect(restoreWizardDraft('p', query)).toMatchObject({ predictorId: 'new-model', evaluationId: 'new-evaluation', batch: null });
    expect(history.replaceState).toHaveBeenCalledOnce();
  });
  it('requires every selected completed result and rejects failed, missing, unrelated or unverified results', () => {
    const selected = [selectedSlide('one'), selectedSlide('two')];
    const states = new Map<string, SelectedStudyState>(selected.map((slide) => [slide.slidePath, { hasRecord: true, execution: execution(slide.slideId) }]));
    expect(selectedBatchReady(selected, states)).toBe(true);
    states.set('/slides/two.svs', { hasRecord: true, execution: execution('two', 'failed') }); expect(selectedBatchReady(selected, states)).toBe(false);
    states.set('/slides/two.svs', { hasRecord: true, execution: execution('unrelated') }); expect(selectedBatchReady(selected, states)).toBe(false);
    states.set('/slides/two.svs', { hasRecord: true, execution: execution('two'), error: new Error('Receipt changed') }); expect(selectedBatchReady(selected, states)).toBe(false);
    states.delete('/slides/two.svs'); expect(selectedBatchReady(selected, states)).toBe(false);
  });
});
describe('focused interpretation screens', () => {
  it('allows a frozen nnMIL predictor to use attention interpretation', () => {
    const html = render('stage=select', draft(), [], 'nnmil').html;
    expect(html).toContain('Choose a frozen ABMIL or nnMIL predictor');
    expect(html).toContain('<option value="predictor" selected="">Selected refit');
    expect(html).not.toContain('attention unsupported');
    expect(render('stage=select', draft(), [], 'mean_pool').html).toContain('attention unsupported');
  });
  it('keeps the results page limited to selected slides and separates the focused viewer from setup', () => {
    const results = render('stage=results', draft(), [record('one'), record('two'), record('unselected')]).html;
    expect(results).toContain('Attention results'); expect(results).toContain('Open attention viewer for one'); expect(results).not.toContain('unselected');
    expect(results).not.toContain('Choose model and shared features'); expect(results).not.toContain('Search slides');
    const viewer = render('stage=viewer&interpretation=study-one&slide=one').html;
    expect(viewer).toContain('Attention one'); expect(viewer).toContain('Back to results');
    expect(viewer).not.toContain('Choose model and shared features'); expect(viewer).not.toContain('Search slides'); expect(viewer).not.toContain('Compute slide attention');
  });
  it('registers every selected running job independently of result-card visibility and blocks results until all finish', () => {
    const ids = Array.from({ length: 12 }, (_, index) => `slide-${index}`);
    const value = draft(ids); const records = ids.map((id) => record(id, 'running'));
    const { client, html } = render('stage=compute', value, records);
    expect(html).toContain('0 / 12 selected slides completed'); expect(html).not.toContain('class="slide-gallery-image"');
    const jobs = client.getQueryCache().findAll({ queryKey: ['compute-job', 'p', 'interpretation'] });
    expect(jobs).toHaveLength(12);
    for (const job of jobs) { const interval = (job.options as QueryObserverOptions).refetchInterval; expect(typeof interval).toBe('function'); }
    expect(render('stage=results', value, records).html).toContain('Selected attention is not ready');
  });
  it('locks context mutations and saved-study entry points while preserving an uncertain request', () => {
    const value = draft(); value.batch!.pending = { selection: value.batch!.request!, operationId: 'stable' };
    const { html } = render('stage=select', value);
    expect(html).toContain('<fieldset class="interpretation-sources" disabled="">');
    expect(html).toContain('<fieldset class="chain-fields" disabled=""><legend class="sr-only">Evidence links');
    expect(html).toMatch(/<button class="text-button" disabled="">Attention one/);
    expect(html).not.toContain('Slide folder on the server'); expect(html).toContain('Imported slides');
  });
});
