import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider, QueryObserver } from '@tanstack/react-query';
import { development, defaultResources, nnmilRecipe, type DevelopmentBatchSpec, type RuntimeRecommendation } from '../api/development';
import DevelopmentBatches from './DevelopmentBatches';
import { canApplyRuntimeRecommendation, EditingRuntimeRecommendation, RuntimeRecommendationView, runtimeRecommendationOptions, runtimeRecommendationSource, scheduleRuntimeRecommendation } from './RuntimeRecommendation';
import type { BatchEditorDraft } from '../lib/batchEditorDraft';
import { sessionDraftKey } from '../lib/sessionDraft';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const spec = (): DevelopmentBatchSpec => ({ version: 1, experimentName: 'CRC KRAS', experimentId: 'experiment', experimentRevision: 1,
  batchName: 'nnMIL folds', inputs, recipe: nnmilRecipe(), mode: 'single', grid: { learningRates: [1e-4], weightDecays: [5e-3], maxEpochs: [40] },
  configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '' });
const recommendation = (): RuntimeRecommendation => ({ version: 1, generatedAt: '2026-09-15T12:00:00Z', applicable: true, basis: 'measured',
  resources: { maxConcurrentRuns: 3, gpuIds: [0, 2], runsPerGpu: 2, cpuThreadsPerRun: 2, dataLoaderWorkers: 2, ramGbPerRun: 12 },
  summary: 'Start with three concurrent runs.', memory: { perRunGpuGb: 4, perRunRamGb: 12, observedRuns: 1 },
  limits: { cpuConcurrency: 5, ramConcurrency: 10, gpuConcurrency: 3, additionalRunsNow: 0 },
  evidence: [{ key: 'workload-key', gpuUuid: 'gpu-uuid', trainingPatches: 6233, evaluationPatches: 45000, peakReservedGpuGb: 3.39, stage: 'fit', epoch: 26, batchId: 'b1', runId: 'run-1' }],
  findings: [{ severity: 'warning', code: 'RESERVATION', message: 'An active batch reserves the GPU.' }],
  hardware: { cpuCount: 36, totalRamGb: 192, availableRamGb: 170, gpus: [] } });
const clients: QueryClient[] = [];
function client() { const value = new QueryClient({ defaultOptions: { queries: { retry: false } } }); clients.push(value); return value; }
afterEach(() => { for (const value of clients.splice(0)) value.clear(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('runtime recommendation request lifecycle', () => {
  it('automatically detects once enabled, and refreshes explicitly without applying resources', async () => {
    const fetcher = vi.spyOn(development, 'runtimeRecommendation').mockResolvedValue(recommendation());
    const value = spec();
    const key = runtimeRecommendationSource(value, '{}')!;
    const observer = new QueryObserver(client(), runtimeRecommendationOptions('project', key, value, false));
    const unsubscribe = observer.subscribe(() => {});
    try {
      expect(fetcher).not.toHaveBeenCalled();
      observer.setOptions(runtimeRecommendationOptions('project', key, value, true));
      await vi.waitFor(() => expect(observer.getCurrentResult().isSuccess).toBe(true));
      expect(fetcher).toHaveBeenCalledExactlyOnceWith('project', value, expect.any(AbortSignal));
      expect(value.resources.maxConcurrentRuns).toBe(1);
      await observer.refetch();
      expect(fetcher).toHaveBeenCalledTimes(2);
      expect(value.resources.maxConcurrentRuns).toBe(1);
    } finally { unsubscribe(); }
  });

  it('cancels an old draft request and isolates its late response from the new draft', async () => {
    const pending: { resolve: (result: RuntimeRecommendation) => void; signal?: AbortSignal }[] = [];
    vi.spyOn(development, 'runtimeRecommendation').mockImplementation((_project, _spec, signal) => new Promise((resolve) => pending.push({ resolve, signal })));
    const first = spec(); const next = { ...first, recipe: { ...first.recipe, batchSize: 64 } };
    const firstKey = runtimeRecommendationSource(first, '{}')!; const nextKey = runtimeRecommendationSource(next, '{}')!;
    const observer = new QueryObserver(client(), runtimeRecommendationOptions('project', firstKey, first, true));
    const unsubscribe = observer.subscribe(() => {});
    try {
      await vi.waitFor(() => expect(pending).toHaveLength(1));
      observer.setOptions(runtimeRecommendationOptions('project', nextKey, next, true));
      await vi.waitFor(() => expect(pending).toHaveLength(2));
      expect(pending[0].signal?.aborted).toBe(true);
      pending[0].resolve(recommendation());
      expect(observer.getCurrentResult().data).toBeUndefined();
      expect(canApplyRuntimeRecommendation(nextKey, firstKey, recommendation())).toBe(false);
      pending[1].resolve({ ...recommendation(), summary: 'For the changed batch.' });
      await vi.waitFor(() => expect(observer.getCurrentResult().data?.sourceKey).toBe(nextKey));
      expect(observer.getCurrentResult().data?.recommendation.summary).toBe('For the changed batch.');
    } finally { unsubscribe(); }
  });

  it('does not apply retained cached settings after a refresh failure', async () => {
    vi.spyOn(development, 'runtimeRecommendation').mockResolvedValueOnce(recommendation()).mockRejectedValueOnce(new Error('Probe unavailable'));
    const value = spec(); const key = runtimeRecommendationSource(value, '{}')!;
    const observer = new QueryObserver(client(), runtimeRecommendationOptions('project', key, value, true));
    const unsubscribe = observer.subscribe(() => {});
    try {
      await vi.waitFor(() => expect(observer.getCurrentResult().isSuccess).toBe(true));
      const result = await observer.refetch();
      expect(result.error?.message).toBe('Probe unavailable');
      expect(result.data?.recommendation).toBeDefined();
      expect(canApplyRuntimeRecommendation(key, result.data?.sourceKey ?? null, result.data?.recommendation, false, result.error)).toBe(false);
    } finally { unsubscribe(); }
  });

  it('waits for typing to settle and cancels hidden or superseded draft work', () => {
    vi.useFakeTimers();
    const ready = vi.fn(); const cancel = scheduleRuntimeRecommendation(() => true, ready);
    vi.advanceTimersByTime(599); expect(ready).not.toHaveBeenCalled();
    cancel(); vi.advanceTimersByTime(1); expect(ready).not.toHaveBeenCalled();
    scheduleRuntimeRecommendation(() => true, ready);
    vi.advanceTimersByTime(600); expect(ready).toHaveBeenCalledExactlyOnceWith();
  });

  it('rechecks unfinished numeric input before issuing a request', () => {
    vi.useFakeTimers(); let valid = true; const ready = vi.fn(); const invalid = vi.fn();
    scheduleRuntimeRecommendation(() => valid, ready, invalid); valid = false;
    vi.advanceTimersByTime(600); expect(ready).not.toHaveBeenCalled();
    expect(invalid).toHaveBeenCalledExactlyOnceWith();
    const value = spec();
    expect(runtimeRecommendationSource(value, '{}')).not.toBe(runtimeRecommendationSource(value, JSON.stringify({ 'shared:Batch size': { source: value.recipe.batchSize, text: '' } })));
    expect(runtimeRecommendationSource(null, '{}')).toBeNull();
  });

  it.each([
    { current: null, result: 'a', refreshing: false },
    { current: 'b', result: 'a', refreshing: false },
    { current: 'a', result: 'a', refreshing: true },
  ])('rejects incomplete, superseded and refreshing suggestions: %j', ({ current, result, refreshing }) => {
    expect(canApplyRuntimeRecommendation(current, result, recommendation(), refreshing)).toBe(false);
  });
  it('requires applicable resources and permits a current measured or estimated recommendation', () => {
    expect(canApplyRuntimeRecommendation('a', 'a', recommendation())).toBe(true);
    expect(canApplyRuntimeRecommendation('a', 'a', { ...recommendation(), basis: 'estimated' })).toBe(true);
    expect(canApplyRuntimeRecommendation('a', 'a', { ...recommendation(), applicable: false })).toBe(false);
    expect(canApplyRuntimeRecommendation('a', 'a', { ...recommendation(), resources: null })).toBe(false);
  });
});

describe('runtime suggestion presentation and batch integration', () => {
  function render(value?: RuntimeRecommendation, error: Error | null = null) {
    return renderToStaticMarkup(<RuntimeRecommendationView recommendation={value} pending={false} error={error}
      canApply={Boolean(value?.applicable && !error)} canRefresh onRefresh={() => {}} onApply={() => {}} />);
  }
  it('explains measured capacity and active reservations separately from a performance optimum', () => {
    const html = render(recommendation());
    for (const text of ['Uses matching run measurements', 'starting points', 'benchmark complete epoch time', 'GPU 0, 2', 'Data workers per loader', 'CPU threads per run', '1 matching measured run', 'Current capacity for additional runs: <strong>0</strong>', 'An active batch reserves the GPU.', 'epoch 26', 'Apply suggested settings', 'Active and frozen batches keep their saved settings']) expect(html).toContain(text);
  });
  it('labels estimates and unavailable hardware without making the ordinary workflow depend on them', () => {
    expect(render({ ...recommendation(), basis: 'estimated' })).toContain('Estimated from this workload');
    const missing = render({ ...recommendation(), applicable: false, basis: 'unavailable', resources: null, evidence: [],
      memory: { perRunGpuGb: null, perRunRamGb: null, observedRuns: 0 }, limits: { cpuConcurrency: 0, ramConcurrency: 0, gpuConcurrency: null, additionalRunsNow: null } });
    expect(missing).toContain('Suggestion unavailable');
    expect(missing).toContain('Current additional capacity is unknown');
    expect(missing).toMatch(/<button[^>]*disabled=""[^>]*>Apply suggested settings/);
    expect(render(undefined, new Error('GPU probe failed'))).toContain('You can continue configuring resources manually');
    expect(render()).toContain('Complete valid batch settings');
  });
  it('marks an existing plan dirty and invalidates its preview when the explicit apply button changes compute settings', () => {
    const value = spec(); const recipeBefore = structuredClone(value.recipe);
    const state = { dirty: false, preview: 'reviewed-plan' as string | null, gpuText: '0' };
    const editor = EditingRuntimeRecommendation({ project: 'project', spec: value, draftKey: '{}', isDraftValid: () => true,
      onApply: (resources) => { value.resources = resources; state.gpuText = resources.gpuIds.join(', '); },
      onEdit: () => { state.dirty = true; state.preview = null; } });
    expect(state.dirty).toBe(false);
    editor.props.onApply(recommendation().resources!);
    expect(value.resources).toEqual(recommendation().resources);
    expect(state).toEqual({ dirty: true, preview: null, gpuText: '0, 2' });
    expect(value.recipe).toEqual(recipeBefore);
    expect(value.inputs).toEqual(inputs);
    expect(value.trainingSeeds).toEqual([42]);
  });

  function renderEditor(page: number, overrides: Partial<BatchEditorDraft> = {}, readOnly = false) {
    const value = spec();
    const draft: BatchEditorDraft = { version: 1, editorRevision: 1, inputs, workingPlan: 'saved-plan', name: value.batchName,
      editorOpen: true, batchPage: page, templateId: 'nnmil', predictorPolicy: { method: 'ensemble', refitPercentile: null },
      recipe: value.recipe, resources: value.resources, mode: 'single', rows: [{ id: 0, recipe: value.recipe }], explicitInitialized: false,
      seeds: '42', lrs: '0.0001', wds: '0.005', epochs: '40', gpus: '0', notes: '', numericDrafts: {}, ...overrides };
    const key = sessionDraftKey('project', 'experiment', 'batch-editor');
    vi.stubGlobal('window', { sessionStorage: { getItem: (requested: string) => requested === key ? JSON.stringify({ version: 1, value: draft }) : null } });
    return renderToStaticMarkup(<QueryClientProvider client={client()}><DevelopmentBatches project="project" experimentName="CRC KRAS" experimentId="experiment"
      experimentRevision={1} inputs={inputs} experimentStage="planning" ownedBatches={[]} ownedDrafts={[]} tab="batches" readOnly={readOnly} onOpenSetup={() => {}} /></QueryClientProvider>);
  }
  it('mounts detection on the compute step and retains all manual controls and worker explanations', () => {
    const html = renderEditor(3);
    expect(html).toContain('Suggested runtime settings');
    for (const text of ['Allowed GPU IDs', 'Concurrent runs', 'Runs per GPU', 'CPU threads per run', 'Data workers per loader', 'CPU threads + 2 × data workers']) expect(html).toContain(text);
    expect(renderEditor(2)).not.toContain('Suggested runtime settings');
    expect(renderEditor(3, {}, true)).not.toContain('Suggested runtime settings');
  });
  it('does not start detection for incomplete training seed lists', () => {
    const fetcher = vi.spyOn(development, 'runtimeRecommendation');
    expect(renderEditor(3, { seeds: '42,' })).toContain('Complete valid batch settings');
    expect(fetcher).not.toHaveBeenCalled();
  });
});
