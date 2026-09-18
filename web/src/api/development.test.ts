import { afterEach, describe, expect, it, vi } from 'vitest';
import { defaultRecipe, defaultResources, parseNumberList, withRecipeDefaults, developmentPollInterval, latestExecution } from './development';
import type { TrainingExecution } from './development';

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('development batch intent', () => {
  it('posts the exact draft for a cancellable read-only runtime recommendation', async () => {
    const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response({ version: 1, applicable: false }));
    vi.stubGlobal('fetch', fetcher);
    const { development } = await import('./development');
    const spec = { version: 1 as const, experimentName: 'nnMIL', batchName: 'Folds', inputs: { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null }, recipe: defaultRecipe(), mode: 'single' as const, grid: { learningRates: [0.0001], weightDecays: [0.005], maxEpochs: [40] }, configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '' };
    const signal = new AbortController().signal;
    await development.runtimeRecommendation('project/one', spec, signal);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/mil-experiments/batches/runtime-recommendation');
    expect(fetcher.mock.calls[1][1]).toMatchObject({ method: 'POST', signal });
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(spec);
  });
  it('keeps CPU workers separate from concurrent runs and defaults to one process per GPU', () => {
    expect(defaultResources()).toMatchObject({ maxConcurrentRuns: 1, runsPerGpu: 1, dataLoaderWorkers: 2, cpuThreadsPerRun: 2 });
    expect(defaultRecipe().checkpointMetric).toBe('validation_auroc');
  });

  it('fills legacy architecture defaults without changing saved optimization settings or mutating the original', () => {
    const legacy = { model: 'abmil', learningRate: 0.001, weightDecay: 0, maxEpochs: 5, optimizer: 'adam' as const, batchSize: 1, bagSize: 20, earlyStopping: false, patience: 2, checkpointMetric: 'validation_loss' as const };
    expect(withRecipeDefaults(legacy)).toMatchObject({ ...legacy, embedDim: 512, attentionDim: 384, numFcLayers: 1, gatedAttention: true, dropout: 0.25, inputDropout: 0, gradientCheckpointing: false });
    expect(legacy).not.toHaveProperty('embedDim');
    expect(withRecipeDefaults({ ...legacy, dropout: 0 })).toHaveProperty('dropout', 0);
    expect(withRecipeDefaults({ ...legacy, bagSize: null })).toHaveProperty('bagSize', null);
  });

  it('polls active executions frequently and checks idle projects less often to discover CLI launches', () => {
    for (const status of ['queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted'] as const) {
      const execution = { status, cancelRequested: true } as TrainingExecution;
      expect(developmentPollInterval({ items: [], executionImplemented: true, executions: [execution] })).toBe(['queued', 'running'].includes(status) ? 3000 : 15000);
    }
    expect(developmentPollInterval()).toBe(false);
  });

  it('uses new project-level execution evidence when a selected-batch cache predates an external launch or completion', () => {
    const running = { status: 'running', updatedAt: '2026-09-11T16:00:00Z' } as TrainingExecution;
    const completed = { status: 'completed', updatedAt: '2026-09-11T16:01:00Z' } as TrainingExecution;
    expect(latestExecution(null, running)).toBe(running);
    expect(latestExecution(running, completed)).toBe(completed);
    expect(latestExecution(completed, running)).toBe(completed);
    expect(latestExecution(null, undefined)).toBeNull();
  });

  it('preserves scientific numeric values and rejects empty, duplicate, fractional-seed and nonfinite entries', () => {
    expect(parseNumberList('1e-4, 0.0003, 0.001', 'LR')).toEqual([0.0001, 0.0003, 0.001]);
    expect(parseNumberList('0, 20, 30', 'Seeds', true)).toEqual([0, 20, 30]);
    for (const value of ['', '1,', '1,,2', '1, 1', 'NaN', 'Infinity', '-1', '1.5']) {
      expect(() => parseNumberList(value, 'Seeds', true)).toThrow();
    }
  });

  it('freezes exactly the reviewed batch and reuses the caller operation ID', async () => {
    const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response({ id: 'frozen-batch' }));
    vi.stubGlobal('fetch', fetcher);
    const { development } = await import('./development');
    const spec = { version: 1 as const, experimentName: 'ABMIL', batchName: 'Search', inputs: { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null }, recipe: defaultRecipe(), mode: 'single' as const, grid: { learningRates: [0.0003], weightDecays: [0.0001], maxEpochs: [100] }, configurations: [], trainingSeeds: [10, 20, 30], resources: defaultResources(), notes: '' };
    await development.freeze('project/one', spec, 'reviewed-hash', 'stable-operation', { tag: 'Search', note: '' });
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/mil-experiments/batches/freeze');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ spec, previewHash: 'reviewed-hash', operationId: 'stable-operation', versionLabel: { tag: 'Search', note: '' } });
  });

  it.each(['launch', 'cancel', 'resume'] as const)('%s uses the frozen batch identity and stable operation ID, without resending editable recipe state', async (action) => {
    const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockResolvedValueOnce(response({ detail: 'Temporary worker error' }, 503))
      .mockResolvedValueOnce(response({ batchId: 'batch/a', status: 'running' }));
    vi.stubGlobal('fetch', fetcher);
    const { development } = await import('./development');
    await expect(development[action]('project/one', 'batch/a', 'same-operation')).rejects.toThrow('Temporary worker error');
    await expect(development[action]('project/one', 'batch/a', 'same-operation')).resolves.toMatchObject({ status: 'running' });
    for (const call of fetcher.mock.calls.slice(1)) {
      expect(call[0]).toBe(`/api/v1/projects/project%2Fone/mil-experiments/batches/batch%2Fa/${action}`);
      expect(call[1].method).toBe('POST');
      expect(JSON.parse(call[1].body)).toEqual({ operationId: 'same-operation' });
    }
  });

  it('reads runtime, execution and results without launching any batch', async () => {
    const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockResolvedValueOnce(response({ available: true })).mockResolvedValueOnce(response(null))
      .mockResolvedValueOnce(response({ status: 'planned', candidates: [], oof: [] }));
    vi.stubGlobal('fetch', fetcher);
    const { development } = await import('./development');
    await development.runtime('project');
    expect(await development.execution('project', 'batch')).toBeNull();
    await development.results('project', 'batch');
    expect(fetcher.mock.calls.map((call) => call[0])).toEqual(['/api/v1/session', '/api/v1/projects/project/mil-experiments/runtime', '/api/v1/projects/project/mil-experiments/batches/batch/execution', '/api/v1/projects/project/mil-experiments/batches/batch/results']);
    expect(fetcher.mock.calls.slice(1).every((call) => call[1].method === undefined)).toBe(true);
  });

  it('reads durable history from the exact selected run without mutating or conflating batch identity', async () => {
    const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response({ runId: 'run/b', rows: [], totalRows: 0, truncated: false }));
    vi.stubGlobal('fetch', fetcher);
    const { development } = await import('./development');
    await development.history('project/one', 'batch/a', 'run/b');
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/mil-experiments/batches/batch%2Fa/runs/run%2Fb/history');
    expect(fetcher.mock.calls[1][1].method).toBeUndefined();
  });
});
