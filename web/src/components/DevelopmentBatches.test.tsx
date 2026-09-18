import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { defaultRecipe, defaultResources } from '../api/development';
import type { DevelopmentResults, FrozenBatch, TrainingExecution, TrainingMetricDetails, TrainingRuntime } from '../api/development';
import { ConfigurationTable, RecipeFields, batchVersionTag, developmentTabs } from './DevelopmentBatches';
import { executionActions, RunTable, ResultsTable, TrainingControls, ExecutionEvidence } from './DevelopmentExecution';
import JobTray from './JobTray';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const batch = {
  id: 'batch', createdAt: '', manifest: {
    kind: 'mil-batch', spec: { version: 1, experimentName: 'ABMIL', batchName: 'Baseline', inputs, recipe: defaultRecipe(), mode: 'single', grid: { learningRates: [0.0003], weightDecays: [0.0001], maxEpochs: [100] }, configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '' },
    configurations: [{ id: 'candidate', number: 1, recipe: defaultRecipe() }],
    splitPlans: [{ id: 'split', planId: 'seed-7-fold-1', partitions: {}, slideCount: 10 }],
    runs: [{ id: 'run', candidateId: 'candidate', splitPlanId: 'split', trainingSeed: 42, status: 'planned' }],
    summary: { configurationCount: 1, trainingSeedCount: 1, splitPlanCount: 1, runCount: 1 },
  },
} as unknown as FrozenBatch;
const runtime: TrainingRuntime = { available: true, python: '/training/python', versions: { torch: '2.8' }, cudaAvailable: false, gpuCount: 0, findings: [] };
function execution(status: TrainingExecution['status'], overrides: Partial<TrainingExecution> = {}): TrainingExecution {
  return { batchId: 'batch', status, sessionName: 'hp-mil-test', logPath: '/tmp/worker.log', outputPath: '/tmp/output', findings: [], runCounts: { total: 1, queued: 0, running: 0, completed: 0, failed: 0, cancelled: 0 }, runs: [{ id: 'run', candidateId: 'candidate', trainingSeed: 42, splitPlanId: 'split', status }], createdAt: '', updatedAt: '', ...overrides };
}
function controls(value?: TrainingExecution | null, available = true) {
  return renderToStaticMarkup(<TrainingControls execution={value} runtime={{ ...runtime, available }} checking={false} pending={null} onAction={() => {}} />);
}

describe('development execution controls', () => {
  it('namespaces identical batch names by stable experiment identity without changing readable names', () => {
    expect(batchVersionTag('Baseline', 'draft-one')).not.toBe(batchVersionTag('Baseline', 'draft-two'));
    expect(batchVersionTag('Baseline', 'draft-one')).toBe('Baseline · draft-one');
    expect(batchVersionTag('B'.repeat(80), `draft-${'a'.repeat(32)}`)).toHaveLength(80);
  });

  it('makes launch explicit and disables it when the runtime is unavailable', () => {
    expect(controls(null)).toContain('Launch batch');
    expect(controls(null, false)).toMatch(/<button[^>]*disabled=""[^>]*>Launch batch/);
    expect(controls(execution('running'))).not.toContain('Launch batch');
    expect(controls(execution('running'))).toContain('Cancel batch');
  });

  it('keeps draining cancellation distinct from a terminal cancelled state and resumes only unfinished runs', () => {
    const draining = execution('running', { cancelRequested: true });
    expect(controls(draining)).toMatch(/<button[^>]*disabled=""[^>]*>Cancellation requested/);
    expect(executionActions(draining)).toEqual({ launch: false, cancel: false, resume: false });
    for (const status of ['cancelled', 'failed', 'interrupted'] as const) {
      expect(controls(execution(status))).toContain('Resume unfinished runs');
      expect(controls(execution(status))).toContain('Completed runs are retained');
    }
    expect(controls(execution('completed', { runCounts: { total: 1, completed: 1, queued: 0, running: 0, failed: 0, cancelled: 0 } }))).not.toContain('Resume unfinished');
  });

  it('displays actual run state, assessment evidence and worker errors without overwriting the frozen plan', () => {
    const live = execution('failed');
    const details: TrainingMetricDetails = { unit: 'patient', classOrder: ['low', 'high'], positiveClass: 'high', patientAggregation: 'mean_probabilities', selected: { accuracy: 0.75, auroc: null }, slide: { accuracy: 0.6 }, patient: { accuracy: 0.75, auroc: null } };
    live.runs[0] = { ...live.runs[0], error: 'Missing feature slide s99', checkpointPath: '/tmp/last.ckpt', metrics: { validation: { ...details, selected: { accuracy: 0.65 } }, assessment: details } };
    const html = renderToStaticMarkup(<RunTable batch={batch} execution={live} />);
    expect(html).toContain('failed');
    expect(html).toContain('Missing feature slide s99');
    expect(html).toContain('accuracy: 0.7500');
    expect(html).toContain('accuracy: 0.6500');
    expect(html).toContain('patient scoring details');
    expect(html).toContain('Checkpoint validation');
    expect(html).toContain('/tmp/last.ckpt');
    expect(batch.manifest.runs[0].status).toBe('planned');
  });

  it('publishes OOF metrics only for complete groups and separates checkpoint validation from assessment', () => {
    const results: DevelopmentResults = { batchId: 'batch', status: 'running', oof: [], candidates: [
      { candidateId: 'candidate', trainingSeed: 42, splitSeed: 7, complete: true, metrics: { auroc: 0.9 }, oofPath: '/tmp/complete-oof.csv', completedRuns: 3, totalRuns: 3 },
      { candidateId: 'candidate', trainingSeed: 43, splitSeed: 7, complete: false, metrics: { auroc: 0.1234 }, oofPath: '/tmp/incomplete-oof.csv', completedRuns: 1, totalRuns: 3 },
    ] };
    const html = renderToStaticMarkup(<ResultsTable batch={batch} results={results} loading={false} />);
    expect(html).toContain('Checkpoints follow each configuration’s frozen evaluation policy');
    expect(html).toContain('Assessment predictions');
    expect(html).toContain('auroc: 0.9000');
    expect(html).toContain('/tmp/complete-oof.csv');
    expect(html).toContain('Waiting for all folds');
    expect(html).not.toContain('0.1234');
    expect(html).not.toContain('/tmp/incomplete-oof.csv');
  });

  it('shows live epoch progress without treating the latest validation loss as a selected checkpoint or assessment', () => {
    const live = execution('running');
    live.runs[0].progress = { epoch: 3, maxEpochs: 10, globalStep: 12, trainingLoss: 0.7, validation: { loss: 0.8 } };
    const html = renderToStaticMarkup(<RunTable batch={batch} execution={live} />);
    expect(html).toContain('Epoch 3 / 10');
    expect(html).toContain('<th>Training loss</th><th>Current validation loss</th>');
    expect(html).toContain('<td>0.7000</td><td>0.8000</td>');
    expect(html).toContain('Checkpoint validation: —');
    expect(html).toContain('Held-out assessment: —');
  });

  it('keeps training state and artifacts inspectable when a run progress file is unreadable', () => {
    const live = execution('running');
    live.runs[0] = { ...live.runs[0], progress: null, progressWarning: 'Progress file could not be read.', outputPath: '/tmp/run-output' };
    const html = renderToStaticMarkup(<RunTable batch={batch} execution={live} />);
    expect(html).toContain('running');
    expect(html).toContain('Progress file could not be read.');
    expect(html).toContain('/tmp/run-output');
  });

  it('renders partial progress while validation metrics have not been written yet', () => {
    const live = execution('running');
    live.runs[0].progress = { epoch: 1, maxEpochs: 10, globalStep: 1 } as NonNullable<TrainingExecution['runs'][number]['progress']>;
    const html = renderToStaticMarkup(<RunTable batch={batch} execution={live} />);
    expect(html).toContain('Epoch 1 / 10');
    expect(html).toContain('<td>—</td><td>—</td>');
  });

  it('offers the ABMIL architecture defaults including zero dropout without requiring CUDA', () => {
    const html = renderToStaticMarkup(<RecipeFields value={{ ...defaultRecipe(), dropout: 0 }} onChange={() => {}} />);
    expect(html).toContain('Embedding dimensions');
    expect(html).toContain('value="512"');
    expect(html).toContain('value="384"');
    expect(html).toContain('Gradient checkpointing');
    expect(html).toContain('Gated attention');
  });

  it('shares real training state with the jobs tray and keeps freezing in a separate module', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['development-batches', 'project'], { items: [batch], executionImplemented: true, executions: [execution('running')] });
    client.setQueryData(['scientific', 'project', 'drafts'], { drafts: [] });
    try {
      const tray = renderToStaticMarkup(<QueryClientProvider client={client}><JobTray projectId="project" /></QueryClientProvider>);
      expect(tray).toContain('1 active job');
      expect(tray).not.toContain('Execution not implemented');
      expect(developmentTabs.map((tab) => tab.id)).toEqual(['setup', 'batches', 'runs', 'results']);
    } finally { client.clear(); }
  });

  it('offers whole-bag training explicitly and never presents null as a zero patch limit', () => {
    const whole = renderToStaticMarkup(<RecipeFields value={{ ...defaultRecipe(), bagSize: null }} onChange={() => {}} />);
    expect(whole).toContain('Use whole bag for training');
    expect(whole).toContain('Train with every available patch in each slide');
    expect(whole).toContain('No patch sampling is applied');
    expect(whole).not.toContain('>Patches per bag<input');
    const table = renderToStaticMarkup(<ConfigurationTable batch={{ configurations: [
      { id: 'whole', number: 1, recipe: { ...defaultRecipe(), bagSize: null } },
      { id: 'sampled', number: 2, recipe: { ...defaultRecipe(), bagSize: 2048 } },
    ] }} />);
    expect(table).toContain('Whole bag (all patches)');
    expect(table).toContain('2048 patches maximum');
  });

  it('discloses optional optimization settings while keeping grid values unambiguous', () => {
    const html = renderToStaticMarkup(<RecipeFields value={{ ...defaultRecipe(), lrScheduler: 'cosine', warmupEpochs: 3, minEpochs: 5, precision: 'bf16-mixed' }} gridMode onChange={() => {}} />);
    expect(html).toContain('Optimization &amp; stopping');
    expect(html).toContain('Model architecture');
    expect(html).toContain('Precision &amp; memory');
    expect(html).toContain('Warmup epochs');
    expect(html).toContain('Minimum training epochs');
    expect(html).toContain('Final LR fraction');
    expect(html).toContain('Accumulate batches');
    expect(html).not.toContain('>Maximum epochs<input');
    expect(html).not.toContain('>Learning rate<input');
  });

  it('distinguishes device-wide telemetry from per-run measurements and exposes the saved history', () => {
    const live = execution('running', {
      resourcePlan: { requestedConcurrency: 6, effectiveConcurrency: 1, cpuSlotsPerRun: 6, cpuLimit: 6, ramLimit: 20, gpuSlotLimit: 1, note: 'GPU slots limit this batch.' },
      telemetry: { path: '/tmp/telemetry.jsonl', intervalSeconds: 15, latest: { at: '2026-09-11T12:00:00Z', host: { cpuCount: 36, totalRamGb: 192, availableRamGb: 170, bootId: 'boot', kernel: 'linux' }, gpus: [{ index: 0, uuid: 'gpu', name: 'A5000', driverVersion: '596', totalMemoryGb: 24, usedMemoryGb: 5, freeMemoryGb: 19, utilizationPercent: 60 }], runs: [{ runId: 'run', pid: 123, rssGb: 2 }] }, peak: { hostUsedRamGb: 22, runRssGb: { run: 3 }, gpuUsedMemoryGb: { '0': 7 } } },
    });
    const html = renderToStaticMarkup(<ExecutionEvidence execution={live} />);
    expect(html).toContain('from 6 requested');
    expect(html).toContain('24.00 GiB total VRAM · Peak 7.00 GiB');
    expect(html).toContain('GPU usage includes other programs');
    expect(html).toContain('/tmp/telemetry.jsonl');
    expect(html).toContain('3.00 GiB');
    live.telemetry!.latest!.runs = [];
    live.runs[0].status = 'completed';
    live.runs[0].progress = { epoch: 10, maxEpochs: 10, globalStep: 40, trainingLoss: 0.3, validation: { loss: 0.4 }, learningRate: 0.00003, cudaPeakAllocatedBytes: 2 ** 30, cudaPeakReservedBytes: 2 * 2 ** 30 };
    const finished = renderToStaticMarkup(<RunTable batch={batch} execution={live} />);
    expect(finished).toContain('Process-tree RAM peak</dt><dd>3.00 GiB');
    expect(finished).toContain('CUDA allocated peak</dt><dd>1.00 GiB');
    expect(finished).toContain('CUDA reserved peak</dt><dd>2.00 GiB');
    expect(finished).toContain('Learning rate</dt><dd>3.000e-5');
  });
});
