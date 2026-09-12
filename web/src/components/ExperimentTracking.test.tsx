import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { defaultRecipe, defaultResources } from '../api/development';
import type { FrozenBatch, TrainingExecution, TrainingHistory, TrainingRun, TrainingRuntime } from '../api/development';
import DevelopmentExecution, { ResultsTable, TrainingControls } from './DevelopmentExecution';
import { EpochProgress, LossHistory, ResourceCards, RunTable, metricValue, runLabel } from './ExperimentTracking';

const run: TrainingRun = { id: 'run-one', candidateId: 'configuration-one', splitPlanId: 'split-one', trainingSeed: 42, status: 'running', progress: { epoch: 2, maxEpochs: 10, globalStep: 20, trainingLoss: 0.0, validation: { loss: null } } };
const batch = { id: 'batch-one', createdAt: '', manifest: {
  spec: { recipe: defaultRecipe(), resources: defaultResources(), batchName: 'Baseline' },
  configurations: [{ id: run.candidateId, number: 1, recipe: defaultRecipe() }],
  splitPlans: [{ id: run.splitPlanId, planId: 'seed:7/fold:0', seed: 7, fold: 0 }],
  runs: [{ ...run, status: 'planned' }],
} } as unknown as FrozenBatch;
const runtime: TrainingRuntime = { available: true, python: '/training/python', versions: { torch: '2.8' }, cudaAvailable: true, gpuCount: 1, findings: [] };
const execution: TrainingExecution = {
  batchId: batch.id, status: 'running', sessionName: 'batch-one', logPath: '/worker.log', outputPath: '/run', runs: [run], findings: [],
  createdAt: '2026-09-12T12:00:00Z', updatedAt: '2026-09-12T12:00:30Z', runCounts: { total: 1, running: 1, queued: 0, completed: 0, failed: 0, cancelled: 0 },
  telemetry: { intervalSeconds: 15, path: '/telemetry.jsonl', latest: { at: '2026-09-12T12:00:30Z', host: { cpuCount: 8, totalRamGb: 32, availableRamGb: 24, bootId: 'boot', kernel: 'linux' }, gpus: [{ index: 0, uuid: 'gpu', name: 'A5000', driverVersion: '596', totalMemoryGb: 24, usedMemoryGb: 5, freeMemoryGb: 19, utilizationPercent: 60 }], runs: [{ runId: run.id, pid: 100, rssGb: 2 }] }, peak: { hostUsedRamGb: 8, runRssGb: { [run.id]: 2.5 }, gpuUsedMemoryGb: { 0: 7 } } },
};
const history: TrainingHistory = { runId: run.id, totalRows: 3, truncated: false, rows: [
  { epoch: 1, trainingLoss: 0.9, validation: { loss: 0.7, auroc: 0.6 }, checkpointUnit: 'patient', learningRate: 0.001 },
  { epoch: 2, trainingLoss: 0.7, validation: { loss: null, auroc: null }, checkpointUnit: 'patient', learningRate: 0.001 },
  { epoch: 3, trainingLoss: 0.5, validation: { loss: 0.4, auroc: 0.8 }, checkpointUnit: 'patient', learningRate: 0.001 },
] };

afterEach(() => { vi.useRealTimers(); });

describe('experiment tracking', () => {
  it('identifies both seeds and displays zero-based folds as human-readable fold numbers', () => {
    expect(runLabel(batch, run)).toBe('Config 1 · Fold 1 · Train seed 42 · Split seed 7');
    expect(runLabel({ ...batch, manifest: { ...batch.manifest, splitPlans: [{ ...batch.manifest.splitPlans[0], seed: 8 }] } }, run)).not.toBe(runLabel(batch, run));
  });

  it('shows zero loss as measured and missing/nonfinite loss as unavailable', () => {
    expect(metricValue(0)).toBe('0.0000');
    expect(metricValue(null)).toBe('—');
    expect(metricValue(Number.NaN)).toBe('—');
    const html = renderToStaticMarkup(<RunTable batch={batch} execution={execution} />);
    expect(html).toContain('<td>0.0000</td><td>—</td>');
    expect(html).toContain('All statuses');
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain('Checkpoint validation: —');
    expect(html).toContain('Held-out assessment: —');
  });

  it('keeps completed early-stopped runs distinct from reaching their maximum epoch budget', () => {
    const html = renderToStaticMarkup(<EpochProgress run={{ ...run, status: 'completed' }} />);
    expect(html).toContain('Epoch 2 / 10');
    expect(html).toContain('Stopped early');
    expect(html).toContain('value="2" max="10"');
    expect(renderToStaticMarkup(<EpochProgress run={{ ...run, progress: undefined }} />)).toContain('No epoch reported');
  });

  it('uses durable epoch history for separate train/validation curves and leaves missing validation values as gaps', () => {
    const html = renderToStaticMarkup(<LossHistory history={history} validationMetric="auroc" />);
    expect(html).toContain('Loss history');
    expect(html).toContain('Validation AUROC');
    expect(html).toContain('Validation uses patient scoring');
    expect(html).toContain('not held-out assessment');
    expect(html).toContain('View numeric curve data');
    expect(html).toMatch(/stroke-dasharray="7 5"><circle[^>]+><\/circle><circle/);
    expect(html).not.toContain('NaN');
  });

  it('distinguishes waiting, missing finished history, invalid history and explicit truncation', () => {
    expect(renderToStaticMarkup(<LossHistory />)).toContain('after the first completed epoch');
    expect(renderToStaticMarkup(<LossHistory finished />)).toContain('No epoch history was recorded');
    const invalid = renderToStaticMarkup(<LossHistory history={{ ...history, warning: 'History file is invalid.' }} />);
    expect(invalid).toContain('History file is invalid.');
    expect(invalid).not.toContain('<svg');
    const truncated = renderToStaticMarkup(<LossHistory history={{ ...history, totalRows: 2100, truncated: true }} />);
    expect(truncated).toContain('latest 3 of 2100');
  });

  it('reports device-wide GPU utilization and marks old samples stale instead of implying live usage', () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date('2026-09-12T12:05:00Z'));
    const html = renderToStaticMarkup(<ResourceCards execution={execution} />);
    expect(html).toContain('60% utilization');
    expect(html).toContain('GPU usage includes other programs');
    expect(html).toContain('resource sample is stale');
    const finished = renderToStaticMarkup(<ResourceCards execution={{ ...execution, status: 'completed' }} />);
    expect(finished).toContain('Last resource sample');
    expect(finished).toContain('recorded measurements, not live device usage');
    expect(finished).not.toContain('resource sample is stale');
  });

  it('does not substitute zero utilization for unavailable hardware telemetry', () => {
    const value = structuredClone(execution);
    value.telemetry!.latest.gpus[0].utilizationPercent = null;
    value.telemetry!.latest.gpuProbeError = 'Device probe timed out';
    const html = renderToStaticMarkup(<ResourceCards execution={value} />);
    expect(html).toContain('Utilization unavailable');
    expect(html).toContain('Device probe timed out');
    expect(html).not.toContain('0% utilization');
    expect(renderToStaticMarkup(<ResourceCards execution={{ ...execution, telemetry: undefined }} />)).toContain('not been recorded yet');
  });

  it('prevents embedded launch and every read-only mutation, including cancellation', () => {
    const props = { runtime, checking: false, pending: null, onAction: () => {}, allowLaunch: false };
    expect(renderToStaticMarkup(<TrainingControls {...props} />)).not.toContain('Launch batch');
    expect(renderToStaticMarkup(<TrainingControls {...props} execution={execution} />)).toContain('Cancel batch');
    const readonly = renderToStaticMarkup(<TrainingControls {...props} execution={execution} readOnly />);
    expect(readonly).not.toContain('Cancel batch');
    expect(readonly).not.toContain('Resume unfinished runs');
    expect(readonly).not.toContain('Checking the training runtime');
  });

  it.each(['planning', 'finished'] as const)('enforces %s view-only rules even with stale running execution data', (stage) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['training-execution', 'project', batch.id], execution);
    client.setQueryData(['training-runtime', 'project'], runtime);
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentExecution project="project" batch={batch} implemented view="batches" stage={stage} knownExecution={execution} /></QueryClientProvider>);
      expect(html).not.toContain('Cancel batch');
      expect(html).not.toContain('Launch batch');
      expect(html).not.toContain('Resume unfinished runs');
      if (stage === 'planning') expect(html).toContain('Runs and results are locked during planning');
    } finally { client.clear(); }
  });

  it('does not label a failed finished experiment successful or publish scores from incomplete folds', () => {
    const html = renderToStaticMarkup(<ResultsTable batch={batch} loading={false} results={{ status: 'failed', oof: [], candidates: [{ candidateId: run.candidateId, trainingSeed: 42, splitSeed: 7, complete: false, metrics: { auroc: 0.9876, accuracy: 0.9876 }, completedRuns: 1, totalRuns: 5, oofPath: '/unfinished.csv' }] }} />);
    expect(html).toContain('incomplete groups');
    expect(html).toContain('Waiting for all folds');
    expect(html).not.toContain('0.9876');
    expect(html).not.toContain('/unfinished.csv');
    expect(html).not.toContain('predictor');
  });
});
