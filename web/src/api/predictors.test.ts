import { afterEach, describe, expect, it, vi } from 'vitest';
import { computePollInterval, hasPatientPredictions, predictorMethodLabel, predictorSeedLabel } from './predictors';
import type { ComputeExecution, EvaluationSelection } from './predictors';
import { fixturePredictor, fixtureSeedEnsemble } from '../testFixtures/predictors';

const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const evaluation: EvaluationSelection = { predictorId: 'configuration-predictor', cohortId: 'configuration-cohort', name: 'External validation' };

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('evaluation result files', () => {
  it('offers patient predictions only when the evaluation wrote them', () => {
    const file = { path: '/p', bytes: 1, sha256: 'x' };
    expect(hasPatientPredictions({ artifacts: { 'slide-predictions.csv': file, 'patient-predictions.csv': file } })).toBe(true);
    // Slide-split evaluations never write patient-predictions.csv.
    expect(hasPatientPredictions({ artifacts: { 'slide-predictions.csv': file }, metrics: { splitUnit: 'slide' } as never })).toBe(false);
    expect(hasPatientPredictions({ artifacts: { 'slide-predictions.csv': file } })).toBe(false);
    // Results without an artifact list fall back to the split unit.
    expect(hasPatientPredictions({ metrics: { splitUnit: 'slide' } as never })).toBe(false);
    expect(hasPatientPredictions({ metrics: { splitUnit: 'patient' } as never })).toBe(true);
    expect(hasPatientPredictions({})).toBe(true);
  });
});

describe('seed ensembles', () => {
  it('names the method and describes the seeds of either kind of predictor', () => {
    expect(predictorMethodLabel('seed_ensemble')).toBe('Seed ensemble');
    expect(predictorMethodLabel('ensemble')).toBe('Fold ensemble');
    expect(predictorSeedLabel(fixtureSeedEnsemble(1).manifest)).toBe('3 training × 1 split seed · 15 models');
    expect(predictorSeedLabel(fixturePredictor(1, 11, 'ensemble').manifest)).toBe('Train 11 / split 42');
  });
  it('lists, reviews and freezes seed ensembles on their own routes', async () => {
    const selection = { experimentId: 'exp', batchId: 'batch', candidateId: 'candidate', name: 'Pooled' };
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockResolvedValueOnce(response({ items: [] }))
      .mockResolvedValueOnce(response({ canFreeze: true, previewHash: 'review', findings: [], manifest: null }))
      .mockResolvedValueOnce(response({ id: 'predictor' }, 201));
    vi.stubGlobal('fetch', fetcher);
    const { predictors } = await import('./predictors');
    await predictors.seedEnsembles('project/one', 'exp/1');
    await predictors.previewSeedEnsemble('project/one', selection);
    await predictors.freezeSeedEnsemble('project/one', selection, 'review', 'build-once');
    expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
      '/api/v1/projects/project%2Fone/predictors/seed-ensembles?experiment_id=exp%2F1',
      '/api/v1/projects/project%2Fone/predictors/seed-ensembles/preview',
      '/api/v1/projects/project%2Fone/predictors/seed-ensembles',
    ]);
    expect(JSON.parse(fetcher.mock.calls[3][1].body)).toEqual({ ...selection, previewHash: 'review', operationId: 'build-once' });
  });
});

describe('experiment predictor and evaluation API contracts', () => {
  it('compares fixed evaluation identities with an authenticated read-only computation', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response({ statistics: { available: true } }));
    vi.stubGlobal('fetch', fetcher);
    const { modelEvaluations } = await import('./predictors');
    await modelEvaluations.compare('project/one', 'left', 'right');
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/evaluation-runs/compare');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ leftEvaluationId: 'left', rightEvaluationId: 'right' });
  });
  it('loads all registry states while asking the server for scoped complete-fold choices', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockResolvedValueOnce(response({ items: [], executionEnabled: false }))
      .mockResolvedValueOnce(response({ items: [], executionEnabled: false }))
      .mockResolvedValueOnce(response({ items: [], executionEnabled: false }));
    vi.stubGlobal('fetch', fetcher);
    const { predictors, modelEvaluations } = await import('./predictors');
    await predictors.list('project/one');
    await predictors.choices('project/one');
    await modelEvaluations.list('project/one');
    expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
      '/api/v1/projects/project%2Fone/predictors?include_inactive=true',
      '/api/v1/projects/project%2Fone/predictors/choices',
      '/api/v1/projects/project%2Fone/evaluation-runs?include_inactive=true',
    ]);
    expect(fetcher.mock.calls[1][1].headers.get('X-HistoPilot-Token')).toBe('session');
  });

  it('saves an explicit predictor/cohort plan without manufacturing inference or launch calls', async () => {
    const preview = { canSave: true, previewHash: 'evaluation-review', findings: [], manifest: { ...evaluation, kind: 'model-evaluation', status: 'planned', results: null }, executionEnabled: false };
    const saved = { id: 'evaluation', lifecycleState: 'active', manifest: preview.manifest };
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockResolvedValueOnce(response(preview)).mockResolvedValueOnce(response(saved));
    vi.stubGlobal('fetch', fetcher);
    const { modelEvaluations } = await import('./predictors');
    expect(await modelEvaluations.preview('p', evaluation)).toEqual(preview);
    expect(await modelEvaluations.save('p', evaluation, 'evaluation-review', 'save-plan')).toEqual(saved);
    expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
      '/api/v1/projects/p/evaluation-runs/preview', '/api/v1/projects/p/evaluation-runs',
    ]);
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ ...evaluation, previewHash: 'evaluation-review', operationId: 'save-plan' });
    expect(saved.manifest.results).toBeNull();
    expect(saved.manifest.status).toBe('planned');
  });

  it('binds feature and pack choices to both single and batch evaluation reviews and submissions', async () => {
    const fetcher = vi.fn().mockImplementation(async (path: string) => response(path.endsWith('/session') ? { token: 'session' } : {}));
    vi.stubGlobal('fetch', fetcher);
    const { modelEvaluations } = await import('./predictors');
    const { bulkEvaluations } = await import('./bulkEvaluations');
    const inference = { loadingPolicy: 'packed' as const, packArtifactId: 'test-pack', batchSize: 1, numWorkers: 0, device: 'cpu' as const, precision: 'float32' as const, patientAggregation: 'mean' as const, decisionThreshold: 0 };
    const single = { ...evaluation, featureBundleId: 'test-features', inference, patientIdentifiers: 'independent' as const };
    const batch = { cohortId: evaluation.cohortId, scope: 'selected' as const, predictorIds: [evaluation.predictorId], featureBundleId: 'test-features', inference, patientIdentifiers: 'independent' as const };
    await modelEvaluations.preview('p', single);
    await modelEvaluations.save('p', single, 'single-review', 'single-operation');
    await bulkEvaluations.preview('p', batch);
    await bulkEvaluations.run('p', batch, { previewHash: 'batch-review', reviewedPredictorIds: batch.predictorIds }, 'batch-operation');
    for (const [, options] of fetcher.mock.calls.slice(1)) expect(JSON.parse(options.body)).toMatchObject({ cohortId: evaluation.cohortId, featureBundleId: 'test-features', inference, patientIdentifiers: 'independent' });
    expect(JSON.parse(fetcher.mock.calls[4][1].body)).toMatchObject({ reviewedPredictorIds: batch.predictorIds, previewHash: 'batch-review', operationId: 'batch-operation' });
  });
});

it('keeps refit training, cancellation and publication as separate authenticated actions', async () => {
  const fetcher = vi.fn().mockImplementation(async (path: string) => response(path.endsWith('/session') ? { token: 'session' } : { status: 'queued' }));
  vi.stubGlobal('fetch', fetcher);
  const { predictors, modelEvaluations } = await import('./predictors');
  await predictors.refitJob('p/a', 'refit/a', 'launch', 'launch');
  await predictors.refitJob('p/a', 'refit/a', 'cancel', 'cancel');
  await predictors.publishRefit('p/a', 'refit/a', 'publish');
  await modelEvaluations.job('p/a', 'eval/a', 'resume', 'retry');
  expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
    '/api/v1/projects/p%2Fa/predictors/refits/refit%2Fa/launch',
    '/api/v1/projects/p%2Fa/predictors/refits/refit%2Fa/cancel',
    '/api/v1/projects/p%2Fa/predictors/refits/refit%2Fa/publish',
    '/api/v1/projects/p%2Fa/evaluation-runs/eval%2Fa/resume',
  ]);
  expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ operationId: 'launch' });
});

describe('compute job polling', () => {
  it('refreshes quickly only while a list has a running or queued job', () => {
    expect(computePollInterval(undefined)).toBe(30000);
    expect(computePollInterval([])).toBe(30000);
    expect(computePollInterval([{ execution: { status: 'completed' } as ComputeExecution }])).toBe(30000);
    expect(computePollInterval([{ execution: { status: 'not_started' } as ComputeExecution }])).toBe(30000);
    for (const status of ['queued', 'running'] as const) {
      expect(computePollInterval([{ execution: { status } as ComputeExecution }])).toBe(5000);
    }
  });
});
