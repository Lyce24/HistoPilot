import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EvaluationSelection, PredictorSelection } from './predictors';

const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const selected: PredictorSelection = { experimentId: 'draft-experiment', batchId: 'configuration-batch', candidateId: 'candidate-model', trainingSeed: 11, splitSeed: 42, name: 'Baseline predictor' };
const evaluation: EvaluationSelection = { predictorId: 'configuration-predictor', cohortId: 'configuration-cohort', name: 'External validation' };

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('experiment predictor and evaluation API contracts', () => {
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

  it('previews a full seed group using the exact stable experiment and batch identifiers', async () => {
    const blocked = { canFreeze: false, previewHash: null, manifest: null, findings: [{ severity: 'error', code: 'PREDICTOR_RUN_INCOMPLETE', message: 'Every fold must finish.' }] };
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response(blocked));
    vi.stubGlobal('fetch', fetcher);
    const { predictors } = await import('./predictors');
    expect(await predictors.preview('project', selected)).toEqual(blocked);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project/predictors/preview');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(selected);
    expect(fetcher.mock.calls).toHaveLength(2);
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).not.toHaveProperty('runIds');
  });

  it('leaves an uncertain freeze to an explicit retry with the same reviewed operation', async () => {
    const saved = { id: 'predictor', lifecycleState: 'active', manifest: { ...selected, kind: 'frozen-predictor' } };
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockRejectedValueOnce(new TypeError('Response lost'))
      .mockResolvedValueOnce(response(saved));
    vi.stubGlobal('fetch', fetcher);
    const { predictors } = await import('./predictors');
    await expect(predictors.freeze('project', selected, 'review-hash', 'stable-operation')).rejects.toThrow('Response lost');
    expect(fetcher.mock.calls).toHaveLength(2);
    expect(await predictors.freeze('project', selected, 'review-hash', 'stable-operation')).toEqual(saved);
    expect(fetcher.mock.calls[1][1].body).toBe(fetcher.mock.calls[2][1].body);
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ ...selected, previewHash: 'review-hash', operationId: 'stable-operation' });
  });

  it('preserves a stale-promotion conflict instead of creating another predictor', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockResolvedValueOnce(response({ code: 'EXPERIMENT_ALREADY_FROZEN', detail: 'Restore the existing predictor.' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { predictors } = await import('./predictors');
    await expect(predictors.freeze('p', selected, 'review-hash', 'operation')).rejects.toMatchObject({ status: 409, code: 'EXPERIMENT_ALREADY_FROZEN' });
    expect(fetcher.mock.calls).toHaveLength(2);
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
});

it('keeps refit planning, training, cancellation and publication as separate authenticated actions', async () => {
  const fetcher = vi.fn().mockImplementation(async (path: string) => response(path.endsWith('/session') ? { token: 'session' } : { status: 'queued' }));
  vi.stubGlobal('fetch', fetcher);
  const { predictors, modelEvaluations } = await import('./predictors');
  const selection = { ...selected, method: 'refit' as const, refitPercentile: 75 };
  await predictors.planRefit('p/a', selection, 'review', 'save');
  await predictors.refitJob('p/a', 'refit/a', 'launch', 'launch');
  await predictors.refitJob('p/a', 'refit/a', 'cancel', 'cancel');
  await predictors.publishRefit('p/a', 'refit/a', 'publish');
  await modelEvaluations.job('p/a', 'eval/a', 'resume', 'retry');
  expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
    '/api/v1/projects/p%2Fa/predictors/refits',
    '/api/v1/projects/p%2Fa/predictors/refits/refit%2Fa/launch',
    '/api/v1/projects/p%2Fa/predictors/refits/refit%2Fa/cancel',
    '/api/v1/projects/p%2Fa/predictors/refits/refit%2Fa/publish',
    '/api/v1/projects/p%2Fa/evaluation-runs/eval%2Fa/resume',
  ]);
  expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ ...selection, previewHash: 'review', operationId: 'save' });
  expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ operationId: 'launch' });
});
