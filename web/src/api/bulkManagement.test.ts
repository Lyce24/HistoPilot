import { afterEach, describe, expect, it, vi } from 'vitest';
import type { BulkEvaluationSelection } from './bulkEvaluations';
import type { PredictorBuildSelection, PredictorSource } from './predictorBuilds';

const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
const source: PredictorSource = { experimentId: 'experiment/one', batchId: 'batch/one', candidateId: 'candidate/one', trainingSeed: 11, splitSeed: 42 };

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('reviewed bulk predictor and evaluation API contracts', () => {
  it('encodes project and evaluation batch IDs for list, details and cancellation', async () => {
    const fetcher = vi.fn().mockImplementation(async (path: string) => response(path.endsWith('/session') ? { token: 'session' } : { items: [] }));
    vi.stubGlobal('fetch', fetcher);
    const { bulkEvaluations } = await import('./bulkEvaluations');
    await bulkEvaluations.list('project/one');
    await bulkEvaluations.get('project/one', 'batch/one');
    await bulkEvaluations.cancel('project/one', 'batch/one', 'cancel-once');
    expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
      '/api/v1/projects/project%2Fone/evaluation-runs/bulk',
      '/api/v1/projects/project%2Fone/evaluation-runs/bulk/batch%2Fone',
      '/api/v1/projects/project%2Fone/evaluation-runs/bulk/batch%2Fone/cancel',
    ]);
    expect(fetcher.mock.calls[3][1].headers.get('X-HistoPilot-Token')).toBe('session');
    expect(JSON.parse(fetcher.mock.calls[3][1].body)).toEqual({ operationId: 'cancel-once' });
  });

  it('reviews all active predictors without supplying IDs and commits only the reviewed snapshot', async () => {
    const preview = { canRun: true, previewHash: 'review-hash', reviewedPredictorIds: ['predictor/one', 'predictor/two'], eligibleCount: 1, blockedCount: 1, items: [] };
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response(preview)).mockResolvedValueOnce(response({ id: 'batch', status: 'queued' }));
    vi.stubGlobal('fetch', fetcher);
    const { bulkEvaluations } = await import('./bulkEvaluations');
    const selection: BulkEvaluationSelection = { cohortId: 'cohort/one', scope: 'all', namePrefix: 'External cohort' };
    const reviewed = await bulkEvaluations.preview('project/one', selection);
    await bulkEvaluations.run('project/one', selection, { previewHash: reviewed.previewHash, reviewedPredictorIds: reviewed.reviewedPredictorIds }, 'run-once');
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/evaluation-runs/bulk/preview');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(selection);
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).not.toHaveProperty('predictorIds');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ ...selection, previewHash: 'review-hash', reviewedPredictorIds: ['predictor/one', 'predictor/two'], operationId: 'run-once' });
    expect(fetcher.mock.calls).toHaveLength(3);
  });

  it('preserves selected predictor IDs and explicitly retries a lost submission with the identical operation', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockRejectedValueOnce(new TypeError('Response lost')).mockResolvedValueOnce(response({ id: 'batch', status: 'queued' }));
    vi.stubGlobal('fetch', fetcher);
    const { bulkEvaluations } = await import('./bulkEvaluations');
    const selection: BulkEvaluationSelection = { cohortId: 'cohort/one', scope: 'selected', predictorIds: ['predictor/two', 'predictor/one'] };
    const review = { previewHash: 'review-hash', reviewedPredictorIds: ['predictor/one', 'predictor/two'] };
    await expect(bulkEvaluations.run('p', selection, review, 'stable-operation')).rejects.toThrow('Response lost');
    expect(fetcher.mock.calls).toHaveLength(2);
    await bulkEvaluations.run('p', selection, review, 'stable-operation');
    expect(fetcher.mock.calls[1][1].body).toBe(fetcher.mock.calls[2][1].body);
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ ...selection, ...review, operationId: 'stable-operation' });
  });

  it('keeps Both and every configuration and seed identity intact through predictor review and creation', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response({ canBuild: true, previewHash: 'review', items: [], findings: [], counts: { total: 4 } })).mockResolvedValueOnce(response({ operationId: 'build-once', status: 'completed', items: [] }));
    vi.stubGlobal('fetch', fetcher);
    const { predictorBuilds, predictorSourceKey } = await import('./predictorBuilds');
    const other = { ...source, trainingSeed: 12, splitSeed: 43 };
    const selection: PredictorBuildSelection = { selections: [source, other], method: 'both', refitPercentile: 75, namePrefix: 'Candidate family' };
    await predictorBuilds.preview('project/one', selection);
    await predictorBuilds.create('project/one', selection, 'review', 'build-once');
    expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
      '/api/v1/projects/project%2Fone/predictors/builds/preview',
      '/api/v1/projects/project%2Fone/predictors/builds',
    ]);
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(selection);
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ ...selection, previewHash: 'review', operationId: 'build-once' });
    expect(predictorSourceKey(source)).not.toBe(predictorSourceKey(other));
    expect(predictorSourceKey(source)).not.toBe(predictorSourceKey({ ...source, trainingSeed: 12 }));
    expect(predictorSourceKey(source)).not.toBe(predictorSourceKey({ ...source, splitSeed: 43 }));
    expect(predictorSourceKey(source)).not.toBe(predictorSourceKey({ ...source, batchId: 'batch/two' }));
    expect(predictorSourceKey(source)).not.toBe(predictorSourceKey({ ...source, candidateId: 'candidate/two' }));
  });

  it('does not recreate Both after a lost response until the same reviewed request is explicitly retried', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockRejectedValueOnce(new TypeError('Response lost')).mockResolvedValueOnce(response({ operationId: 'build-once', status: 'partial', items: [] }));
    vi.stubGlobal('fetch', fetcher);
    const { predictorBuilds } = await import('./predictorBuilds');
    const selection: PredictorBuildSelection = { selections: [source], method: 'both', refitPercentile: 50 };
    await expect(predictorBuilds.create('p', selection, 'review', 'build-once')).rejects.toThrow('Response lost');
    expect(fetcher.mock.calls).toHaveLength(2);
    await predictorBuilds.create('p', selection, 'review', 'build-once');
    expect(fetcher.mock.calls[1][1].body).toBe(fetcher.mock.calls[2][1].body);
  });
});
