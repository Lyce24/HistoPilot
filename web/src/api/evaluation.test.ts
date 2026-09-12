import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EvaluationSpec } from './evaluation';

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const spec: EvaluationSpec = {
  protocolId: 'development', developmentFeatureBundleId: 'development-bundle', datasetId: 'shared-dataset', featureBundleId: 'test-bundle',
  target: null, eligibility: [{ field: 'cohort', op: 'eq', value: 'test' }], patientIdentifiers: 'shared',
  inference: { loadingPolicy: 'packed', packArtifactId: 'test-pack', batchSize: 1, numWorkers: 2, device: 'auto', precision: 'float32', patientAggregation: 'mean', decisionThreshold: 0.5 },
};

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('test cohort persistence API', () => {
  it('saves the cohort independently of model runs and freezes the exact reviewed revision', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { taggedFreeze: true } }))
      .mockResolvedValueOnce(json({ id: 'draft/id', revision: 1, payload: { type: 'evaluation-cohort', spec } }))
      .mockResolvedValueOnce(json({ id: 'draft/id', revision: 2, payload: { type: 'evaluation-cohort', spec } }))
      .mockResolvedValueOnce(json({ previewHash: 'checked-content', canFreeze: true, executionEnabled: false }))
      .mockResolvedValueOnce(json({ id: 'frozen', manifest: { kind: 'evaluation-cohort', spec } }));
    vi.stubGlobal('fetch', fetcher);
    const { evaluation } = await import('./evaluation');
    const created = await evaluation.saveDraft('project/one', 'Later cohort', spec);
    const saved = await evaluation.saveDraft('project/one', 'Renamed cohort', spec, created);
    const preview = await evaluation.preview('project/one', saved);
    await evaluation.freeze('project/one', saved, preview.previewHash, 'freeze-operation', { tag: 'held-out-v1', note: 'Prepared during model development' });
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/drafts');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ kind: 'experiment', name: 'Later cohort', payload: { type: 'evaluation-cohort', spec } });
    expect(fetcher.mock.calls[2][1].method).toBe('PATCH');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toMatchObject({ expectedRevision: 1, name: 'Renamed cohort' });
    expect(fetcher.mock.calls[3][0]).toBe('/api/v1/projects/project%2Fone/drafts/draft%2Fid/evaluation-preview');
    expect(JSON.parse(fetcher.mock.calls[3][1].body)).toEqual({ expectedRevision: 2 });
    expect(JSON.parse(fetcher.mock.calls[4][1].body)).toEqual({ expectedRevision: 2, previewHash: 'checked-content', operationId: 'freeze-operation', versionLabel: { tag: 'held-out-v1', note: 'Prepared during model development' } });
    expect(spec).not.toHaveProperty('split');
    expect(spec).not.toHaveProperty('predictorId');
  });

  it('propagates stale-review conflicts without freezing a different draft or dropping its tag', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { taggedFreeze: true } }))
      .mockResolvedValueOnce(json({ detail: 'Selected feature coverage changed. Review this cohort again.' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { evaluation } = await import('./evaluation');
    await expect(evaluation.freeze('project', { id: 'draft', revision: 3 }, 'previous-check', 'operation', { tag: 'test-v2', note: '' })).rejects.toMatchObject({ status: 409, message: 'Selected feature coverage changed. Review this cohort again.' });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toMatchObject({ expectedRevision: 3, previewHash: 'previous-check', versionLabel: { tag: 'test-v2' } });
  });

  it('lists only evaluation drafts and preserves freshness evidence on frozen cohorts', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ drafts: [
        { id: 'protocol', payload: { type: 'analysis-protocol' } },
        { id: 'model', payload: { type: 'mil-experiment' } },
        { id: 'test', payload: { type: 'evaluation-cohort', spec } },
      ] }))
      .mockResolvedValueOnce(json({ items: [{ id: 'cohort', current: false, findings: [{ severity: 'error', code: 'STALE_FEATURE_BUNDLE', message: 'Verify test features.' }] }] }));
    vi.stubGlobal('fetch', fetcher);
    const { evaluation } = await import('./evaluation');
    expect((await evaluation.drafts('project')).drafts.map((draft) => draft.id)).toEqual(['test']);
    expect((await evaluation.list('project')).items[0]).toMatchObject({ current: false, findings: [{ code: 'STALE_FEATURE_BUNDLE' }] });
  });
});
