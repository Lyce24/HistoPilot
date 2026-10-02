import { describe, expect, it } from 'vitest';
import type { Interpretation } from '../api/interpretation';
import { interpretationStudies } from './interpretationStudies';

function saved(id: string, minute: number, values: Record<string, unknown> = {}): Interpretation {
  return { id, createdAt: new Date(Date.UTC(2026, 8, 30, 10, minute)).toISOString(), contentHash: id, manifest: {
    kind: 'model-interpretation', name: id, predictorId: 'model', experimentId: 'experiment', encoderId: 'uni', method: 'refit', memberCount: 1,
    featureBundleId: 'bundle', packArtifactId: null, slideFolder: '/slides', slides: [{}], ...values,
  } } as unknown as Interpretation;
}

describe('saved attention studies', () => {
  it('groups records saved together for one model, features and folder, newest study first', () => {
    const records = [saved('a', 0), saved('other-model', 5, { predictorId: 'other' }), saved('b', 20, { slides: [{}, {}] }), saved('c', 45), saved('d', 80), saved('pack', 21, { packArtifactId: 'pack-1' })];
    const studies = interpretationStudies([...records].reverse());
    expect(studies.map((study) => study.records.map((record) => record.id))).toEqual([['d'], ['pack'], ['other-model'], ['a', 'b', 'c']]);
    // A study continues while each record follows the last within the gap, however long it runs.
    expect(studies.at(-1)).toMatchObject({ key: 'a', predictorId: 'model', experimentId: 'experiment', slides: 4, createdAt: records[0].createdAt });
    expect(interpretationStudies(records, 10).map((study) => study.key)).toEqual(['d', 'c', 'pack', 'b', 'other-model', 'a']);
    expect(interpretationStudies([])).toEqual([]);
  });
});
