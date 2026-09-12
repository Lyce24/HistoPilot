import { describe, expect, it } from 'vitest';
import { fixturePredictor } from '../testFixtures/predictors';
import { fixtureExperiment } from '../testFixtures/evaluations';
import { evaluationExperiments, experimentPredictors } from './evaluationSelection';

const items = [fixturePredictor(1, 11, 'ensemble', 'one'), fixturePredictor(1, 11, 'refit', 'one'), fixturePredictor(1, 11, 'ensemble', 'two')];
describe('experiment-first evaluation selection', () => {
  it('defaults to no predictors and supports an explicit selection of multiple experiments', () => {
    expect(experimentPredictors(items, [], 'both')).toEqual([]);
    expect(experimentPredictors(items, ['one', 'two'], 'both')).toHaveLength(3);
    expect(experimentPredictors(items, ['one', 'two'], 'refit')).toEqual([items[1]]);
    expect(experimentPredictors(items, ['one'], 'ensemble')).toEqual([items[0]]);
  });
  it('shows planning/pending/Skip experiments without inventing ready predictions', () => {
    const records = [fixtureExperiment('planning', 'planning'), fixtureExperiment('pending', 'running'), fixtureExperiment('skip')];
    records[2].predictorPolicy = { method: 'skip', refitPercentile: null };
    const options = evaluationExperiments(records, items, ['missing']);
    expect(options.find((item) => item.id === 'planning')?.description).toBe('Planning · No ready predictors');
    expect(options.find((item) => item.id === 'pending')?.ready).toBe(0);
    expect(options.find((item) => item.id === 'skip')?.ready).toBe(0);
    expect(options.find((item) => item.id === 'missing')?.description).toContain('unavailable');
    expect(options.find((item) => item.id === 'one')).toMatchObject({ ensemble: 1, refit: 1, ready: 2 });
  });
  it('ignores archived/trashed weights while preserving same-name experiment identities', () => {
    const retained = [items[0], { ...items[1], lifecycleState: 'archived' as const }, { ...items[2], lifecycleState: 'trashed' as const }];
    expect(experimentPredictors(retained, ['one', 'two'], 'both')).toEqual([items[0]]);
    expect(evaluationExperiments([], items).map((item) => item.id)).toEqual(['one', 'two']);
  });
});
