import { describe, expect, it } from 'vitest';
import { fixturePredictor } from '../testFixtures/predictors';
import { experimentPredictorLink, groupPredictors, predictorConfigurationLabel, predictorMatches } from './predictorGroups';


describe('experiment predictor grouping', () => {
  it('keeps 3 seeds × 15 configurations × both methods as 90 predictors across five folds', () => {
    const items = Array.from({ length: 15 }, (_, configuration) => [11, 22, 33].flatMap((seed) => (['ensemble', 'refit'] as const).map((method) => fixturePredictor(configuration + 1, seed, method)))).flat();
    const originalIds = items.map((item) => item.id);
    const groups = groupPredictors([...items].reverse());
    expect(groups).toHaveLength(1); expect(groups[0].items).toHaveLength(90);
    expect(groups[0].items.filter((item) => item.manifest.method === 'ensemble')).toHaveLength(45);
    expect(groups[0].items.filter((item) => item.manifest.method === 'refit')).toHaveLength(45);
    expect(groups[0].items.map((item) => item.id)).toEqual(originalIds);
    expect(items.map((item) => item.id)).toEqual(originalIds);
  });

  it('separates same-name experiments and matches configuration, experiment and predictor methods', () => {
    const first = fixturePredictor(1, 11, 'ensemble', 'one');
    const second = fixturePredictor(1, 11, 'refit', 'two');
    expect(groupPredictors([first, second]).map((group) => group.id)).toEqual(['one', 'two']);
    expect(predictorMatches(first, 'one', 'ensemble', 'candidate-1')).toBe(true);
    expect(predictorMatches(first, 'two', 'all', '')).toBe(false);
    expect(predictorMatches(second, '', 'ensemble', '')).toBe(false);
    expect(predictorMatches(second, '', 'refit', ' SAME STUDY ')).toBe(true);
    expect(predictorConfigurationLabel({ candidateId: 'candidate-hash', candidateNumber: 7 })).toBe('Configuration 7');
    expect(experimentPredictorLink('legacy/one', 'predictor?two')).toBe('#experiments?experiment=legacy%2Fone&tab=predictors&predictor=predictor%3Ftwo');
  });
});
