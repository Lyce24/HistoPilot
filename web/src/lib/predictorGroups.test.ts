import { describe, expect, it } from 'vitest';
import { fixturePredictor, fixtureSeedEnsemble } from '../testFixtures/predictors';
import type { ExperimentFiling, Objective } from '../api/objectives';
import { experimentPredictorLink, groupPredictors, predictorConfigurationLabel, predictorMatches, predictorOptionGroups } from './predictorGroups';


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
  it('lists a configuration\'s seed ensemble before its per-seed predictors and finds it by its seeds', () => {
    const pooled = fixtureSeedEnsemble(1);
    const perSeed = fixturePredictor(1, 11, 'ensemble');
    expect(groupPredictors([perSeed, pooled])[0].items.map((item) => item.id)).toEqual([pooled.id, perSeed.id]);
    expect(predictorMatches(pooled, '', 'seed_ensemble', '3 training')).toBe(true);
    expect(predictorMatches(pooled, '', 'ensemble', '')).toBe(false);
  });

  it('orders a picker\'s groups as the project is organized and names objective, role and member', () => {
    const objectives = (['aim-1', 'aim-2'] as const).map((id, index): Objective => ({ id, name: `Aim ${index + 1}`, rank: index ? 'secondary' : 'primary', description: '', revision: 1, createdAt: '', updatedAt: '' }));
    const filing = (objectiveId: string | null, role: ExperimentFiling['role'], series: string | null = null, seriesMember: string | null = null): ExperimentFiling => ({ objectiveId, role, series, seriesMember });
    const items = ['loose', 'loco-tcga', 'main', 'loco-cptac', 'clinical', 'gone'].map((experiment) => fixturePredictor(1, 11, 'refit', experiment));
    const filings = {
      main: filing('aim-1', 'primary'), clinical: filing('aim-1', 'comparison'),
      'loco-tcga': filing('aim-2', 'generalization', 'LOCO', 'tcga'), 'loco-cptac': filing('aim-2', 'generalization', 'LOCO', 'cptac'),
      gone: filing('objective-removed', 'variant'),
    };
    const groups = predictorOptionGroups(items, filings, objectives);
    expect(groups.map((group) => group.key)).toEqual(['main', 'clinical', 'loco-cptac', 'loco-tcga', 'gone', 'loose']);
    expect(groups[2].label).toBe('Aim 2 › Same study name · Generalization › cptac');
    expect(groups[4].label).toBe('No objective › Same study name · Variant');
    expect(groups.at(-1)?.label).toBe('No objective › Same study name');
    // Without objectives the groups are named by their experiment alone.
    expect(new Set(predictorOptionGroups(items, {}, []).map((group) => group.label))).toEqual(new Set(['Same study name']));
  });

  it('groups a predictor saved without its experiment instead of failing', () => {
    const legacy = fixturePredictor(1, 11, 'refit');
    const manifest = legacy.manifest as Partial<typeof legacy.manifest>;
    delete manifest.experimentId; delete manifest.experiment; delete manifest.batchId; delete manifest.candidateId;
    const groups = predictorOptionGroups([legacy, fixturePredictor(1, 11, 'refit', 'other')], {}, []);
    expect(groups.flatMap((group) => group.items.map((item) => item.id))).toContain(legacy.id);
  });
});
