import { describe, expect, it } from 'vitest';
import { defaultRecipe, defaultResources, type DevelopmentBatchSpec } from '../api/development';
import type { ModelExperiment } from '../api/experiments';
import type { ProtocolSpec } from '../api/scientific';
import { experimentPredictorCount, plannedConfigurationCount } from './experimentPredictors';

const spec: DevelopmentBatchSpec = { version: 1, experimentName: 'Count', batchName: 'Grid', inputs: { protocolId: 'p', featureBundleId: 'f', loadingPolicy: 'native', packArtifactId: null }, recipe: defaultRecipe(), mode: 'grid', grid: { learningRates: [0.0001, 0.0002, 0.0003, 0.0004, 0.0005], weightDecays: [0, 0.0001, 0.001], maxEpochs: [100] }, configurations: [], trainingSeeds: [1, 2, 3], resources: defaultResources(), notes: '' };
const record = { batchPlans: [{ id: 'grid', spec }], batches: [], configurationLocked: false } as unknown as ModelExperiment;
const protocol = { split: { mode: 'kfold', seeds: [42], folds: 5 } } as ProtocolSpec;

describe('experiment predictor planning counts', () => {
  it('turns 225 fold runs into 90 predictors across both methods', () => {
    expect(experimentPredictorCount(record, { method: 'both', refitPercentile: 75 }, protocol)).toEqual({ groups: 45, foldRuns: 225, ensembles: 45, refits: 45, total: 90 });
  });
  it('includes split seeds but never multiplies predictors by folds', () => {
    expect(experimentPredictorCount(record, { method: 'refit', refitPercentile: 50 }, { ...protocol, split: { ...protocol.split, seeds: [42, 2026], folds: 10 } })).toEqual({ groups: 90, foldRuns: 900, ensembles: 0, refits: 90, total: 90 });
    expect(experimentPredictorCount(record, { method: 'skip', refitPercentile: null }, protocol)).toEqual({ groups: 45, foldRuns: 225, ensembles: 0, refits: 0, total: 0 });
  });
  it('deduplicates default-equivalent explicit recipes like backend expansion', () => {
    const { dropout: _dropout, ...implicit } = defaultRecipe();
    expect(plannedConfigurationCount({ ...spec, mode: 'explicit', configurations: [defaultRecipe(), implicit, { ...defaultRecipe(), learningRate: 0.0001 }] })).toBe(2);
  });
  it('counts the frozen manifest once even when submission retains editable recipe history', () => {
    const submitted = { ...record, configurationLocked: true, submission: {}, batches: [{ state: 'active', manifest: { summary: { configurationCount: 15, trainingSeedCount: 3, runCount: 225 }, splitPlans: Array.from({ length: 5 }, (_, fold) => ({ fold, seed: 42 })) } }] } as unknown as ModelExperiment;
    expect(experimentPredictorCount(submitted, { method: 'both', refitPercentile: 75 }, protocol)?.total).toBe(90);
  });
  it('waits for a saved k-fold protocol instead of inventing a split seed count', () => {
    expect(experimentPredictorCount(record, { method: 'ensemble', refitPercentile: null })).toBeNull();
    expect(experimentPredictorCount(record, { method: 'ensemble', refitPercentile: null }, { ...protocol, split: { ...protocol.split, mode: 'holdout' } })).toBeNull();
  });
});
