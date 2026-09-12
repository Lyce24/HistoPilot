import { withRecipeDefaults, type DevelopmentBatchSpec } from '../api/development';
import type { ExperimentPredictorPolicy, ModelExperiment } from '../api/experiments';
import type { ProtocolSpec } from '../api/scientific';
import { sameJSON } from './json';

export const defaultPredictorPolicy = (): ExperimentPredictorPolicy => ({ method: 'ensemble', refitPercentile: null });
export const predictorPolicyLabel = (policy: ExperimentPredictorPolicy) => ({ skip: 'Skip', refit: 'Refit', ensemble: 'Ensemble', both: 'Both' })[policy.method];
export const includesRefit = (policy: ExperimentPredictorPolicy) => policy.method === 'refit' || policy.method === 'both';

/** The service deduplicates scientific configurations, including repeated explicit rows. */
export function plannedConfigurationCount(spec: Pick<DevelopmentBatchSpec, 'mode' | 'grid' | 'configurations'>) {
  if (spec.mode === 'grid') return new Set(spec.grid.learningRates).size * new Set(spec.grid.weightDecays).size * new Set(spec.grid.maxEpochs).size;
  if (spec.mode === 'explicit') {
    const unique: ReturnType<typeof withRecipeDefaults>[] = [];
    for (const recipe of spec.configurations) {
      const normalized = withRecipeDefaults(recipe);
      if (!unique.some((item) => sameJSON(item, normalized))) unique.push(normalized);
    }
    return unique.length;
  }
  return 1;
}

/** Planning estimate; the frozen backend manifest remains authoritative at submission. */
export function experimentPredictorCount(record: ModelExperiment, policy: ExperimentPredictorPolicy, protocol?: ProtocolSpec) {
  let groups = 0;
  let folds = 0;
  const batches = record.batches.filter((batch) => batch.state === 'active');
  for (const batch of batches) {
    const summary = batch.manifest.summary;
    const splitSeeds = new Set(batch.manifest.splitPlans.map((plan) => plan.seed ?? 0)).size;
    groups += summary.configurationCount * summary.trainingSeedCount * splitSeeds;
    folds += summary.runCount;
  }
  // Submitted records retain their planning recipes as history: never count them twice.
  if (!record.configurationLocked && !record.submission && (record.batchPlans?.length ?? 0) > 0) {
    if (!protocol || protocol.split.mode !== 'kfold') return null;
    for (const plan of record.batchPlans ?? []) {
      const planGroups = plannedConfigurationCount(plan.spec) * new Set(plan.spec.trainingSeeds).size * new Set(protocol.split.seeds).size;
      groups += planGroups;
      folds += planGroups * protocol.split.folds;
    }
  }
  const ensembles = policy.method === 'ensemble' || policy.method === 'both' ? groups : 0;
  const refits = includesRefit(policy) ? groups : 0;
  return { groups, foldRuns: folds, ensembles, refits, total: ensembles + refits };
}
