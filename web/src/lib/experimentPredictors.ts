import { withRecipeDefaults, type DevelopmentBatchSpec } from '../api/development';
import type { ExperimentPredictorPolicy, ModelExperiment } from '../api/experiments';
import type { ProtocolSpec } from '../api/scientific';
import { sameJSON } from './json';

export const defaultPredictorPolicy = (): ExperimentPredictorPolicy => ({ method: 'ensemble', refitPercentile: null });
export const predictorPolicyLabel = (policy: ExperimentPredictorPolicy) => ({ skip: 'Skip', refit: 'Refit', ensemble: 'Ensemble', both: 'Both' })[policy.method];
export const includesRefit = (policy: ExperimentPredictorPolicy) => policy.method === 'refit' || policy.method === 'both';

/** Only older recipes inherit an experiment-wide policy. New batches save their own. */
export const batchPredictorPolicy = (spec?: Pick<DevelopmentBatchSpec, 'predictorPolicy'>, fallback?: ExperimentPredictorPolicy | null): ExperimentPredictorPolicy => spec?.predictorPolicy ?? fallback ?? defaultPredictorPolicy();

function countsFor(groups: number, foldRuns: number, policy: ExperimentPredictorPolicy) {
  const ensembles = policy.method === 'ensemble' || policy.method === 'both' ? groups : 0;
  const refits = includesRefit(policy) ? groups : 0;
  return { groups, foldRuns, ensembles, refits, total: ensembles + refits };
}

export function plannedBatchPredictorCount(spec: DevelopmentBatchSpec, protocol?: ProtocolSpec, fallback?: ExperimentPredictorPolicy | null) {
  if (!protocol || protocol.split.mode !== 'kfold') return null;
  const seedGroups = new Set(spec.trainingSeeds).size * new Set(protocol.split.seeds).size;
  const configurations = plannedConfigurationCount(spec);
  const groups = (spec.candidateSelection === 'best_validation' ? 1 : configurations) * seedGroups;
  return countsFor(groups, configurations * seedGroups * protocol.split.folds, batchPredictorPolicy(spec, fallback));
}

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
export function experimentPredictorCount(record: ModelExperiment, fallback?: ExperimentPredictorPolicy | null, protocol?: ProtocolSpec) {
  const total = countsFor(0, 0, defaultPredictorPolicy());
  const add = (value: typeof total) => { for (const key of Object.keys(total) as (keyof typeof total)[]) total[key] += value[key]; };
  const batches = record.batches.filter((batch) => batch.state === 'active');
  for (const batch of batches) {
    const summary = batch.manifest.summary;
    const splitSeeds = new Set(batch.manifest.splitPlans.map((plan) => plan.seed ?? 0)).size;
    const policy = record.predictorPolicies?.[batch.id] ?? batchPredictorPolicy(batch.manifest.spec, fallback ?? record.predictorPolicy);
    const configurations = batch.manifest.spec?.candidateSelection === 'best_validation' ? 1 : summary.configurationCount;
    add(countsFor(configurations * summary.trainingSeedCount * splitSeeds, summary.runCount, policy));
  }
  // Submitted records retain their planning recipes as history: never count them twice.
  if (!record.configurationLocked && !record.submission && (record.batchPlans?.length ?? 0) > 0) {
    if (!protocol || protocol.split.mode !== 'kfold') return null;
    for (const plan of record.batchPlans ?? []) {
      add(plannedBatchPredictorCount(plan.spec, protocol, fallback ?? record.predictorPolicy)!);
    }
  }
  return total;
}
