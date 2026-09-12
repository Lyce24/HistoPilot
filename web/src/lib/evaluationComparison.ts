import type { FrozenPredictor, ModelEvaluation } from '../api/predictors';

export type EvaluationMetric = 'auroc' | 'accuracy';
export function evaluationMetric(item: ModelEvaluation, key: EvaluationMetric): number | null {
  const metrics = item.execution?.result?.metrics?.selected;
  const value = metrics?.[key];
  return item.execution?.status === 'completed' && metrics?.available && typeof value === 'number' && Number.isFinite(value) ? value : null;
}
export const evaluationUnit = (item: ModelEvaluation) => item.execution?.result?.metrics?.unit ?? 'unavailable';
export interface EvaluationPair {
  key: string; source: FrozenPredictor['manifest']; ensemble?: ModelEvaluation; refit?: ModelEvaluation;
}
export interface EvaluationComparison {
  key: string; cohortId: string; unit: 'slide' | 'patient'; classOrder: string[]; positiveClass: string | null;
  threshold: number | null; patientAggregation: string; labeledCount: number | null;
  sources: EvaluationPair[]; pairs: EvaluationPair[]; ensembleOnly: number; refitOnly: number;
}

/** Pair the same batch/configuration/seeds only within identical test scoring contexts. */
export function compareEvaluationMethods(records: ModelEvaluation[], predictors: FrozenPredictor[]): EvaluationComparison[] {
  const predictorById = new Map(predictors.map((item) => [item.id, item.manifest]));
  const contexts = new Map<string, { context: Omit<EvaluationComparison, 'sources' | 'pairs' | 'ensembleOnly' | 'refitOnly'>; sources: Map<string, EvaluationPair> }>();
  // Newest completed result wins within each source/method, never the highest score.
  const completed = records.filter((item) => item.execution?.status === 'completed').sort((a, b) => b.createdAt.localeCompare(a.createdAt) || b.id.localeCompare(a.id));
  for (const record of completed) {
    const source = predictorById.get(record.manifest.predictorId), metrics = record.execution?.result?.metrics;
    if (!source || !metrics?.selected?.available || !['slide', 'patient'].includes(metrics.unit)) continue;
    const contextKey = JSON.stringify([record.manifest.cohortId, metrics.unit, metrics.classOrder, metrics.positiveClass, metrics.decisionThreshold ?? null, metrics.patientAggregation, metrics.selected.count ?? null]);
    const entry = contexts.get(contextKey) ?? { context: { key: contextKey, cohortId: record.manifest.cohortId, unit: metrics.unit, classOrder: metrics.classOrder, positiveClass: metrics.positiveClass,
      threshold: metrics.decisionThreshold ?? null, patientAggregation: metrics.patientAggregation, labeledCount: metrics.selected.count ?? null }, sources: new Map<string, EvaluationPair>() };
    const key = JSON.stringify([source.experimentId, source.batchId, source.candidateId, source.trainingSeed, source.splitSeed]);
    const pair = entry.sources.get(key) ?? { key, source };
    const method = source.method ?? 'ensemble';
    if (!pair[method]) pair[method] = record;
    entry.sources.set(key, pair); contexts.set(contextKey, entry);
  }
  return [...contexts.values()].map(({ context, sources }) => {
    const rows = [...sources.values()];
    return { ...context, sources: rows, pairs: rows.filter((item) => item.ensemble && item.refit), ensembleOnly: rows.filter((item) => item.ensemble && !item.refit).length, refitOnly: rows.filter((item) => !item.ensemble && item.refit).length };
  }).sort((a, b) => a.cohortId.localeCompare(b.cohortId) || a.unit.localeCompare(b.unit) || a.key.localeCompare(b.key));
}

/** Both method means use the exact same source pairs for each metric. */
export function pairedEvaluationMean(pairs: EvaluationPair[], metric: EvaluationMetric) {
  const values = pairs.flatMap((pair) => {
    const ensemble = pair.ensemble ? evaluationMetric(pair.ensemble, metric) : null;
    const refit = pair.refit ? evaluationMetric(pair.refit, metric) : null;
    return ensemble === null || refit === null ? [] : [{ ensemble, refit }];
  });
  return { count: values.length, ensemble: values.length ? values.reduce((sum, item) => sum + item.ensemble, 0) / values.length : null,
    refit: values.length ? values.reduce((sum, item) => sum + item.refit, 0) / values.length : null };
}
