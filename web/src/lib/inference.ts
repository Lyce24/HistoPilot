import type { EvaluationCohort, EvaluationSpec } from '../api/evaluation';
import type { InferenceUnitSummary } from '../api/inference';
import type { ModelEvaluation } from '../api/predictors';

/** Unlabeled cohorts that receive predictions only; `review` is their earlier name. */
export const isInferencePurpose = (purpose?: string | null) => purpose === 'inference' || purpose === 'review';
export const isInferenceCohort = (cohort?: Pick<EvaluationCohort, 'manifest'> | null) => isInferencePurpose(cohort?.manifest.spec.purpose);
/** Runs created from inference cohorts, including earlier slide-review runs. */
export const isInferenceRun = (record?: Pick<ModelEvaluation, 'manifest'> | null) => isInferencePurpose(record?.manifest.purpose);

export type CohortKind = 'evaluation' | 'inference' | 'unlabeled-evaluation';
export function cohortKind(spec: Pick<EvaluationSpec, 'purpose' | 'target'>): CohortKind {
  if (isInferencePurpose(spec.purpose)) return 'inference';
  return spec.target === null ? 'unlabeled-evaluation' : 'evaluation';
}
export const cohortKindLabel: Record<CohortKind, string> = {
  evaluation: 'Evaluation · labeled',
  inference: 'Inference · unlabeled',
  'unlabeled-evaluation': 'Unlabeled evaluation (earlier)',
};

export const percent = (value: number | null | undefined, digits = 1) =>
  typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—';
export const decimal = (value: number | null | undefined, digits = 3) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';

/** Stable class colors from the H&E-inspired palette, reused across every inference chart. */
const CLASS_COLORS = ['#006b5f', '#6e5aa7', '#c86e93', '#356580', '#8a5810', '#00a58f', '#b4433c', '#656d6a'];
export const classColor = (index: number) => CLASS_COLORS[index % CLASS_COLORS.length];

/** Compact "ND 212 · LG 60" text for tables, largest classes first. */
export function predictedMix(summary?: Pick<InferenceUnitSummary, 'predicted'> | null, limit = 4) {
  if (!summary) return '—';
  const rows = summary.predicted.filter((item) => item.count > 0).sort((a, b) => b.count - a.count);
  const shown = rows.slice(0, limit).map((item) => `${item.label} ${item.count.toLocaleString()}`);
  return rows.length > limit ? `${shown.join(' · ')} · +${rows.length - limit}` : shown.join(' · ') || 'No predictions';
}

/** Records whose confidence lies below a bin-edge cutoff, read from the frozen histogram. */
export function countBelow(distribution: InferenceUnitSummary['confidence'], cutoff: number) {
  const bins = distribution.edges.length - 1;
  const last = Math.min(bins, Math.max(0, Math.round(cutoff * bins)));
  return Object.values(distribution.counts).reduce((sum, counts) => sum + counts.slice(0, last).reduce((total, value) => total + value, 0), 0);
}

export const unanimousShare = (summary?: InferenceUnitSummary | null) =>
  summary?.ensemble && summary.ensemble.records ? summary.ensemble.unanimous / summary.ensemble.records : null;

/**
 * Batches belong to the mode of the runs they created. Archived cohorts are absent
 * from the active cohort list, so the cohort is only a fallback for empty batches.
 */
export function isInferenceBatch(batch: { cohortId: string; items: { evaluationId?: string | null }[] }, inferenceRunIds: ReadonlySet<string>, inferenceCohortIds: ReadonlySet<string>) {
  const ids = batch.items.map((item) => item.evaluationId).filter((id): id is string => Boolean(id));
  return ids.length ? ids.some((id) => inferenceRunIds.has(id)) : inferenceCohortIds.has(batch.cohortId);
}

/** Runs whose predictions can be paired case by case with this one. */
export function comparableRuns(record: Pick<ModelEvaluation, 'id' | 'manifest'>, runs: ModelEvaluation[]) {
  const context = (item: Pick<ModelEvaluation, 'manifest'>) => JSON.stringify([item.manifest.cohortId, item.manifest.target ?? null, (item.manifest.inference as { patientAggregation?: string } | undefined)?.patientAggregation ?? null]);
  return runs.filter((item) => item.id !== record.id && context(item) === context(record) && item.execution?.status === 'completed' && item.lifecycleState !== 'trashed');
}

export function isUnitSummary(value: unknown): value is InferenceUnitSummary {
  return Boolean(value && typeof value === 'object' && Array.isArray((value as InferenceUnitSummary).predicted));
}
