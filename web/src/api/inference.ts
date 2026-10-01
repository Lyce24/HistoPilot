import { downloadRequestedArtifact, request } from './client';
import type { ResourcePolicy } from './development';
import type { Interpretation, VisualizeItem } from './interpretation';

/** Label-free description of one prediction unit; never derived from outcomes. */
export interface InferenceDistribution {
  mean: number | null;
  quantiles: { p10: number | null; p25: number | null; median: number | null; p75: number | null; p90: number | null };
  edges: number[];
  counts: Record<string, number[]>;
}
export interface InferencePredictedClass { label: string; count: number; fraction: number | null; meanConfidence: number | null }
export interface InferenceUnitSummary {
  count: number;
  predicted: InferencePredictedClass[];
  confidence: InferenceDistribution;
  margin: InferenceDistribution;
  binary?: {
    positiveClass: string; threshold: number;
    positiveProbability: { edges: number[]; counts: number[] };
    sweep: { threshold: number; positive: number }[];
    nearThreshold: { band: number; count: number }[];
  };
  ensemble?: {
    memberCount: number; records: number; unanimous: number; disagreements: number; meanSpread: number;
    agreement: { agree: number; count: number }[];
  };
}
/** Frozen by the worker in `summary.json` and the job result. */
export interface InferenceResultSummary {
  /** Label-blind evaluations write the same summary; their purpose says so. */
  purpose: 'inference' | 'evaluation'; unit: 'slide' | 'patient'; classOrder: string[]; positiveClass?: string | null;
  decisionThreshold: number; patientAggregation: string; memberCount: number;
  /** Per-record member probabilities are omitted when predictions.json would grow too large. */
  memberProbabilities?: 'recorded' | 'omitted_for_size' | 'single_model';
  slide: InferenceUnitSummary; patient: InferenceUnitSummary | { available: false; count: 0; reason: string };
  selected: InferenceUnitSummary | { available: false; count: 0; reason: string };
}
export interface InferenceBreakdownRow { value: string; count: number; counts: Record<string, number>; meanConfidence: number }
export interface InferenceComparison {
  evaluationId: string; name: string; predictorId: string; decisionThreshold: number; predictionsSha256: string;
  count: number; agreement: number | null; kappa: number | null; disagreements: number; matrix: number[][];
}
export interface InferenceSummary extends InferenceUnitSummary {
  evaluationId: string; name: string; purpose: 'inference' | 'evaluation'; predictorId: string; cohortId: string;
  unit: 'slide' | 'patient'; task: string; classOrder: string[]; positiveClass: string | null;
  decisionThreshold: number; patientAggregation: string; patients: number;
  source: { predictionsSha256: string };
  development: { comparable: false } | { comparable: true; patients: number; records: number; unknown?: number; shared: Record<string, number>; new: Record<string, number> };
  attributes: { key: string; label: string }[];
  breakdown?: { attribute: string; label: string; rows: InferenceBreakdownRow[]; otherValues: number; other?: { count: number; counts: Record<string, number> } };
  comparison?: InferenceComparison;
}
export interface InferenceSummaryQuery { unit: 'selected' | 'slide' | 'patient'; attribute?: string | null; comparisonId?: string | null }
export interface InferenceAttentionResult { evaluationId: string; items: VisualizeItem[]; interpretations: Interpretation[] }

const base = (project: string, evaluation: string) => `/projects/${encodeURIComponent(project)}/evaluation-runs/${encodeURIComponent(evaluation)}`;
const post = (value: unknown, signal?: AbortSignal) => ({ method: 'POST', body: JSON.stringify(value), signal });

export const inferenceRuns = {
  summary: (project: string, evaluation: string, query: InferenceSummaryQuery, signal?: AbortSignal) =>
    request<InferenceSummary>(`${base(project, evaluation)}/inference/summary`, post({ unit: query.unit, attribute: query.attribute || null, comparisonId: query.comparisonId || null }, signal)),
  export: (project: string, evaluation: string, unit: InferenceSummaryQuery['unit'], attributes: string[] | null = null) =>
    downloadRequestedArtifact(`${base(project, evaluation)}/inference/export`, 'inference-predictions.csv', post({ unit, attributes })),
  attention: (project: string, evaluation: string, slideIds: string[], operationId: string, resources?: ResourcePolicy) =>
    request<InferenceAttentionResult>(`${base(project, evaluation)}/attention`, post({ slideIds, operationId, ...(resources ? { resources } : {}) })),
};
