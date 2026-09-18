export interface PatientAnalysisSettings {
  version: 1; confidenceLevel: 0.95; bootstrapResamples: number; bootstrapSeed: number; oneSlideSeed: number;
}
export const defaultPatientAnalysis = (): PatientAnalysisSettings => ({ version: 1, confidenceLevel: 0.95, bootstrapResamples: 2000, bootstrapSeed: 42, oneSlideSeed: 42 });
export interface ConfidenceInterval { lower: number; upper: number }
export type RankingMetric = 'auroc' | 'auprc';
export interface BootstrapResult {
  available: boolean; reason?: string; method?: string; confidenceLevel?: number; seed?: number;
  resamples?: number; validResamples?: number; excludedResamples?: number; patientCount?: number;
  estimates?: Partial<Record<RankingMetric, number>>;
  intervals?: Partial<Record<RankingMetric, ConfidenceInterval>>; note?: string; degenerate?: boolean;
}
export interface PatientAnalysis {
  policy: PatientAnalysisSettings; uncertainty: BootstrapResult;
  oneSlidePerPatient: { method?: string; seed?: number; slideIds?: string[]; patientCount?: number;
    metrics?: Partial<Record<RankingMetric, number>>; uncertainty: BootstrapResult };
}
export interface PatientComparison {
  leftEvaluationId: string; rightEvaluationId: string; leftName: string; rightName: string;
  cohortId: string; analysis: PatientAnalysisSettings; difference: 'left_minus_right';
  predictionsSha256: { left: string; right: string };
  statistics: Omit<BootstrapResult, 'estimates'> & { estimates?: Partial<Record<RankingMetric, { left: number; right: number; difference: number }>> };
}
export function formatStatistic(value?: number | null, interval?: ConfidenceInterval) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return interval ? `${value.toFixed(3)} (${interval.lower.toFixed(3)}–${interval.upper.toFixed(3)})` : value.toFixed(3);
}
