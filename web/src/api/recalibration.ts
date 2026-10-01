import { request } from './client';

/** How well risks match outcomes; the risk is the positive class's probability, or the
 * predicted class's confidence for a multiclass target. */
export interface CalibrationMetrics {
  count: number; brierScore: number; logLoss: number; ece: number;
  meanPredictedRisk: number; observedFraction: number; observedExpectedRatio: number | null;
  /** Outcome regressed on logit(risk): 1 and 0 when calibrated; null when not estimable. */
  calibrationSlope: number | null; calibrationIntercept: number | null;
  bins: { lower: number; upper: number; count: number; meanPredicted: number; observedFraction: number }[];
}
export interface Recalibration {
  evaluationId: string; unit: 'slide' | 'patient'; classOrder: string[]; positiveClass: string | null;
  /** Platt scaling for binary targets, temperature scaling for multiclass ones. */
  method: 'platt' | 'temperature';
  parameters: { slope: number; intercept: number } | { temperature: number };
  /** The predictor's out-of-fold development predictions the map is fitted on. */
  development: { units: number; seedGroups: number; original: CalibrationMetrics; recalibrated: CalibrationMetrics };
  cohort: { units: number; original: CalibrationMetrics; recalibrated: CalibrationMetrics };
  developmentExcluded: number;
  reference: { id: string; name: string } | null;
  developmentSources: { batchId: string; candidateId: string; trainingSeed: number; splitSeed: number; analysisInputHash: string }[];
}

/** A run's calibration as predicted and after a map fitted on its predictor's development predictions. */
export const recalibration = {
  get: (project: string, evaluation: string, unit: 'selected' | 'slide' | 'patient', referenceId: string | null, signal?: AbortSignal) =>
    request<Recalibration>(`/projects/${encodeURIComponent(project)}/evaluation-runs/${encodeURIComponent(evaluation)}/recalibration`,
      { method: 'POST', body: JSON.stringify({ unit, referenceId }), signal }),
};
