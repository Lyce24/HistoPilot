import { downloadArtifact, request } from './client';
import type { ComputeExecution } from './predictors';

/** One slide in the dataset shift analysis of a run (histopilot/shift.py). */
export interface ShiftSlide {
  slideId: string;
  set: 'development' | 'cohort';
  /** Mean cosine distance to the nearest development slides; development slides leave themselves out. */
  distance: number;
  /** Share of development slides at or below this distance. */
  percentile: number;
  /** Beyond the development slides' own 99th percentile. */
  beyond: boolean;
  /** Out-of-fold probability that the slide is a cohort slide; cohort slides only. */
  domainScore?: number;
  x: number;
  y: number;
}

export interface ShiftSummary {
  developmentSlides: number;
  cohortSlides: number;
  dimensions: number;
  domainAuroc: { value: number | null; lower: number; upper: number };
  beyond: { count: number; fraction: number; lower: number | null; upper: number | null; threshold: number };
  distance: Record<'development' | 'cohort', Record<'p05' | 'p25' | 'median' | 'p75' | 'p95', number>>;
  verdict: 'little' | 'moderate' | 'strong' | null;
  verdictText: string | null;
}

export interface ShiftReport {
  version: number;
  method: Record<string, unknown>;
  summary: ShiftSummary;
  slides: ShiftSlide[];
  inputHash: string;
}

export interface ShiftFinding { severity: 'error' | 'warning'; code: string; message: string }

export interface ShiftStatus {
  evaluationId: string;
  analysisId: string;
  exists: boolean;
  execution: (ComputeExecution & { progress?: { completed?: number; total?: number } | null }) | null;
  findings: ShiftFinding[];
  canRun: boolean;
  report: ShiftReport | null;
}

export interface ShiftPreview {
  evaluationId: string; analysisId: string; exists: boolean; status: string;
  development: { batchId: string; slides: number };
  cohort: { cohortId: string; slides: number; sharedSlidesExcluded: number };
  featureDim: number; findings: ShiftFinding[]; canRun: boolean; previewHash: string;
}

export interface ShiftGroupMetrics {
  group: string;
  available: boolean; count: number; reason?: string;
  auroc?: number | null; balancedAccuracy?: number | null; accuracy?: number | null;
  percentileRange?: [number, number];
}

export interface ShiftPerformance {
  evaluationId: string; analysisId: string; unit: string;
  overall: ShiftGroupMetrics;
  thirds: ShiftGroupMetrics[];
  range: ShiftGroupMetrics[];
}

const base = (project: string, evaluation: string) =>
  `/projects/${encodeURIComponent(project)}/evaluation-runs/${encodeURIComponent(evaluation)}/shift`;

/** A run's dataset shift: its cohort against its predictor's development slides. */
export const shift = {
  status: (project: string, evaluation: string, signal?: AbortSignal) =>
    request<ShiftStatus>(base(project, evaluation), { signal }),
  preview: (project: string, evaluation: string) =>
    request<ShiftPreview>(`${base(project, evaluation)}/preview`, { method: 'POST', body: '{}' }),
  start: (project: string, evaluation: string, previewHash: string, operationId: string) =>
    request<ComputeExecution>(base(project, evaluation), { method: 'POST', body: JSON.stringify({ previewHash, operationId }) }),
  performance: (project: string, evaluation: string, unit: 'selected' | 'slide' | 'patient', referenceId: string | null, signal?: AbortSignal) =>
    request<ShiftPerformance>(`${base(project, evaluation)}/performance`, { method: 'POST', body: JSON.stringify({ unit, referenceId }), signal }),
};

/** A frozen predictor's model card, as Markdown, saved under the predictor's name. */
export function downloadModelCard(project: string, predictor: string, name: string) {
  return downloadArtifact(
    `/projects/${encodeURIComponent(project)}/predictors/${encodeURIComponent(predictor)}/card?format=markdown`,
    `${name} model card.md`,
  );
}
