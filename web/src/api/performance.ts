import { request } from './client';
import type { TrainingMetrics } from './development';

/** One group's metrics: labeled units only, as in the run's own metrics. */
export type GroupMetrics = TrainingMetrics & { predictionCount?: number; unlabeledCount?: number };
export interface PerformanceBreakdown {
  evaluationId: string; unit: 'slide' | 'patient'; attribute: string; label: string;
  classOrder: string[]; positiveClass: string | null; decisionThreshold: number;
  overall: GroupMetrics;
  rows: (GroupMetrics & { value: string })[];
  otherValues: number; other?: GroupMetrics;
  /** Label-blind runs: units from development patients, left out as they are from every metric. */
  developmentExcluded: number;
  source: { predictionsSha256: string };
}

const base = (project: string, evaluation: string) => `/projects/${encodeURIComponent(project)}/evaluation-runs/${encodeURIComponent(evaluation)}`;

export const runPerformance = {
  /** `referenceId` scores against a reference standard of the cohort instead of its labels. */
  breakdown: (project: string, evaluation: string, unit: 'selected' | 'slide' | 'patient', attribute: string, referenceId: string | null, signal?: AbortSignal) =>
    request<PerformanceBreakdown>(`${base(project, evaluation)}/performance/breakdown`, { method: 'POST', body: JSON.stringify({ unit, attribute, referenceId }), signal }),
};
