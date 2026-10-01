import { downloadArtifact, request } from './client';
import type { LifecycleState } from './lifecycle';
import type { EvaluationMetrics } from './predictors';
import type { Finding } from './scientific';

/** One dataset column mapped to a class set, for every slide of one cohort. */
export interface ReferenceSelection {
  cohortId: string; name: string; datasetIds: string[]; field: string;
  classes: string[]; labels: Record<string, string>;
}
export interface ReferenceSummary {
  slides: number; labeledSlides: number; missingSlides: number; unmappedSlides: number; unmatchedSlides: number;
  classCounts: Record<string, number>;
  /** Patients whose slides map to different classes; patient-level scores leave them unlabeled. */
  conflictingPatients: number;
  /** The column's values over the cohort's matched slides, with the class each maps to. */
  values: { value: string; count: number; label: string | null }[];
  valuesTruncated: boolean;
}
export interface ReferenceManifest {
  kind: 'reference-standard'; name: string; cohortId: string; cohort: { id: string; contentHash: string };
  datasetId: string; datasets: { id: string; contentHash: string }[]; field: string; matchedBy: 'slideId';
  classes: string[]; labels: Record<string, string>;
  /** Present on a single record; lists omit it. */
  memberships?: { slideId: string; label: string | null }[];
  summary: ReferenceSummary; findings: Finding[];
}
export interface ReferenceStandard {
  id: string; createdAt: string; contentHash: string; lifecycleState?: LifecycleState; manifest: ReferenceManifest;
}
export interface ReferencePreview { canSave: boolean; previewHash: string | null; manifest: ReferenceManifest | null; findings: Finding[] }

export interface AgreementPair {
  left: string; right: string; count: number; agreement: number | null; kappa: number | null;
  /** Three or more classes: near misses on the class order count as partial agreement. */
  weightedKappa?: number | null;
  disagreements: number; matrix: number[][];
}
export interface RunAgreement {
  evaluationId: string; unit: 'slide' | 'patient'; classOrder: string[];
  sources: { id: string; name: string; kind: 'run' | 'cohort' | 'reference'; labeled?: number; reason?: string }[];
  pairs: AgreementPair[];
  /** Units from development patients, left out as they are from every metric. */
  developmentExcluded: number;
  source: { predictionsSha256: string };
}

const base = (project: string) => `/projects/${encodeURIComponent(project)}`;
const run = (project: string, evaluation: string) => `${base(project)}/evaluation-runs/${encodeURIComponent(evaluation)}`;
const post = (value: unknown, signal?: AbortSignal) => ({ method: 'POST', body: JSON.stringify(value), signal });

/** Labels attached to a frozen cohort after it was frozen, and runs scored against them. */
export const references = {
  list: (project: string, cohortId?: string) => request<{ items: ReferenceStandard[] }>(`${base(project)}/reference-standards?${new URLSearchParams({ ...(cohortId ? { cohort_id: cohortId } : {}), include_inactive: 'true' })}`),
  get: (project: string, id: string) => request<ReferenceStandard>(`${base(project)}/reference-standards/${encodeURIComponent(id)}`),
  preview: (project: string, selection: ReferenceSelection) => request<ReferencePreview>(`${base(project)}/reference-standards/preview`, post(selection)),
  save: (project: string, selection: ReferenceSelection, previewHash: string, operationId: string) =>
    request<ReferenceStandard>(`${base(project)}/reference-standards`, post({ ...selection, previewHash, operationId })),
  /** A run's scores against a reference, or against its cohort's labels when `referenceId` is null. */
  scores: (project: string, evaluation: string, referenceId: string | null, signal?: AbortSignal) =>
    request<EvaluationMetrics>(`${run(project, evaluation)}/scores`, post({ referenceId }, signal)),
  agreement: (project: string, evaluation: string, unit: 'selected' | 'slide' | 'patient', signal?: AbortSignal) =>
    request<RunAgreement>(`${run(project, evaluation)}/agreement`, post({ unit }, signal)),
  /** A run's metrics or scored table against a reference, shaped like the run's own downloads. */
  download: (project: string, evaluation: string, referenceId: string, filename: ReferenceArtifact, savedAs: string = filename) =>
    downloadArtifact(`${run(project, evaluation)}/reference-standards/${encodeURIComponent(referenceId)}/${filename}`, savedAs),
};
export type ReferenceArtifact = 'metrics.json' | 'slide-predictions.csv' | 'patient-predictions.csv';

/** A reference applies to a run on the same cohort whose classes it maps to, unless it is in Trash. */
export function referenceFits(standard: ReferenceStandard, cohortId: string, classes: readonly string[]) {
  return standard.lifecycleState !== 'trashed' && standard.manifest.cohortId === cohortId
    && [...standard.manifest.classes].sort().join('\u0000') === [...classes].sort().join('\u0000');
}
