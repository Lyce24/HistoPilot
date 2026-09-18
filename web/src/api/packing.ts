import { request } from './client';
import type { Finding } from './scientific';

export interface FeaturePackSpec {
  featureSetId: string;
  action: 'pack' | 'validate' | 'attach';
  outputPath?: string | null;
  existingPath?: string | null;
  dtype: 'preserve' | 'float16';
}

export interface FeatureValidationReport {
  valid: boolean;
  tensorValidationComplete: boolean;
  provenanceComplete: boolean;
  featureSetId: string;
  sourceContentHash: string;
  slideCount: number;
  totalPatches: number;
  dimensions: number;
  sourceDtype: string | string[];
  current?: boolean;
  jobId?: string;
  validatedAt?: string;
  findings?: Finding[];
}

export interface FeaturePackArtifact {
  id: string;
  materializationId: string;
  outputPath: string;
  featureSetId: string;
  sourceContentHash: string;
  slideCount: number;
  totalPatches: number;
  dimensions: number;
  outputDtype: string;
  sourceDtype?: string;
  dtypePolicy: 'preserve' | 'float16';
  preservesSourcePrecision?: boolean;
  origin?: 'existing' | 'created';
  verification?: string;
  current?: boolean;
  findings?: Finding[];
  validation: FeatureValidationReport;
  [key: string]: unknown;
}

export interface FeaturePackPreview {
  spec: FeaturePackSpec;
  previewHash: string;
  canRun: boolean;
  findings: Finding[];
  slideCount: number;
  patchCount: number;
  dimensions: number | null;
  sourceDtype: string | string[] | null;
  outputDtype: string | null;
  estimatedBytes: number | null;
  availableBytes: number | null;
  outputPath: string | null;
  tmuxAvailable: boolean;
  formatAvailable?: boolean;
  existingPath?: string | null;
  matchesFeatures?: boolean;
  verification?: 'headers';
  packInspection?: FeaturePackInspection | null;
}

export interface FeaturePackInspection {
  format: string;
  formatVariant: string;
  slideCount: number;
  totalPatches: number;
  dimensions: number;
  sourceDtype: string | null;
  /** "reduced" packs hold the source at a lower float precision and verify against the cast source. */
  precision: 'exact' | 'reduced';
  outputDtype: string;
  featureBytes: number;
  coordinateBytes: number;
  totalBytes: number;
  expectedFeatureBytes: number | null;
  expectedCoordinateBytes: number;
  sourceContainerBytes: number;
  sourcePatchCount: number;
  missingSlideCount: number;
  extraSlideCount: number;
  mismatchedSlideCount: number;
  missingSlides: string[];
  extraSlides: string[];
  mismatchedSlides: { slideId: string; sourcePatches: number; packPatches: number }[];
}

export interface FeaturePackSelection {
  featureSetId: string;
  artifactId: string | null;
  artifact: FeaturePackArtifact | null;
  current: boolean;
  findings: Finding[];
}

export type FeaturePackState = 'queued' | 'starting' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

export interface FeaturePackJob {
  progressWarning?: string | null;
  id: string;
  state: FeaturePackState;
  spec: FeaturePackSpec;
  featureSetId: string;
  outputPath: string | null;
  sessionName: string;
  logPath: string;
  createdAt: string;
  updatedAt: string;
  progress?: {
    stage: string;
    completed: number | null;
    total: number | null;
    unit: string;
    currentSlide: string | null;
    percent: number | null;
  } | null;
  result?: { validation: FeatureValidationReport; artifact: FeaturePackArtifact | null } | null;
  error?: string | null;
  logs?: string;
}

export interface FeaturePackJobs {
  jobs: FeaturePackJob[];
  artifacts: FeaturePackArtifact[];
  tmuxAvailable: boolean;
  formatAvailable: boolean;
  defaultOutputRoot: string;
  selections?: Record<string, string | null>;
}

const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/feature-packs`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });

export const packing = {
  jobs: (project: string) => request<FeaturePackJobs>(prefix(project)),
  preview: (project: string, spec: FeaturePackSpec) =>
    request<FeaturePackPreview>(`${prefix(project)}/preview`, post(spec)),
  start: (project: string, spec: FeaturePackSpec, previewHash: string, operationId: string) =>
    request<FeaturePackJob>(prefix(project), post({ ...spec, previewHash, operationId })),
  job: (project: string, id: string) =>
    request<FeaturePackJob>(`${prefix(project)}/${encodeURIComponent(id)}`),
  cancel: (project: string, id: string) =>
    request<FeaturePackJob>(`${prefix(project)}/${encodeURIComponent(id)}/cancel`, post({})),
  validation: (project: string, featureSetId: string) =>
    request<FeatureValidationReport | null>(`/projects/${encodeURIComponent(project)}/features/${encodeURIComponent(featureSetId)}/validation`),
  selection: (project: string, featureSetId: string) =>
    request<FeaturePackSelection>(`/projects/${encodeURIComponent(project)}/features/${encodeURIComponent(featureSetId)}/pack-selection`),
  select: (project: string, featureSetId: string, artifactId: string | null) =>
    request<FeaturePackSelection>(`/projects/${encodeURIComponent(project)}/features/${encodeURIComponent(featureSetId)}/pack-selection`, { method: 'PUT', body: JSON.stringify({ artifactId }) }),
  artifact: (project: string, artifactId: string) =>
    request<FeaturePackArtifact>(`${prefix(project)}/artifacts/${encodeURIComponent(artifactId)}`),
};

export const featurePackActive = (job: Pick<FeaturePackJob, 'state'> | undefined) =>
  job?.state === 'queued' || job?.state === 'starting' || job?.state === 'running' || job?.state === 'cancelling';

/** The review identity includes every option that changes the worker's output. */
export function featurePackSpecKey(spec: FeaturePackSpec): string {
  return JSON.stringify([spec.featureSetId, spec.action, spec.action === 'pack' ? spec.dtype : 'preserve', spec.action === 'pack' ? spec.outputPath?.trim() || null : null, spec.action === 'attach' ? spec.existingPath?.trim() || null : null]);
}
