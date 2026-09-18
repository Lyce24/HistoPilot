import { request } from './client';
import type { Finding } from './scientific';
import type { SlideListSource } from './slideLists';

export interface TridentOption {
  name: string;
  flag: string;
  label: string;
  group: 'execution' | 'slides' | 'segmentation' | 'patching' | 'features';
  type: 'boolean' | 'integer' | 'number' | 'string' | 'string[]' | 'integer[]';
  default: unknown;
  advanced: boolean;
  nullable?: boolean;
  choices?: string[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  description: string;
}
export interface TridentRuntime {
  available?: boolean;
  ready?: boolean;
  python?: string;
  script?: string;
  tmux?: boolean | string;
  message?: string;
  findings?: Finding[];
  [key: string]: unknown;
}
export interface TridentCatalog {
  source: string;
  schemaVersion: number;
  options: TridentOption[];
  defaults: Record<string, unknown>;
  patchEncoders: string[];
  slideEncoders: string[];
  managedOptions?: string[];
  defaultOutputPath?: string;
  tmuxAvailable?: boolean;
  runtime?: TridentRuntime;
}
export interface ExtractionSpec {
  /** Null runs against a slide folder alone; a dataset narrows whatever the source selected. */
  datasetId: string | null;
  slideRoot?: string | null;
  slideList?: SlideListSource | null;
  recursive?: boolean;
  outputPath: string;
  options: Record<string, unknown>;
}
export interface TridentOutputLayout {
  jobDir: string;
  coordsDir: string;
  patchesDir: string;
  featuresDir: string;
  contoursDir: string;
  geojsonDir: string;
  thumbnailsDir: string;
  coordinatePattern: string;
  featurePattern: string;
  featureKind: 'patch' | 'slide';
}
export interface SlideListSummary {
  source: 'list' | 'folder' | 'dataset';
  listPath: string | null;
  filename?: string;
  sha256: string | null;
  root: string;
  initialCount: number;
  selectedCount: number;
  declaresMpp: boolean;
  datasetFiltered: boolean;
  outsideCount: number;
  outsideExamples: string[];
  unlistedCount: number;
  unlistedExamples: string[];
}
export interface ExtractionPreview {
  spec: ExtractionSpec;
  previewHash: string;
  canRun: boolean;
  findings: Finding[];
  slideCount: number;
  slideList?: SlideListSummary | null;
  outputLayout: TridentOutputLayout;
  runtime: TridentRuntime;
  command: string[] | string;
}
export type ExtractionState =
  | 'queued'
  | 'starting'
  | 'running'
  | 'cancelling'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted';
export interface ExtractionProgress {
  stage: string;
  stages: {
    id: 'preparing' | 'segmentation' | 'coordinates' | 'patch_features' | 'slide_features' | 'validation';
    label: string;
    status: 'pending' | 'active' | 'complete' | 'stopped';
  }[];
  label: string;
  detail: string;
  completed: number | null;
  total: number | null;
  unit: 'slides' | 'patches' | 'files';
  percent: number | null;
  currentSlide: string | null;
  /** Whole-run wall-clock duration; stage estimates are separate. */
  elapsedSeconds: number | null;
  stageElapsedSeconds?: number | null;
  /** Remaining time applies only to the stage or batch identified by scope. */
  etaSeconds: number | null;
  ratePerSecond: number | null;
  scope: 'stage' | 'batch' | null;
  warnings: string[];
}
export interface ExtractionJob {
  id: string;
  state: ExtractionState;
  createdAt: string;
  updatedAt: string;
  spec: ExtractionSpec;
  outputPath: string;
  logPath: string;
  sessionName: string;
  error?: string | null;
  logs?: string;
  outputLayout?: TridentOutputLayout;
  progress?: ExtractionProgress;
  result?: {
    outputLayout?: TridentOutputLayout;
    featurePath?: string;
    featureDirectory?: string;
    completedSlides?: number;
    missingSlides?: number;
    findings?: Finding[];
    [key: string]: unknown;
  } | null;
}

const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/extractions`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });

export const trident = {
  catalog: (project: string) => request<TridentCatalog>(`${prefix(project)}/catalog`),
  preview: (project: string, spec: ExtractionSpec) =>
    request<ExtractionPreview>(`${prefix(project)}/preview`, post(spec)),
  start: (project: string, spec: ExtractionSpec, previewHash: string, operationId: string) =>
    request<ExtractionJob>(prefix(project), post({ ...spec, previewHash, operationId })),
  jobs: (project: string) => request<{ jobs: ExtractionJob[] }>(prefix(project)),
  job: (project: string, id: string) =>
    request<ExtractionJob>(`${prefix(project)}/${encodeURIComponent(id)}`),
  cancel: (project: string, id: string) =>
    request<ExtractionJob>(`${prefix(project)}/${encodeURIComponent(id)}/cancel`, post({})),
};

/** Keep in-progress input editable; serialize each control using the server's option schema. */
export function normalizeTridentOptions(
  definitions: TridentOption[],
  values: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    definitions.map((option) => {
      const source = values[option.name] ?? option.default;
      const value = typeof source === 'string' ? source.trim() : source;
      if (value === '' || value === null || value === undefined) return [option.name, null];
      if (option.type === 'integer' || option.type === 'number')
        return [option.name, Number(value)];
      if (option.type.endsWith('[]')) {
        const parts = Array.isArray(value)
          ? value
          : String(value).split(/[\s,]+/).filter(Boolean);
        return [option.name, option.type === 'integer[]' ? parts.map(Number) : parts];
      }
      return [option.name, option.type === 'string' ? String(value).trim() : value];
    }),
  );
}

export const extractionActive = (job: ExtractionJob | undefined) =>
  job?.state === 'queued' || job?.state === 'starting' || job?.state === 'running' || job?.state === 'cancelling';
