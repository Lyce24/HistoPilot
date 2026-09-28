import type { VersionLabel } from './scientific';
import type { DemoPipeline } from './demo';

export type Page =
  | 'overview'
  | 'dataset'
  | 'cohort'
  | 'features'
  | 'experimental-setup'
  | 'legacy-protocol'
  | 'experiments'
  | 'post-development'
  | 'source-cv'
  | 'selection'
  | 'predictor'
  | 'test-data'
  | 'evaluation'
  | 'inference'
  | 'clinical-utility'
  | 'interpretation'
  | 'cleanup'
  | 'operations'
  | 'task-center'
  | 'system';
export interface Source {
  id: string;
  path: string;
  createdAt: string;
  readOnly?: boolean;
  status?: string;
  name?: string;
  role?: 'data' | 'slides' | 'features';
  importStatus?: string;
}
export interface InitialConfig {
  task?: 'binary_classification' | 'multiclass_classification';
  targetColumn?: string;
  positiveLabel?: string;
  seed?: number;
  folds?: number;
  encoderId?: string;
  milId?: string;
}
export interface ProjectSummary {
  id: string;
  name: string;
  description: string;
  storagePath: string;
  mode: 'local' | 'synthetic-demo';
  lifecycleState?: 'active' | 'archived' | 'trashed';
  createdAt: string;
  updatedAt: string;
  config: InitialConfig;
  sources: Source[];
  available: boolean;
  unavailableReason?: string;
  dataPath?: string;
  slidePath?: string;
  featurePath?: string;
}
export interface ProjectInput {
  name: string;
  storagePath: string;
  description?: string;
  dataPath?: string;
  slidePath?: string;
  featurePath?: string;
  config?: InitialConfig;
}
/** `GET /projects/{id}/workspace`: a local project, or the read-only BLCA demo with `demoPipeline`. */
export interface Workspace {
  project: ProjectSummary;
  mode: 'local' | 'synthetic-demo';
  executionEnabled: boolean;
  demoPipeline?: DemoPipeline;
  /** Local projects only: storage diagnostics (not shown in the interface) and record counts. */
  scientificStorage?: Record<string, unknown>;
  scientificSummary?: { datasetCount: number; protocolCount: number; featureCount: number };
  dataset: {
    id: string;
    name?: string;
    versionLabel?: VersionLabel | null;
    patientCount: number;
    specimenCount: number;
    slideCount: number;
    fallbackSlideCount?: number;
    groupCount?: number;
    unlinkedSlideCount?: number;
  };
  sources: Source[];
}
export interface FilesystemRoot {
  path: string;
  name: string;
}
export interface DirectoryListing {
  path: string;
  parent: string | null;
  entries: { name: string; path: string; kind: string }[];
  truncated: boolean;
}
export interface SystemStatus {
  mode: string;
  workspace: string;
  storage: { engine: string; journalMode: string; schemaVersion: number };
  control: { cudaModelsLoaded: boolean; process: string };
  /** `tmuxAvailable`: tmux is available to host the Task Center runner. */
  workers: { executionEnabled: boolean; status: string; tmuxAvailable?: boolean; nativeExecutionImplemented?: boolean };
  sourcesReadOnly: boolean;
}

export interface SystemCompute {
  sampledAt: string;
  sampleIntervalSeconds: number | null;
  host: { hostname: string; platform: string; release: string; uptimeSeconds: number | null };
  cpu: {
    model: string | null;
    logicalCores: number | null;
    physicalCores: number | null;
    availableCores: number | null;
    utilizationPercent: number | null;
    loadAverage: [number, number, number] | null;
    status: 'available' | 'unavailable';
    message: string | null;
  };
  memory: {
    totalBytes: number | null;
    usedBytes: number | null;
    availableBytes: number | null;
    utilizationPercent: number | null;
    swapTotalBytes: number | null;
    swapUsedBytes: number | null;
    status: 'available' | 'unavailable';
    message: string | null;
  };
  gpu: {
    status: 'available' | 'unavailable' | 'error';
    message: string | null;
    devices: Array<{
      index: number;
      name: string;
      uuid: string | null;
      driverVersion: string | null;
      utilizationPercent: number | null;
      memoryTotalBytes: number | null;
      memoryUsedBytes: number | null;
      memoryFreeBytes: number | null;
      memoryUtilizationPercent: number | null;
      temperatureCelsius: number | null;
      powerWatts: number | null;
      powerLimitWatts: number | null;
    }>;
  };
  disks: Array<{
    path: string;
    role: 'workspace' | 'data';
    totalBytes: number | null;
    usedBytes: number | null;
    freeBytes: number | null;
    utilizationPercent: number | null;
    status: 'available' | 'unavailable';
    message: string | null;
  }>;
}
