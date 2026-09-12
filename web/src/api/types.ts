import type { VersionLabel } from './scientific';

export type Page =
  | 'overview'
  | 'dataset'
  | 'cohort'
  | 'features'
  | 'experiments'
  | 'post-development'
  | 'source-cv'
  | 'selection'
  | 'predictor'
  | 'test-data'
  | 'evaluation'
  | 'clinical-utility'
  | 'interpretation'
  | 'reports'
  | 'example-results'
  | 'explorer'
  | 'provenance'
  | 'cleanup'
  | 'system';
export interface Patient {
  id: string;
  site: string;
  specimenType: string;
  msi: string;
  braf: string;
  kras: string;
  partition: string;
}
export interface Slide {
  id: string;
  patientId: string;
  specimenId: string;
  site: string;
  status: string;
  filename: string;
}
export interface Encoder {
  id: string;
  name: string;
  description: string;
  dimensions: number;
  adapter: string;
}
export interface MilModel {
  id: string;
  name: string;
  description: string;
  adapter: string;
}
export interface FeatureSet {
  id: string;
  encoderId: string;
  slides: number;
  dimensions: number;
  status: string;
}
export interface Result {
  id: string;
  encoderId: string;
  milId: string;
  auroc: number;
  auprc: number;
  accuracy: number;
  seed: number;
  splitId: string;
  featureSetId: string;
}
export interface Cohort {
  id: string;
  name: string;
  datasetId: string;
  target: string;
  filters: { specimenType: string; msi: string; braf: string };
  patientIds: string[];
  slideIds: string[];
  splitId: string;
  createdAt: string;
}
export interface Experiment {
  id: string;
  status: string;
  datasetId: string;
  cohortId: string;
  cohortSnapshot: Cohort;
  splitId: string;
  encoderId: string;
  milId: string;
  featureSetId: string | null;
  seeds: number[];
  folds: number;
  aggregation: string;
  createdAt: string;
}
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
export interface Manifest extends Record<string, unknown> {
  result: { id: string };
  run: { id: string };
  experiment: { id: string };
  features: { id: string };
  split: { id: string };
  dataset: { id: string };
}
export interface Workspace {
  project: ProjectSummary;
  mode: 'local' | 'synthetic-demo';
  executionEnabled: boolean;
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
  };
  patients: Patient[];
  slides: Slide[];
  encoders: Encoder[];
  milModels: MilModel[];
  featureSets: FeatureSet[];
  split: { id: string; seed: number; groupBy: string };
  results: Result[];
  cohortSnapshots: Cohort[];
  drafts: Experiment[];
  sources: Source[];
  exampleManifests: Manifest[];
}
export interface CohortInput {
  datasetId: string;
  specimenType: string;
  msi: string;
  braf: string;
}
export interface ExperimentInput {
  cohortId: string;
  pairs: string[];
  seeds: number[];
  folds: number;
  aggregation: 'mean' | 'max';
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
export interface Job {
  id: string;
  status: string;
  name?: string;
}
export interface Jobs {
  jobs: Job[];
  executionEnabled: boolean;
}
export interface SystemStatus {
  mode: string;
  workspace: string;
  storage: { engine: string; journalMode: string; schemaVersion: number };
  control: { cudaModelsLoaded: boolean; process: string };
  workers: { executionEnabled: boolean; status: string };
  sourcesReadOnly: boolean;
}
