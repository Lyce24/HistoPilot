import { request, requestScientificSave } from './client';

export interface Finding {
  severity: 'error' | 'warning' | 'info';
  code: string;
  message: string;
  count?: number;
}
export interface TableSource {
  path?: string;
  contentBase64?: string;
  filename?: string;
  sheet?: string;
}
export interface AttributeMapping {
  key: string;
  sourceColumn: string;
  owner: 'slide' | 'patient';
  type:
    'text' | 'categorical' | 'ordered_categorical' | 'integer' | 'decimal' | 'boolean' | 'date';
  missingValues?: string[];
  categories?: string[];
}
export interface ImportSpec {
  parentId?: string;
  source: TableSource;
  slideIdColumn: string;
  patientIdColumn?: string;
  patientIdFallback?: 'unresolved' | 'slide_id';
  slideRoot?: string;
  recursive: boolean;
  includeMissingSlides: boolean;
  missingValues: string[];
  attributes: AttributeMapping[];
  patientSource?: TableSource;
  patientSourceKind?: 'crosswalk' | 'patients';
  patientSourceSlideIdColumn?: string;
  patientSourcePatientIdColumn?: string;
  patientAttributes?: AttributeMapping[];
}
export interface Inspection {
  headers: string[];
  sheets: string[];
  sheet: string | null;
  rows: Record<string, string | null>[];
  rowCount: number;
  columnSummaries?: Record<
    string,
    { examples: string[]; distinctCount: number; missingCount: number }
  >;
  fingerprint: string;
  findings: Finding[];
}
export interface ScientificDraft<T = ImportSpec | ProtocolSpec> {
  id: string;
  projectId: string;
  kind: 'import' | 'experiment';
  name: string;
  payload: { type: 'dataset-import' | 'analysis-protocol' | 'mil-experiment'; spec: T };
  revision: number;
  status: 'editable' | 'frozen';
  createdAt: string;
  updatedAt: string;
}
export interface DataRecord {
  slideId: string;
  patientId: string | null;
  patientIdSource?: 'source' | 'crosswalk' | 'slide_fallback' | 'unresolved';
  slidePath: string | null;
  attributes: Record<string, string | number | boolean | null>;
}
export interface ImportSummary {
  sourceRowCount: number;
  slideCount: number;
  mappedPatientCount: number;
  verifiedPatientCount?: number;
  fallbackSlideCount?: number;
  unlinkedSlideCount: number;
  matchedSlideCount: number;
  missingSlideCount: number;
  unmatchedFileCount: number;
  excludedRowCount: number;
  scannedFileCount: number;
}
export interface ImportPreview {
  draftId: string;
  revision: number;
  previewHash: string;
  canFreeze: boolean;
  findings: Finding[];
  summary: ImportSummary;
  dictionary: AttributeMapping[];
  records: DataRecord[];
  recordsTruncated: boolean;
}
export interface VersionLabelInput {
  tag: string;
  note: string;
}
export interface VersionLabel extends VersionLabelInput {
  revision: number;
  createdAt: string;
  updatedAt: string;
}
export interface VersionLabelResource {
  id: string;
  versionLabel?: VersionLabel | null;
}
export interface DatasetVersion {
  id: string;
  projectId: string;
  createdAt: string;
  contentHash: string;
  versionLabel?: VersionLabel | null;
  manifest: {
    name?: string;
    kind?: string;
    summary?: ImportSummary;
    dictionary?: AttributeMapping[];
    provenance?: unknown;
    [key: string]: unknown;
  };
  artifacts: Record<string, { sha256: string; sizeBytes: number }>;
}
export interface RecordsPage {
  records: DataRecord[];
  total: number;
  offset: number;
  limit: number;
}
export interface DatasetQuery {
  field?: string;
  compare?: string;
  search: string;
  filters: { field: string; values: (string | null)[] }[];
  offset: number;
  limit: number;
}
export interface DatasetQueryResult extends RecordsPage {
  totalSlides: number;
  summary: {
    slideCount: number;
    mappedPatientCount: number;
    unlinkedSlideCount: number;
    verifiedPatientCount?: number;
    fallbackSlideCount?: number;
    groupCount?: number;
  };
  distribution: {
    kind: 'categorical' | 'numeric' | 'none';
    unit: 'slide' | 'patient' | 'group';
    counts: { value: string | null; count: number }[];
    missingCount: number;
    total: number;
    unlinkedSlideCount?: number;
    truncated?: boolean;
    otherCount?: number;
    omittedCategoryCount?: number;
  };
  valueCounts: { value: string | null; count: number }[];
  valuesTruncated: boolean;
  valueCountsUnit: 'slide';
  crossTab?: {
    rows: (string | null)[];
    columns: (string | null)[];
    counts: number[][];
    unit: string;
    total?: number;
    displayedCount?: number;
    omittedCount?: number;
    truncated?: boolean;
    unlinkedSlideCount?: number;
  } | null;
}
export type ConditionValue = string | number | boolean | null | (string | number)[];
export interface Condition {
  field: string;
  op: 'eq' | 'ne' | 'in' | 'not_in' | 'regex' | 'lt' | 'lte' | 'gt' | 'gte' | 'exists';
  value: ConditionValue;
}
export interface ProtocolCohortStats {
  totalSlides: number;
  patientCount: number;
  fallbackSlideCount: number;
  groupCount: number;
  unlinkedSlideCount: number;
  sample: DataRecord[];
}
export interface ProtocolPartitionStats {
  selection: 'rules' | 'remaining' | 'none';
  directMatches: ProtocolCohortStats;
  expanded: ProtocolCohortStats;
}
export interface ProtocolExploreRequest {
  datasetId: string;
  targetField?: string;
  eligibility: Condition[];
  rules: { train: Condition[]; val: Condition[]; test: Condition[] };
  splitMode: ProtocolSpec['split']['mode'];
  split?: ProtocolSpec['split'];
}
export interface ProtocolExploration {
  selectionBasis?: 'pools';
  datasetId: string;
  splitMode: ProtocolExploreRequest['splitMode'];
  valid: boolean;
  dataset: ProtocolCohortStats;
  cohort: ProtocolCohortStats | null;
  partitions: Record<'train' | 'val' | 'test', ProtocolPartitionStats> | null;
  unassigned: ProtocolCohortStats | null;
  target: {
    field: string;
    values: { value: string | null; slides: number }[];
    distinctCount: number;
  } | null;
  findings: Finding[];
}
export interface ProtocolSpec {
  datasetId: string;
  target: {
    field: string;
    task: '' | 'binary_classification' | 'multiclass_classification';
    unit: 'patient' | 'slide';
    classes: string[];
    labels: Record<string, string>;
    positiveClass?: string;
    missing: 'block' | 'exclude';
    unmapped: 'block' | 'exclude';
  };
  predictors: string[];
  eligibility: Condition[];
  split: {
    version?: 1 | 2 | 3;
    pools?: {
      source: 'rules' | 'imported';
      trainSelection: 'rules' | 'remaining';
      validationSource: 'training_fraction' | 'fixed';
      rules: { train: Condition[]; val: Condition[]; test: Condition[] };
      imported?: ProtocolSpec['split']['imported'];
    };
    mode:
      | 'rules'
      | 'kfold'
      | 'holdout'
      | 'imported'
      | 'monte_carlo'
      | 'leave_one_domain_out'
      | 'nested_kfold'
      | 'held_out';
    validationFraction?: number;
    testFraction?: number;
    repeats?: number;
    outerFolds?: number;
    innerFolds?: number;
    stratify?: boolean;
    domainField?: string;
    domainPolicy?: 'all' | 'selected';
    heldOutDomains?: string[];
    heldOutSource?: 'fractions' | 'rules' | 'imported';
    folds: number;
    seeds: number[];
    ratios: { train: number; val: number; test: number };
    rules: { train: Condition[]; val: Condition[]; test: Condition[] };
    imported?: {
      partitionField?: string;
      foldField?: string;
      partitionLabels: Record<string, 'train' | 'val' | 'test' | 'trainval'>;
      foldLabels: Record<string, number>;
      testFoldLabels: (string | number)[];
    };
  };
  constraints: { minPatientsPerClass: number; minPatientsPerPartition: number };
  featureSetId?: string | null;
  featurePackId?: string | null;
}
export interface PartitionCounts {
  slides: number;
  patients: number;
  groups?: number;
  fallbackSlides?: number;
  classes: Record<string, number>;
}
export interface ProtocolPreview {
  previewHash: string;
  canFreeze: boolean;
  findings: Finding[];
  summary: {
    datasetId: string;
    totalSlides: number;
    eligibleSlides: number;
    includedSlides: number;
    includedPatients: number;
    includedGroups?: number;
    fallbackSlideCount?: number;
    unlinkedSlideCount?: number;
    excludedSlides: number;
    classCounts: Record<string, number>;
    patientClassCounts: Record<string, number>;
    grouping: string;
    algorithm: string;
    strategy?: ProtocolSpec['split']['mode'];
    splitVersion?: 1 | 2 | 3;
    finalPlanCount?: number;
    poolCounts?: Record<'train' | 'val' | 'test', PartitionCounts>;
    validationSource?: 'training_fraction' | 'fixed';
    validationFraction?: number;
    roleDescriptions?: Record<string, string>;
    evaluationPlanCount?: number;
    innerPlanCount?: number;
    oofCoverage?: {
      groups: number;
      testedGroups: number;
      minTestAppearances: number;
      maxTestAppearances: number;
      complete: boolean;
    };
  };
  partitions: {
    seed: number;
    fold: number | null;
    planId?: string;
    phase?: 'evaluation' | 'inner' | 'outer' | 'final';
    pool?: 'training' | 'external_test';
    excludedValidation?: { groups: number; slides: number; groupIds?: string[] };
    repeat?: number;
    outerFold?: number;
    innerFold?: number;
    domain?: string;
    tune?: PartitionCounts;
    train: PartitionCounts;
    val: PartitionCounts;
    test: PartitionCounts;
  }[];
  spec: ProtocolSpec;
  executionEnabled: false;
}
export interface ExecutionPreflight {
  protocolId: string;
  scientificReady: boolean;
  executionEnabled: false;
  executionReady: false;
  tensorValidationComplete?: boolean;
  fullFeatureValidationComplete?: boolean;
  provenanceComplete?: boolean;
  findings: Finding[];
}
export interface FeatureSpec {
  datasetId: string;
  path: string;
  encoderId?: string;
  fileSuffix: '.h5' | '.hdf5';
  idSuffix: string;
  recursive: boolean;
  layout?: 'auto' | 'flat' | 'trident';
  coordinatesPath?: string | null;
  sourceExtractionJobId?: string | null;
}
export interface FeaturePreview {
  previewHash: string;
  canFreeze: boolean;
  findings: Finding[];
  validationLevel: 'headers';
  layout?: {
    kind: 'flat' | 'trident';
    featureDirectory: string;
    coordinatesDirectory: string | null;
    encoderId: string | null;
    jobDirectory: string | null;
  };
  summary: {
    slideCount: number;
    matchedSlides: number;
    missingSlides: number;
    orphanFiles: number;
    dimensions: number | null;
    patchCount: number;
  };
  files: {
    slideId: string;
    path: string;
    patchCount: number;
    dimensions: number;
    dtype: string;
    sizeBytes: number;
    mtimeNs: number;
    coordinatePath?: string | null;
    coordinateSource?: 'embedded' | 'trident-patches' | null;
    coordinateSpace?: 'level0_pixels' | 'unspecified';
    attributes?: {
      file: Record<string, unknown>;
      features: Record<string, unknown>;
      coords: Record<string, unknown>;
    };
  }[];
}
export interface Configuration {
  id: string;
  projectId: string;
  contentHash: string;
  createdAt: string;
  versionLabel?: VersionLabel | null;
  manifest: {
    kind: 'protocol' | 'feature';
    datasetId: string;
    spec: ProtocolSpec | FeatureSpec;
    summary: ProtocolPreview['summary'] | FeaturePreview['summary'];
    partitions?: ProtocolPreview['partitions'];
    files?: FeaturePreview['files'];
    layout?: FeaturePreview['layout'];
    findings?: Finding[];
    [key: string]: unknown;
  };
}
const prefix = (project: string) => `/projects/${encodeURIComponent(project)}`;
const body = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const scientific = {
  drafts: (project: string) =>
    request<{ drafts: ScientificDraft[] }>(`${prefix(project)}/drafts`),
  saveDraft: <T>(
    project: string,
    input: {
      kind: 'import' | 'experiment';
      name: string;
      payload: { type: 'dataset-import' | 'analysis-protocol' | 'mil-experiment'; spec: T };
    },
    current?: { id: string; revision: number },
  ) =>
    current
      ? request<ScientificDraft<T>>(
          `${prefix(project)}/drafts/${encodeURIComponent(current.id)}`,
          {
            method: 'PATCH',
            body: JSON.stringify({
              expectedRevision: current.revision,
              name: input.name,
              payload: input.payload,
            }),
          },
        )
      : request<ScientificDraft<T>>(`${prefix(project)}/drafts`, body(input)),
  draft: <T>(project: string, id: string) =>
    request<ScientificDraft<T>>(`${prefix(project)}/drafts/${encodeURIComponent(id)}`),
  datasets: (project: string) =>
    request<{ datasets: DatasetVersion[] }>(`${prefix(project)}/datasets`),
  dataset: (project: string, id: string) =>
    request<DatasetVersion>(`${prefix(project)}/datasets/${encodeURIComponent(id)}`),
  setVersionLabel: (
    project: string,
    resourceType: 'dataset' | 'configuration',
    id: string,
    label: { tag: string; note: string; expectedRevision: number },
  ) => requestScientificSave<VersionLabel>(
    `${prefix(project)}/${resourceType === 'dataset' ? 'datasets' : 'configurations'}/${encodeURIComponent(id)}/label`,
    { method: 'PUT', body: JSON.stringify(label) },
    'versionLabels',
  ),
  records: (project: string, id: string, offset = 0) =>
    request<RecordsPage>(
      `${prefix(project)}/datasets/${encodeURIComponent(id)}/records?offset=${offset}&limit=200`,
    ),
  queryDataset: (project: string, id: string, query: DatasetQuery) =>
    request<DatasetQueryResult>(
      `${prefix(project)}/datasets/${encodeURIComponent(id)}/query`,
      body(query),
    ),
  inspect: (project: string, source: TableSource) =>
    request<Inspection>(`${prefix(project)}/imports/inspect`, body({ source })),
  importPreview: (project: string, id: string, expectedRevision: number) =>
    request<ImportPreview>(
      `${prefix(project)}/imports/${encodeURIComponent(id)}/preview`,
      body({ expectedRevision }),
    ),
  importFreeze: (project: string, id: string, expectedRevision: number, previewHash: string, versionLabel: VersionLabelInput, operationId: string) =>
    requestScientificSave<DatasetVersion>(
      `${prefix(project)}/imports/${encodeURIComponent(id)}/freeze`,
      body({
        expectedRevision,
        previewHash,
        versionLabel,
        operationId,
      }),
      'taggedFreeze',
    ),
  protocolPreflight: (project: string, id: string) =>
    request<ExecutionPreflight>(
      `${prefix(project)}/protocols/${encodeURIComponent(id)}/preflight`,
    ),
  exploreProtocol: (project: string, input: ProtocolExploreRequest) =>
    request<ProtocolExploration>(`${prefix(project)}/protocols/explore`, body(input)),
  protocolPreview: (project: string, id: string, expectedRevision: number) =>
    request<ProtocolPreview>(
      `${prefix(project)}/protocols/${encodeURIComponent(id)}/preview`,
      body({ expectedRevision }),
    ),
  protocolFreeze: (
    project: string,
    id: string,
    expectedRevision: number,
    previewHash: string,
    versionLabel: VersionLabelInput,
    operationId: string,
  ) =>
    requestScientificSave<Configuration>(
      `${prefix(project)}/protocols/${encodeURIComponent(id)}/freeze`,
      body({
        expectedRevision,
        previewHash,
        versionLabel,
        operationId,
      }),
      'taggedFreeze',
    ),
  configurations: (project: string, kind: 'protocol' | 'feature') =>
    request<{ configurations: Configuration[] }>(
      `${prefix(project)}/configurations?kind=${kind}`,
    ),
  configuration: (project: string, id: string) =>
    request<Configuration>(`${prefix(project)}/configurations/${encodeURIComponent(id)}`),
  featurePreview: (project: string, spec: FeatureSpec) =>
    request<FeaturePreview>(`${prefix(project)}/features/preview`, body(spec)),
  featureFreeze: (project: string, spec: FeatureSpec, previewHash: string, versionLabel: VersionLabelInput, operationId: string) =>
    requestScientificSave<Configuration>(
      `${prefix(project)}/features/freeze`,
      body({ ...spec, previewHash, versionLabel, operationId }),
      'taggedFreeze',
    ),
};
