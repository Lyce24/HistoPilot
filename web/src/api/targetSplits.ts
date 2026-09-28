import { request, requestScientificSave } from './client';
import type { Condition, DataRecord, Finding, ProtocolCohortStats, ProtocolSpec, ScientificDraft, VersionLabel, VersionLabelInput } from './scientific';

export type TargetSplitUnit = 'slide' | 'patient';

export interface TargetSplitSpec {
  datasetId: string;
  /** Historical specifications without this field retain patient grouping. */
  splitUnit?: TargetSplitUnit;
  target: ProtocolSpec['target'];
  /** Omitted inherits training; null preserves an unlabeled testing/inference set. */
  testTarget?: ProtocolSpec['target'] | null;
  predictors: string[];
  eligibility: Condition[];
  split: {
    method: 'random' | 'rules' | 'imported';
    testFraction: number;
    seed: number;
    stratify: boolean;
    stratifyField?: string | null;
    trainRules: Condition[];
    testRules: Condition[];
    /** Rules method only: testing takes every eligible slide or group that training does not select. */
    testRemaining?: boolean;
    partitionField?: string | null;
    trainValues: string[];
    testValues: string[];
  };
}

export interface TargetSplitSummary extends TargetSplitTestingShare {
  splitUnit?: TargetSplitUnit;
  totalSlides: number;
  includedSlides: number;
  includedPatients: number;
  excludedSlides: number;
  trainingSlides: number;
  testingSlides: number;
  trainingPatients: number;
  testingPatients: number;
  trainingGroups: number;
  testingGroups: number;
  classCounts: Record<string, number>;
  trainingClassCounts: Record<string, number>;
  testingClassCounts: Record<string, number>;
  trainingPatientClassCounts?: Record<string, number>;
  testingPatientClassCounts?: Record<string, number>;
  testingPurpose?: 'independent' | 'inference';
  selectedTrainingSlides?: number;
  selectedTestingSlides?: number;
  selectedTrainingPatients?: number;
  selectedTestingPatients?: number;
}

/** Random splits only: the split units (slides or patient groups) in testing and in total. */
export interface TargetSplitTestingShare {
  testingUnits?: number;
  splitUnits?: number;
  achievedTestFraction?: number;
}

/** "3 of 12 slides (25%)": the testing share a random split reached, or null for other methods. */
export function testingShareText(summary: TargetSplitTestingShare, unit: TargetSplitUnit): string | null {
  if (summary.achievedTestFraction === undefined || summary.testingUnits === undefined || summary.splitUnits === undefined) return null;
  const units = unit === 'slide' ? 'slides' : 'patient groups';
  return `${summary.testingUnits.toLocaleString()} of ${summary.splitUnits.toLocaleString()} ${units} (${Number((summary.achievedTestFraction * 100).toPrecision(3))}%)`;
}

export interface TargetSplitMembership extends Pick<DataRecord, 'slideId' | 'patientId' | 'patientIdSource'> {
  label: string | null;
  partition: 'train' | 'test';
}

export interface TargetSplitTargetDistribution {
  field: string;
  values: { value: string | null; slides: number; patients: number }[];
  distinctCount: number;
  classCounts: Record<string, number>;
  patientClassCounts: Record<string, number>;
  missingSlides: number;
  unmappedSlides: number;
  mixedValuePatients?: number;
  mixedLabelPatients: number;
  missingLabelPatients: number;
  unmappedLabelPatients: number;
  labeledPatients: number;
  unlabeledPatients: number;
}
export interface TargetSplitPartition {
  slides: number;
  patients: number;
  groups: number;
  fallbackSlides: number;
  unlinkedSlides: number;
  target?: TargetSplitTargetDistribution | null;
  selection?: {
    mode: 'rules' | 'remaining' | 'none' | 'imported' | 'random';
    directMatches: ProtocolCohortStats;
    expanded: ProtocolCohortStats;
    assigned: ProtocolCohortStats;
  };
}
export interface TargetSplitPartitionRequest {
  datasetId: string;
  splitUnit?: TargetSplitUnit;
  eligibility: Condition[];
  split: TargetSplitSpec['split'];
  targetFields?: { train?: string; test?: string };
  target?: ProtocolSpec['target'];
  testTarget?: ProtocolSpec['target'] | null;
}
export interface TargetSplitPartitionPreview {
  dataset?: ProtocolCohortStats;
  cohort?: ProtocolCohortStats;
  summary: TargetSplitTestingShare & { totalSlides: number; eligibleSlides: number; selectedSlides: number; excludedSlides: number; trainingSlides: number; testingSlides: number; trainingPatients: number; testingPatients: number };
  partitions: Record<'train' | 'test', TargetSplitPartition>;
  findings: Finding[];
  valid: boolean;
  membershipStatus?: 'fixed' | 'provisional';
}

export const TARGET_SPLIT_STEPS = ['Dataset & cohort', 'Training & Testing split', 'Prediction Targets', 'Review & Freeze'] as const;

export function targetDefinitionReady(target: ProtocolSpec['target'] | null | undefined): boolean {
  return Boolean(target?.field && target.task && target.classes.length >= 2 && target.classes.every(Boolean)
    && new Set(target.classes).size === target.classes.length && Object.keys(target.labels).length
    && Object.entries(target.labels).every(([raw, label]) => Boolean(raw) && target.classes.includes(label))
    && target.classes.every((label) => Object.values(target.labels).includes(label))
    && (target.task !== 'binary_classification' || target.classes.length === 2 && target.classes.includes(target.positiveClass ?? '')));
}

export function targetSplitTestingIssue(spec: TargetSplitSpec): string | null {
  if (!spec.testTarget) return null;
  const testing = spec.testTarget;
  if (testing.task !== spec.target.task || testing.unit !== spec.target.unit
    || JSON.stringify(testing.classes) !== JSON.stringify(spec.target.classes) || testing.positiveClass !== spec.target.positiveClass) {
    return 'Testing targets must use the training task, label unit, class order and positive class.';
  }
  if (testing.field === spec.target.field && Object.entries(testing.labels).some(([raw, label]) => Object.hasOwn(spec.target.labels, raw) && spec.target.labels[raw] !== label)) {
    return 'The same target field must preserve training label mappings. Use the training mapping, or choose a different testing field.';
  }
  return null;
}

export function targetSplitPartitionRequest(spec: TargetSplitSpec, targetFields?: { train?: string; test?: string }, rawOnly = false): TargetSplitPartitionRequest {
  const testing = spec.testTarget === undefined ? spec.target : spec.testTarget;
  const mapped = !rawOnly && !targetSplitTestingIssue(spec) && targetDefinitionReady(spec.target) && (testing === null || targetDefinitionReady(testing));
  return { datasetId: spec.datasetId, ...(spec.splitUnit ? { splitUnit: spec.splitUnit } : {}), eligibility: spec.eligibility, split: spec.split,
    targetFields: targetFields ?? { train: spec.target.field || undefined, test: testing?.field || undefined },
    ...(mapped ? { target: spec.target, ...(spec.testTarget !== undefined ? { testTarget: spec.testTarget } : {}) }
      : spec.testTarget === null ? { testTarget: null } : {}),
  };
}

/** Missing units are legacy patient-grouped artifacts, never new slide defaults. */
export const targetSplitUnit = (spec: Pick<TargetSplitSpec, 'splitUnit'>): TargetSplitUnit => spec.splitUnit ?? 'patient';

export function targetSplitWithUnit(spec: TargetSplitSpec, unit: TargetSplitUnit): TargetSplitSpec {
  return { ...spec, splitUnit: unit, target: { ...spec.target, unit }, ...(spec.testTarget ? { testTarget: { ...spec.testTarget, unit } } : {}) };
}

/** Independent testing mappings must keep the same model output definition. */
export function targetSplitTrainingTarget(spec: TargetSplitSpec, target: ProtocolSpec['target']): Pick<TargetSplitSpec, 'target' | 'testTarget'> {
  return { target, ...(spec.testTarget ? { testTarget: { ...spec.testTarget, task: target.task, unit: target.unit, classes: target.classes, positiveClass: target.positiveClass } } : {}) };
}

export interface TargetSplitPreview {
  spec: TargetSplitSpec;
  summary: TargetSplitSummary;
  memberships: TargetSplitMembership[];
  findings: Finding[];
  partitions?: Record<'train' | 'test', TargetSplitPartition>;
  canFreeze: boolean;
  previewHash: string;
}

export interface TargetSplit {
  id: string;
  projectId: string;
  contentHash: string;
  createdAt: string;
  versionLabel?: VersionLabel | null;
  evaluationCohortId?: string | null;
  /** Detail and freeze responses only. Freezing derives the testing cohort; reads never create it. */
  testCohort?: TargetSplitTestCohort;
  /** Freeze only: the version is frozen, but its testing cohort still needs an explicit retry. */
  testCohortError?: { code: string; message: string } | null;
  manifest: {
    kind: 'target-split';
    datasetId: string;
    spec: TargetSplitSpec;
    summary: TargetSplitSummary;
    memberships: TargetSplitMembership[];
    partitions?: Record<'train' | 'test', TargetSplitPartition>;
    findings: Finding[];
    previewHash: string;
    [key: string]: unknown;
  };
}

/** `required` with no `id`: the testing set has no derived evaluation or inference cohort yet. */
export interface TargetSplitTestCohort {
  required: boolean;
  id: string | null;
  state: 'active' | 'archived' | 'trashed' | null;
}

export type TargetSplitDraft = ScientificDraft<TargetSplitSpec>;

export function newTargetSplitSpec(datasetId = '', seed = 42): TargetSplitSpec {
  return {
    datasetId, splitUnit: 'slide', target: { field: '', task: '', unit: 'slide', classes: [], labels: {}, missing: 'block', unmapped: 'block' },
    predictors: [], eligibility: [],
    split: { method: 'random', testFraction: 0.2, seed, stratify: false, trainRules: [], testRules: [], trainValues: [], testValues: [] },
  };
}

/** Keep each selection method free of settings from a different method. */
export function targetSplitMethod(split: TargetSplitSpec['split'], method: TargetSplitSpec['split']['method']): TargetSplitSpec['split'] {
  return { ...split, method,
    ...(method !== 'rules' ? { trainRules: [], testRules: [], testRemaining: undefined } : {}),
    ...(method !== 'imported' ? { partitionField: undefined, trainValues: [], testValues: [] } : {}),
  };
}

/** Testing uses either its own conditions or the remainder outside training; `restore` returns set-aside conditions. */
export function targetSplitTestingRemainder(split: TargetSplitSpec['split'], remaining: boolean, restore: Condition[] = []): TargetSplitSpec['split'] {
  return remaining ? { ...split, testRemaining: true, testRules: [] } : { ...split, testRemaining: undefined, testRules: restore };
}

export function targetSplitSetupLink(record: Pick<TargetSplit, 'id' | 'manifest'>): string {
  return `#experimental-setup?${new URLSearchParams({ dataset: record.manifest.datasetId, targetSplit: record.id })}`;
}

const prefix = (project: string) => `/projects/${encodeURIComponent(project)}`;
const body = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const targetSplits = {
  list: (project: string) => request<{ configurations: TargetSplit[] }>(`${prefix(project)}/configurations?kind=target-split`),
  get: (project: string, id: string) => request<TargetSplit>(`${prefix(project)}/target-splits/${encodeURIComponent(id)}`),
  /** Idempotent: derives the testing cohort once per frozen version, or returns the existing one. */
  createTestCohort: (project: string, id: string) => request<{ evaluationCohortId: string | null }>(`${prefix(project)}/target-splits/${encodeURIComponent(id)}/test-cohort`, { method: 'POST' }),
  partitionPreview: (project: string, input: TargetSplitPartitionRequest) => request<TargetSplitPartitionPreview>(`${prefix(project)}/target-splits/partition-preview`, body(input)),
  preview: (project: string, id: string, expectedRevision: number) => request<TargetSplitPreview>(`${prefix(project)}/target-splits/${encodeURIComponent(id)}/preview`, body({ expectedRevision })),
  freeze: (project: string, id: string, expectedRevision: number, previewHash: string, versionLabel: VersionLabelInput, operationId: string) =>
    requestScientificSave<TargetSplit>(`${prefix(project)}/target-splits/${encodeURIComponent(id)}/freeze`, body({ expectedRevision, previewHash, versionLabel, operationId }), 'taggedFreeze'),
};
