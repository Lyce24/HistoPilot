import { request, requestScientificSave } from './client';
import type { Condition, Finding, ProtocolSpec, ScientificDraft, VersionLabel, VersionLabelInput } from './scientific';

export interface EvaluationInference {
  loadingPolicy: 'per_slide' | 'packed';
  packArtifactId: string | null;
  batchSize: number;
  numWorkers: number;
  device: 'auto' | 'cpu' | 'cuda';
  precision: 'float32' | 'float16' | 'bfloat16';
  patientAggregation: 'mean' | 'max';
  decisionThreshold: number;
}

export interface EvaluationSpec {
  protocolId: string;
  developmentFeatureBundleId: string;
  datasetId: string;
  featureBundleId: string;
  target: ProtocolSpec['target'] | null;
  eligibility: Condition[];
  patientIdentifiers: 'shared' | 'independent';
  inference: EvaluationInference;
}

export type EvaluationDraft = Omit<ScientificDraft<EvaluationSpec>, 'payload'> & {
  payload: { type: 'evaluation-cohort'; spec: EvaluationSpec };
};

export interface EvaluationSummary {
  includedSlides: number;
  includedPatients: number;
  excludedSlides: number;
  labeledSlides: number;
  classCounts: Record<string, number>;
  developmentSlideOverlap: number;
  developmentPatientOverlap: number;
}

export interface EvaluationPreview {
  spec: EvaluationSpec;
  target: ProtocolSpec['target'];
  summary: EvaluationSummary;
  coverage: {
    selectedSlideIds: string[];
    featureSlideCount: number;
    missingFeatureSlideIds: string[];
    missingPackSlideIds: string[];
    packChecked: boolean;
  };
  overlap: { slideIds: string[]; patientIds: string[]; patientsComparable: boolean; sourceSlideIds?: string[] };
  compatibility: {
    development: { dimensions: number | null; encoderId: string | null };
    evaluation: { dimensions: number | null; encoderId: string | null };
  };
  findings: Finding[];
  canFreeze: boolean;
  executionEnabled: false;
  previewHash: string;
}

export interface EvaluationCohort {
  id: string;
  projectId: string;
  contentHash: string;
  createdAt: string;
  versionLabel?: VersionLabel | null;
  current?: boolean;
  findings?: Finding[];
  executionEnabled?: false;
  manifest: {
    kind: 'evaluation-cohort';
    datasetId: string;
    spec: EvaluationSpec;
    target: ProtocolSpec['target'];
    summary: EvaluationSummary;
    findings: Finding[];
  };
}

const base = (project: string) => `/projects/${encodeURIComponent(project)}`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });

export const evaluation = {
  list: (project: string) => request<{ items: EvaluationCohort[] }>(`${base(project)}/evaluation-cohorts`),
  get: (project: string, id: string) => request<EvaluationCohort>(`${base(project)}/evaluation-cohorts/${encodeURIComponent(id)}`),
  drafts: async (project: string) => {
    const response = await request<{ drafts: { payload: { type: string } }[] }>(`${base(project)}/drafts`);
    return { drafts: response.drafts.filter((draft) => draft.payload.type === 'evaluation-cohort') as EvaluationDraft[] };
  },
  draft: (project: string, id: string) => request<EvaluationDraft>(`${base(project)}/drafts/${encodeURIComponent(id)}`),
  saveDraft: (project: string, name: string, spec: EvaluationSpec, current?: { id: string; revision: number }) => {
    const payload = { type: 'evaluation-cohort', spec };
    return current
      ? request<EvaluationDraft>(`${base(project)}/drafts/${encodeURIComponent(current.id)}`, {
        method: 'PATCH', body: JSON.stringify({ expectedRevision: current.revision, name, payload }),
      })
      : request<EvaluationDraft>(`${base(project)}/drafts`, post({ kind: 'experiment', name, payload }));
  },
  preview: (project: string, draft: { id: string; revision: number }) => request<EvaluationPreview>(
    `${base(project)}/drafts/${encodeURIComponent(draft.id)}/evaluation-preview`, post({ expectedRevision: draft.revision }),
  ),
  freeze: (project: string, draft: { id: string; revision: number }, previewHash: string, operationId: string, versionLabel: VersionLabelInput) => requestScientificSave<EvaluationCohort>(
    `${base(project)}/drafts/${encodeURIComponent(draft.id)}/evaluation-freeze`,
    post({ expectedRevision: draft.revision, previewHash, operationId, versionLabel }), 'taggedFreeze',
  ),
};
