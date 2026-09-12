import { downloadArtifact, request } from './client';
import type { ResourcePolicy, TrainingRecipe, TrainingMetricDetails, TrainingMetrics } from './development';
import type { LifecycleState } from './lifecycle';
import type { Finding, ProtocolSpec } from './scientific';
import type { MILExperimentSpec } from './mil';
import type { EvaluationInference, EvaluationPreview } from './evaluation';

export interface PredictorChoice {
  experimentId: string; experimentName: string; batchId: string; batchName: string;
  candidateId: string; candidateNumber: number; trainingSeed: number; splitSeed: number;
  completedRuns: number; totalRuns: number; eligible: boolean; reason: string | null;
  existingPredictorId: string | null; recipe: TrainingRecipe;
  existingPredictorIds?: Partial<Record<PredictorMethod, string>>;
  existingRefitId?: string | null; eligibleMethods?: PredictorMethod[];
}
export type PredictorMethod = 'ensemble' | 'refit';
export const predictorMethodLabel = (method?: PredictorMethod) => method === 'refit' ? 'Refit' : 'Fold ensemble';
export interface PredictorSelection {
  experimentId: string; batchId: string; candidateId: string; trainingSeed: number; splitSeed: number; name: string;
  method?: PredictorMethod; refitPercentile?: number;
}
export interface PredictorManifest extends PredictorSelection {
  candidateNumber?: number;
  kind: 'frozen-predictor'; runIds: string[];
  checkpoints: { runId: string; path: string; sha256: string; bytes: number }[];
  sourceCheckpoints?: { runId: string; path: string; sha256: string; bytes: number }[];
  target: ProtocolSpec['target']; recipe: TrainingRecipe;
  inputs: { protocol: { id: string; contentHash: string }; features: { bundle: { id: string; contentHash: string }; [key: string]: unknown }; loading: MILExperimentSpec; resolvedLoading?: { policy?: string; packArtifactId?: string | null; packPath?: string | null } };
  aggregation: 'mean_probability' | 'single_model';
  experiment?: { id: string; name: string };
  epochBudget?: { percentile: number; epochs: number; foldBestEpochs: { runId: string; bestEpoch: number; source?: string }[]; rounding: string; interpolation: string };
  trainingSlideCount?: number; trainingPatientCount?: number;
  refitId?: string;
}
export interface FrozenPredictor {
  id: string; createdAt: string; contentHash: string; lifecycleState: LifecycleState;
  manifest: PredictorManifest;
}
export interface PredictorPreview {
  canFreeze: boolean; previewHash: string | null; findings: Finding[]; manifest: PredictorManifest | RefitBuild['manifest'] | null;
}
export interface EvaluationMetrics extends TrainingMetricDetails {
  decisionThreshold?: number;
  slide: TrainingMetrics & { predictionCount?: number; unlabeledCount?: number };
  patient: TrainingMetrics & { predictionCount?: number; unlabeledCount?: number };
  selected: TrainingMetrics & { predictionCount?: number; unlabeledCount?: number };
}
export interface ComputeExecution {
  progressWarning?: string | null;
  status: 'not_started' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  cancellationRequested?: boolean; error?: string | null; sessionName?: string; logPath?: string; updatedAt?: string;
  result?: { metrics?: EvaluationMetrics; [key: string]: unknown } | null;
  progress?: { epoch?: number; maxEpochs?: number; trainingLoss?: number | null; completedModels?: number; totalModels?: number; slideCount?: number; completedPairs?: number; totalPairs?: number; currentSlide?: string; completedSlides?: number; totalSlides?: number } | null;
}
export const computeActive = (job?: ComputeExecution) => Boolean(job && ['queued', 'running'].includes(job.status));
export const computeStatusLabel = (job?: ComputeExecution) => job?.cancellationRequested && computeActive(job) ? 'Cancelling' : ({ not_started: 'Ready to run', completed: 'Completed', queued: 'Queued', running: 'Running', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted' }[job?.status ?? 'not_started']);
export interface RefitBuild {
  id: string; createdAt: string; contentHash: string; lifecycleState: LifecycleState;
  manifest: Omit<PredictorManifest, 'kind'> & { kind: 'predictor-refit'; resources?: ResourcePolicy };
  execution?: ComputeExecution; predictorId?: string | null;
}
export interface EvaluationSelection { predictorId: string; cohortId: string; name: string; featureBundleId?: string; inference?: EvaluationInference; patientIdentifiers?: 'shared' | 'independent' }
export interface ModelEvaluation {
  id: string; createdAt: string; contentHash: string; lifecycleState: LifecycleState;
  manifest: EvaluationSelection & { kind: 'model-evaluation'; experimentId: string; status: 'planned'; coverage?: EvaluationPreview['coverage']; overlap?: EvaluationPreview['overlap']; [key: string]: unknown };
  execution?: ComputeExecution;
}
export interface ModelEvaluationPreview {
  canSave: boolean; previewHash: string | null; findings: Finding[]; manifest: ModelEvaluation['manifest'] | null; executionEnabled: boolean;
}
const base = (project: string) => `/projects/${encodeURIComponent(project)}`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const predictors = {
  list: (project: string) => request<{ items: FrozenPredictor[]; executionEnabled: boolean }>(`${base(project)}/predictors?include_inactive=true`),
  choices: (project: string) => request<{ items: PredictorChoice[]; executionEnabled: boolean }>(`${base(project)}/predictors/choices`),
  preview: (project: string, selection: PredictorSelection) => request<PredictorPreview>(`${base(project)}/predictors/preview`, post(selection)),
  freeze: (project: string, selection: PredictorSelection, previewHash: string, operationId: string) => request<FrozenPredictor>(`${base(project)}/predictors/freeze`, post({ ...selection, previewHash, operationId })),
  refits: (project: string) => request<{ items: RefitBuild[] }>(`${base(project)}/predictors/refits?include_inactive=true`),
  planRefit: (project: string, selection: PredictorSelection, previewHash: string, operationId: string) => request<RefitBuild>(`${base(project)}/predictors/refits`, post({ ...selection, previewHash, operationId })),
  refitExecution: (project: string, id: string) => request<ComputeExecution>(`${base(project)}/predictors/refits/${encodeURIComponent(id)}/execution`),
  refitJob: (project: string, id: string, action: 'launch' | 'resume' | 'cancel', operationId: string, resources?: ResourcePolicy) => request<ComputeExecution>(`${base(project)}/predictors/refits/${encodeURIComponent(id)}/${action}`, post({ operationId, ...(resources ? { resources } : {}) })),
  publishRefit: (project: string, id: string, operationId: string) => request<FrozenPredictor>(`${base(project)}/predictors/refits/${encodeURIComponent(id)}/publish`, post({ operationId })),
};
export const modelEvaluations = {
  list: (project: string) => request<{ items: ModelEvaluation[]; executionEnabled: boolean }>(`${base(project)}/evaluation-runs?include_inactive=true`),
  preview: (project: string, selection: EvaluationSelection) => request<ModelEvaluationPreview>(`${base(project)}/evaluation-runs/preview`, post(selection)),
  save: (project: string, selection: EvaluationSelection, previewHash: string, operationId: string) => request<ModelEvaluation>(`${base(project)}/evaluation-runs`, post({ ...selection, previewHash, operationId })),
  execution: (project: string, id: string) => request<ComputeExecution>(`${base(project)}/evaluation-runs/${encodeURIComponent(id)}/execution`),
  job: (project: string, id: string, action: 'launch' | 'resume' | 'cancel', operationId: string) => request<ComputeExecution>(`${base(project)}/evaluation-runs/${encodeURIComponent(id)}/${action}`, post({ operationId })),
  download: (project: string, id: string, filename: 'slide-predictions.csv' | 'patient-predictions.csv' | 'predictions.json' | 'metrics.json') => downloadArtifact(`${base(project)}/evaluation-runs/${encodeURIComponent(id)}/artifacts/${filename}`, filename),
};
