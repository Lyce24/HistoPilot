import { downloadArtifact, request } from './client';
import type { TaskState } from './taskCenter';
import type { ResourcePolicy, TrainingRecipe, TrainingMetricDetails, TrainingMetrics } from './development';
import type { LifecycleState } from './lifecycle';
import type { Finding, ProtocolSpec } from './scientific';
import type { MILExperimentSpec } from './mil';
import type { EvaluationInference, EvaluationPreview } from './evaluation';
import type { PatientComparison } from './statistics';
import type { InferenceResultSummary } from './inference';

export interface PredictorChoice {
  experimentId: string; experimentName: string; batchId: string; batchName: string;
  candidateId: string; candidateNumber: number; trainingSeed: number; splitSeed: number;
  completedRuns: number; totalRuns: number; eligible: boolean; reason: string | null;
  existingPredictorId: string | null; recipe: TrainingRecipe;
  existingPredictorIds?: Partial<Record<PredictorMethod, string>>;
  existingRefitId?: string | null; eligibleMethods?: PredictorMethod[];
}
export type PredictorMethod = 'ensemble' | 'refit' | 'seed_ensemble';
export const predictorMethodLabel = (method?: PredictorMethod) => method === 'refit' ? 'Refit' : method === 'seed_ensemble' ? 'Seed ensemble' : 'Fold ensemble';
export interface PredictorSelection {
  experimentId: string; batchId: string; candidateId: string; trainingSeed: number; splitSeed: number; name: string;
  method?: PredictorMethod; refitPercentile?: number;
}
/** A seed ensemble pools every seed group of one configuration, so it has no single seed. */
export interface PredictorManifest extends Omit<PredictorSelection, 'trainingSeed' | 'splitSeed'> {
  trainingSeed?: number; splitSeed?: number;
  trainingSeeds?: number[]; splitSeeds?: number[];
  seedGroups?: { trainingSeed: number; splitSeed: number; runIds: string[] }[];
  candidateNumber?: number;
  kind: 'frozen-predictor'; runIds: string[];
  checkpoints: { runId: string; path: string; sha256: string; bytes: number }[];
  sourceCheckpoints?: { runId: string; path: string; sha256: string; bytes: number }[];
  target: ProtocolSpec['target']; recipe: TrainingRecipe;
  inputs: { protocol: { id: string; contentHash: string }; features: { bundle: { id: string; contentHash: string }; [key: string]: unknown }; loading: MILExperimentSpec; resolvedLoading?: { policy?: string; packArtifactId?: string | null; packPath?: string | null } };
  aggregation: 'mean_probability' | 'mean_logit' | 'single_model';
  experiment?: { id: string; name: string };
  epochBudget?: { percentile: number; epochs: number; foldBestEpochs: { runId: string; bestEpoch: number; source?: string; checkpointSelection?: 'best_validation' | 'latest' | 'final_epoch'; validationBestEpoch?: number }[]; rounding: string; interpolation: string };
  trainingSlideCount?: number; trainingPatientCount?: number;
  refitId?: string;
}
/** "Train 42 / split 42" for one seed group; the seed counts and models for a seed ensemble. */
export function predictorSeedLabel(manifest: Pick<PredictorManifest, 'trainingSeed' | 'splitSeed' | 'trainingSeeds' | 'splitSeeds' | 'checkpoints'>) {
  if (manifest.trainingSeeds) {
    const training = manifest.trainingSeeds.length, split = manifest.splitSeeds?.length ?? 1;
    return `${training} training × ${split} split seed${split === 1 ? '' : 's'} · ${manifest.checkpoints.length} models`;
  }
  return `Train ${manifest.trainingSeed} / split ${manifest.splitSeed}`;
}
export interface SeedEnsembleChoice {
  experimentId: string; experimentName: string; batchId: string; batchName: string;
  candidateId: string; candidateNumber: number; trainingSeeds: number[]; splitSeeds: number[];
  seedGroups: number; members: number; completedRuns: number;
  eligible: boolean; reason: string | null; existingPredictorId: string | null;
}
export interface SeedEnsembleSelection { experimentId: string; batchId: string; candidateId: string; name: string }
export interface FrozenPredictor {
  id: string; createdAt: string; contentHash: string; lifecycleState: LifecycleState;
  manifest: PredictorManifest;
}
export interface PredictorPreview {
  canFreeze: boolean; previewHash: string | null; findings: Finding[]; manifest: PredictorManifest | RefitBuild['manifest'] | null;
}
export interface EvaluationMetrics extends TrainingMetricDetails {
  decisionThreshold?: number;
  /** Label-blind runs: slides of development patients were predicted but not scored. */
  developmentExcluded?: { slides: number; patients: number; reason: string };
  /** Label-blind runs are scored by the control service from the frozen cohort labels, or
   * from a reference standard attached to the cohort later. */
  scoredBy?: { method: string; predictionsSha256: string; cohort?: { id: string; contentHash: string }; reference?: { id: string; contentHash: string } };
  /** Scores against a reference standard name it. */
  reference?: { id: string; name: string };
  /** Patients whose slides disagree under a reference standard, left unlabeled for patient scores. */
  conflictingPatients?: number;
  /** List reads of a run not yet opened carry point metrics; its bootstrap intervals come with the run itself. */
  analysisPending?: boolean;
  /** Present on evaluations of frozen splits; slide-split evaluations have no patient results. */
  splitUnit?: 'slide' | 'patient';
  slide: TrainingMetrics & { predictionCount?: number; unlabeledCount?: number };
  patient: TrainingMetrics & { predictionCount?: number; unlabeledCount?: number };
  selected: TrainingMetrics & { predictionCount?: number; unlabeledCount?: number };
}
export interface ComputeExecution {
  progressWarning?: string | null;
  status: 'not_started' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  cancellationRequested?: boolean; error?: string | null; sessionName?: string; logPath?: string; updatedAt?: string;
  /** "task-center" for Task Center jobs, which report their queue state; "tmux" (or absent) for
   * read-only records created before it, whose stored `sessionName` is never shown. */
  executor?: 'task-center' | 'tmux';
  /** `runnerAlive` is null when the service cannot tell; false means queued work cannot start. */
  task?: { id: string; state: TaskState; attempt: number; waitingReason: string | null; held: boolean; ownerKey?: string; runnerAlive?: boolean | null } | null;
  waitingReason?: string | null;
  /** A label-blind run's `metrics` are scored by the service; `metricsError` says why they are missing. */
  result?: { metrics?: EvaluationMetrics; metricsError?: { code: string; message: string }; purpose?: 'inference' | 'evaluation'; labelsWithheld?: boolean; summary?: InferenceResultSummary; [key: string]: unknown } | null;
  progress?: { epoch?: number; maxEpochs?: number; trainingLoss?: number | null; completedModels?: number; totalModels?: number; slideCount?: number; completedPairs?: number; totalPairs?: number; currentSlide?: string; completedSlides?: number; totalSlides?: number } | null;
}
/**
 * Whether a completed evaluation wrote `patient-predictions.csv`. Slide-split evaluations never
 * do; the saved artifact list says so directly, and results without one fall back to the split unit.
 */
export function hasPatientPredictions(result?: ComputeExecution['result']) {
  const artifacts = result?.artifacts;
  if (artifacts && typeof artifacts === 'object') return Object.hasOwn(artifacts, 'patient-predictions.csv');
  return result?.metrics?.splitUnit !== 'slide';
}
export const computeActive = (job?: ComputeExecution) => Boolean(job && ['queued', 'running'].includes(job.status));
/**
 * Poll only as fast as a list's own jobs change. A project with nothing running
 * should not keep the control service reading its records every few seconds.
 */
export const computePollInterval = (items?: readonly { execution?: ComputeExecution }[]) =>
  items?.some((item) => computeActive(item.execution)) ? 5000 : 30000;
export const computeStatusLabel = (job?: ComputeExecution) => job?.cancellationRequested && computeActive(job) ? 'Cancelling' : ({ not_started: 'Ready to run', completed: 'Completed', queued: 'Queued', running: 'Running', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted' }[job?.status ?? 'not_started']);
export interface RefitBuild {
  id: string; createdAt: string; contentHash: string; lifecycleState: LifecycleState;
  manifest: Omit<PredictorManifest, 'kind'> & { kind: 'predictor-refit'; resources?: ResourcePolicy };
  execution?: ComputeExecution; predictorId?: string | null;
}
export interface EvaluationSelection { predictorId: string; cohortId: string; name: string; featureBundleId?: string; inference?: EvaluationInference; patientIdentifiers?: 'shared' | 'independent' }
export interface ModelEvaluation {
  id: string; createdAt: string; contentHash: string; lifecycleState: LifecycleState;
  manifest: EvaluationSelection & { kind: 'model-evaluation'; experimentId: string; status: 'planned'; purpose?: 'inference' | 'review'; target?: ProtocolSpec['target']; coverage?: EvaluationPreview['coverage']; overlap?: EvaluationPreview['overlap']; [key: string]: unknown };
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
  refits: (project: string) => request<{ items: RefitBuild[] }>(`${base(project)}/predictors/refits?include_inactive=true`),
  refitExecution: (project: string, id: string) => request<ComputeExecution>(`${base(project)}/predictors/refits/${encodeURIComponent(id)}/execution`),
  refitJob: (project: string, id: string, action: 'launch' | 'resume' | 'cancel', operationId: string, resources?: ResourcePolicy) => request<ComputeExecution>(`${base(project)}/predictors/refits/${encodeURIComponent(id)}/${action}`, post({ operationId, ...(resources ? { resources } : {}) })),
  publishRefit: (project: string, id: string, operationId: string) => request<FrozenPredictor>(`${base(project)}/predictors/refits/${encodeURIComponent(id)}/publish`, post({ operationId })),
  seedEnsembles: (project: string, experimentId?: string) => request<{ items: SeedEnsembleChoice[] }>(`${base(project)}/predictors/seed-ensembles${experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : ''}`),
  previewSeedEnsemble: (project: string, selection: SeedEnsembleSelection) => request<PredictorPreview>(`${base(project)}/predictors/seed-ensembles/preview`, post(selection)),
  freezeSeedEnsemble: (project: string, selection: SeedEnsembleSelection, previewHash: string, operationId: string) => request<FrozenPredictor>(`${base(project)}/predictors/seed-ensembles`, post({ ...selection, previewHash, operationId })),
};
export const modelEvaluations = {
  compare: (project: string, leftEvaluationId: string, rightEvaluationId: string, signal?: AbortSignal) => request<PatientComparison>(`${base(project)}/evaluation-runs/compare`, { ...post({ leftEvaluationId, rightEvaluationId }), signal }),
  list: (project: string) => request<{ items: ModelEvaluation[]; executionEnabled: boolean }>(`${base(project)}/evaluation-runs?include_inactive=true`),
  preview: (project: string, selection: EvaluationSelection) => request<ModelEvaluationPreview>(`${base(project)}/evaluation-runs/preview`, post(selection)),
  save: (project: string, selection: EvaluationSelection, previewHash: string, operationId: string) => request<ModelEvaluation>(`${base(project)}/evaluation-runs`, post({ ...selection, previewHash, operationId })),
  execution: (project: string, id: string) => request<ComputeExecution>(`${base(project)}/evaluation-runs/${encodeURIComponent(id)}/execution`),
  job: (project: string, id: string, action: 'launch' | 'resume' | 'cancel', operationId: string) => request<ComputeExecution>(`${base(project)}/evaluation-runs/${encodeURIComponent(id)}/${action}`, post({ operationId })),
  download: (project: string, id: string, filename: 'slide-predictions.csv' | 'patient-predictions.csv' | 'predictions.json' | 'metrics.json' | 'summary.json') => downloadArtifact(`${base(project)}/evaluation-runs/${encodeURIComponent(id)}/artifacts/${filename}`, filename),
};
