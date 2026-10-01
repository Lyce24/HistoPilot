import { fromTemplate, recipePreset, templates } from '../lib/templates';
import { downloadArtifact, request } from './client';
import type { MILExperimentSpec, MILExperimentPreview } from './mil';
import type { Finding, VersionLabel } from './scientific';
import type { ExperimentPredictorPolicy } from './experiments';
import { type PatientAnalysisSettings, type PatientAnalysis, type ConfidenceInterval } from './statistics';

export interface TrainingRecipe {
  inputMode?: 'image' | 'clinical' | 'multimodal';
  clinicalFields?: { field: string; kind: 'numeric' | 'categorical' }[];
  model: string; learningRate: number; weightDecay: number; maxEpochs: number;
  optimizer: 'adam' | 'adamw' | 'sgd'; batchSize: number; bagSize: number | null;
  earlyStopping: boolean; patience: number;
  checkpointMetric: 'validation_loss' | 'validation_auroc' | 'validation_accuracy';
  // Optional on older frozen plans; the executor applies the same defaults.
  embedDim?: number; attentionDim?: number; numFcLayers?: number; gatedAttention?: boolean;
  dropout?: number; inputDropout?: number; gradientCheckpointing?: boolean;
  precision?: '32-true' | '16-mixed' | 'bf16-mixed'; gradientClipNorm?: number;
  accumulateGradBatches?: number; lrScheduler?: 'none' | 'cosine' | 'plateau' | 'step'; warmupEpochs?: number;
  finalLrFraction?: number; earlyStoppingMinDelta?: number; minEpochs?: number;
  lossType?: 'ce' | 'bce' | 'focal'; classWeighting?: 'none' | 'inverse_prevalence';
  classWeights?: number[] | null; focalGamma?: number; labelSmoothing?: number;
  patientAggregation?: 'mean_probabilities' | 'mean_logits';
  ensembleAggregation?: 'mean_probability' | 'mean_logit';
  adamBetas?: [number, number]; adamEps?: number;
  aggregatorLearningRate?: number | null; headLearningRate?: number | null;
  lrStepSize?: number; lrGamma?: number; lrPlateauPatience?: number;
  samplingStrategy?: 'slide_uniform' | 'patient_natural' | 'cohort_balanced' | 'cohort_label_balanced';
  classWeightedSampling?: boolean; samplingPositivePrevalence?: number; cohortColumn?: string;
  instanceDropout?: number; featureNoiseStd?: number;
  bagCurriculum?: boolean; bagCurriculumStart?: number; bagCurriculumEnd?: number; bagCurriculumWarmupEpochs?: number;
  evalBagSize?: number | null; evalBatchSize?: number | null;
  minValidationPositives?: number | null; fixedEpochBudget?: number | null;
  analysis?: PatientAnalysisSettings | null; decisionThreshold?: number | null;
  bagSizeMode?: 'fixed' | 'training_median'; bagSizeFraction?: number;
  nnmilFeatureSampling?: boolean; nnmilWindowStrideDivisor?: number;
  nnmilWindowShuffle?: boolean; nnmilWindowSeed?: number;
  /** When on, each training seed draws its own feature-window order; the fixed seed is then unused. */
  nnmilWindowSeedFromTraining?: boolean;
  nnmilWindowAggregation?: 'mean_logits' | 'mean_probabilities';
  nnmilBatchSampler?: 'patient_weighted' | 'class_balanced' | 'auc_stratified';
  nnmilCheckpointSelection?: 'best_validation' | 'latest';
  weightDecayPolicy?: 'all' | 'weights_only'; lrScheduleInterval?: 'epoch' | 'step';
}
export interface ResourcePolicy {
  maxConcurrentRuns: number; gpuIds: number[]; runsPerGpu: number;
  cpuThreadsPerRun: number; dataLoaderWorkers: number; ramGbPerRun: number;
}
/** Metrics a controlled comparison can name as primary; loss is never used to compare. */
export type ComparisonMetric = 'auroc' | 'auprc' | 'balancedAccuracy' | 'macroF1' | 'accuracy';
/** Every configuration of the batch against one reference configuration, on shared folds and seeds. */
export interface ComparisonSpec {
  /** One-based configuration number of the reference arm. */
  reference: number; primaryMetric: ComparisonMetric;
}
export interface DevelopmentBatchSpec {
  version: 1; experimentId?: string; experimentRevision?: number; experimentName: string; batchName: string; inputs: MILExperimentSpec;
  recipe: TrainingRecipe; mode: 'single' | 'grid' | 'explicit';
  grid: { learningRates: number[]; weightDecays: number[]; maxEpochs: number[] };
  configurations: TrainingRecipe[]; trainingSeeds: number[]; notes: string;
  /** Saved only by batches planned before the Task Center; it now decides parallelism and devices. */
  resources?: ResourcePolicy;
  /** Omitted only on saved batches created before predictor choices belonged to each batch. */
  predictorPolicy?: ExperimentPredictorPolicy;
  selectionMetric?: TrainingRecipe['checkpointMetric'] | null;
  candidateSelection?: 'best_validation' | 'all' | null;
  /** Omitted unless the batch declares a controlled comparison (explicit mode, 2–8 configurations, all built). */
  comparison?: ComparisonSpec | null;
}
/** A frozen dataset column offered as a clinical input, with the reason it is refused, if any. */
export interface ClinicalFieldChoice {
  field: string; owner: 'slide' | 'patient'; type: string; kind: 'numeric' | 'categorical'; excluded: string | null;
}
export interface PlannedRun {
  id: string; candidateId: string; trainingSeed: number; splitPlanId: string; status: 'planned';
}
export interface NnMILPlanningRow {
  candidateId: string; splitPlanId: string; trainingSlideCount: number; trainingPatientCount: number;
  medianPatchCount: number; bagSize: number | null; featureDimension: number; windowCount: number;
  minPatchCount: number; maxPatchCount: number; paddedSlides: number; truncatedSlides: number;
  patchCountP05: number; patchCountP95: number; fingerprint: string;
  inputMemoryMiB?: number;
}
export interface BatchManifest {
  kind: 'mil-batch'; version: 1; datasetId: string; spec: DevelopmentBatchSpec;
  configurations: { id: string; number: number; recipe: TrainingRecipe }[];
  /** `domain`: the site a leave-one-site-out plan holds out; a held-out design's plan has no fold. */
  splitPlans: { id: string; planId: string; seed?: number; fold?: number | null; domain?: string; phase?: string; slideCount: number; partitions: Record<string, number> }[];
  runs: PlannedRun[];
  summary: { configurationCount: number; trainingSeedCount: number; splitPlanCount: number; runCount: number };
  executionImplemented: boolean; previewHash: string; resolvedInputs: MILExperimentPreview;
  nnmilPlanning?: NnMILPlanningRow[];
}
export interface BatchPreview extends BatchManifest { canFreeze: boolean; findings: Finding[] }
export interface FrozenBatch { id: string; createdAt: string; manifest: BatchManifest; versionLabel?: VersionLabel | null }
export type TrainingStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
export interface TrainingMetrics {
  available?: boolean; count?: number; reason?: string; loss?: number | null; accuracy?: number | null;
  balancedAccuracy?: number | null; macroF1?: number | null; auroc?: number | null; auprc?: number | null;
  classCounts?: Record<string, number>; missingClasses?: string[]; confusionMatrix?: number[][];
  confidenceIntervals?: Partial<Record<'auroc' | 'auprc', ConfidenceInterval>>;
}
export interface TrainingMetricDetails {
  unit: 'slide' | 'patient'; classOrder: string[]; positiveClass: string | null;
  patientAggregation: string; slide: TrainingMetrics; patient: TrainingMetrics; selected: TrainingMetrics;
  patientAnalysis?: PatientAnalysis;
}
export interface TrainingRun {
  id: string; candidateId: string; trainingSeed: number; splitPlanId: string; status: TrainingStatus;
  metrics?: { validation: TrainingMetricDetails; assessment: TrainingMetricDetails }; error?: string; checkpointPath?: string; outputPath?: string;
  progress?: { epoch: number; maxEpochs: number; globalStep: number; trainingLoss: number | null; validation: TrainingMetrics; learningRate?: number; cudaPeakAllocatedBytes?: number; cudaPeakReservedBytes?: number } | null;
  progressWarning?: string | null;
  result?: { epochsCompleted?: number | null; bestEpoch?: number | null; selectedEpoch?: number | null } | null;
}
export interface TrainingHistoryRow {
  /** One-based completed epoch, matching the live progress snapshot. */
  epoch: number; trainingLoss: number | null; validation: TrainingMetrics;
  learningRate: number | null; checkpointUnit: 'slide' | 'patient' | null;
}
export interface TrainingHistory {
  runId: string; rows: TrainingHistoryRow[]; totalRows: number; truncated: boolean; warning?: string;
  /** Derived from the complete validated history, even when chart rows are truncated. */
  stopping?: { enabled: boolean | null; patience: number | null; remaining: number | null; waitCount: number | null;
    minEpochs: number | null; epoch: number | null; status: 'tracking' | 'disabled' | 'unavailable'; reason?: string | null };
}
export interface TrainingResourceSample {
  at: string;
  host: Omit<TrainingHost, 'totalRamGb' | 'availableRamGb' | 'bootId' | 'kernel'> & {
    totalRamGb: number | null; availableRamGb: number | null; bootId?: string; kernel?: string;
  };
  gpus: TrainingGPU[]; gpuProbeError?: string;
  runs: { runId: string; pid: number; rssGb: number | null }[];
}
export interface TrainingResourceHistory {
  batchId: string; rows: TrainingResourceSample[]; totalRows: number; truncated: boolean; warning?: string;
  /** A bounded tail read may only know the number of rows in the inspected window. */
  totalRowsIsLowerBound?: boolean;
}
export interface TrainingExecution {
  batchId: string; status: TrainingStatus; sessionName: string; logPath: string; outputPath: string;
  cancelRequested?: boolean; findings: Finding[];
  runCounts: { total: number; queued: number; running: number; completed: number; failed: number; cancelled: number; interrupted?: number };
  runs: TrainingRun[]; createdAt: string; updatedAt: string;
  resourcePlan?: { requestedConcurrency: number; effectiveConcurrency: number; cpuSlotsPerRun: number; cpuLimit: number; ramLimit: number; gpuSlotLimit: number | null; note: string };
  computePath?: string; computeVersion?: string; provenancePath?: string;
  /** Batches launched through the Task Center have no tmux session of their own. */
  executor?: 'task-center' | 'tmux';
  taskCenter?: { runnerAlive: boolean; queued: number; running: number; held: boolean; waitingReason: string | null; ownerKey: string | null } | null;
  telemetry?: {
    path: string; intervalSeconds: number;
    latest: TrainingResourceSample;
    peak: { hostUsedRamGb: number; runRssGb: Record<string, number>; gpuUsedMemoryGb: Record<string, number> };
  };
}
export interface TrainingHost {
  cpuCount: number; totalRamGb: number; availableRamGb: number; bootId: string; kernel: string;
  /** Older workers did not record CPU utilization; missing is not zero. */
  cpuUtilizationPercent?: number | null;
}
export interface TrainingGPU { index: number; uuid: string; name: string; driverVersion: string; totalMemoryGb: number | null; usedMemoryGb: number | null; freeMemoryGb: number | null; utilizationPercent: number | null }
export interface TrainingRuntime {
  available: boolean; python: string; versions: Record<string, string | null>;
  cudaAvailable: boolean; gpuCount: number; findings: Finding[];
  host?: TrainingHost; gpus?: TrainingGPU[]; gpuProbeError?: string;
}
export interface CandidateResult {
  candidateId: string; trainingSeed: number; splitSeed: number; complete: boolean;
  metrics: TrainingMetrics | null; oofPath?: string | null;
  completedRuns: number; totalRuns: number; assessmentSlideCount?: number; metricDetails?: TrainingMetricDetails;
  selectionScore?: number | null; selected?: boolean;
}
export interface DevelopmentResults {
  batchId?: string; status: TrainingStatus | 'not-started' | 'planned'; candidates: CandidateResult[];
  oof: { path: string; candidateId: string; trainingSeed: number; splitSeed: number; slideCount: number }[];
  selectionNote?: string; findings?: Finding[];
  selection?: { metric: TrainingRecipe['checkpointMetric']; unit: string; ready: boolean; selectedCandidateId: string | null; note: string };
}
export interface DevelopmentBatchList { items: FrozenBatch[]; executionImplemented: boolean; executions?: TrainingExecution[] }
const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/mil-experiments/batches`;
const batchPrefix = (project: string, batch: string) => `${prefix(project)}/${encodeURIComponent(batch)}`;
const body = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const development = {
  list: (project: string) => request<DevelopmentBatchList>(prefix(project)),
  runtime: (project: string) => request<TrainingRuntime>(`/projects/${encodeURIComponent(project)}/mil-experiments/runtime`),
  execution: (project: string, batch: string) => request<TrainingExecution | null>(`${batchPrefix(project, batch)}/execution`),
  history: (project: string, batch: string, run: string) => request<TrainingHistory>(`${batchPrefix(project, batch)}/runs/${encodeURIComponent(run)}/history`),
  resourceHistory: (project: string, batch: string, signal?: AbortSignal) => request<TrainingResourceHistory>(`${batchPrefix(project, batch)}/resources/history`, { signal }),
  launch: (project: string, batch: string, operationId: string) => request<TrainingExecution>(`${batchPrefix(project, batch)}/launch`, body({ operationId })),
  cancel: (project: string, batch: string, operationId: string) => request<TrainingExecution>(`${batchPrefix(project, batch)}/cancel`, body({ operationId })),
  resume: (project: string, batch: string, operationId: string) => request<TrainingExecution>(`${batchPrefix(project, batch)}/resume`, body({ operationId })),
  results: (project: string, batch: string) => request<DevelopmentResults>(`${batchPrefix(project, batch)}/results`),
  downloadOOF: (project: string, batch: string, candidate: string, trainingSeed: number, splitSeed: number, unit: 'slide' | 'patient') => downloadArtifact(
    `${batchPrefix(project, batch)}/oof/${encodeURIComponent(candidate)}/${trainingSeed}/${splitSeed}/${unit}.csv`,
    `oof-${candidate}-${trainingSeed}-${splitSeed}-${unit}.csv`,
  ),
  preview: (project: string, spec: DevelopmentBatchSpec) => request<BatchPreview>(`${prefix(project)}/preview`, body(spec)),
  clinicalFields: (project: string, protocolId: string) => request<{ fields: ClinicalFieldChoice[] }>(
    `/projects/${encodeURIComponent(project)}/mil-experiments/clinical-fields?protocolId=${encodeURIComponent(protocolId)}`),
};

// Starting recipes are server-owned: see lib/templates.ts.
export const defaultRecipe = (): TrainingRecipe => fromTemplate(templates.recipes.default);
export const experimentalRecipeDefaults = templates.recipes.experimentalDefaults as Partial<TrainingRecipe>;
// Older saved recipes can omit values that were defaults when they were created.
export const withRecipeDefaults = (recipe: TrainingRecipe): TrainingRecipe => ({ ...experimentalRecipeDefaults, ...defaultRecipe(), ...recipe,
  analysis: recipe.analysis ?? null, decisionThreshold: recipe.decisionThreshold ?? null, checkpointMetric: recipe.checkpointMetric ?? 'validation_loss',
  learningRate: recipe.learningRate === undefined ? 0.0003 : recipe.learningRate,
  weightDecay: recipe.weightDecay === undefined ? 0.0001 : recipe.weightDecay,
  maxEpochs: recipe.maxEpochs === undefined ? 100 : recipe.maxEpochs, patience: recipe.patience === undefined ? 15 : recipe.patience });
export const oceanPathRecipe = (preset: 'standard' | 'kras'): TrainingRecipe => recipePreset(preset === 'standard' ? 'oceanpath' : 'oceanpath-kras');
/** Editable nnMIL method template with HistoPilot's patient protocol and optimizer defaults.
 * New recipes give each training seed its own feature-window order, so the seed spread
 * includes that variation; saved recipes keep their own setting. */
export const nnmilRecipe = (): TrainingRecipe => recipePreset('nnmil');
export const defaultResources = (): ResourcePolicy => fromTemplate(templates.resources.training);
export const managedByTaskCenter = (execution?: Pick<TrainingExecution, 'executor' | 'taskCenter'> | null) => execution?.executor === 'task-center' || Boolean(execution?.taskCenter);
export const trainingActive = (execution?: TrainingExecution | null) => execution?.status === 'queued' || execution?.status === 'running';
export const developmentPollInterval = (data?: DevelopmentBatchList) => data?.executionImplemented ? data.executions?.some(trainingActive) ? 3000 : 15000 : false;
/** A project-level refresh can discover a CLI launch before the selected-batch query. */
export function latestExecution(selected?: TrainingExecution | null, listed?: TrainingExecution) {
  if (!selected) return listed ?? selected;
  if (!listed) return selected;
  return Date.parse(listed.updatedAt) > Date.parse(selected.updatedAt) ? listed : selected;
}

export function parseNumberList(value: string, label: string, integer = false, minimum = 0): number[] {
  const entries = value.split(',').map((item) => item.trim());
  if (!entries.length || entries.some((item) => !item)) throw new Error(`${label}: enter comma-separated numbers without empty values.`);
  const numbers = entries.map(Number);
  if (numbers.some((n) => !Number.isFinite(n) || n < minimum || (integer && !Number.isSafeInteger(n)))) throw new Error(`${label}: enter ${integer ? 'whole ' : ''}numbers of at least ${minimum}.`);
  if (new Set(numbers).size !== numbers.length) throw new Error(`${label}: values must be distinct.`);
  return numbers;
}

/** A split plan as people read it: the site it holds out, its fold, or the held-out assessment. */
export function splitPlanLabel(split: { fold?: number | null; domain?: string }) {
  if (split.domain !== undefined) return `Held-out ${split.domain}`;
  return typeof split.fold === 'number' ? `Fold ${split.fold + 1}` : 'Held-out assessment';
}
