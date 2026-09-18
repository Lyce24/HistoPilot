import { downloadArtifact, request } from './client';
import type { MILExperimentSpec, MILExperimentPreview } from './mil';
import type { Finding, VersionLabel } from './scientific';
import type { ExperimentPredictorPolicy } from './experiments';
import { defaultPatientAnalysis, type PatientAnalysisSettings, type PatientAnalysis, type ConfidenceInterval } from './statistics';

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
  nnmilWindowAggregation?: 'mean_logits' | 'mean_probabilities';
  nnmilBatchSampler?: 'patient_weighted' | 'class_balanced' | 'auc_stratified';
  nnmilCheckpointSelection?: 'best_validation' | 'latest';
  weightDecayPolicy?: 'all' | 'weights_only'; lrScheduleInterval?: 'epoch' | 'step';
}
export interface ResourcePolicy {
  maxConcurrentRuns: number; gpuIds: number[]; runsPerGpu: number;
  cpuThreadsPerRun: number; dataLoaderWorkers: number; ramGbPerRun: number;
}
export interface RuntimeRecommendation {
  version: 1; generatedAt: string; applicable: boolean;
  basis: 'measured' | 'estimated' | 'hardware_only' | 'unavailable';
  resources: ResourcePolicy | null; summary: string;
  memory: { perRunGpuGb: number | null; perRunRamGb: number | null; observedRuns: number };
  limits: { cpuConcurrency: number; ramConcurrency: number; gpuConcurrency: number | null; additionalRunsNow: number | null };
  evidence: { key: string; gpuUuid: string; trainingPatches: number; evaluationPatches: number; peakReservedGpuGb: number; stage: 'fit' | 'assessment'; epoch: number; batchId: string; runId: string }[];
  findings: Finding[];
  hardware?: { cpuCount: number; totalRamGb: number; availableRamGb: number; gpus: TrainingGPU[] };
}
export interface DevelopmentBatchSpec {
  version: 1; experimentId?: string; experimentRevision?: number; experimentName: string; batchName: string; inputs: MILExperimentSpec;
  recipe: TrainingRecipe; mode: 'single' | 'grid' | 'explicit';
  grid: { learningRates: number[]; weightDecays: number[]; maxEpochs: number[] };
  configurations: TrainingRecipe[]; trainingSeeds: number[]; resources: ResourcePolicy; notes: string;
  /** Omitted only on saved batches created before predictor choices belonged to each batch. */
  predictorPolicy?: ExperimentPredictorPolicy;
  selectionMetric?: TrainingRecipe['checkpointMetric'] | null;
  candidateSelection?: 'best_validation' | 'all' | null;
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
  splitPlans: { id: string; planId: string; seed?: number; fold?: number; phase?: string; slideCount: number; partitions: Record<string, number> }[];
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
  runtimeRecommendation: (project: string, spec: DevelopmentBatchSpec, signal?: AbortSignal) => request<RuntimeRecommendation>(`${prefix(project)}/runtime-recommendation`, { ...body(spec), signal }),
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
  freeze: (project: string, spec: DevelopmentBatchSpec, previewHash: string, operationId: string, versionLabel: { tag: string; note: string }) =>
    request<FrozenBatch>(`${prefix(project)}/freeze`, body({ spec, previewHash, operationId, versionLabel })),
};

export const defaultRecipe = (): TrainingRecipe => ({ model: 'abmil', learningRate: 0.0003, weightDecay: 0.0001, maxEpochs: 40, optimizer: 'adamw', batchSize: 1, bagSize: 4096, earlyStopping: true, patience: 8, checkpointMetric: 'validation_auroc', analysis: defaultPatientAnalysis(), decisionThreshold: 0.5, embedDim: 512, attentionDim: 384, numFcLayers: 1, gatedAttention: true, dropout: 0.25, inputDropout: 0, gradientCheckpointing: false, precision: '32-true', gradientClipNorm: 0, accumulateGradBatches: 1, lrScheduler: 'none', warmupEpochs: 0, finalLrFraction: 0.01, earlyStoppingMinDelta: 0, minEpochs: 1 });
export const experimentalRecipeDefaults = {
  lossType: 'ce', classWeighting: 'none', classWeights: null, focalGamma: 2, labelSmoothing: 0,
  patientAggregation: 'mean_probabilities', ensembleAggregation: 'mean_probability', adamBetas: [0.9, 0.999] as [number, number], adamEps: 1e-8,
  aggregatorLearningRate: null, headLearningRate: null, lrStepSize: 10, lrGamma: 0.5, lrPlateauPatience: 5,
  samplingStrategy: 'slide_uniform', classWeightedSampling: false, samplingPositivePrevalence: 0.4,
  cohortColumn: 'cohort', instanceDropout: 0, featureNoiseStd: 0, bagCurriculum: false,
  bagCurriculumStart: 512, bagCurriculumEnd: 8000, bagCurriculumWarmupEpochs: 5,
  evalBagSize: null, evalBatchSize: null, minValidationPositives: null, fixedEpochBudget: null,
  bagSizeMode: 'fixed', bagSizeFraction: 0.5,
  nnmilFeatureSampling: true, nnmilWindowStrideDivisor: 4, nnmilWindowShuffle: true, nnmilWindowSeed: 42,
  nnmilWindowAggregation: 'mean_logits', nnmilBatchSampler: 'patient_weighted', nnmilCheckpointSelection: 'best_validation',
  weightDecayPolicy: 'all', lrScheduleInterval: 'epoch',
} satisfies Partial<TrainingRecipe>;
// Older saved recipes can omit values that were defaults when they were created.
export const withRecipeDefaults = (recipe: TrainingRecipe): TrainingRecipe => ({ ...experimentalRecipeDefaults, ...defaultRecipe(), ...recipe,
  analysis: recipe.analysis ?? null, decisionThreshold: recipe.decisionThreshold ?? null, checkpointMetric: recipe.checkpointMetric ?? 'validation_loss',
  learningRate: recipe.learningRate === undefined ? 0.0003 : recipe.learningRate,
  weightDecay: recipe.weightDecay === undefined ? 0.0001 : recipe.weightDecay,
  maxEpochs: recipe.maxEpochs === undefined ? 100 : recipe.maxEpochs, patience: recipe.patience === undefined ? 15 : recipe.patience });
export const oceanPathRecipe = (preset: 'standard' | 'kras'): TrainingRecipe => ({
  ...defaultRecipe(), ...(preset === 'kras' ? { weightDecay: 0.01 } : {}), maxEpochs: preset === 'standard' ? 20 : 40, minEpochs: preset === 'standard' ? 10 : 0,
  patience: preset === 'standard' ? 5 : 8, lrScheduler: 'cosine', finalLrFraction: preset === 'standard' ? 0.01 : 0.001,
  gradientClipNorm: 1, checkpointMetric: 'validation_auroc', patientAggregation: 'mean_probabilities', ensembleAggregation: 'mean_probability',
  bagSize: preset === 'standard' ? null : 4096,
  ...(preset === 'kras' ? { lossType: 'bce', classWeighting: 'none' } : {}),
});
/** Editable nnMIL method template with HistoPilot's patient protocol and optimizer defaults. */
export const nnmilRecipe = (): TrainingRecipe => ({
  ...defaultRecipe(), ...experimentalRecipeDefaults, model: 'nnmil', attentionDim: 256,
  dropout: 0.25, batchSize: 32, bagSize: null, bagSizeMode: 'training_median', bagSizeFraction: 0.5,
  optimizer: 'adamw', lrScheduler: 'cosine', warmupEpochs: 5, maxEpochs: 100, patience: 10, minEpochs: 1,
  evalBagSize: null, evalBatchSize: 1, checkpointMetric: 'validation_auroc',
});
export const defaultResources = (): ResourcePolicy => ({ maxConcurrentRuns: 1, gpuIds: [0], runsPerGpu: 1, cpuThreadsPerRun: 2, dataLoaderWorkers: 2, ramGbPerRun: 8 });
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
