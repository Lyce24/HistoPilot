import { request } from './client';
import type { MILExperimentSpec, MILExperimentPreview } from './mil';
import type { Finding, VersionLabel } from './scientific';
import type { ExperimentPredictorPolicy } from './experiments';

export interface TrainingRecipe {
  model: string; learningRate: number; weightDecay: number; maxEpochs: number;
  optimizer: 'adam' | 'adamw' | 'sgd'; batchSize: number; bagSize: number | null;
  earlyStopping: boolean; patience: number;
  checkpointMetric: 'validation_loss' | 'validation_auroc' | 'validation_accuracy';
  // Optional on older frozen plans; the executor applies the same defaults.
  embedDim?: number; attentionDim?: number; numFcLayers?: number; gatedAttention?: boolean;
  dropout?: number; inputDropout?: number; gradientCheckpointing?: boolean;
  precision?: '32-true' | '16-mixed' | 'bf16-mixed'; gradientClipNorm?: number;
  accumulateGradBatches?: number; lrScheduler?: 'none' | 'cosine'; warmupEpochs?: number;
  finalLrFraction?: number; earlyStoppingMinDelta?: number; minEpochs?: number;
}
export interface ResourcePolicy {
  maxConcurrentRuns: number; gpuIds: number[]; runsPerGpu: number;
  cpuThreadsPerRun: number; dataLoaderWorkers: number; ramGbPerRun: number;
}
export interface DevelopmentBatchSpec {
  version: 1; experimentId?: string; experimentRevision?: number; experimentName: string; batchName: string; inputs: MILExperimentSpec;
  recipe: TrainingRecipe; mode: 'single' | 'grid' | 'explicit';
  grid: { learningRates: number[]; weightDecays: number[]; maxEpochs: number[] };
  configurations: TrainingRecipe[]; trainingSeeds: number[]; resources: ResourcePolicy; notes: string;
  /** Omitted only on saved batches created before predictor choices belonged to each batch. */
  predictorPolicy?: ExperimentPredictorPolicy;
}
export interface PlannedRun {
  id: string; candidateId: string; trainingSeed: number; splitPlanId: string; status: 'planned';
}
export interface BatchManifest {
  kind: 'mil-batch'; version: 1; datasetId: string; spec: DevelopmentBatchSpec;
  configurations: { id: string; number: number; recipe: TrainingRecipe }[];
  splitPlans: { id: string; planId: string; seed?: number; fold?: number; phase?: string; slideCount: number; partitions: Record<string, number> }[];
  runs: PlannedRun[];
  summary: { configurationCount: number; trainingSeedCount: number; splitPlanCount: number; runCount: number };
  executionImplemented: boolean; previewHash: string; resolvedInputs: MILExperimentPreview;
}
export interface BatchPreview extends BatchManifest { canFreeze: boolean; findings: Finding[] }
export interface FrozenBatch { id: string; createdAt: string; manifest: BatchManifest; versionLabel?: VersionLabel | null }
export type TrainingStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
export interface TrainingMetrics {
  available?: boolean; count?: number; reason?: string; loss?: number | null; accuracy?: number | null;
  balancedAccuracy?: number | null; macroF1?: number | null; auroc?: number | null; auprc?: number | null;
  classCounts?: Record<string, number>; missingClasses?: string[]; confusionMatrix?: number[][];
}
export interface TrainingMetricDetails {
  unit: 'slide' | 'patient'; classOrder: string[]; positiveClass: string | null;
  patientAggregation: string; slide: TrainingMetrics; patient: TrainingMetrics; selected: TrainingMetrics;
}
export interface TrainingRun {
  id: string; candidateId: string; trainingSeed: number; splitPlanId: string; status: TrainingStatus;
  metrics?: { validation: TrainingMetricDetails; assessment: TrainingMetricDetails }; error?: string; checkpointPath?: string; outputPath?: string;
  progress?: { epoch: number; maxEpochs: number; globalStep: number; trainingLoss: number | null; validation: TrainingMetrics; learningRate?: number; cudaPeakAllocatedBytes?: number; cudaPeakReservedBytes?: number } | null;
  progressWarning?: string | null;
}
export interface TrainingHistoryRow {
  /** One-based completed epoch, matching the live progress snapshot. */
  epoch: number; trainingLoss: number | null; validation: TrainingMetrics;
  learningRate: number | null; checkpointUnit: 'slide' | 'patient' | null;
}
export interface TrainingHistory {
  runId: string; rows: TrainingHistoryRow[]; totalRows: number; truncated: boolean; warning?: string;
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
    latest: { at: string; host: TrainingHost; gpus: TrainingGPU[]; gpuProbeError?: string; runs: { runId: string; pid: number; rssGb: number }[] };
    peak: { hostUsedRamGb: number; runRssGb: Record<string, number>; gpuUsedMemoryGb: Record<string, number> };
  };
}
export interface TrainingHost { cpuCount: number; totalRamGb: number; availableRamGb: number; bootId: string; kernel: string }
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
}
export interface DevelopmentResults {
  batchId?: string; status: TrainingStatus | 'not-started' | 'planned'; candidates: CandidateResult[];
  oof: { path: string; candidateId: string; trainingSeed: number; splitSeed: number; slideCount: number }[];
  selectionNote?: string; findings?: Finding[];
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
  launch: (project: string, batch: string, operationId: string) => request<TrainingExecution>(`${batchPrefix(project, batch)}/launch`, body({ operationId })),
  cancel: (project: string, batch: string, operationId: string) => request<TrainingExecution>(`${batchPrefix(project, batch)}/cancel`, body({ operationId })),
  resume: (project: string, batch: string, operationId: string) => request<TrainingExecution>(`${batchPrefix(project, batch)}/resume`, body({ operationId })),
  results: (project: string, batch: string) => request<DevelopmentResults>(`${batchPrefix(project, batch)}/results`),
  preview: (project: string, spec: DevelopmentBatchSpec) => request<BatchPreview>(`${prefix(project)}/preview`, body(spec)),
  freeze: (project: string, spec: DevelopmentBatchSpec, previewHash: string, operationId: string, versionLabel: { tag: string; note: string }) =>
    request<FrozenBatch>(`${prefix(project)}/freeze`, body({ spec, previewHash, operationId, versionLabel })),
};

export const defaultRecipe = (): TrainingRecipe => ({ model: 'abmil', learningRate: 0.0003, weightDecay: 0.0001, maxEpochs: 40, optimizer: 'adamw', batchSize: 1, bagSize: 4096, earlyStopping: true, patience: 8, checkpointMetric: 'validation_loss', embedDim: 512, attentionDim: 384, numFcLayers: 1, gatedAttention: true, dropout: 0.25, inputDropout: 0, gradientCheckpointing: false, precision: '32-true', gradientClipNorm: 0, accumulateGradBatches: 1, lrScheduler: 'none', warmupEpochs: 0, finalLrFraction: 0.01, earlyStoppingMinDelta: 0, minEpochs: 1 });
// Older saved recipes can omit values that were defaults when they were created.
export const withRecipeDefaults = (recipe: TrainingRecipe): TrainingRecipe => ({ ...defaultRecipe(), ...recipe, maxEpochs: recipe.maxEpochs === undefined ? 100 : recipe.maxEpochs, patience: recipe.patience === undefined ? 15 : recipe.patience });
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
