import { request } from './client';
import type { BatchManifest, DevelopmentBatchSpec, FrozenBatch, TrainingExecution } from './development';
import type { LifecycleState } from './lifecycle';
import type { MILExperimentSpec } from './mil';
import type { Configuration, ProtocolSpec, ScientificDraft } from './scientific';
import type { ComputeExecution, PredictorManifest, PredictorMethod } from './predictors';

export interface ExperimentBatch extends FrozenBatch {
  key: string; name: string; state: LifecycleState; status: string;
  inputSnapshot?: Record<string, unknown>; execution?: TrainingExecution | null;
}
export type ExperimentStage = 'planning' | 'running' | 'finished';
export interface ExperimentPredictorPolicy {
  method: 'skip' | 'refit' | 'ensemble' | 'both';
  refitPercentile: number | null;
}
export interface ExperimentPredictorItem {
  key: string;
  source: { experimentId: string; batchId: string; candidateId: string; trainingSeed: number; splitSeed: number };
  method: PredictorMethod; configurationNumber: number; foldCount: number; runIds: string[];
  refitPercentile?: number | null;
  status: 'waiting' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'skipped';
  recordId: string | null; predictorId: string | null;
  epochBudget: PredictorManifest['epochBudget'] | null; execution: ComputeExecution | null;
  error: { code: string; message: string } | null;
}
export interface ExperimentPredictorExecution {
  status: 'queued' | 'waiting' | 'running' | 'cancelling' | 'completed' | 'cancelled' | 'attention' | 'interrupted';
  counts: { total: number; ensemble: number; refit: number; completed: number; waiting: number; active: number; failed: number; cancelled: number; skipped?: number };
  items?: ExperimentPredictorItem[];
  error: { code: string; message: string } | null; updatedAt: string | null; sessionName: string | null; logPath: string | null;
  retryable: boolean; cancellable: boolean;
  /** "task-center" for Task Center coordinators; "tmux" (or absent) for read-only coordinators
   * created before it, whose stored `sessionName` is never shown. */
  executor?: 'task-center' | 'tmux';
  waitingReason?: string | null;
  /** Task Center coordinators only; null when the service cannot tell. */
  runnerAlive?: boolean | null;
}
export interface ExperimentBatchPlan { id: string; spec: DevelopmentBatchSpec }
export interface ExperimentSubmission {
  operationId: string; expectedRevision: number; submittedAt: string;
  status: 'launching' | 'submitted' | 'attention'; batchIds: string[];
  error: { code: string; message: string } | null; retryable: boolean;
}
export interface SetupDesign {
  splitUnit?: 'slide' | 'patient';
  datasetId: string;
  targetSplitId: string;
  trainingSplit: ProtocolSpec['split'];
}
export interface SetupInput extends SetupDesign {
  expectedRevision: number;
  featureBundleId: string;
  loadingPolicy: MILExperimentSpec['loadingPolicy'];
  packArtifactId: string | null;
}
/** One execution status for a submitted experiment, shared by the library, the detail header
 * and the Task Center's planned list. Live work first; `needs-attention` covers failed or interrupted runs and a
 * coordinator that stopped, and `waiting` work that resumes by itself (a busy workspace, earlier
 * batches). */
export type ExperimentExecutionStatus = 'queued' | 'running' | 'waiting' | 'held' | 'needs-attention' | 'cancelled' | 'completed';
/** Before submission a record is `created`, `planned` or `ready` (a frozen design). */
export type ExperimentStatus = 'created' | 'planned' | 'ready' | ExperimentExecutionStatus;
export interface ModelExperiment {
  setupVersion?: 1 | null;
  setupDesign?: SetupDesign | null;
  setupStatus?: 'draft' | 'frozen';
  frozenSetupId?: string | null;
  frozenSetup?: Configuration | null;
  id: string; key: string; name: string; notes: string; tags: string[];
  /** An {@link ExperimentStatus}; older services also sent failed, interrupted, unknown, partial,
   * scheduled or cancelling, which {@link experimentExecutionStatus} maps. */
  revision: number; state: LifecycleState; status: string; legacy: boolean;
  /** Why the experiment waits or needs attention, when the service knows one reason. */
  statusReason?: string | null;
  createdAt: string; updatedAt: string; inputs: MILExperimentSpec | null; executionImplemented?: boolean;
  batches: ExperimentBatch[]; drafts: ScientificDraft[]; predictorId: string | null; inputSnapshot?: Record<string, unknown> | null;
  predictorIds?: Partial<Record<'ensemble' | 'refit', string>>;
  predictors?: { id: string; method: 'ensemble' | 'refit'; batchId: string; candidateId: string; trainingSeed: number; splitSeed: number; lifecycleState: LifecycleState }[];
  stage?: ExperimentStage; configurationLocked?: boolean;
  batchPlans?: ExperimentBatchPlan[]; submission?: ExperimentSubmission | null;
  predictorPolicy?: ExperimentPredictorPolicy | null;
  predictorPolicies?: Record<string, ExperimentPredictorPolicy> | null;
  predictorExecution?: ExperimentPredictorExecution | null;
}
export interface ExperimentBatchSummary extends Omit<ExperimentBatch, 'manifest' | 'inputSnapshot' | 'execution'> {
  manifest: Pick<BatchManifest, 'kind' | 'version' | 'summary'> & {
    spec: Pick<DevelopmentBatchSpec, 'experimentId' | 'experimentRevision' | 'experimentName' | 'batchName' | 'inputs' | 'predictorPolicy'>;
  };
}
export interface ModelExperimentSummary extends Omit<ModelExperiment, 'batches' | 'drafts' | 'inputSnapshot'> {
  summary?: boolean; inputSnapshot?: null; batches: ExperimentBatchSummary[];
  drafts: Omit<ScientificDraft, 'payload'>[];
}
export interface ExperimentInput {
  name: string; notes?: string; tags?: string[]; inputs?: MILExperimentSpec | null;
  predictorPolicy?: ExperimentPredictorPolicy;
}
/** A new experiment states its setup version: 1 for an experiment design, null for the legacy form. */
export interface CreateExperimentInput extends ExperimentInput { setupVersion: 1 | null; sourceExperimentId?: string; operationId: string }
export interface UpdateExperimentInput extends ExperimentInput { expectedRevision: number; batchPlans?: ExperimentBatchPlan[] }
const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/model-experiments`;
export const experiments = {
  summaries: (project: string) => request<{ items: ModelExperimentSummary[] }>(`${prefix(project)}?summary=true`),
  get: (project: string, id: string) => request<ModelExperiment>(`${prefix(project)}/${encodeURIComponent(id)}`),
  create: (project: string, input: CreateExperimentInput) =>
    request<ModelExperiment>(prefix(project), { method: 'POST', body: JSON.stringify(input) }),
  update: (project: string, id: string, input: UpdateExperimentInput) =>
    request<ModelExperiment>(`${prefix(project)}/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(input) }),
  setupInputs: (project: string, id: string, input: SetupInput) =>
    request<ModelExperiment>(`${prefix(project)}/${encodeURIComponent(id)}/setup-inputs`, { method: 'POST', body: JSON.stringify(input) }),
  freezeSetup: (project: string, id: string, input: { expectedRevision: number; operationId: string }) =>
    request<ModelExperiment>(`${prefix(project)}/${encodeURIComponent(id)}/freeze-setup`, { method: 'POST', body: JSON.stringify(input) }),
  submit: (project: string, id: string, input: { expectedRevision: number; operationId: string }) =>
    request<ModelExperiment>(`${prefix(project)}/${encodeURIComponent(id)}/submit`, { method: 'POST', body: JSON.stringify(input) }),
  predictorAction: (project: string, id: string, action: 'resume' | 'cancel', operationId: string) =>
    request<ExperimentPredictorExecution>(`${prefix(project)}/${encodeURIComponent(id)}/predictors/${action}`, { method: 'POST', body: JSON.stringify({ operationId }) }),
};
/** Prefer the server's lifecycle; derive a conservative state for older saved responses. */
export function experimentStage(item: Pick<ModelExperiment, 'stage' | 'status' | 'configurationLocked'>): ExperimentStage {
  if (item.stage) return item.stage;
  if (['completed', 'cancelled'].includes(item.status)) return 'finished';
  if (item.configurationLocked || ['running', 'queued', 'scheduled', 'cancelling', 'failed', 'interrupted', 'waiting', 'held', 'needs-attention'].includes(item.status)) return 'running';
  return 'planning';
}
export const experimentStageLabel: Record<ExperimentStage, string> = { planning: 'Planning', running: 'Running', finished: 'Finished' };
/** Older services' statuses in the current vocabulary; planning statuses pass through. */
const legacyStatuses: Record<string, ExperimentExecutionStatus> = {
  scheduled: 'queued', launching: 'queued', cancelling: 'running',
  failed: 'needs-attention', interrupted: 'needs-attention', unknown: 'needs-attention', attention: 'needs-attention',
  partial: 'cancelled',
};
export const experimentExecutionStatus = (status: string): string => legacyStatuses[status] ?? status;
/** The Task Center's words for queued work its runner cannot start (see `task_execution`). */
export const runnerStoppedReason = (reason?: string | null) => Boolean(reason && /runner is stopped/i.test(reason));
/**
 * How often a list of experiments is re-read: 3 s while work runs or waits its turn, 5 s while
 * held (it moves once released in the Task Center), 20 s while it waits for a stopped runner,
 * 15 s otherwise. The fastest item decides.
 */
export function experimentPollInterval(data?: { items: { status: string; statusReason?: string | null }[] }) {
  const intervals = (data?.items ?? []).map((item) => {
    const status = experimentExecutionStatus(item.status);
    if (status === 'waiting' && runnerStoppedReason(item.statusReason)) return 20000;
    if (['running', 'queued', 'waiting'].includes(status)) return 3000;
    return status === 'held' ? 5000 : 15000;
  });
  return intervals.length ? Math.min(...intervals) : 15000;
}
const statusLabels: Record<ExperimentStatus, string> = {
  created: 'Created', planned: 'Planned', ready: 'Ready to run',
  queued: 'Queued', running: 'Running', waiting: 'Waiting', held: 'Held',
  'needs-attention': 'Needs attention', cancelled: 'Cancelled', completed: 'Completed',
};
export const experimentStatusLabel = (status: string) => {
  const current = experimentExecutionStatus(status);
  return statusLabels[current as ExperimentStatus] ?? current.replace(/^./, (letter) => letter.toUpperCase());
};
/** Execution tones follow the Task Center: active work in the product colour, waiting work
 * neutral, and only what needs a person asks for attention. */
const statusTones: Partial<Record<ExperimentStatus, string>> = {
  running: 'green', held: 'amber', 'needs-attention': 'orange', completed: 'success',
};
export const experimentStatusTone = (status: string) => statusTones[experimentExecutionStatus(status) as ExperimentStatus] ?? 'neutral';
/**
 * The service's reason for an experiment that is not simply running or finished: why it is
 * queued, waiting or held, or what needs attention. Planning statuses carry none.
 */
export function experimentStatusReason(item: Pick<ModelExperiment, 'status' | 'statusReason'>): string | null {
  const status = experimentExecutionStatus(item.status);
  return item.statusReason && ['queued', 'waiting', 'held', 'needs-attention'].includes(status) ? item.statusReason : null;
}
/** Filter choices on the Experiments page, in the order they matter. */
export const experimentStatusFilters: Record<string, string> = Object.fromEntries(
  (['ready', 'queued', 'running', 'waiting', 'held', 'needs-attention', 'cancelled', 'completed'] as const).map((status) => [status, statusLabels[status]]),
);
