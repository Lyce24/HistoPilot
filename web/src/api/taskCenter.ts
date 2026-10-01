import { downloadArtifact, fetchArtifactBlob, request } from './client';

/** Machine-wide task queue served by the Task Center runner. These states exist
 * only in the task store; record status files keep their own vocabularies. */
export type TaskState = 'blocked' | 'queued' | 'starting' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';
export type TaskLane = 'gpu' | 'cpu';
export type OwnerAction = 'hold' | 'release' | 'stop' | 'cancel' | 'retry' | 'move';
export type MovePosition = 'top' | 'up' | 'down' | 'bottom';
export type TaskStateCounts = Partial<Record<TaskState, number>>;

export interface TaskCenterRunner {
  alive: boolean; heartbeatAt: string | null; pid: number | null; state: string | null; message: string | null;
  codeHash: string | null; codeCurrent: boolean; autostart: boolean;
  /** Set when the runner was started from a different checkout than this service. */
  otherCheckout?: string | null; checkoutRoot?: string | null;
}
/**
 * What a start or restart did. `started: false` carries a `reason` (for example
 * "tmux unavailable"); `pending` means the old runner is still finishing a step and
 * the service starts the new one when it exits.
 */
export interface RunnerStartResult {
  started: boolean; alive?: boolean; reason?: string; note?: string; pending?: boolean; session?: string;
  stopped?: Record<string, unknown>;
}
export interface RunnerActionResponse { runner: TaskCenterRunner; result: RunnerStartResult | null }
export interface TaskCenterGpu {
  index: number; name: string; slots: number; usedSlots: number; foreignSlots: number;
  totalMemoryGb: number | null; usedMemoryGb: number | null; committedMemoryGb: number; utilizationPercent: number | null;
}
export interface TaskCenterSummary {
  runner: TaskCenterRunner;
  paused: boolean;
  capacity: {
    gpus: TaskCenterGpu[];
    cpu: { logical: number; physical: number | null; committedThreads: number; reserveThreads: number; cpuTaskSlots: number; usedCpuTasks: number };
    ram: { totalGb: number; availableGb: number; reserveGb: number; committedGb?: number };
    /** The GPU probe failed: GPU capacity is unknown, not idle. */
    gpuProbeError?: string;
  };
  counts: TaskStateCounts;
  /** Failed or interrupted tasks that ended in the last 24 hours. */
  recentFailures?: number;
  eta: { seconds: number; basis: 'measured' | 'estimated' } | null;
  foreignLeases: ForeignLease[];
  /** The legacy lease registry could not be read: foreign load may be missing, not zero. */
  leaseError?: string | null;
  workspace: string; updatedAt: string;
}
export interface ForeignLease { kind: string | null; gpu: number | null; cpus: number; ramGb: number; runsPerGpu?: number | null; batchId?: string; runId?: string }
export interface TaskOwnerRef {
  key: string; kind: string; id: string; title: string; projectId: string; projectFolder: string;
  /** This workspace's name for the project; null for another workspace's project. */
  projectName?: string | null;
  sameWorkspace: boolean; held: boolean; queueSeq: number;
}
export interface TaskProgress {
  epoch?: number; maxEpochs?: number; completed?: number; total?: number;
  completedModels?: number; totalModels?: number; completedPairs?: number; totalPairs?: number;
  completedSlides?: number; totalSlides?: number; updatedAt?: string;
  trainingLoss?: number | null; learningRate?: number | null; phase?: string; message?: string; label?: string; unit?: string;
  validation?: { loss?: number | null; auroc?: number | null; accuracy?: number | null; available?: boolean } | null;
  cudaPeakReservedBytes?: number; cudaPeakAllocatedBytes?: number;
  [key: string]: unknown;
}
/** Why an unsuccessful task stopped, in plain words, and whether retrying it is safe. */
export interface TaskFailure {
  title: string; cause: string; advice: string | null; detail: string;
  retry: 'safe' | 'check' | 'after-fix' | 'unknown';
}
export interface TaskExit {
  reason: string | null; returncode: number | null; error: string | null;
  stopReason?: 'cancel' | 'pause' | null; killed?: boolean; lost?: boolean; signalled?: boolean;
}
export interface TaskResources {
  privateRamGb?: number | null; peakPrivateRamGb?: number | null; cpuCores?: number | null; meanCpuCores?: number | null;
  meanConcurrency?: number | null; gpuName?: string | null; vramGb?: number | null; peakVramGb?: number | null; sampledAt?: string | null;
}
export interface TaskItem {
  id: string; kind: string; title: string; state: TaskState; attempt: number; priority: 'interactive' | 'normal';
  lane: TaskLane; gpu: number | null; owner: TaskOwnerRef; group: { kind: string; id: string } | null;
  labels: Record<string, unknown>; queuePosition: number | null; waitingReason: string | null;
  progress: TaskProgress | null;
  resources: TaskResources | null;
  request: { lane: TaskLane; cpuThreads: number; dataWorkers: number; ramGb: number; vramGb: number; service?: boolean };
  exit: TaskExit | null;
  error?: string | null;
  failure?: TaskFailure | null;
  /** A cancel or pause asked for and not yet carried out. */
  stopRequest?: 'cancel' | 'pause' | null; stopRequestedAt?: string | null;
  createdAt: string; queuedAt: string | null; startedAt: string | null; finishedAt: string | null; updatedAt: string;
  link: string | null; logPath: string | null; actions: { cancel: boolean; retry: boolean };
  /** Concluded, but the runner may still requeue it (for example an undecided auto-resume). */
  awaitingRequeue?: boolean;
}
export interface TaskEvent {
  seq: number; at: string; attempt: number | null; fromState: TaskState | null; toState: TaskState | null;
  detail: Record<string, unknown> | string | null;
}
export interface TaskAttempt { attempt: number; startedAt: string | null; endedAt: string | null; state: TaskState | null; exitReason: string | null; gpu: number | null }
export interface TaskDependency { task: string; condition?: 'succeeded' | 'terminal'; state: TaskState | null; title?: string | null }
export interface TaskDetail extends TaskItem {
  events: TaskEvent[]; logTail: string | null; logTruncated?: boolean; logSize?: number | null;
  command?: { argv: string[]; cwd: string | null; env: Record<string, string>; log: string | null; progress: string | null; result: string | null };
  pid?: number | null; sessionName?: string | null;
  attempts?: TaskAttempt[]; dependencies?: TaskDependency[]; dependents?: TaskDependency[];
  taskTitles?: Record<string, string>;
}
export interface TaskOwner extends TaskOwnerRef {
  position: number | null; counts: TaskStateCounts; lanes: Partial<Record<TaskLane, number>>; createdAt: string;
  etaSeconds: number | null; link: string | null;
  /** The server's reason the first task in line waits (or that the owner is held). */
  waitingReason?: string | null;
  /** `inference` when the owner's runs are on an unlabeled cohort. */
  purpose?: string | null;
  labels?: Record<string, unknown>;
  actions: { hold: boolean; release: boolean; stop: boolean; cancel: boolean; retry: boolean; moveUp: boolean; moveDown: boolean };
}
export interface TaskCenterSnapshot { summary: TaskCenterSummary; running: TaskItem[]; owners: TaskOwner[]; pendingCount: number; updatedAt: string }
export interface TaskFailureSummary {
  taskId: string; title: string; state: TaskState; reason: string | null; message: string | null; cause: string | null;
  retry: TaskFailure['retry'] | null; at: string | null;
}
export interface TaskHistoryGroup {
  owner: TaskOwner;
  finished: { total: number; succeeded: number; failed: number; cancelled: number; interrupted: number };
  lastFinishedAt: string | null; firstStartedAt: string | null; lastFailure: TaskFailureSummary | null;
}
export interface TaskHistory { groups: TaskHistoryGroup[]; total: number; offset: number; limit: number }
export interface HistoryQuery { project?: string; kind?: string; state?: string; limit?: number; offset?: number }
/**
 * What a stage page asks about its runs. One of: an owner key; an owner kind and id; a
 * science record kind and id (matched on task labels); a project (optionally some task
 * kinds); or nothing for the whole machine.
 */
export interface RollupScope {
  owner?: string; ownerKind?: string; ownerId?: string; recordKind?: string; recordId?: string;
  /** Several records of `recordKind`, comma-separated (for example one attention batch). */
  recordIds?: string;
  project?: string; kinds?: string;
  /**
   * The training batches the page still keeps (an experiment's retained batches and its
   * current submission), comma-separated: concluded tasks of other batches (trashed, or from
   * an earlier plan) are left out. Live work always counts.
   */
  batchIds?: string;
}
export type RollupState = 'not-started' | 'queued' | 'running' | 'held' | 'stopping' | 'attention' | 'runner-stopped' | 'completed' | 'cancelled';
export interface TaskRollup {
  scope: RollupScope; state: RollupState; counts: TaskStateCounts;
  byKind: Record<string, { counts: TaskStateCounts; completed: number; total: number }>;
  /** Tasks completed of all tasks; only for an owner or record scope. */
  progress: { completed: number; total: number } | null;
  live: number; active: number; pending: number; held: boolean;
  /** The owner's place among queued owners, and the first waiting task's place in line. */
  position: number | null; queuePosition: number | null;
  waitingReason: string | null; eta: { seconds: number; basis: 'measured' | 'estimated' } | null;
  runnerAlive: boolean; paused: boolean; stopRequest: 'cancel' | 'pause' | null;
  lastFailure: TaskFailureSummary | null;
  /** Machine and project scopes: failures from the last 24 hours. */
  recentFailures: number | null;
  /**
   * Whether a retry would resume anything now: the owner's retry for an owner scope, a retry
   * of the record's own tasks for a record scope; null for a project or the machine. Older
   * services omit it.
   */
  retryable?: boolean | null;
  current: { taskId: string; title: string; kind: string; labels: Record<string, unknown>; progress: TaskProgress | null; startedAt: string | null } | null;
  startedAt: string | null; finishedAt: string | null;
  ownerKey: string | null; ownerKind: string | null; ownerId: string | null; title: string | null;
  projectId: string | null; projectName: string | null;
  /** `#task-center?owner=…&task=…&project=…` for this scope. */
  href: string; updatedAt: string;
}
export type SuggestionBinding = 'throughput' | 'gpu_memory' | 'ram' | 'cpu' | 'run_count' | 'default';
export interface CapacitySuggestion {
  version: 2; basis: 'measured' | 'estimated' | 'hardware_only'; parallelGpuTasks: number; perGpu: Record<string, number>;
  binding: SuggestionBinding;
  limits: { gpuMemory: number | null; ram: number | null; cpu: number | null; throughput: number | null; runCount: number | null };
  /** cpuThreads is what admission reserves per task (threads + data workers); cpuCores is measured use. */
  perTask: { vramGb: number | null; ramGb: number | null; cpuCores: number | null; cpuThreads?: number | null };
  throughput: { observed: { concurrency: number; secondsPerEpoch: number; runs: number }[]; trend: 'rising' | 'flat' | 'falling' | 'unknown'; note: string | null };
  evidence: { runs: number; batches: number; latest: string | null };
  explanation: string[]; findings: { code: string; severity: 'error' | 'warning' | 'info'; message: string }[];
}
export interface CapacitySettings {
  gpuSlots: Record<string, number>; defaultGpuSlots: number; cpuTaskSlots: number | null;
  reserves: { cpuThreads: number; ramGb: number | null; vramGb: number | null };
  defaults: { cpuThreadsPerRun: number; dataLoaderWorkers: number };
  paused: boolean; autoResume: boolean; cancelGraceSeconds: number; stallMinutes: number;
}
export interface CapacityResponse {
  settings: CapacitySettings;
  effective: { gpuSlots: Record<string, number>; cpuTaskSlots: number; reserves: { cpuThreads: number; ramGb: number; vramGb: number } };
  suggestion: CapacitySuggestion | null;
}
export interface CapacityPatch {
  parallelGpuTasks?: number; gpuSlots?: Record<string, number>; cpuTaskSlots?: number | null;
  paused?: boolean; autoResume?: boolean; defaults?: { cpuThreadsPerRun: number; dataLoaderWorkers: number };
}
export interface TaskQuery { state?: string; owner?: string; project?: string; kind?: string; limit?: number; offset?: number }

const base = '/task-center';
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export function taskQueryString(params: TaskQuery | HistoryQuery | Record<string, string | number | undefined> = {}) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value !== undefined && value !== '') query.set(key, String(value));
  const text = query.toString();
  return text ? `?${text}` : '';
}
export const taskCenter = {
  snapshot: () => request<TaskCenterSnapshot>(`${base}/snapshot`),
  rollup: (scope: RollupScope = {}) => request<TaskRollup>(`${base}/rollup${taskQueryString(scope as Record<string, string | undefined>)}`),
  history: (params: HistoryQuery = {}) => request<TaskHistory>(`${base}/history${taskQueryString(params)}`),
  tasks: (params: TaskQuery = {}) => request<{ tasks: TaskItem[]; hasMore?: boolean; offset?: number }>(`${base}/tasks${taskQueryString(params)}`),
  task: (id: string) => request<TaskDetail>(`${base}/tasks/${encodeURIComponent(id)}`),
  /** The whole log as text (the detail carries only its tail). */
  log: async (id: string) => (await fetchArtifactBlob(`${base}/tasks/${encodeURIComponent(id)}/log`)).text(),
  downloadLog: (id: string, filename: string) => downloadArtifact(`${base}/tasks/${encodeURIComponent(id)}/log?download=true`, filename),
  owner: (key: string) => request<TaskOwner>(`${base}/owners/${encodeURIComponent(key)}`),
  taskAction: (id: string, action: 'cancel' | 'retry', operationId: string) =>
    request<TaskItem>(`${base}/tasks/${encodeURIComponent(id)}/${action}`, post({ operationId })),
  ownerAction: (key: string, action: OwnerAction, operationId: string, position?: MovePosition) =>
    request<TaskOwner>(`${base}/owners/${encodeURIComponent(key)}/${action}`, post({ operationId, ...(action === 'move' ? { position } : {}) })),
  capacity: () => request<CapacityResponse>(`${base}/capacity`),
  updateCapacity: (patch: CapacityPatch & { operationId: string }) =>
    request<CapacityResponse>(`${base}/capacity`, { method: 'PUT', body: JSON.stringify(patch) }),
  startRunner: (operationId: string) => request<RunnerActionResponse>(`${base}/runner/start`, post({ operationId })),
  restartRunner: (operationId: string) => request<RunnerActionResponse>(`${base}/runner/restart`, post({ operationId })),
};

/** What to tell the user after Start or Restart runner; null when the runner simply runs. */
export function runnerActionOutcome(response?: RunnerActionResponse | null): { message: string; pending: boolean } | null {
  const result = response?.result;
  if (!response || !result) return null;
  if (result.pending) return { message: result.note || 'The old runner is finishing its current step; it restarts when that ends.', pending: true };
  if (!response.runner.alive && result.started === false) {
    return { message: `The runner did not start${result.reason ? `: ${result.reason}` : ''}.`, pending: false };
  }
  return null;
}

/** Shared cache keys; the page, the job tray and the experiment queue bar read the same entries. */
export const taskCenterKeys = {
  all: ['task-center'] as const,
  snapshot: ['task-center', 'snapshot'] as const,
  rollup: (scope: RollupScope) => ['task-center', 'rollup', rollupScopeKey(scope)] as const,
  history: (params: HistoryQuery) => ['task-center', 'history', taskQueryString(params)] as const,
  capacity: ['task-center', 'capacity'] as const,
  owner: (key: string) => ['task-center', 'owner', key] as const,
  tasks: (view: string, ...rest: string[]) => ['task-center', 'tasks', view, ...rest] as const,
  task: (id: string) => ['task-center', 'task', id] as const,
};

const liveStates: readonly TaskState[] = ['blocked', 'queued', 'starting', 'running', 'stopping'];
export const taskLive = (task?: { state: TaskState } | TaskState | null) => {
  const state = typeof task === 'string' ? task : task?.state;
  return Boolean(state && liveStates.includes(state));
};
const stateLabels: Record<TaskState, string> = {
  blocked: 'Waiting on dependencies', queued: 'Queued', starting: 'Starting', running: 'Running', stopping: 'Stopping',
  succeeded: 'Completed', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted',
};
export const taskStateLabel = (state: TaskState | string) => stateLabels[state as TaskState] ?? state.replace(/^./, (letter) => letter.toUpperCase());
/** Active work reads as the product colour, waiting work stays neutral, and only
 * failures ask for attention. */
const stateTones: Record<TaskState, string> = {
  blocked: 'neutral', queued: 'neutral', starting: 'green', running: 'green', stopping: 'amber',
  succeeded: 'success', failed: 'orange', cancelled: 'neutral', interrupted: 'amber',
};
export const taskStateTone = (state: TaskState | string) => stateTones[state as TaskState] ?? 'neutral';

export function taskCounts(counts?: TaskStateCounts | null) {
  const value = (state: TaskState) => counts?.[state] ?? 0;
  return {
    running: value('starting') + value('running') + value('stopping'),
    queued: value('queued'), blocked: value('blocked'),
    succeeded: value('succeeded'), failed: value('failed'), cancelled: value('cancelled'), interrupted: value('interrupted'),
  };
}
export const taskCenterLive = (summary?: Pick<TaskCenterSummary, 'counts'> | null) => {
  const counts = taskCounts(summary?.counts);
  return counts.running + counts.queued + counts.blocked > 0;
};
export const taskCenterPollInterval = (summary?: Pick<TaskCenterSummary, 'counts'> | null) => taskCenterLive(summary) ? 2000 : 10000;

/** "4 running · 41 queued", with other states only when present. */
export function ownerCountsText(counts?: TaskStateCounts | null) {
  const value = taskCounts(counts);
  return [`${value.running} running`, `${value.queued} queued`,
    value.blocked ? `${value.blocked} waiting on dependencies` : '', value.succeeded ? `${value.succeeded} completed` : '',
    value.failed ? `${value.failed} failed` : '', value.interrupted ? `${value.interrupted} interrupted` : '',
    value.cancelled ? `${value.cancelled} cancelled` : ''].filter(Boolean).join(' · ');
}

export function formatDuration(seconds?: number | null) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.round(seconds);
  if (total < 60) return `${total} s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes} m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours} h ${minutes % 60} m`;
  return `${Math.floor(hours / 24)} d ${hours % 24} h`;
}
export function secondsBetween(start?: string | null, end: string | number | null = Date.now()) {
  const from = start ? Date.parse(start) : NaN;
  const to = typeof end === 'number' ? end : end ? Date.parse(end) : NaN;
  return Number.isFinite(from) && Number.isFinite(to) ? Math.max(0, (to - from) / 1000) : null;
}

/** One readable progress line for fold runs and compute jobs. */
export function taskProgress(progress?: TaskProgress | null): { text: string; value: number; max: number } | null {
  if (!progress) return null;
  const pair = (value: unknown, max: unknown, text: (value: number, max: number) => string) =>
    typeof value === 'number' && typeof max === 'number' && max > 0 ? { text: text(value, max), value: Math.min(value, max), max } : null;
  return pair(progress.epoch, progress.maxEpochs, (value, max) => `Epoch ${value} / ${max}`)
    ?? pair(progress.completedModels, progress.totalModels, (value, max) => `${value} / ${max} models`)
    ?? pair(progress.completedPairs, progress.totalPairs, (value, max) => `${value} / ${max} slide–checkpoint pairs`)
    ?? pair(progress.completedSlides, progress.totalSlides, (value, max) => `${value} / ${max} slides`)
    ?? pair(progress.completed, progress.total, (value, max) => `${value} / ${max}`);
}

const exitLabels: Record<string, string> = {
  ok: 'Finished', error: 'Error', oom: 'Out of memory', cuda_failure: 'CUDA device failure', cancelled: 'Cancelled',
  paused: 'Stopped and requeued', interrupted: 'Interrupted', lost: 'Process lost', 'already-complete': 'Already complete',
  busy: 'Output busy; requeued',
};
export const exitReasonLabel = (reason?: string | null) => reason ? exitLabels[reason] ?? reason : null;

const ownerKinds: Record<string, string> = {
  experiment: 'Experiment', 'mil-batch': 'Training batch', 'evaluation-batch': 'Batch · labeled cohort',
  'predictor-refit': 'Refit', refit: 'Refit', 'model-evaluation': 'Run · labeled cohort', evaluation: 'Run · labeled cohort',
  interpretation: 'Interpretation', 'attention-interpretation': 'Interpretation', 'model-interpretation': 'Interpretation',
  extraction: 'Feature extraction', 'feature-pack': 'Feature packing', archive: 'Study archive',
};
const capitalized = (kind: string) => kind.replaceAll('-', ' ').replace(/^./, (letter) => letter.toUpperCase());
/** An owner's kind; a run or batch says whether its cohort is labeled, as in Apply models. */
export const ownerKindLabel = (kind: string, purpose?: string | null) => {
  if (purpose === 'inference' && kind === 'evaluation-batch') return 'Batch · unlabeled cohort';
  if (purpose === 'inference' && (kind === 'model-evaluation' || kind === 'evaluation')) return 'Run · unlabeled cohort';
  return ownerKinds[kind] ?? capitalized(kind);
};
const taskKinds: Record<string, string> = {
  'mil-fold': 'Fold run', 'mil-collect': 'Results collection', 'compute-job': 'Compute job',
  'predictor-coordinator': 'Predictor creation', 'bulk-submit': 'Bulk submission',
  extraction: 'Feature extraction', 'extraction-validation': 'Extraction check', packing: 'Feature packing', archive: 'Study archive',
};
const computeKinds: Record<string, string> = {
  refit: 'Refit training', 'predictor-refit': 'Refit training', evaluation: 'Predictions', 'model-evaluation': 'Predictions',
  interpretation: 'Attention maps', 'model-interpretation': 'Attention maps', 'attention-interpretation': 'Attention maps',
};
/** A task's kind; compute jobs say which job (refit, a run's predictions, attention). */
export const taskKindLabel = (kind: string, labels?: Record<string, unknown> | null) => {
  if (kind === 'compute-job' && labels) {
    const compute = typeof labels.computeKind === 'string' ? computeKinds[labels.computeKind] : undefined;
    if (compute) return compute;
  }
  return taskKinds[kind] ?? capitalized(kind);
};
const modelNames: Record<string, string> = { abmil: 'ABMIL', nnmil: 'nnMIL', transmil: 'TransMIL', clam: 'CLAM', dsmil: 'DSMIL', meanpool: 'Mean pooling', maxpool: 'Max pooling' };
/**
 * A task title as the queue shows it. Refit titles end in a repeated "· refit" and do not
 * name the model; the model label, when recorded, replaces that suffix.
 */
export function taskDisplayTitle(task: Pick<TaskItem, 'kind' | 'title' | 'labels'>) {
  if (task.kind !== 'compute-job' || !['refit', 'predictor-refit'].includes(String(task.labels?.computeKind))) return task.title;
  const model = typeof task.labels?.model === 'string' ? modelNames[task.labels.model.toLowerCase()] ?? task.labels.model : null;
  const title = task.title.replace(/\s·\s*refit$/i, '');
  return model ? `${title} · ${model}` : title;
}

/**
 * Task links carry their project (`?project=<id>#…`). Inside the same project a
 * hash link keeps the page loaded; another project in this workspace reloads; a
 * task from another workspace has no link at all.
 */
export function taskCenterLink(link: string | null | undefined, projectId?: string) {
  if (!link) return null;
  if (link.startsWith('#')) return link;
  const hash = link.indexOf('#');
  const query = new URLSearchParams(link.slice(link.startsWith('?') ? 1 : 0, hash < 0 ? undefined : hash));
  if (projectId && hash >= 0 && query.get('project') === projectId) return link.slice(hash);
  return link;
}

/** Loss, validation and phase lines for a running task, beside its main progress. */
export function taskProgressDetails(progress?: TaskProgress | null): string[] {
  if (!progress) return [];
  const number = (value: unknown, digits = 4) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : null;
  const lines: string[] = [];
  const loss = number(progress.trainingLoss);
  if (loss) lines.push(`Training loss ${loss}`);
  const validation = progress.validation;
  if (validation && validation.available !== false) {
    const parts = [number(validation.loss) ? `loss ${number(validation.loss)}` : '', number(validation.auroc, 3) ? `AUROC ${number(validation.auroc, 3)}` : ''].filter(Boolean);
    if (parts.length) lines.push(`Validation ${parts.join(' · ')}`);
  }
  for (const key of ['phase', 'label', 'message'] as const) {
    const value = progress[key];
    if (typeof value === 'string' && value.trim() && !lines.includes(value.trim())) lines.push(value.trim());
  }
  return lines;
}
/** Measured peak GPU memory from a fold's progress record, in GiB. */
export function measuredVramGb(task: Pick<TaskItem, 'progress' | 'resources'>) {
  const measured = task.resources?.peakVramGb ?? task.resources?.vramGb;
  if (typeof measured === 'number' && Number.isFinite(measured)) return measured;
  const bytes = task.progress?.cudaPeakReservedBytes;
  return typeof bytes === 'number' && Number.isFinite(bytes) && bytes > 0 ? bytes / 1024 ** 3 : null;
}

export interface TaskCenterRoute { owner?: string; task?: string; project?: string; kind?: string; state?: string }
const routeKeys = ['owner', 'task', 'project', 'kind', 'state'] as const;
/** `#task-center?owner=…&task=…&project=…&kind=…&state=…`, empty values dropped. */
/** Everything the Task Center runs, for the places that describe it. Sentence case, no final stop. */
export const TASK_CENTER_WORK = 'training folds and results, refits and predictors, predictor runs on cohorts, attention maps, feature extraction and validation, feature packing, and study archives';
export const taskCenterWork = (capitalized = false) => capitalized ? TASK_CENTER_WORK[0].toUpperCase() + TASK_CENTER_WORK.slice(1) : TASK_CENTER_WORK;
export function taskCenterHref(route: TaskCenterRoute = {}) {
  const query = new URLSearchParams();
  for (const key of routeKeys) if (route[key]) query.set(key, route[key]!);
  const text = query.toString();
  return text ? `#task-center?${text}` : '#task-center';
}
export function readTaskCenterRoute(hash: string = typeof window === 'undefined' ? '' : window.location.hash): TaskCenterRoute {
  const parameters = new URLSearchParams(hash.split('?')[1] ?? '');
  const route: TaskCenterRoute = {};
  for (const key of routeKeys) { const value = parameters.get(key); if (value) route[key] = value; }
  return route;
}
/** A stable cache key for a rollup scope, whatever order its fields were given in. */
export function rollupScopeKey(scope: RollupScope) {
  return taskQueryString(Object.fromEntries(Object.entries(scope).filter(([, value]) => value).sort(([a], [b]) => a.localeCompare(b))));
}
const rollupLive = (rollup?: Pick<TaskRollup, 'state'> | null) => Boolean(rollup && ['queued', 'running', 'held', 'stopping', 'runner-stopped'].includes(rollup.state));
/** Fast only while something runs: 3 s active, 5 s waiting, 30 s otherwise. */
export function rollupPollInterval(rollup?: Pick<TaskRollup, 'state' | 'active'> | null) {
  if (!rollup) return 30000;
  if (rollup.state === 'running' || rollup.state === 'stopping') return 3000;
  return rollupLive(rollup) ? 5000 : 30000;
}
export const rollupIsLive = rollupLive;
/** Stage Resume/Retry buttons appear only when a retry would resume something. */
export const rollupRetryable = (rollup?: Pick<TaskRollup, 'retryable'> | null) => Boolean(rollup) && rollup!.retryable !== false;
