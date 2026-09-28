import type { CapacityResponse, CapacitySuggestion, TaskCenterSnapshot, TaskCenterSummary, TaskDetail, TaskHistoryGroup, TaskItem, TaskOwner, TaskRollup } from '../api/taskCenter';

export function fixtureSummary(counts: TaskCenterSummary['counts'], changes: Partial<TaskCenterSummary> = {}): TaskCenterSummary {
  return {
    runner: { alive: true, heartbeatAt: '2026-09-27T10:00:00Z', pid: 10, state: 'running', message: null, codeHash: 'abc', codeCurrent: true, autostart: true },
    paused: false,
    capacity: {
      gpus: [{ index: 0, name: 'NVIDIA RTX A5000', slots: 5, usedSlots: 4, foreignSlots: 1, totalMemoryGb: 24, usedMemoryGb: 7.2, committedMemoryGb: 9.2, utilizationPercent: 52 }],
      cpu: { logical: 36, physical: 18, committedThreads: 16, reserveThreads: 2, cpuTaskSlots: 9, usedCpuTasks: 1 },
      ram: { totalGb: 188.7, availableGb: 150.2, reserveGb: 9.4 },
    },
    counts, eta: { seconds: 7800, basis: 'measured' }, foreignLeases: [], workspace: '/workspace', updatedAt: '2026-09-27T10:00:01Z', ...changes,
  };
}

export function fixtureTask(changes: Partial<TaskItem> = {}): TaskItem {
  return {
    id: 'task-1', kind: 'mil-fold', title: 'Baseline · Config 1 · Fold 1 · Train seed 42 · Split seed 7', state: 'running', attempt: 1, priority: 'normal', lane: 'gpu', gpu: 0,
    owner: { key: 'owner-1', kind: 'experiment', id: 'exp', title: 'KRAS study', projectId: 'project', projectFolder: '/projects/kras', sameWorkspace: true, held: false, queueSeq: 1 },
    group: { kind: 'mil-batch', id: 'batch' }, labels: {}, queuePosition: 1, waitingReason: null, progress: { epoch: 23, maxEpochs: 100 },
    resources: { privateRamGb: 4.4, peakPrivateRamGb: 5.1, cpuCores: 2.3, meanConcurrency: 3.9 },
    request: { lane: 'gpu', cpuThreads: 2, dataWorkers: 2, ramGb: 6, vramGb: 2.3 }, exit: null,
    createdAt: '2026-09-27T09:00:00Z', queuedAt: '2026-09-27T09:00:00Z', startedAt: '2026-09-27T09:30:00Z', finishedAt: null, updatedAt: '2026-09-27T10:00:00Z',
    link: '?project=project#experiments?experiment=exp&tab=runs&batch=batch', logPath: '/projects/kras/training/batch/runs/r1/run.log',
    actions: { cancel: true, retry: false }, ...changes,
  };
}

export function fixtureOwner(changes: Partial<TaskOwner> = {}): TaskOwner {
  return {
    key: 'owner-1', kind: 'experiment', id: 'exp', title: 'KRAS study', projectId: 'project', projectFolder: '/projects/kras', sameWorkspace: true, held: false, queueSeq: 1,
    position: 1, counts: { running: 4, queued: 41, succeeded: 20 }, lanes: { gpu: 45 }, createdAt: '2026-09-27T09:00:00Z', etaSeconds: 7800,
    link: '?project=project#experiments?experiment=exp&tab=runs', actions: { hold: true, release: false, stop: true, cancel: true, retry: false, moveUp: false, moveDown: true },
    ...changes,
  };
}

export function fixtureSuggestion(changes: Partial<CapacitySuggestion> = {}): CapacitySuggestion {
  return {
    version: 2, basis: 'measured', parallelGpuTasks: 5, perGpu: { '0': 5 }, binding: 'throughput',
    limits: { gpuMemory: 10, ram: 25, cpu: 14, throughput: 5, runCount: null },
    perTask: { vramGb: 2.3, ramGb: 5, cpuCores: 2.4, cpuThreads: 4 },
    throughput: { observed: [{ concurrency: 4, secondsPerEpoch: 16.8, runs: 13 }, { concurrency: 2.8, secondsPerEpoch: 15.9, runs: 2 }], trend: 'rising', note: null },
    evidence: { runs: 20, batches: 2, latest: '2026-09-26T12:00:00Z' },
    explanation: ['Measured 20 runs on this machine: 1.2–1.8 GiB GPU memory and about 4.5 GiB RAM per run.', 'GPU memory would allow 10, RAM 25, CPU 14.'],
    findings: [], ...changes,
  };
}

export function fixtureCapacity(changes: Partial<CapacityResponse> = {}): CapacityResponse {
  return {
    settings: { gpuSlots: { '0': 4 }, defaultGpuSlots: 4, cpuTaskSlots: null, reserves: { cpuThreads: 2, ramGb: null, vramGb: null }, defaults: { cpuThreadsPerRun: 2, dataLoaderWorkers: 2 }, paused: false, autoResume: true, cancelGraceSeconds: 30, stallMinutes: 30 },
    effective: { gpuSlots: { '0': 4 }, cpuTaskSlots: 9, reserves: { cpuThreads: 2, ramGb: 9.4, vramGb: 1.2 } },
    suggestion: fixtureSuggestion(), ...changes,
  };
}

export function fixtureDetail(changes: Partial<TaskDetail> = {}): TaskDetail {
  return {
    ...fixtureTask(), events: [{ seq: 1, at: '2026-09-27T09:00:00Z', attempt: 1, fromState: null, toState: 'queued', detail: null }, { seq: 2, at: '2026-09-27T09:30:00Z', attempt: 1, fromState: 'queued', toState: 'running', detail: { pid: 42, gpu: 0 } }],
    logTail: 'epoch 23 loss 0.41\n', logTruncated: false, logSize: 19,
    command: { argv: ['/venv/bin/python', '-u', '-m', 'histopilot.workers.managed_fold', '/projects/kras/training/batch/plan.json'], cwd: '/projects/kras/training/batch/compute', env: { PYTHONUNBUFFERED: '1' }, log: '/projects/kras/training/batch/runs/r1/run.log', progress: '/projects/kras/training/batch/runs/r1/progress.json', result: null },
    pid: 42, sessionName: null,
    attempts: [{ attempt: 1, startedAt: '2026-09-27T09:30:00Z', endedAt: null, state: 'running', exitReason: null, gpu: 0 }],
    dependencies: [], dependents: [{ task: 'collect', title: 'Baseline · Final results', state: 'blocked' }], taskTitles: { collect: 'Baseline · Final results' },
    ...changes,
  };
}

export function fixtureRollup(changes: Partial<TaskRollup> = {}): TaskRollup {
  return {
    scope: { ownerKind: 'experiment', ownerId: 'exp', project: 'project' }, state: 'running',
    counts: { running: 3, queued: 12, failed: 1, succeeded: 14 },
    byKind: { 'mil-fold': { counts: { running: 3, queued: 12, succeeded: 14, failed: 1 }, completed: 14, total: 30 } },
    progress: { completed: 14, total: 30 }, live: 15, active: 3, pending: 12, held: false, position: 1, queuePosition: 4,
    waitingReason: null, eta: { seconds: 2400, basis: 'measured' }, runnerAlive: true, paused: false, stopRequest: null,
    lastFailure: null, recentFailures: null, retryable: true, current: null, startedAt: '2026-09-27T09:00:00Z', finishedAt: null,
    ownerKey: 'owner-1', ownerKind: 'experiment', ownerId: 'exp', title: 'KRAS study', projectId: 'project', projectName: 'KRAS',
    href: '#task-center?owner=owner-1&project=project', updatedAt: '2026-09-27T10:00:00Z', ...changes,
  };
}

export function fixtureSnapshot(changes: Partial<TaskCenterSnapshot> = {}): TaskCenterSnapshot {
  return { summary: fixtureSummary({ running: 1, queued: 41 }), running: [fixtureTask()], owners: [fixtureOwner()], pendingCount: 41, updatedAt: '2026-09-27T10:00:01Z', ...changes };
}

export function fixtureHistoryGroup(changes: Partial<TaskHistoryGroup> = {}): TaskHistoryGroup {
  return {
    owner: fixtureOwner({ counts: { succeeded: 60, failed: 1 }, position: null, actions: { hold: false, release: false, stop: false, cancel: false, retry: true, moveUp: false, moveDown: false } }),
    finished: { total: 61, succeeded: 60, failed: 1, cancelled: 0, interrupted: 0 }, lastFinishedAt: '2026-09-27T10:05:00Z', firstStartedAt: '2026-09-27T09:00:00Z',
    lastFailure: { taskId: 'task-9', title: 'Predictors · KRAS study', state: 'failed', reason: 'error', message: 'The project was busy', cause: 'Another HistoPilot operation was changing this project when the task started, so it stopped without changing anything.', retry: 'safe', at: '2026-09-27T10:05:00Z' },
    ...changes,
  };
}
