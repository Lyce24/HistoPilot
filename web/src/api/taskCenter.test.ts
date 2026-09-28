import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  exitReasonLabel, formatDuration, measuredVramGb, ownerCountsText, ownerKindLabel, readTaskCenterRoute, rollupPollInterval, rollupScopeKey, runnerActionOutcome,
  secondsBetween, taskCenterHref, taskCenterLink, taskCenterLive, taskCenterPollInterval, taskCounts, taskDisplayTitle, taskKindLabel,
  taskLive, taskProgress, taskProgressDetails, taskQueryString, taskStateLabel, taskStateTone, type TaskState,
} from './taskCenter';
import { fixtureSummary } from '../testFixtures/taskCenter';

const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('Task Center API contract', () => {
  it('reads filtered tasks, one task and one queue owner', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }));
    for (let index = 0; index < 4; index++) fetcher.mockResolvedValueOnce(response({}));
    vi.stubGlobal('fetch', fetcher);
    const { taskCenter } = await import('./taskCenter');
    await taskCenter.tasks({ state: 'starting,running,stopping', limit: 100 });
    await taskCenter.tasks({ owner: 'owner/1', project: '', kind: undefined });
    await taskCenter.task('task/1');
    await taskCenter.owner('owner/1');
    expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
      '/api/v1/task-center/tasks?state=starting%2Crunning%2Cstopping&limit=100',
      '/api/v1/task-center/tasks?owner=owner%2F1',
      '/api/v1/task-center/tasks/task%2F1',
      '/api/v1/task-center/owners/owner%2F1',
    ]);
    expect(fetcher.mock.calls[1][1].headers.get('X-HistoPilot-Token')).toBe('session');
  });

  it('sends every action with its operation ID and only a move with a position', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }));
    for (let index = 0; index < 7; index++) fetcher.mockResolvedValueOnce(response({}));
    vi.stubGlobal('fetch', fetcher);
    const { taskCenter } = await import('./taskCenter');
    await taskCenter.taskAction('task/1', 'cancel', 'op-1');
    await taskCenter.taskAction('task-2', 'retry', 'op-2');
    await taskCenter.ownerAction('owner/1', 'move', 'op-3', 'top');
    await taskCenter.ownerAction('owner-1', 'hold', 'op-4', 'top');
    await taskCenter.updateCapacity({ operationId: 'op-5', parallelGpuTasks: 5 });
    await taskCenter.startRunner('op-6');
    await taskCenter.restartRunner('op-7');
    const calls = fetcher.mock.calls.slice(1).map(([path, init]) => [path, init.method, JSON.parse(init.body)]);
    expect(calls).toEqual([
      ['/api/v1/task-center/tasks/task%2F1/cancel', 'POST', { operationId: 'op-1' }],
      ['/api/v1/task-center/tasks/task-2/retry', 'POST', { operationId: 'op-2' }],
      ['/api/v1/task-center/owners/owner%2F1/move', 'POST', { operationId: 'op-3', position: 'top' }],
      ['/api/v1/task-center/owners/owner-1/hold', 'POST', { operationId: 'op-4' }],
      ['/api/v1/task-center/capacity', 'PUT', { operationId: 'op-5', parallelGpuTasks: 5 }],
      ['/api/v1/task-center/runner/start', 'POST', { operationId: 'op-6' }],
      ['/api/v1/task-center/runner/restart', 'POST', { operationId: 'op-7' }],
    ]);
  });

  it('says why a runner start did not happen, or that a restart follows the current step', () => {
    const stopped = { ...fixtureSummary({}).runner, alive: false };
    expect(runnerActionOutcome({ runner: stopped, result: { started: false, reason: 'tmux unavailable' } }))
      .toEqual({ message: 'The runner did not start: tmux unavailable.', pending: false });
    expect(runnerActionOutcome({ runner: stopped, result: { started: false } })?.message).toBe('The runner did not start.');
    const pending = runnerActionOutcome({ runner: fixtureSummary({}).runner, result: { started: false, pending: true, note: 'The old runner is finishing its current step; it restarts when that ends.' } });
    expect(pending).toEqual({ message: 'The old runner is finishing its current step; it restarts when that ends.', pending: true });
    // A runner that is alive (or just started) needs no note.
    expect(runnerActionOutcome({ runner: fixtureSummary({}).runner, result: { started: false, alive: true } })).toBeNull();
    expect(runnerActionOutcome({ runner: stopped, result: { started: true, note: 'already starting' } })).toBeNull();
    expect(runnerActionOutcome({ runner: stopped, result: null })).toBeNull();
    expect(runnerActionOutcome(undefined)).toBeNull();
  });

  it('reads the snapshot, a scoped rollup, grouped history, paged tasks and the whole log', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }));
    for (let index = 0; index < 5; index++) fetcher.mockResolvedValueOnce(response({}));
    fetcher.mockResolvedValueOnce(new Response('whole log', { status: 200 }));
    vi.stubGlobal('fetch', fetcher);
    const { taskCenter } = await import('./taskCenter');
    await taskCenter.snapshot();
    await taskCenter.rollup({ ownerKind: 'experiment', ownerId: 'exp/1', project: 'p1' });
    await taskCenter.rollup();
    await taskCenter.history({ project: 'p1', state: 'failed', limit: 15, offset: 15 });
    await taskCenter.tasks({ owner: 'o', state: 'history', limit: 50, offset: 0 });
    expect(await taskCenter.log('task/1')).toBe('whole log');
    expect(fetcher.mock.calls.slice(1).map(([path]) => path)).toEqual([
      '/api/v1/task-center/snapshot',
      '/api/v1/task-center/rollup?ownerKind=experiment&ownerId=exp%2F1&project=p1',
      '/api/v1/task-center/rollup',
      '/api/v1/task-center/history?project=p1&state=failed&limit=15&offset=15',
      '/api/v1/task-center/tasks?owner=o&state=history&limit=50&offset=0',
      '/api/v1/task-center/tasks/task%2F1/log',
    ]);
  });

  it('builds query strings without empty filters', () => {
    expect(taskQueryString()).toBe('');
    expect(taskQueryString({ state: 'history', limit: 100, owner: '' })).toBe('?state=history&limit=100');
  });
});

describe('Task Center state helpers', () => {
  const states: TaskState[] = ['blocked', 'queued', 'starting', 'running', 'stopping', 'succeeded', 'failed', 'cancelled', 'interrupted'];

  it('labels every task state, treats pending and active work as live, and never uses an unstyled tone', () => {
    expect(states.map(taskStateLabel)).toEqual(['Waiting on dependencies', 'Queued', 'Starting', 'Running', 'Stopping', 'Completed', 'Failed', 'Cancelled', 'Interrupted']);
    expect(states.filter((state) => taskLive(state))).toEqual(['blocked', 'queued', 'starting', 'running', 'stopping']);
    expect(taskLive({ state: 'failed' })).toBe(false);
    expect(taskLive(null)).toBe(false);
    const styled = ['neutral', 'success', 'green', 'warning', 'amber', 'orange', 'purple', 'frozen'];
    for (const state of states) expect(styled).toContain(taskStateTone(state));
    expect(taskStateTone('failed')).toBe('orange');
    expect(taskStateTone('something-new')).toBe('neutral');
    expect(taskStateLabel('something-new')).toBe('Something-new');
  });

  it('polls every two seconds while anything is live and every ten seconds otherwise', () => {
    expect(taskCenterPollInterval({ counts: { queued: 1 } })).toBe(2000);
    expect(taskCenterPollInterval({ counts: { blocked: 1, succeeded: 4 } })).toBe(2000);
    expect(taskCenterPollInterval({ counts: { stopping: 1 } })).toBe(2000);
    expect(taskCenterPollInterval({ counts: { succeeded: 3, failed: 1 } })).toBe(10000);
    expect(taskCenterPollInterval(undefined)).toBe(10000);
    expect(taskCenterLive({ counts: {} })).toBe(false);
  });

  it('summarizes owner counts with running work first and quiet zeros elsewhere', () => {
    expect(taskCounts({ starting: 1, running: 3, stopping: 1, queued: 2 })).toMatchObject({ running: 5, queued: 2, blocked: 0 });
    expect(ownerCountsText({ running: 4, queued: 41 })).toBe('4 running · 41 queued');
    expect(ownerCountsText({ blocked: 1, succeeded: 20, failed: 2 })).toBe('0 running · 0 queued · 1 waiting on dependencies · 20 completed · 2 failed');
  });

  it('formats durations and progress for folds and compute jobs', () => {
    expect(formatDuration(42)).toBe('42 s');
    expect(formatDuration(7800)).toBe('2 h 10 m');
    expect(formatDuration(3 * 86400 + 3600)).toBe('3 d 1 h');
    expect(formatDuration(null)).toBe('—');
    expect(secondsBetween('2026-09-27T10:00:00Z', '2026-09-27T10:02:00Z')).toBe(120);
    expect(secondsBetween(null, '2026-09-27T10:02:00Z')).toBeNull();
    expect(taskProgress({ epoch: 23, maxEpochs: 100 })).toEqual({ text: 'Epoch 23 / 100', value: 23, max: 100 });
    expect(taskProgress({ completedModels: 2, totalModels: 5 })?.text).toBe('2 / 5 models');
    expect(taskProgress({ completedSlides: 7, totalSlides: 9 })?.text).toBe('7 / 9 slides');
    expect(taskProgress({ epoch: 1 })).toBeNull();
    expect(exitReasonLabel('oom')).toBe('Out of memory');
    expect(exitReasonLabel(null)).toBeNull();
  });

  it('keeps same-project links in the page, reloads for another project and drops missing links', () => {
    const link = '?project=p1#experiments?experiment=e&tab=runs&batch=b';
    expect(taskCenterLink(link, 'p1')).toBe('#experiments?experiment=e&tab=runs&batch=b');
    expect(taskCenterLink(link, 'p2')).toBe(link);
    expect(taskCenterLink('#evaluation', 'p1')).toBe('#evaluation');
    expect(taskCenterLink(null, 'p1')).toBeNull();
  });
});

describe('Task Center links, labels and polling', () => {
  it('writes and reads deep links with only the filters given', () => {
    expect(taskCenterHref()).toBe('#task-center');
    expect(taskCenterHref({ owner: 'owner-1', task: 't/1', project: 'p1', kind: undefined, state: '' })).toBe('#task-center?owner=owner-1&task=t%2F1&project=p1');
    expect(readTaskCenterRoute('#task-center?owner=owner-1&task=t%2F1&kind=mil-fold&other=x')).toEqual({ owner: 'owner-1', task: 't/1', kind: 'mil-fold' });
    expect(readTaskCenterRoute('#task-center')).toEqual({});
    expect(rollupScopeKey({ project: 'p', ownerKind: 'experiment', ownerId: 'e' })).toBe(rollupScopeKey({ ownerId: 'e', ownerKind: 'experiment', project: 'p' }));
  });

  it('names compute jobs by what they do and inference by its purpose', () => {
    expect(taskKindLabel('compute-job', { computeKind: 'refit' })).toBe('Refit training');
    expect(taskKindLabel('compute-job', { computeKind: 'evaluation', purpose: 'inference' })).toBe('Inference');
    expect(taskKindLabel('compute-job', { computeKind: 'interpretation' })).toBe('Attention maps');
    expect(taskKindLabel('compute-job')).toBe('Compute job');
    expect(taskKindLabel('extraction')).toBe('Feature extraction');
    expect(ownerKindLabel('evaluation-batch', 'inference')).toBe('Inference batch');
    expect(ownerKindLabel('model-evaluation', 'inference')).toBe('Inference');
    expect(ownerKindLabel('model-evaluation')).toBe('Evaluation');
    expect(ownerKindLabel('feature-pack')).toBe('Feature packing');
  });

  it('drops the repeated refit suffix and names the model when recorded', () => {
    const refit = { kind: 'compute-job', title: 'Refit · gej3 · config 1 · seed 42 / split 42 · refit', labels: { computeKind: 'refit' } };
    expect(taskDisplayTitle(refit)).toBe('Refit · gej3 · config 1 · seed 42 / split 42');
    expect(taskDisplayTitle({ ...refit, labels: { computeKind: 'refit', model: 'nnmil' } })).toBe('Refit · gej3 · config 1 · seed 42 / split 42 · nnMIL');
    expect(taskDisplayTitle({ kind: 'mil-fold', title: 'Fold · refit', labels: {} })).toBe('Fold · refit');
  });

  it('describes loss, validation and measured GPU memory for running work', () => {
    expect(taskProgressDetails({ epoch: 3, maxEpochs: 10, trainingLoss: 0.12345, validation: { loss: 0.2, auroc: 0.91234, available: true }, message: 'Scoring slides' }))
      .toEqual(['Training loss 0.1235', 'Validation loss 0.2000 · AUROC 0.912', 'Scoring slides']);
    expect(taskProgressDetails(null)).toEqual([]);
    expect(measuredVramGb({ progress: { cudaPeakReservedBytes: 2 * 1024 ** 3 }, resources: null })).toBe(2);
    expect(measuredVramGb({ progress: null, resources: { peakVramGb: 3.5 } })).toBe(3.5);
    expect(measuredVramGb({ progress: null, resources: null })).toBeNull();
  });

  it('polls a rollup fast only while its work runs', () => {
    expect(rollupPollInterval({ state: 'running', active: 2 })).toBe(3000);
    expect(rollupPollInterval({ state: 'queued', active: 0 })).toBe(5000);
    expect(rollupPollInterval({ state: 'runner-stopped', active: 0 })).toBe(5000);
    expect(rollupPollInterval({ state: 'completed', active: 0 })).toBe(30000);
    expect(rollupPollInterval({ state: 'attention', active: 0 })).toBe(30000);
    expect(rollupPollInterval(undefined)).toBe(30000);
  });
});
