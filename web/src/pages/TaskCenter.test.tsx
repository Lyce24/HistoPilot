import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import type { Workspace } from '../api/types';
import type { ModelExperimentSummary } from '../api/experiments';
import { batchTemplate } from '../components/DevelopmentBatches';
import TaskCenter, { FailureExplanation, ParallelTasks, RunnerStrip, SuggestionCallout, TaskDetails, TaskHistory, TaskName, drawerDepth, ownerState, ownerWaitingReason, routeHistoryMode, runnerOutcomeNote, taskCenterErrorMessage, trappedFocus, withCurrentOption } from './TaskCenter';
import { fixtureCapacity, fixtureDetail, fixtureHistoryGroup, fixtureOwner, fixtureSnapshot, fixtureSuggestion, fixtureSummary, fixtureTask } from '../testFixtures/taskCenter';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
const workspace = { mode: 'local', project: { id: 'project', name: 'Demo', lifecycleState: 'active' } } as Workspace;
const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const ready = { id: 'ready-exp', key: 'draft:ready-exp', name: 'Frozen baseline', notes: '', tags: [], revision: 3, state: 'active', status: 'ready', legacy: false, createdAt: '', updatedAt: '', inputs, batches: [], drafts: [], predictorId: null, stage: 'planning', frozenSetupId: 'setup', batchPlans: [{ id: 'plan', spec: batchTemplate('baseline', inputs, 'Frozen baseline') }] } as ModelExperimentSummary;
const idle = actions();
function actions() { return { run: async () => undefined, pending: null, isPending: () => false }; }

function seeded(changes: { summary?: ReturnType<typeof fixtureSummary>; capacity?: ReturnType<typeof fixtureCapacity> } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['task-center', 'snapshot'], fixtureSnapshot({
    summary: changes.summary ?? fixtureSummary({ running: 4, queued: 41, succeeded: 20 }, { recentFailures: 2 }),
    running: [
      fixtureTask({ progress: { epoch: 23, maxEpochs: 100, trainingLoss: 0.4123, cudaPeakReservedBytes: 1.5 * 1024 ** 3 } }),
      fixtureTask({ id: 'task-2', title: 'Inference study · Fold 3', link: null, progress: null, resources: null, owner: { ...fixtureTask().owner, key: 'owner-2', title: 'Inference study', projectId: 'elsewhere', sameWorkspace: false }, actions: { cancel: false, retry: false } }),
      fixtureTask({ id: 'task-4', kind: 'compute-job', labels: { computeKind: 'refit', recordId: 'r1' }, title: 'Refit · study3 · config 1 · seed 42 / split 42 · refit', owner: { ...fixtureTask().owner, key: 'owner-4', title: 'study3', projectId: 'study-project', projectName: 'study' }, stopRequest: 'cancel', link: '?project=study-project#experiments?experiment=e&tab=runs' }),
    ],
    owners: [
      fixtureOwner({ waitingReason: 'Waiting for a GPU slot (4/4)', counts: { running: 4, queued: 41, succeeded: 20, failed: 1 } }),
      fixtureOwner({ key: 'owner-2', id: 'inference', title: 'Inference study', projectId: 'elsewhere', sameWorkspace: false, held: true, position: 2, link: null, counts: { running: 1, queued: 3 } }),
      fixtureOwner({ key: 'owner-3', id: 'bulk', kind: 'evaluation-batch', purpose: 'inference', title: 'Unlabeled cohort', projectId: 'study-project', projectName: 'study', position: 3, link: '?project=study-project#apply?batch=bulk', counts: { queued: 3 } }),
    ],
  }));
  client.setQueryData(['task-center', 'capacity'], changes.capacity ?? fixtureCapacity());
  client.setQueryData(['model-experiments', 'project', 'summary'], { items: [ready, { ...ready, id: 'running-exp', status: 'running', stage: 'running' }] });
  client.setQueryData(['task-center', 'history', '?state=history&limit=15&offset=0'], { total: 2, offset: 0, limit: 15, groups: [
    fixtureHistoryGroup(),
    fixtureHistoryGroup({ owner: fixtureOwner({ key: 'owner-9', id: 'r1', kind: 'predictor-refit', title: 'Refit · seed 42', projectId: 'other-project', projectName: 'Study B', position: null, link: '?project=other-project#post-development?tab=refits&refit=r1', actions: { hold: false, release: false, stop: false, cancel: false, retry: false, moveUp: false, moveDown: false } }), finished: { total: 1, succeeded: 1, failed: 0, cancelled: 0, interrupted: 0 }, lastFailure: null }),
  ] });
  return client;
}
const render = (client: QueryClient) => renderToStaticMarkup(<QueryClientProvider client={client}><TaskCenter workspace={workspace} /></QueryClientProvider>);
const failing = (keys: string[][], error: Error) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, retryOnMount: false, staleTime: Infinity } } });
  clients.push(client);
  for (const key of keys) client.getQueryCache().build(client, { queryKey: key }).setState({ status: 'error', error });
  return renderToStaticMarkup(<QueryClientProvider client={client}><TaskCenter workspace={workspace} /></QueryClientProvider>);
};

describe('Task Center page', () => {
  it('summarizes the runner, queue, recent failures and time to empty', () => {
    const html = render(seeded());
    expect(html).toContain('<h1>Task Center</h1>');
    // The header names every kind of work the queue runs, not only model jobs.
    expect(html).toContain('Training folds and results, refits and predictors, predictor runs on cohorts, attention maps, feature extraction and validation, feature packing, and study archives from every project on this machine');
    expect(html).toContain('Pause queue');
    expect(html).toContain('Queue active');
    expect(html).toContain('4 running · 41 queued · 2 failed in the last 24 h · about 2 h 10 m left');
    expect(html).not.toContain('Start runner');
    expect(html).not.toContain('Restart runner');
  });

  it('shows per-GPU slots and memory, CPU and RAM meters, and the parallel GPU task setting', () => {
    const html = render(seeded());
    expect(html).toContain('GPU 0 · NVIDIA RTX A5000');
    expect(html).toContain('4 of 5 in use');
    expect(html).toContain('1 used by jobs outside the Task Center');
    expect(html).toContain('7.2 GiB of 24 GiB used');
    expect(html).toContain('9.2 GiB reserved by running tasks');
    expect(html).toContain('16 of 34 committed');
    expect(html).toContain('1 of 9 CPU task slots in use');
    expect(html).toContain('39 GiB of 189 GiB in use');
    expect(html).toContain('150 GiB available · 9.4 GiB kept free');
    expect(html).toMatch(/Parallel GPU tasks<input[^>]*value="4"/);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Apply<\/button>/);
  });

  it('offers the suggestion with its limiting factor and an expandable breakdown', () => {
    const html = render(seeded());
    expect(html).toContain('<strong>Suggested: 5</strong> · limited by throughput');
    expect(html).toContain('>Use suggestion<');
    expect(html).toContain('Based on 20 measured runs in 2 batches on this machine');
    expect(html).toContain('<summary>Why 5?</summary>');
    for (const text of ['GPU memory', '>10<', '>25<', '>14<', 'Limiting', '2.3 GiB per task', '4 threads reserved per task', 'Throughput was still rising', '16.8', 'GPU memory would allow 10, RAM 25, CPU 14.']) expect(html).toContain(text);
    expect(html).not.toContain('No blocking findings');
  });

  it('lists running tasks with progress, measured memory, kind labels and details links', () => {
    const html = render(seeded());
    const running = html.slice(html.indexOf('<h2>Running</h2>'), html.indexOf('<h2>Queue</h2>'));
    expect(running).toContain('href="#task-center?task=task-1"');
    expect(running).toContain('Epoch 23 / 100');
    expect(running).toContain('Training loss 0.4123');
    expect(running).toContain('4.4 GiB RAM');
    expect(running).toContain('1.5 GiB GPU memory (peak)');
    expect(running).toContain('2.3 GiB GPU memory reserved');
    expect(running).toContain('>Cancel<');
    expect(running).toContain('>Details<');
    expect(running).toContain('Inference study · Another workspace');
    // A refit reads as a refit, not "Compute job", without the repeated suffix, and names its project.
    expect(running).toContain('Refit training');
    expect(running).toContain('Refit · study3 · config 1 · seed 42 / split 42<');
    expect(running).toContain('study3 · Project study');
    expect(running).toContain('Cancel requested');
    expect(running.match(/>Cancel</g)).toHaveLength(1);
  });

  it('orders owners with counts, held state, the server waiting reason and allowed actions', () => {
    const html = render(seeded());
    const queue = html.slice(html.indexOf('<h2>Queue</h2>'), html.indexOf('<h2>Capacity</h2>'));
    expect(queue).toContain('href="#task-center?owner=owner-1"');
    expect(queue).toContain('href="#experiments?experiment=exp&amp;tab=runs"');
    expect(queue).toContain('4 running · 41 queued · 20 completed · 1 failed');
    expect(queue).toContain('>1 failed<');
    expect(queue).toContain('Waiting for a GPU slot (4/4)');
    expect(queue).toContain('About 2 h 10 m left');
    expect(queue).toContain('>Hold<');
    expect(queue).toContain('Stop &amp; hold');
    expect(queue).toContain('Show tasks');
    expect(queue).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Move Demo study to the top of the queue"/);
    expect(queue).toMatch(/<button(?![^>]*disabled)[^>]*aria-label="Move Demo study down"/);
    expect(queue).toContain('>Held<');
    expect(queue).toContain('Managed from its own workspace');
    expect(queue).not.toContain('Move Inference study');
    expect(queue).toContain('Batch · unlabeled cohort · Project study');
  });

  it('lists frozen setups of this project that have not started, linking to Experiments', () => {
    const html = render(seeded());
    const planned = html.slice(html.indexOf('<h2>Planned in this project</h2>'), html.indexOf('<h2>History</h2>'));
    expect(planned).toContain('Frozen baseline');
    expect(planned).toContain('1 batch · 1 training group, each over every fold');
    expect(planned).toContain('href="#experiments?experiment=ready-exp"');
    expect(planned).toContain('Open in Experiments');
    expect(planned.match(/Open in Experiments/g)).toHaveLength(1);
  });

  it('groups finished tasks by owner with results, the failure cause, retry and server filters', () => {
    const html = render(seeded());
    const history = html.slice(html.indexOf('<h2>History</h2>'));
    expect(history).toContain('All finished');
    expect(history).toContain('This project');
    expect(history).toContain('All kinds');
    expect(history).toContain('60 completed · 1 failed');
    expect(history).toContain('The project was busy');
    expect(history).toContain('safe to retry');
    expect(history).toContain('href="#task-center?task=task-9"');
    expect(history).toContain('>Retry failed<');
    expect(history).toContain('Refit · Project Study B');
    expect(history).toContain('href="?project=other-project#post-development?tab=refits&amp;refit=r1"');
  });

  it('explains a failure in plain words above the traceback, with whether retry is safe', () => {
    const html = renderToStaticMarkup(<FailureExplanation failure={{ title: 'The project was busy', cause: 'Another HistoPilot operation was changing this project.', advice: 'Retry once the other operation has finished.', detail: 'BlockingIOError: [Errno 11] Resource temporarily unavailable', retry: 'safe' }} error={'Traceback (most recent call last):\nBlockingIOError: [Errno 11]'} />);
    expect(html.indexOf('The project was busy')).toBeLessThan(html.indexOf('Traceback'));
    expect(html).toContain('Safe to retry');
    expect(html).toContain('<summary>Full error</summary>');
  });

  it('shows a task drawer with command, measurements, attempts, dependents and an accessible log', () => {
    const client = seeded();
    client.setQueryData(['task-center', 'task', 'task-1'], fixtureDetail({ attempts: [
      { attempt: 1, startedAt: '2026-09-27T08:00:00Z', endedAt: '2026-09-27T08:10:00Z', state: 'failed', exitReason: 'oom', gpu: 0 },
      { attempt: 2, startedAt: '2026-09-27T09:30:00Z', endedAt: null, state: 'running', exitReason: null, gpu: 0 },
    ], labels: { configurationNumber: 1, fold: 2, trainingSeed: 42, splitSeed: 7 } }));
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><TaskDetails id="task-1" onClose={() => {}} actions={idle} project="project" /></QueryClientProvider>);
    expect(html).toContain('role="dialog" aria-modal="true" aria-labelledby=');
    expect(html).toMatch(/aria-labelledby="([^"]+)".*<strong id="\1">Baseline · Config 1/);
    expect(html).toContain('histopilot.workers.managed_fold');
    expect(html).toContain('/projects/demo/training/batch/compute');
    expect(html).toContain('PYTHONUNBUFFERED=1');
    expect(html).toContain('Peak RAM 5.1 GiB');
    expect(html).toContain('Shared the GPU with 3.9 tasks on average');
    expect(html).toContain('Attempts (2)');
    expect(html).toContain('Out of memory');
    expect(html).toContain('Baseline · Final results');
    expect(html).toContain('<dt>Fold</dt><dd>2</dd>');
    expect(html).toContain('aria-label="Log of Baseline · Config 1 · Fold 1 · Train seed 42 · Split seed 7"');
    expect(html).toContain('Download log');
    expect(html).toContain('>Cancel task<');
  });

  it('offers to resume a paused queue and to start or restart the runner', () => {
    const paused = render(seeded({ summary: fixtureSummary({ queued: 2 }, { paused: true, runner: { ...fixtureSummary({}).runner, alive: false } }) }));
    expect(paused).toContain('Resume queue');
    expect(paused).toContain('Queue paused');
    expect(paused).toContain('Start runner');
    expect(paused).toContain('Queued tasks start once the runner is running.');
    const outdated = renderToStaticMarkup(<RunnerStrip summary={fixtureSummary({}, { runner: { ...fixtureSummary({}).runner, codeCurrent: false } })} actions={idle} />);
    expect(outdated).toContain('Restart runner');
    expect(outdated).toContain('running tasks continue');
  });

  it('keeps a failed start or a pending restart visible until the runner it describes runs', () => {
    const runner = fixtureSummary({}).runner;
    const failed = { runner: { ...runner, alive: false }, result: { started: false, reason: 'tmux unavailable' } };
    expect(runnerOutcomeNote(failed, { ...runner, alive: false })).toBe('The runner did not start: tmux unavailable.');
    expect(runnerOutcomeNote(failed, runner)).toBeNull();
    const pending = { runner: { ...runner, codeCurrent: false }, result: { started: false, pending: true, note: 'The old runner is finishing its current step; it restarts when that ends.' } };
    expect(runnerOutcomeNote(pending, { ...runner, codeCurrent: false })).toBe('The old runner is finishing its current step; it restarts when that ends.');
    expect(runnerOutcomeNote(pending, { ...runner, alive: false })).toBe('The old runner is finishing its current step; it restarts when that ends.');
    expect(runnerOutcomeNote(pending, runner)).toBeNull();
    expect(runnerOutcomeNote(null, { ...runner, alive: false })).toBeNull();
  });

  it('explains an unavailable or outdated service without inventing state', () => {
    expect(taskCenterErrorMessage(new ApiError('Not Found', 404))).toContain('does not include the Task Center yet');
    expect(taskCenterErrorMessage(new ApiError('Store locked.', 503, 'TASK_CENTER_UNAVAILABLE'))).toBe('The task store is unavailable. Store locked.');
    const html = failing([['task-center', 'snapshot']], new ApiError('Not Found', 404));
    expect(html).toContain('does not include the Task Center yet');
    expect(html).not.toContain('0 running');
  });

  it('points to the terminal instead of offering start or restart when automatic start is off', () => {
    const runner = fixtureSummary({}).runner;
    const stopped = renderToStaticMarkup(<RunnerStrip summary={fixtureSummary({ queued: 2 }, { runner: { ...runner, alive: false, autostart: false } })} actions={idle} />);
    expect(stopped).toContain('histopilot runner start');
    expect(stopped).toContain('Queued tasks start once the runner is running.');
    expect(stopped).not.toContain('Start runner');
    const outdated = renderToStaticMarkup(<RunnerStrip summary={fixtureSummary({}, { runner: { ...runner, codeCurrent: false, autostart: false } })} actions={idle} />);
    expect(outdated).toContain('Restart it from a terminal');
    expect(outdated).not.toContain('Restart runner');
  });

  it('lets GPUs with different limits be unified to any valid value', () => {
    const client = new QueryClient();
    clients.push(client);
    const mixed = fixtureCapacity({ settings: { ...fixtureCapacity().settings, gpuSlots: { '1': 2 } }, effective: { ...fixtureCapacity().effective, gpuSlots: { '0': 4, '1': 2 } } });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><ParallelTasks capacity={mixed} actions={idle} /></QueryClientProvider>);
    expect(html).toMatch(/Parallel GPU tasks<input[^>]*value="4"/);
    expect(html).toMatch(/<button(?![^>]*disabled)[^>]*>Apply<\/button>/);
    expect(html).toContain('GPUs currently use different limits');
    expect(html).toContain('>Use suggestion<');
  });

  it('does not invent empty lists or repeat the error when the snapshot cannot be read', () => {
    const html = failing([['task-center', 'snapshot'], ['task-center', 'capacity']], new ApiError('Store locked.', 503, 'TASK_CENTER_UNAVAILABLE'));
    expect(html).toContain('The task store is unavailable. Store locked.');
    expect(html.match(/Store locked\./g)).toHaveLength(1);
    expect(html).not.toContain('Nothing is running.');
    expect(html).not.toContain('The queue is empty.');
    expect(html).toContain('<h2>Running</h2>');
  });

  it('hides the queue panels on a service without the Task Center', () => {
    const html = failing([['task-center', 'snapshot']], new ApiError('Not Found', 404));
    expect(html).not.toContain('<h2>Running</h2>');
    expect(html).not.toContain('<h2>History</h2>');
    expect(html).toContain('<h2>Planned in this project</h2>');
  });

  it('labels a copy of this project in another workspace as foreign and never links its record', () => {
    const task = fixtureTask({ owner: { ...fixtureTask().owner, sameWorkspace: false } });
    const html = renderToStaticMarkup(<TaskName task={task} project="project" />);
    expect(html).toContain('Demo study · Another workspace');
    expect(html).not.toContain('#experiments');
    expect(html).toContain('href="#task-center?task=task-1"');
    const other = renderToStaticMarkup(<TaskName task={fixtureTask({ owner: { ...fixtureTask().owner, projectId: 'p2', projectName: 'Study B' } })} project="project" />);
    expect(other).toContain('Demo study · Project Study B');
  });

  it('keeps the suggestion quiet when it matches and explains hardware defaults', () => {
    const matching = renderToStaticMarkup(<SuggestionCallout suggestion={fixtureSuggestion()} current={5} busy={false} onUse={() => {}} />);
    expect(matching).toContain('Matches the current setting.');
    expect(matching).not.toContain('Use suggestion');
    const fallback = renderToStaticMarkup(<SuggestionCallout suggestion={fixtureSuggestion({ basis: 'hardware_only', binding: 'default', parallelGpuTasks: 3, throughput: { observed: [], trend: 'unknown', note: null }, evidence: { runs: 0, batches: 0, latest: null } })} current={4} busy={false} onUse={() => {}} />);
    expect(fallback).toContain('<strong>Suggested: 3</strong> · hardware default');
    expect(fallback).toContain('conservative default');
    expect(fallback).toContain('No measured runs');
  });

  it('reports why an owner waits from the server, else its first reasoned task or held state', () => {
    const pending = [fixtureTask({ id: 'a', state: 'queued', owner: { ...fixtureTask().owner, key: 'other' }, waitingReason: 'Waiting for RAM' }), fixtureTask({ id: 'b', state: 'queued', waitingReason: 'Waiting for a GPU slot (5/5)' })];
    expect(ownerWaitingReason(fixtureOwner({ waitingReason: 'Waiting for CPU threads' }), pending)).toBe('Waiting for CPU threads');
    expect(ownerWaitingReason(fixtureOwner(), pending)).toBe('Waiting for a GPU slot (5/5)');
    expect(ownerWaitingReason(fixtureOwner({ held: true }), [])).toContain('Held');
    expect(ownerWaitingReason(fixtureOwner({ counts: { blocked: 1 } }), [])).toBe('Waiting for earlier tasks to finish.');
    expect(ownerWaitingReason(fixtureOwner(), [])).toBeNull();
  });

  it('pushes a history entry only when a task opens, so Back closes the drawer', () => {
    expect(routeHistoryMode({}, { task: 't' })).toBe('push');
    expect(routeHistoryMode({ owner: 'o' }, { owner: 'o', task: 't' })).toBe('push');
    expect(routeHistoryMode({ task: 't' }, { task: 'u' })).toBe('push');
    expect(routeHistoryMode({ task: 't' }, { task: 't', state: 'failed' })).toBe('replace');
    expect(routeHistoryMode({ task: 't' }, {})).toBe('replace');
    expect(routeHistoryMode({}, { project: 'p', kind: 'mil-fold' })).toBe('replace');
    expect(drawerDepth({ taskCenterDrawer: 2 })).toBe(2);
    for (const state of [null, undefined, {}, { taskCenterDrawer: 0 }, { taskCenterDrawer: 'x' }, { taskCenterDrawer: 1.5 }]) expect(drawerDepth(state)).toBe(0);
  });

  it('keeps Tab inside the task drawer', () => {
    const items = ['close', 'retry', 'log'];
    expect(trappedFocus(items, 'log', false)).toBe('close');
    expect(trappedFocus(items, 'close', true)).toBe('log');
    expect(trappedFocus(items, 'retry', false)).toBeNull();
    expect(trappedFocus(items, null, false)).toBe('close');
    expect(trappedFocus(items, null, true)).toBe('log');
    expect(trappedFocus([], null, false)).toBeNull();
  });

  it('names an owner’s state as the stage chips do', () => {
    expect(ownerState(fixtureOwner({ counts: { cancelled: 3 } })).text).toBe('Cancelled');
    expect(ownerState(fixtureOwner({ counts: { succeeded: 29, cancelled: 1 } })).text).toBe('Cancelled');
    expect(ownerState(fixtureOwner({ counts: { succeeded: 3 } }))).toEqual({ text: 'Completed', tone: 'success' });
    expect(ownerState(fixtureOwner({ counts: { succeeded: 3, interrupted: 1, cancelled: 1 } })).text).toBe('Needs attention');
    expect(ownerState(fixtureOwner({ counts: { running: 1, failed: 1 } })).text).toBe('Running');
    expect(ownerState(fixtureOwner({ held: true, counts: { queued: 1 } })).text).toBe('Held');
    expect(ownerState(fixtureOwner({ counts: { blocked: 1 } })).text).toBe('Queued');
  });

  it('reads the history filters from the URL and shows a state its list does not offer', () => {
    expect(withCurrentOption([['', 'All']] as const, '')).toEqual([['', 'All']]);
    expect(withCurrentOption([['', 'All']] as const, 'failed', (value) => value.toUpperCase())).toEqual([['', 'All'], ['failed', 'FAILED']]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    clients.push(client);
    client.setQueryData(['task-center', 'history', '?state=failed%2Cinterrupted&project=project&kind=compute-job&limit=15&offset=0'], { total: 1, offset: 0, limit: 15, groups: [fixtureHistoryGroup()] });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><TaskHistory project="project" actions={idle} route={{ state: 'failed,interrupted', project: 'project', kind: 'compute-job' }} /></QueryClientProvider>);
    expect(html).toContain('<option value="failed,interrupted" selected="">Failed or Interrupted</option>');
    expect(html).toContain('<option value="project" selected="">This project</option>');
    expect(html).toContain('<option value="compute-job" selected="">Refits, evaluations and attention</option>');
    expect(html).toContain('60 completed · 1 failed');
  });
});
