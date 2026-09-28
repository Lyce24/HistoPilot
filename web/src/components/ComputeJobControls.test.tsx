import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider, type QueryObserverOptions } from '@tanstack/react-query';
import ComputeJobControls, { computeExecutionQuery, recordBehindRollup } from './ComputeJobControls';
import type { ComputeExecution } from '../api/predictors';
import { taskCenterKeys } from '../api/taskCenter';
import { fixtureRollup } from '../testFixtures/taskCenter';
import { legacyRecordNote } from './LegacyRecordNote';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function render(initial: ComputeExecution, { kind = 'refit', readOnly = false, rollup, inference = false }: { kind?: 'refit' | 'evaluation' | 'interpretation'; readOnly?: boolean; rollup?: ReturnType<typeof fixtureRollup>; inference?: boolean } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } });
  clients.push(client);
  if (rollup) client.setQueryData(taskCenterKeys.rollup({ recordKind: kind, recordId: 'job', project: 'p' }), rollup);
  return renderToStaticMarkup(<QueryClientProvider client={client}><ComputeJobControls project="p" id="job" kind={kind} initial={initial} readOnly={readOnly} inference={inference} /></QueryClientProvider>);
}
const single = (changes: Parameters<typeof fixtureRollup>[0] = {}) => fixtureRollup({ scope: { recordKind: 'refit', recordId: 'job', project: 'p' }, counts: { running: 1 }, live: 1, active: 1, pending: 0, progress: { completed: 0, total: 1 }, byKind: {}, href: '#task-center?owner=o&task=t&project=p', current: { taskId: 't', title: 'Refit', kind: 'compute-job', labels: {}, progress: { epoch: 3, maxEpochs: 8 }, startedAt: null }, eta: null, ...changes });

describe('compute job science action and run status', () => {
  it('offers launch only for a saved, unlaunched active record', () => {
    expect(render({ status: 'not_started' })).toContain('Train refit model');
    expect(render({ status: 'not_started' }, { kind: 'evaluation' })).toContain('Run evaluation');
    expect(render({ status: 'not_started' }, { kind: 'evaluation', inference: true })).toContain('Run inference');
    expect(render({ status: 'not_started' }, { kind: 'interpretation' })).toContain('Compute slide attention');
    expect(render({ status: 'not_started' }, { kind: 'interpretation', readOnly: true })).not.toContain('Compute slide attention');
    expect(render({ status: 'completed', executor: 'task-center' })).not.toContain('Train refit model');
  });

  it('shows a Task Center job as one status chip with a deep link, and no cancel, log or tmux hint', () => {
    const html = render({ status: 'running', executor: 'task-center', sessionName: 'tc-abc', logPath: '/job/worker.log', progress: { epoch: 3, maxEpochs: 8, trainingLoss: 0.2 } }, { rollup: single() });
    expect(html).toContain('>Running<');
    expect(html).toContain('>epoch 3 / 8<');
    expect(html).toContain('href="#task-center?owner=o&amp;task=t&amp;project=p"');
    for (const text of ['Cancel job', '/job/worker.log', 'tmux attach', 'Job log']) expect(html).not.toContain(text);
  });

  it('offers the one retry when the job stopped short, next to the failure', () => {
    const html = render({ status: 'failed', executor: 'task-center' }, { rollup: single({ state: 'attention', live: 0, active: 0, current: null, lastFailure: { taskId: 't', title: 'Refit', state: 'failed', reason: 'oom', message: 'Out of GPU memory', cause: '', retry: 'safe', at: null } }) });
    expect(html).toContain('>Needs attention<');
    expect(html).toContain('>out of GPU memory<');
    expect(html).toContain('>Resume<');
    expect(render({ status: 'interrupted', executor: 'task-center' }, { kind: 'evaluation', rollup: single({ state: 'attention', live: 0, active: 0 }) })).toContain('>Resume<');
    expect(render({ status: 'cancelled', executor: 'task-center' }, { kind: 'interpretation', rollup: single({ state: 'cancelled', live: 0, active: 0 }) })).toContain('>Resume<');
    expect(render({ status: 'failed', executor: 'task-center' }, { readOnly: true, rollup: single({ state: 'attention' }) })).not.toContain('>Resume<');
    // A retry that would resume nothing is not offered.
    expect(render({ status: 'failed', executor: 'task-center' }, { rollup: single({ state: 'attention', live: 0, active: 0, retryable: false }) })).not.toContain('>Resume<');
  });

  it('follows jobs through the task store instead of polling the record', () => {
    render({ status: 'running', executor: 'task-center' }, { rollup: single() });
    const managed = clients.at(-1)?.getQueryCache().find({ queryKey: ['compute-job', 'p', 'refit', 'job'] })?.options as QueryObserverOptions | undefined;
    expect(managed?.refetchInterval).toBeUndefined();
    expect(computeExecutionQuery('p', 'refit', 'job', { status: 'running', executor: 'tmux' }, true)).not.toHaveProperty('refetchInterval');
  });

  it('reads the record again on mount: what a list handed in may predate a Task Center change', () => {
    const options = computeExecutionQuery('p', 'evaluation', 'job', { status: 'running', executor: 'task-center' }, true);
    expect(options.queryKey).toEqual(['compute-job', 'p', 'evaluation', 'job']);
    expect(options.initialDataUpdatedAt).toBe(0);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 10000 } } });
    clients.push(client);
    renderToStaticMarkup(<QueryClientProvider client={client}><ComputeJobControls project="p" id="job" kind="evaluation" initial={{ status: 'running', executor: 'task-center' }} /></QueryClientProvider>);
    const query = client.getQueryCache().find({ queryKey: ['compute-job', 'p', 'evaluation', 'job'] });
    expect(query?.state.dataUpdatedAt).toBe(0);
    expect(query?.isStaleByTime(10000)).toBe(true);
  });

  it('re-reads a record the task store already finished, failed or cancelled', () => {
    const running = { status: 'running', executor: 'task-center' } as const;
    const done = single({ state: 'completed', live: 0, active: 0 });
    expect(recordBehindRollup(done, running)).toBe(true);
    expect(recordBehindRollup(single({ state: 'attention', live: 0, active: 0 }), { ...running, status: 'queued' })).toBe(true);
    expect(recordBehindRollup(single({ state: 'cancelled', live: 0, active: 0 }), running)).toBe(true);
    // Still live, not in the store yet, already up to date, or created before the Task Center: nothing to do.
    expect(recordBehindRollup(single(), running)).toBe(false);
    expect(recordBehindRollup(single({ state: 'not-started' }), running)).toBe(false);
    expect(recordBehindRollup(done, { status: 'completed', executor: 'task-center' })).toBe(false);
    expect(recordBehindRollup(done, { status: 'running', executor: 'tmux' })).toBe(false);
    expect(recordBehindRollup(undefined, running)).toBe(false);
  });

  it('uses historical status for inactive stopped records while reading active ones', () => {
    render({ status: 'completed' }, { readOnly: true });
    const inactive = clients.at(-1)?.getQueryCache().find({ queryKey: ['compute-job', 'p', 'refit', 'job'] })?.options as QueryObserverOptions | undefined;
    expect(inactive?.enabled).toBe(false);
    render({ status: 'running' }, { readOnly: true });
    const active = clients.at(-1)?.getQueryCache().find({ queryKey: ['compute-job', 'p', 'refit', 'job'] })?.options as QueryObserverOptions | undefined;
    expect(active?.enabled).toBe(true);
  });

  it('does not claim an unqueried job is ready to run', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    clients.push(client);
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><ComputeJobControls project="p" id="new" kind="evaluation" /></QueryClientProvider>);
    expect(html).toContain('Checking job status…');
    expect(html).not.toContain('Run evaluation');
  });

  it('offers status recovery and distinguishes cached state after a failed refresh', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    clients.push(client);
    const key = ['compute-job', 'p', 'evaluation', 'new'];
    client.setQueryData(key, { status: 'not_started' });
    client.getQueryCache().find({ queryKey: key })!.setState({ status: 'error', error: new Error('Disconnected') });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><ComputeJobControls project="p" id="new" kind="evaluation" /></QueryClientProvider>);
    expect(html).toContain('Retry job status');
    expect(html).toContain('Showing the last loaded job status');
    expect(html).toMatch(/disabled="">Run evaluation/);
  });

  it('shows a job created before the Task Center read-only: saved status and a note, no actions or session', () => {
    const legacy = render({ status: 'interrupted', executor: 'tmux', sessionName: 'hp-refit', logPath: '/job/worker.log', progress: { epoch: 2, maxEpochs: 8 }, error: 'Created before the Task Center; it did not finish.' });
    expect(legacy).toContain('>Interrupted<');
    expect(legacy).toContain('Epoch 2 / 8');
    expect(legacy).toContain('Created before the Task Center; it did not finish.');
    expect(legacy).toContain(legacyRecordNote);
    expect(legacy).toContain('/job/worker.log');
    for (const text of ['hp-refit', 'tmux attach', 'Cancel job', '>Resume<', 'Restore this record']) expect(legacy).not.toContain(text);
    // A record without an executor also predates the Task Center; a finished one keeps its status.
    const finished = render({ status: 'completed', sessionName: 'hp-refit' });
    expect(finished).toContain('>Completed<');
    expect(finished).toContain(legacyRecordNote);
    // A saved record never launched is not one: it can still be run.
    expect(render({ status: 'not_started' })).not.toContain(legacyRecordNote);
  });

  it('shows a progress-read warning with the status', () => {
    const html = render({ status: 'running', executor: 'task-center', progressWarning: 'Progress file could not be read. The worker state is still available.' }, { kind: 'evaluation', rollup: single() });
    expect(html).toContain('Progress file could not be read');
  });
});
