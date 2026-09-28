import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import RunStatusChip, { actionShown, describeRollup, rollupSettled, segmentText } from './RunStatusChip';
import { taskCenterKeys, type RollupScope } from '../api/taskCenter';
import { fixtureRollup } from '../testFixtures/taskCenter';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
const now = Date.parse('2026-09-27T15:00:00Z');
const text = (copy: ReturnType<typeof describeRollup>) => [copy.headline, ...copy.parts, copy.linkText].filter(Boolean).join(' · ');

describe('run status words', () => {
  it('says how many run, wait and failed, and when the queue empties', () => {
    expect(text(describeRollup(fixtureRollup(), { now }))).toBe('3 running · 12 queued · 1 failed · ~40 m left · View in Task Center →');
    const single = fixtureRollup({ live: 1, active: 1, pending: 0, counts: { running: 1 }, current: { taskId: 't', title: 'Refit', kind: 'compute-job', labels: {}, progress: { epoch: 3, maxEpochs: 8 }, startedAt: null }, eta: { seconds: 300, basis: 'measured' } });
    expect(text(describeRollup(single, { now }))).toBe('Running · epoch 3 / 8 · ~5 m left · View in Task Center →');
  });

  it('gives a queued run its place in line and the reason it waits', () => {
    const queued = fixtureRollup({ state: 'queued', active: 0, counts: { queued: 3 }, queuePosition: 4, waitingReason: 'GPU slots 5/5', eta: null });
    expect(text(describeRollup(queued, { now }))).toBe('Queued · #4 in line · waiting: GPU slots 5/5 · View in Task Center →');
    expect(describeRollup({ ...queued, paused: true }, { now }).headline).toBe('Queued · queue paused');
  });

  it('sends held work, a stopped runner and a requested cancel to the Task Center', () => {
    expect(text(describeRollup(fixtureRollup({ state: 'held' }), { now }))).toBe('Held in Task Center · Release there →');
    expect(text(describeRollup(fixtureRollup({ state: 'runner-stopped' }), { now }))).toBe('Queued, but the runner is stopped · Start it in Task Center →');
    expect(describeRollup(fixtureRollup({ state: 'stopping', stopRequest: 'cancel' }), { now }).headline).toBe('Cancel requested');
  });

  it('names a failure in plain words and points to its details', () => {
    const failed = fixtureRollup({ state: 'attention', live: 0, active: 0, pending: 0, counts: { succeeded: 29, failed: 1 }, progress: { completed: 29, total: 30 }, lastFailure: { taskId: 't', title: 'Fold 3', state: 'failed', reason: 'oom', message: 'Out of GPU memory', cause: '', retry: 'safe', at: null } });
    expect(text(describeRollup(failed, { now }))).toBe('Needs attention · out of GPU memory · 29 of 30 done · Details →');
    const interrupted = { ...failed, lastFailure: { ...failed.lastFailure!, state: 'interrupted' as const, message: 'Process lost' } };
    expect(text(describeRollup(interrupted, { now }))).toBe('Needs attention · process lost · 29 of 30 done · Details →');
    const generic = { ...failed, lastFailure: { ...failed.lastFailure!, state: 'interrupted' as const, message: 'The task failed' } };
    expect(describeRollup(generic, { now }).parts[0]).toBe('interrupted');
    const several = { ...failed, counts: { succeeded: 27, failed: 3 }, lastFailure: { ...failed.lastFailure!, message: 'The task failed' } };
    expect(text(describeRollup(several, { now }))).toBe('Needs attention · 29 of 30 done · 3 failed · Details →');
  });

  it('says when finished work completed and how long it took', () => {
    const done = fixtureRollup({ state: 'completed', live: 0, active: 0, pending: 0, counts: { succeeded: 30 }, startedAt: '2026-09-27T13:50:00Z', finishedAt: '2026-09-27T14:02:00Z', eta: null });
    const copy = describeRollup(done, { now });
    expect(copy.headline).toMatch(/^Completed \d{1,2}:\d{2}/);
    expect(copy.parts).toContain('took 12 m');
    expect(text(describeRollup(fixtureRollup({ state: 'not-started', counts: {}, byKind: {} }), { now }))).toBe('Not started');
    expect(text(describeRollup(undefined, { unavailable: true }))).toBe('Run status unavailable · Open Task Center →');
  });

  it('adds per-kind progress, marking finished segments', () => {
    const rollup = fixtureRollup({ byKind: { 'mil-fold': { counts: {}, completed: 30, total: 30 }, 'mil-collect': { counts: {}, completed: 2, total: 2 }, 'compute-job': { counts: {}, completed: 4, total: 12 } } });
    expect(segmentText(rollup, [{ label: 'Training', kinds: ['mil-fold', 'mil-collect'] }, { label: 'Predictors', kinds: ['compute-job'] }, { label: 'Nothing', kinds: ['absent'] }]))
      .toEqual(['Training 32/32 ✓', 'Predictors 4/12']);
  });
});

describe('RunStatusChip', () => {
  const render = (scope: RollupScope, data: ReturnType<typeof fixtureRollup>, props: Partial<Parameters<typeof RunStatusChip>[0]> = {}) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    clients.push(client);
    client.setQueryData(taskCenterKeys.rollup(scope), data);
    return renderToStaticMarkup(<QueryClientProvider client={client}><RunStatusChip scope={scope} {...props} /></QueryClientProvider>);
  };
  const scope = { ownerKind: 'experiment', ownerId: 'exp', project: 'project' };

  it('links to the filtered Task Center view and offers the page action only when it applies', () => {
    const running = render(scope, fixtureRollup(), { label: 'Training', primaryAction: <button type="button">Resume</button> });
    expect(running).toContain('href="#task-center?owner=owner-1&amp;project=project"');
    expect(running).toContain('>Training<');
    expect(running).not.toContain('Resume');
    const failed = render(scope, fixtureRollup({ state: 'attention', lastFailure: null }), { variant: 'row', primaryAction: <button type="button">Resume</button> });
    expect(failed).toContain('Resume');
    expect(failed).toContain('run-status-row');
    expect(failed).toContain('data-state="attention"');
  });

  it('offers Resume only when a retry would resume something', () => {
    const attention = fixtureRollup({ state: 'attention', live: 0, active: 0, pending: 0 });
    expect(actionShown(attention)).toBe(true);
    expect(actionShown({ ...attention, retryable: false })).toBe(false);
    expect(actionShown({ ...attention, state: 'cancelled', retryable: false })).toBe(false);
    expect(actionShown({ ...attention, retryable: undefined })).toBe(true);
    expect(actionShown({ ...attention, state: 'not-started', retryable: false })).toBe(true);
    expect(actionShown({ ...attention, state: 'running' })).toBe(false);
    expect(actionShown(undefined)).toBe(true);
    expect(render(scope, { ...attention, retryable: false }, { primaryAction: <button type="button">Resume</button> })).not.toContain('Resume');
  });

  it('settles when live work ends, not on a first read', () => {
    const running = fixtureRollup();
    const done = fixtureRollup({ state: 'completed', live: 0, active: 0, pending: 0 });
    expect(rollupSettled(running, done)).toBe(true);
    expect(rollupSettled(undefined, done)).toBe(false);
    expect(rollupSettled(done, done)).toBe(false);
    expect(rollupSettled(fixtureRollup({ state: 'queued' }), fixtureRollup({ state: 'attention' }))).toBe(true);
  });

  it('can stay hidden until work exists, or say it is ready', () => {
    const empty = fixtureRollup({ state: 'not-started', counts: {}, byKind: {} });
    expect(render(scope, empty, { hideWhenNotStarted: true })).toBe('');
    expect(render(scope, empty, { notStartedText: 'Ready to run' })).toContain('Ready to run');
  });
});
