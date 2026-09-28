import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import JobTray, { JobTrayLinks, jobTrayStatus, liveTaskCount, taskJobLink } from './JobTray';
import { taskCenterKeys } from '../api/taskCenter';
import { fixtureRollup, fixtureTask } from '../testFixtures/taskCenter';

const clients: QueryClient[] = [];
function client() {
  const value = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(value);
  return value;
}
const render = (value: QueryClient) => renderToStaticMarkup(<QueryClientProvider client={value}><JobTray /></QueryClientProvider>);
afterEach(() => clients.splice(0).forEach((value) => value.clear()));
const machineKey = taskCenterKeys.rollup({});

const machine = (changes: Parameters<typeof fixtureRollup>[0] = {}) => fixtureRollup({ scope: {}, progress: null, recentFailures: 0, ownerKey: null, href: '#task-center', ...changes });

describe('Task Center summary tray', () => {
  it('reports running, queued and failed tasks and the time left from the machine rollup', () => {
    const value = client();
    value.setQueryData(machineKey, machine({ counts: { running: 3, starting: 1, queued: 40, blocked: 1, succeeded: 12 }, recentFailures: 1 }));
    const html = render(value);
    expect(html).toContain('4 running · 41 queued · 1 failed · ~40 m left');
    expect(html).not.toContain('is-idle');
    // The status line is not a live region; only state changes are announced.
    expect(html).not.toMatch(/<span role="status">4 running/);
  });

  it('recedes when nothing is active and reads no extraction records of its own', () => {
    const value = client();
    value.setQueryData(machineKey, machine({ state: 'completed', live: 0, active: 0, pending: 0, counts: { succeeded: 3 }, eta: null }));
    expect(render(value)).toContain('is-idle');
    expect(render(value)).toContain('0 running · 0 queued');
    expect(value.getQueryCache().findAll({ queryKey: ['extractions'] })).toHaveLength(0);
  });

  it('names a paused queue and a stopped runner only when they hold work back', () => {
    expect(jobTrayStatus(machine({ counts: { queued: 2 }, paused: true, eta: null }))).toBe('0 running · 2 queued · queue paused');
    expect(jobTrayStatus(machine({ counts: { queued: 2 }, runnerAlive: false, eta: null }))).toBe('0 running · 2 queued · runner stopped');
    expect(jobTrayStatus(machine({ counts: {}, runnerAlive: false, paused: true, eta: null }))).toBe('0 running · 0 queued');
  });

  it('waits for the rollup before reporting counts', () => {
    const value = client();
    expect(render(value)).toContain('Checking tasks…');
    expect(render(value)).not.toContain('0 running');
  });

  it('keeps the last known counts with an explicit stale status when the rollup fails', () => {
    const value = client();
    value.setQueryData(machineKey, machine({ counts: { running: 1 }, eta: null }));
    value.getQueryCache().find({ queryKey: machineKey })!.setState({ status: 'error', error: new Error('Connection lost') });
    expect(render(value)).toContain('1 running · 0 queued · status may be outdated');
    const fresh = new QueryClient({ defaultOptions: { queries: { retry: false, retryOnMount: false, staleTime: Infinity } } });
    clients.push(fresh);
    fresh.getQueryCache().build(fresh, { queryKey: machineKey }).setState({ status: 'error', error: new Error('Connection lost') });
    expect(render(fresh)).toContain('Task status unavailable');
  });

  it('counts only tasks the live list can show when saying how many more there are', () => {
    // live = 15 also counts a concluded task awaiting its automatic resume.
    expect(liveTaskCount(fixtureRollup({ counts: { running: 3, queued: 11, blocked: 1, failed: 2, succeeded: 9 }, live: 16 }))).toBe(15);
    expect(liveTaskCount(undefined)).toBe(0);
  });

  it('opens every task in the Task Center, not its record', () => {
    expect(taskJobLink(fixtureTask()).link).toBe('#task-center?task=task-1');
    expect(taskJobLink(fixtureTask()).detail).toBe('Epoch 23 / 100');
    expect(taskJobLink(fixtureTask({ link: null, state: 'queued', progress: null, waitingReason: 'Waiting for a GPU slot (4/4)' }))).toMatchObject({ detail: 'Waiting for a GPU slot (4/4)', status: 'Queued' });
  });

  it('lists each live task as a link to its Task Center detail', () => {
    const html = renderToStaticMarkup(<JobTrayLinks jobs={[taskJobLink(fixtureTask())]} />);
    expect(html).toContain('href="#task-center?task=task-1"');
    expect(html).toContain('Epoch 23 / 100');
  });
});
