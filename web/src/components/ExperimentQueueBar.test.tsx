import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { taskCenterKeys, type RollupScope } from '../api/taskCenter';
import ExperimentQueueBar, { experimentNeedsResume, experimentRollupScope } from './ExperimentQueueBar';
import { fixtureOwner, fixtureRollup } from '../testFixtures/taskCenter';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function render(rollup: ReturnType<typeof fixtureRollup> | null, { ownerKey, managed = true, batchIds }: { ownerKey?: string; managed?: boolean; batchIds?: string[] } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, retryOnMount: false, staleTime: Infinity } } });
  clients.push(client);
  const scope: RollupScope = { ...(ownerKey ? { owner: ownerKey } : { ownerKind: 'experiment', ownerId: 'exp', project: 'project' }), ...(batchIds?.length ? { batchIds: [...batchIds].sort().join(',') } : {}) };
  if (rollup) client.setQueryData(taskCenterKeys.rollup(scope), rollup);
  return renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentQueueBar project="project" experimentId="exp" ownerKey={ownerKey} managed={managed} batchIds={batchIds} /></QueryClientProvider>);
}

describe('experiment run status', () => {
  it('shows training and predictor progress, failures and time left with a deep link, and no queue controls', () => {
    const html = render(fixtureRollup({ byKind: { 'mil-fold': { counts: {}, completed: 30, total: 30 }, 'predictor-coordinator': { counts: {}, completed: 0, total: 1 }, 'compute-job': { counts: {}, completed: 4, total: 11 } } }));
    expect(html).toContain('Training 30/30 ✓');
    expect(html).toContain('Predictor jobs 4/12');
    expect(html).toContain('~40 m left');
    expect(html).toContain('href="#task-center?owner=owner-1&amp;project=project"');
    for (const label of ['>Top<', '>Up<', '>Hold<', '>Cancel<', 'Stop &amp; hold', 'Resume']) expect(html).not.toContain(label);
  });

  it('offers one Resume when the experiment needs attention', () => {
    const html = render(fixtureRollup({ state: 'attention', live: 0, active: 0, pending: 0, lastFailure: { taskId: 't', title: 'Predictors · Study', state: 'failed', reason: 'error', message: 'The project was busy', cause: '', retry: 'safe', at: null } }), { ownerKey: 'owner-1' });
    expect(html).toContain('>Needs attention<');
    expect(html).toContain('>the project was busy<');
    expect(html.match(/>Resume</g)).toHaveLength(1);
    expect(html).toContain('Details →');
    expect(experimentNeedsResume({ state: 'cancelled' })).toBe(true);
    expect(experimentNeedsResume({ state: 'running' })).toBe(false);
  });

  it('offers no Resume when the owner retry would resume nothing', () => {
    const html = render(fixtureRollup({ state: 'attention', live: 0, active: 0, pending: 0, retryable: false }), { ownerKey: 'owner-1' });
    expect(html).toContain('>Needs attention<');
    expect(html).not.toContain('>Resume<');
    expect(experimentNeedsResume({ state: 'attention', retryable: false })).toBe(false);
    // Older services do not say; the button stays.
    expect(experimentNeedsResume({ state: 'attention' })).toBe(true);
  });

  it('counts only the batches the experiment keeps (trashed or replaced batches keep their tasks)', () => {
    expect(experimentRollupScope('project', 'exp', 'owner-1', ['b', 'a', 'b'])).toEqual({ owner: 'owner-1', batchIds: 'a,b' });
    expect(experimentRollupScope('project', 'exp', null, [])).toEqual({ ownerKind: 'experiment', ownerId: 'exp', project: 'project' });
    const html = render(fixtureRollup({ state: 'completed', live: 0, active: 0, pending: 0 }), { ownerKey: 'owner-1', batchIds: ['kept', 'current'] });
    expect(html).toContain('data-state="completed"');
  });

  it('says so when nothing is queued yet, and stays hidden for experiments outside the Task Center', () => {
    expect(render(fixtureRollup({ state: 'not-started', counts: {}, byKind: {} }))).toContain('Nothing from this experiment is queued yet');
    expect(render(fixtureRollup({ state: 'not-started', counts: {}, byKind: {} }), { managed: false })).toBe('');
    expect(render(fixtureRollup(), { managed: false })).toContain('3 running');
  });
});
