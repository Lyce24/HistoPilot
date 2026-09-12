import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider, type QueryObserverOptions } from '@tanstack/react-query';
import ComputeJobControls from './ComputeJobControls';
import type { ComputeExecution } from '../api/predictors';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function render(status: ComputeExecution['status'], kind: 'refit' | 'evaluation' | 'interpretation' = 'refit', readOnly = false, cancelling = false) {
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } });
  clients.push(client);
  return renderToStaticMarkup(<QueryClientProvider client={client}><ComputeJobControls project="p" id="job" kind={kind} initial={{ status, cancellationRequested: cancelling }} readOnly={readOnly} /></QueryClientProvider>);
}
describe('predictor and evaluation job controls', () => {
  it('offers launch only for a saved, unlaunched active record', () => {
    expect(render('not_started')).toContain('Train refit model');
    expect(render('not_started', 'evaluation')).toContain('Run evaluation');
    expect(render('not_started', 'interpretation')).toContain('Compute slide attention');
    expect(render('not_started', 'interpretation', true)).not.toContain('Compute slide attention');
    expect(render('not_started', 'refit', true)).not.toContain('Train refit model');
    expect(render('completed')).not.toContain('Train refit model');
  });
  it('keeps cancellation visible for an active job and waits for stop confirmation', () => {
    expect(render('running', 'refit', true)).toContain('Cancel job');
    expect(render('queued', 'evaluation', false, true)).toContain('Cancellation requested');
    expect(render('cancelled')).toContain('Resume refit training');
    expect(render('interrupted', 'evaluation')).toContain('Retry evaluation');
    expect(render('interrupted', 'interpretation')).toContain('Retry attention computation');
  });
  it('uses historical status for inactive stopped records while continuing active-job polling', () => {
    render('completed', 'refit', true);
    const inactive = clients.at(-1)?.getQueryCache().find({ queryKey: ['compute-job', 'p', 'refit', 'job'] })?.options as QueryObserverOptions | undefined;
    expect(inactive?.enabled).toBe(false);
    render('running', 'refit', true);
    const active = clients.at(-1)?.getQueryCache().find({ queryKey: ['compute-job', 'p', 'refit', 'job'] })?.options as QueryObserverOptions | undefined;
    expect(active?.enabled).toBe(true);
  });
});
