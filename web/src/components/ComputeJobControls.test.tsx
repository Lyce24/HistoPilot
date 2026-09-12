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

  it('does not claim an unqueried job is ready to run', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    clients.push(client);
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><ComputeJobControls project="p" id="new" kind="evaluation" /></QueryClientProvider>);
    expect(html).toContain('Checking job status…');
    expect(html).not.toContain('Ready to run');
    expect(html).not.toContain('Run evaluation');
  });

  it('shows progress-read warnings while keeping active job cancellation available', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    clients.push(client);
    const initial: ComputeExecution = { status: 'running', progress: null, progressWarning: 'Progress file could not be read. The worker state is still available.' };
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><ComputeJobControls project="p" id="new" kind="evaluation" initial={initial} /></QueryClientProvider>);
    expect(html).toContain('Progress file could not be read');
    expect(html).toContain('Cancel job');
    expect(html).not.toContain('Epoch');
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
});
