import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import JobTray from './JobTray';

const clients: QueryClient[] = [];
function client() {
  const value = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(value);
  return value;
}
const render = (value: QueryClient) => renderToStaticMarkup(<QueryClientProvider client={value}><JobTray projectId="project" /></QueryClientProvider>);
const running = { id: 'refit', manifest: { name: 'Refit model' }, execution: { status: 'running' } };
afterEach(() => clients.splice(0).forEach((value) => value.clear()));

describe('compute job summary', () => {
  it('waits for every compute family before reporting completed counts', () => {
    const value = client();
    value.setQueryData(['development-batches', 'project'], { items: [], executionImplemented: true, executions: [] });
    expect(render(value)).toContain('Checking compute jobs…');
    expect(render(value)).not.toContain('0 fold runs completed');
    value.setQueryData(['refit-builds', 'project'], { items: [running] });
    expect(render(value)).toContain('At least 1 active job · checking remaining jobs…');
  });

  it('shows refit activity even when this service cannot launch training', () => {
    const value = client();
    value.setQueryData(['development-batches', 'project'], { items: [], executionImplemented: false, executions: [] });
    value.setQueryData(['refit-builds', 'project'], { items: [running] });
    value.setQueryData(['model-evaluations', 'project'], { items: [] });
    value.setQueryData(['interpretations', 'project'], { items: [] });
    expect(render(value)).toContain('1 active job · 0 fold runs completed');
    expect(render(value)).not.toContain('Execution unavailable');
  });

  it('retains known activity with an explicit stale status when a query fails', () => {
    const value = client();
    value.setQueryData(['refit-builds', 'project'], { items: [running] });
    value.getQueryCache().find({ queryKey: ['refit-builds', 'project'] })!.setState({ status: 'error', error: new Error('Connection lost') });
    expect(render(value)).toContain('1 active job · 0 fold runs completed · status may be outdated');
  });
});
