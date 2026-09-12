import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { SystemStatus } from '../api/types';
import System from './System';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function render(workers: SystemStatus['workers']) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['system'], { mode: 'local', workspace: '/projects', storage: { engine: 'SQLite', journalMode: 'wal', schemaVersion: 1 }, control: { process: 'Control service', cudaModelsLoaded: false }, sourcesReadOnly: true, workers } satisfies SystemStatus);
  return renderToStaticMarkup(<QueryClientProvider client={client}><System /></QueryClientProvider>);
}

describe('system capability reporting', () => {
  it('distinguishes missing extraction runtime from implemented native execution', () => {
    const html = render({ executionEnabled: false, nativeExecutionImplemented: true, tmuxAvailable: true, status: 'TRIDENT runtime unavailable.' });
    expect(html).toContain('Native execution implemented');
    expect(html).toContain('TRIDENT feature extraction');
    expect(html).toContain('Extraction runtime setup required');
    expect(html).toContain('tmux available');
    expect(html).toContain('href="#experiments"');
    expect(html).toContain('href="#features"');
    expect(html).not.toContain('future isolated workers');
    expect(html).not.toContain('Planned');
    expect(html).not.toContain('DuckDB');
  });

  it('reports extraction readiness and unavailable persistence separately', () => {
    const html = render({ executionEnabled: true, nativeExecutionImplemented: true, tmuxAvailable: false, status: 'Review module runtimes.' });
    expect(html).toContain('Runtime ready');
    expect(html).toContain('tmux unavailable');
    expect(html).toContain('Feature tensors &amp; coordinates');
    expect(html).toContain('Dataset and analytical tables');
  });
});
