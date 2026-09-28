import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalOperations from './LocalOperations';
import type { Workspace } from '../api/types';
import type { OperationsInventory } from '../api/operations';
import { extractionActive } from '../api/trident';
import { featurePackActive } from '../api/packing';
import type { ExtractionJob } from '../api/trident';
import type { FeaturePackJob } from '../api/packing';
import { legacyRecordNote } from '../components/LegacyRecordNote';

const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.restoreAllMocks(); });
function render(inventory: OperationsInventory, failed = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['operations', 'project'], inventory);
  client.setQueryData(['operation-sources', 'project'], { sources: [{ id: 'source', name: 'Moved slides', path: '/missing/slides', role: 'slides', permitted: true, available: false }], referenceCount: 1, missingReferences: [{ path: '/missing/slides/A.svs', reason: 'missing' }] });
  client.setQueryData(['operation-archives', 'project'], { jobs: failed ? [{ id: 'archive', action: 'verify', status: 'failed', createdAt: '2026-09-16T10:00:00Z', sessionName: 'archive-session', logPath: '/logs/archive.log', error: 'Checksum verification failed.', result: null }] : [] });
  return renderToStaticMarkup(<QueryClientProvider client={client}><LocalOperations workspace={{ project: { id: 'project' } } as Workspace} /></QueryClientProvider>);
}
const empty: OperationsInventory = { projectId: 'project', jobs: [], reservations: [], capacity: { cpus: 8, availableRamGb: 16 }, note: '' };

describe('project operations workspace', () => {
  it('keeps waiting preparation jobs active for cancellation and polling', () => {
    expect(extractionActive({ state: 'queued' } as ExtractionJob)).toBe(true);
    expect(featurePackActive({ state: 'queued' } as FeaturePackJob)).toBe(true);
  });
  it('points compute work to the Task Center and keeps export gated on active project jobs', () => {
    const html = render({ ...empty, jobs: [{ key: 'extraction:x', id: 'x', kind: 'extraction', name: 'Slide embeddings', job: { status: 'running', cancellable: true } }] });
    expect(html).toContain('Study backups &amp; sources');
    expect(html).toContain('href="#task-center?project=project"');
    expect(html).toContain('Export waits for 1 active project job to finish');
    expect(html).not.toContain('Unified job queue');
    expect(html).not.toContain('Cancel job');
    expect(html).not.toContain('Host reservations across all projects');
    const idle = render(empty);
    expect(idle).not.toContain('Export waits for');
  });
  it('explains external source policy and preserves honest failed verification status', () => {
    const html = render(empty, true);
    expect(html).toContain('External slides, feature folders and outputs remain external references');
    expect(html).toContain('Checksum verification failed.');
    expect(html).not.toContain('Checksums verified');
    expect(html).toContain('/missing/slides/A.svs');
    expect(html).toContain('existing frozen versions keep their recorded paths');
  });
  it('shows an archive operation created before the Task Center read-only, without its session', () => {
    const html = render(empty, true);
    expect(html).toContain(legacyRecordNote);
    expect(html).toContain('/logs/archive.log');
    for (const text of ['archive-session', 'tmux attach', 'Reconnect', 'Retry saved operation', 'Cancel archive operation', '>Resume<']) expect(html).not.toContain(text);
  });
});
