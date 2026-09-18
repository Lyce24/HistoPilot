import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalOperations, { cancelPipelineJob } from './LocalOperations';
import { lifecycle } from '../api/lifecycle';
import type { Workspace } from '../api/types';
import type { OperationsInventory } from '../api/operations';
import { extractionActive } from '../api/trident';
import { featurePackActive } from '../api/packing';
import type { ExtractionJob } from '../api/trident';
import type { FeaturePackJob } from '../api/packing';

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
  it('uses a fresh cancellation receipt after acknowledgement for a resumed job', async () => {
    const cancel = vi.spyOn(lifecycle, 'cancel').mockResolvedValue({ key: 'configuration:job', job: {} });
    const receipts = new Map<string, string>();
    await cancelPipelineJob('project', 'configuration:job', receipts);
    await cancelPipelineJob('project', 'configuration:job', receipts);
    expect(cancel.mock.calls[0][2]).not.toEqual(cancel.mock.calls[1][2]);
    expect(receipts.size).toBe(0);
  });
  it('retries an uncertain cancellation with its existing receipt', async () => {
    const cancel = vi.spyOn(lifecycle, 'cancel').mockRejectedValueOnce(new Error('Connection lost')).mockResolvedValueOnce({ key: 'configuration:job', job: {} });
    const receipts = new Map<string, string>();
    await expect(cancelPipelineJob('project', 'configuration:job', receipts)).rejects.toThrow('Connection lost');
    await cancelPipelineJob('project', 'configuration:job', receipts);
    expect(cancel.mock.calls[0][2]).toEqual(cancel.mock.calls[1][2]);
    expect(receipts.size).toBe(0);
  });
  it('keeps waiting preparation jobs active for cancellation and polling', () => {
    expect(extractionActive({ state: 'queued' } as ExtractionJob)).toBe(true);
    expect(featurePackActive({ state: 'queued' } as FeaturePackJob)).toBe(true);
  });
  it('shows unified jobs, cancellation and reservation context', () => {
    const html = render({ ...empty, jobs: [{ key: 'extraction:x', id: 'x', kind: 'extraction', name: 'Slide embeddings', job: { status: 'queued', cancellable: true, waitingReason: 'Waiting for GPU capacity.' } }], reservations: [{ batchId: 'batch', runId: 'run', cpus: 4, ramGb: 8, gpu: 0, runsPerGpu: 1 }] });
    expect(html).toContain('Slide embeddings');
    expect(html).toContain('Waiting for GPU capacity.');
    expect(html).toContain('Cancel job');
    expect(html).toContain('Host reservations across all projects');
    expect(html).toContain('4 CPU slots');
  });
  it('explains external source policy and preserves honest failed verification status', () => {
    const html = render(empty, true);
    expect(html).toContain('External slides, feature folders and outputs remain external references');
    expect(html).toContain('Checksum verification failed.');
    expect(html).not.toContain('Checksums verified');
    expect(html).toContain('/missing/slides/A.svs');
    expect(html).toContain('existing frozen versions keep their recorded paths');
    expect(html).toContain('tmux attach -t archive-session');
  });
});
