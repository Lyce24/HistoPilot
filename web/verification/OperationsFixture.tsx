/** Offline operations UI fixture with in-memory receipts; no worker/server starts. */
import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalOperations from '../src/pages/LocalOperations';
import type { Workspace } from '../src/api/types';
import type { ArchiveJob } from '../src/api/operations';
import '../src/styles.css';
import '../src/local-workspace.css';

const state = { pipelineRunning: true, holdInventory: true, releaseInventory: null as null | (() => void), loseCancel: false, loseArchive: false, acceptedArchives: 0 };
const calls: { path: string; body: Record<string, unknown> | null }[] = [];
const cancelled = new Set<string>();
const jobs: (ArchiveJob & { archivePath: string })[] = [];
Object.assign(window, { __operationsState: state, __operationsCalls: calls });
const reply = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
window.fetch = async (input, init) => {
  const path = String(input), body = init?.body ? JSON.parse(String(init.body)) : null;
  calls.push({ path, body });
  if (path === '/api/v1/session') return reply({ token: 'offline' });
  if (path.endsWith('/cleanup/cancel')) {
    if (!cancelled.has(body.operationId)) { cancelled.add(body.operationId); state.pipelineRunning = false; }
    if (state.loseCancel) { state.loseCancel = false; throw new TypeError('Invented lost acknowledgement'); }
    return reply({ key: body.key, job: { status: state.pipelineRunning ? 'running' : 'cancelled' } });
  }
  if (path.endsWith('/operations')) {
    if (state.holdInventory) await new Promise<void>((resolve) => { state.releaseInventory = () => { state.holdInventory = false; resolve(); }; });
    return reply({ projectId: 'offline', jobs: [{ key: 'configuration:job', id: 'job', name: 'Synthetic training batch', kind: 'mil-batch', job: { status: state.pipelineRunning ? 'running' : 'cancelled', cancellable: state.pipelineRunning } }], reservations: [], capacity: { cpus: 8, availableRamGb: 24 }, note: '' });
  }
  if (path.endsWith('/operations/sources')) return reply({ sources: [], referenceCount: 2, missingReferences: [{ path: '/external/slides/missing.svs', reason: 'missing' }], note: '' });
  if (path.endsWith('/operations/archives')) {
    if (init?.method === 'POST') {
      let job = jobs.find((item) => item.archivePath === body.archivePath && ['running', 'starting'].includes(item.status));
      if (!job) {
        state.acceptedArchives++;
        job = { id: `archive-${state.acceptedArchives}`, action: body.action, archivePath: body.archivePath, status: 'running', createdAt: new Date().toISOString(), sessionName: 'offline-fixture-no-real-worker', logPath: '/not-created/worker.log', error: null, result: null, progress: { stage: 'copy', completed: 1, total: 3, file: 'records.json' } };
        jobs.push(job);
      }
      if (state.loseArchive) { state.loseArchive = false; throw new TypeError('Invented lost acknowledgement'); }
      return reply(job);
    }
    return reply({ jobs });
  }
  const action = path.match(/\/operations\/archives\/(archive-\d+)\/(cancel|retry)$/);
  if (action) {
    const job = jobs.find((item) => item.id === action[1])!;
    job.status = action[2] === 'cancel' ? 'cancelled' : 'running';
    job.progress = { stage: 'copy', completed: 0, total: 3, file: '' };
    return reply(job);
  }
  throw new Error(`Unexpected offline request: ${path}`);
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
function Fixture() {
  const [connection, setConnection] = useState(0);
  return <QueryClientProvider client={client}><main style={{ maxWidth: 1280, padding: 16, margin: '20px auto' }}><p>Offline operations verification · invented data · no workers or server</p><button className="btn btn-secondary" onClick={() => setConnection(connection + 1)}>Reconnect view</button><LocalOperations key={connection} workspace={{ project: { id: 'offline' } } as Workspace} /></main></QueryClientProvider>;
}
createRoot(document.getElementById('app')!).render(<Fixture />);
