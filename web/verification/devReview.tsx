/** Offline browser harness. All project data is invented and all API traffic is mocked. */
import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from '../src/App';
import WorkspaceErrorBoundary from '../src/components/WorkspaceErrorBoundary';
import ComputeJobControls from '../src/components/ComputeJobControls';
import { InferenceFixture } from './InferenceFixture';
import type { Workspace } from '../src/api/types';
import '../src/styles.css';
import '../src/local-workspace.css';
import '../src/scientific.css';
import '../src/clinical-workspace.css';
import '../src/roadmap.css';

const project = {
  id: 'offline-review', name: 'Offline review project', mode: 'local', config: {},
  description: 'Invented data for interface verification only.', storagePath: '/offline-only',
  createdAt: '', updatedAt: '', sources: [], available: true, lifecycleState: 'active',
};
const workspace = {
  mode: 'local', project, dataset: { id: '', slideCount: 0, patientCount: 0, specimenCount: 0 },
  patients: [], slides: [], results: [], encoders: [], milModels: [], featureSets: [],
  cohortSnapshots: [], drafts: [], sources: [], exampleManifests: {},
} as unknown as Workspace;
const responses: Record<string, unknown> = {
  '/session': { token: 'offline-only', scientificCapabilities: { versionLabels: true, taggedFreeze: true } },
  '/projects': { projects: [project], defaultStoragePath: '/offline-only' },
  '/projects/offline-review/workspace': workspace,
};
for (const [path, value] of Object.entries({
  drafts: { drafts: [] }, datasets: { datasets: [] },
  'configurations?kind=protocol': { configurations: [] },
  'configurations?kind=feature': { configurations: [] },
  'mil-experiments/batches': { items: [], executions: [], executionImplemented: true },
  'feature-bundles': { items: [] }, 'evaluation-cohorts': { items: [] },
  'predictors?include_inactive=true': { items: [] }, 'evaluation-runs?include_inactive=true': { items: [] },
  'clinical-analyses': { items: [] }, interpretations: { items: [] }, 'predictors/refits?include_inactive=true': { items: [] },
})) responses[`/projects/offline-review/${path}`] = value;

const traffic: { path: string; method: string; matched: boolean }[] = [];
const operations: string[] = [];
Object.assign(window, { __reviewTraffic: traffic, __reviewOperations: operations });
window.fetch = async (input, init) => {
  const path = String(input).replace(/^\/api\/v1/, '');
  const method = init?.method ?? 'GET';
  if (path === '/projects/offline-review/predictors/refits/retry-fixture/execution') {
    return new Response(JSON.stringify({ status: operations.length ? 'queued' : 'not_started' }));
  }
  if (path === '/projects/offline-review/predictors/refits/retry-fixture/launch' && method === 'POST') {
    operations.push(JSON.parse(String(init?.body)).operationId);
    if (operations.length === 1) throw new TypeError('Fixture accepted the job but lost its response.');
    return new Response(JSON.stringify({ status: 'queued' }));
  }
  const matched = Object.hasOwn(responses, path) && method === 'GET';
  traffic.push({ path, method, matched });
  return new Response(JSON.stringify(matched ? responses[path] : { detail: `Offline fixture has no ${method} ${path}` }), {
    status: matched ? 200 : 404, headers: { 'Content-Type': 'application/json' },
  });
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity }, mutations: { retry: false } } });
let fault = true;
function RecoverableView() {
  if (fault) throw new Error('Deliberately injected display fault for offline verification.');
  return <p id="recovered-view">The view recovered successfully.</p>;
}
function CrashFixture() {
  return <main style={{ padding: 24 }}>
    <button className="btn btn-secondary" onClick={() => { fault = false; }}>Resolve fixture fault</button>
    <WorkspaceErrorBoundary><RecoverableView /></WorkspaceErrorBoundary>
  </main>;
}
function Review() {
  const [view, setView] = useState('Project workspace');
  return <QueryClientProvider client={client}>
    <nav aria-label="Offline verification scenarios" style={{ position: 'relative', zIndex: 200, display: 'flex', flexWrap: 'wrap', gap: 8, padding: 12, background: '#e8eef2' }}>
      <strong style={{ padding: 8 }}>Offline fixtures · no server</strong>
      {['Project workspace', 'Inference controls', 'Compute retry', 'Rendering recovery'].map((name) => <button key={name} className="btn btn-secondary" onClick={() => setView(name)}>{name}</button>)}
    </nav>
    {view === 'Project workspace' ? <App /> : view === 'Inference controls' ? <main style={{ padding: 24, maxWidth: 1080, margin: '0 auto' }}><InferenceFixture /></main>
      : view === 'Compute retry' ? <main style={{ padding: 24 }}><h1>Uncertain launch recovery</h1><ComputeJobControls project={project.id} id="retry-fixture" kind="refit" /></main>
        : <CrashFixture />}
  </QueryClientProvider>;
}
const url = new URL(location.href);
url.searchParams.set('project', project.id);
url.hash = 'overview';
history.replaceState({}, '', url);
createRoot(document.getElementById('app')!).render(<Review />);
