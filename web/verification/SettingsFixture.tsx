/** Offline settings concurrency fixture: invented data, no control service. */
import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Settings } from '../src/pages/LocalWorkspace';
import type { InitialConfig, Workspace } from '../src/api/types';
import '../src/styles.css';
import '../src/local-workspace.css';

let saved: InitialConfig = { seed: 7 };
const requests: { config: InitialConfig; expectedConfig: InitialConfig; status: number }[] = [];
const transport: { delayed: boolean; release?: () => void } = { delayed: false };
const workspace = () => ({
  project: { id: 'settings-fixture', name: 'Offline settings', config: { ...saved } },
  encoders: [], milModels: [],
}) as unknown as Workspace;
const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
});
Object.assign(window, { __settingsRequests: requests, __settingsTransport: transport });
window.fetch = async (input, init) => {
  const path = String(input);
  if (path === '/api/v1/session') return reply({ token: 'offline' });
  if (transport.delayed) await new Promise<void>((resolve) => { transport.release = resolve; });
  if (path === '/api/v1/projects/settings-fixture/workspace') return reply(workspace());
  if (path === '/api/v1/projects/settings-fixture' && init?.method === 'PATCH') {
    const payload = JSON.parse(String(init.body));
    const conflict = JSON.stringify(payload.expectedConfig) !== JSON.stringify(saved)
      && JSON.stringify(payload.config) !== JSON.stringify(saved);
    requests.push({ ...payload, status: conflict ? 409 : 200 });
    if (conflict) return reply({ code: 'PROJECT_CONFIG_CONFLICT', detail: 'Project settings changed in another tab. Your edits have not been saved.' }, 409);
    saved = payload.config;
    return reply(workspace().project);
  }
  throw new Error(`Unexpected offline request: ${path}`);
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
function Fixture() {
  const [current, setCurrent] = useState(workspace);
  return <QueryClientProvider client={client}><main style={{ maxWidth: 960, margin: '24px auto', padding: 16 }}>
    <p>Offline settings verification · invented data · no server</p>
    <button type="button" className="btn btn-secondary" onClick={() => {
      saved = { seed: 11 };
      setCurrent(workspace());
    }}>Simulate change in another tab</button>
    <Settings workspace={current} />
  </main></QueryClientProvider>;
}
createRoot(document.getElementById('app')!).render(<Fixture />);
