import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { access, approveRequest } from '../api/access';
import type { AgentRequest, ExposureLevel } from '../api/access';
import { ErrorNotice, Panel } from './ui';

export const exposureMeaning: Record<ExposureLevel, string> = {
  none: 'AI agents see nothing of this project.',
  metadata: 'Agents see names, designs, statuses and aggregate results. Patient and slide IDs are pseudonymized; paths, free text, per-case values, images and exports are withheld. This lowers risk; it is not de-identification.',
  full: 'Agents see everything, including slide images and case-level exports. Use it only for data you are free to send to an AI provider.',
};

/** Raising a project to full exposure asks for its name, so it is never a slip. */
export const exposureChangeReady = (level: ExposureLevel, typedName: string, projectName: string) =>
  level !== 'full' || typedName.trim() === projectName.trim();

const errorOf = (reason: unknown) => (reason instanceof Error ? reason : new Error(String(reason)));

export default function AgentAccess({ project, projectName }: { project: string; projectName: string }) {
  const client = useQueryClient();
  const exposure = useQuery({ queryKey: ['ai-exposure', project], queryFn: () => access.exposure(project), staleTime: 30_000 });
  const tokens = useQuery({ queryKey: ['agent-tokens', project], queryFn: () => access.tokens(project), staleTime: 30_000 });
  const requests = useQuery({ queryKey: ['agent-requests', project], queryFn: () => access.requests(project), refetchInterval: 30_000 });
  const current = exposure.data?.level ?? 'none';
  const [level, setLevel] = useState<ExposureLevel | null>(null);
  const [typedName, setTypedName] = useState('');
  const [tokenName, setTokenName] = useState('');
  const [created, setCreated] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const chosen = level ?? current;
  const refresh = () => Promise.all(['ai-exposure', 'agent-tokens', 'agent-requests', 'workspace'].map((key) => client.invalidateQueries({ queryKey: [key, project] })));

  async function act(work: () => Promise<unknown>) {
    setBusy(true); setError(null);
    try { await work(); await refresh(); } catch (reason) { setError(errorOf(reason)); } finally { setBusy(false); }
  }
  const changeExposure = () => act(async () => { await access.setExposure(project, chosen, current); setLevel(null); setTypedName(''); });
  const createToken = () => act(async () => { const token = await access.createToken(project, ['read', 'preview'], tokenName.trim(), 7); setCreated(token.token ?? ''); setTokenName(''); });
  const approve = (entry: AgentRequest) => act(() => approveRequest(entry));
  const pending = (requests.data?.requests ?? []).filter((entry) => entry.state === 'pending');

  return <Panel title="AI agent access" subtitle="Whether AI agents may work on this project, with which tokens, and the changes they asked for.">
    <ErrorNotice error={error ?? exposure.error ?? tokens.error ?? requests.error} />
    <p><strong>Exposure: {current}.</strong> {exposureMeaning[current]}</p>
    <div className="operations-form">
      <label className="operations-field">Change to<select value={chosen} disabled={busy} onChange={(event) => setLevel(event.target.value as ExposureLevel)}>
        <option value="none">none</option><option value="metadata">metadata</option><option value="full">full</option></select></label>
      {chosen === 'full' && current !== 'full' ? <label className="operations-field">Type the project name to allow full exposure<input value={typedName} disabled={busy} onChange={(event) => setTypedName(event.target.value)} placeholder={projectName} /></label> : null}
      <button type="button" className="btn btn-secondary" disabled={busy || chosen === current || !exposureChangeReady(chosen, typedName, projectName)} onClick={() => void changeExposure()}>Save exposure</button>
    </div>
    {chosen !== current ? <p className="muted">{exposureMeaning[chosen]}</p> : null}
    <h3>Tokens</h3>
    {current === 'none' ? <p className="muted">Tokens need an exposure level other than none.</p> : <div className="operations-form">
      <label className="operations-field">Token name<input value={tokenName} disabled={busy} onChange={(event) => setTokenName(event.target.value)} placeholder="Chat app" /></label>
      <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => void createToken()}>Create read and preview token</button>
    </div>}
    {created ? <p className="callout" role="status">Copy this token now; it is not shown again: <code>{created}</code></p> : null}
    <ul className="agent-access-list">{(tokens.data?.tokens ?? []).map((token) => <li key={token.id}>
      <span>{token.name || token.id} · {token.scopes.join(', ')} · {token.state} · expires {token.expiresAt.slice(0, 10)}</span>
      {token.state === 'active' ? <button type="button" className="btn btn-small btn-secondary" disabled={busy} onClick={() => void act(() => access.revokeToken(token.id))}>Revoke</button> : null}
    </li>)}</ul>
    <h3>Requests from AI</h3>
    {pending.length ? <ul className="agent-access-list">{pending.map((entry) => <li key={entry.id}>
      <span><strong>{entry.tokenName || entry.tokenId}</strong> asks: <code>{entry.method} {entry.path}</code></span>
      <details><summary>Request body</summary><pre>{JSON.stringify(entry.body, null, 2)}</pre></details>
      <button type="button" className="btn btn-small btn-primary" disabled={busy} onClick={() => void approve(entry)}>Approve</button>
      <button type="button" className="btn btn-small btn-secondary" disabled={busy} onClick={() => void act(() => access.resolve(entry.id, 'declined'))}>Decline</button>
    </li>)}</ul> : <p className="muted">No changes are waiting for approval.</p>}
  </Panel>;
}
