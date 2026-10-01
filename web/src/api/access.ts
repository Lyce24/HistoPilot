import { ApiError, request } from './client';

/** What AI agents may see of a project; see docs/agents.md. */
export type ExposureLevel = 'none' | 'metadata' | 'full';

export interface AccessToken {
  id: string; name: string; projectId: string; scopes: string[];
  createdAt: string; expiresAt: string; revokedAt: string | null; lastUsedAt: string | null;
  state: 'active' | 'revoked' | 'expired';
  /** Present once, in the answer that created the token. */
  token?: string;
}

/** How a claimed request's replay ended: it ran, the service refused it, or no answer came. */
export type ReplayOutcome = 'approved' | 'failed' | 'unknown';

export interface AgentRequest {
  id: string; projectId: string; tokenId: string; tokenName: string;
  method: string; path: string; query: string; body: unknown;
  state: 'pending' | 'approving' | ReplayOutcome | 'declined' | 'expired';
  createdAt: string; expiresAt: string;
}

const encoded = encodeURIComponent;
const send = (method: string, body?: unknown) => (body === undefined ? { method } : { method, body: JSON.stringify(body) });

/** The route an agent asked for, relative to the API root the browser's client adds. */
export const replayPath = (entry: Pick<AgentRequest, 'path' | 'query'>) =>
  entry.path.replace(/^\/api\/v1/, '') + (entry.query ? `?${entry.query}` : '');

export const access = {
  exposure: (project: string) => request<{ projectId: string; level: ExposureLevel }>(`/projects/${encoded(project)}/ai-exposure`),
  setExposure: (project: string, level: ExposureLevel, expectedLevel: ExposureLevel) =>
    request<{ projectId: string; level: ExposureLevel }>(`/projects/${encoded(project)}/ai-exposure`, send('PUT', { level, expectedLevel })),
  tokens: (project: string) => request<{ tokens: AccessToken[] }>(`/tokens?project=${encoded(project)}`),
  createToken: (project: string, scopes: string[], name: string, days: number) =>
    request<AccessToken>('/tokens', send('POST', { projectId: project, scopes, name, days })),
  revokeToken: (id: string) => request<AccessToken>(`/tokens/${encoded(id)}/revoke`, send('POST')),
  requests: (project: string) => request<{ requests: AgentRequest[] }>(`/agent-requests?project=${encoded(project)}`),
  /** Takes a pending request for one replay; a second approver is refused. */
  claim: (id: string) => request<AgentRequest>(`/agent-requests/${encoded(id)}/claim`, send('POST')),
  /** Runs the agent's request as this person; the service checks it again. */
  replay: (entry: AgentRequest) => request<unknown>(replayPath(entry), send(entry.method, entry.body ?? undefined)),
  resolve: (id: string, outcome: ReplayOutcome | 'declined', status?: number) =>
    request<AgentRequest>(`/agent-requests/${encoded(id)}/resolve`, send('POST', { outcome, ...(status ? { status } : {}) })),
};

/** Approve a request: claim it so it runs once, replay it, then record how the replay ended. */
export async function approveRequest(entry: AgentRequest): Promise<unknown> {
  await access.claim(entry.id);
  let result: unknown;
  try {
    result = await access.replay(entry);
  } catch (reason) {
    // A refusal changed nothing; without an answer the change may have run.
    const status = reason instanceof ApiError ? reason.status : undefined;
    await access.resolve(entry.id, status !== undefined && status < 500 ? 'failed' : 'unknown', status);
    throw reason;
  }
  await access.resolve(entry.id, 'approved', 200);
  return result;
}
