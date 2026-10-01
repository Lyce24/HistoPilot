import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AgentAccess, { exposureChangeReady } from './AgentAccess';
import { access, approveRequest, replayPath, type AgentRequest } from '../api/access';
import { ApiError } from '../api/client';

const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.restoreAllMocks(); });

function render(level: 'none' | 'metadata' | 'full', requests: unknown[] = []) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['ai-exposure', 'project'], { projectId: 'project', level });
  client.setQueryData(['agent-tokens', 'project'], { tokens: [{ id: 'abc', name: 'Chat app', projectId: 'project', scopes: ['read', 'preview'], createdAt: '', expiresAt: '2026-10-07T00:00:00Z', revokedAt: null, lastUsedAt: null, state: 'active' }] });
  client.setQueryData(['agent-requests', 'project'], { requests });
  return renderToStaticMarkup(<QueryClientProvider client={client}><AgentAccess project="project" projectName="Study" /></QueryClientProvider>);
}

describe('AI agent access', () => {
  it('explains the level and keeps tokens off private projects', () => {
    const html = render('none');
    expect(html).toContain('Exposure: none.');
    expect(html).toContain('AI agents see nothing of this project.');
    expect(html).toContain('Tokens need an exposure level other than none.');
  });
  it('lists tokens and the changes agents asked for', () => {
    const html = render('metadata', [{ id: 'request-1', projectId: 'project', tokenId: 'abc', tokenName: 'Chat app', method: 'PATCH', path: '/api/v1/projects/project', query: '', body: { config: { seed: 7 } }, state: 'pending', createdAt: '', expiresAt: '' }]);
    expect(html).toContain('Chat app · read, preview · active');
    expect(html).toContain('Revoke');
    expect(html).toContain('PATCH /api/v1/projects/project');
    expect(html).toContain('Approve');
    expect(render('metadata')).toContain('No changes are waiting for approval.');
  });
  it('asks for the project name before full exposure and replays requests under the API root', () => {
    expect(exposureChangeReady('full', 'Stud', 'Study')).toBe(false);
    expect(exposureChangeReady('full', ' Study ', 'Study')).toBe(true);
    expect(exposureChangeReady('metadata', '', 'Study')).toBe(true);
    expect(replayPath({ path: '/api/v1/projects/p/drafts', query: 'x=1' })).toBe('/projects/p/drafts?x=1');
  });
  it('claims a request before replaying it and records how the replay ended', async () => {
    const entry = { id: 'request-1', path: '/api/v1/projects/p', query: '', method: 'PATCH', body: {} } as AgentRequest;
    const steps: string[] = [];
    vi.spyOn(access, 'claim').mockImplementation(async (id) => { steps.push(`claim ${id}`); return entry; });
    vi.spyOn(access, 'resolve').mockImplementation(async (id, outcome, status) => { steps.push(`resolve ${outcome} ${status}`); return entry; });
    const replay = vi.spyOn(access, 'replay').mockResolvedValue({ ok: true });
    await approveRequest(entry);
    expect(steps).toEqual(['claim request-1', 'resolve approved 200']);
    steps.length = 0;
    replay.mockRejectedValueOnce(new ApiError('Stale preview', 409, 'PREVIEW_CHANGED'));
    await expect(approveRequest(entry)).rejects.toThrow('Stale preview');
    expect(steps).toEqual(['claim request-1', 'resolve failed 409']);
    steps.length = 0;
    replay.mockRejectedValueOnce(new Error('network down'));
    await expect(approveRequest(entry)).rejects.toThrow('network down');
    expect(steps).toEqual(['claim request-1', 'resolve unknown undefined']);
  });
});
