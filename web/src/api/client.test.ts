import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });

describe('control service client', () => {
  it('reads compute telemetry through the authenticated service with query cancellation', async () => {
    const sample = { sampledAt: '2026-09-12T17:00:00Z' };
    const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' })).mockResolvedValueOnce(json(sample));
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('./client');
    const controller = new AbortController();
    expect(await api.systemCompute(controller.signal)).toEqual(sample);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/system/compute');
    expect(fetcher.mock.calls[1][1].signal).toBe(controller.signal);
    expect(fetcher.mock.calls[1][1].headers.get('X-HistoPilot-Token')).toBe('session');
    expect(fetcher.mock.calls[1][1].cache).toBe('no-store');
  });

  it('shares renewal when an old JSON or image request returns 401 after a newer session exists', async () => {
    let rejectImage!: (response: Response) => void;
    const delayedImage = new Promise<Response>((resolve) => { rejectImage = resolve; });
    const fetcher = vi.fn().mockImplementation(async (url: string, init: RequestInit) => {
      if (url.endsWith('/session')) return json({ token: fetcher.mock.calls.filter(([path]) => path.endsWith('/session')).length === 1 ? 'old' : 'new' });
      const token = new Headers(init.headers).get('X-HistoPilot-Token');
      if (token === 'new') return url.endsWith('/image') ? new Response('image') : json({ ok: true });
      return url.endsWith('/image') ? delayedImage : json({ detail: 'Expired' }, 401);
    });
    vi.stubGlobal('fetch', fetcher);
    const { request, fetchArtifactBlob } = await import('./client');
    const image = fetchArtifactBlob('/image');
    await expect(request('/projects')).resolves.toEqual({ ok: true });
    rejectImage(json({ detail: 'Expired' }, 401));
    expect(await (await image).text()).toBe('image');
    expect(fetcher.mock.calls.filter(([path]) => path.endsWith('/session'))).toHaveLength(2);
  });

  it('does not replay a mutation after a lost connection and explains the uncertain outcome', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'));
    vi.stubGlobal('fetch', fetcher);
    const { request } = await import('./client');
    await expect(request('/launch', { method: 'POST', body: '{}' })).rejects.toMatchObject({
      status: 0, code: 'SERVICE_UNREACHABLE', message: expect.stringContaining('check the saved record or job status before retrying'),
    });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it.each(['connection', 'json'])('keeps an uncertain %s failure distinct from server rejection so the UI retains its operation ID', async (failure) => {
    const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' }));
    if (failure === 'connection') fetcher.mockRejectedValueOnce(new TypeError('Response lost'));
    else fetcher.mockResolvedValueOnce(new Response('<html>Response lost</html>'));
    vi.stubGlobal('fetch', fetcher);
    const { ApiError, request } = await import('./client');
    const error = await request('/launch', { method: 'POST', body: '{"operationId":"keep-me"}' }).catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(Error);
    expect(error).not.toBeInstanceOf(ApiError);
    expect(error).toHaveProperty('message', expect.stringContaining('check the saved record or job status before retrying'));
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('can retry session bootstrap after a connection failure', async () => {
    const fetcher = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(json({ token: 'session' })).mockResolvedValueOnce(json({ ok: true }));
    vi.stubGlobal('fetch', fetcher);
    const { request } = await import('./client');
    await expect(request('/projects')).rejects.toMatchObject({ code: 'SERVICE_UNREACHABLE' });
    await expect(request('/projects')).resolves.toEqual({ ok: true });
  });

  it.each([null, {}, { token: 123 }, { token: ' ' }])('rejects an invalid session payload: %j', async (payload) => {
    const fetcher = vi.fn().mockResolvedValueOnce(json(payload));
    vi.stubGlobal('fetch', fetcher);
    const { request } = await import('./client');
    await expect(request('/projects')).rejects.toMatchObject({ status: 401, message: 'The service did not return a valid session token.' });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('explains a successful HTTP response containing an HTML proxy page', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(new Response('<html>Proxy</html>')));
    const { request } = await import('./client');
    await expect(request('/projects')).rejects.toMatchObject({ code: 'INVALID_SERVICE_RESPONSE', message: expect.stringContaining('unreadable response') });
  });

  it('preserves cancellation and does not fetch for a request cancelled during session bootstrap', async () => {
    const controller = new AbortController();
    let finishSession!: (response: Response) => void;
    const fetcher = vi.fn().mockReturnValueOnce(new Promise<Response>((resolve) => { finishSession = resolve; }));
    vi.stubGlobal('fetch', fetcher);
    const { request } = await import('./client');
    const pending = request('/projects', { signal: controller.signal });
    controller.abort();
    finishSession(json({ token: 'session' }));
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetcher).toHaveBeenCalledTimes(1);
    await expect(request('/projects', { signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('names the invalid form field when FastAPI supplies a validation location', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: [{ loc: ['body', 'resources', 'workers'], msg: 'Must be an integer' }] }, 422)));
    const { request } = await import('./client');
    await expect(request('/preview')).rejects.toMatchObject({ message: 'resources.workers: Must be an integer' });
  });

  it('shares session bootstrap and authenticates both reads and mutations', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(json({ token: 'session-one' }))
      .mockResolvedValueOnce(json({ id: 'project' }))
      .mockResolvedValueOnce(json({ projects: [], defaultStoragePath: '/workspace' }));
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('./client');
    await api.createProject({ name: 'Bladder', storagePath: '/workspace/bladder' });
    await api.projects();
    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(fetcher.mock.calls[0][0]).toBe('/api/v1/session');
    const mutation = fetcher.mock.calls[1][1];
    expect(mutation.method).toBe('POST');
    expect(mutation.headers.get('X-HistoPilot-Token')).toBe('session-one');
    expect(JSON.parse(mutation.body)).toMatchObject({ name: 'Bladder' });
    expect(fetcher.mock.calls[2][1].headers.get('X-HistoPilot-Token')).toBe('session-one');
  });
  it('renews an expired session once without silently returning fixture data', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(json({ token: 'old' }))
      .mockResolvedValueOnce(json({ detail: 'Expired' }, 401))
      .mockResolvedValueOnce(json({ token: 'new' }))
      .mockResolvedValueOnce(json({ detail: 'Still unauthorized' }, 401));
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('./client');
    await expect(api.projectWorkspace('project')).rejects.toMatchObject({
      status: 401,
      message: 'Still unauthorized',
    });
    expect(fetcher).toHaveBeenCalledTimes(4);
    expect(fetcher.mock.calls[3][1].headers.get('X-HistoPilot-Token')).toBe('new');
  });
  it('surfaces service validation and permits a bodyless response', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: [{ msg: 'Choose a supported MIL model' }] }, 422))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetcher);
    const { api, request } = await import('./client');
    await expect(api.createProject({ name: 'Bladder', storagePath: '/workspace/bladder', config: { milId: 'unknown' } }))
      .rejects.toMatchObject({ status: 422, message: 'Choose a supported MIL model' });
    await expect(request('/records/one', { method: 'DELETE' })).resolves.toBeUndefined();
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/records/one');
  });
  it('creates folders in the selected purpose and surfaces existing-folder conflicts', async () => {
    const created = { path: '/data/features/Bladder Ω', parent: '/data/features', name: 'Bladder Ω' };
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json(created, 201))
      .mockResolvedValueOnce(json({ detail: 'A folder with this name already exists.' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('./client');
    await expect(api.createDirectory('/data/features', 'Bladder Ω')).resolves.toEqual(created);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/filesystem/directories');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({
      parentPath: '/data/features', name: 'Bladder Ω', purpose: 'source',
    });
    await expect(api.createDirectory('/workspace', 'existing', 'storage')).rejects.toMatchObject({
      status: 409, message: 'A folder with this name already exists.',
    });
    expect(JSON.parse(fetcher.mock.calls[2][1].body).purpose).toBe('storage');
  });
  describe('reads that meet a busy project', () => {
    const busy = () => json({ detail: 'Another operation is changing this workspace.', code: 'PROJECT_BUSY' }, 409);
    afterEach(() => { vi.useRealTimers(); });

    it('waits with a doubling backoff and reads again', async () => {
      vi.useFakeTimers();
      const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
        .mockResolvedValueOnce(busy()).mockResolvedValueOnce(busy()).mockResolvedValueOnce(json({ items: [] }));
      vi.stubGlobal('fetch', fetcher);
      const { request } = await import('./client');
      const pending = request('/projects/p/model-experiments');
      await vi.advanceTimersByTimeAsync(249);
      expect(fetcher).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(1);
      expect(fetcher).toHaveBeenCalledTimes(3);
      await vi.advanceTimersByTimeAsync(500);
      await expect(pending).resolves.toEqual({ items: [] });
      expect(fetcher).toHaveBeenCalledTimes(4);
      expect(fetcher.mock.calls[3][1].headers.get('X-HistoPilot-Token')).toBe('session');
    });

    it('gives up with the busy error after its retry budget', async () => {
      vi.useFakeTimers();
      const fetcher = vi.fn().mockImplementation(async (url: string) => (url.endsWith('/session') ? json({ token: 'session' }) : busy()));
      vi.stubGlobal('fetch', fetcher);
      const { BUSY_READ_RETRY_DELAYS_MS, request } = await import('./client');
      const pending = request('/projects/p/operations/sources').catch((reason: unknown) => reason);
      await vi.advanceTimersByTimeAsync(BUSY_READ_RETRY_DELAYS_MS.reduce((total, delay) => total + delay, 0));
      expect(await pending).toMatchObject({ status: 409, code: 'PROJECT_BUSY' });
      expect(fetcher).toHaveBeenCalledTimes(1 + 1 + BUSY_READ_RETRY_DELAYS_MS.length);
    });

    it('never replays a mutation, whose operation ID the UI keeps for an explicit retry', async () => {
      const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' })).mockResolvedValueOnce(busy());
      vi.stubGlobal('fetch', fetcher);
      const { request } = await import('./client');
      await expect(request('/projects/p/model-experiments/e/submit', { method: 'POST', body: '{"operationId":"keep"}' }))
        .rejects.toMatchObject({ status: 409, code: 'PROJECT_BUSY' });
      expect(fetcher).toHaveBeenCalledTimes(2);
    });

    it('does not retry other conflicts', async () => {
      const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
        .mockResolvedValueOnce(json({ detail: 'Reload.', code: 'REVISION_CONFLICT' }, 409));
      vi.stubGlobal('fetch', fetcher);
      const { request } = await import('./client');
      await expect(request('/projects/p/drafts/d')).rejects.toMatchObject({ code: 'REVISION_CONFLICT' });
      expect(fetcher).toHaveBeenCalledTimes(2);
    });

    it('stops waiting as soon as the query is cancelled', async () => {
      vi.useFakeTimers();
      const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' })).mockResolvedValue(busy());
      vi.stubGlobal('fetch', fetcher);
      const { request } = await import('./client');
      const controller = new AbortController();
      const pending = request('/projects/p/drafts', { signal: controller.signal }).catch((reason: unknown) => reason);
      await vi.advanceTimersByTimeAsync(100);
      controller.abort();
      expect(await pending).toMatchObject({ name: 'AbortError' });
      await vi.advanceTimersByTimeAsync(10000);
      expect(fetcher).toHaveBeenCalledTimes(2);
    });
  });

  it.each(['PREVIEW_STALE', 'FEATURE_BUNDLE_INVALID', 'VERSION_TAG_CONFLICT'])('preserves structured %s errors for the correct recovery flow', async (code) => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: 'Review could not be saved.', code }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { request } = await import('./client');
    await expect(request('/projects/project/feature-bundles/freeze', { method: 'POST', body: '{}' }))
      .rejects.toMatchObject({ status: 409, message: 'Review could not be saved.', code });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
