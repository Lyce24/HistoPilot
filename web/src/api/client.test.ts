import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });

describe('control service client', () => {
  it('sends the editor baseline and preserves project-setting conflicts without retrying', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: 'Reload saved settings.', code: 'PROJECT_CONFIG_CONFLICT' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('./client');
    await expect(api.updateProject('project/one', { config: { seed: 13 }, expectedConfig: { seed: 7 } }))
      .rejects.toMatchObject({ code: 'PROJECT_CONFIG_CONFLICT', status: 409 });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ config: { seed: 13 }, expectedConfig: { seed: 7 } });
  });

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
    await expect(request('/workspace')).resolves.toEqual({ ok: true });
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
    await expect(request('/workspace')).rejects.toMatchObject({ code: 'SERVICE_UNREACHABLE' });
    await expect(request('/workspace')).resolves.toEqual({ ok: true });
  });

  it.each([null, {}, { token: 123 }, { token: ' ' }])('rejects an invalid session payload: %j', async (payload) => {
    const fetcher = vi.fn().mockResolvedValueOnce(json(payload));
    vi.stubGlobal('fetch', fetcher);
    const { request } = await import('./client');
    await expect(request('/workspace')).rejects.toMatchObject({ status: 401, message: 'The service did not return a valid session token.' });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('explains a successful HTTP response containing an HTML proxy page', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(new Response('<html>Proxy</html>')));
    const { request } = await import('./client');
    await expect(request('/workspace')).rejects.toMatchObject({ code: 'INVALID_SERVICE_RESPONSE', message: expect.stringContaining('unreadable response') });
  });

  it('preserves cancellation and does not fetch for a request cancelled during session bootstrap', async () => {
    const controller = new AbortController();
    let finishSession!: (response: Response) => void;
    const fetcher = vi.fn().mockReturnValueOnce(new Promise<Response>((resolve) => { finishSession = resolve; }));
    vi.stubGlobal('fetch', fetcher);
    const { request } = await import('./client');
    const pending = request('/workspace', { signal: controller.signal });
    controller.abort();
    finishSession(json({ token: 'session' }));
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetcher).toHaveBeenCalledTimes(1);
    await expect(request('/workspace', { signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
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
      .mockResolvedValueOnce(json({ id: 'saved-cohort' }))
      .mockResolvedValueOnce(json({ jobs: [], executionEnabled: false }));
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('./client');
    await api.saveCohort({ datasetId: 'test', specimenType: 'Any', msi: 'Any', braf: 'Any' });
    await api.jobs();
    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(fetcher.mock.calls[0][0]).toBe('/api/v1/session');
    const mutation = fetcher.mock.calls[1][1];
    expect(mutation.method).toBe('POST');
    expect(mutation.headers.get('X-HistoPilot-Token')).toBe('session-one');
    expect(JSON.parse(mutation.body)).toMatchObject({ datasetId: 'test' });
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
    await expect(api.workspace()).rejects.toMatchObject({
      status: 401,
      message: 'Still unauthorized',
    });
    expect(fetcher).toHaveBeenCalledTimes(4);
    expect(fetcher.mock.calls[3][1].headers.get('X-HistoPilot-Token')).toBe('new');
  });
  it('surfaces service validation and permits a bodyless draft deletion response', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: [{ msg: 'Fold count exceeds patients' }] }, 422))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('./client');
    await expect(
      api.createExperiments({
        cohortId: 'cohort',
        pairs: ['encoder:mil'],
        seeds: [42],
        folds: 5,
        aggregation: 'mean',
      }),
    ).rejects.toMatchObject({ status: 422, message: 'Fold count exceeds patients' });
    await expect(api.deleteExperiment('draft/one')).resolves.toBeUndefined();
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/experiments/draft%2Fone');
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
