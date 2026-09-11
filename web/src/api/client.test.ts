import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });

describe('control service client', () => {
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
