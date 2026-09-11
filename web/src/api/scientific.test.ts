import { afterEach, describe, expect, it, vi } from 'vitest';

const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe('version label metadata API', () => {
  it('freezes each scientific kind with its label in one request and preserves the caller retry identity', async () => {
    const versionLabel = { tag: 'reviewed-v1', note: 'Reviewed records\nChecked coverage' };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { taggedFreeze: true } }))
      .mockImplementation(() => Promise.resolve(json({ id: 'saved', versionLabel })));
    vi.stubGlobal('fetch', fetcher);
    const { scientific } = await import('./scientific');
    await scientific.importFreeze('project', 'draft', 4, 'dataset-preview', versionLabel, 'import:one-intent');
    await scientific.protocolFreeze('project', 'protocol-draft', 2, 'protocol-preview', versionLabel, 'protocol:one-intent');
    await scientific.featureFreeze('project', { datasetId: 'dataset', path: '/features', fileSuffix: '.h5', idSuffix: '', recursive: false }, 'feature-preview', versionLabel, 'feature:one-intent');
    expect(fetcher).toHaveBeenCalledTimes(4);
    for (const [, init] of fetcher.mock.calls.slice(1)) {
      expect(init.method).toBe('POST');
      expect(JSON.parse(init.body).versionLabel).toEqual(versionLabel);
    }
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ expectedRevision: 4, previewHash: 'dataset-preview', versionLabel, operationId: 'import:one-intent' });
    expect(JSON.parse(fetcher.mock.calls[2][1].body).operationId).toBe('protocol:one-intent');
    expect(JSON.parse(fetcher.mock.calls[3][1].body)).toMatchObject({ datasetId: 'dataset', path: '/features', operationId: 'feature:one-intent' });
  });

  it('saves dataset and configuration labels with explicit revisions outside scientific manifests', async () => {
    const saved = { tag: 'baseline-v1', note: 'First line\nSecond line', revision: 3 };
    const cleared = { tag: '', note: '', revision: 4 };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { versionLabels: true, taggedFreeze: true } }))
      .mockResolvedValueOnce(json(saved))
      .mockResolvedValueOnce(json(cleared));
    vi.stubGlobal('fetch', fetcher);
    const { scientific } = await import('./scientific');
    await expect(scientific.setVersionLabel('project/one', 'dataset', 'dataset/one', {
      tag: saved.tag, note: saved.note, expectedRevision: 2,
    })).resolves.toEqual(saved);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/datasets/dataset%2Fone/label');
    expect(fetcher.mock.calls[1][1].method).toBe('PUT');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({
      tag: saved.tag, note: saved.note, expectedRevision: 2,
    });
    await expect(scientific.setVersionLabel('project/one', 'configuration', 'feature/one', {
      tag: '', note: '', expectedRevision: 3,
    })).resolves.toEqual(cleared);
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/project%2Fone/configurations/feature%2Fone/label');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ tag: '', note: '', expectedRevision: 3 });
  });

  it('surfaces concurrent update conflicts without retrying a stale write', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { versionLabels: true, taggedFreeze: true } }))
      .mockResolvedValueOnce(json({ detail: 'Version label changed; refresh and try again.' }, 409))
      .mockResolvedValueOnce(json({ id: 'dataset', versionLabel: { tag: 'newer', revision: 2 } }));
    vi.stubGlobal('fetch', fetcher);
    const { scientific } = await import('./scientific');
    await expect(scientific.setVersionLabel('project', 'dataset', 'dataset', {
      tag: 'my draft', note: 'keep this note', expectedRevision: 1,
    })).rejects.toMatchObject({ status: 409, message: 'Version label changed; refresh and try again.' });
    expect(fetcher).toHaveBeenCalledTimes(2);
    await expect(scientific.dataset('project', 'dataset')).resolves.toMatchObject({
      versionLabel: { tag: 'newer', revision: 2 },
    });
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/project/datasets/dataset');
  });

  it('blocks freeze and label writes when the running server lacks the required capability', async () => {
    const fetcher = vi.fn().mockImplementation(() => Promise.resolve(json({ token: 'old-server' })));
    vi.stubGlobal('fetch', fetcher);
    const { scientific } = await import('./scientific');
    await expect(scientific.importFreeze('project', 'draft', 1, 'preview', { tag: 'my-tag', note: '' }, 'one-intent'))
      .rejects.toMatchObject({ status: 405, message: expect.stringContaining('Restart HistoPilot') });
    await expect(scientific.setVersionLabel('project', 'dataset', 'dataset', { tag: 'my-tag', note: '', expectedRevision: 0 }))
      .rejects.toMatchObject({ status: 405, message: expect.stringContaining('tag and note have been kept') });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('rechecks capabilities after session renewal before retrying a freeze', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'supported', scientificCapabilities: { taggedFreeze: true } }))
      .mockResolvedValueOnce(json({ detail: 'Expired' }, 401))
      .mockResolvedValueOnce(json({ token: 'older-server' }));
    vi.stubGlobal('fetch', fetcher);
    const { scientific } = await import('./scientific');
    await expect(scientific.importFreeze('project', 'draft', 1, 'preview', { tag: 'my-tag', note: '' }, 'one-intent'))
      .rejects.toMatchObject({ status: 405, message: expect.stringContaining('Restart HistoPilot') });
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it('can retry the preserved naming draft after the server restarts without reloading the page', async () => {
    const versionLabel = { tag: 'keep-this-tag', note: 'Keep this note too' };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'old-server' }))
      .mockResolvedValueOnce(json({ token: 'restarted', scientificCapabilities: { taggedFreeze: true } }))
      .mockResolvedValueOnce(json({ id: 'saved', versionLabel }));
    vi.stubGlobal('fetch', fetcher);
    const { scientific } = await import('./scientific');
    const save = () => scientific.importFreeze('project', 'draft', 1, 'preview', versionLabel, 'same-intent');
    await expect(save()).rejects.toMatchObject({ status: 405 });
    await expect(save()).resolves.toMatchObject({ versionLabel });
    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/session');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toMatchObject({ versionLabel, operationId: 'same-intent' });
  });

  it('explains missing save routes while preserving genuine resource-not-found errors', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { versionLabels: true } }))
      .mockResolvedValueOnce(json({ detail: 'Method Not Allowed' }, 405))
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { versionLabels: true } }))
      .mockResolvedValueOnce(json({ detail: 'Not Found' }, 404))
      .mockResolvedValueOnce(json({ token: 'session', scientificCapabilities: { versionLabels: true } }))
      .mockResolvedValueOnce(json({ detail: 'The frozen dataset does not exist.' }, 404));
    vi.stubGlobal('fetch', fetcher);
    const { scientific } = await import('./scientific');
    const save = () => scientific.setVersionLabel('project', 'dataset', 'dataset', { tag: 'tag', note: '', expectedRevision: 0 });
    await expect(save()).rejects.toMatchObject({ message: expect.stringContaining('Restart HistoPilot') });
    await expect(save()).rejects.toMatchObject({ message: expect.stringContaining('Restart HistoPilot') });
    await expect(save()).rejects.toMatchObject({ status: 404, message: 'The frozen dataset does not exist.' });
  });
});
