import { afterEach, describe, expect, it, vi } from 'vitest';

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('immutable feature bundle API', () => {
  it('reviews and freezes the exact feature and pack references with review identity and version label', async () => {
    const spec = { featureSetId: 'feature/version', packArtifactIds: ['pack/one', 'pack/two'] };
    const label = { tag: 'UNI baseline', note: 'Reviewed float32 features\nTwo verified pack copies' };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ spec, previewHash: 'review-hash', canFreeze: true }))
      .mockResolvedValueOnce(json({ id: 'frozen-bundle', manifest: { spec }, versionLabel: label }));
    vi.stubGlobal('fetch', fetcher);
    const { bundles } = await import('./bundles');
    await expect(bundles.preview('project/one', spec)).resolves.toMatchObject({ previewHash: 'review-hash' });
    await expect(bundles.freeze('project/one', spec, 'review-hash', 'bundle:one-intent', label)).resolves.toMatchObject({ id: 'frozen-bundle' });
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/feature-bundles/preview');
    expect(fetcher.mock.calls[1][1].method).toBe('POST');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(spec);
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/project%2Fone/feature-bundles/freeze');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ ...spec, previewHash: 'review-hash', operationId: 'bundle:one-intent', versionLabel: label });
    expect(fetcher.mock.calls.some(([url]) => String(url).includes('pack-selection'))).toBe(false);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it('retains explicit features-only contents and retrieves immutable bundles by ID', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ canFreeze: true }))
      .mockResolvedValueOnce(json({ items: [{ id: 'bundle/one' }] }))
      .mockResolvedValueOnce(json({ id: 'bundle/one', current: true }));
    vi.stubGlobal('fetch', fetcher);
    const { bundles } = await import('./bundles');
    await bundles.preview('project', { featureSetId: 'feature', packArtifactIds: [] });
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ featureSetId: 'feature', packArtifactIds: [] });
    await expect(bundles.list('project')).resolves.toEqual({ items: [{ id: 'bundle/one' }] });
    await expect(bundles.get('project', 'bundle/one')).resolves.toMatchObject({ id: 'bundle/one', current: true });
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/project/feature-bundles');
    expect(fetcher.mock.calls[3][0]).toBe('/api/v1/projects/project/feature-bundles/bundle%2Fone');
  });

  it('surfaces stale review rejection without changing the pack list or retrying the write', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: 'Bundle review changed; review again.' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { bundles } = await import('./bundles');
    await expect(bundles.freeze('project', { featureSetId: 'feature', packArtifactIds: ['pack'] }, 'old-preview', 'same-intent', { tag: 'My bundle', note: '' })).rejects.toMatchObject({ status: 409, message: 'Bundle review changed; review again.' });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(JSON.parse(fetcher.mock.calls[1][1].body).packArtifactIds).toEqual(['pack']);
  });
});
