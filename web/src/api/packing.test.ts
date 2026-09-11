import { afterEach, describe, expect, it, vi } from 'vitest';
import { featurePackActive, featurePackSpecKey } from './packing';
import type { FeaturePackSpec } from './packing';

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
});
const spec: FeaturePackSpec = { featureSetId: 'feature/one', action: 'pack', dtype: 'preserve', outputPath: '/packs/one' };

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

describe('feature packing requests', () => {
  it('submits the reviewed destination, precision and stable operation identity with authentication', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ previewHash: 'review', spec }))
      .mockResolvedValueOnce(json({ id: 'job/one', state: 'starting' }));
    vi.stubGlobal('fetch', fetcher);
    const { packing } = await import('./packing');
    await packing.preview('project/one', spec);
    await packing.start('project/one', spec, 'review', 'one-intent');
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/feature-packs/preview');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(spec);
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/project%2Fone/feature-packs');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ ...spec, previewHash: 'review', operationId: 'one-intent' });
    expect(fetcher.mock.calls[2][1].headers.get('X-HistoPilot-Token')).toBe('session');
  });

  it('reloads durable jobs, reads latest validation and cancels an identified job', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ jobs: [{ id: 'job/one', state: 'running' }], artifacts: [] }))
      .mockResolvedValueOnce(json(null))
      .mockResolvedValueOnce(json({ id: 'job/one', state: 'running', logs: 'Reading slide A' }))
      .mockResolvedValueOnce(json({ id: 'job/one', state: 'cancelling' }));
    vi.stubGlobal('fetch', fetcher);
    const { packing } = await import('./packing');
    await expect(packing.jobs('project')).resolves.toMatchObject({ jobs: [{ state: 'running' }] });
    await expect(packing.validation('project', 'feature/one')).resolves.toBeNull();
    await expect(packing.job('project', 'job/one')).resolves.toMatchObject({ logs: 'Reading slide A' });
    await expect(packing.cancel('project', 'job/one')).resolves.toMatchObject({ state: 'cancelling' });
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/project/features/feature%2Fone/validation');
    expect(fetcher.mock.calls[3][0]).toBe('/api/v1/projects/project/feature-packs/job%2Fone');
    expect(fetcher.mock.calls[4][0]).toBe('/api/v1/projects/project/feature-packs/job%2Fone/cancel');
    expect(fetcher.mock.calls[4][1].method).toBe('POST');
  });

  it('surfaces source changes as a conflict without retrying the launch', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ detail: 'Feature files changed. Review again.' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { packing } = await import('./packing');
    await expect(packing.start('project', spec, 'stale', 'same-intent')).rejects.toMatchObject({ status: 409, message: 'Feature files changed. Review again.' });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('reviews an existing folder without treating it as a creation destination', async () => {
    const attached = { featureSetId: 'features', action: 'attach' as const, dtype: 'preserve' as const, existingPath: '/mmap/blca', outputPath: null };
    const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' })).mockResolvedValueOnce(json({ spec: attached, canRun: true }));
    vi.stubGlobal('fetch', fetcher);
    const { packing } = await import('./packing');
    await packing.preview('project', attached);
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual(attached);
  });

  it('persists a chosen pack and an explicit switch back to original files', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(json({ token: 'session' }))
      .mockResolvedValueOnce(json({ featureSetId: 'feature/one', artifactId: 'pack/one', current: true }))
      .mockResolvedValueOnce(json({ featureSetId: 'feature/one', artifactId: 'pack/one', current: true }))
      .mockResolvedValueOnce(json({ featureSetId: 'feature/one', artifactId: null, current: true }));
    vi.stubGlobal('fetch', fetcher);
    const { packing } = await import('./packing');
    await expect(packing.selection('project', 'feature/one')).resolves.toMatchObject({ artifactId: 'pack/one' });
    await packing.select('project', 'feature/one', 'pack/one');
    await packing.select('project', 'feature/one', null);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project/features/feature%2Fone/pack-selection');
    expect(fetcher.mock.calls[2][1].method).toBe('PUT');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ artifactId: 'pack/one' });
    expect(JSON.parse(fetcher.mock.calls[3][1].body)).toEqual({ artifactId: null });
  });
});

describe('feature job review identity and active states', () => {
  it('invalidates review when source version, action, destination or precision changes', () => {
    const original = featurePackSpecKey(spec);
    for (const change of [{ featureSetId: 'other' }, { action: 'validate' as const }, { dtype: 'float16' as const }, { outputPath: '/other' }]) {
      expect(featurePackSpecKey({ ...spec, ...change })).not.toBe(original);
    }
    expect(featurePackSpecKey({ ...spec, outputPath: ' /packs/one ' })).toBe(original);
    expect(featurePackSpecKey({ ...spec, action: 'validate', outputPath: '/ignored' })).toBe(featurePackSpecKey({ ...spec, action: 'validate', outputPath: null }));
    const attached = { ...spec, action: 'attach' as const, existingPath: '/mmap/one' };
    expect(featurePackSpecKey(attached)).not.toBe(featurePackSpecKey({ ...attached, existingPath: '/mmap/two' }));
    expect(featurePackSpecKey(attached)).toBe(featurePackSpecKey({ ...attached, existingPath: ' /mmap/one ', outputPath: '/ignored' }));
  });

  it('keeps cancelling workers active while all terminal states stop polling', () => {
    for (const state of ['starting', 'running', 'cancelling'] as const) expect(featurePackActive({ state })).toBe(true);
    for (const state of ['succeeded', 'failed', 'cancelled', 'interrupted'] as const) expect(featurePackActive({ state })).toBe(false);
    expect(featurePackActive(undefined)).toBe(false);
  });
});
