import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanupApplyRequest, cleanupJobActive, cleanupPollInterval, cleanupReviewMatches } from './lifecycle';
import type { CleanupInventory, CleanupItem, CleanupPreview } from './lifecycle';

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });
const inventory = { projectId: 'p', revision: 4, projectState: 'active', items: [], audit: [], note: '' } as CleanupInventory;
const preview = { action: 'trash', keys: ['dataset:a', 'configuration:b'], revision: 4, previewHash: 'review-hash', canApply: true, blockers: [], requiredKeys: [], recordCount: 2, note: '' } satisfies CleanupPreview;
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });

describe('workspace cleanup contracts', () => {
  it('requires an exact reviewed selection, action and project revision', () => {
    expect(cleanupReviewMatches(preview, inventory, 'trash', [...preview.keys].reverse())).toBe(true);
    expect(cleanupReviewMatches(preview, { ...inventory, revision: 5 }, 'trash', preview.keys)).toBe(false);
    expect(cleanupReviewMatches(preview, inventory, 'archive', preview.keys)).toBe(false);
    expect(cleanupReviewMatches(preview, inventory, 'trash', ['dataset:a'])).toBe(false);
    expect(cleanupReviewMatches(preview, inventory, 'trash', ['dataset:a', 'dataset:a'])).toBe(false);
    expect(cleanupReviewMatches({ ...preview, keys: ['dataset:a', 'dataset:a'] }, inventory, 'trash', preview.keys)).toBe(false);
    expect(cleanupReviewMatches(null, inventory, 'trash', preview.keys)).toBe(false);
  });

  it('never adds dependent records to a confirmation implicitly', () => {
    const required = { ...preview, requiredKeys: ['configuration:downstream'] };
    expect(() => cleanupApplyRequest(required, 'one-intent')).toThrow('resolve every blocker');
    expect(required.keys).toEqual(preview.keys);
    expect(() => cleanupApplyRequest({ ...preview, canApply: false }, 'one-intent')).toThrow();
    expect(() => cleanupApplyRequest({ ...preview, blockers: [{ code: 'ACTIVE_JOB', message: 'Stop the job', keys: ['configuration:b'] }] }, 'one-intent')).toThrow();
  });

  it('polls cancellation and active work until jobs become terminal', () => {
    for (const status of ['queued', 'pending', 'active', 'running', 'cancelling', 'stopping', 'cancel_requested']) {
      const item = { job: { status, cancellable: false } } as CleanupItem;
      expect(cleanupJobActive(item)).toBe(true);
      expect(cleanupPollInterval({ ...inventory, items: [item] })).toBe(3000);
    }
    expect(cleanupPollInterval({ ...inventory, items: [{ job: { status: 'cancelled', cancellable: false } } as CleanupItem] })).toBe(15000);
    expect(cleanupJobActive({} as CleanupItem)).toBe(false);
  });

  it('previews only the explicit keys before any apply request', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response(preview));
    vi.stubGlobal('fetch', fetcher);
    const { lifecycle } = await import('./lifecycle');
    await lifecycle.preview('project/one', 'trash', ['dataset:a']);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/cleanup/preview');
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ action: 'trash', keys: ['dataset:a'] });
    expect(fetcher.mock.calls.some(([path]) => String(path).endsWith('/apply'))).toBe(false);
  });

  it('retries the same reviewed operation after a lost response without resending changed selection', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockRejectedValueOnce(new TypeError('Network response lost'))
      .mockResolvedValueOnce(response({ action: 'trash', revision: 5, changed: preview.keys }));
    vi.stubGlobal('fetch', fetcher);
    const { lifecycle } = await import('./lifecycle');
    const intent = cleanupApplyRequest(preview, 'stable-operation');
    await expect(lifecycle.apply('project', intent)).rejects.toThrow('Network response lost');
    await lifecycle.apply('project', intent);
    expect(fetcher.mock.calls[1][1].body).toBe(fetcher.mock.calls[2][1].body);
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ action: 'trash', keys: preview.keys, previewHash: 'review-hash', operationId: 'stable-operation' });
    expect(intent.keys).not.toBe(preview.keys);
  });

  it('preserves stale-preview errors and sends cancellation using a caller-owned stable ID', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' }))
      .mockResolvedValueOnce(response({ detail: 'Review is stale', code: 'STALE_PREVIEW' }, 409))
      .mockResolvedValueOnce(response({ key: 'packing:one', job: { status: 'cancelling' } }));
    vi.stubGlobal('fetch', fetcher);
    const { lifecycle } = await import('./lifecycle');
    await expect(lifecycle.apply('p', cleanupApplyRequest(preview, 'intent'))).rejects.toMatchObject({ status: 409, code: 'STALE_PREVIEW' });
    await lifecycle.cancel('p', 'packing:one', 'cancel-intent');
    expect(fetcher.mock.calls[2][0]).toBe('/api/v1/projects/p/cleanup/cancel');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ key: 'packing:one', operationId: 'cancel-intent' });
  });
});
