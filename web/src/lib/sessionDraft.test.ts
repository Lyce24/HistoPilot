import { afterEach, describe, expect, it, vi } from 'vitest';
import { readSessionDraft, sessionDraftKey, writeSessionDraft } from './sessionDraft';

afterEach(() => vi.unstubAllGlobals());

describe('experiment browser recovery', () => {
  function storage() {
    const entries = new Map<string, string>();
    const sessionStorage = { getItem: (key: string) => entries.get(key) ?? null, setItem: (key: string, value: string) => entries.set(key, value), removeItem: (key: string) => entries.delete(key) };
    vi.stubGlobal('window', { sessionStorage });
    return sessionStorage;
  }
  const valid = (value: unknown): value is { raw: string; revision: number } => Boolean(value && typeof value === 'object' && 'raw' in value && typeof value.raw === 'string' && 'revision' in value && typeof value.revision === 'number');

  it('retains incomplete input together with its saved baseline and isolates records and projects', () => {
    storage();
    const key = sessionDraftKey('project:one', 'experiment', 'batch');
    expect(writeSessionDraft(key, { raw: '42,', revision: 7 })).toBe(true);
    expect(readSessionDraft(key, valid)).toEqual({ raw: '42,', revision: 7 });
    expect(readSessionDraft(sessionDraftKey('project', 'one:experiment', 'batch'), valid)).toBeNull();
    expect(readSessionDraft(sessionDraftKey('project:one', 'other', 'batch'), valid)).toBeNull();
    expect(writeSessionDraft(key, null)).toBe(true);
    expect(readSessionDraft(key, valid)).toBeNull();
  });

  it('ignores malformed, incompatible and oversized storage without applying it to a plan', () => {
    const entries = storage();
    for (const raw of ['{broken', JSON.stringify({ version: 2, value: { raw: '42', revision: 1 } }), JSON.stringify({ version: 1, value: { raw: 42 } }), ' '.repeat(1_000_001)]) {
      entries.setItem('key', raw);
      expect(readSessionDraft('key', valid)).toBeNull();
    }
  });

  it('reports unavailable or full storage instead of claiming edits were recovered', () => {
    vi.stubGlobal('window', { get sessionStorage() { throw new Error('Storage denied'); } });
    expect(readSessionDraft('key', valid)).toBeNull();
    expect(writeSessionDraft('key', { raw: '42', revision: 1 })).toBe(false);
    storage();
    expect(writeSessionDraft('key', { raw: 'x'.repeat(1_000_001), revision: 1 })).toBe(false);
  });
});
