import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  clearEditorRecovery,
  editorRecoveryKey,
  isEditorRecovery,
  readEditorRecovery,
  recoveredStep,
} from './editorRecovery';
import { writeSessionDraft } from './sessionDraft';

afterEach(() => vi.unstubAllGlobals());

function storage() {
  const entries = new Map<string, string>();
  vi.stubGlobal('window', {
    sessionStorage: {
      getItem: (key: string) => entries.get(key) ?? null,
      setItem: (key: string, value: string) => entries.set(key, value),
      removeItem: (key: string) => entries.delete(key),
    },
  });
  return entries;
}

const draft = {
  id: 'draft-1', projectId: 'project', kind: 'import', name: 'Saved import', revision: 3,
  status: 'editable', createdAt: '', updatedAt: '',
  payload: { type: 'dataset-import', spec: { source: { path: '/a.csv' } } },
};
const recovery = { version: 1, name: 'Edited import', spec: { source: { path: '/b.csv' } }, draft, step: 1 };

describe('scientific editor recovery', () => {
  it('restores unsaved editor input for the same project and editor only', () => {
    storage();
    writeSessionDraft(editorRecoveryKey('project', 'dataset'), recovery);
    expect(readEditorRecovery('project', 'dataset')).toEqual(recovery);
    expect(readEditorRecovery('project', 'protocol')).toBeNull();
    expect(readEditorRecovery('other', 'dataset')).toBeNull();
    clearEditorRecovery('project', 'dataset');
    expect(readEditorRecovery('project', 'dataset')).toBeNull();
  });

  it('keeps input that was never saved as a draft', () => {
    storage();
    writeSessionDraft(editorRecoveryKey('project', 'test-cohort'), { ...recovery, draft: null });
    expect(readEditorRecovery('project', 'test-cohort')?.draft).toBeNull();
  });

  it('rejects entries that could not have come from this editor', () => {
    const entries = storage();
    const key = editorRecoveryKey('project', 'protocol');
    for (const value of [
      { ...recovery, version: 2 },
      { ...recovery, spec: 'not an object' },
      { ...recovery, step: -1 },
      { ...recovery, step: 1.5 },
      { ...recovery, draft: { ...draft, revision: 'three' } },
      { ...recovery, draft: { ...draft, status: 'published' } },
      { ...recovery, draft: { ...draft, payload: { type: 'analysis-protocol' } } },
    ]) {
      expect(isEditorRecovery(value)).toBe(false);
      entries.set(key, JSON.stringify({ version: 1, value }));
      expect(readEditorRecovery('project', 'protocol')).toBeNull();
    }
  });

  it('reopens before a review step, because recovery never restores a server check', () => {
    expect(recoveredStep(undefined, 3)).toBe(0);
    expect(recoveredStep(0, 3)).toBe(0);
    expect(recoveredStep(2, 3)).toBe(2);
    expect(recoveredStep(4, 3)).toBe(3);
  });

  it('does not fail the editor when browser storage is unavailable', () => {
    vi.stubGlobal('window', { get sessionStorage() { throw new Error('Storage denied'); } });
    expect(readEditorRecovery('project', 'dataset')).toBeNull();
    expect(() => clearEditorRecovery('project', 'dataset')).not.toThrow();
  });
});
