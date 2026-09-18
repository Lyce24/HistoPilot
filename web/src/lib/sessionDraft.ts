import { useLayoutEffect, useState } from 'react';

const MAX_DRAFT_BYTES = 1_000_000;
export const sessionDraftKey = (project: string, experiment: string, kind: string) =>
  `histopilot:experiment-draft:v1:${encodeURIComponent(project)}:${encodeURIComponent(experiment)}:${encodeURIComponent(kind)}`;

/** Browser drafts are recovery copies, never authoritative saved experiment inputs. */
export function readSessionDraft<T>(key: string, validate: (value: unknown) => value is T): T | null {
  try {
    const raw = window.sessionStorage.getItem(key);
    if (!raw || raw.length > MAX_DRAFT_BYTES) return null;
    const envelope: unknown = JSON.parse(raw);
    if (!envelope || typeof envelope !== 'object' || !('version' in envelope) || envelope.version !== 1 || !('value' in envelope)) return null;
    return validate(envelope.value) ? envelope.value : null;
  } catch { return null; }
}

export function writeSessionDraft<T>(key: string, value: T | null): boolean {
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else {
      const raw = JSON.stringify({ version: 1, value });
      if (raw.length > MAX_DRAFT_BYTES) return false;
      window.sessionStorage.setItem(key, raw);
    }
    return true;
  } catch { return false; }
}

/** Commit before the next navigation event, including edits followed immediately by a click. */
export function useSessionDraftBackup<T>(key: string, value: T | null, validate?: (value: unknown) => value is T): { error: string | null } {
  const [error, setError] = useState<string | null>(null);
  const serialized = JSON.stringify(value);
  const restorable = value === null || !validate || validate(value);
  useLayoutEffect(() => {
    const saved = restorable && writeSessionDraft(key, JSON.parse(serialized));
    setError(saved ? null : 'Browser draft recovery is unavailable. Save your changes before leaving this page.');
  }, [key, serialized, restorable]);
  return { error };
}
