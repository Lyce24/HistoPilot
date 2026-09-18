import type { ScientificDraft } from '../api/scientific';
import { readSessionDraft, useSessionDraftBackup, writeSessionDraft } from './sessionDraft';

/**
 * Unsaved scientific editor input, retained for this browser tab only.
 *
 * The project folder still owns every saved draft and frozen version. A recovery
 * copy restores the editor after module navigation or a reload; it never becomes
 * a saved record on its own and never replaces a newer server revision.
 */
export interface EditorRecovery<S, D = ScientificDraft<S>> {
  version: 1;
  /** The edited name, which may differ from the saved draft name. */
  name: string;
  /** The edited specification, which may differ from the saved draft payload. */
  spec: S;
  /** The saved draft this editor continues, or null for work never saved. */
  draft: D | null;
  /** The open step, so recovery returns to the same place in the workflow. */
  step: number;
}

export type EditorRecoveryKind = 'dataset' | 'protocol' | 'test-cohort';

export const editorRecoveryKey = (project: string, kind: EditorRecoveryKind) =>
  `histopilot:editor-draft:v1:${encodeURIComponent(project)}:${kind}`;

const object = (value: unknown): value is Record<string, unknown> =>
  Boolean(value && typeof value === 'object' && !Array.isArray(value));

function isDraft(value: unknown): boolean {
  return object(value)
    && typeof value.id === 'string' && value.id.length > 0 && value.id.length <= 256
    && typeof value.name === 'string' && value.name.length <= 400
    && typeof value.revision === 'number' && Number.isSafeInteger(value.revision) && value.revision >= 0
    && (value.status === 'editable' || value.status === 'frozen')
    && object(value.payload) && object(value.payload.spec);
}

/** Structure only: the service revalidates every specification before saving or freezing. */
export function isEditorRecovery<S, D>(value: unknown): value is EditorRecovery<S, D> {
  return object(value)
    && value.version === 1
    && typeof value.name === 'string' && value.name.length <= 400
    && object(value.spec)
    && (value.draft === null || isDraft(value.draft))
    && typeof value.step === 'number' && Number.isSafeInteger(value.step) && value.step >= 0;
}

export function readEditorRecovery<S, D = ScientificDraft<S>>(
  project: string,
  kind: EditorRecoveryKind,
): EditorRecovery<S, D> | null {
  return readSessionDraft<EditorRecovery<S, D>>(editorRecoveryKey(project, kind), isEditorRecovery);
}

export function clearEditorRecovery(project: string, kind: EditorRecoveryKind): void {
  writeSessionDraft(editorRecoveryKey(project, kind), null);
}

/**
 * Keep a recovery copy of the open editor while it holds unsaved input.
 *
 * Pass null once the editor matches its saved record, so a completed save or
 * freeze does not leave stale input behind. The returned error explains that
 * recovery is unavailable; callers guard navigation with it.
 */
export function useEditorRecoveryBackup<S, D = ScientificDraft<S>>(
  project: string,
  kind: EditorRecoveryKind,
  value: EditorRecovery<S, D> | null,
): { error: string | null } {
  return useSessionDraftBackup(editorRecoveryKey(project, kind), value, isEditorRecovery);
}

/**
 * A recovered step, clamped to the last step the editor can render from saved
 * input alone. Review and freeze steps depend on a server check that recovery
 * deliberately does not restore, so they reopen at the step before them.
 */
export function recoveredStep(step: number | undefined, lastRestorable: number): number {
  return typeof step === 'number' && step >= 1 ? Math.min(step, lastRestorable) : 0;
}
