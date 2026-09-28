import type { ComputeExecution } from '../api/predictors';

/**
 * Records created before the Task Center (2026-09-27) ran in their own tmux session. The service
 * no longer runs, cancels or watches them: they keep their saved status and are read-only.
 * (Unrelated to the experiment registry's `legacy` flag, which marks historical experiment drafts.)
 */
export const legacyRecordNote = 'Created before the Task Center. This record is read-only; clone it or preview it again to run it.';

/** Stored records say "tmux" or nothing; only Task Center records say "task-center". */
export const createdBeforeTaskCenter = (record?: { executor?: string } | null) => Boolean(record && record.executor !== 'task-center');

/** A compute job that ran before the Task Center; a saved record never launched is not one. */
export const computeCreatedBeforeTaskCenter = (job?: ComputeExecution | null) => Boolean(job && job.status !== 'not_started' && createdBeforeTaskCenter(job));

export default function LegacyRecordNote() {
  return <p className="muted legacy-record-note" role="note">{legacyRecordNote}</p>;
}
