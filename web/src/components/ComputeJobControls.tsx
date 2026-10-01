import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { computeActive, computeStatusLabel, modelEvaluations, predictors, type ComputeExecution } from '../api/predictors';
import { interpretations } from '../api/interpretation';
import { rollupIsLive, taskCenterKeys, type TaskRollup } from '../api/taskCenter';
import { definiteRejection } from '../lib/taskCenterActions';
import RunStatusChip, { useRunRollup } from './RunStatusChip';
import LegacyRecordNote, { computeCreatedBeforeTaskCenter } from './LegacyRecordNote';
import { Badge, ErrorNotice } from './ui';

type Kind = 'refit' | 'evaluation' | 'interpretation';
type Props = {
  project: string; id: string; kind: Kind; initial?: ComputeExecution;
  readOnly?: boolean; readOnlyReason?: string; onComplete?: () => void;
  /** `row` on a record page; `chip` inside a list or a slide panel. */
  variant?: 'chip' | 'row';
};
const queryFamily: Record<Kind, string> = { refit: 'refit-builds', interpretation: 'interpretations', evaluation: 'model-evaluations' };

/** Requests and their retry IDs belong to exactly one compute record. */
export default function ComputeJobControls(props: Props) {
  return <ComputeJobState key={JSON.stringify([props.project, props.kind, props.id])} {...props} />;
}

/** A run's job predicts its cohort, labeled or not; the service scores a labeled one afterwards. */
export const computeLaunchLabel = (kind: Kind) => kind === 'refit' ? 'Train refit model' : kind === 'interpretation' ? 'Compute slide attention' : 'Run predictions';
/** Every stage action that requeues stopped work says "Resume" (see the Area C log). */
export const computeRetryLabel = () => 'Resume';
const stopped = (job?: ComputeExecution) => Boolean(job && ['failed', 'cancelled', 'interrupted'].includes(job.status));

/**
 * One compute record's execution, shared by the record page and its controls. A list or a
 * record page hands in what it last read as `initial`, but that can predate a change made
 * in the Task Center (the job finished, or a cancel or failure was applied there), so it
 * counts as stale and is read again on mount. The task store, not the record, is followed.
 */
export function computeExecutionQuery(project: string, kind: Kind, id: string, initial: ComputeExecution | undefined, enabled: boolean) {
  return {
    queryKey: ['compute-job', project, kind, id],
    queryFn: () => kind === 'refit' ? predictors.refitExecution(project, id) : kind === 'interpretation' ? interpretations.execution(project, id) : modelEvaluations.execution(project, id),
    initialData: initial, initialDataUpdatedAt: 0, enabled,
  };
}

/**
 * The task store says the record's work is over (completed, failed or cancelled) while the
 * record still says queued or running: it was read before the Task Center changed it.
 */
export const recordBehindRollup = (rollup: TaskRollup | undefined, record: ComputeExecution | undefined) =>
  Boolean(rollup && record && rollup.state !== 'not-started' && !rollupIsLive(rollup) && computeActive(record) && record.executor === 'task-center');

/**
 * One compute record's science action (Run, or Retry/Resume when it stopped short) and its run
 * status. Task Center jobs show the shared status chip, which links to the task for its
 * queue place, log, resources and Cancel; the record is re-read once the task settles.
 * Jobs created before the Task Center are read-only: their saved status and a note.
 */
function ComputeJobState({ project, id, kind, initial, readOnly = false, readOnlyReason, onComplete, variant = 'row' }: Props) {
  const client = useQueryClient();
  // Cleanup listings include historical status for trashed records. Their
  // ordinary execution endpoint deliberately rejects new direct access.
  const shouldRead = !readOnly || !initial || computeActive(initial);
  const query = computeExecutionQuery(project, kind, id, initial, shouldRead);
  const { queryKey } = query;
  const job = useQuery(query);
  const [pending, setPending] = useState<{ action: 'launch' | 'resume'; operation: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const state = shouldRead ? job.data : initial ?? job.data;
  const active = computeActive(state);
  // Task Center jobs always say so; older records (executor "tmux" or absent) are read-only.
  const legacy = computeCreatedBeforeTaskCenter(state);
  const followed = Boolean(state && !legacy && state.status !== 'not_started');
  // The chip reads the same rollup (one cache entry).
  const rollup = useRunRollup({ recordKind: kind, recordId: id, project }, followed).data;
  const settledFor = useRef<string | null>(null);
  async function run(action: 'launch' | 'resume') {
    if (submitting.current || (!pending && readOnly)) return;
    submitting.current = true;
    const request = pending ?? { action, operation: crypto.randomUUID() };
    setPending(request); setBusy(true); setError(null);
    try {
      const result = kind === 'refit' ? await predictors.refitJob(project, id, request.action, request.operation) : kind === 'interpretation' ? await interpretations.job(project, id, request.action, request.operation) : await modelEvaluations.job(project, id, request.action, request.operation);
      client.setQueryData(queryKey, result); setPending(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('Job action failed.'));
      // A definite refusal can be sent again as a new request; a lost response or a server
      // error may have been applied, so a repeat reuses the same operation ID.
      if (definiteRejection(reason)) setPending(null);
    } finally { submitting.current = false; setBusy(false); }
    // A failed refresh must not replay a successfully accepted action.
    void client.invalidateQueries({ queryKey: [queryFamily[kind], project] });
    void client.invalidateQueries({ queryKey: ['cleanup', project] });
    void client.invalidateQueries({ queryKey: taskCenterKeys.all });
  }
  function settled(update: TaskRollup) {
    // The chip's live-to-done transition and the stale-record check see the same update.
    if (settledFor.current === update.updatedAt) return;
    settledFor.current = update.updatedAt;
    void client.invalidateQueries({ queryKey });
    void client.invalidateQueries({ queryKey: [queryFamily[kind], project] });
    onComplete?.();
  }
  // Reached from a Task Center link after the job finished, or changed there: the chip never
  // saw it live, so read the record again whenever the task store says the work is over.
  const behind = followed && recordBehindRollup(rollup, state);
  useEffect(() => { if (behind && rollup) settled(rollup); }, [behind, rollup]);
  const retry = !pending && !readOnly && !legacy && stopped(state)
    ? <button type="button" className="btn btn-primary btn-small" disabled={busy || job.isError} onClick={() => void run('resume')}>{computeRetryLabel()}</button> : null;
  return <div className="compute-job-controls">
    <ErrorNotice error={error ?? (shouldRead ? job.error : null)} />
    {state?.progressWarning ? <p className="callout callout-warning" role="status">{state.progressWarning}</p> : null}
    {shouldRead && job.isError ? <><p className="muted">{state ? 'Showing the last loaded job status. Refresh to check the current state.' : 'Job status could not be loaded. Refresh before starting work.'}</p><button type="button" className="btn btn-secondary" disabled={job.isFetching} onClick={() => void job.refetch()}>{job.isFetching ? 'Refreshing…' : 'Retry job status'}</button></> : null}
    {pending && !busy ? <p className="callout callout-warning" role="status">The {pending.action} response was lost. The request may already have been accepted. Retry uses the same request and operation ID.</p> : null}
    {!state ? <p role="status"><Badge>{job.isError ? 'Job status unavailable' : 'Checking job status…'}</Badge></p>
      : state.status === 'not_started' ? <div className="inline-actions">
        <Badge>{computeStatusLabel(state)}</Badge>
        {!pending && !readOnly ? <button className="btn btn-primary" disabled={busy || job.isError} onClick={() => void run('launch')}>{computeLaunchLabel(kind)}</button> : null}
      </div>
        : legacy ? <LegacyJob state={state} />
          : <RunStatusChip scope={{ recordKind: kind, recordId: id, project }} variant={variant} primaryAction={retry} onSettled={settled} notStartedText={computeStatusLabel(state)} />}
    {pending && !busy ? <div className="inline-actions"><button className="btn btn-secondary" onClick={() => void run(pending.action)}>Retry {pending.action} request</button></div> : null}
    {readOnly && !legacy && state?.status !== 'completed' && !active ? <p className="muted">{readOnlyReason ?? 'Restore this record to run it.'}</p> : null}
  </div>;
}

/** A job created before the Task Center: its saved status, read-only. */
function LegacyJob({ state }: { state: ComputeExecution }) {
  return <>
    <div className="inline-actions">
      <Badge tone={state.status === 'completed' ? 'success' : 'neutral'}>{computeStatusLabel(state)}</Badge>
      {state.progress?.epoch !== undefined ? <span>Epoch {state.progress.epoch} / {state.progress.maxEpochs}</span> : null}
    </div>
    {state.error ? <p className="callout" role="status">{state.error}</p> : null}
    <LegacyRecordNote />
    {state.logPath ? <details><summary>Job log</summary><code className="record-path">{state.logPath}</code></details> : null}
  </>;
}
