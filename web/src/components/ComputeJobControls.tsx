import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { computeActive, computeStatusLabel, modelEvaluations, predictors, type ComputeExecution } from '../api/predictors';
import { interpretations } from '../api/interpretation';
import { Badge, ErrorNotice } from './ui';

/** Durable jobs have one explicit action and stable operation ID across lost responses. */
export default function ComputeJobControls({ project, id, kind, initial, readOnly = false, readOnlyReason, onComplete }: {
  project: string; id: string; kind: 'refit' | 'evaluation' | 'interpretation'; initial?: ComputeExecution;
  readOnly?: boolean; readOnlyReason?: string; onComplete?: () => void;
}) {
  const client = useQueryClient();
  const queryKey = ['compute-job', project, kind, id];
  // Cleanup listings include historical status for trashed records. Their
  // ordinary execution endpoint deliberately rejects new direct access.
  const shouldPoll = !readOnly || !initial || computeActive(initial);
  const job = useQuery({ queryKey, queryFn: () => kind === 'refit' ? predictors.refitExecution(project, id) : kind === 'interpretation' ? interpretations.execution(project, id) : modelEvaluations.execution(project, id), initialData: initial,
    enabled: shouldPoll, refetchInterval: (query) => shouldPoll ? computeActive(query.state.data) ? 2000 : 10000 : false });
  const [pending, setPending] = useState<{ action: 'launch' | 'resume' | 'cancel'; operation: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const state = shouldPoll ? job.data : initial ?? job.data;
  const active = computeActive(state);
  async function run(action: 'launch' | 'resume' | 'cancel') {
    if (busy) return;
    const request = pending ?? { action, operation: crypto.randomUUID() };
    setPending(request); setBusy(true); setError(null);
    try {
      const result = kind === 'refit' ? await predictors.refitJob(project, id, request.action, request.operation) : kind === 'interpretation' ? await interpretations.job(project, id, request.action, request.operation) : await modelEvaluations.job(project, id, request.action, request.operation);
      client.setQueryData(queryKey, result); setPending(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('Job action failed.'));
      if (reason instanceof ApiError) setPending(null);
    } finally { setBusy(false); }
    // A failed refresh must not replay a successfully accepted action.
    void client.invalidateQueries({ queryKey: [kind === 'refit' ? 'refit-builds' : kind === 'interpretation' ? 'interpretations' : 'model-evaluations', project] });
    void client.invalidateQueries({ queryKey: ['cleanup', project] });
  }
  return <div className="compute-job-controls">
    <p role="status"><Badge tone={state?.status === 'completed' ? 'success' : active ? 'warning' : 'neutral'}>{computeStatusLabel(state)}</Badge></p>
    {state?.progress?.epoch !== undefined ? <p>Epoch {state.progress.epoch} / {state.progress.maxEpochs}{typeof state.progress.trainingLoss === 'number' ? ` · training loss ${state.progress.trainingLoss.toFixed(4)}` : ''}</p> : null}
    {kind === 'interpretation' && state?.progress?.completedPairs !== undefined ? <p>{state.progress.completedPairs} / {state.progress.totalPairs} slide–checkpoint pairs computed{state.progress.currentSlide ? ` · current slide: ${state.progress.currentSlide}` : ''}</p> : state?.progress?.completedModels !== undefined ? <p>{state.progress.completedModels} / {state.progress.totalModels} model checkpoints evaluated{state.progress.slideCount !== undefined ? ` · ${state.progress.slideCount} slides` : ''}</p> : null}
    <ErrorNotice error={error ?? (shouldPoll ? job.error : null) ?? (state?.error ? new Error(state.error) : null)} />
    <div className="inline-actions">
      {pending && !busy ? <button className="btn btn-secondary" onClick={() => void run(pending.action)}>Retry {pending.action} request</button> : null}
      {!pending && !readOnly && state?.status === 'not_started' ? <button className="btn btn-primary" disabled={busy || job.isError} onClick={() => void run('launch')}>{kind === 'refit' ? 'Train refit model' : kind === 'interpretation' ? 'Compute slide attention' : 'Run evaluation'}</button> : null}
      {!pending && !readOnly && state && ['failed', 'cancelled', 'interrupted'].includes(state.status) ? <button className="btn btn-secondary" disabled={busy || job.isError} onClick={() => void run('resume')}>{kind === 'refit' ? 'Resume refit training' : kind === 'interpretation' ? 'Retry attention computation' : 'Retry evaluation'}</button> : null}
      {!pending && active ? <button className="btn btn-secondary" disabled={busy || state?.cancellationRequested} onClick={() => void run('cancel')}>{state?.cancellationRequested ? 'Cancellation requested…' : 'Cancel job'}</button> : null}
      {onComplete && state?.status === 'completed' ? <button className="btn btn-secondary" onClick={onComplete}>Refresh completed result</button> : null}
    </div>
    {state?.logPath ? <details><summary>Job log and session</summary><code className="record-path">{state.logPath}</code>{state.sessionName ? <code className="record-path">tmux attach -t {state.sessionName}</code> : null}</details> : null}
    {active ? <p className="muted">This job continues after you leave this page. Cancellation waits for the worker to stop before records can be deleted.</p> : null}
    {readOnly && state?.status !== 'completed' && !active ? <p className="muted">{readOnlyReason ?? 'Restore this record to run it.'}</p> : null}
  </div>;
}
