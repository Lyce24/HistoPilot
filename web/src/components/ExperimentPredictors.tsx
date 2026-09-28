import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { experimentStage, experiments, type ExperimentPredictorExecution, type ModelExperiment } from '../api/experiments';
import { predictors, predictorMethodLabel, type FrozenPredictor } from '../api/predictors';
import { taskCenterHref } from '../api/taskCenter';
import { predictorConfigurationLabel, predictorMatches } from '../lib/predictorGroups';
import { batchPredictorPolicy } from '../lib/experimentPredictors';
import { Badge, ErrorNotice, Panel } from './ui';
import LegacyRecordNote, { createdBeforeTaskCenter } from './LegacyRecordNote';
import './ExperimentPredictors.css';

const statusLabel: Record<ExperimentPredictorExecution['status'], string> = {
  queued: 'Queued', waiting: 'Waiting', running: 'Creating predictors', cancelling: 'Cancelling',
  completed: 'Predictors complete', cancelled: 'Predictor creation cancelled', attention: 'Needs attention', interrupted: 'Interrupted',
};
const pageSize = 25;

/**
 * Predictors of an experiment. `progress` (the Runs tab) says how many exist and how many
 * are still to come; Task Center experiments leave jobs, logs, cancel and resume to the Task
 * Center and the experiment status line. `library` (the Results tab) lists ready predictors.
 * Coordinators created before the Task Center keep their job table here, read-only.
 */
/** A coordinator started before the Task Center; one cancelled before it ever started has no session. */
export const coordinatorCreatedBeforeTaskCenter = (execution: ExperimentPredictorExecution) => createdBeforeTaskCenter(execution) && Boolean(execution.sessionName);

export default function ExperimentPredictors({ project, record, view = 'all' }: { project: string; record: ModelExperiment; view?: 'all' | 'progress' | 'library' }) {
  const stage = experimentStage(record);
  const managed = record.predictorExecution?.executor === 'task-center';
  const showLibrary = view !== 'progress';
  const query = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), enabled: showLibrary || !managed, refetchInterval: stage === 'running' && showLibrary ? 30000 : false });
  const [method, setMethod] = useState('all');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState<number | null>(null);
  const [workPage, setWorkPage] = useState(0);
  const [workFilter, setWorkFilter] = useState('all');
  const items = (query.data?.items ?? []).filter((item) => item.manifest.experimentId === record.id && item.lifecycleState !== 'trashed');
  const ready = items.filter((item) => item.lifecycleState === 'active');
  const filtered = items.filter((item) => predictorMatches(item, record.id, method, search));
  const selectedPredictor = typeof window === 'undefined' ? null : new URLSearchParams(window.location.hash.split('?')[1]).get('predictor');
  const linkedPage = Math.floor(Math.max(0, filtered.findIndex((item) => item.id === selectedPredictor)) / pageSize);
  const currentPage = Math.min(page ?? linkedPage, Math.max(0, Math.ceil(filtered.length / pageSize) - 1));
  const execution = record.predictorExecution;
  const hasPredictorPlan = Boolean(record.predictorPolicies || record.predictorPolicy);
  const policies = record.predictorPolicies ? Object.values(record.predictorPolicies) : record.batches.map((batch) => batchPredictorPolicy(batch.manifest.spec, record.predictorPolicy));
  const skipped = policies.length > 0 ? policies.every((policy) => policy.method === 'skip') : record.predictorPolicy?.method === 'skip';
  const workItems = (execution?.items ?? []).filter((item) => workFilter === 'all' || item.status === workFilter);
  const currentWorkPage = Math.min(workPage, Math.max(0, Math.ceil(workItems.length / pageSize) - 1));
  const batchNames = new Map(record.batches.map((batch) => [batch.id, batch.manifest.spec.batchName]));
  const configurations = new Map(record.batches.flatMap((batch) => batch.manifest.configurations.map((candidate) => [JSON.stringify([batch.id, candidate.id]), candidate.number] as const)));
  const evaluationLink = `#evaluate-models?${new URLSearchParams({ experiment: record.id })}`;

  const creation = view !== 'library';
  if (view === 'progress' && managed && execution) {
    const planned = execution.counts.total - (execution.counts.skipped ?? 0);
    return <section className="experiment-predictor-summary" aria-label="Predictor creation">
      <strong>Predictors: {execution.counts.completed.toLocaleString()} of {planned.toLocaleString()} created</strong>
      <progress aria-label="Predictors created" value={execution.counts.completed} max={Math.max(1, planned)} />
      <span className="muted">{execution.counts.ensemble} ensembles · {execution.counts.refit} refits{execution.counts.waiting ? ` · ${execution.counts.waiting} wait for their batch` : ''}{execution.counts.skipped ? ` · ${execution.counts.skipped} omitted by validation selection` : ''}. Ready predictors are listed under Predictors; jobs, logs and failures are in the Task Center.</span>
    </section>;
  }
  return <Panel title={creation && stage === 'running' ? 'Predictor creation' : 'Predictors'} subtitle="One predictor per method and complete configuration / training-seed / split-seed group.">
    {execution && creation ? <div className="experiment-predictor-progress">
      <div className="inline-actions"><Badge tone={execution.status === 'completed' ? 'success' : ['attention', 'interrupted'].includes(execution.status) ? 'warning' : 'neutral'}>{statusLabel[execution.status]}</Badge><strong>{execution.counts.completed.toLocaleString()} / {(execution.counts.total - (execution.counts.skipped ?? 0)).toLocaleString()} created</strong></div>
      <progress aria-label="Predictors created" value={execution.counts.completed} max={Math.max(1, execution.counts.total - (execution.counts.skipped ?? 0))} />
      <p className="muted">{execution.counts.ensemble} ensembles · {execution.counts.refit} refits · {execution.counts.waiting} waiting · {execution.counts.active} queued or running{execution.counts.skipped ? ` · ${execution.counts.skipped} omitted by validation selection` : ''}{execution.counts.failed ? ` · ${execution.counts.failed} failed` : ''}{execution.counts.cancelled ? ` · ${execution.counts.cancelled} cancelled` : ''}</p>
      {execution.counts.waiting ? <p className="muted">Waiting jobs need their source batch to complete. Predictor jobs run through the <a href={taskCenterHref({ project })}>Task Center</a>.</p> : null}
      {execution.executor === 'task-center' && execution.runnerAlive === false && ['queued', 'waiting', 'running'].includes(execution.status) ? <p className="callout" role="status">The Task Center runner is not running; predictor jobs start once it runs. <a className="text-link" href={taskCenterHref({ project })}>Start it in the Task Center →</a></p> : null}
      {execution.error ? <p className="callout" role="status">{execution.error.message}</p> : null}
      {coordinatorCreatedBeforeTaskCenter(execution) ? <LegacyRecordNote /> : stage === 'running' && record.state === 'active' ? <PredictorActions project={project} record={record} execution={execution} /> : null}
      {execution.items?.length ? <details open={stage === 'running'}><summary>Predictor jobs · {execution.items.length}</summary>
        <label className="label experiment-predictor-tools">Job status<select className="field" value={workFilter} onChange={(event) => { setWorkFilter(event.target.value); setWorkPage(0); }}><option value="all">All jobs</option><option value="waiting">Waiting</option><option value="queued">Queued</option><option value="running">Running</option><option value="completed">Completed</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option></select></label>
        <div className="experiment-predictor-table"><table><thead><tr><th>Batch / configuration</th><th>Seeds</th><th>Method</th><th>Status</th><th>Progress</th></tr></thead><tbody>{workItems.slice(currentWorkPage * pageSize, (currentWorkPage + 1) * pageSize).map((item) => <tr key={item.key}>
          <td>{batchNames.get(item.source.batchId) ?? item.source.batchId}<small>Configuration {item.configurationNumber} · {item.foldCount} folds</small></td>
          <td>Training {item.source.trainingSeed}<small>Split {item.source.splitSeed}</small></td>
          <td>{predictorMethodLabel(item.method)}{item.epochBudget ? <small>P{item.epochBudget.percentile} → {item.epochBudget.epochs} epochs</small> : item.refitPercentile != null ? <small>Refit budget P{item.refitPercentile}</small> : null}</td>
          <td>{item.status === 'completed' ? 'Ready' : item.status.replace(/^./, (letter) => letter.toUpperCase())}</td>
          <td>{item.execution?.progress?.epoch !== undefined ? <>Epoch {item.execution.progress.epoch} / {item.execution.progress.maxEpochs ?? item.epochBudget?.epochs ?? '—'}{typeof item.execution.progress.trainingLoss === 'number' ? <small>Loss {item.execution.progress.trainingLoss.toFixed(4)}</small> : null}</> : item.method === 'ensemble' && item.status === 'completed' ? `${item.foldCount} checkpoints` : '—'}{item.error ? <small role="status">{item.error.message}</small> : null}{item.execution?.error ? <small>{item.execution.error}</small> : null}{item.status === 'queued' && item.execution?.waitingReason ? <small>{item.execution.waitingReason}</small> : null}{item.execution?.progressWarning ? <small>{item.execution.progressWarning}</small> : null}</td>
        </tr>)}</tbody></table></div>
        {!workItems.length ? <p className="muted">No jobs match this status.</p> : null}
        <Pagination count={workItems.length} page={currentWorkPage} setPage={setWorkPage} label="Predictor jobs" />
      </details> : null}
      {execution.logPath ? <details><summary>Predictor worker details</summary><code className="record-path">{execution.logPath}</code>{execution.updatedAt ? <p className="muted">Updated {execution.updatedAt}</p> : null}</details> : null}
    </div> : execution ? null : !creation ? null : skipped ? <p className="muted">Predictor creation was skipped in every batch. This experiment contains cross-validation results only.</p> : record.submission?.status !== 'submitted' && stage === 'running' ? <p className="callout">Predictor creation waits until experiment submission is complete. Resolve the submission notice above to continue.</p> : !hasPredictorPlan ? <p className="muted">Predictors from this historical experiment are retained here. <a href={`#post-development?${new URLSearchParams({ tab: 'refits', experiment: record.id })}`}>Open historical refit jobs</a></p> : view === 'progress' ? <p className="muted">Predictors are created after each batch’s folds finish. Ready predictors are listed under Predictors.</p> : null}
    {showLibrary ? <>
    <ErrorNotice error={query.error} />
    <div className="experiment-predictor-tools"><strong>{ready.length.toLocaleString()} ready to evaluate</strong>{ready.length ? <a className="btn btn-primary btn-small" href={evaluationLink}>Evaluate predictors →</a> : null}</div>
    {items.length ? <>
      <div className="experiment-predictor-tools"><label className="label">Predictor method<select className="field" value={method} onChange={(event) => { setMethod(event.target.value); setPage(0); }}><option value="all">All methods</option><option value="ensemble">Ensemble</option><option value="refit">Refit</option></select></label><label className="label">Find a predictor<input className="field" type="search" placeholder="Name, configuration or seed" value={search} onChange={(event) => { setSearch(event.target.value); setPage(0); }} /></label></div>
      <div className="experiment-predictor-table"><table><thead><tr><th>Predictor</th><th>Source</th><th>Seeds</th><th>Method</th><th>Evaluation</th></tr></thead><tbody>{filtered.slice(currentPage * pageSize, (currentPage + 1) * pageSize).map((item) => <tr key={item.id}>
        <td><details open={selectedPredictor === item.id || undefined}><summary>{item.manifest.name}</summary><PredictorDetails item={item} /></details></td>
        <td>{batchNames.get(item.manifest.batchId) ?? item.manifest.batchId}<small>{predictorConfigurationLabel({ ...item.manifest, candidateNumber: configurations.get(JSON.stringify([item.manifest.batchId, item.manifest.candidateId])) ?? item.manifest.candidateNumber })}</small></td>
        <td>Training {item.manifest.trainingSeed}<small>Split {item.manifest.splitSeed}</small></td>
        <td>{predictorMethodLabel(item.manifest.method)}<small>{item.manifest.method === 'refit' ? `${item.manifest.epochBudget?.epochs ?? '—'} epochs · P${item.manifest.epochBudget?.percentile ?? '—'}` : `${item.manifest.checkpoints.length} fold checkpoints`}</small></td>
        <td>{item.lifecycleState === 'active' ? <a href={`#evaluate-models?${new URLSearchParams({ experiment: record.id, predictor: item.id })}`}>Evaluate</a> : <Badge>Archived</Badge>}</td>
      </tr>)}</tbody></table></div>
      {!filtered.length ? <p className="muted">No predictors match these filters.</p> : null}
      <Pagination count={filtered.length} page={currentPage} setPage={setPage} label="Predictor library" />
    </> : query.isPending ? <p role="status">Loading predictors…</p> : !query.isError ? <p className="muted">{skipped ? 'To create predictors, use this experiment as a template and change the predictor choices in its batches before submission.' : 'Ready predictors appear here after their fold evidence and checkpoints are verified.'}</p> : null}
    </> : null}
  </Panel>;
}

function PredictorDetails({ item }: { item: FrozenPredictor }) {
  const budget = item.manifest.epochBudget;
  return <div><p className="muted">{item.id}</p><p>{item.manifest.method === 'refit' ? 'One model trained on the complete development cohort.' : item.manifest.aggregation === 'mean_logit' ? 'Mean logits from the frozen fold checkpoints.' : 'Mean probabilities from the frozen fold checkpoints.'}</p>{budget ? <p>Selected fold checkpoint epochs: {budget.foldBestEpochs.map((fold) => fold.bestEpoch).join(', ')}. P{budget.percentile}, rounded up: {budget.epochs} epochs.</p> : null}<details><summary>Exact predictor snapshot</summary><pre className="experiment-snapshot">{JSON.stringify(item.manifest, null, 2)}</pre></details></div>;
}

function Pagination({ count, page, setPage, label }: { count: number; page: number; setPage: (page: number) => void; label: string }) {
  if (count <= pageSize) return null;
  return <nav className="inline-actions" aria-label={`${label} pages`}><button className="btn btn-secondary btn-small" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><span>Page {page + 1} of {Math.ceil(count / pageSize)}</span><button className="btn btn-secondary btn-small" disabled={(page + 1) * pageSize >= count} onClick={() => setPage(page + 1)}>Next</button></nav>;
}

function PredictorActions({ project, record, execution }: { project: string; record: ModelExperiment; execution: ExperimentPredictorExecution }) {
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const [pending, setPending] = useState<{ action: 'resume' | 'cancel'; operationId: string } | null>(null);
  async function run(action: 'resume' | 'cancel') {
    if (inFlight.current) return;
    const input = pending ?? { action, operationId: crypto.randomUUID() };
    inFlight.current = true; setBusy(true); setError(null); setPending(input);
    try {
      const result = await experiments.predictorAction(project, record.id, input.action, input.operationId);
      client.setQueryData<ModelExperiment>(['model-experiment', project, record.id], (current) => current ? { ...current, predictorExecution: result } : current);
      setPending(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('Could not update predictor jobs.'));
      if (reason instanceof ApiError) setPending(null);
    } finally { inFlight.current = false; setBusy(false); }
    void client.invalidateQueries({ queryKey: ['model-experiment', project, record.id] });
    void client.invalidateQueries({ queryKey: ['model-experiments', project] });
    void client.invalidateQueries({ queryKey: ['predictors', project] });
  }
  return <div><ErrorNotice error={error} />{pending && !busy ? <p className="callout">The predictor action response was lost. Retry uses the same request identity.</p> : null}<div className="inline-actions">
    {pending ? <button className="btn btn-secondary" disabled={busy} onClick={() => void run(pending.action)}>{busy ? 'Updating predictor jobs…' : `Retry ${pending.action} request`}</button> : <>
      {execution.retryable ? <button className="btn btn-secondary" disabled={busy} onClick={() => void run('resume')}>Resume predictor creation</button> : null}
      {execution.cancellable ? <button className="btn btn-secondary" disabled={busy} onClick={() => void run('cancel')}>Cancel remaining predictors</button> : null}
    </>}
  </div></div>;
}
