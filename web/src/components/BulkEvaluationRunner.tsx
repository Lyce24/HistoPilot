import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { bulkEvaluations, type BulkEvaluationSelection, type EvaluationBatch } from '../api/bulkEvaluations';
import { predictorMethodLabel, type FrozenPredictor } from '../api/predictors';
import type { EvaluationCohort } from '../api/evaluation';
import { versionLabelText } from '../lib/versionLabels';
import { cleanupLink } from '../lib/hashRoute';
import { useBatchReview } from './useBatchReview';
import { Badge, ErrorNotice } from './ui';
import './RunWorkspace.css';

export default function BulkEvaluationRunner({ project, predictors, cohorts, linkedCohort = '', onOpenEvaluation }: { project: string; predictors: FrozenPredictor[]; cohorts: EvaluationCohort[]; linkedCohort?: string; onOpenEvaluation: (id: string) => void }) {
  const client = useQueryClient();
  const [cohortId, setCohortId] = useState(linkedCohort);
  const [scope, setScope] = useState<'all' | 'selected'>('all');
  const [selected, setSelected] = useState<string[]>([]);
  const [search, setSearch] = useState('');
  const [method, setMethod] = useState('all');
  const [name, setName] = useState('');
  const [batchId, setBatchId] = useState('');
  const active = predictors.filter((item) => item.lifecycleState === 'active');
  const visible = active.filter((item) => (method === 'all' || (item.manifest.method ?? 'ensemble') === method) && `${item.manifest.name} ${item.manifest.trainingSeed} ${item.manifest.splitSeed} ${item.manifest.experiment?.name ?? ''}`.toLowerCase().includes(search.toLowerCase()));
  const cohort = cohorts.find((item) => item.id === cohortId);
  const ready = cohort?.current && !cohort.findings?.some((finding) => finding.severity === 'error');
  const batches = useQuery({ queryKey: ['evaluation-batches', project], queryFn: () => bulkEvaluations.list(project), refetchInterval: 5000 });
  const action = useBatchReview(
    (selection: BulkEvaluationSelection) => bulkEvaluations.preview(project, selection),
    (selection, preview, operation) => bulkEvaluations.run(project, selection, { previewHash: preview.previewHash, reviewedPredictorIds: preview.reviewedPredictorIds }, operation),
    (preview) => preview.canRun,
    (result) => result.items.every((item) => ['skipped', 'cancelled'].includes(item.status) || Boolean(item.execution && item.execution.status !== 'not_started')),
    async (result) => { setBatchId(result.id); await Promise.all(['evaluation-batches', 'model-evaluations', 'cleanup'].map((key) => client.invalidateQueries({ queryKey: [key, project] }))); },
  );
  function reset() { action.reset(); }
  return <div className="run-workspace">
    <ErrorNotice error={action.error ?? batches.error} />
    <fieldset className="chain-fields" disabled={action.locked} onChange={reset}>
      <legend className="sr-only">Run predictors on a test cohort</legend>
      <label className="label">Test cohort for all predictors<select className="field" value={cohortId} onChange={(event) => setCohortId(event.target.value)}><option value="">Choose a test cohort</option>{cohorts.map((item) => <option key={item.id} value={item.id} disabled={!item.current || item.findings?.some((finding) => finding.severity === 'error')}>{versionLabelText(item, 'Test cohort')} · {item.manifest.summary.includedSlides} slides</option>)}</select></label>
      <label className="label">Evaluation batch name (optional)<input className="field" value={name} maxLength={60} onChange={(event) => setName(event.target.value)} placeholder="For example: External validation" /></label>
      <div className="run-methods chain-wide" role="group" aria-label="Predictors to evaluate"><label className={scope === 'all' ? 'selected' : ''}><input type="radio" name="evaluation-scope" value="all" checked={scope === 'all'} onChange={() => setScope('all')} />All predictors ({active.length})</label><label className={scope === 'selected' ? 'selected' : ''}><input type="radio" name="evaluation-scope" value="selected" checked={scope === 'selected'} onChange={() => setScope('selected')} />Selected predictors ({selected.length})</label></div>
    </fieldset>
    <p className="muted">Every selected predictor runs on the same test cohort, with separate results for each seed and method. The review lists incompatible predictors and excludes them explicitly. Each batch supports up to 256 predictors.</p>
    {scope === 'selected' ? <><div className="run-toolbar"><label className="label run-search">Find predictors<input className="field" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Experiment, name or seed" /></label><label className="label">Predictor method filter<select className="field" value={method} onChange={(event) => setMethod(event.target.value)}><option value="all">All methods</option><option value="ensemble">Ensemble</option><option value="refit">Refit</option></select></label></div><div className="run-selection-bar"><strong>{selected.length} selected</strong><button className="text-button" disabled={action.locked} onClick={() => { reset(); setSelected([...new Set([...selected,...visible.map((item) => item.id)])]); }}>Select all shown</button><button className="text-button" disabled={action.locked} onClick={() => { reset(); setSelected([]); }}>Clear selection</button></div><div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-check"><span className="sr-only">Select</span></th><th>Predictor</th><th>Experiment</th><th>Method</th><th className="run-number">Training seed</th><th className="run-number">Split seed</th></tr></thead><tbody>{visible.map((item) => <tr key={item.id}><td><input type="checkbox" aria-label={`Evaluate ${item.manifest.name}`} disabled={action.locked} checked={selected.includes(item.id)} onChange={(event) => { reset(); setSelected(event.target.checked ? [...selected,item.id] : selected.filter((id) => id !== item.id)); }} /></td><td>{item.manifest.name}</td><td>{item.manifest.experiment?.name ?? item.manifest.experimentId}</td><td>{predictorMethodLabel(item.manifest.method)}</td><td>{item.manifest.trainingSeed}</td><td>{item.manifest.splitSeed}</td></tr>)}</tbody></table></div></> : null}
    {!active.length ? <p className="callout">No active predictors are available. <a href="#post-development">Build predictors</a> or restore archived records first.</p> : null}
    <button className="btn btn-primary" disabled={action.locked || !ready || !active.length || (scope === 'selected' && !selected.length)} onClick={() => void action.preview({ cohortId, scope, ...(scope === 'selected' ? { predictorIds: selected } : {}), ...(name.trim() ? { namePrefix: name.trim() } : {}) })}>{action.busy ? 'Checking compatibility…' : scope === 'all' ? 'Review all predictors' : 'Review selected predictors'}</button>
    {action.review ? <div className="run-bulk-review"><h3>Review evaluation batch</h3><p><strong>{action.review.preview.eligibleCount}</strong> compatible predictors will run · <strong>{action.review.preview.blockedCount}</strong> excluded. The reviewed predictor list is fixed for this batch.</p><div className="run-table-scroll"><table className="run-table"><thead><tr><th>Predictor</th><th>Method</th><th>Compatibility</th><th>Details</th></tr></thead><tbody>{action.review.preview.items.map((item) => <tr key={item.predictorId}><td>{item.predictorName}</td><td>{predictorMethodLabel(item.method)}</td><td><Badge tone={item.eligible ? 'success' : 'warning'}>{item.eligible ? 'Will run' : 'Excluded'}</Badge></td><td>{item.findings.map((finding) => finding.message).join(' ') || 'Inputs verified'}</td></tr>)}</tbody></table></div>
      {action.review.preview.canRun ? <><label className="development-check"><input type="checkbox" checked={action.acknowledged} disabled={action.busy || action.submitted} onChange={(event) => action.setAcknowledged(event.target.checked)} />I reviewed the test cohort, predictor list and exclusions.</label><button className="btn btn-primary" disabled={action.busy || (!action.acknowledged && !action.submitted)} onClick={() => void action.apply()}>{action.busy ? 'Submitting evaluation jobs…' : action.submitted ? 'Retry unfinished submissions' : action.review.selection.scope === 'all' ? 'Run all compatible predictors' : 'Run selected compatible predictors'}</button></> : null}
    </div> : null}
    {action.result ? <div className="callout" role="status"><p>Evaluation batch saved. Each predictor has its own job and results below.</p>{action.submitted ? <button className="text-button" disabled={action.busy} onClick={action.reset}>Start a new review; keep this evaluation batch</button> : null}</div> : null}
    {(batches.data?.items.length ?? 0) > 0 ? <div className="run-bulk-review"><h3>Evaluation batches</h3><label className="label">Evaluation batch<select className="field" value={batchId} onChange={(event) => setBatchId(event.target.value)}><option value="">Choose a submitted batch</option>{batches.data?.items.map((item) => <option key={item.id} value={item.id}>{item.name ?? item.id} · {item.status} · {item.items.length} predictors</option>)}</select></label>{batchId ? <EvaluationBatchStatus key={batchId} project={project} id={batchId} onOpen={onOpenEvaluation} /> : null}</div> : null}
  </div>;
}

function EvaluationBatchStatus({ project, id, onOpen }: { project: string; id: string; onOpen: (id: string) => void }) {
  const client = useQueryClient();
  const record = useQuery({ queryKey: ['evaluation-batch', project, id], queryFn: () => bulkEvaluations.get(project, id), refetchInterval: 3000 });
  const [operation, setOperation] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  async function cancel() {
    if (busy) return;
    const op = operation ?? crypto.randomUUID(); setOperation(op); setBusy(true); setError(null);
    try { const result = await bulkEvaluations.cancel(project, id, op); client.setQueryData(['evaluation-batch', project, id], result); setOperation(null); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Cancel request failed.')); }
    finally { setBusy(false); }
  }
  const batch: EvaluationBatch | undefined = record.data;
  return <><ErrorNotice error={error ?? record.error} />{batch ? <><div className="run-selection-bar"><strong>{batch.status}</strong>{['queued','running'].includes(batch.status) ? <button className="btn btn-secondary" disabled={busy || batch.cancelRequested} onClick={() => void cancel()}>{busy || batch.cancelRequested ? 'Cancelling…' : operation ? 'Retry cancellation' : 'Cancel evaluation batch'}</button> : null}<a href={cleanupLink(batch.id)}>Manage batch record</a></div><div className="run-table-scroll"><table className="run-table"><thead><tr><th>Predictor</th><th>Method</th><th>Status</th><th>Details</th><th>Results</th></tr></thead><tbody>{batch.items.map((item) => <tr key={item.predictorId}><td>{item.predictorName ?? item.predictorId}</td><td>{predictorMethodLabel(item.method)}</td><td>{item.status}</td><td>{item.error ?? item.findings?.map((finding) => finding.message).join(' ')}</td><td>{item.evaluationId ? <button className="text-button" onClick={() => onOpen(item.evaluationId!)}>Open evaluation</button> : '—'}</td></tr>)}</tbody></table></div></> : <p>Loading evaluation batch…</p>}</>;
}
