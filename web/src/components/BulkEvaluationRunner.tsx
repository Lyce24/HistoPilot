import { Fragment, useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { bulkEvaluations, type BulkEvaluationSelection, type EvaluationBatch } from '../api/bulkEvaluations';
import { predictorMethodLabel, type FrozenPredictor } from '../api/predictors';
import type { ModelExperimentSummary } from '../api/experiments';
import type { EvaluationCohort } from '../api/evaluation';
import { versionLabelText } from '../lib/versionLabels';
import { groupPredictors, predictorMatches, predictorConfigurationLabel, experimentPredictorLink } from '../lib/predictorGroups';
import { evaluationExperiments, experimentPredictors, type EvaluationMethod } from '../lib/evaluationSelection';
import { RecordManageButton } from './RecordManagement';
import { shortRecordId } from '../lib/recordLabels';
import { useBatchReview } from './useBatchReview';
import EvaluationExperimentPicker from './EvaluationExperimentPicker';
import EvaluationInputSettings, { initialEvaluationInputs, evaluationExecutionSelection, EvaluationCoverageSummary, type EvaluationExecutionInputs } from './EvaluationInputSettings';
import { reportEditorValidity } from './NumericField';
import { Badge, ErrorNotice } from './ui';
import { StageCreateButton, StageBackButton, StageContinueButton, StagePage, StageSteps } from './StageWorkflow';
import './RunWorkspace.css';

export default function BulkEvaluationRunner({ project, predictors, experiments = [], experimentsLoading, cohorts, linkedCohort = '', linkedExperiment = '', experimentIds, onExperimentsChange, onLockChange, onOpenEvaluation }: {
  project: string; predictors: FrozenPredictor[]; experiments?: ModelExperimentSummary[]; experimentsLoading?: boolean;
  cohorts: EvaluationCohort[]; linkedCohort?: string; linkedExperiment?: string; experimentIds?: string[];
  onExperimentsChange?: (ids: string[]) => void; onLockChange?: (locked: boolean) => void; onOpenEvaluation: (id: string) => void;
}) {
  const client = useQueryClient();
  const [inputStep, setInputStep] = useState<'experiments' | 'inputs' | 'results'>('experiments');
  const [cohortId, setCohortId] = useState(linkedCohort);
  const [localExperimentIds, setExperimentIds] = useState<string[]>(() => linkedExperiment ? [linkedExperiment] : []);
  const sourceIds = experimentIds ?? localExperimentIds;
  const [individual, setIndividual] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [search, setSearch] = useState('');
  const [method, setMethod] = useState<EvaluationMethod>('both');
  const [name, setName] = useState('');
  const [batchId, setBatchId] = useState('');
  const [executionInputs, setExecutionInputs] = useState<EvaluationExecutionInputs | null>(null);
  const editor = useRef<HTMLFieldSetElement>(null);
  const options = evaluationExperiments(experiments, predictors, sourceIds);
  const candidates = experimentPredictors(predictors, sourceIds, method);
  const eligibleIds = new Set(candidates.map((item) => item.id));
  const draftIds = individual ? selected.filter((id) => eligibleIds.has(id)) : [...eligibleIds];
  const visible = candidates.filter((item) => predictorMatches(item, '', 'all', search));
  const groups = groupPredictors(visible);
  const cohort = cohorts.find((item) => item.id === cohortId);
  const ready = Boolean(cohort);
  const inputs = executionInputs ?? initialEvaluationInputs(cohort);
  const action = useBatchReview(
    (selection: BulkEvaluationSelection) => bulkEvaluations.preview(project, selection),
    (selection, preview, operation) => bulkEvaluations.run(project, selection, { previewHash: preview.previewHash, reviewedPredictorIds: preview.reviewedPredictorIds }, operation),
    (preview) => preview.canRun,
    (result) => result.items.every((item) => ['skipped', 'cancelled'].includes(item.status) || Boolean(item.execution && item.execution.status !== 'not_started')),
    async (result) => { setBatchId(result.id); await Promise.all(['evaluation-batches', 'model-evaluations', 'cleanup'].map((key) => client.invalidateQueries({ queryKey: [key, project] }))); },
  );
  const fixed = action.locked || Boolean(action.review);
  const page = action.review ? 'review' : action.result ? 'results' : inputStep;
  useEffect(() => { onLockChange?.(fixed); }, [fixed, onLockChange]);
  const selectedIds = action.review?.selection.predictorIds ?? draftIds;
  const tooMany = selectedIds.length > 256;
  const selectedSet = new Set(selectedIds);
  const arrivals = action.review ? draftIds.filter((id) => !selectedSet.has(id)).length : 0;
  const totals = candidates.reduce((value, item) => { value[item.manifest.method ?? 'ensemble']++; return value; }, { ensemble: 0, refit: 0 });
  function changeExperiments(ids: string[]) { action.reset(); setExperimentIds(ids); onExperimentsChange?.(ids); }
  const batchName = (source: FrozenPredictor['manifest']) => experiments.find((item) => item.id === source.experimentId)?.batches.find((item) => item.id === source.batchId)?.name ?? `Batch ${shortRecordId(source.batchId)}`;
  return <div className="run-workspace">
    <ErrorNotice error={action.error} />
    <StageSteps label="Experiment evaluation steps" current={page} disabled={fixed} steps={[
      { id: 'experiments', title: 'Experiments', description: 'Choose development results', complete: sourceIds.length > 0 },
      { id: 'inputs', title: 'Evaluation inputs', description: 'Methods, test cohort and features', disabled: !sourceIds.length, complete: Boolean(action.review || action.result) },
      { id: 'review', title: 'Review and run', description: 'Check compatibility and exclusions', disabled: !action.review },
      { id: 'results', title: 'Batch results', description: 'Monitor evaluations', disabled: !batchId },
    ]} onChange={(next) => { if (next === 'experiments' || next === 'inputs') { action.reset(); setInputStep(next); } else if (next === 'results' && batchId) setInputStep('results'); }} />
    <StagePage pageKey={page}>
    {page === 'experiments' ? <>
    <EvaluationExperimentPicker options={options} selected={sourceIds} onChange={changeExperiments} disabled={fixed} loading={experimentsLoading} />
    <div className="stage-actions"><p className="muted">{draftIds.length} {draftIds.length === 1 ? 'predictor' : 'predictors'} to evaluate from {sourceIds.length} selected {sourceIds.length === 1 ? 'experiment' : 'experiments'}</p><StageContinueButton disabled={!sourceIds.length || fixed} onClick={() => setInputStep('inputs')}>Continue to evaluation inputs</StageContinueButton></div>
    </> : null}
    {page === 'inputs' ? <>
    <section className="evaluation-step" aria-labelledby="evaluation-method-title"><h3 id="evaluation-method-title">2. Choose methods and test cohort</h3>
      <fieldset ref={editor} className="chain-fields" disabled={fixed} onChange={() => action.reset()}>
        <legend className="sr-only">Predictor methods and evaluation inputs</legend>
        <div className="run-methods chain-wide" role="group" aria-label="Predictor methods to evaluate">{(['both', 'ensemble', 'refit'] as const).map((value) => <label key={value} className={method === value ? 'selected' : ''}><input type="radio" name="evaluation-method" value={value} checked={method === value} onChange={() => setMethod(value)} />{value === 'both' ? 'Both available methods' : value === 'ensemble' ? 'Ensemble' : 'Refit'}</label>)}</div>
        <label className="label">Test cohort for selected experiments<select className="field" value={cohortId} onChange={(event) => { setCohortId(event.target.value); setExecutionInputs(null); }}><option value="">Choose a test cohort</option>{cohorts.map((item) => <option key={item.id} value={item.id}>{versionLabelText(item, 'Test cohort')} · {item.manifest.summary.includedSlides} slides</option>)}</select></label>
        <label className="label">Evaluation batch name (optional)<input className="field" value={name} maxLength={60} onChange={(event) => setName(event.target.value)} placeholder="For example: Ensemble and refit comparison" /></label>
        <div className="chain-wide"><EvaluationInputSettings project={project} cohort={cohort} value={inputs} onChange={(value) => { setExecutionInputs(value); action.reset(); }} disabled={fixed} /></div>
      </fieldset>
      <p className="muted">{totals.ensemble} ensemble and {totals.refit} refit predictors are ready for this selection. Both includes whichever methods are available; it does not build missing predictors. Compatibility review excludes unsupported inputs. Each batch supports up to 256 predictors.</p>
      {sourceIds.length && !candidates.length ? <p className="callout">The selected experiments have no ready predictors for this method. Follow their progress in <a href="#experiments">Experiments</a>. Batches using Skip produce no predictors; copy an experiment as a template to choose another policy.</p> : !sourceIds.length ? <p className="callout">Select one or more experiments above to choose predictors for evaluation.</p> : null}
      <details className="evaluation-advanced"><summary>Advanced: choose individual predictors</summary>
        <label className="development-check"><input type="checkbox" checked={individual} disabled={fixed} onChange={(event) => { action.reset(); setIndividual(event.target.checked); if (event.target.checked) setSelected([...eligibleIds]); }} />Choose a subset of the selected experiments’ ready predictors</label>
        {individual ? <><div className="run-toolbar"><label className="label run-search">Find individual predictors<input className="field" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Batch, configuration, name or seed" /></label><button className="text-button" disabled={fixed} onClick={() => { action.reset(); setSelected([...new Set([...selected, ...visible.map((item) => item.id)])]); }}>Select all shown predictors</button><button className="text-button" disabled={fixed} onClick={() => { action.reset(); setSelected([]); }}>Clear predictors</button></div>
          {draftIds.some((id) => !visible.some((item) => item.id === id)) ? <p className="muted">Some selected predictors are hidden by search and remain included.</p> : null}</> : null}
        {visible.length ? <div className="run-table-scroll evaluation-predictor-table"><table className="run-table"><thead><tr>{individual ? <th className="run-check"><span className="sr-only">Select</span></th> : null}<th>Batch / configuration</th><th>Method</th><th className="run-number">Training seed</th><th className="run-number">Split seed</th><th>Weights</th></tr></thead><tbody>{groups.map((group) => <Fragment key={group.id}><tr className="run-group"><th colSpan={individual ? 6 : 5}><a href={experimentPredictorLink(group.id)}>{group.name}</a> · {group.items.length} predictors</th></tr>{group.items.map((item) => <tr key={item.id}>{individual ? <td><input type="checkbox" aria-label={`Evaluate predictor ${item.manifest.name} (${item.id})`} disabled={fixed} checked={selectedSet.has(item.id)} onChange={(event) => { action.reset(); setSelected(event.target.checked ? [...selected, item.id] : selected.filter((id) => id !== item.id)); }} /></td> : null}<td><strong title={item.manifest.candidateId}>{predictorConfigurationLabel(item.manifest)}</strong><small>{batchName(item.manifest)} · {item.manifest.name}</small></td><td>{predictorMethodLabel(item.manifest.method)}</td><td>{item.manifest.trainingSeed}</td><td>{item.manifest.splitSeed}</td><td>{item.manifest.checkpoints.length} {item.manifest.method === 'refit' ? 'refit model' : 'fold models'}</td></tr>)}</Fragment>)}</tbody></table></div> : null}
      </details>
    </section>
    <div className="run-selection-bar"><strong>{selectedIds.length} {selectedIds.length === 1 ? 'predictor' : 'predictors'} {action.review ? 'fixed for review' : 'to evaluate'} from {sourceIds.length} selected {sourceIds.length === 1 ? 'experiment' : 'experiments'}</strong></div>
    {tooMany ? <p className="callout" role="status">This selection contains {selectedIds.length} predictors. Choose fewer experiments or methods, or use individual selection to stay within 256 predictors per batch.</p> : null}
    <div className="stage-actions"><StageBackButton disabled={fixed} onClick={() => setInputStep('experiments')}>Back</StageBackButton>
    {!action.review ? <StageContinueButton disabled={action.locked || !ready || !selectedIds.length || tooMany} onClick={() => { if (reportEditorValidity(editor.current)) void action.preview({ cohortId, scope: 'selected', predictorIds: selectedIds, ...evaluationExecutionSelection(inputs), ...(name.trim() ? { namePrefix: name.trim() } : {}) }); }}>{action.busy ? 'Checking compatibility…' : 'Review experiment evaluation'}</StageContinueButton> : null}</div>
    </> : null}
    {action.review ? <div className="run-bulk-review"><h3>3. Review experiment evaluation</h3><p><strong>{action.review.preview.eligibleCount}</strong> compatible predictors will run · <strong>{action.review.preview.blockedCount}</strong> excluded. Prediction tasks, class encoding, extracted features, pack coverage and development overlap are checked for every predictor. The reviewed predictor list is fixed for this batch.</p>
      {arrivals ? <p className="callout" role="status">{arrivals} additional ready {arrivals === 1 ? 'predictor is' : 'predictors are'} excluded from this review. Change the selection and review again to include new arrivals.</p> : null}
      <div className="run-table-scroll"><table className="run-table"><thead><tr><th>Experiment / predictor</th><th>Method</th><th>Compatibility</th><th>Details</th></tr></thead><tbody>{action.review.preview.items.map((item) => { const source = predictors.find((candidate) => candidate.id === item.predictorId)?.manifest; return <tr key={item.predictorId}><td>{source?.experiment?.name ?? shortRecordId(source?.experimentId ?? '')}<small>{item.predictorName}</small></td><td>{predictorMethodLabel(item.method)}</td><td><Badge tone={item.eligible ? 'success' : 'warning'}>{item.eligible ? 'Will run' : 'Excluded'}</Badge></td><td>{item.findings.map((finding) => finding.message).join(' ') || 'Inputs verified'}<EvaluationCoverageSummary manifest={item.evaluationManifest} /></td></tr>; })}</tbody></table></div>
      {action.review.preview.canRun ? <label className="development-check"><input type="checkbox" checked={action.acknowledged} disabled={action.busy || action.submitted} onChange={(event) => action.setAcknowledged(event.target.checked)} />I reviewed the test cohort, predictor list and exclusions.</label> : null}
      <div className="stage-actions">
        {!action.submitted ? <StageBackButton disabled={action.busy} onClick={action.reset}>Back to evaluation inputs</StageBackButton> : null}
        {action.review.preview.canRun ? <button className="btn btn-primary" disabled={action.busy || (!action.acknowledged && !action.submitted)} onClick={() => void action.apply()}>{action.busy ? 'Submitting evaluation jobs…' : action.submitted ? 'Retry unfinished submissions' : 'Run reviewed predictors'}</button> : null}
      </div>
    </div> : null}
    {action.result && action.submitted ? <div className="callout" role="status"><p>The evaluation batch is saved. Retry the unfinished submissions, or keep this batch and inspect the jobs already accepted.</p><button type="button" className="btn btn-secondary" disabled={action.busy} onClick={() => { action.reset(); setInputStep('results'); }}>Keep this batch and view submitted results</button></div> : null}
    {action.result && page === 'results' ? <div className="callout" role="status"><p>Evaluation batch saved. Each predictor has its own job and results below.</p>{action.submitted ? <button className="text-button" disabled={action.busy} onClick={action.reset}>Start a new review; keep this evaluation batch</button> : null}</div> : null}
    {page === 'results' && batchId ? <><EvaluationBatchStatus key={batchId} project={project} id={batchId} onOpen={onOpenEvaluation} /><div className="stage-actions"><StageCreateButton onClick={() => { action.reset(); setInputStep('experiments'); }}>Create evaluation batch</StageCreateButton></div></> : null}
    </StagePage>
  </div>;
}

export function EvaluationBatchStatus({ project, id, onOpen }: { project: string; id: string; onOpen: (id: string) => void }) {
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
  return <><ErrorNotice error={error ?? record.error} />{batch ? <><div className="run-selection-bar"><strong>{batch.status}</strong>{['queued','running'].includes(batch.status) ? <button className="btn btn-secondary" disabled={busy || batch.cancelRequested} onClick={() => void cancel()}>{busy || batch.cancelRequested ? 'Cancelling…' : operation ? 'Retry cancellation' : 'Cancel evaluation batch'}</button> : null}<RecordManageButton recordKey={`configuration:${batch.id}`} name={batch.name ?? "Evaluation batch"} /></div><div className="run-table-scroll"><table className="run-table"><thead><tr><th>Predictor</th><th>Method</th><th>Status</th><th>Details</th><th>Results</th></tr></thead><tbody>{batch.items.map((item) => <tr key={item.predictorId}><td>{item.predictorName ?? item.predictorId}</td><td>{predictorMethodLabel(item.method)}</td><td>{item.status}</td><td>{item.error ?? item.findings?.map((finding) => finding.message).join(' ')}</td><td>{item.evaluationId ? <button className="text-button" onClick={() => onOpen(item.evaluationId!)}>Open evaluation</button> : '—'}</td></tr>)}</tbody></table></div></> : <p>Loading evaluation batch…</p>}</>;
}
