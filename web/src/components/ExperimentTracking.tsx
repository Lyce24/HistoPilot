import PatientAnalysisResults from './PatientAnalysisResults';
import { useId, useState } from 'react';
import { queryOptions, useQuery } from '@tanstack/react-query';
import { development } from '../api/development';
import type { FrozenBatch, PlannedRun, TrainingExecution, TrainingHistory, TrainingMetricDetails, TrainingMetrics, TrainingRecipe, TrainingRun } from '../api/development';
import { finiteNumber } from '../lib/evidenceCharts';
import { downloadJSON } from '../lib/download';
import CurveChart from './CurveChart';
import { Badge, ErrorNotice } from './ui';
import './ExperimentTracking.css';

export { ResourceCards } from './RunResourceUsage';

export const gib = (value?: number | null) => finiteNumber(value) ? `${value.toFixed(2)} GiB` : 'Unavailable';
export const metricValue = (value?: number | null) => finiteNumber(value) ? value.toFixed(4) : '—';
export function metricsText(metrics?: TrainingMetrics | null) {
  if (!metrics) return '—';
  if (metrics.available === false) return metrics.reason ?? 'Unavailable';
  const names = (['accuracy', 'auroc', 'auprc', 'loss', 'balancedAccuracy', 'macroF1'] as const).filter((name) => metrics[name] !== undefined);
  return names.length ? names.map((name) => `${name}: ${finiteNumber(metrics[name]) ? metrics[name].toFixed(4) : 'Unavailable'}`).join(' · ') : '—';
}

export function MetricEvidence({ details }: { details?: TrainingMetricDetails }) {
  if (!details) return null;
  return <details><summary>{details.unit} scoring details</summary><p>Class order: {details.classOrder.join(', ')}{details.positiveClass ? ` · Positive class: ${details.positiveClass}` : ''}</p><p>Slide: {metricsText(details.slide)}</p><p>Patient ({details.patientAggregation.replaceAll('_', ' ')}): {metricsText(details.patient)}</p><PatientAnalysisResults value={details.patientAnalysis} />{details.selected.missingClasses?.length ? <p>Missing classes: {details.selected.missingClasses.join(', ')}. AUROC may be unavailable.</p> : null}</details>;
}

function RunDiagnostics({ run, peakRamGb }: { run?: TrainingRun; peakRamGb?: number }) {
  const progress = run?.progress;
  const available = [progress?.learningRate, progress?.globalStep, progress?.cudaPeakAllocatedBytes, progress?.cudaPeakReservedBytes, peakRamGb].some(finiteNumber);
  return <><h4>Training diagnostics</h4>{available ? <dl className="experiment-run-facts">
    <div><dt>Learning rate</dt><dd>{finiteNumber(progress?.learningRate) ? progress.learningRate.toExponential(3) : 'Unavailable'}</dd></div>
    <div><dt>Global step</dt><dd>{finiteNumber(progress?.globalStep) ? progress.globalStep.toLocaleString() : 'Unavailable'}</dd></div>
    <div><dt>CUDA allocated peak</dt><dd>{gib(finiteNumber(progress?.cudaPeakAllocatedBytes) ? progress.cudaPeakAllocatedBytes / 2 ** 30 : undefined)}</dd></div>
    <div><dt>CUDA reserved peak</dt><dd>{gib(finiteNumber(progress?.cudaPeakReservedBytes) ? progress.cudaPeakReservedBytes / 2 ** 30 : undefined)}</dd></div>
    <div><dt>Process-tree RAM peak</dt><dd>{gib(peakRamGb)}</dd></div>
  </dl> : <p className="muted">No training diagnostics have been recorded for this run.</p>}
    {finiteNumber(peakRamGb) ? <p className="muted">Process-tree RAM is sampled. Shared pages may be counted more than once.</p> : null}
  </>;
}

export function runLabel(batch: FrozenBatch, run: PlannedRun | TrainingRun) {
  const configuration = batch.manifest.configurations.find((item) => item.id === run.candidateId);
  const split = batch.manifest.splitPlans.find((item) => item.id === run.splitPlanId);
  return `Config ${configuration?.number ?? '?'} · ${finiteNumber(split?.fold) ? `Fold ${split.fold + 1}` : split?.planId ?? 'Unknown split'} · Train seed ${run.trainingSeed}${split?.seed !== undefined ? ` · Split seed ${split.seed}` : ''}`;
}

export function EpochProgress({ run }: { run?: TrainingRun }) {
  const progress = run?.progress;
  const finalEpoch = run?.status === 'completed' && wholeEpoch(run.result?.epochsCompleted) ? run.result.epochsCompleted : null;
  const epoch = finalEpoch ?? progress?.epoch;
  if (!finiteNumber(epoch) || !progress || !finiteNumber(progress.maxEpochs) || progress.maxEpochs <= 0) return <span className="muted">{finalEpoch === null ? 'No epoch reported' : `${finalEpoch} completed epochs`}</span>;
  return <div className="experiment-epoch"><span>Epoch {epoch} / {progress.maxEpochs}</span><progress aria-label="Completed epochs" value={Math.min(epoch, progress.maxEpochs)} max={progress.maxEpochs} />{run?.status === 'completed' && epoch < progress.maxEpochs ? <small>Stopped early</small> : null}</div>;
}

export function LossHistory({ history, validationMetric = 'loss', finished = false, illustrative = false, latestCheckpoint = false }: { history?: TrainingHistory; validationMetric?: 'loss' | 'accuracy' | 'auroc'; finished?: boolean; illustrative?: boolean; latestCheckpoint?: boolean }) {
  const rows = history?.rows ?? [];
  if (history?.warning) return <p className="callout callout-warning" role="status">{history.warning}</p>;
  if (!rows.length) return <p className="callout">{finished ? 'No epoch history was recorded for this run.' : 'Loss history appears after the first completed epoch.'}</p>;
  const unit = rows.find((row) => row.checkpointUnit)?.checkpointUnit;
  const description = `${illustrative ? 'Synthetic example values for each epoch.' : 'Recorded at each completed epoch.'} ${latestCheckpoint ? `Validation${unit ? ` uses ${unit} scoring and` : ''} monitors training. Evaluation uses the latest completed epoch.` : `Validation${unit ? ` uses ${unit} scoring and` : ''} selects checkpoints; it is not held-out assessment.`}`;
  const xRange = [rows[0].epoch, Math.max(rows[0].epoch + 1, rows[rows.length - 1].epoch)] as const;
  const trainingPoints = rows.map((row) => ({ x: row.epoch, y: row.trainingLoss }));
  const validationPoints = rows.map((row) => ({ x: row.epoch, y: row.validation.loss ?? null }));
  const hasLoss = [...trainingPoints, ...validationPoints].some((point) => finiteNumber(point.y));
  return <>
    {history?.truncated ? <p className="muted">Showing the latest {rows.length} of {history.totalRows} completed epochs. The complete history remains in the run output directory.</p> : null}
    <div className="experiment-history-charts">
      {hasLoss ? <CurveChart title="Loss history" description={description} xLabel="Epoch" integerX yLabel="Loss" xRange={xRange} series={[{ label: 'Training loss', points: trainingPoints }, { label: 'Validation loss', points: validationPoints, dashed: true }]} /> : <p className="callout">No loss measurements are available in the recorded history.</p>}
      {validationMetric !== 'loss' ? <CurveChart title={`Validation ${validationMetric === 'auroc' ? 'AUROC' : 'accuracy'}`} description={`${latestCheckpoint ? 'Training monitor' : 'Checkpoint selection metric'}${unit ? ` · ${unit} scoring` : ''}. Missing values remain gaps.`} xLabel="Epoch" integerX yLabel={validationMetric === 'auroc' ? 'AUROC' : 'Accuracy'} xRange={xRange} yRange={[0, 1]} series={[{ label: `Validation ${validationMetric === 'auroc' ? 'AUROC' : 'accuracy'}`, points: rows.map((row) => ({ x: row.epoch, y: row.validation[validationMetric] ?? null })) }]} /> : null}
    </div>
    {history ? <button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${illustrative ? 'synthetic-' : ''}${history.runId}-epoch-history.json`, illustrative ? { ...history, synthetic: true, executable: false } : history)}>Export {illustrative ? 'synthetic ' : ''}epoch history</button> : null}
  </>;
}

export function runHistoryOptions(project: string, batchId: string, run: TrainingRun) {
  const active = ['queued', 'running'].includes(run.status);
  // A terminal transition must read the final epoch even when the poll stopped
  // just before the worker wrote it. The status key also refreshes resumed runs.
  return queryOptions({ queryKey: ['training-history', project, batchId, run.id, run.status], queryFn: () => development.history(project, batchId, run.id), refetchInterval: active ? 3000 : false });
}

const wholeEpoch = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value > 0;

export function runStoppingMetric(run: TrainingRun, recipe?: TrainingRecipe, history?: TrainingHistory) {
  const validHistory = history?.runId === run.id && !history.warning ? history : undefined;
  if (run.status === 'completed') {
    const epoch = wholeEpoch(run.result?.epochsCompleted) ? run.result.epochsCompleted : wholeEpoch(validHistory?.totalRows) ? validHistory.totalRows : null;
    return { label: 'Stopped epoch', value: epoch === null ? 'Not available' : `Epoch ${epoch}`, detail: epoch === null ? 'The final completed epoch has not been recorded.' : 'Actual completed training epochs' };
  }
  if (['failed', 'cancelled', 'interrupted'].includes(run.status)) return { label: 'Patience', value: 'Paused', detail: `Run ${run.status}; no active early-stopping countdown.` };
  const stopping = validHistory?.stopping;
  if (recipe?.earlyStopping === false || stopping?.status === 'disabled' || stopping?.enabled === false) {
    return { label: 'Patience', value: 'Disabled', detail: stopping?.reason ?? 'Training uses its configured epoch budget.' };
  }
  if (stopping?.status !== 'tracking' || stopping.enabled !== true || !wholeEpoch(stopping.patience)
    || !Number.isSafeInteger(stopping.remaining) || stopping.remaining === null || stopping.remaining < 0 || stopping.remaining > stopping.patience) {
    return { label: 'Patience', value: 'Not available yet', detail: stopping?.reason ?? 'Reported after completed validation epochs.' };
  }
  const floorPending = stopping.remaining === 0 && wholeEpoch(stopping.minEpochs) && (stopping.epoch ?? 0) < stopping.minEpochs;
  return { label: 'Patience', value: `${stopping.remaining} epoch${stopping.remaining === 1 ? '' : 's'} left`,
    detail: `Of ${stopping.patience} without sufficient validation improvement.${floorPending ? ` Minimum epochs delay stopping until epoch ${stopping.minEpochs}.` : ''}` };
}

interface RunDetailsProps { batch: FrozenBatch; run?: TrainingRun; peakRamGb?: number; project?: string; history?: TrainingHistory; illustrative?: boolean }

/** A single history request supplies both the patience tile and the epoch charts. */
function LiveRunDetails(props: RunDetailsProps & { project: string; run: TrainingRun }) {
  const history = useQuery(runHistoryOptions(props.project, props.batch.id, props.run));
  return <RunDetailsContent {...props} history={history.data} historyPending={history.isPending} historyError={history.error} onHistoryRetry={() => void history.refetch()} />;
}

const detailTabs = [
  { id: 'overview', label: 'Overview' },
  { id: 'checkpoints', label: 'Checkpoints' },
  { id: 'diagnostics', label: 'Diagnostics' },
  { id: 'artifacts', label: 'Artifacts' },
] as const;
type DetailTab = typeof detailTabs[number]['id'];

export function RunDetails(props: RunDetailsProps) {
  return props.project && props.run && !props.history && !props.illustrative
    ? <LiveRunDetails {...props} project={props.project} run={props.run} /> : <RunDetailsContent {...props} />;
}

function RunDetailsContent({ batch, run, peakRamGb, project, history, illustrative = false, historyPending = false, historyError = null, onHistoryRetry }: RunDetailsProps & { historyPending?: boolean; historyError?: Error | null; onHistoryRetry?: () => void }) {
  const [tab, setTab] = useState<DetailTab>('overview');
  const id = useId();
  if (!run) return <p className="muted">This run has not been queued yet.</p>;
  const recipe = batch.manifest.configurations.find((item) => item.id === run.candidateId)?.recipe;
  const validationMetric = recipe?.checkpointMetric === 'validation_auroc' ? 'auroc' : recipe?.checkpointMetric === 'validation_accuracy' ? 'accuracy' : 'loss';
  const latestCheckpoint = recipe?.model.toLowerCase() === 'nnmil' && recipe.nnmilCheckpointSelection === 'latest';
  const stopping = runStoppingMetric(run, recipe, historyError ? undefined : history);
  return <section className="experiment-run-detail" aria-label="Selected run details">
    <div className="experiment-tracking-heading"><div><span className="experiment-detail-eyebrow">Selected run</span><h3>{runLabel(batch, run)}</h3></div><Badge tone={run.status === 'failed' || run.status === 'interrupted' ? 'warning' : run.status === 'completed' ? 'success' : 'neutral'}>{run.status}</Badge></div>
    {run.error ? <p className="callout callout-warning" role="status">{run.error}</p> : null}
    {run.progressWarning ? <p className="callout callout-warning" role="status">{run.progressWarning}</p> : null}
    <div className="experiment-run-metrics" aria-label="Latest reported run metrics">
      <div><span>Completed epochs</span><EpochProgress run={run} /></div>
      <div><span>Training loss</span><strong>{metricValue(run.progress?.trainingLoss)}</strong><small>Latest completed epoch</small></div>
      <div><span>Validation {validationMetric === 'auroc' ? 'AUROC' : validationMetric}</span><strong>{run.progress?.validation?.available === false ? '—' : metricValue(run.progress?.validation?.[validationMetric])}</strong><small>{latestCheckpoint ? 'Latest training monitor' : 'Latest checkpoint selection metric'}</small></div>
      <div><span>{stopping.label}</span><strong className="experiment-run-text-metric">{stopping.value}</strong><small>{stopping.detail}</small></div>
    </div>
    <div className="experiment-run-tabs" role="tablist" aria-label="Run details" onKeyDown={(event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const current = detailTabs.findIndex((item) => item.id === tab);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? detailTabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + detailTabs.length) % detailTabs.length;
      setTab(detailTabs[next].id);
      event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
    }}>
      {detailTabs.map((item) => <button type="button" role="tab" key={item.id} id={`${id}-tab-${item.id}`} aria-controls={`${id}-panel-${item.id}`} aria-selected={tab === item.id} tabIndex={tab === item.id ? 0 : -1} onClick={() => setTab(item.id)}>{item.label}</button>)}
    </div>
    <div className="experiment-run-panel" role="tabpanel" id={`${id}-panel-overview`} aria-labelledby={`${id}-tab-overview`} hidden={tab !== 'overview'} tabIndex={0}>
      <ErrorNotice error={historyError} />
      {historyError ? <><p className="muted">History refresh failed. Any curves shown are the last successfully loaded measurements.</p><button className="btn btn-secondary btn-small" type="button" onClick={onHistoryRetry}>Retry epoch history</button></> : null}
      {historyPending ? <p role="status">Loading epoch history…</p> : history ? <LossHistory history={history} validationMetric={validationMetric} finished={!['queued', 'running'].includes(run.status)} illustrative={illustrative} latestCheckpoint={latestCheckpoint} /> : <p className="muted">{illustrative ? 'No synthetic epoch history is included for this run.' : project ? 'No epoch history has been recorded yet.' : 'Open this run in its project to load epoch history.'}</p>}
    </div>
    <div className="experiment-run-panel" role="tabpanel" id={`${id}-panel-checkpoints`} aria-labelledby={`${id}-tab-checkpoints`} hidden={tab !== 'checkpoints'} tabIndex={0}>
      <h4>Checkpoint validation and held-out assessment</h4>
      <p className="muted">{latestCheckpoint ? 'The nnMIL protocol selects the latest completed epoch. Validation monitors training.' : 'Validation selects the checkpoint.'} Held-out assessment measures the saved checkpoint on the assessment partition.</p>
      <div className="experiment-checkpoint-grid"><section><h5>Checkpoint validation</h5><p>Checkpoint validation: {metricsText(run.metrics?.validation.selected)}</p><MetricEvidence details={run.metrics?.validation} /></section><section><h5>Held-out assessment</h5><p>Held-out assessment: {metricsText(run.metrics?.assessment.selected)}</p><MetricEvidence details={run.metrics?.assessment} /></section></div>
    </div>
    <div className="experiment-run-panel" role="tabpanel" id={`${id}-panel-diagnostics`} aria-labelledby={`${id}-tab-diagnostics`} hidden={tab !== 'diagnostics'} tabIndex={0}><RunDiagnostics run={run} peakRamGb={peakRamGb} /></div>
    <div className="experiment-run-panel" role="tabpanel" id={`${id}-panel-artifacts`} aria-labelledby={`${id}-tab-artifacts`} hidden={tab !== 'artifacts'} tabIndex={0}>
      <h4>Run artifacts</h4>{illustrative && run.checkpointPath ? <p className="muted">Reference only; no checkpoint file</p> : null}<dl className="experiment-run-facts experiment-artifact-paths"><div><dt>Run ID</dt><dd><code>{run.id}</code></dd></div>{run.checkpointPath ? <div><dt>Checkpoint</dt><dd><code>{run.checkpointPath}</code></dd></div> : null}{run.outputPath ? <div><dt>Output</dt><dd><code>{run.outputPath}</code></dd></div> : null}</dl>
      {!run.checkpointPath && !run.outputPath ? <p className="muted">No checkpoint or output path has been recorded for this run.</p> : null}
    </div>
  </section>;
}

export function RunTable({ batch, execution, project, histories, illustrative = false }: { batch: FrozenBatch; execution?: TrainingExecution | null; project?: string; histories?: Record<string, TrainingHistory>; illustrative?: boolean }) {
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [selection, setSelection] = useState<string | null>(null);
  const id = useId();
  const actual = new Map(execution?.runs.map((run) => [run.id, run]) ?? []);
  const query = search.trim().toLocaleLowerCase();
  const filtered = batch.manifest.runs.filter((planned) => (filter === 'all' || (actual.get(planned.id)?.status ?? 'planned') === filter)
    && (!query || `${runLabel(batch, planned)} ${planned.id} ${planned.candidateId} ${planned.splitPlanId}`.toLocaleLowerCase().includes(query)));
  const pages = Math.max(1, Math.ceil(filtered.length / 50));
  const current = Math.min(page, pages - 1);
  const visible = filtered.slice(current * 50, (current + 1) * 50);
  const selected = visible.find((planned) => planned.id === selection) ?? visible.find((planned) => actual.get(planned.id)?.status === 'running') ?? visible[0];
  return <div className="experiment-run-tracker">
    <div className="experiment-tracking-heading"><div><h3>Runs <span className="muted">({batch.manifest.runs.length})</span></h3><p className="experiment-tracking-hint">Select a run to inspect its metrics, checkpoints, and diagnostics.</p></div></div>
    <div className="experiment-run-toolbar">
      <label className="experiment-run-search" htmlFor={`${id}-search`}><span>Search runs</span><input type="search" id={`${id}-search`} className="field" placeholder="Configuration, fold, seed, or run ID" value={search} onChange={(event) => { setSearch(event.target.value); setPage(0); }} /></label>
      <label className="experiment-run-filter" htmlFor={`${id}-filter`}>Status<select id={`${id}-filter`} className="field" value={filter} onChange={(event) => { setFilter(event.target.value); setPage(0); }}><option value="all">All statuses</option>{['planned', 'queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted'].map((status) => <option key={status} value={status}>{status.charAt(0).toUpperCase() + status.slice(1)}</option>)}</select></label>
      {search || filter !== 'all' ? <button type="button" className="btn btn-secondary btn-small" onClick={() => { setSearch(''); setFilter('all'); setPage(0); }}>Clear filters</button> : null}
      <span className="experiment-run-count" role="status">{filtered.length === batch.manifest.runs.length ? `${filtered.length} runs` : `${filtered.length} of ${batch.manifest.runs.length} runs`}</span>
    </div>
    <div className="development-table experiment-runs-table"><table><caption className="sr-only">Batch runs. Select a run name to view its details below.</caption><thead><tr><th>Run</th><th>Status</th><th>Epoch</th><th>Training loss</th><th>Current validation loss</th></tr></thead><tbody>{visible.map((planned) => {
      const run = actual.get(planned.id);
      return <tr key={planned.id} className={selected?.id === planned.id ? 'is-selected' : undefined}><th scope="row"><button type="button" className="experiment-run-select" aria-pressed={selected?.id === planned.id} onClick={() => setSelection(planned.id)}>{runLabel(batch, planned)}</button></th><td><Badge tone={run?.status === 'failed' || run?.status === 'interrupted' ? 'warning' : run?.status === 'completed' ? 'success' : 'neutral'}>{run?.status ?? 'planned'}</Badge>{run?.progressWarning ? <small>Progress unavailable</small> : null}</td><td><EpochProgress run={run} /></td><td>{metricValue(run?.progress?.trainingLoss)}</td><td>{run?.progress?.validation?.available === false ? '—' : metricValue(run?.progress?.validation?.loss)}</td></tr>;
    })}</tbody></table></div>
    {!filtered.length ? <p className="muted">No runs match these filters.</p> : null}
    <p className="experiment-tracking-hint">Epoch progress counts completed epochs; early stopping can finish before the maximum.</p>
    {pages > 1 ? <div className="inline-actions"><button type="button" className="btn btn-secondary btn-small" disabled={!current} onClick={() => setPage(current - 1)}>Previous</button><span>Page {current + 1} of {pages}</span><button type="button" className="btn btn-secondary btn-small" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Next</button></div> : null}
    {selected ? <RunDetails key={selected.id} project={project} batch={batch} run={actual.get(selected.id)} peakRamGb={execution?.telemetry?.peak.runRssGb[selected.id]} history={histories?.[selected.id]} illustrative={illustrative} /> : null}
  </div>;
}
