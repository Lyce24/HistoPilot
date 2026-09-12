import { useId, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { development, trainingActive } from '../api/development';
import type { FrozenBatch, PlannedRun, TrainingExecution, TrainingHistory, TrainingMetricDetails, TrainingMetrics, TrainingRun } from '../api/development';
import { finiteNumber } from '../lib/evidenceCharts';
import { downloadJSON } from '../lib/download';
import CurveChart from './CurveChart';
import { Badge, ErrorNotice } from './ui';
import './ExperimentTracking.css';

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
  return <details><summary>{details.unit} scoring details</summary><p>Class order: {details.classOrder.join(', ')}{details.positiveClass ? ` · Positive class: ${details.positiveClass}` : ''}</p><p>Slide: {metricsText(details.slide)}</p><p>Patient ({details.patientAggregation.replaceAll('_', ' ')}): {metricsText(details.patient)}</p>{details.selected.missingClasses?.length ? <p>Missing classes: {details.selected.missingClasses.join(', ')}. AUROC may be unavailable.</p> : null}</details>;
}

function RunDiagnostics({ run, peakRamGb }: { run?: TrainingRun; peakRamGb?: number }) {
  const progress = run?.progress;
  if (progress?.learningRate === undefined && progress?.cudaPeakAllocatedBytes === undefined && peakRamGb === undefined) return null;
  return <details><summary>Training diagnostics</summary>
    {finiteNumber(progress?.learningRate) ? <p>Learning rate: {progress.learningRate.toExponential(3)}</p> : null}
    {finiteNumber(progress?.cudaPeakAllocatedBytes) ? <p>Run CUDA allocator peak: {gib(progress.cudaPeakAllocatedBytes / 2 ** 30)} allocated / {gib(progress.cudaPeakReservedBytes === undefined ? undefined : progress.cudaPeakReservedBytes / 2 ** 30)} reserved</p> : null}
    {peakRamGb !== undefined ? <p>Sampled process-tree RAM peak: {gib(peakRamGb)}. Shared pages may be counted more than once.</p> : null}
  </details>;
}

export function runLabel(batch: FrozenBatch, run: PlannedRun | TrainingRun) {
  const configuration = batch.manifest.configurations.find((item) => item.id === run.candidateId);
  const split = batch.manifest.splitPlans.find((item) => item.id === run.splitPlanId);
  return `Config ${configuration?.number ?? '?'} · ${finiteNumber(split?.fold) ? `Fold ${split.fold + 1}` : split?.planId ?? 'Unknown split'} · Train seed ${run.trainingSeed}${split?.seed !== undefined ? ` · Split seed ${split.seed}` : ''}`;
}

export function EpochProgress({ run }: { run?: TrainingRun }) {
  const progress = run?.progress;
  if (!progress || !finiteNumber(progress.epoch) || !finiteNumber(progress.maxEpochs) || progress.maxEpochs <= 0) return <span className="muted">No epoch reported</span>;
  return <div className="experiment-epoch"><span>Epoch {progress.epoch} / {progress.maxEpochs}</span><progress aria-label="Completed epochs" value={Math.min(progress.epoch, progress.maxEpochs)} max={progress.maxEpochs} />{run?.status === 'completed' && progress.epoch < progress.maxEpochs ? <small>Stopped early</small> : null}</div>;
}

export function ResourceCards({ execution }: { execution: TrainingExecution }) {
  const telemetry = execution.telemetry;
  const sample = telemetry?.latest;
  const current = trainingActive(execution);
  if (!sample) return <p className="muted">Resource measurements have not been recorded yet.</p>;
  const stale = current && Date.now() - Date.parse(sample.at) > Math.max(45000, telemetry.intervalSeconds * 3000);
  return <section className="experiment-resource-section" aria-label="Recorded resource usage">
    <div className="experiment-tracking-heading"><h3>{current ? 'Resource usage' : 'Last resource sample'}</h3><small>Recorded {sample.at ? new Date(sample.at).toLocaleString() : 'time unavailable'}</small></div>
    <div className="experiment-resource-cards">
      {sample.gpus.map((gpu) => <div className="experiment-resource-card" key={gpu.index}><span>GPU {gpu.index} · {gpu.name}</span><strong>{finiteNumber(gpu.utilizationPercent) ? `${gpu.utilizationPercent.toFixed(0)}% utilization` : 'Utilization unavailable'}</strong><small>{gib(gpu.usedMemoryGb)} used / {gib(gpu.totalMemoryGb)} total</small><small>Observed peak: {gib(telemetry.peak.gpuUsedMemoryGb[String(gpu.index)])}</small></div>)}
      <div className="experiment-resource-card"><span>Host memory</span><strong>{gib(sample.host.availableRamGb)} available</strong><small>{gib(sample.host.totalRamGb)} total RAM</small></div>
    </div>
    <p className="muted">Samples every {telemetry.intervalSeconds}s. GPU usage includes other programs.{!current ? ' These are recorded measurements, not live device usage.' : ''}</p>
    {stale ? <p className="callout callout-warning" role="status">The resource sample is stale. Measurements may no longer reflect the active runs.</p> : null}
    {sample.gpuProbeError ? <p className="callout callout-warning" role="status">GPU measurements unavailable: {sample.gpuProbeError}</p> : null}
  </section>;
}

export function LossHistory({ history, validationMetric = 'loss', finished = false }: { history?: TrainingHistory; validationMetric?: 'loss' | 'accuracy' | 'auroc'; finished?: boolean }) {
  const rows = history?.rows ?? [];
  if (history?.warning) return <p className="callout callout-warning" role="status">{history.warning}</p>;
  if (!rows.length) return <p className="callout">{finished ? 'No epoch history was recorded for this run.' : 'Loss history appears after the first completed epoch.'}</p>;
  const unit = rows.find((row) => row.checkpointUnit)?.checkpointUnit;
  const description = `Recorded at each completed epoch. Validation${unit ? ` uses ${unit} scoring and` : ''} selects checkpoints; it is not held-out assessment.`;
  const xRange = [rows[0].epoch, Math.max(rows[0].epoch + 1, rows[rows.length - 1].epoch)] as const;
  const trainingPoints = rows.map((row) => ({ x: row.epoch, y: row.trainingLoss }));
  const validationPoints = rows.map((row) => ({ x: row.epoch, y: row.validation.loss ?? null }));
  const hasLoss = [...trainingPoints, ...validationPoints].some((point) => finiteNumber(point.y));
  return <>
    {history?.truncated ? <p className="muted">Showing the latest {rows.length} of {history.totalRows} completed epochs. The complete history remains in the run output directory.</p> : null}
    <div className="experiment-history-charts">
      {hasLoss ? <CurveChart title="Loss history" description={description} xLabel="Epoch" integerX yLabel="Loss" xRange={xRange} series={[{ label: 'Training loss', points: trainingPoints }, { label: 'Validation loss', points: validationPoints, dashed: true }]} /> : <p className="callout">No loss measurements are available in the recorded history.</p>}
      {validationMetric !== 'loss' ? <CurveChart title={`Validation ${validationMetric === 'auroc' ? 'AUROC' : 'accuracy'}`} description={`Checkpoint selection metric${unit ? ` · ${unit} scoring` : ''}. Missing values remain gaps.`} xLabel="Epoch" integerX yLabel={validationMetric === 'auroc' ? 'AUROC' : 'Accuracy'} xRange={xRange} yRange={[0, 1]} series={[{ label: `Validation ${validationMetric === 'auroc' ? 'AUROC' : 'accuracy'}`, points: rows.map((row) => ({ x: row.epoch, y: row.validation[validationMetric] ?? null })) }]} /> : null}
    </div>
    {history ? <button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${history.runId}-epoch-history.json`, history)}>Export epoch history</button> : null}
  </>;
}

function RunHistory({ project, batch, run }: { project: string; batch: FrozenBatch; run: TrainingRun }) {
  const active = ['queued', 'running'].includes(run.status);
  // A terminal transition must read the final epoch even when the poll stopped
  // just before the worker wrote it. The status key also refreshes resumed runs.
  const history = useQuery({ queryKey: ['training-history', project, batch.id, run.id, run.status], queryFn: () => development.history(project, batch.id, run.id), refetchInterval: active ? 3000 : false });
  const recipe = batch.manifest.configurations.find((item) => item.id === run.candidateId)?.recipe;
  const metric = recipe?.checkpointMetric === 'validation_auroc' ? 'auroc' : recipe?.checkpointMetric === 'validation_accuracy' ? 'accuracy' : 'loss';
  return <><ErrorNotice error={history.error} />{history.isError ? <><p className="muted">History refresh failed. Any curves shown are the last successfully loaded measurements.</p><button className="btn btn-secondary btn-small" type="button" onClick={() => void history.refetch()}>Retry epoch history</button></> : null}{history.isPending ? <p role="status">Loading epoch history…</p> : history.data ? <LossHistory history={history.data} validationMetric={metric} finished={!active} /> : null}</>;
}

export function RunDetails({ batch, run, peakRamGb, project }: { batch: FrozenBatch; run?: TrainingRun; peakRamGb?: number; project?: string }) {
  if (!run) return <p className="muted">This run has not been queued yet.</p>;
  return <section className="experiment-run-detail" aria-label="Selected run details">
    <div className="experiment-tracking-heading"><h3>{runLabel(batch, run)}</h3><Badge tone={run.status === 'failed' || run.status === 'interrupted' ? 'warning' : run.status === 'completed' ? 'success' : 'neutral'}>{run.status}</Badge></div>
    {run.error ? <p className="callout callout-warning" role="status">{run.error}</p> : null}
    {run.progressWarning ? <p className="callout callout-warning" role="status">{run.progressWarning}</p> : null}
    {project ? <RunHistory key={run.id} project={project} batch={batch} run={run} /> : null}
    <details><summary>Checkpoint validation and held-out assessment</summary><p>Checkpoint validation: {metricsText(run.metrics?.validation.selected)}</p><MetricEvidence details={run.metrics?.validation} /><p>Held-out assessment: {metricsText(run.metrics?.assessment.selected)}</p><MetricEvidence details={run.metrics?.assessment} /></details>
    <RunDiagnostics run={run} peakRamGb={peakRamGb} />
    {run.checkpointPath || run.outputPath ? <details><summary>Run artifacts</summary>{run.checkpointPath ? <p>Checkpoint: <code>{run.checkpointPath}</code></p> : null}{run.outputPath ? <p>Output: <code>{run.outputPath}</code></p> : null}</details> : null}
  </section>;
}

export function RunTable({ batch, execution, project }: { batch: FrozenBatch; execution?: TrainingExecution | null; project?: string }) {
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState('all');
  const [selection, setSelection] = useState<string | null>(null);
  const id = useId();
  const actual = new Map(execution?.runs.map((run) => [run.id, run]) ?? []);
  const filtered = batch.manifest.runs.filter((planned) => filter === 'all' || (actual.get(planned.id)?.status ?? 'planned') === filter);
  const pages = Math.max(1, Math.ceil(filtered.length / 50));
  const current = Math.min(page, pages - 1);
  const visible = filtered.slice(current * 50, (current + 1) * 50);
  const selected = visible.find((planned) => planned.id === selection) ?? visible.find((planned) => actual.get(planned.id)?.status === 'running') ?? visible[0];
  return <div className="experiment-run-tracker">
    <div className="experiment-tracking-heading"><h3>Runs <span className="muted">({batch.manifest.runs.length})</span></h3><label className="experiment-run-filter" htmlFor={`${id}-filter`}>Status<select id={`${id}-filter`} className="field" value={filter} onChange={(event) => { setFilter(event.target.value); setPage(0); }}><option value="all">All statuses</option>{['planned', 'queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted'].map((status) => <option key={status} value={status}>{status.charAt(0).toUpperCase() + status.slice(1)}</option>)}</select></label></div>
    <div className="development-table experiment-runs-table"><table><thead><tr><th>Run</th><th>Status</th><th>Epoch</th><th>Training loss</th><th>Current validation loss</th></tr></thead><tbody>{visible.map((planned) => {
      const run = actual.get(planned.id);
      return <tr key={planned.id} className={selected?.id === planned.id ? 'is-selected' : undefined}><th scope="row"><button type="button" className="experiment-run-select" aria-pressed={selected?.id === planned.id} onClick={() => setSelection(planned.id)}>{runLabel(batch, planned)}</button></th><td><Badge tone={run?.status === 'failed' || run?.status === 'interrupted' ? 'warning' : run?.status === 'completed' ? 'success' : 'neutral'}>{run?.status ?? 'planned'}</Badge>{run?.progressWarning ? <small>Progress unavailable</small> : null}</td><td><EpochProgress run={run} /></td><td>{metricValue(run?.progress?.trainingLoss)}</td><td>{metricValue(run?.progress?.validation?.loss)}</td></tr>;
    })}</tbody></table></div>
    {!filtered.length ? <p className="muted">No runs match this status.</p> : null}
    <p className="muted">Select a run for loss history and scoring details. Epoch progress counts completed epochs; early stopping can finish before the maximum.</p>
    {pages > 1 ? <div className="inline-actions"><button type="button" className="btn btn-secondary btn-small" disabled={!current} onClick={() => setPage(current - 1)}>Previous</button><span>Page {current + 1} of {pages}</span><button type="button" className="btn btn-secondary btn-small" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Next</button></div> : null}
    {selected ? <RunDetails key={selected.id} project={project} batch={batch} run={actual.get(selected.id)} peakRamGb={execution?.telemetry?.peak.runRssGb[selected.id]} /> : null}
  </div>;
}
