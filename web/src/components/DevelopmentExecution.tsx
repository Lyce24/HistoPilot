import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { development, trainingActive, latestExecution } from '../api/development';
import type { DevelopmentResults, FrozenBatch, TrainingExecution, TrainingMetricDetails, TrainingMetrics, TrainingRun, TrainingRuntime } from '../api/development';
import { Badge, ErrorNotice } from './ui';
import { Findings } from './ScientificUI';
import { downloadJSON } from '../lib/download';

export function executionActions(execution?: TrainingExecution | null) {
  return {
    launch: !execution,
    cancel: trainingActive(execution) && !execution?.cancelRequested,
    resume: Boolean(execution && ['failed', 'cancelled', 'interrupted'].includes(execution.status)
      && execution.runCounts.completed < execution.runCounts.total),
  };
}

export default function DevelopmentExecution({ project, batch, implemented, knownExecution, view, allowChanges = true }: {
  project: string; batch: FrozenBatch; implemented: boolean; knownExecution?: TrainingExecution;
  view: 'batches' | 'runs' | 'results'; allowChanges?: boolean;
}) {
  const client = useQueryClient();
  const [error, setError] = useState<Error | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const operationIds = useRef<Partial<Record<'launch' | 'cancel' | 'resume', string>>>({});
  const submitting = useRef(false);
  const runtime = useQuery({ queryKey: ['training-runtime', project], queryFn: () => development.runtime(project), enabled: implemented, staleTime: 30000 });
  const executionQuery = useQuery({
    queryKey: ['training-execution', project, batch.id], queryFn: () => development.execution(project, batch.id), enabled: implemented,
    refetchInterval: (query) => trainingActive(latestExecution(query.state.data, knownExecution)) ? 3000 : false,
  });
  const execution = latestExecution(executionQuery.data, knownExecution);
  const results = useQuery({
    queryKey: ['development-results', project, batch.id], queryFn: () => development.results(project, batch.id),
    enabled: implemented && view === 'results',
    refetchInterval: trainingActive(execution) ? 3000 : false,
  });
  const status = execution?.status;
  useEffect(() => {
    if (status && status !== 'running' && status !== 'queued') {
      void client.invalidateQueries({ queryKey: ['development-results', project, batch.id] });
    }
  }, [status, client, project, batch.id]);
  async function act(action: 'launch' | 'cancel' | 'resume') {
    if (submitting.current || (!allowChanges && action !== 'cancel')) return;
    submitting.current = true; setPending(action); setError(null);
    const operationId = operationIds.current[action] ?? crypto.randomUUID();
    operationIds.current[action] = operationId;
    try {
      const updated = await development[action](project, batch.id, operationId);
      client.setQueryData(['training-execution', project, batch.id], updated);
      delete operationIds.current[action];
      await Promise.all([
        client.invalidateQueries({ queryKey: ['development-batches', project] }),
        client.invalidateQueries({ queryKey: ['model-experiments', project] }),
        client.invalidateQueries({ queryKey: ['model-experiment', project] }),
        client.invalidateQueries({ queryKey: ['development-results', project, batch.id] }),
      ]);
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Training action failed.')); }
    finally { submitting.current = false; setPending(null); }
  }
  return <div className="development-execution">
    <ErrorNotice error={error ?? executionQuery.error ?? runtime.error ?? (view === 'results' ? results.error : null)} />
    {implemented ? <>
      {allowChanges || executionActions(execution).cancel ? <TrainingControls execution={execution} runtime={runtime.data} checking={executionQuery.isPending || executionQuery.isError} pending={pending} onAction={(action) => void act(action)} /> : <p className="muted">Restore this record to Active to launch or resume training.</p>}
      {executionQuery.isError || runtime.isError ? <button type="button" className="btn btn-secondary btn-small" onClick={() => { void executionQuery.refetch(); void runtime.refetch(); }}>Retry training status</button> : null}
      {runtime.data ? <details><summary>Device &amp; runtime details</summary><p>{runtime.data.available ? 'Available' : 'Unavailable'} · {runtime.data.cudaAvailable ? `${runtime.data.gpuCount} CUDA GPUs` : 'No CUDA GPU detected'}</p><Findings findings={runtime.data.findings} />{runtime.data.gpus?.map((gpu) => <p key={gpu.index}>GPU {gpu.index}: {gpu.name} · {gib(gpu.freeMemoryGb)} free / {gib(gpu.totalMemoryGb)} total · Driver {gpu.driverVersion}</p>)}<dl className="development-paths"><dt>Python</dt><dd><code>{runtime.data.python}</code></dd>{Object.entries(runtime.data.versions).map(([name, version]) => <div key={name}><dt>{name}</dt><dd>{version ?? 'Unavailable'}</dd></div>)}</dl></details> : null}
      {execution ? <ExecutionEvidence execution={execution} /> : <p className="muted">This batch is frozen and has not been launched. Launch currently supports ABMIL classification with a version 4 k-fold development protocol.</p>}
    </> : <p className="muted">Training execution is unavailable from this service. Frozen plans remain available for review and export.</p>}
    {view === 'runs' ? <RunTable batch={batch} execution={execution} /> : null}
    {view === 'results' ? <ResultsTable batch={batch} results={results.data} loading={implemented && results.isPending} /> : null}
  </div>;
}

export function TrainingControls({ execution, runtime, checking, pending, onAction }: {
  execution?: TrainingExecution | null; runtime?: TrainingRuntime; checking: boolean; pending: string | null;
  onAction: (action: 'launch' | 'cancel' | 'resume') => void;
}) {
  const allowed = executionActions(execution);
  return <div className="development-training-controls">
    <div className="inline-actions">
      {allowed.launch ? <button type="button" className="btn btn-primary" disabled={Boolean(pending) || checking || !runtime?.available} onClick={() => onAction('launch')}>{pending === 'launch' ? 'Launching…' : 'Launch batch'}</button> : null}
      {trainingActive(execution) ? <button type="button" className="btn btn-secondary" disabled={Boolean(pending) || !allowed.cancel} onClick={() => onAction('cancel')}>{execution?.cancelRequested ? 'Cancellation requested' : pending === 'cancel' ? 'Requesting cancellation…' : 'Cancel batch'}</button> : null}
      {allowed.resume ? <button type="button" className="btn btn-primary" disabled={Boolean(pending) || !runtime?.available} onClick={() => onAction('resume')}>{pending === 'resume' ? 'Resuming…' : 'Resume unfinished runs'}</button> : null}
      <Badge>{execution?.status ?? 'Not launched'}</Badge>
    </div>
    {!runtime ? <p className="muted">Checking the training runtime…</p> : !runtime.available ? <><p className="callout">Training cannot launch until the runtime is available.</p><Findings findings={runtime.findings} /></> : null}
    {allowed.resume ? <p className="muted">Completed runs are retained. Unfinished runs resume their latest checkpoint when available; otherwise they restart.</p> : null}
    {execution?.cancelRequested && trainingActive(execution) ? <p role="status">Cancellation is being applied. Status will update when the worker and active runs stop.</p> : null}
  </div>;
}

export function ExecutionEvidence({ execution }: { execution: TrainingExecution }) {
  const counts = execution.runCounts;
  const telemetry = execution.telemetry;
  return <>
    <div className="development-run-counts" aria-live="polite"><strong>{counts.completed} / {counts.total} completed</strong><span>{counts.running} running</span><span>{counts.queued} queued</span><span>{counts.failed} failed</span><span>{counts.cancelled} cancelled</span>{counts.interrupted ? <span>{counts.interrupted} interrupted</span> : null}</div>
    <Findings findings={execution.findings} />
    {execution.resourcePlan ? <p>At launch: up to <strong>{execution.resourcePlan.effectiveConcurrency}</strong> concurrent runs from {execution.resourcePlan.requestedConcurrency} requested. {execution.resourcePlan.note}</p> : null}
    {telemetry?.latest ? <details className="setup-details"><summary>Resource usage &amp; recorded peaks</summary>
      <div className="development-telemetry"><p>Host RAM available: <strong>{gib(telemetry.latest.host.availableRamGb)}</strong></p>
      {telemetry.latest.gpus.map((gpu) => <p key={gpu.index}>GPU {gpu.index} device usage: <strong>{gib(gpu.usedMemoryGb)}</strong> / {gib(gpu.totalMemoryGb)}<br />Observed peak: {gib(telemetry.peak.gpuUsedMemoryGb[String(gpu.index)])}</p>)}</div>
      <p className="muted">Samples every {telemetry.intervalSeconds}s. GPU usage includes other programs. Process-tree RAM can count shared pages more than once; short peaks can occur between samples.</p>
      {telemetry.latest.runs.length ? <div className="development-table"><table><thead><tr><th>Run</th><th>Current process-tree RAM</th><th>Observed peak RAM</th></tr></thead><tbody>{telemetry.latest.runs.map((run) => <tr key={run.runId}><td><code>{run.runId.slice(0, 16)}…</code></td><td>{gib(run.rssGb)}</td><td>{gib(telemetry.peak.runRssGb[run.runId])}</td></tr>)}</tbody></table></div> : <p className="muted">No active run measurements. Saved per-run peaks are available under Runs.</p>}
      {telemetry.latest.gpuProbeError ? <p role="status">GPU measurements unavailable: {telemetry.latest.gpuProbeError}</p> : null}
      <p className="muted">Recorded at {telemetry.latest.at}</p>
    </details> : null}
    <details><summary>Worker and saved artifacts</summary><dl className="development-paths"><dt>Session</dt><dd><code>{execution.sessionName}</code></dd><dt>Worker log</dt><dd><code>{execution.logPath}</code></dd><dt>Output directory</dt><dd><code>{execution.outputPath}</code></dd>{execution.computePath ? <><dt>Pinned training code</dt><dd><code>{execution.computePath}</code></dd></> : null}{execution.provenancePath ? <><dt>Attempt and driver history</dt><dd><code>{execution.provenancePath}</code></dd></> : null}{telemetry?.path ? <><dt>Resource history</dt><dd><code>{telemetry.path}</code></dd></> : null}<dt>Updated</dt><dd>{execution.updatedAt}</dd></dl></details>
  </>;
}

const gib = (value?: number | null) => typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)} GiB` : 'Unavailable';

function metricsText(metrics?: TrainingMetrics | null) {
  if (!metrics) return '—';
  if (metrics.available === false) return metrics.reason ?? 'Unavailable';
  const values = (['accuracy', 'auroc', 'auprc', 'loss', 'balancedAccuracy', 'macroF1'] as const).filter((name) => metrics[name] !== undefined);
  return values.length ? values.map((name) => `${name}: ${typeof metrics[name] === 'number' && Number.isFinite(metrics[name]) ? metrics[name]!.toFixed(4) : 'Unavailable'}`).join(' · ') : '—';
}

function MetricEvidence({ details }: { details?: TrainingMetricDetails }) {
  if (!details) return null;
  return <details><summary>{details.unit} scoring details</summary><p>Class order: {details.classOrder.join(', ')}{details.positiveClass ? ` · Positive class: ${details.positiveClass}` : ''}</p><p>Slide: {metricsText(details.slide)}</p><p>Patient ({details.patientAggregation.replaceAll('_', ' ')}): {metricsText(details.patient)}</p>{details.selected.missingClasses?.length ? <p>Missing classes: {details.selected.missingClasses.join(', ')}. AUROC may be unavailable.</p> : null}</details>;
}

function RunDiagnostics({ run, peakRamGb }: { run?: TrainingRun; peakRamGb?: number }) {
  const progress = run?.progress;
  if (progress?.learningRate === undefined && progress?.cudaPeakAllocatedBytes === undefined && peakRamGb === undefined) return null;
  return <details><summary>Training diagnostics</summary>
    {progress?.learningRate !== undefined ? <p>Learning rate: {progress.learningRate.toExponential(3)}</p> : null}
    {progress?.cudaPeakAllocatedBytes !== undefined ? <p>Run CUDA allocator peak: {gib(progress.cudaPeakAllocatedBytes / 2 ** 30)} allocated / {gib(progress.cudaPeakReservedBytes === undefined ? undefined : progress.cudaPeakReservedBytes / 2 ** 30)} reserved</p> : null}
    {peakRamGb !== undefined ? <p>Sampled process-tree RAM peak: {gib(peakRamGb)}. Shared pages may be counted more than once.</p> : null}
  </details>;
}

export function RunTable({ batch, execution }: { batch: FrozenBatch; execution?: TrainingExecution | null }) {
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(batch.manifest.runs.length / 50));
  const current = Math.min(page, pages - 1);
  const candidates = new Map(batch.manifest.configurations.map((item) => [item.id, item.number]));
  const plans = new Map(batch.manifest.splitPlans.map((item) => [item.id, item.planId]));
  const actual = new Map(execution?.runs.map((run) => [run.id, run]) ?? []);
  return <><div className="development-table"><table><thead><tr><th>Configuration</th><th>Training seed</th><th>Frozen split plan</th><th>Status</th><th>Checkpoint validation</th><th>Held-out assessment / artifacts</th></tr></thead><tbody>{batch.manifest.runs.slice(current * 50, (current + 1) * 50).map((planned) => {
    const run = actual.get(planned.id);
    return <tr key={planned.id}><td>{candidates.get(planned.candidateId)}</td><td>{planned.trainingSeed}</td><td>{plans.get(planned.splitPlanId)}</td><td>{run?.status ?? 'planned'}{run?.progress ? <div className="development-epoch">Epoch {run.progress.epoch} / {run.progress.maxEpochs}{typeof run.progress.trainingLoss === 'number' ? <p>Training loss: {run.progress.trainingLoss.toFixed(4)}</p> : null}{typeof run.progress.validation?.loss === 'number' ? <p>Current validation loss: {run.progress.validation?.loss.toFixed(4)}</p> : null}</div> : null}{run?.progressWarning ? <p className="callout callout-warning" role="status">{run.progressWarning}</p> : null}<RunDiagnostics run={run} peakRamGb={execution?.telemetry?.peak.runRssGb[planned.id]} /></td><td>{metricsText(run?.metrics?.validation.selected)}<MetricEvidence details={run?.metrics?.validation} /></td><td>{metricsText(run?.metrics?.assessment.selected)}<MetricEvidence details={run?.metrics?.assessment} />{run?.error ? <p role="status">{run.error}</p> : null}{run?.checkpointPath || run?.outputPath ? <details><summary>Run artifacts</summary>{run.checkpointPath ? <p>Checkpoint: <code>{run.checkpointPath}</code></p> : null}{run.outputPath ? <p>Output: <code>{run.outputPath}</code></p> : null}</details> : null}</td></tr>;
  })}</tbody></table></div><div className="inline-actions"><button type="button" className="btn btn-secondary btn-small" disabled={!current} onClick={() => setPage(current - 1)}>Previous</button><span>Page {current + 1} of {pages}</span><button type="button" className="btn btn-secondary btn-small" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Next</button></div></>;
}

export function ResultsTable({ batch, results, loading }: { batch: FrozenBatch; results?: DevelopmentResults; loading: boolean }) {
  const numbers = new Map(batch.manifest.configurations.map((item) => [item.id, item.number]));
  return <>
    <p>Validation selects each run’s checkpoint. Assessment predictions use its held-out fold. Complete folds form OOF predictions for one configuration, training seed and split seed; test-cohort evaluation remains separate.</p>
    {results?.selectionNote ? <p className="muted">{results.selectionNote}</p> : null}
    {results?.findings ? <Findings findings={results.findings} /> : null}
    {loading ? <p role="status">Loading development results…</p> : !results?.candidates.length ? <p>No completed development results yet. Frozen plans do not contain measured scores or OOF predictions.</p> : <>
      <button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${batch.manifest.spec.batchName}-results.json`, results)}>Export development results</button>
      <div className="development-table"><table><thead><tr><th>Configuration</th><th>Training seed</th><th>Split seed</th><th>Coverage</th><th>OOF metrics</th><th>Predictions</th></tr></thead><tbody>{results.candidates.map((result) => <tr key={`${result.candidateId}-${result.trainingSeed}-${result.splitSeed}`}><td>{numbers.get(result.candidateId) ?? result.candidateId}</td><td>{result.trainingSeed}</td><td>{result.splitSeed}</td><td>{result.complete ? 'Complete' : 'Incomplete'} · {result.completedRuns} / {result.totalRuns} runs{result.assessmentSlideCount !== undefined ? <p>{result.assessmentSlideCount} assessment slides</p> : null}</td><td>{result.complete ? <>{metricsText(result.metrics)}<MetricEvidence details={result.metricDetails} /></> : 'Waiting for all folds'}</td><td>{result.complete && result.oofPath ? <code>{result.oofPath}</code> : 'Unavailable'}</td></tr>)}</tbody></table></div>
    </>}
  </>;
}
