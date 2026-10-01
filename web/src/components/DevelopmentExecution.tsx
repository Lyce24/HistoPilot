import { formatStatistic } from '../api/statistics';
import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { development, managedByTaskCenter, trainingActive, latestExecution } from '../api/development';
import { taskCenterHref } from '../api/taskCenter';
import type { CandidateResult, DevelopmentResults, FrozenBatch, TrainingExecution, TrainingRuntime } from '../api/development';
import { Badge, ErrorNotice } from './ui';
import { ConfirmAction } from '../lib/taskCenterActions';
import LegacyRecordNote from './LegacyRecordNote';
import { Findings } from './ScientificUI';
import { downloadJSON } from '../lib/download';
import { gib, metricValue, metricsText, MetricEvidence, ResourceCards, RunTable } from './ExperimentTracking';
export { RunTable } from './ExperimentTracking';
export type ExperimentExecutionStage = 'planning' | 'running' | 'finished';


export default function DevelopmentExecution({ project, batch, implemented, knownExecution, view, allowChanges = true, readOnly = false, stage }: {
  project: string; batch: FrozenBatch; implemented: boolean; knownExecution?: TrainingExecution;
  view: 'batches' | 'runs' | 'results'; allowChanges?: boolean; readOnly?: boolean; stage?: ExperimentExecutionStage;
}) {
  const client = useQueryClient();
  const canChange = allowChanges && !readOnly && stage !== 'finished' && stage !== 'planning';
  const trackingEnabled = implemented && stage !== 'planning';
  const resultsEnabled = trackingEnabled && view === 'results';
  const [error, setError] = useState<Error | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const operationIds = useRef<Partial<Record<'launch' | 'cancel', string>>>({});
  const submitting = useRef(false);
  const executionQuery = useQuery({
    queryKey: ['training-execution', project, batch.id], queryFn: () => development.execution(project, batch.id), enabled: trackingEnabled,
    refetchInterval: (query) => trainingActive(latestExecution(query.state.data, knownExecution)) ? 5000 : false,
  });
  const execution = latestExecution(executionQuery.data, knownExecution);
  // Task Center batches are stopped and resumed from the Task Center (and the experiment's one
  // Resume). Only a batch never launched outside an experiment offers Launch, after the runtime check.
  const launchable = stage === undefined && !execution;
  const runtime = useQuery({ queryKey: ['training-runtime', project], queryFn: () => development.runtime(project), enabled: trackingEnabled && canChange && launchable, staleTime: 30000 });
  const managed = managedByTaskCenter(execution) || managedByTaskCenter(knownExecution);
  const ownerKey = execution?.taskCenter?.ownerKey ?? knownExecution?.taskCenter?.ownerKey ?? null;
  const taskCenterLink = managed ? taskCenterHref(ownerKey ? { owner: ownerKey } : {}) : undefined;
  const results = useQuery({
    queryKey: ['development-results', project, batch.id], queryFn: () => development.results(project, batch.id),
    enabled: resultsEnabled,
    // Partial results change only when a group's last fold finishes.
    refetchInterval: trainingActive(execution) ? 10000 : false,
  });
  const status = execution?.status;
  useEffect(() => {
    if (status && status !== 'running' && status !== 'queued') {
      void client.invalidateQueries({ queryKey: ['development-results', project, batch.id] });
    }
  }, [status, client, project, batch.id]);
  async function act(action: 'launch' | 'cancel') {
    if (submitting.current || !canChange || (stage !== undefined && action === 'launch')) return;
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
  if (stage === 'planning') return <p className="muted">Inputs and batches remain editable until the design is frozen. Runs and results unlock once the experiment starts.</p>;
  return <div className="development-execution">
    <ErrorNotice error={error ?? executionQuery.error ?? (canChange && launchable ? runtime.error : null) ?? (resultsEnabled ? results.error : null)} />
    {implemented ? <>
      {view !== 'results' ? <>
        {managed ? <ManagedBatchCancel execution={execution} pending={pending} readOnly={!canChange} onCancel={() => void act('cancel')} /> : <TrainingControls execution={execution} runtime={runtime.data} checking={executionQuery.isPending || executionQuery.isError} pending={pending} onLaunch={() => void act('launch')} allowLaunch={stage === undefined} readOnly={!canChange} />}
        {executionQuery.isError ? <p className="callout callout-warning" role="status">Tracking could not refresh. Any run status and measurements shown are the last known values.</p> : null}
        {executionQuery.isError || (canChange && launchable && runtime.isError) ? <button type="button" className="btn btn-secondary btn-small" onClick={() => { void executionQuery.refetch(); if (canChange && launchable) void runtime.refetch(); }}>Retry training status</button> : null}
        {execution ? <ExecutionStatus execution={execution} /> : <p className="muted">{stage ? 'This batch has not been queued yet.' : 'This batch is frozen and has not been launched.'}</p>}
      </> : null}
    </> : <p className="muted">Training execution is unavailable from this service. Frozen plans remain available for review and export.</p>}
    {view === 'runs' ? <RunTable project={trackingEnabled ? project : undefined} batch={batch} execution={execution} taskCenterHref={taskCenterLink} /> : null}
    {implemented && view !== 'results' && !managed ? execution ? <ExecutionEvidence project={trackingEnabled ? project : undefined} execution={execution} showStatus={false} /> : canChange && runtime.data ? <DeviceRuntime runtime={runtime.data} /> : null : null}
    {view === 'results' ? <>
      {/* The experiment can keep running (other batches, predictors) after this batch's folds are done. */}
      {stage !== 'finished' && (trainingActive(execution) || (stage === 'running' && !execution)) ? <p className="callout" role="status">Partial results: groups appear as their folds finish.</p> : null}
      {resultsEnabled && results.isError ? <><p className="callout callout-warning" role="status">Results could not refresh.{results.data ? ' Showing the last successfully loaded results.' : ''}</p><button type="button" className="btn btn-secondary btn-small" onClick={() => void results.refetch()}>Retry results</button></> : null}
      {!results.isError || results.data ? <ResultsTable project={trackingEnabled ? project : undefined} batch={batch} results={results.data} loading={resultsEnabled && results.isPending} /> : null}
    </> : null}
  </div>;
}

/**
 * One batch of a Task Center experiment can be cancelled on its own: the Task Center offers
 * the whole experiment or single folds, but no batch-wide action. The experiment's one Resume
 * (status row above) resumes it.
 */
export function ManagedBatchCancel({ execution, pending, readOnly, onCancel }: { execution?: TrainingExecution | null; pending: string | null; readOnly: boolean; onCancel: () => void }) {
  if (readOnly || !trainingActive(execution)) return null;
  if (execution?.cancelRequested) return <p className="muted" role="status">Cancelling this batch. Runs stop after saving what they can.</p>;
  return <div className="inline-actions"><ConfirmAction label="Cancel batch" busy={pending === 'cancel'} busyLabel="Requesting cancellation…" disabled={Boolean(pending)} question="Cancel this batch? Its runs stop after saving what they can; other batches keep running." confirmLabel="Cancel batch" onConfirm={onCancel} /></div>;
}

/**
 * A batch the Task Center does not run: Launch for one never launched (outside an experiment),
 * or, for a batch launched before the Task Center, its saved status, read-only.
 */
export function TrainingControls({ execution, runtime, checking, pending, onLaunch, allowLaunch = true, readOnly = false }: {
  execution?: TrainingExecution | null; runtime?: TrainingRuntime; checking: boolean; pending: string | null;
  onLaunch: () => void; allowLaunch?: boolean; readOnly?: boolean;
}) {
  const launch = !execution && allowLaunch && !readOnly;
  return <div className="development-training-controls">
    <div className="inline-actions">
      {launch ? <button type="button" className="btn btn-primary" disabled={Boolean(pending) || checking || !runtime?.available} onClick={onLaunch}>{pending === 'launch' ? 'Launching…' : 'Launch batch'}</button> : null}
      <Badge>{execution?.status ?? 'Not launched'}</Badge>
    </div>
    {launch ? !runtime ? <p className="muted">Checking the training runtime…</p> : !runtime.available ? <><p className="callout">Training cannot launch until the runtime is available.</p><Findings findings={runtime.findings} /></> : null : null}
    {execution ? <LegacyRecordNote /> : null}
  </div>;
}

export function ExecutionStatus({ execution }: { execution: TrainingExecution }) {
  const counts = execution.runCounts;
  return <>
    <div className="development-run-counts" aria-live="polite"><strong>{counts.completed} / {counts.total} completed</strong><span>{counts.running} running</span><span>{counts.queued} queued</span><span>{counts.failed} failed</span><span>{counts.cancelled} cancelled</span>{counts.interrupted ? <span>{counts.interrupted} interrupted</span> : null}</div>
    {execution.findings.length ? <Findings findings={execution.findings} /> : null}
  </>;
}

export function DeviceRuntime({ execution, runtime }: { execution?: TrainingExecution; runtime?: TrainingRuntime }) {
  const sample = execution?.telemetry?.latest;
  return <details className="experiment-worker-details"><summary>Device &amp; runtime details</summary>
    {sample ? <><p className="muted">Recorded by this batch’s worker.</p><dl className="development-paths"><div><dt>CPU cores</dt><dd>{sample.host.cpuCount}</dd></div><div><dt>Host RAM</dt><dd>{gib(sample.host.totalRamGb)}</dd></div><div><dt>Kernel</dt><dd>{sample.host.kernel || 'Not recorded'}</dd></div>{sample.gpus.map((gpu) => <div key={gpu.index}><dt>GPU {gpu.index}</dt><dd>{gpu.name} · {gib(gpu.totalMemoryGb)} VRAM · Driver {gpu.driverVersion}</dd></div>)}</dl></> : <p className="muted">Worker device measurements have not been recorded yet.</p>}
    {execution?.computeVersion ? <p>Pinned compute version: <code>{execution.computeVersion}</code></p> : null}
    {runtime ? <><h4>Current runtime readiness</h4><p>{runtime.available ? 'Available' : 'Unavailable'} · {runtime.cudaAvailable ? `${runtime.gpuCount} CUDA GPUs` : 'No CUDA GPU detected'}</p><Findings findings={runtime.findings} />{runtime.gpus?.map((gpu) => <p key={gpu.index}>GPU {gpu.index}: {gpu.name} · {gib(gpu.freeMemoryGb)} free / {gib(gpu.totalMemoryGb)} total · Driver {gpu.driverVersion}</p>)}<dl className="development-paths"><div><dt>Python</dt><dd><code>{runtime.python}</code></dd></div>{Object.entries(runtime.versions).map(([name, version]) => <div key={name}><dt>{name}</dt><dd>{version ?? 'Unavailable'}</dd></div>)}</dl></> : null}
  </details>;
}

export function ExecutionEvidence({ execution, project, showStatus = true }: { execution: TrainingExecution; project?: string; showStatus?: boolean }) {
  const telemetry = execution.telemetry;
  return <section className="experiment-execution-resources" aria-label="Batch resource usage and worker details">
    {showStatus ? <ExecutionStatus execution={execution} /> : null}
    <ResourceCards project={project} execution={execution} />
    {execution.resourcePlan || telemetry?.latest ? <details className="experiment-worker-details"><summary>Resource scheduling and per-run memory</summary>
      {execution.resourcePlan ? <p>At launch: up to <strong>{execution.resourcePlan.effectiveConcurrency}</strong> concurrent runs from {execution.resourcePlan.requestedConcurrency} requested. {execution.resourcePlan.note}</p> : null}
      {telemetry?.latest ? <>
      <p className="muted">Process-tree RAM can count shared pages more than once; short peaks can occur between samples.</p>
      {telemetry.latest.runs.length ? <div className="development-table"><table><thead><tr><th>Run</th><th>{trainingActive(execution) ? 'Latest' : 'Last recorded'} process-tree RAM</th><th>Observed peak RAM</th></tr></thead><tbody>{telemetry.latest.runs.map((run) => <tr key={run.runId}><td><code>{run.runId}</code></td><td>{gib(run.rssGb)}</td><td>{gib(telemetry.peak.runRssGb[run.runId])}</td></tr>)}</tbody></table></div> : <p className="muted">No active run measurements. Saved per-run peaks are available under Runs.</p>}
      </> : null}
    </details> : null}
    <details className="experiment-worker-details"><summary>Worker and saved artifacts</summary><dl className="development-paths">{managedByTaskCenter(execution) ? <><dt>Execution</dt><dd>Managed by the Task Center</dd></> : null}<dt>Worker log</dt><dd><code>{execution.logPath}</code></dd><dt>Output directory</dt><dd><code>{execution.outputPath}</code></dd>{execution.computePath ? <><dt>Pinned training code</dt><dd><code>{execution.computePath}</code></dd></> : null}{execution.provenancePath ? <><dt>Attempt and driver history</dt><dd><code>{execution.provenancePath}</code></dd></> : null}{telemetry?.path ? <><dt>Resource history</dt><dd><code>{telemetry.path}</code></dd></> : null}<dt>Updated</dt><dd>{execution.updatedAt}</dd></dl></details>
    <DeviceRuntime execution={execution} />
  </section>;
}

export type ResultScoringUnit = 'selected' | 'patient' | 'slide';
export function candidateDisplayMetrics(result: CandidateResult, unit: ResultScoringUnit) {
  if (!result.complete) return null;
  return unit === 'selected' ? result.metrics : result.metricDetails?.[unit] ?? null;
}

export function OOFPredictionDownloads({ project, batchId, candidate, units = ['patient', 'slide'] }: { project: string; batchId: string; candidate: CandidateResult; units?: ('slide' | 'patient')[] }) {
  const [pending, setPending] = useState<'slide' | 'patient' | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const inFlight = useRef(false);
  if (!candidate.complete) return null;
  async function download(unit: 'slide' | 'patient') {
    if (inFlight.current) return;
    inFlight.current = true; setPending(unit); setError(null);
    try { await development.downloadOOF(project, batchId, candidate.candidateId, candidate.trainingSeed, candidate.splitSeed, unit); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Could not download OOF predictions.')); }
    finally { inFlight.current = false; setPending(null); }
  }
  return <div><ErrorNotice error={error} /><div className="inline-actions">{units.map((unit) => <button key={unit} type="button" className="btn btn-secondary btn-small" disabled={pending !== null} onClick={() => void download(unit)}>{pending === unit ? 'Downloading…' : `Download ${unit} OOF predictions`}</button>)}</div></div>;
}

export function ResultsTable({ project, batch, results, loading }: { project?: string; batch: FrozenBatch; results?: DevelopmentResults; loading: boolean }) {
  const [unit, setUnit] = useState<ResultScoringUnit>('selected');
  const numbers = new Map(batch.manifest.configurations.map((item) => [item.id, item.number]));
  const complete = results?.candidates.filter((result) => result.complete).length ?? 0;
  const incomplete = (results?.candidates.length ?? 0) - complete;
  const recipes = new Map(batch.manifest.configurations.map((item) => [item.id, item.recipe]));
  return <>
    <p>Checkpoints follow each configuration’s frozen evaluation policy. Assessment predictions use held-out folds. Out-of-fold (OOF) scores combine all completed folds for one configuration and seed pair.</p>
    {results?.selection ? <p>{results.selection.ready ? `Selected configuration ${numbers.get(results.selection.selectedCandidateId!) ?? results.selection.selectedCandidateId}` : 'Configuration selection awaits all validation results'} · {results.selection.unit} {results.selection.metric.replaceAll('_', ' ')} · mean across folds and seeds.</p> : null}
    {results?.findings ? <Findings findings={results.findings} /> : null}
    {loading ? <p role="status">Loading experiment results…</p> : !results?.candidates.length ? <p>No complete configuration results are available. Failed or cancelled runs do not produce successful results.</p> : <>
      <div className="experiment-result-summary"><p><strong>{complete}</strong> complete configuration / seed groups</p>{incomplete ? <p><strong>{incomplete}</strong> incomplete groups · scores unavailable</p> : null}</div>
      <label className="label">OOF metrics by prediction unit<select className="field" value={unit} onChange={(event) => setUnit(event.target.value as ResultScoringUnit)}><option value="selected">Primary target unit</option><option value="patient">Patient</option><option value="slide">Slide</option></select><small>This changes the displayed assessment metrics. Checkpoint and configuration selection retain their frozen validation metric and unit.</small></label>
      <div className="development-table"><table><thead><tr><th>Configuration</th><th>Learning rate</th><th>Weight decay</th><th>Train / split seed</th><th>Coverage</th><th>Scoring unit</th><th>Validation selection score</th><th>OOF AUROC (95% CI)</th><th>OOF AUPRC (95% CI)</th><th>OOF accuracy</th></tr></thead><tbody>{results.candidates.map((result) => {
        const recipe = recipes.get(result.candidateId);
        const metrics = candidateDisplayMetrics(result, unit);
        const measured = metrics && metrics.available !== false;
        return <tr key={`${result.candidateId}-${result.trainingSeed}-${result.splitSeed}`}><th scope="row">{numbers.get(result.candidateId) ?? result.candidateId}{result.selected ? ' · Selected' : ''}</th><td>{recipe?.learningRate ?? '—'}</td><td>{recipe?.weightDecay ?? '—'}</td><td>{result.trainingSeed} / {result.splitSeed}</td><td>{result.complete ? 'Complete' : 'Incomplete'} · {result.completedRuns} / {result.totalRuns} runs{!result.complete ? <small> · Waiting for all folds</small> : null}</td><td>{unit === 'selected' ? result.metricDetails?.unit ?? 'See details' : unit}</td><td>{metricValue(result.selectionScore)}</td>{(['auroc', 'auprc'] as const).map((metric) => <td key={metric}>{measured ? formatStatistic(metrics[metric], metrics.confidenceIntervals?.[metric]) : '—'}</td>)}<td>{measured ? metricValue(metrics.accuracy) : '—'}</td></tr>;
      })}</tbody></table></div>
      <p className="muted">— means unavailable. Incomplete groups are excluded from score comparison. Primary metrics use each configuration's frozen scoring unit. Patient-level intervals require verified patient IDs and resample patients. The results export includes both scoring units and their available evidence.</p>
      <div className="inline-actions"><button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${batch.manifest.spec.batchName}-results.json`, results)}>Export results</button></div>
      <details><summary>Scoring details and prediction files</summary>{results.selectionNote ? <p className="muted">{results.selectionNote}</p> : null}{results.candidates.filter((result) => result.complete).map((result) => <div key={`${result.candidateId}-${result.trainingSeed}-${result.splitSeed}`}><h4>Configuration {numbers.get(result.candidateId) ?? result.candidateId} · Train seed {result.trainingSeed} · Split seed {result.splitSeed}</h4><p>{metricsText(result.metrics)}</p><MetricEvidence details={result.metricDetails} />{project ? <OOFPredictionDownloads project={project} batchId={batch.id} candidate={result} /> : null}{result.oofPath ? <p>OOF predictions: <code>{result.oofPath}</code></p> : null}</div>)}</details>
    </>}
  </>;
}
