import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { development, trainingActive, latestExecution } from '../api/development';
import type { DevelopmentResults, FrozenBatch, TrainingExecution, TrainingRuntime } from '../api/development';
import { Badge, ErrorNotice } from './ui';
import { Findings } from './ScientificUI';
import { downloadJSON } from '../lib/download';
import { gib, metricValue, metricsText, MetricEvidence, ResourceCards, RunTable } from './ExperimentTracking';
export { RunTable } from './ExperimentTracking';
export type ExperimentExecutionStage = 'planning' | 'running' | 'finished';

export function executionActions(execution?: TrainingExecution | null) {
  return {
    launch: !execution,
    cancel: trainingActive(execution) && !execution?.cancelRequested,
    resume: Boolean(execution && ['failed', 'cancelled', 'interrupted'].includes(execution.status)
      && execution.runCounts.completed < execution.runCounts.total),
  };
}

export default function DevelopmentExecution({ project, batch, implemented, knownExecution, view, allowChanges = true, readOnly = false, stage }: {
  project: string; batch: FrozenBatch; implemented: boolean; knownExecution?: TrainingExecution;
  view: 'batches' | 'runs' | 'results'; allowChanges?: boolean; readOnly?: boolean; stage?: ExperimentExecutionStage;
}) {
  const client = useQueryClient();
  const canChange = allowChanges && !readOnly && stage !== 'finished' && stage !== 'planning';
  const trackingEnabled = implemented && stage !== 'planning';
  const resultsEnabled = trackingEnabled && view === 'results' && stage !== 'running';
  const [error, setError] = useState<Error | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const operationIds = useRef<Partial<Record<'launch' | 'cancel' | 'resume', string>>>({});
  const submitting = useRef(false);
  const runtime = useQuery({ queryKey: ['training-runtime', project], queryFn: () => development.runtime(project), enabled: trackingEnabled && canChange, staleTime: 30000 });
  const executionQuery = useQuery({
    queryKey: ['training-execution', project, batch.id], queryFn: () => development.execution(project, batch.id), enabled: trackingEnabled,
    refetchInterval: (query) => trainingActive(latestExecution(query.state.data, knownExecution)) ? 3000 : false,
  });
  const execution = latestExecution(executionQuery.data, knownExecution);
  const results = useQuery({
    queryKey: ['development-results', project, batch.id], queryFn: () => development.results(project, batch.id),
    enabled: resultsEnabled,
    refetchInterval: trainingActive(execution) ? 3000 : false,
  });
  const status = execution?.status;
  useEffect(() => {
    if (status && status !== 'running' && status !== 'queued') {
      void client.invalidateQueries({ queryKey: ['development-results', project, batch.id] });
    }
  }, [status, client, project, batch.id]);
  async function act(action: 'launch' | 'cancel' | 'resume') {
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
  if (stage === 'planning') return <p className="muted">Inputs and batches remain editable until the experiment is submitted. Runs and results are locked during planning.</p>;
  return <div className="development-execution">
    <ErrorNotice error={error ?? executionQuery.error ?? (canChange ? runtime.error : null) ?? (resultsEnabled ? results.error : null)} />
    {implemented ? <>
      {view !== 'results' ? <>
        <TrainingControls execution={execution} runtime={runtime.data} checking={executionQuery.isPending || executionQuery.isError} pending={pending} onAction={(action) => void act(action)} allowLaunch={stage === undefined} readOnly={!canChange} />
        {executionQuery.isError ? <p className="callout callout-warning" role="status">Tracking could not refresh. Any run status and measurements shown are the last known values.</p> : null}
        {executionQuery.isError || (canChange && runtime.isError) ? <button type="button" className="btn btn-secondary btn-small" onClick={() => { void executionQuery.refetch(); if (canChange) void runtime.refetch(); }}>Retry training status</button> : null}
        {execution ? <ExecutionStatus execution={execution} /> : <p className="muted">{stage ? 'This submitted batch has not been queued yet.' : 'This batch is frozen and has not been launched.'}</p>}
      </> : null}
    </> : <p className="muted">Training execution is unavailable from this service. Frozen plans remain available for review and export.</p>}
    {view === 'runs' ? <RunTable project={trackingEnabled ? project : undefined} batch={batch} execution={execution} /> : null}
    {implemented && view !== 'results' ? execution ? <ExecutionEvidence project={trackingEnabled ? project : undefined} execution={execution} runtime={canChange ? runtime.data : undefined} showStatus={false} /> : canChange && runtime.data ? <DeviceRuntime runtime={runtime.data} /> : null : null}
    {view === 'results' ? stage === 'running' ? <p className="muted">Results unlock when the experiment finishes.</p> : <>
      {resultsEnabled && results.isError ? <><p className="callout callout-warning" role="status">Results could not refresh.{results.data ? ' Showing the last successfully loaded results.' : ''}</p><button type="button" className="btn btn-secondary btn-small" onClick={() => void results.refetch()}>Retry results</button></> : null}
      {!results.isError || results.data ? <ResultsTable batch={batch} results={results.data} loading={resultsEnabled && results.isPending} /> : null}
    </> : null}
  </div>;
}

export function TrainingControls({ execution, runtime, checking, pending, onAction, allowLaunch = true, readOnly = false }: {
  execution?: TrainingExecution | null; runtime?: TrainingRuntime; checking: boolean; pending: string | null;
  onAction: (action: 'launch' | 'cancel' | 'resume') => void; allowLaunch?: boolean; readOnly?: boolean;
}) {
  const actions = executionActions(execution);
  const allowed = { launch: actions.launch && allowLaunch && !readOnly, cancel: actions.cancel && !readOnly, resume: actions.resume && !readOnly };
  return <div className="development-training-controls">
    <div className="inline-actions">
      {allowed.launch ? <button type="button" className="btn btn-primary" disabled={Boolean(pending) || checking || !runtime?.available} onClick={() => onAction('launch')}>{pending === 'launch' ? 'Launching…' : 'Launch batch'}</button> : null}
      {trainingActive(execution) && !readOnly ? <button type="button" className="btn btn-secondary" disabled={Boolean(pending) || !allowed.cancel} onClick={() => onAction('cancel')}>{execution?.cancelRequested ? 'Cancellation requested' : pending === 'cancel' ? 'Requesting cancellation…' : 'Cancel batch'}</button> : null}
      {allowed.resume ? <button type="button" className="btn btn-primary" disabled={Boolean(pending) || checking || !runtime?.available} onClick={() => onAction('resume')}>{pending === 'resume' ? 'Resuming…' : 'Resume unfinished runs'}</button> : null}
      <Badge>{execution?.status ?? 'Not launched'}</Badge>
    </div>
    {allowed.launch || allowed.resume ? !runtime ? <p className="muted">Checking the training runtime…</p> : !runtime.available ? <><p className="callout">Training cannot launch until the runtime is available.</p><Findings findings={runtime.findings} /></> : null : null}
    {allowed.resume ? <p className="muted">Completed runs are retained. Unfinished runs resume their latest checkpoint when available; otherwise they restart.</p> : null}
    {execution?.cancelRequested && trainingActive(execution) ? <p role="status">Cancellation is being applied. Status will update when the worker and active runs stop.</p> : null}
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

export function ExecutionEvidence({ execution, project, runtime, showStatus = true }: { execution: TrainingExecution; project?: string; runtime?: TrainingRuntime; showStatus?: boolean }) {
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
    <details className="experiment-worker-details"><summary>Worker and saved artifacts</summary><dl className="development-paths"><dt>Session</dt><dd><code>{execution.sessionName}</code></dd><dt>Worker log</dt><dd><code>{execution.logPath}</code></dd><dt>Output directory</dt><dd><code>{execution.outputPath}</code></dd>{execution.computePath ? <><dt>Pinned training code</dt><dd><code>{execution.computePath}</code></dd></> : null}{execution.provenancePath ? <><dt>Attempt and driver history</dt><dd><code>{execution.provenancePath}</code></dd></> : null}{telemetry?.path ? <><dt>Resource history</dt><dd><code>{telemetry.path}</code></dd></> : null}<dt>Updated</dt><dd>{execution.updatedAt}</dd></dl></details>
    <DeviceRuntime execution={execution} runtime={runtime} />
  </section>;
}

export function ResultsTable({ batch, results, loading }: { batch: FrozenBatch; results?: DevelopmentResults; loading: boolean }) {
  const numbers = new Map(batch.manifest.configurations.map((item) => [item.id, item.number]));
  const complete = results?.candidates.filter((result) => result.complete).length ?? 0;
  const incomplete = (results?.candidates.length ?? 0) - complete;
  const recipes = new Map(batch.manifest.configurations.map((item) => [item.id, item.recipe]));
  return <>
    <p>Validation selects each run’s checkpoint. Assessment predictions use its held-out fold. OOF scores combine all completed folds for one configuration and seed pair.</p>
    {results?.findings ? <Findings findings={results.findings} /> : null}
    {loading ? <p role="status">Loading experiment results…</p> : !results?.candidates.length ? <p>No complete configuration results are available. Failed or cancelled runs do not produce successful results.</p> : <>
      <div className="experiment-result-summary"><p><strong>{complete}</strong> complete configuration / seed groups</p>{incomplete ? <p><strong>{incomplete}</strong> incomplete groups · scores unavailable</p> : null}</div>
      <div className="development-table"><table><thead><tr><th>Configuration</th><th>Learning rate</th><th>Weight decay</th><th>Train / split seed</th><th>Coverage</th><th>Scoring unit</th><th>OOF AUROC</th><th>OOF accuracy</th></tr></thead><tbody>{results.candidates.map((result) => {
        const recipe = recipes.get(result.candidateId);
        const measured = result.complete && result.metrics?.available !== false;
        return <tr key={`${result.candidateId}-${result.trainingSeed}-${result.splitSeed}`}><th scope="row">{numbers.get(result.candidateId) ?? result.candidateId}</th><td>{recipe?.learningRate ?? '—'}</td><td>{recipe?.weightDecay ?? '—'}</td><td>{result.trainingSeed} / {result.splitSeed}</td><td>{result.complete ? 'Complete' : 'Incomplete'} · {result.completedRuns} / {result.totalRuns} runs{!result.complete ? <small> · Waiting for all folds</small> : null}</td><td>{result.metricDetails?.unit ?? 'See details'}</td><td>{measured ? metricValue(result.metrics?.auroc) : '—'}</td><td>{measured ? metricValue(result.metrics?.accuracy) : '—'}</td></tr>;
      })}</tbody></table></div>
      <p className="muted">— means unavailable. Incomplete groups are excluded from score comparison.</p>
      <div className="inline-actions"><button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${batch.manifest.spec.batchName}-results.json`, results)}>Export results</button></div>
      <details><summary>Scoring details and prediction files</summary>{results.selectionNote ? <p className="muted">{results.selectionNote}</p> : null}{results.candidates.filter((result) => result.complete).map((result) => <div key={`${result.candidateId}-${result.trainingSeed}-${result.splitSeed}`}><h4>Configuration {numbers.get(result.candidateId) ?? result.candidateId} · Train seed {result.trainingSeed} · Split seed {result.splitSeed}</h4><p>{metricsText(result.metrics)}</p><MetricEvidence details={result.metricDetails} />{result.oofPath ? <p>OOF predictions: <code>{result.oofPath}</code></p> : null}</div>)}</details>
    </>}
  </>;
}
