import { Fragment, useEffect, useId, useRef, useState, type MouseEvent, type ReactNode, type RefObject } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import type { Workspace } from '../api/types';
import { experimentPollInterval, experiments } from '../api/experiments';
import {
  exitReasonLabel, formatDuration, measuredVramGb, ownerCountsText, ownerKindLabel, readTaskCenterRoute, runnerActionOutcome, secondsBetween, taskCenter,
  taskCenterHref, taskCenterKeys, taskCenterLink, taskCenterWork, taskCenterPollInterval, taskCounts, taskDisplayTitle, taskKindLabel, taskLive, taskProgress, taskProgressDetails,
  taskStateLabel, taskStateTone,
  type CapacityResponse, type CapacitySuggestion, type RunnerActionResponse, type TaskCenterRoute, type TaskCenterRunner, type TaskCenterSummary, type TaskDetail,
  type TaskEvent, type TaskFailure, type TaskHistoryGroup, type TaskItem, type TaskOwner, type TaskOwnerRef,
} from '../api/taskCenter';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import { Findings } from '../components/ScientificUI';
import TaskOwnerActions from '../components/TaskOwnerActions';
import { ConfirmAction, TaskCenterActionNotice, useTaskCenterActions, type TaskCenterActions } from '../lib/taskCenterActions';
import { plannedConfigurationCount } from '../lib/experimentPredictors';
import { preparationLink } from '../lib/preparationRoute';
import './TaskCenter.css';

type Actions = Pick<TaskCenterActions, 'run' | 'pending'> & Partial<Pick<TaskCenterActions, 'isPending'>>;
type Navigate = (next: TaskCenterRoute) => void;
const gib = (value?: number | null) => value == null || !Number.isFinite(value) ? 'Unavailable' : `${value.toLocaleString(undefined, { maximumFractionDigits: value < 10 ? 1 : 0 })} GiB`;
const clock = (value?: string | null) => {
  const date = value ? new Date(value) : null;
  return date && !Number.isNaN(date.valueOf()) ? date.toLocaleString() : '—';
};
const noNavigation: Navigate = () => undefined;

export function taskCenterErrorMessage(error: Error) {
  if (error instanceof ApiError && error.status === 404) return 'The running HistoPilot service does not include the Task Center yet. Restart HistoPilot, then refresh this page.';
  if (error instanceof ApiError && error.code === 'TASK_CENTER_UNAVAILABLE') return `The task store is unavailable. ${error.message}`;
  return `Task Center status could not be refreshed. ${error.message}`;
}

/**
 * How a route change enters the browser history: opening a task pushes an entry, so Back
 * closes the drawer; filter changes replace the current entry.
 */
export function routeHistoryMode(current: TaskCenterRoute, next: TaskCenterRoute): 'push' | 'replace' {
  return next.task && next.task !== current.task ? 'push' : 'replace';
}
/** How many task entries this page pushed on top of the view the drawer opened from. */
export function drawerDepth(state: unknown) {
  const depth = (state as { taskCenterDrawer?: unknown } | null)?.taskCenterDrawer;
  return typeof depth === 'number' && Number.isInteger(depth) && depth > 0 ? depth : 0;
}
const withDepth = (state: unknown, depth: number) => {
  const rest = { ...(state && typeof state === 'object' ? state as Record<string, unknown> : {}) };
  delete rest.taskCenterDrawer;
  return depth ? { ...rest, taskCenterDrawer: depth } : rest;
};
const sameFilters = (a: TaskCenterRoute, b: TaskCenterRoute) => a.owner === b.owner && a.project === b.project && a.kind === b.kind && a.state === b.state;
const focusable = (element: Element | null): element is HTMLElement => element instanceof HTMLElement && element !== document.body;

/**
 * The page's filters and open task live in the hash (`#task-center?owner=…&task=…`), so a
 * refresh or a link from a stage page lands on the same view. Changes go through the History
 * API without a hash change, which would scroll the workspace back to the top: opening a task
 * pushes an entry (Back closes it), filters replace. Closing a drawer this page opened steps
 * back over its entries; focus returns to the link that opened it.
 */
export function useTaskCenterRoute(): [TaskCenterRoute, Navigate] {
  const [route, setRoute] = useState(() => readTaskCenterRoute());
  const current = useRef(route);
  current.current = route;
  const opener = useRef<HTMLElement | null>(null);
  useEffect(() => {
    const update = () => {
      if (window.location.hash.replace(/^#/, '').split('?')[0] !== 'task-center') return;
      const next = readTaskCenterRoute();
      if (next.task && !current.current.task && focusable(document.activeElement)) opener.current = document.activeElement;
      setRoute(next);
    };
    window.addEventListener('hashchange', update);
    window.addEventListener('popstate', update);
    return () => { window.removeEventListener('hashchange', update); window.removeEventListener('popstate', update); };
  }, []);
  useEffect(() => {
    if (route.task || !opener.current) return;
    const element = opener.current;
    opener.current = null;
    if (element.isConnected) element.focus({ preventScroll: true });
  }, [route.task]);
  const navigate: Navigate = (next) => {
    const before = current.current;
    const depth = drawerDepth(window.history.state);
    if (before.task && !next.task && depth && sameFilters(before, next)) {
      // Back to the view the drawer opened from; popstate then updates the route.
      window.history.go(-depth);
      return;
    }
    const url = new URL(window.location.href);
    url.hash = taskCenterHref(next);
    if (routeHistoryMode(before, next) === 'push') {
      if (!before.task && focusable(document.activeElement)) opener.current = document.activeElement;
      // A drawer reached by a link (no entry of ours below it) closes in place instead.
      window.history.pushState(withDepth(window.history.state, before.task ? depth && depth + 1 : 1), '', url);
    } else {
      window.history.replaceState(withDepth(window.history.state, next.task ? depth : 0), '', url);
    }
    current.current = next;
    setRoute(next);
  };
  return [route, navigate];
}

/** A link inside the page: a real href (copy, open in a new tab) that navigates in place on a plain click. */
function RouteLink({ route, navigate, children, className, label }: { route: TaskCenterRoute; navigate: Navigate; children: ReactNode; className?: string; label?: string }) {
  const onClick = (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); navigate(route);
  };
  return <a className={className} href={taskCenterHref(route)} aria-label={label} onClick={onClick}>{children}</a>;
}

/** Re-render every `interval` ms while `active`, so elapsed times keep moving between polls. */
function useNow(active: boolean, interval = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), interval);
    return () => window.clearInterval(timer);
  }, [active, interval]);
  return now;
}

/** Whether an element is on screen; polling of below-the-fold panels waits until it is. */
function useInView<T extends Element>() {
  const ref = useRef<T>(null);
  const [visible, setVisible] = useState(typeof IntersectionObserver === 'undefined');
  useEffect(() => {
    const element = ref.current;
    if (!element || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver((entries) => setVisible(entries.some((entry) => entry.isIntersecting)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, visible] as const;
}

function Meter({ label, value, max, text, note }: { label: string; value: number | null | undefined; max: number | null | undefined; text: string; note?: string }) {
  const measured = typeof value === 'number' && typeof max === 'number' && Number.isFinite(value) && max > 0;
  return <div className="tc-meter">
    <div className="tc-meter-label"><span>{label}</span><strong>{text}</strong></div>
    {measured ? <progress max={max} value={Math.min(Math.max(value, 0), max)} aria-label={label} /> : <div className="tc-meter-unavailable" aria-hidden="true" />}
    {note ? <small>{note}</small> : null}
  </div>;
}

/**
 * A start that did not happen says why; a pending restart says it follows the old
 * runner's current step. Each note stays until the runner it describes runs.
 */
export function runnerOutcomeNote(response: RunnerActionResponse | null, runner: TaskCenterRunner) {
  const outcome = runnerActionOutcome(response);
  if (!outcome) return null;
  if (outcome.pending) return runner.alive && runner.codeCurrent ? null : outcome.message;
  return runner.alive ? null : outcome.message;
}

export function RunnerStrip({ summary, actions }: { summary: TaskCenterSummary; actions: Actions }) {
  const { runner } = summary;
  const counts = taskCounts(summary.counts);
  const waiting = counts.queued + counts.blocked;
  const outdated = runner.alive && !runner.codeCurrent;
  const pending = (kind: string) => actions.isPending ? actions.isPending(`runner:${kind}`) : actions.pending === `runner:${kind}`;
  const busy = pending('start') || pending('restart');
  const [response, setResponse] = useState<RunnerActionResponse | null>(null);
  const runnerAction = async (kind: 'start' | 'restart') => {
    setResponse(null);
    const result = await actions.run(`runner:${kind}`, (operationId) => kind === 'start' ? taskCenter.startRunner(operationId) : taskCenter.restartRunner(operationId));
    if (result) setResponse(result);
  };
  const outcome = runnerOutcomeNote(response, runner);
  const silent = secondsBetween(runner.heartbeatAt, summary.updatedAt);
  // With automatic start off the service will not launch a runner (a restart
  // would only stop it), so point to the terminal instead of offering a button.
  const manual = !runner.autostart;
  const note = !runner.alive ? `${waiting ? 'Queued tasks start once the runner is running. ' : ''}${manual ? 'Automatic start is off for this service; start it in a terminal with “histopilot runner start”.' : waiting ? '' : 'It starts with HistoPilot, or start it here.'}`.trim()
    : outdated && runner.otherCheckout ? `Started from another checkout (${runner.otherCheckout}). ${manual ? 'Restart it from a terminal in this checkout' : 'Restart it to run this checkout’s code'}; running tasks continue.`
    : outdated ? manual ? 'Running older task code. Restart it from a terminal to use the current version; running tasks continue.' : 'Running older task code. Restart to use the current version; running tasks continue.'
      : silent !== null && silent > 60 ? `No heartbeat for ${formatDuration(silent)}.` : null;
  const failed = summary.recentFailures ?? 0;
  return <section className="tc-strip" aria-label="Runner and queue state">
    <div>
      <strong>Runner</strong>
      <Badge tone={runner.alive ? 'green' : 'amber'}>{runner.alive ? 'Running' : 'Stopped'}</Badge>
      {note ? <span>{note}</span> : null}
      {runner.message ? <span>{runner.message}</span> : null}
      {outcome ? <span role="status">{outcome}</span> : null}
      {manual ? null : !runner.alive ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => void runnerAction('start')}>{pending('start') ? 'Starting…' : 'Start runner'}</button>
        : outdated ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => void runnerAction('restart')}>{pending('restart') ? 'Restarting…' : 'Restart runner'}</button> : null}
    </div>
    <div>
      <Badge tone={summary.paused ? 'amber' : 'neutral'}>{summary.paused ? 'Queue paused' : 'Queue active'}</Badge>
      <span>{counts.running} running · {waiting} queued{failed ? ` · ${failed} failed in the last 24 h` : ''}{summary.eta && (counts.running || waiting) ? ` · about ${formatDuration(summary.eta.seconds)} left${summary.eta.basis === 'estimated' ? ' (estimated)' : ''}` : ''}</span>
      <span>Updated <time dateTime={summary.updatedAt}>{clock(summary.updatedAt)}</time></span>
    </div>
  </section>;
}

export function CapacityMeters({ summary }: { summary: TaskCenterSummary }) {
  const { gpus, cpu, ram } = summary.capacity;
  const usableThreads = Math.max(0, cpu.logical - cpu.reserveThreads);
  return <>
    {summary.capacity.gpuProbeError ? <p className="callout callout-warning" role="status">GPU status is unknown: {summary.capacity.gpuProbeError} GPU meters below may be missing, not idle.</p> : null}
    {summary.leaseError ? <p className="callout callout-warning" role="status">Jobs outside the Task Center could not be read: {summary.leaseError} Their load is missing from these meters, not zero.</p> : null}
    <div className="tc-capacity-grid">
      {gpus.map((gpu) => <article key={gpu.index} className="tc-device" aria-label={`GPU ${gpu.index}: ${gpu.name}`}>
        <div className="tc-device-title"><h3>GPU {gpu.index} · {gpu.name}</h3><span>{gpu.utilizationPercent == null ? 'Utilization unavailable' : `${Math.round(gpu.utilizationPercent)}% busy`}</span></div>
        <Meter label="Task slots" value={gpu.usedSlots} max={gpu.slots} text={`${gpu.usedSlots} of ${gpu.slots} in use`} note={gpu.foreignSlots ? `${gpu.foreignSlots} used by jobs outside the Task Center` : undefined} />
        <Meter label="GPU memory" value={gpu.usedMemoryGb} max={gpu.totalMemoryGb} text={gpu.totalMemoryGb == null ? 'Unavailable' : `${gib(gpu.usedMemoryGb)} of ${gib(gpu.totalMemoryGb)} used`} note={`${gib(gpu.committedMemoryGb)} reserved by running tasks`} />
      </article>)}
      {!gpus.length ? <article className="tc-device"><div className="tc-device-title"><h3>GPU</h3></div><p className="muted">No GPU detected. GPU tasks wait until one is available.</p></article> : null}
      <article className="tc-device" aria-label="CPU and RAM">
        <div className="tc-device-title"><h3>CPU &amp; RAM</h3><span>{cpu.physical ? `${cpu.physical} cores · ` : ''}{cpu.logical} threads</span></div>
        <Meter label="CPU threads" value={cpu.committedThreads} max={usableThreads} text={`${cpu.committedThreads} of ${usableThreads} committed`} note={`${cpu.reserveThreads} kept free · ${cpu.usedCpuTasks} of ${cpu.cpuTaskSlots} CPU task slots in use`} />
        <Meter label="RAM" value={ram.totalGb - ram.availableGb} max={ram.totalGb} text={`${gib(ram.totalGb - ram.availableGb)} of ${gib(ram.totalGb)} in use`} note={`${gib(ram.availableGb)} available · ${gib(ram.reserveGb)} kept free`} />
      </article>
    </div>
  </>;
}

const bindingLabels: Record<CapacitySuggestion['binding'], string> = {
  throughput: 'throughput', gpu_memory: 'GPU memory', ram: 'RAM', cpu: 'CPU threads', run_count: 'the number of runs', default: 'hardware defaults',
};
const trendLabels: Record<CapacitySuggestion['throughput']['trend'], string> = {
  rising: 'Throughput was still rising at the highest measured level.', flat: 'Throughput levelled off at the measured levels.',
  falling: 'Throughput fell at higher concurrency.', unknown: 'Not enough measurements to see a throughput trend.',
};
function basisText(suggestion: CapacitySuggestion) {
  if (suggestion.basis === 'measured') {
    const { runs, batches, latest } = suggestion.evidence;
    return `Based on ${runs} measured run${runs === 1 ? '' : 's'} in ${batches} batch${batches === 1 ? '' : 'es'} on this machine${latest ? `, latest ${clock(latest)}` : ''}.`;
  }
  return suggestion.basis === 'estimated' ? 'Estimated from the queued work; no matching runs have been measured on this machine yet.'
    : 'Nothing to size yet, so this is a conservative default for this machine.';
}

export function SuggestionCallout({ suggestion, current, busy, onUse }: { suggestion: CapacitySuggestion; current: number | null; busy: boolean; onUse: () => void }) {
  const value = suggestion.parallelGpuTasks;
  const { limits, perTask } = suggestion;
  const rows = [
    { id: 'gpu_memory', label: 'GPU memory', limit: limits.gpuMemory, basis: perTask.vramGb != null ? `${gib(perTask.vramGb)} per task` : null },
    { id: 'ram', label: 'RAM', limit: limits.ram, basis: perTask.ramGb != null ? `${gib(perTask.ramGb)} per task` : null },
    { id: 'cpu', label: 'CPU', limit: limits.cpu, basis: perTask.cpuThreads != null ? `${perTask.cpuThreads} threads reserved per task` : perTask.cpuCores != null ? `${perTask.cpuCores.toLocaleString(undefined, { maximumFractionDigits: 1 })} cores per task` : null },
    { id: 'throughput', label: 'Throughput', limit: limits.throughput, basis: suggestion.throughput.observed.length ? trendLabels[suggestion.throughput.trend] : 'No measured runs' },
    { id: 'run_count', label: 'Runs waiting', limit: limits.runCount, basis: null },
  ] as const;
  const perGpu = Object.entries(suggestion.perGpu);
  return <div className="tc-suggestion" role="note" aria-label="Suggested parallel GPU tasks">
    <div className="tc-suggestion-head">
      <p><strong>Suggested: {value}</strong> · {suggestion.binding === 'default' ? 'hardware default' : `limited by ${bindingLabels[suggestion.binding]}`}</p>
      {value !== current ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={onUse}>Use suggestion</button> : <span className="muted">Matches the current setting.</span>}
    </div>
    <p className="muted">{basisText(suggestion)}</p>
    <details className="tc-breakdown"><summary>Why {value}?</summary>
      <div className="tc-table-wrap"><table className="tc-table tc-limits"><thead><tr><th scope="col">Limit</th><th scope="col">Tasks per GPU</th><th scope="col">Basis</th></tr></thead><tbody>
        {rows.map((row) => <tr key={row.id}><th scope="row">{row.label}{suggestion.binding === row.id ? <> <Badge>Limiting</Badge></> : null}</th><td>{row.limit ?? 'No limit'}</td><td>{row.basis ?? '—'}</td></tr>)}
      </tbody></table></div>
      {perGpu.length > 1 ? <p>Per GPU: {perGpu.map(([index, count]) => `GPU ${index}: ${count}`).join(' · ')}</p> : null}
      {suggestion.throughput.observed.length ? <div className="tc-table-wrap"><table className="tc-table tc-throughput"><caption>Measured epoch time by concurrency</caption><thead><tr><th scope="col">Tasks running together</th><th scope="col">Seconds per epoch</th><th scope="col">Runs</th></tr></thead><tbody>
        {suggestion.throughput.observed.map((row) => <tr key={row.concurrency}><td>{row.concurrency.toLocaleString(undefined, { maximumFractionDigits: 1 })}</td><td>{row.secondsPerEpoch.toLocaleString(undefined, { maximumFractionDigits: 1 })}</td><td>{row.runs}</td></tr>)}
      </tbody></table></div> : null}
      {suggestion.throughput.note ? <p className="muted">{suggestion.throughput.note}</p> : null}
      {suggestion.explanation.length ? <ul className="tc-explanation">{suggestion.explanation.map((sentence) => <li key={sentence}>{sentence}</li>)}</ul> : null}
      {suggestion.findings.length ? <Findings findings={suggestion.findings} /> : null}
    </details>
  </div>;
}

export function ParallelTasks({ capacity, actions }: { capacity: CapacityResponse; actions: Actions }) {
  const detected = Object.values(capacity.effective.gpuSlots);
  const uniform = detected.length && detected.every((count) => count === detected[0]) ? detected[0] : null;
  // GPUs with different limits have no single current value: any valid entry may be applied to unify them.
  const mixed = detected.length > 0 && uniform === null;
  const current = uniform ?? capacity.settings.defaultGpuSlots;
  const hint = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const text = draft ?? String(current);
  const value = Number(text.trim());
  const valid = /^\d+$/.test(text.trim()) && value >= 1 && value <= 16;
  const applying = actions.isPending ? actions.isPending('["capacity",{"parallelGpuTasks"', true) : Boolean(actions.pending?.includes('parallelGpuTasks'));
  async function apply(next: number) {
    const result = await actions.run(JSON.stringify(['capacity', { parallelGpuTasks: next }]), (operationId) => taskCenter.updateCapacity({ operationId, parallelGpuTasks: next }));
    if (result) setDraft(null);
  }
  return <section className="tc-parallel" aria-label="Parallel GPU tasks">
    <div className="tc-parallel-control">
      <label className="label">Parallel GPU tasks<input className="field" type="number" inputMode="numeric" min={1} max={16} step={1} value={text} disabled={applying} aria-invalid={!valid} aria-describedby={hint} onChange={(event) => setDraft(event.target.value)} /></label>
      <button type="button" className="btn btn-secondary" disabled={applying || !valid || (!mixed && value === current)} onClick={() => void apply(value)}>{applying ? 'Applying…' : 'Apply'}</button>
    </div>
    <p className="muted" id={hint}>{!valid ? 'Enter a whole number from 1 to 16.' : mixed ? 'GPUs currently use different limits; applying sets every GPU.' : 'Tasks that may share each GPU at once. Running tasks are not interrupted by a change.'}</p>
    {capacity.suggestion ? <SuggestionCallout suggestion={capacity.suggestion} current={mixed ? null : current} busy={applying} onUse={() => { setDraft(null); void apply(capacity.suggestion!.parallelGpuTasks); }} /> : null}
  </section>;
}

/** "gej · Experiment" or "Another workspace": where an owner's work comes from. */
export function ownerWhere(owner: Pick<TaskOwnerRef, 'sameWorkspace' | 'projectId' | 'projectName'>, project: string) {
  if (!owner.sameWorkspace) return 'Another workspace';
  if (owner.projectId === project) return null;
  return owner.projectName ? `Project ${owner.projectName}` : 'Another project';
}

export function TaskName({ task, project, navigate = noNavigation }: { task: TaskItem; project: string; navigate?: Navigate }) {
  const where = ownerWhere(task.owner, project);
  const title = taskDisplayTitle(task);
  return <div className="tc-name">
    <RouteLink className="tc-task-link" route={{ ...readTaskCenterRoute(), task: task.id }} navigate={navigate} label={`Details: ${title}`}>{title}</RouteLink>
    <small>{task.owner.title}{where ? ` · ${where}` : ''}</small>
  </div>;
}

function TaskProgressCell({ task }: { task: TaskItem }) {
  const progress = taskProgress(task.progress);
  const details = taskProgressDetails(task.progress).slice(0, 1);
  if (!progress) return <>{task.state === 'running' ? 'Started' : taskStateLabel(task.state)}{details.length ? <small>{details[0]}</small> : null}</>;
  return <div className="tc-progress"><span>{progress.text}</span><progress aria-label={`${task.title}: progress`} value={progress.value} max={progress.max} />{details.length ? <small>{details[0]}</small> : null}</div>;
}
const device = (task: TaskItem) => task.lane === 'gpu' ? task.gpu == null ? 'GPU' : `GPU ${task.gpu}` : 'CPU';
const taskIdentity = (id: string, action: 'cancel' | 'retry') => JSON.stringify(['task', id, action]);
const pendingFor = (actions: Actions, identity: string) => actions.isPending ? actions.isPending(identity) : actions.pending === identity;

function eventDetail(detail: TaskEvent['detail']) {
  if (detail == null) return '';
  if (typeof detail === 'string') return detail;
  return Object.entries(detail).map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`).join(' · ');
}

const retryWords: Record<TaskFailure['retry'], { tone: string; text: string }> = {
  safe: { tone: 'green', text: 'Safe to retry' },
  check: { tone: 'amber', text: 'Check the machine before retrying' },
  'after-fix': { tone: 'orange', text: 'Fix the cause before retrying' },
  unknown: { tone: 'neutral', text: 'Retrying repeats the same work' },
};

/** The plain-language cause of a failure, shown above any traceback. */
export function FailureExplanation({ failure, error }: { failure: TaskFailure; error?: string | null }) {
  const retry = retryWords[failure.retry];
  const traceback = error && error.includes('\n') ? error : null;
  return <section className="tc-failure" aria-label="Why it stopped">
    <div className="tc-failure-head"><strong>{failure.title}</strong><Badge tone={retry.tone}>{retry.text}</Badge></div>
    <p>{failure.cause}</p>
    {failure.advice ? <p className="muted">{failure.advice}</p> : null}
    {failure.detail && failure.detail !== failure.cause ? <p className="tc-failure-detail"><code>{failure.detail}</code></p> : null}
    {traceback ? <details><summary>Full error</summary><pre className="tc-log" aria-label="Full error">{traceback}</pre></details> : null}
  </section>;
}

const labelNames: Record<string, string> = {
  batchName: 'Batch', configurationNumber: 'Configuration', fold: 'Fold', trainingSeed: 'Training seed', splitSeed: 'Split seed',
  model: 'Model', purpose: 'Purpose', computeKind: 'Job', recordKind: 'Record', recordId: 'Record ID', runId: 'Run', batchId: 'Batch ID', experimentId: 'Experiment',
};
function labelRows(labels: Record<string, unknown>) {
  return Object.entries(labels).filter(([key, value]) => labelNames[key] && value !== null && value !== undefined && value !== '')
    .map(([key, value]) => ({ key, name: labelNames[key], value: String(value) }));
}

function LogView({ task }: { task: TaskDetail }) {
  const [full, setFull] = useState<string | null>(null);
  const [state, setState] = useState<'idle' | 'loading' | 'error'>('idle');
  const [error, setError] = useState<Error | null>(null);
  const logId = useId();
  async function openFull() {
    setState('loading'); setError(null);
    try { setFull(await taskCenter.log(task.id)); setState('idle'); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('The log could not be read.')); setState('error'); }
  }
  async function download() {
    setError(null);
    const name = (task.logPath ?? `${task.id}.log`).split('/').at(-1) ?? `${task.id}.log`;
    try { await taskCenter.downloadLog(task.id, name); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('The log could not be downloaded.')); }
  }
  const text = full ?? task.logTail;
  return <section className="tc-log-section" aria-labelledby={logId}>
    <div className="tc-log-head">
      <h3 id={logId}>{full !== null ? 'Full log' : task.logTruncated ? 'Log (last part)' : 'Log'}{task.logSize ? <small> · {(task.logSize / 1024).toLocaleString(undefined, { maximumFractionDigits: 0 })} KiB</small> : null}</h3>
      {task.logPath ? <div className="tc-row-actions">
        {task.logTruncated && full === null ? <button type="button" className="btn btn-secondary btn-small" disabled={state === 'loading'} onClick={() => void openFull()}>{state === 'loading' ? 'Reading…' : 'Show full log'}</button> : null}
        <button type="button" className="btn btn-secondary btn-small" onClick={() => void download()}>Download log</button>
      </div> : null}
    </div>
    <ErrorNotice error={error} />
    <pre className="tc-log" tabIndex={0} aria-label={`Log of ${task.title}`}>{text || 'The log is empty.'}</pre>
  </section>;
}

/**
 * Where Tab goes inside a modal: wraps from the last control to the first (and back with
 * Shift), and pulls focus back in from outside. Null lets the browser move it.
 */
export function trappedFocus<T>(items: readonly T[], active: T | null, backwards: boolean): T | null {
  if (!items.length) return null;
  const index = active === null ? -1 : items.indexOf(active);
  if (index < 0) return backwards ? items[items.length - 1] : items[0];
  if (backwards && index === 0) return items[items.length - 1];
  if (!backwards && index === items.length - 1) return items[0];
  return null;
}
const tabbable = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

/** Escape closes the modal wherever focus is; Tab stays inside it. */
function useModalKeys(dialog: RefObject<HTMLElement | null>, onClose: () => void) {
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      const root = dialog.current;
      if (event.defaultPrevented || !root) return;
      if (event.key === 'Escape') { event.preventDefault(); close.current(); return; }
      if (event.key !== 'Tab') return;
      const items = [...root.querySelectorAll<HTMLElement>(tabbable)].filter((element) => element.tabIndex >= 0 && element.getClientRects().length > 0);
      const active = document.activeElement instanceof HTMLElement && root.contains(document.activeElement) ? document.activeElement : null;
      const target = trappedFocus(items, active, event.shiftKey);
      if (target) { event.preventDefault(); target.focus(); }
    };
    document.addEventListener('keydown', keydown);
    return () => document.removeEventListener('keydown', keydown);
  }, [dialog]);
}

/**
 * Everything about one task: why it stopped (above the traceback), what it ran, what it
 * measured, its attempts, dependencies and log. Addressed by `?task=` so a refresh or a
 * stage link opens it; a modal side drawer on wide screens, full screen on phones.
 */
export function TaskDetails({ id, onClose, actions, project = '', navigate = noNavigation }: { id: string; onClose: () => void; actions?: Actions; project?: string; navigate?: Navigate }) {
  // Concluded tasks awaiting an automatic resume keep being read until they are requeued.
  const detail = useQuery({ queryKey: taskCenterKeys.task(id), queryFn: () => taskCenter.task(id), refetchIntervalInBackground: false, refetchInterval: (query) => taskLive(query.state.data) ? 3000 : query.state.data?.awaitingRequeue ? 5000 : false });
  const task = detail.data;
  const now = useNow(Boolean(task && ['starting', 'running', 'stopping'].includes(task.state)));
  const request = task?.request;
  const progress = taskProgress(task?.progress);
  const progressLines = taskProgressDetails(task?.progress);
  const vram = task ? measuredVramGb(task) : null;
  const resources = task?.resources;
  const labels = task ? labelRows(task.labels) : [];
  const link = task?.owner.sameWorkspace ? taskCenterLink(task.link, project) : null;
  const title = task ? taskDisplayTitle(task) : 'Task details';
  const where = task ? ownerWhere(task.owner, project) : null;
  const cancelIdentity = taskIdentity(id, 'cancel');
  const retryIdentity = taskIdentity(id, 'retry');
  const closeButton = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEffect(() => { closeButton.current?.focus({ preventScroll: true }); }, [id]);
  useModalKeys(dialog, onClose);
  const act = (action: 'cancel' | 'retry') => actions && void actions.run(taskIdentity(id, action), (operationId) => taskCenter.taskAction(id, action, operationId));
  const managed = Boolean(task?.owner.sameWorkspace && actions);
  return <div ref={dialog} className="tc-drawer" role="dialog" aria-modal="true" aria-labelledby={titleId}>
    <div className="tc-drawer-head">
      <div className="tc-drawer-title"><strong id={titleId}>{title}</strong>{task ? <span><Badge tone={taskStateTone(task.state)}>{taskStateLabel(task.state)}</Badge> {taskKindLabel(task.kind, task.labels)}{task.attempt > 1 ? ` · attempt ${task.attempt}` : ''}</span> : null}</div>
      <button ref={closeButton} type="button" className="icon-button" aria-label="Close task details" onClick={onClose}><Icon name="close" /></button>
    </div>
    <div className="tc-drawer-body">
      {detail.isPending ? <p role="status">Loading task details…</p> : null}
      <ErrorNotice error={detail.error ?? null} />
      {task ? <>
        {task.stopRequest === 'cancel' && taskLive(task) ? <p className="callout" role="status">Cancel requested{task.stopRequestedAt ? ` at ${clock(task.stopRequestedAt)}` : ''}. The task saves what it can and stops.</p>
          : task.stopRequest === 'pause' && taskLive(task) ? <p className="callout" role="status">Stopping to hold{task.stopRequestedAt ? ` since ${clock(task.stopRequestedAt)}` : ''}; it returns to the queue.</p> : null}
        {task.failure ? <FailureExplanation failure={task.failure} error={task.exit?.error ?? task.error} /> : null}
        {task.awaitingRequeue ? <p className="callout" role="status">Waiting to be resumed automatically.</p> : null}
        <div className="tc-row-actions tc-drawer-actions">
          {managed && task.actions.retry ? <button type="button" className="btn btn-primary btn-small" disabled={pendingFor(actions!, retryIdentity)} onClick={() => act('retry')}>{pendingFor(actions!, retryIdentity) ? 'Retrying…' : 'Retry'}</button> : null}
          {managed && task.actions.cancel ? <ConfirmAction label="Cancel task" busyLabel="Cancelling…" busy={pendingFor(actions!, cancelIdentity)} question={`Cancel “${title}”? It saves what it can and stops.`} confirmLabel="Cancel task" onConfirm={() => act('cancel')} /> : null}
          {link ? <a className="text-link" href={link}>Open its record →</a> : null}
          <RouteLink className="text-link" route={{ owner: task.owner.key }} navigate={navigate}>All tasks of {task.owner.title} →</RouteLink>
        </div>
        <dl className="tc-facts">
          <div><dt>Owner</dt><dd>{task.owner.title} · {ownerKindLabel(task.owner.kind)}{where ? ` · ${where}` : ''}</dd></div>
          {labels.map((row) => <div key={row.key}><dt>{row.name}</dt><dd>{row.value}</dd></div>)}
          {progress || progressLines.length ? <div><dt>Progress</dt><dd>{progress ? progress.text : ''}{progressLines.map((line) => <small key={line}>{line}</small>)}</dd></div> : null}
          {task.waitingReason ? <div><dt>Waiting</dt><dd>{task.waitingReason}{task.queuePosition ? ` · #${task.queuePosition} in line` : ''}</dd></div> : null}
          <div><dt>Queued · started · finished</dt><dd>{clock(task.queuedAt)} · {clock(task.startedAt)} · {clock(task.finishedAt)}</dd></div>
          {task.startedAt ? <div><dt>{task.finishedAt ? 'Took' : 'Running for'}</dt><dd>{formatDuration(secondsBetween(task.startedAt, task.finishedAt ?? now))}</dd></div> : null}
          <div><dt>Device</dt><dd>{device(task)}{resources?.gpuName ? ` · ${resources.gpuName}` : ''}</dd></div>
          {request ? <div><dt>Requested</dt><dd>{request.cpuThreads} CPU thread{request.cpuThreads === 1 ? '' : 's'}{request.dataWorkers ? ` + ${request.dataWorkers} data workers` : ''} · {gib(request.ramGb)} RAM{request.vramGb ? ` · ${gib(request.vramGb)} GPU memory` : ''}</dd></div> : null}
          {resources || vram != null ? <div><dt>Measured</dt><dd>
            {resources?.peakPrivateRamGb != null ? `Peak RAM ${gib(resources.peakPrivateRamGb)}` : resources?.privateRamGb != null ? `RAM ${gib(resources.privateRamGb)}` : 'RAM not measured'}
            {vram != null ? ` · peak GPU memory ${gib(vram)}` : ''}
            {resources?.meanCpuCores != null ? ` · ${resources.meanCpuCores.toLocaleString(undefined, { maximumFractionDigits: 1 })} CPU cores on average` : resources?.cpuCores != null ? ` · ${resources.cpuCores.toLocaleString(undefined, { maximumFractionDigits: 1 })} CPU cores` : ''}
            {resources?.meanConcurrency != null ? <small>Shared the GPU with {resources.meanConcurrency.toLocaleString(undefined, { maximumFractionDigits: 1 })} tasks on average</small> : null}
          </dd></div> : null}
          {task.exit ? <div><dt>Exit</dt><dd>{exitReasonLabel(task.exit.reason) ?? 'Unknown'}{task.exit.returncode != null ? ` · code ${task.exit.returncode}` : ''}{task.exit.stopReason ? ` · stopped for ${task.exit.stopReason}` : ''}{task.exit.killed ? ' · killed' : ''}{task.exit.lost ? ' · process lost' : ''}</dd></div> : null}
          {task.pid ? <div><dt>Process</dt><dd>PID {task.pid}</dd></div> : null}
        </dl>
        {task.attempts && task.attempts.length > 1 ? <details className="tc-events" open><summary>Attempts ({task.attempts.length})</summary><div className="tc-table-wrap"><table className="tc-table tc-attempts"><thead><tr><th scope="col">Attempt</th><th scope="col">Started</th><th scope="col">Ended</th><th scope="col">Result</th></tr></thead><tbody>
          {task.attempts.map((attempt) => <tr key={attempt.attempt}><th scope="row">{attempt.attempt}</th><td>{clock(attempt.startedAt)}</td><td>{clock(attempt.endedAt)}</td><td>{attempt.state ? taskStateLabel(attempt.state) : '—'}{attempt.exitReason && attempt.exitReason !== 'ok' ? <small>{exitReasonLabel(attempt.exitReason)}</small> : null}</td></tr>)}
        </tbody></table></div></details> : null}
        {task.dependencies?.length || task.dependents?.length ? <details className="tc-events"><summary>Dependencies</summary>
          {task.dependencies?.length ? <><p className="muted">Starts after</p><ul className="tc-deps">{task.dependencies.map((item) => <li key={item.task}><RouteLink route={{ ...readTaskCenterRoute(), task: item.task }} navigate={navigate}>{task.taskTitles?.[item.task] ?? item.title ?? item.task}</RouteLink> · {item.state ? taskStateLabel(item.state) : 'Unknown'}{item.condition === 'terminal' ? ' (whatever its result)' : ''}</li>)}</ul></> : null}
          {task.dependents?.length ? <><p className="muted">Then starts</p><ul className="tc-deps">{task.dependents.map((item) => <li key={item.task}><RouteLink route={{ ...readTaskCenterRoute(), task: item.task }} navigate={navigate}>{item.title ?? item.task}</RouteLink> · {item.state ? taskStateLabel(item.state) : 'Unknown'}</li>)}</ul></> : null}
        </details> : null}
        {task.command ? <details className="tc-events"><summary>Command and files</summary><dl className="tc-facts tc-command">
          <div><dt>Command</dt><dd><code>{task.command.argv.join(' ')}</code></dd></div>
          {task.command.cwd ? <div><dt>Working folder</dt><dd><code>{task.command.cwd}</code></dd></div> : null}
          {Object.keys(task.command.env).length ? <div><dt>Environment</dt><dd><code>{Object.entries(task.command.env).map(([key, value]) => `${key}=${value}`).join(' ')}</code></dd></div> : null}
          {task.command.log ? <div><dt>Log</dt><dd><code>{task.command.log}</code></dd></div> : null}
          {task.command.progress ? <div><dt>Progress file</dt><dd><code>{task.command.progress}</code></dd></div> : null}
          {task.command.result ? <div><dt>Result file</dt><dd><code>{task.command.result}</code></dd></div> : null}
        </dl></details> : task.logPath ? <p className="muted">Log: <code>{task.logPath}</code></p> : null}
        {task.events.length ? <details className="tc-events"><summary>State changes ({task.events.length})</summary><ol>{[...task.events].sort((a, b) => a.seq - b.seq).map((event) => <li key={event.seq}>
          <time dateTime={event.at}>{clock(event.at)}</time><span>{event.fromState ? `${taskStateLabel(event.fromState)} → ` : ''}{event.toState ? taskStateLabel(event.toState) : 'Updated'}{event.attempt ? ` · attempt ${event.attempt}` : ''}</span>{eventDetail(event.detail) ? <small>{eventDetail(event.detail)}</small> : null}
        </li>)}</ol></details> : null}
        <LogView task={task} />
      </> : null}
    </div>
  </div>;
}

const awaitingNote = 'Waiting to be resumed automatically';

function TaskActionButtons({ task, actions, navigate, retry = false }: { task: TaskItem; actions: Actions; navigate: Navigate; retry?: boolean }) {
  const managed = task.owner.sameWorkspace;
  const title = taskDisplayTitle(task);
  const act = (action: 'cancel' | 'retry') => void actions.run(taskIdentity(task.id, action), (operationId) => taskCenter.taskAction(task.id, action, operationId));
  const cancelling = pendingFor(actions, taskIdentity(task.id, 'cancel'));
  const retrying = pendingFor(actions, taskIdentity(task.id, 'retry'));
  return <div className="tc-row-actions">
    {task.stopRequest === 'cancel' && taskLive(task) ? <small className="tc-note">Cancel requested</small>
      : managed && task.actions.cancel && (taskLive(task) || task.awaitingRequeue) ? <ConfirmAction label="Cancel" busyLabel="Cancelling…" busy={cancelling} ariaLabel={`Cancel ${title}`} question={`Cancel “${title}”?`} confirmLabel="Cancel task" onConfirm={() => act('cancel')} /> : null}
    {managed && retry && task.actions.retry ? <button type="button" className="btn btn-secondary btn-small" disabled={retrying} aria-label={`Retry ${title}`} onClick={() => act('retry')}>{retrying ? 'Retrying…' : 'Retry'}</button> : null}
    <RouteLink className="text-button" route={{ ...readTaskCenterRoute(), task: task.id }} navigate={navigate} label={`Details and log: ${title}`}>Details</RouteLink>
  </div>;
}

function MemoryCell({ task }: { task: TaskItem }) {
  const vram = measuredVramGb(task);
  const ram = task.resources?.privateRamGb;
  return <>
    {ram != null ? `${gib(ram)} RAM` : '—'}
    {vram != null ? <small>{gib(vram)} GPU memory (peak)</small> : task.request.vramGb ? <small>{gib(task.request.vramGb)} GPU memory reserved</small> : null}
  </>;
}

export function RunningTasks({ tasks, project, actions, paused, navigate = noNavigation, filtered = false }: { tasks: TaskItem[]; project: string; actions: Actions; paused: boolean; navigate?: Navigate; filtered?: boolean }) {
  const now = useNow(tasks.length > 0);
  if (!tasks.length) return <p className="muted">{filtered ? 'Nothing matching these filters is running.' : paused ? 'Nothing is running. The queue is paused, so no new tasks start.' : 'Nothing is running.'}</p>;
  return <div className="tc-table-wrap"><table className="tc-table tc-running">
    <thead><tr><th scope="col">Task</th><th scope="col">Device</th><th scope="col">Progress</th><th scope="col">Elapsed</th><th scope="col">Memory</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
    <tbody>{tasks.map((task) => <tr key={task.id}>
      <th scope="row"><TaskName task={task} project={project} navigate={navigate} />{task.state !== 'running' ? <Badge tone={taskStateTone(task.state)}>{taskStateLabel(task.state)}</Badge> : null}{task.waitingReason ? <small className="tc-note">{task.waitingReason}</small> : null}</th>
      <td>{device(task)}<small>{taskKindLabel(task.kind, task.labels)}</small></td>
      <td><TaskProgressCell task={task} /></td>
      <td>{formatDuration(secondsBetween(task.startedAt, now))}</td>
      <td><MemoryCell task={task} /></td>
      <td><TaskActionButtons task={task} actions={actions} navigate={navigate} /></td>
    </tr>)}</tbody>
  </table></div>;
}

/** The server names why an owner's first task waits (or that it is held or blocked). */
export function ownerWaitingReason(owner: TaskOwner, pending: TaskItem[] = []) {
  if (owner.waitingReason) return owner.waitingReason;
  const reason = pending.find((task) => task.owner.key === owner.key && task.waitingReason)?.waitingReason;
  if (reason) return reason;
  if (owner.held) return 'Held. Its queued tasks start after you release it.';
  const counts = taskCounts(owner.counts);
  if (!counts.running && !counts.queued && counts.blocked) return 'Waiting for earlier tasks to finish.';
  return null;
}

const ownerPageSize = 50;
const ownerStates = [['', 'All tasks'], ['live', 'Unfinished'], ['failed,interrupted', 'Failed or interrupted'], ['succeeded', 'Completed'], ['cancelled', 'Cancelled']] as const;
/** A select's options plus the value it was given when that is not one of them (for example `state=failed`). */
export function withCurrentOption(options: readonly (readonly [string, string])[], value: string, label = (text: string) => text) {
  return options.some(([option]) => option === value) ? options : [...options, [value, label(value)] as const];
}
const stateFilterLabel = (value: string) => value.split(',').map((state) => taskStateLabel(state)).join(' or ');
function OwnerTasks({ owner, project, actions, navigate, live, stateFilter, onStateChange }: { owner: TaskOwner; project: string; actions: Actions; navigate: Navigate; live: boolean; stateFilter?: string; onStateChange?: (state: string) => void }) {
  const [page, setPage] = useState(0);
  const [localState, setLocalState] = useState(stateFilter ?? '');
  // In the owner view the filter lives in the URL; an expanded queue row keeps its own.
  const state = onStateChange ? stateFilter ?? '' : localState;
  const setState = onStateChange ?? setLocalState;
  const tasks = useQuery({
    queryKey: taskCenterKeys.tasks('owner', owner.key, state, String(page)),
    queryFn: () => taskCenter.tasks({ owner: owner.key, state: state || undefined, limit: ownerPageSize, offset: page * ownerPageSize }),
    refetchInterval: live ? 5000 : false, refetchIntervalInBackground: false,
  });
  const items = tasks.data?.tasks ?? [];
  return <div className="tc-owner-tasks-panel">
    <div className="tc-filters"><label className="label">Tasks<select className="field" value={state} onChange={(event) => { setState(event.target.value); setPage(0); }}>{withCurrentOption(ownerStates, state, stateFilterLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label></div>
    {tasks.isPending ? <p role="status">Loading tasks…</p> : null}
    {tasks.error && !tasks.data ? <ErrorNotice error={tasks.error} /> : null}
    {!tasks.isPending && !items.length && !tasks.error ? <p className="muted">No tasks match.</p> : null}
    {items.length ? <div className="tc-table-wrap"><table className="tc-table tc-owner-tasks">
      <thead><tr><th scope="col">Task</th><th scope="col">State</th><th scope="col">Device</th><th scope="col">Progress</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
      <tbody>{items.map((task) => <tr key={task.id}>
        <th scope="row"><div className="tc-name"><RouteLink className="tc-task-link" route={{ ...readTaskCenterRoute(), task: task.id }} navigate={navigate}>{taskDisplayTitle(task)}</RouteLink><small>{taskKindLabel(task.kind, task.labels)}{task.attempt > 1 ? ` · attempt ${task.attempt}` : ''}</small></div></th>
        <td><Badge tone={taskStateTone(task.state)}>{taskStateLabel(task.state)}</Badge>{task.stopRequest === 'cancel' && taskLive(task) ? <small>Cancel requested</small> : task.waitingReason ? <small>{task.waitingReason}</small> : task.awaitingRequeue ? <small>{awaitingNote}</small> : task.failure ? <small>{task.failure.title}</small> : task.exit?.reason && !taskLive(task) && task.exit.reason !== 'ok' ? <small>{exitReasonLabel(task.exit.reason)}</small> : null}</td>
        <td>{device(task)}</td>
        <td>{taskProgress(task.progress)?.text ?? '—'}</td>
        <td><TaskActionButtons task={task} actions={actions} navigate={navigate} retry /></td>
      </tr>)}</tbody>
    </table></div> : null}
    {page > 0 || tasks.data?.hasMore ? <nav className="tc-pager" aria-label={`Tasks of ${owner.title}`}><button type="button" className="btn btn-secondary btn-small" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><span>Page {page + 1}</span><button type="button" className="btn btn-secondary btn-small" disabled={!tasks.data?.hasMore} onClick={() => setPage(page + 1)}>Next</button></nav> : null}
  </div>;
}

export function QueueTable({ owners, project, actions, navigate = noNavigation, filtered = false }: { owners: TaskOwner[]; pending?: TaskItem[]; project: string; actions: Actions; navigate?: Navigate; filtered?: boolean }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  if (!owners.length) return <p className="muted">{filtered ? 'Nothing matching these filters is queued.' : 'The queue is empty.'}</p>;
  return <div className="tc-table-wrap"><table className="tc-table tc-queue">
    <thead><tr><th scope="col" className="tc-position">#</th><th scope="col">Owner</th><th scope="col">Tasks</th><th scope="col">Status</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
    <tbody>{owners.map((owner) => {
      const link = owner.sameWorkspace ? taskCenterLink(owner.link, project) : null;
      const where = ownerWhere(owner, project);
      const reason = ownerWaitingReason(owner);
      const counts = taskCounts(owner.counts);
      const open = expanded === owner.key;
      return <Fragment key={owner.key}>
        <tr>
          <td className="tc-position">{owner.position ?? '—'}</td>
          <th scope="row"><div className="tc-name"><RouteLink className="tc-task-link" route={{ owner: owner.key }} navigate={navigate}>{owner.title}</RouteLink><small>{ownerKindLabel(owner.kind, owner.purpose)}{where ? ` · ${where}` : ''}{link ? <> · <a href={link}>Open record</a></> : null}</small></div>
            <button type="button" className="text-button tc-expand" aria-expanded={open} aria-label={`${open ? 'Hide tasks' : 'Show tasks'}: ${owner.title}`} onClick={() => setExpanded(open ? null : owner.key)}>{open ? 'Hide tasks' : 'Show tasks'}</button></th>
          <td>{ownerCountsText(owner.counts)}{owner.etaSeconds ? <small>About {formatDuration(owner.etaSeconds)} left</small> : null}</td>
          <td>{owner.held ? <Badge tone="amber">Held</Badge> : counts.running ? <Badge tone="green">Running</Badge> : <Badge>Queued</Badge>}{counts.failed ? <> <Badge tone="orange">{counts.failed} failed</Badge></> : null}{reason ? <small>{reason}</small> : null}</td>
          <td>{owner.sameWorkspace ? <TaskOwnerActions owner={owner} actions={actions} /> : <small>Managed from its own workspace</small>}</td>
        </tr>
        {open ? <tr className="tc-detail-row"><td colSpan={5}><OwnerTasks owner={owner} project={project} actions={actions} navigate={navigate} live /></td></tr> : null}
      </Fragment>;
    })}</tbody>
  </table></div>;
}

export function PlannedSetups({ project }: { project: string }) {
  const summaries = useQuery({ queryKey: ['model-experiments', project, 'summary'], queryFn: () => experiments.summaries(project), refetchInterval: (query) => experimentPollInterval(query.state.data), refetchIntervalInBackground: false });
  const planned = (summaries.data?.items ?? []).filter((item) => item.status === 'ready' && item.state === 'active');
  if (summaries.isPending) return <p role="status">Loading frozen setups…</p>;
  if (summaries.error && !summaries.data) return <ErrorNotice error={summaries.error} />;
  if (!planned.length) return <p className="muted">No frozen setups are waiting to start in this project.</p>;
  return <ul className="tc-planned">{planned.map((item) => {
    const plans = item.batchPlans ?? [];
    const groups = plans.reduce((total, plan) => total + plannedConfigurationCount(plan.spec) * plan.spec.trainingSeeds.length, 0);
    return <li key={item.id}>
      <div className="tc-name"><strong>{item.name}</strong><small>{plans.length} batch{plans.length === 1 ? '' : 'es'}{groups ? ` · ${groups} training group${groups === 1 ? '' : 's'}, each over every fold` : ''}</small></div>
      <a className="text-link" href={preparationLink('experiments', {}, { experiment: item.id })}>Open in Experiments →</a>
    </li>;
  })}</ul>;
}

const historyStates = [['history', 'All finished'], ['succeeded', 'Completed'], ['failed', 'Failed'], ['interrupted', 'Interrupted'], ['cancelled', 'Cancelled']] as const;
const historyKinds = [['', 'All kinds'], ['mil-fold,mil-collect', 'Training'], ['compute-job', 'Refits, evaluations and attention'], ['predictor-coordinator', 'Predictor creation'], ['bulk-submit', 'Bulk submission'], ['extraction,extraction-validation', 'Feature extraction'], ['packing', 'Feature packing'], ['archive', 'Study archives']] as const;
const historyPageSize = 15;
function historyResult(group: TaskHistoryGroup) {
  const { finished } = group;
  return [finished.succeeded ? `${finished.succeeded} completed` : '', finished.failed ? `${finished.failed} failed` : '', finished.interrupted ? `${finished.interrupted} interrupted` : '', finished.cancelled ? `${finished.cancelled} cancelled` : ''].filter(Boolean).join(' · ');
}

/**
 * Finished work grouped by owner, filtered and paged by the server; polls only while on
 * screen. Its filters are the page's (`state`, `project`, `kind` in the URL), so a refresh or
 * a shared link keeps them.
 */
export function TaskHistory({ project, actions, quiet = false, route = {}, navigate = noNavigation }: { project: string; actions: Actions; quiet?: boolean; route?: TaskCenterRoute; navigate?: Navigate }) {
  const state = route.state || 'history';
  const scope = route.project ? 'project' : 'all';
  const kind = route.kind ?? '';
  const [page, setPage] = useState(0);
  const [ref, visible] = useInView<HTMLDivElement>();
  const params = { state, project: route.project, kind: kind || undefined, limit: historyPageSize, offset: page * historyPageSize };
  const history = useQuery({ queryKey: taskCenterKeys.history(params), queryFn: () => taskCenter.history(params), refetchInterval: visible ? 15000 : false, refetchIntervalInBackground: false });
  const groups = history.data?.groups ?? [];
  const total = history.data?.total ?? 0;
  const change = (next: TaskCenterRoute) => { navigate({ ...route, ...next }); setPage(0); };
  return <div ref={ref}>
    <div className="tc-filters">
      <label className="label">Result<select className="field" value={state} onChange={(event) => change({ state: event.target.value === 'history' ? undefined : event.target.value })}>{withCurrentOption(historyStates, state, stateFilterLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label className="label">Projects<select className="field" value={scope} onChange={(event) => change({ project: event.target.value === 'project' ? project : undefined })}><option value="all">All projects</option><option value="project">{route.project && route.project !== project ? 'The linked project' : 'This project'}</option></select></label>
      <label className="label">Kind<select className="field" value={kind} onChange={(event) => change({ kind: event.target.value || undefined })}>{withCurrentOption(historyKinds, kind).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
    </div>
    {history.isPending ? <p role="status">Loading finished tasks…</p> : null}
    <ErrorNotice error={quiet ? null : history.error ?? null} />
    {groups.length ? <div className="tc-table-wrap"><table className="tc-table tc-history">
      <thead><tr><th scope="col">Owner</th><th scope="col">Result</th><th scope="col">Last finished</th><th scope="col">Span</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
      <tbody>{groups.map((group) => {
        const owner = group.owner;
        const where = ownerWhere(owner, project);
        const link = owner.sameWorkspace ? taskCenterLink(owner.link, project) : null;
        const failure = group.lastFailure;
        const retrying = pendingFor(actions, JSON.stringify(['owner', owner.key, 'retry', null]));
        return <tr key={owner.key}>
          <th scope="row"><div className="tc-name"><RouteLink className="tc-task-link" route={{ owner: owner.key }} navigate={navigate}>{owner.title}</RouteLink><small>{ownerKindLabel(owner.kind, owner.purpose)}{where ? ` · ${where}` : ''}{link ? <> · <a href={link}>Open record</a></> : null}</small></div></th>
          <td>{historyResult(group)}{failure ? <small><RouteLink route={{ ...route, task: failure.taskId }} navigate={navigate}>{failure.message ?? 'Failed'}</RouteLink>{failure.retry === 'safe' ? ' · safe to retry' : ''}</small> : null}</td>
          <td>{clock(group.lastFinishedAt)}</td>
          <td>{formatDuration(secondsBetween(group.firstStartedAt, group.lastFinishedAt))}</td>
          <td><div className="tc-row-actions">
            {owner.sameWorkspace && owner.actions.retry ? <button type="button" className="btn btn-secondary btn-small" disabled={retrying} aria-label={`Retry failed tasks of ${owner.title}`} onClick={() => void actions.run(JSON.stringify(['owner', owner.key, 'retry', null]), (operationId) => taskCenter.ownerAction(owner.key, 'retry', operationId))}>{retrying ? 'Retrying…' : 'Retry failed'}</button> : null}
            <RouteLink className="text-button" route={{ owner: owner.key }} navigate={navigate} label={`Tasks of ${owner.title}`}>Tasks</RouteLink>
          </div></td>
        </tr>;
      })}</tbody>
    </table></div> : !history.isPending && !history.error ? <p className="muted">No finished tasks{scope === 'project' ? ' in this project' : ''} match these filters.</p> : null}
    {total > historyPageSize ? <nav className="tc-pager" aria-label="Finished task pages"><button type="button" className="btn btn-secondary btn-small" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><span>Page {page + 1} of {Math.ceil(total / historyPageSize)}</span><button type="button" className="btn btn-secondary btn-small" disabled={(page + 1) * historyPageSize >= total} onClick={() => setPage(page + 1)}>Next</button></nav> : null}
  </div>;
}

/**
 * An owner's state in the stage chips' words (see the rollup): failures need attention until
 * retried, any cancelled task reads as cancelled, and only a clean run reads as completed.
 */
export function ownerState(owner: Pick<TaskOwner, 'held' | 'counts'>): { text: string; tone?: string } {
  const counts = taskCounts(owner.counts);
  if (owner.held) return { text: 'Held', tone: 'amber' };
  if (counts.running) return { text: 'Running', tone: 'green' };
  if (counts.queued || counts.blocked) return { text: 'Queued' };
  if (counts.failed || counts.interrupted) return { text: 'Needs attention', tone: 'orange' };
  if (counts.cancelled) return { text: 'Cancelled' };
  return { text: 'Completed', tone: 'success' };
}
function OwnerStateBadge({ owner }: { owner: Pick<TaskOwner, 'held' | 'counts'> }) {
  const state = ownerState(owner);
  return <Badge tone={state.tone}>{state.text}</Badge>;
}

/** One owner's whole story, for links from stage pages: counts, queue controls and every task. */
function OwnerFocus({ ownerKey, project, actions, navigate, route }: { ownerKey: string; project: string; actions: Actions; navigate: Navigate; route: TaskCenterRoute }) {
  const owner = useQuery({ queryKey: taskCenterKeys.owner(ownerKey), queryFn: () => taskCenter.owner(ownerKey), refetchIntervalInBackground: false, refetchInterval: (query) => { const counts = taskCounts(query.state.data?.counts); return counts.running + counts.queued + counts.blocked ? 3000 : 30000; } });
  const data = owner.data;
  if (owner.isPending) return <p role="status">Loading this owner…</p>;
  if (!data) return <ErrorNotice error={owner.error} />;
  const counts = taskCounts(data.counts);
  const live = counts.running + counts.queued + counts.blocked > 0;
  const where = ownerWhere(data, project);
  const link = data.sameWorkspace ? taskCenterLink(data.link, project) : null;
  const reason = ownerWaitingReason(data);
  return <section className="tc-owner-focus" aria-label={`Tasks of ${data.title}`}>
    <div className="tc-owner-focus-head">
      <div className="tc-name"><strong>{data.title}</strong><small>{ownerKindLabel(data.kind, data.purpose)}{where ? ` · ${where}` : ''}{data.position ? ` · #${data.position} in the queue` : ''}{link ? <> · <a href={link}>Open its record →</a></> : null}</small></div>
      <div><OwnerStateBadge owner={data} /></div>
    </div>
    <p className="tc-owner-counts">{ownerCountsText(data.counts)}{data.etaSeconds ? ` · about ${formatDuration(data.etaSeconds)} left` : ''}</p>
    {reason ? <p className="muted">{reason}</p> : null}
    {data.sameWorkspace ? <TaskOwnerActions owner={data} actions={actions} /> : <p className="muted">Managed from its own workspace.</p>}
    <OwnerTasks key={ownerKey} owner={data} project={project} actions={actions} navigate={navigate} live={live} stateFilter={route.state} onStateChange={(state) => navigate({ ...route, state: state || undefined })} />
  </section>;
}

function FilterBar({ route, navigate, owners, project }: { route: TaskCenterRoute; navigate: Navigate; owners: TaskOwner[]; project: string }) {
  const parts = [
    route.owner ? `Owner: ${owners.find((owner) => owner.key === route.owner)?.title ?? 'selected owner'}` : '',
    route.project ? (route.project === project ? 'This project' : `Project ${route.project}`) : '',
    route.kind ? `Kind: ${taskKindLabel(route.kind)}` : '',
    route.state ? `State: ${route.state}` : '',
  ].filter(Boolean);
  if (!parts.length) return null;
  return <div className="tc-filter-bar" role="note"><span>Showing {parts.join(' · ')}</span><RouteLink className="text-link" route={route.task ? { task: route.task } : {}} navigate={navigate}>Show everything</RouteLink></div>;
}

const matchesRoute = (route: TaskCenterRoute) => (owner: Pick<TaskOwnerRef, 'key' | 'projectId' | 'sameWorkspace'>, kind?: string) =>
  (!route.owner || owner.key === route.owner)
  && (!route.project || (owner.sameWorkspace && owner.projectId === route.project))
  && (!route.kind || !kind || route.kind.split(',').includes(kind));

export default function TaskCenter({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const [route, navigate] = useTaskCenterRoute();
  const live = { refetchIntervalInBackground: false } as const;
  // One read for the summary, running tasks and the live queue.
  const snapshot = useQuery({ queryKey: taskCenterKeys.snapshot, queryFn: taskCenter.snapshot, ...live, refetchInterval: (query) => query.state.error ? 10000 : taskCenterPollInterval(query.state.data?.summary) });
  const capacity = useQuery({ queryKey: taskCenterKeys.capacity, queryFn: taskCenter.capacity, ...live, refetchInterval: 30000 });
  const actions = useTaskCenterActions([['model-experiment', project], ['model-experiments', project], ['development-batches', project], ['training-execution', project]]);
  const client = useQueryClient();
  const [refreshing, setRefreshing] = useState(false);
  const summary = snapshot.data?.summary;
  const paused = summary?.paused ?? capacity.data?.settings.paused;
  const pauseIdentity = (next: boolean) => JSON.stringify(['capacity', { paused: next }]);
  // When the snapshot cannot be read the callout above explains why; the panels
  // then show neither invented empty states nor the same error again.
  const down = Boolean(snapshot.error && !snapshot.data);
  const unsupported = down && snapshot.error instanceof ApiError && snapshot.error.status === 404;
  const panelError = (query: { error: Error | null; data?: unknown }) => !down && query.error && !query.data ? query.error : null;
  const matches = matchesRoute(route);
  const running = (snapshot.data?.running ?? []).filter((task) => matches(task.owner, task.kind));
  const owners = (snapshot.data?.owners ?? []).filter((owner) => matches(owner));
  const filtered = Boolean(route.owner || route.project || route.kind);
  function setPaused(next: boolean) {
    void actions.run(pauseIdentity(next), (operationId) => taskCenter.updateCapacity({ operationId, paused: next }));
  }
  async function refresh() {
    // Polling runs in the background; the button reports only the refresh the user asked for.
    setRefreshing(true);
    try { await Promise.all([client.invalidateQueries({ queryKey: taskCenterKeys.all }), client.invalidateQueries({ queryKey: ['model-experiments', project, 'summary'] })]); }
    finally { setRefreshing(false); }
  }
  const closeTask = () => navigate({ ...route, task: undefined });
  const pausing = actions.isPending(pauseIdentity(true)) || actions.isPending(pauseIdentity(false));
  return <div className={`task-center${route.task ? ' has-drawer' : ''}`}>
    <PageHeader eyebrow="LOCAL WORKSPACE" title="Task Center" description={`${taskCenterWork(true)} from every project on this machine, in the order they will run.`} actions={<>
      <button type="button" className="btn btn-secondary" disabled={paused === undefined || pausing} title={paused ? undefined : 'Running tasks continue; queued tasks wait until you resume.'} onClick={() => setPaused(!paused)}>{paused ? 'Resume queue' : 'Pause queue'}</button>
      <button type="button" className="btn btn-secondary" disabled={refreshing} onClick={() => void refresh()}><Icon name="reset" />{refreshing ? 'Refreshing…' : 'Refresh'}</button>
    </>} />
    <TaskCenterActionNotice actions={actions} />
    {snapshot.error ? <div className="callout callout-warning" role="alert">{taskCenterErrorMessage(snapshot.error)}{snapshot.data ? ' The last known state is shown.' : ''}</div> : null}
    {snapshot.isPending ? <p role="status">Reading the task queue…</p> : null}
    {summary ? <RunnerStrip summary={summary} actions={actions} /> : null}
    <FilterBar route={route} navigate={navigate} owners={snapshot.data?.owners ?? []} project={project} />
    {route.owner && !unsupported ? <Panel title="Owner" subtitle="Every task of this experiment or record, with its queue controls.">
      <OwnerFocus ownerKey={route.owner} project={project} actions={actions} navigate={navigate} route={route} />
    </Panel> : null}
    {unsupported ? null : <>
      <Panel title="Running" subtitle="Cancelling stops the task after it saves what it can. Details shows its progress, measurements and log.">
        {snapshot.isPending ? <p role="status">Loading running tasks…</p> : snapshot.data ? <RunningTasks tasks={running} project={project} actions={actions} paused={Boolean(paused)} navigate={navigate} filtered={filtered} /> : null}
      </Panel>
      {route.owner ? null : <Panel title="Queue" subtitle="Owners run in this order; each experiment or record keeps its tasks together. Held owners keep their place but start nothing.">
        {snapshot.isPending ? <p role="status">Loading the queue…</p> : snapshot.data ? <QueueTable owners={owners} project={project} actions={actions} navigate={navigate} filtered={filtered} /> : null}
      </Panel>}
      <Panel title="Capacity" subtitle="A task starts when a GPU slot, GPU memory, RAM and CPU threads are free. Jobs outside the Task Center count against the same capacity.">
        {summary ? <CapacityMeters summary={summary} /> : null}
        {capacity.data ? <ParallelTasks capacity={capacity.data} actions={actions} /> : capacity.isPending ? <p role="status">Reading capacity settings…</p> : null}
        <ErrorNotice error={panelError(capacity)} />
        {summary?.foreignLeases.length ? <details className="tc-foreign"><summary>Other jobs using this machine ({summary.foreignLeases.length})</summary><ul>{summary.foreignLeases.map((lease, index) => <li key={index}>{lease.kind ? ownerKindLabel(lease.kind) : 'Compute job'}{lease.batchId ? ` · batch ${lease.batchId}` : ''}{lease.runId ? ` · run ${lease.runId}` : ''} · {lease.gpu === null ? 'CPU' : `GPU ${lease.gpu}`} · {lease.cpus} CPU threads · {gib(lease.ramGb)} RAM</li>)}</ul></details> : null}
      </Panel>
    </>}
    {route.owner ? null : <Panel title="Planned in this project" subtitle="Frozen setups that have not started. Start them from Experiments; their tasks then join the end of the queue.">
      {workspace.project.lifecycleState === 'trashed' ? <p className="muted">This project is in Trash.</p> : <PlannedSetups project={project} />}
    </Panel>}
    {unsupported || route.owner ? null : <Panel title="History" subtitle="Finished tasks grouped by the experiment or record they belong to, newest first.">
      <TaskHistory key={`${route.project ?? ''}:${route.kind ?? ''}:${route.state ?? ''}`} project={project} actions={actions} quiet={down} route={route} navigate={navigate} />
    </Panel>}
    {route.task ? <>
      <button type="button" className="tc-drawer-backdrop" aria-label="Close task details" tabIndex={-1} onClick={closeTask} />
      <TaskDetails key={route.task} id={route.task} onClose={closeTask} actions={actions} project={project} navigate={navigate} />
    </> : null}
  </div>;
}
