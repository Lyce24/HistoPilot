import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import {
  formatDuration, rollupIsLive, rollupPollInterval, rollupRetryable, secondsBetween, taskCenter, taskCenterKeys, taskProgress,
  type RollupScope, type TaskRollup,
} from '../api/taskCenter';
import './RunStatusChip.css';

export type RunStatusTone = 'idle' | 'active' | 'waiting' | 'attention' | 'done';
export interface RunStatusSegment { label: string; kinds: string[] }
export interface RunStatusCopy { tone: RunStatusTone; headline: string; parts: string[]; linkText: string | null }

const lowerFirst = (text: string) => text.replace(/^[A-Z](?=[a-z])/, (letter) => letter.toLowerCase());
const clockTime = (value: string | null, now: number) => {
  const at = value ? new Date(value) : null;
  if (!at || Number.isNaN(at.valueOf())) return '';
  const today = new Date(now).toDateString() === at.toDateString();
  return today ? at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : at.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
};
const failedCount = (rollup: TaskRollup) => rollup.progress
  ? (rollup.counts.failed ?? 0) + (rollup.counts.interrupted ?? 0)
  : rollup.recentFailures ?? 0;

/** "Training 30/30 · Predictors 4/12" from the rollup's per-kind counts. */
export function segmentText(rollup: TaskRollup, segments: RunStatusSegment[] = []) {
  return segments.flatMap((segment) => {
    const rows = segment.kinds.map((kind) => rollup.byKind[kind]).filter((row) => row !== undefined);
    const total = rows.reduce((sum, row) => sum + row.total, 0);
    if (!total) return [];
    const completed = rows.reduce((sum, row) => sum + row.completed, 0);
    return [`${segment.label} ${completed}/${total}${completed === total ? ' ✓' : ''}`];
  });
}

/**
 * The words of a run status, one vocabulary for every stage: not started · queued · running
 * · held · stopping · needs attention · runner stopped · completed (· cancelled).
 */
export function describeRollup(rollup: TaskRollup | undefined, { segments = [], now = Date.now(), unavailable = false }: { segments?: RunStatusSegment[]; now?: number; unavailable?: boolean } = {}): RunStatusCopy {
  if (!rollup) return unavailable
    ? { tone: 'idle', headline: 'Run status unavailable', parts: [], linkText: 'Open Task Center →' }
    : { tone: 'idle', headline: 'Checking run status…', parts: [], linkText: null };
  const done = segmentText(rollup, segments);
  const eta = rollup.eta?.seconds ? [`~${formatDuration(rollup.eta.seconds)} left`] : [];
  const failed = failedCount(rollup);
  const failedPart = failed ? [`${failed} failed`] : [];
  const progress = rollup.progress && rollup.progress.total > 1 && !segments.length ? [`${rollup.progress.completed} of ${rollup.progress.total} done`] : [];
  switch (rollup.state) {
    case 'not-started':
      return { tone: 'idle', headline: 'Not started', parts: done, linkText: null };
    case 'queued': {
      const waiting = rollup.waitingReason ? [`waiting: ${lowerFirst(rollup.waitingReason.replace(/\.$/, ''))}`] : [];
      return {
        tone: 'waiting', headline: rollup.paused ? 'Queued · queue paused' : 'Queued',
        parts: [...done, ...(rollup.queuePosition ? [`#${rollup.queuePosition} in line`] : []), ...waiting, ...eta],
        linkText: 'View in Task Center →',
      };
    }
    case 'running': {
      const single = rollup.live === 1 && rollup.active === 1;
      const current = single ? taskProgress(rollup.current?.progress)?.text : null;
      if (single) return { tone: 'active', headline: 'Running', parts: [...done, ...(current ? [lowerFirst(current)] : []), ...failedPart, ...eta], linkText: 'View in Task Center →' };
      return {
        tone: 'active', headline: `${rollup.active} running`,
        parts: [...done, ...(rollup.pending ? [`${rollup.pending} queued`] : []), ...failedPart, ...eta],
        linkText: 'View in Task Center →',
      };
    }
    case 'held':
      return { tone: 'waiting', headline: 'Held in Task Center', parts: done, linkText: 'Release there →' };
    case 'stopping':
      return { tone: 'waiting', headline: rollup.stopRequest === 'cancel' ? 'Cancel requested' : 'Stopping', parts: done, linkText: 'View in Task Center →' };
    case 'runner-stopped':
      return { tone: 'attention', headline: 'Queued, but the runner is stopped', parts: done, linkText: 'Start it in Task Center →' };
    case 'attention': {
      // The contract's words: "Needs attention · <reason>", as the experiment list and the
      // Task Center say it.
      const failure = rollup.lastFailure;
      const specific = failure?.message && failure.message !== 'The task failed' ? lowerFirst(failure.message) : null;
      const reason = specific ?? (failure && failed <= 1 ? failure.state === 'interrupted' ? 'interrupted' : 'failed' : null);
      return { tone: 'attention', headline: 'Needs attention', parts: [...(reason ? [reason] : []), ...done, ...progress, ...(failed > 1 ? [`${failed} failed`] : [])], linkText: 'Details →' };
    }
    case 'cancelled':
      return { tone: 'idle', headline: 'Cancelled', parts: [...done, ...progress], linkText: 'Details →' };
    case 'completed': {
      const took = secondsBetween(rollup.startedAt, rollup.finishedAt);
      const at = clockTime(rollup.finishedAt, now);
      return {
        tone: 'done', headline: at ? `Completed ${at}` : 'Completed',
        parts: [...done, ...(took ? [`took ${formatDuration(took)}`] : [])],
        linkText: rollup.progress ? 'Details →' : null,
      };
    }
    default:
      return { tone: 'idle', headline: String(rollup.state), parts: done, linkText: 'View in Task Center →' };
  }
}

/** The rollup a stage chip shows; fast polling only while its work is live. */
export function useRunRollup(scope: RollupScope | null, enabled = true) {
  return useQuery({
    queryKey: taskCenterKeys.rollup(scope ?? {}),
    queryFn: () => taskCenter.rollup(scope ?? {}),
    enabled: enabled && scope !== null,
    refetchIntervalInBackground: false,
    refetchInterval: (query) => query.state.error ? 30000 : rollupPollInterval(query.state.data),
    // A service without the rollup (404) is not retried on every render.
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  });
}

/** Live work ended between two reads of the same scope. */
export const rollupSettled = (before: TaskRollup | undefined, after: TaskRollup | undefined) =>
  Boolean(before && after && rollupIsLive(before) && !rollupIsLive(after));

/** Calls `onSettled` when the rollup's live work ends (for pages that show no chip). */
export function useRollupSettled(rollup: TaskRollup | undefined, onSettled: (rollup: TaskRollup) => void) {
  const previous = useRef<TaskRollup | undefined>(undefined);
  const settled = useRef(onSettled);
  settled.current = onSettled;
  useEffect(() => {
    const before = previous.current;
    previous.current = rollup;
    if (rollup && rollupSettled(before, rollup)) settled.current(rollup);
  }, [rollup]);
}

/** Resume/Retry show only where the work stopped short and a retry would resume something. */
export function actionShown(rollup: TaskRollup | undefined) {
  if (!rollup) return true;
  if (rollup.state === 'attention' || rollup.state === 'cancelled') return rollupRetryable(rollup);
  return rollup.state === 'not-started' || rollup.state === 'completed';
}

/**
 * One run status for a stage page: a deep link into the Task Center, the page's one science
 * action (Run, or Resume when a retry would resume something), and nothing operational.
 * `onSettled` fires when live work ends, so the page can refresh its science record.
 */
export default function RunStatusChip({ scope, variant = 'chip', label, segments, primaryAction, onSettled, hideWhenNotStarted = false, enabled = true, notStartedText }: {
  scope: RollupScope | null; variant?: 'chip' | 'row' | 'banner'; label?: string; segments?: RunStatusSegment[];
  primaryAction?: ReactNode; onSettled?: (rollup: TaskRollup) => void; hideWhenNotStarted?: boolean; enabled?: boolean;
  /** Replaces "Not started" (for example "Ready to run"). */
  notStartedText?: string;
}) {
  const query = useRunRollup(scope, enabled);
  const rollup = query.data;
  const previous = useRef<TaskRollup | undefined>(undefined);
  const settled = useRef(onSettled);
  settled.current = onSettled;
  const [announcement, setAnnouncement] = useState('');
  const copy = describeRollup(rollup, { segments, unavailable: Boolean(query.error && !rollup) });
  useEffect(() => {
    const before = previous.current;
    previous.current = rollup;
    if (!rollup) return;
    // Announce state changes only, never every progress tick.
    if (before && before.state !== rollup.state) setAnnouncement(`${label ? `${label}: ` : ''}${copy.headline}`);
    if (rollupSettled(before, rollup)) settled.current?.(rollup);
  }, [rollup, label, copy.headline]);
  if (!scope || (hideWhenNotStarted && rollup?.state === 'not-started')) return null;
  const headline = rollup?.state === 'not-started' && notStartedText ? notStartedText : copy.headline;
  const href = rollup?.href ?? '#task-center';
  const showAction = Boolean(primaryAction) && actionShown(rollup);
  return <div className={`run-status run-status-${variant} run-status-${copy.tone}`} data-state={rollup?.state ?? (query.error ? 'unavailable' : 'loading')}>
    <span className="run-status-dot" aria-hidden="true" />
    <span className="run-status-text">
      {label ? <strong className="run-status-label">{label}</strong> : null}
      <span className="run-status-headline">{headline}</span>
      {copy.parts.map((part) => <span key={part} className="run-status-part">{part}</span>)}
    </span>
    {showAction ? <span className="run-status-action">{primaryAction}</span> : null}
    {copy.linkText ? <a className="run-status-link" href={href}>{copy.linkText}</a> : null}
    <span className="sr-only" role="status">{announcement}</span>
  </div>;
}
