import { useId, useRef, useState } from 'react';
import { useQueryClient, type QueryKey } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { taskCenterKeys } from '../api/taskCenter';
import { useWorkspaceNavigationGuard } from './workspaceNavigation';
import './taskCenterActions.css';

/**
 * Send one Task Center request with a stable operation ID. A definite rejection
 * clears the receipt; a lost or server-side failure keeps it so a repeat of the
 * same request cannot be applied twice. `identity` must describe the whole
 * request (target, action and body).
 *
 * A receipt only serves an immediate repeat. Once a different request is sent
 * (e.g. Release after a lost Hold), replaying the older ID later would return
 * its stored outcome instead of applying the new intent, so it is dropped.
 */
export async function runTaskCenterOperation<T>(receipts: Map<string, string>, identity: string, call: (operationId: string) => Promise<T>, inFlight: ReadonlySet<string> = new Set()): Promise<T> {
  for (const key of [...receipts.keys()]) if (key !== identity && !inFlight.has(key)) receipts.delete(key);
  const operationId = receipts.get(identity) ?? crypto.randomUUID();
  receipts.set(identity, operationId);
  try {
    const result = await call(operationId);
    receipts.delete(identity);
    return result;
  } catch (reason) {
    if (definiteRejection(reason)) receipts.delete(identity);
    throw reason;
  }
}

/**
 * A request the server definitely refused (a 4xx other than a timeout) can be sent again
 * with a new operation ID. Anything else (a lost response, a 5xx) may have been applied,
 * so a repeat must reuse the same operation ID.
 */
export const definiteRejection = (reason: unknown) => reason instanceof ApiError && reason.status < 500 && reason.status !== 408;

export interface TaskCenterActions {
  run: <T>(identity: string, call: (operationId: string) => Promise<T>) => Promise<T | undefined>;
  /** Identity of the most recent request in flight, if any. */
  pending: string | null;
  /** Whether this request (or, with a prefix match, any request for this target) is in flight. */
  isPending: (identity: string, prefix?: boolean) => boolean;
  /** Identity whose last response was lost; repeating it reuses its operation ID. */
  uncertain: string | null;
  error: Error | null;
  dismiss: () => void;
}

/**
 * Task Center requests for one surface, guarded against navigation while any is in flight.
 * Different targets may act at once; the same request is never sent twice concurrently.
 */
export function useTaskCenterActions(invalidate: QueryKey[] = []): TaskCenterActions {
  const client = useQueryClient();
  const receipts = useRef(new Map<string, string>());
  const inFlight = useRef(new Set<string>());
  const [flying, setFlying] = useState<string[]>([]);
  const [uncertain, setUncertain] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const pending = flying.at(-1) ?? null;
  useWorkspaceNavigationGuard(pending ? 'A Task Center request is still pending. Leaving now may hide its outcome.' : null);
  async function run<T>(identity: string, call: (operationId: string) => Promise<T>) {
    if (inFlight.current.has(identity)) return undefined;
    inFlight.current.add(identity);
    setFlying([...inFlight.current]); setError(null); setUncertain(null);
    try {
      return await runTaskCenterOperation(receipts.current, identity, call, inFlight.current);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The Task Center request failed.'));
      setUncertain(receipts.current.has(identity) ? identity : null);
      return undefined;
    } finally {
      inFlight.current.delete(identity);
      setFlying([...inFlight.current]);
      for (const queryKey of [taskCenterKeys.all, ...invalidate]) void client.invalidateQueries({ queryKey });
    }
  }
  const isPending = (identity: string, prefix = false) => prefix ? flying.some((item) => item.startsWith(identity)) : flying.includes(identity);
  return { run, pending, isPending, uncertain, error, dismiss: () => { setError(null); } };
}

export function TaskCenterActionNotice({ actions }: { actions: Pick<TaskCenterActions, 'error' | 'uncertain' | 'pending'> }) {
  if (!actions.error) return null;
  return <div className="callout callout-warning" role="alert">
    {actions.error.message}
    {actions.uncertain && !actions.pending ? ' The request may already have been applied. Repeating it is safe: it reuses the same request identity.' : ''}
  </div>;
}

/**
 * A destructive action that asks first, inside the page (not a browser dialog): the first
 * press shows the question with a confirming and a dismissing button.
 */
export function ConfirmAction({ label, busyLabel, question, confirmLabel, busy = false, disabled = false, ariaLabel, onConfirm, className = 'btn btn-secondary btn-small' }: {
  label: string; busyLabel?: string; question: string; confirmLabel: string; busy?: boolean; disabled?: boolean; ariaLabel?: string;
  onConfirm: () => void; className?: string;
}) {
  const [asking, setAsking] = useState(false);
  const questionId = useId();
  if (busy) return <button type="button" className={className} disabled>{busyLabel ?? label}</button>;
  if (!asking) return <button type="button" className={className} disabled={disabled} aria-label={ariaLabel} onClick={() => setAsking(true)}>{label}</button>;
  return <span className="confirm-action" role="alertdialog" aria-labelledby={questionId}>
    <span id={questionId}>{question}</span>
    <button type="button" className="btn btn-primary btn-small" autoFocus onClick={() => { setAsking(false); onConfirm(); }}>{confirmLabel}</button>
    <button type="button" className="text-button" onClick={() => setAsking(false)}>Keep</button>
  </span>;
}
