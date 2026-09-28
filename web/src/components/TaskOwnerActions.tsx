import { taskCenter, type MovePosition, type OwnerAction, type TaskOwner } from '../api/taskCenter';
import { ConfirmAction, type TaskCenterActions } from '../lib/taskCenterActions';
import './TaskOwnerActions.css';

export const ownerActionIdentity = (key: string, action: OwnerAction, position?: MovePosition) => JSON.stringify(['owner', key, action, position ?? null]);
/** Every identity of one owner's actions starts with this prefix. */
const ownerPrefix = (key: string) => JSON.stringify(['owner', key]).slice(0, -1);

const confirmations: Partial<Record<OwnerAction, (title: string) => { question: string; confirm: string }>> = {
  stop: (title) => ({ question: `Stop the running tasks of “${title}” and hold it? They return to the queue and continue when you release it.`, confirm: 'Stop & hold' }),
  cancel: (title) => ({ question: `Cancel every unfinished task of “${title}”? Completed work is kept.`, confirm: 'Cancel tasks' }),
};

/**
 * Queue controls for one owner. The server says which actions apply; this only offers them.
 * Only this owner's controls wait while one of its requests is in flight.
 */
export default function TaskOwnerActions({ owner, actions }: { owner: TaskOwner; actions: Pick<TaskCenterActions, 'run' | 'pending'> & Partial<Pick<TaskCenterActions, 'isPending'>> }) {
  const busy = actions.isPending ? actions.isPending(ownerPrefix(owner.key), true) : Boolean(actions.pending);
  const allowed = owner.actions;
  const identity = (action: OwnerAction, position?: MovePosition) => ownerActionIdentity(owner.key, action, position);
  const pending = (action: OwnerAction, position?: MovePosition) => actions.isPending ? actions.isPending(identity(action, position)) : actions.pending === identity(action, position);
  function act(action: OwnerAction, position?: MovePosition) {
    void actions.run(identity(action, position), (operationId) => taskCenter.ownerAction(owner.key, action, operationId, position));
  }
  const label = (action: OwnerAction, idle: string, working: string, position?: MovePosition) => pending(action, position) ? working : idle;
  const confirmed = (action: 'stop' | 'cancel', idle: string, working: string, ariaLabel: string) => {
    const copy = confirmations[action]!(owner.title);
    return <ConfirmAction label={idle} busyLabel={working} busy={pending(action)} disabled={busy} ariaLabel={ariaLabel} question={copy.question} confirmLabel={copy.confirm} onConfirm={() => act(action)} />;
  };
  return <div className="task-owner-actions">
    {allowed.moveUp || allowed.moveDown ? <div className="task-owner-move" role="group" aria-label={`Queue position of ${owner.title}`}>
      <button type="button" className="btn btn-secondary btn-small" disabled={busy || !allowed.moveUp} aria-label={`Move ${owner.title} to the top of the queue`} onClick={() => act('move', 'top')}>Top</button>
      <button type="button" className="btn btn-secondary btn-small" disabled={busy || !allowed.moveUp} aria-label={`Move ${owner.title} up`} onClick={() => act('move', 'up')}>Up</button>
      <button type="button" className="btn btn-secondary btn-small" disabled={busy || !allowed.moveDown} aria-label={`Move ${owner.title} down`} onClick={() => act('move', 'down')}>Down</button>
    </div> : null}
    {allowed.release ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} aria-label={`Release ${owner.title}`} onClick={() => act('release')}>{label('release', 'Release', 'Releasing…')}</button>
      : allowed.hold ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} aria-label={`Hold ${owner.title}`} onClick={() => act('hold')}>{label('hold', 'Hold', 'Holding…')}</button> : null}
    {allowed.stop ? confirmed('stop', 'Stop & hold', 'Stopping…', `Stop & hold ${owner.title}`) : null}
    {allowed.retry ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} aria-label={`Retry failed tasks of ${owner.title}`} onClick={() => act('retry')}>{label('retry', 'Retry failed', 'Retrying…')}</button> : null}
    {allowed.cancel ? confirmed('cancel', 'Cancel', 'Cancelling…', `Cancel ${owner.title}`) : null}
  </div>;
}
