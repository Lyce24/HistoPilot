import { useQueryClient } from '@tanstack/react-query';
import { rollupRetryable, taskCenter, type RollupScope, type TaskOwner, type TaskRollup } from '../api/taskCenter';
import { TaskCenterActionNotice, useTaskCenterActions } from '../lib/taskCenterActions';
import RunStatusChip, { useRunRollup } from './RunStatusChip';
import { ownerActionIdentity } from './TaskOwnerActions';
import './ExperimentQueueBar.css';

/** Another workspace on this machine can hold a copy of the same project and experiment IDs; only this workspace's owner is ours. */
export const experimentOwner = (owners: TaskOwner[] | undefined, project: string, experimentId: string) =>
  owners?.find((owner) => owner.sameWorkspace && owner.kind === 'experiment' && owner.id === experimentId && owner.projectId === project);

/** What the experiment's one status line counts: fold runs and their collections, then predictors. */
export const experimentSegments = [
  { label: 'Training', kinds: ['mil-fold', 'mil-collect'] },
  { label: 'Predictor jobs', kinds: ['predictor-coordinator', 'compute-job'] },
];
/**
 * Resume is offered when the experiment stopped short (failed, interrupted or cancelled work)
 * and the owner's retry would resume something, as the Task Center's own Retry.
 */
export const experimentNeedsResume = (rollup?: Pick<TaskRollup, 'state' | 'retryable'> | null) =>
  (rollup?.state === 'attention' || rollup?.state === 'cancelled') && rollupRetryable(rollup);

/**
 * The experiment's rollup scope. `batchIds` (retained batches and the current submission)
 * leaves out concluded tasks of trashed or replaced batches, which keep their tasks.
 */
export function experimentRollupScope(project: string, experimentId: string, ownerKey?: string | null, batchIds: string[] = []): RollupScope {
  const batches = [...new Set(batchIds)].sort().join(',');
  return { ...(ownerKey ? { owner: ownerKey } : { ownerKind: 'experiment', ownerId: experimentId, project }), ...(batches ? { batchIds: batches } : {}) };
}

/**
 * The experiment's run status: one line (training and predictor progress, queue place,
 * failures, time left) linking to the experiment in the Task Center, plus one Resume when it
 * needs attention. Queue controls, logs and resources live in the Task Center.
 * `managed` is false for experiments submitted before the Task Center: their batches run in
 * their own tmux sessions, so the line appears only if the task store knows the experiment.
 */
export default function ExperimentQueueBar({ project, experimentId, ownerKey, managed = true, batchIds }: { project: string; experimentId: string; ownerKey?: string | null; waitingReason?: string | null; managed?: boolean; batchIds?: string[] }) {
  const scope = experimentRollupScope(project, experimentId, ownerKey, batchIds);
  const rollup = useRunRollup(scope);
  const client = useQueryClient();
  const actions = useTaskCenterActions([['model-experiment', project], ['model-experiments', project], ['development-batches', project], ['training-execution', project], ['predictors', project]]);
  // Older experiments may have no tasks at all: say nothing rather than "not started".
  if (!managed && (!rollup.data || rollup.data.state === 'not-started')) return null;
  const key = rollup.data?.ownerKey ?? ownerKey ?? null;
  const resuming = key ? actions.isPending(ownerActionIdentity(key, 'retry')) : false;
  const resume = key && experimentNeedsResume(rollup.data)
    ? <button type="button" className="btn btn-primary btn-small" disabled={resuming} onClick={() => void actions.run(ownerActionIdentity(key, 'retry'), (operationId) => taskCenter.ownerAction(key, 'retry', operationId))}>{resuming ? 'Resuming…' : 'Resume'}</button>
    : null;
  const refresh = () => {
    for (const queryKey of [['model-experiment', project], ['model-experiments', project], ['development-batches', project], ['training-execution', project], ['predictors', project], ['development-results', project]]) void client.invalidateQueries({ queryKey });
  };
  return <section className="experiment-queue-bar" aria-label="Experiment run status">
    <RunStatusChip scope={scope} variant="row" segments={experimentSegments} primaryAction={resume} onSettled={refresh} hideWhenNotStarted={!managed} notStartedText="Nothing from this experiment is queued yet" />
    <TaskCenterActionNotice actions={actions} />
  </section>;
}
