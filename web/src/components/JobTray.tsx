import { useEffect, useId, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { formatDuration, taskCenter, taskCenterHref, taskCenterKeys, taskDisplayTitle, taskProgress, taskStateLabel, type TaskItem, type TaskRollup } from '../api/taskCenter';
import { useRunRollup } from './RunStatusChip';
import { Badge, ErrorNotice, Icon } from './ui';
import './JobTray.css';

interface JobLink { id: string; name: string; link: string; detail: string; status: string }

export function JobTrayLinks({ jobs }: { jobs: JobLink[] }) {
  return <ul className="detail-list job-tray-list">{jobs.map((job) => <li key={job.id}><a href={job.link} aria-label={`Details: ${job.name}`}>{job.name}</a><span>{job.detail}</span><Badge>{job.status}</Badge></li>)}</ul>;
}

/** Every task opens its detail in the Task Center; the record link lives there. */
export function taskJobLink(task: TaskItem): JobLink {
  return { id: task.id, name: taskDisplayTitle(task), link: taskCenterHref({ task: task.id }), detail: taskProgress(task.progress)?.text ?? task.waitingReason ?? task.owner.title, status: taskStateLabel(task.state) };
}

/** "4 running · 41 queued · 1 failed · ~2 h left". */
export function jobTrayStatus(rollup: Pick<TaskRollup, 'counts' | 'paused' | 'runnerAlive' | 'recentFailures' | 'eta'> | undefined) {
  if (!rollup) return '';
  const count = (state: keyof TaskRollup['counts']) => rollup.counts[state] ?? 0;
  const running = count('starting') + count('running') + count('stopping');
  const waiting = count('queued') + count('blocked');
  return [`${running} running`, `${waiting} queued`, rollup.recentFailures ? `${rollup.recentFailures} failed` : '',
    rollup.paused && waiting ? 'queue paused' : '', !rollup.runnerAlive && waiting ? 'runner stopped' : '',
    rollup.eta?.seconds && (running || waiting) ? `~${formatDuration(rollup.eta.seconds)} left` : ''].filter(Boolean).join(' · ');
}

/**
 * Tasks the live list can show (running, queued or waiting on dependencies). The rollup's
 * `live` also counts concluded tasks awaiting an automatic resume, which that list leaves out.
 */
export const liveTaskCount = (rollup?: Pick<TaskRollup, 'counts'> | null) =>
  (['blocked', 'queued', 'starting', 'running', 'stopping'] as const).reduce((sum, state) => sum + (rollup?.counts[state] ?? 0), 0);

/**
 * The machine's run status on every page: the same rollup the stage chips read, the live
 * tasks (each opening its Task Center detail) and one link to the Task Center.
 */
export default function JobTray({ inline = false }: { inline?: boolean }) {
  const [open, setOpen] = useState(false);
  const contentId = useId();
  const root = useRef<HTMLElement>(null);
  const machine = useRunRollup({});
  const rollup = machine.data;
  const live = (rollup?.live ?? 0) > 0;
  const tasks = useQuery({ queryKey: taskCenterKeys.tasks('tray'), queryFn: () => taskCenter.tasks({ state: 'live', limit: 5 }), enabled: open, refetchIntervalInBackground: false, refetchInterval: live ? 5000 : 30000 });
  const loading = machine.isPending;
  const error = machine.error ?? null;
  const text = jobTrayStatus(rollup);
  const status = error ? text ? `${text} · status may be outdated` : 'Task status unavailable'
    : loading ? 'Checking tasks…' : text;
  // Nothing running, queued or recently failed: the tray recedes until there is activity to report.
  const idle = !open && !error && !loading && !live && !rollup?.recentFailures;
  const liveTasks = tasks.data?.tasks ?? [];
  // Announce only a change of state (work started, finished or failed), not each count.
  const [announcement, setAnnouncement] = useState('');
  const previous = useRef<string | null>(null);
  const phase = !rollup ? null : live ? 'active' : rollup.recentFailures ? 'failed' : 'idle';
  useEffect(() => {
    if (phase && previous.current && previous.current !== phase) setAnnouncement(phase === 'active' ? 'Tasks are running.' : phase === 'failed' ? 'A task failed. See the Task Center.' : 'All tasks have finished.');
    previous.current = phase;
  }, [phase]);
  useEffect(() => {
    if (!open || inline) return;
    const close = (event: KeyboardEvent | PointerEvent) => {
      if (event instanceof KeyboardEvent ? event.key === 'Escape' : !root.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener('keydown', close);
    window.addEventListener('pointerdown', close);
    return () => { window.removeEventListener('keydown', close); window.removeEventListener('pointerdown', close); };
  }, [open, inline]);
  return (
    <aside
      ref={root}
      className={`job-tray ${inline ? 'job-tray-inline' : ''} ${open ? 'open' : ''} ${idle ? 'is-idle' : ''}`}
      aria-label="Task Center summary"
    >
      <button
        type="button"
        className="job-tray-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls={contentId}
      >
        <Icon name="clock" />
        <strong>Tasks</strong>
        <span>{status}</span>
        <Icon name={open ? 'down' : 'chevron'} />
      </button>
      <span className="sr-only" role="status">{announcement}</span>
      {open ? (
        <div className="job-tray-content" id={contentId}>
          <ErrorNotice error={error ?? tasks.error ?? null} />
          {error ? <><p className="muted">Some statuses could not be refreshed. Counts shown may be outdated.</p><button type="button" className="btn btn-secondary btn-small" disabled={machine.isFetching} onClick={() => { void machine.refetch(); void tasks.refetch(); }}>Retry task status</button></> : null}
          {loading || tasks.isPending ? <p className="muted" role="status">Checking tasks…</p> : null}
          {liveTasks.length ? <JobTrayLinks jobs={liveTasks.map(taskJobLink)} /> : null}
          {liveTaskCount(rollup) > liveTasks.length && liveTasks.length ? <p className="muted">{liveTaskCount(rollup) - liveTasks.length} more in the Task Center.</p> : null}
          {rollup?.lastFailure && rollup.recentFailures ? <p className="job-tray-failure"><Badge tone="orange">Failed</Badge> <a href={taskCenterHref({ task: rollup.lastFailure.taskId })}>{rollup.lastFailure.title}</a>{rollup.lastFailure.message ? ` · ${rollup.lastFailure.message}` : ''}</p> : null}
          {!loading && !tasks.isPending && !live && !rollup?.recentFailures ? <p className="muted">Nothing is running or queued. Training, refits, evaluations, inference and attention maps appear here once submitted.</p> : null}
          <div className="job-tray-links"><a className="text-link" href={rollup?.lastFailure && rollup.recentFailures ? taskCenterHref({ task: rollup.lastFailure.taskId }) : '#task-center'}>Open Task Center →</a></div>
        </div>
      ) : null}
    </aside>
  );
}
