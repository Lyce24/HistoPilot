import { useId, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { development, developmentPollInterval, trainingActive } from '../api/development';
import { computeActive, computePollInterval, computeStatusLabel, modelEvaluations, predictors } from '../api/predictors';
import { interpretations } from '../api/interpretation';
import { Badge, ErrorNotice, Icon } from './ui';

export default function JobTray({ inline = false, projectId }: { inline?: boolean; projectId?: string }) {
  const [open, setOpen] = useState(false);
  const contentId = useId();
  const jobs = useQuery({ queryKey: ['development-batches', projectId], queryFn: () => development.list(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => developmentPollInterval(query.state.data) });
  const refits = useQuery({ queryKey: ['refit-builds', projectId], queryFn: () => predictors.refits(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const evaluations = useQuery({ queryKey: ['model-evaluations', projectId], queryFn: () => modelEvaluations.list(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const attention = useQuery({ queryKey: ['interpretations', projectId], queryFn: () => interpretations.list(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const compute = [...(refits.data?.items ?? []).map((item) => ({ ...item, link: `#experiments?experiment=${encodeURIComponent(item.manifest.experimentId)}&tab=predictors`, kindLabel: 'Refit' })), ...(evaluations.data?.items ?? []).map((item) => ({ ...item, link: `#evaluation?predictor=${encodeURIComponent(item.manifest.predictorId)}`, kindLabel: 'Evaluation' })), ...(attention.data?.items ?? []).map((item) => ({ ...item, link: `#interpretation?interpretation=${encodeURIComponent(item.id)}`, kindLabel: 'Attention' }))].filter((item) => item.execution && item.execution.status !== 'not_started');
  const executions = jobs.data?.executions ?? [];
  const active = executions.filter(trainingActive).length + compute.filter((item) => computeActive(item.execution)).length;
  const completed = executions.reduce((sum, execution) => sum + execution.runCounts.completed, 0);
  const queries = [jobs, refits, evaluations, attention];
  const loading = Boolean(projectId) && queries.some((query) => query.isPending);
  const error = queries.find((query) => query.error)?.error ?? null;
  const summary = active ? `${active} active job${active === 1 ? '' : 's'} · ${completed} fold runs completed`
    : completed ? `No active jobs · ${completed} fold runs completed`
      : executions.length || compute.length ? 'No active jobs' : 'No jobs yet';
  const status = !projectId ? 'Demonstration workspace'
    : error ? active ? `${summary} · status may be outdated` : 'Status unavailable'
      : loading ? active ? `At least ${active} active job${active === 1 ? '' : 's'} · checking remaining jobs…` : 'Checking compute jobs…'
        : summary;
  // A project with nothing running should not carry a prominent panel; it recedes
  // until there is activity to report.
  const idle = !open && !error && !loading && !active && !executions.length && !compute.length;
  return (
    <aside
      className={`job-tray ${inline ? 'job-tray-inline' : ''} ${open ? 'open' : ''} ${idle ? 'is-idle' : ''}`}
      aria-label="Training, evaluation and interpretation jobs"
    >
      <button
        type="button"
        className="job-tray-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls={contentId}
      >
        <Icon name="clock" />
        <strong>Compute jobs</strong>
        <span role="status">{status}</span>
        <Icon name={open ? 'down' : 'chevron'} />
      </button>
      {open ? (
        <div className="job-tray-content" id={contentId}>
          <ErrorNotice error={error} />
          {error ? <><p className="muted">Some job statuses could not be refreshed. Existing counts may be outdated.</p><button type="button" className="btn btn-secondary btn-small" disabled={queries.some((query) => query.isFetching)} onClick={() => { for (const query of queries) void query.refetch(); }}>Retry job status</button></> : null}
          {loading ? <p className="muted" role="status">Checking training, refit, evaluation and attention jobs…</p> : null}
          {executions.length ? (
            <ul className="detail-list">
              {executions.map((job) => (
                <li key={job.batchId}>
                  <a href="#experiments">{jobs.data?.items.find((batch) => batch.id === job.batchId)?.manifest.spec.batchName ?? job.batchId}</a>
                  <span>{job.runCounts.completed}/{job.runCounts.total} completed</span>
                  <Badge>{job.cancelRequested && trainingActive(job) ? 'Cancelling' : job.status}</Badge>
                </li>
              ))}
            </ul>
          ) : projectId && !jobs.isPending && !jobs.isError ? (
            <p className="muted">
              No launched MIL batches. Configure and freeze a batch in Experiments, then launch it when ready. Manage extraction, feature validation and packing jobs in Features.
            </p>
          ) : null}
          {projectId && jobs.data?.executionImplemented === false ? <p className="muted">Training launch is unavailable from this service. Other compute jobs are listed separately below.</p> : null}
          {compute.length ? <ul className="detail-list">{compute.map((job) => <li key={job.id}><a href={job.link}>{job.manifest.name}</a><span>{job.kindLabel}</span><Badge>{computeStatusLabel(job.execution)}</Badge></li>)}</ul> : null}
          <a className="text-link" href="#experiments">
            Experiments →
          </a>
          {projectId ? <a className="text-link" href="#operations">All pipeline jobs & backups →</a> : null}
        </div>
      ) : null}
    </aside>
  );
}
