import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { development, developmentPollInterval, trainingActive } from '../api/development';
import { computeActive, computeStatusLabel, modelEvaluations, predictors } from '../api/predictors';
import { interpretations } from '../api/interpretation';
import { Badge, ErrorNotice, Icon } from './ui';

export default function JobTray({ inline = false, projectId }: { inline?: boolean; projectId?: string }) {
  const [open, setOpen] = useState(false);
  const jobs = useQuery({ queryKey: ['development-batches', projectId], queryFn: () => development.list(projectId!), enabled: Boolean(projectId), refetchInterval: (query) => developmentPollInterval(query.state.data) });
  const refits = useQuery({ queryKey: ['refit-builds', projectId], queryFn: () => predictors.refits(projectId!), enabled: Boolean(projectId), refetchInterval: 10000 });
  const evaluations = useQuery({ queryKey: ['model-evaluations', projectId], queryFn: () => modelEvaluations.list(projectId!), enabled: Boolean(projectId), refetchInterval: 10000 });
  const attention = useQuery({ queryKey: ['interpretations', projectId], queryFn: () => interpretations.list(projectId!), enabled: Boolean(projectId), refetchInterval: 10000 });
  const compute = [...(refits.data?.items ?? []).map((item) => ({ ...item, link: '#post-development', kindLabel: 'Refit' })), ...(evaluations.data?.items ?? []).map((item) => ({ ...item, link: `#evaluation?predictor=${encodeURIComponent(item.manifest.predictorId)}`, kindLabel: 'Evaluation' })), ...(attention.data?.items ?? []).map((item) => ({ ...item, link: `#interpretation?interpretation=${encodeURIComponent(item.id)}`, kindLabel: 'Attention' }))].filter((item) => item.execution && item.execution.status !== 'not_started');
  const executions = jobs.data?.executions ?? [];
  const active = executions.filter(trainingActive).length + compute.filter((item) => computeActive(item.execution)).length;
  const completed = executions.reduce((sum, execution) => sum + execution.runCounts.completed, 0);
  return (
    <aside
      className={`job-tray ${inline ? 'job-tray-inline' : ''} ${open ? 'open' : ''}`}
      aria-label="Training, evaluation and interpretation jobs"
    >
      <button
        className="job-tray-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls="job-tray-content"
      >
        <Icon name="clock" />
        <strong>Compute jobs</strong>
        <span>
          {!projectId ? 'Demonstration workspace' : jobs.isError || refits.isError || evaluations.isError || attention.isError
            ? 'Status unavailable'
            : jobs.isPending
              ? 'Connecting…'
              : jobs.data?.executionImplemented
                ? active ? `${active} active job${active === 1 ? '' : 's'} · ${completed} fold runs completed` : `${completed} fold runs completed`
                : 'Execution unavailable'}
        </span>
        <Icon name={open ? 'down' : 'chevron'} />
      </button>
      {open ? (
        <div className="job-tray-content" id="job-tray-content">
          <ErrorNotice error={jobs.error ?? refits.error ?? evaluations.error ?? attention.error} />
          {executions.length ? (
            <ul className="detail-list">
              {executions.map((job) => (
                <li key={job.batchId}>
                  <strong>{jobs.data?.items.find((batch) => batch.id === job.batchId)?.manifest.spec.batchName ?? job.batchId}</strong>
                  <span>{job.runCounts.completed}/{job.runCounts.total} completed</span>
                  <Badge>{job.cancelRequested && trainingActive(job) ? 'Cancelling' : job.status}</Badge>
                </li>
              ))}
            </ul>
          ) : !jobs.isError ? (
            <p className="muted">
              No launched MIL batches. Configure and freeze a batch in Experiments, then launch it when ready. Manage extraction, feature validation and packing jobs in Features.
            </p>
          ) : null}
          {compute.length ? <ul className="detail-list">{compute.map((job) => <li key={job.id}><a href={job.link}>{job.manifest.name}</a><span>{job.kindLabel}</span><Badge>{computeStatusLabel(job.execution)}</Badge></li>)}</ul> : null}
          <a className="text-link" href="#experiments">
            Experiments →
          </a>
        </div>
      ) : null}
    </aside>
  );
}
