import { useId, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { development, developmentPollInterval, trainingActive, type DevelopmentBatchList } from '../api/development';
import { computeActive, computePollInterval, computeStatusLabel, modelEvaluations, predictors } from '../api/predictors';
import { isInferenceRun } from '../lib/inference';
import { interpretations } from '../api/interpretation';
import { extractionActive, trident, type ExtractionJob } from '../api/trident';
import { Badge, ErrorNotice, Icon } from './ui';

interface JobLink { id: string; name: string; link: string; detail: string; status: string }

export function JobTrayLinks({ jobs }: { jobs: JobLink[] }) {
  return <ul className="detail-list">{jobs.map((job) => <li key={job.id}><a href={job.link} aria-label={`View progress: ${job.name}`}>{job.name}</a><span>{job.detail}</span><Badge>{job.status}</Badge></li>)}</ul>;
}

export function extractionJobLink(job: ExtractionJob): JobLink {
  const progress = job.progress;
  const count = progress?.completed != null && progress.total != null ? ` · ${progress.completed}/${progress.total} ${progress.unit}` : '';
  const encoder = String(job.spec.options.slide_encoder || job.spec.options.patch_encoder || 'Slide features');
  return { id: job.id, name: `${encoder} extraction · ${job.id}`, link: `#features?extraction=${encodeURIComponent(job.id)}`, detail: `${progress?.label || 'Extraction'}${count}`, status: job.state };
}

export const computeJobLink = (kind: 'refit' | 'evaluation' | 'interpretation', id: string) => kind === 'refit' ? `#post-development?tab=refits&refit=${encodeURIComponent(id)}` : `#${kind}?${kind}=${encodeURIComponent(id)}`;

export function trainingJobLinks(data?: DevelopmentBatchList): JobLink[] {
  return (data?.executions ?? []).map((job) => {
    const batch = data?.items.find((item) => item.id === job.batchId);
    const experiment = batch?.manifest.spec.experimentId || `legacy-${job.batchId}`;
    return { id: job.batchId, name: batch?.manifest.spec.batchName ?? job.batchId, link: `#experiments?experiment=${encodeURIComponent(experiment)}&tab=runs&batch=${encodeURIComponent(job.batchId)}`, detail: `${job.runCounts.completed}/${job.runCounts.total} completed`, status: job.cancelRequested && trainingActive(job) ? 'Cancelling' : job.status };
  });
}

export default function JobTray({ inline = false, projectId }: { inline?: boolean; projectId?: string }) {
  const [open, setOpen] = useState(false);
  const contentId = useId();
  const jobs = useQuery({ queryKey: ['development-batches', projectId], queryFn: () => development.list(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => developmentPollInterval(query.state.data) });
  const refits = useQuery({ queryKey: ['refit-builds', projectId], queryFn: () => predictors.refits(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const evaluations = useQuery({ queryKey: ['model-evaluations', projectId], queryFn: () => modelEvaluations.list(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const attention = useQuery({ queryKey: ['interpretations', projectId], queryFn: () => interpretations.list(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const extractions = useQuery({ queryKey: ['extractions', projectId, 'jobs'], queryFn: () => trident.jobs(projectId!), enabled: Boolean(projectId), refetchIntervalInBackground: false, refetchInterval: (query) => query.state.data?.jobs.some(extractionActive) ? 3000 : 30000 });
  const extractionJobs = extractions.data?.jobs ?? [];
  const compute = [...(refits.data?.items ?? []).map((item) => ({ ...item, link: computeJobLink('refit', item.id), kindLabel: 'Refit' })), ...(evaluations.data?.items ?? []).map((item) => isInferenceRun(item) ? { ...item, link: `#inference?evaluation=${encodeURIComponent(item.id)}`, kindLabel: 'Inference' } : { ...item, link: computeJobLink('evaluation', item.id), kindLabel: 'Evaluation' }), ...(attention.data?.items ?? []).map((item) => ({ ...item, link: computeJobLink('interpretation', item.id), kindLabel: 'Attention' }))].filter((item) => item.execution && item.execution.status !== 'not_started');
  const executions = jobs.data?.executions ?? [];
  const active = executions.filter(trainingActive).length + compute.filter((item) => computeActive(item.execution)).length + extractionJobs.filter(extractionActive).length;
  const completed = executions.reduce((sum, execution) => sum + execution.runCounts.completed, 0);
  const queries = [jobs, refits, evaluations, attention, extractions];
  const loading = Boolean(projectId) && queries.some((query) => query.isPending);
  const error = queries.find((query) => query.error)?.error ?? null;
  const summary = active ? `${active} active job${active === 1 ? '' : 's'} · ${completed} fold runs completed`
    : completed ? `No active jobs · ${completed} fold runs completed`
      : executions.length || compute.length || extractionJobs.length ? 'No active jobs' : 'No jobs yet';
  const status = !projectId ? 'Demonstration workspace'
    : error ? active ? `${summary} · status may be outdated` : 'Status unavailable'
      : loading ? active ? `At least ${active} active job${active === 1 ? '' : 's'} · checking remaining jobs…` : 'Checking compute jobs…'
        : summary;
  // A project with nothing running should not carry a prominent panel; it recedes
  // until there is activity to report.
  const idle = !open && !error && !loading && !active && !executions.length && !compute.length && !extractionJobs.length;
  return (
    <aside
      className={`job-tray ${inline ? 'job-tray-inline' : ''} ${open ? 'open' : ''} ${idle ? 'is-idle' : ''}`}
      aria-label="Extraction, training, evaluation and interpretation jobs"
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
          {loading ? <p className="muted" role="status">Checking extraction, training, refit, evaluation and attention jobs…</p> : null}
          {extractionJobs.length ? <JobTrayLinks jobs={extractionJobs.map(extractionJobLink)} /> : null}
          {executions.length ? <JobTrayLinks jobs={trainingJobLinks(jobs.data)} /> : projectId && !jobs.isPending && !jobs.isError && !extractionJobs.length && !compute.length ? (
            <p className="muted">No launched MIL batches. Configure and freeze a batch in Experiments, then launch it when ready. Start extraction in Slide features.</p>
          ) : null}
          {projectId && jobs.data?.executionImplemented === false ? <p className="muted">Training launch is unavailable from this service. Other compute jobs are listed separately below.</p> : null}
          {compute.length ? <JobTrayLinks jobs={compute.map((job) => ({ id: job.id, link: job.link, name: job.manifest.name, detail: job.kindLabel, status: computeStatusLabel(job.execution) }))} /> : null}
          <a className="text-link" href="#experiments">
            Experiments →
          </a>
          {projectId ? <a className="text-link" href="#operations">All pipeline jobs & backups →</a> : null}
        </div>
      ) : null}
    </aside>
  );
}
