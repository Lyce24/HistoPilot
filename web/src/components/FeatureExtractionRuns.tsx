import { extractionActive, type ExtractionJob } from '../api/trident';
import { preparationLink, type PreparationContext } from '../lib/preparationRoute';
import ExtractionProgress, { extractionModelLabel, extractionStateLabel, extractionTaskLabel } from './ExtractionProgress';
import { Badge, Icon, Panel } from './ui';

/** Extraction exists before a feature source or bundle can be saved. */
export default function FeatureExtractionRuns({ jobs, context, isPending }: {
  jobs: ExtractionJob[];
  context: PreparationContext;
  isPending: boolean;
}) {
  if (isPending) return <p className="muted" role="status">Checking extraction runs…</p>;
  if (!jobs.length) return null;
  const active = jobs.filter(extractionActive);
  const finished = jobs.filter((job) => !extractionActive(job));
  const link = (job: ExtractionJob) => preparationLink('features', context, { extraction: job.id });
  return <Panel title="Extraction runs" subtitle="Follow slide processing here. Completed features can then be inspected and saved as a bundle."
    actions={active.length ? <Badge tone="purple">{active.length} active</Badge> : undefined}>
    <div className="pfm-extraction-runs">
      {active.map((job) => <article key={job.id} className="pfm-extraction-run" aria-label={`${extractionModelLabel(job)} extraction`}>
        <div className="pfm-extraction-heading">
          <div><h3><a href={link(job)}>{extractionModelLabel(job)} · {extractionTaskLabel(job)}</a></h3>
            <p className="muted">Started <time dateTime={job.createdAt}>{new Date(job.createdAt).toLocaleString()}</time></p>
          </div>
          <Badge tone="purple">{extractionStateLabel[job.state]}</Badge>
          <a className="btn btn-secondary btn-small" href={link(job)}>View progress <Icon name="arrow" size={15} /></a>
        </div>
        <ExtractionProgress job={job} />
      </article>)}
      {finished.length ? <details className="pfm-extraction-history" open={!active.length || undefined}>
        <summary>Previous extraction runs ({finished.length})</summary>
        <div className="pfm-source-list">{finished.map((job) => <a key={job.id} className="pfm-version-card" href={link(job)}>
          <strong>{extractionModelLabel(job)} · {extractionTaskLabel(job)}</strong>
          <span>{extractionStateLabel[job.state]}</span>
          <small><time dateTime={job.createdAt}>{new Date(job.createdAt).toLocaleString()}</time></small>
          <span>View run &amp; outputs <Icon name="arrow" size={14} /></span>
        </a>)}</div>
      </details> : null}
    </div>
  </Panel>;
}
