import { useState } from 'react';
import { useJobs } from '../api/queries';
import { Badge, ErrorNotice, Icon } from './ui';

export default function JobTray({ inline = false }: { inline?: boolean }) {
  const [open, setOpen] = useState(false);
  const jobs = useJobs();
  return (
    <aside
      className={`job-tray ${inline ? 'job-tray-inline' : ''} ${open ? 'open' : ''}`}
      aria-label="Compute jobs"
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
          {jobs.isError
            ? 'Status unavailable'
            : jobs.isPending
              ? 'Connecting…'
              : jobs.data?.executionEnabled
                ? `${jobs.data.jobs.length} jobs`
                : 'Execution not implemented'}
        </span>
        <Icon name={open ? 'down' : 'chevron'} />
      </button>
      {open ? (
        <div className="job-tray-content" id="job-tray-content">
          <ErrorNotice error={jobs.error} />
          {jobs.data?.jobs.length ? (
            <ul className="detail-list">
              {jobs.data.jobs.map((job) => (
                <li key={job.id}>
                  <strong>{job.name ?? job.id}</strong>
                  <Badge>{job.status}</Badge>
                </li>
              ))}
            </ul>
          ) : !jobs.isError ? (
            <p className="muted">
              No compute jobs. Experiment drafts are saved by the service; isolated worker
              execution will be connected in a later implementation.
            </p>
          ) : null}
          <a className="text-link" href="#system">
            System & storage →
          </a>
        </div>
      ) : null}
    </aside>
  );
}
