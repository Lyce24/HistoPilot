import type { ReactNode } from 'react';
const paths: Record<string, ReactNode> = {
  overview: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  dataset: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7" />
    </>
  ),
  cohort: <path d="M3 4h18l-7 8v7l-4 2v-9z" />,
  features: <path d="m12 2 9 5-9 5-9-5zM3 12l9 5 9-5M3 17l9 5 9-5" />,
  experiments: <path d="M9 3h6M10 3v7L4 20a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1l-6-10V3M7 16h10" />,
  evaluation: <path d="M4 3v17h17M8 15v-4M13 15V7M18 15V4" />,
  explorer: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="3" />
      <circle cx="10" cy="10" r="3" />
      <path d="m12 12 5 5" />
    </>
  ),
  provenance: (
    <>
      <circle cx="6" cy="5" r="2" />
      <circle cx="18" cy="12" r="2" />
      <circle cx="6" cy="19" r="2" />
      <path d="M6 7v10M8 5h3a7 7 0 0 1 7 5M8 19h3a7 7 0 0 0 7-5" />
    </>
  ),
  system: (
    <>
      <rect x="3" y="3" width="18" height="7" rx="2" />
      <rect x="3" y="14" width="18" height="7" rx="2" />
      <path d="M7 6h.01M7 17h.01M11 6h6M11 17h6" />
    </>
  ),
  arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  chevron: <path d="m9 5 7 7-7 7" />,
  down: <path d="m6 9 6 6 6-6" />,
  check: <path d="m5 12 4 4L19 6" />,
  plus: <path d="M12 5v14M5 12h14" />,
  minus: <path d="M5 12h14" />,
  download: <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />,
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v6M12 7v.1" />
    </>
  ),
  search: (
    <>
      <circle cx="10" cy="10" r="6" />
      <path d="m15 15 6 6" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  arrowUp: <path d="M7 17 17 7M7 7h10v10" />,
  lock: (
    <>
      <rect x="5" y="10" width="14" height="11" rx="2" />
      <path d="M8 10V6a4 4 0 0 1 8 0v4" />
    </>
  ),
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  reset: <path d="M3 10a9 9 0 1 1 2 8M3 4v6h6" />,
  patient: (
    <>
      <circle cx="12" cy="7" r="4" />
      <path d="M4 21v-2a8 8 0 0 1 16 0v2" />
    </>
  ),
  folder: <path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11H3z" />,
  branch: (
    <>
      <circle cx="6" cy="5" r="2" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="6" cy="19" r="2" />
      <path d="M6 7v10M6 13h5a7 7 0 0 0 7-5" />
    </>
  ),
};
export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] ?? paths.info}
    </svg>
  );
}
export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}
export function Panel({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      <div className="panel-header">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}
/**
 * One empty state everywhere: what is missing, why, and the action that fills it.
 * `action` belongs here rather than only in the page header, so the next step is
 * where the reader is already looking.
 */
export function EmptyState({ title, description, icon = 'folder', action }: {
  title: string;
  description: string;
  icon?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon" aria-hidden="true"><Icon name={icon} size={22} /></span>
      <h3>{title}</h3>
      <p>{description}</p>
      {action ? <div className="empty-state-action">{action}</div> : null}
    </div>
  );
}
/**
 * A metric card presents one measurement. A value carrying no digits is a state
 * ("Sampled from training", "Unavailable"), not a measurement, so it is set as
 * readable text rather than in the numeric display size. Pass `variant` to
 * override the inference.
 */
export function Metric({
  label,
  value,
  note,
  variant,
}: {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  variant?: 'value' | 'text';
}) {
  const kind = variant ?? (typeof value === 'string' && !/\d/.test(value) ? 'text' : 'value');
  return (
    <article className="card metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value${kind === 'text' ? ' metric-value-text' : ''}`}>{value}</div>
      <div className="metric-note">{note}</div>
    </article>
  );
}
export function ErrorNotice({ error }: { error: Error | null }) {
  return error ? (
    <div className="callout callout-warning" role="alert">
      {error.message}
    </div>
  ) : null;
}
