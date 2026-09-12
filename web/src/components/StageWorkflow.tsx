import { useEffect, useId, useRef, type ReactNode } from 'react';
import { Icon } from './ui';
import { RecordManageButton, RecordManagementScope, useHasRecordManagementScope } from './RecordManagement';
import './StageWorkflow.css';

export interface StageStep {
  id: string;
  title: string;
  description?: string;
  disabled?: boolean;
  complete?: boolean;
  buttonId?: string;
}

/** A repeated click on the current module opens its library without discarding local edits. */
export function useStageLibrary(onOpen: () => void) {
  const handler = useRef(onOpen);
  useEffect(() => { handler.current = onOpen; });
  useEffect(() => {
    const open = () => handler.current();
    window.addEventListener('histopilot:stage-library', open);
    return () => window.removeEventListener('histopilot:stage-library', open);
  }, []);
}

/** Navigation changes the active page; completion remains owned by each workflow. */
export function StageSteps({ steps, current, onChange, label = 'Workflow steps', disabled = false }: {
  steps: readonly StageStep[];
  current: string;
  onChange: (id: string) => void;
  label?: string;
  disabled?: boolean;
}) {
  return <nav className="stage-steps" aria-label={label} onKeyDown={(event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')];
    const index = buttons.indexOf(event.target as HTMLButtonElement);
    if (index < 0 || !buttons.length) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus();
    buttons[next]?.click();
  }}>
    <ol>{steps.map((step, index) => <li key={step.id}>
      <button type="button" id={step.buttonId} className={`stage-step${step.complete ? ' is-complete' : ''}`}
        aria-current={step.id === current ? 'step' : undefined} disabled={disabled || step.disabled}
        onClick={() => { if (!disabled && !step.disabled && step.id !== current) onChange(step.id); }}>
        <span className="stage-step-number" aria-hidden="true">{step.complete && current !== step.id ? <Icon name="check" size={15} /> : index + 1}</span>
        <span className="stage-step-copy"><strong>{step.title}</strong>{step.description ? <small>{step.description}</small> : null}</span>
        {step.disabled ? <Icon name="lock" size={13} /> : null}
      </button>
    </li>)}</ol>
  </nav>;
}

/** Retain child state across steps. Never key/remount a form just to move the viewport. */
export function StagePage({ pageKey, children, className = '' }: {
  pageKey: string | number;
  children: ReactNode;
  className?: string;
}) {
  const page = useRef<HTMLDivElement>(null);
  const previous = useRef(pageKey);
  useEffect(() => {
    if (previous.current === pageKey) return;
    previous.current = pageKey;
    const frame = window.requestAnimationFrame(() => {
      const element = page.current;
      if (!element || element.closest('[hidden]')) return;
      // Include the page heading and step navigation, rather than landing halfway down a form.
      const top = element.closest('.stage-workspace') ?? element.closest('.clinical-workspace') ?? element;
      top.scrollIntoView({ block: 'start', behavior: 'instant' });
      element.focus({ preventScroll: true });
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches && typeof element.animate === 'function') {
        element.animate([{ opacity: 0.35, transform: 'translateY(5px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 160, easing: 'ease-out' });
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [pageKey]);
  return <div className={`stage-page ${className}`.trim()} ref={page} tabIndex={-1} data-stage-page={pageKey}>{children}</div>;
}

export function StageLibrary({ project, title, backLabel, children }: {
  project: string;
  title: string;
  backLabel?: string;
  children: ReactNode;
}) {
  const managedByStage = useHasRecordManagementScope();
  const library = <section className="stage-library card" aria-label={title}>
    <div className="stage-library-body">{children}</div>
  </section>;
  return managedByStage ? library : <RecordManagementScope project={project} backLabel={backLabel ?? `Back to ${title.toLowerCase()}`}>{library}</RecordManagementScope>;
}

/** Every library starts with the same compact controls; each phase owns its filters. */
export function StageLibraryToolbar({ search, onSearch, searchLabel, placeholder, children, count, total, actions, onReset }: {
  search: string;
  onSearch: (value: string) => void;
  searchLabel: string;
  placeholder?: string;
  children?: ReactNode;
  count?: number;
  total?: number;
  actions?: ReactNode;
  onReset?: () => void;
}) {
  const searchId = useId();
  return <div className="stage-library-tools">
    <div className="stage-library-toolbar">
      <label className="label stage-library-search" htmlFor={searchId}>Search
        <span className="stage-library-search-field"><Icon name="search" size={16} />
          <input id={searchId} type="search" className="field" aria-label={searchLabel} placeholder={placeholder ?? searchLabel} value={search} onChange={(event) => onSearch(event.target.value)} />
        </span>
      </label>
      {children}
      {actions ? <div className="stage-library-tool-actions inline-actions">{actions}</div> : null}
    </div>
    {count !== undefined || onReset ? <div className="stage-library-summary">
      {count !== undefined ? <span role="status" aria-live="polite">{total !== undefined && count !== total ? `${count} of ${total} records` : `${count} ${count === 1 ? 'record' : 'records'}`}</span> : null}
      {onReset ? <button type="button" className="text-button" onClick={onReset}>Clear filters</button> : null}
    </div> : null}
  </div>;
}

/** Open the shared record page without navigating away from this stage. */
export function StageRecordManageButton({ type, id, name }: {
  type: 'dataset' | 'configuration' | 'draft' | 'packing' | 'extraction';
  id: string;
  name: string;
}) {
  return <RecordManageButton recordKey={`${type}:${id}`} name={name} />;
}
