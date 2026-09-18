import { StageBackButton } from './StageActions';
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { cleanupPollInterval, lifecycle, lifecycleLabel } from '../api/lifecycle';
import ExperimentLifecycle from './ExperimentLifecycle';
import { Badge, ErrorNotice, PageHeader } from './ui';

interface RecordSelection { recordKey: string; name: string }
const ManagementContext = createContext<((record: RecordSelection) => void) | null>(null);
export function useHasRecordManagementScope() { return Boolean(useContext(ManagementContext)); }

/** A record stays selected after archiving removes its row from the active library. */
export function RecordManagementScope({ project, backLabel, children }: {
  project: string; backLabel: string; children: ReactNode;
}) {
  const [record, setRecord] = useState<RecordSelection | null>(null);
  const [locked, setLocked] = useState(false);
  const trigger = useRef<HTMLElement | null>(null);
  function close() {
    if (locked) return;
    setRecord(null);
    requestAnimationFrame(() => { if (trigger.current?.isConnected) trigger.current.focus(); });
  }
  useEffect(() => {
    const openLibrary = () => { if (!locked) setRecord(null); };
    window.addEventListener('histopilot:stage-library', openLibrary);
    return () => window.removeEventListener('histopilot:stage-library', openLibrary);
  }, [locked]);
  return <ManagementContext.Provider value={(next) => {
    if (locked) return;
    trigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setRecord(next);
  }}>
    {record ? <RecordManagementPage project={project} record={record} backLabel={backLabel} locked={locked} onClose={close} onLockChange={setLocked} /> : null}
    <div hidden={Boolean(record)}>{children}</div>
  </ManagementContext.Provider>;
}

function RecordManagementPage({ project, record, backLabel, locked, onClose, onLockChange }: {
  project: string; record: RecordSelection; backLabel: string; locked: boolean;
  onClose: () => void; onLockChange: (locked: boolean) => void;
}) {
  const page = useRef<HTMLDivElement>(null);
  const inventory = useQuery({ queryKey: ['cleanup', project], queryFn: () => lifecycle.inventory(project),
    refetchInterval: (query) => cleanupPollInterval(query.state.data) });
  const item = inventory.data?.items.find((candidate) => candidate.key === record.recordKey);
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      page.current?.scrollIntoView({ block: 'start', behavior: 'instant' });
      page.current?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(frame);
  }, [record.recordKey]);
  return <div className="clinical-workspace record-management-page" data-record-management ref={page} tabIndex={-1}>
      <PageHeader eyebrow="RECORD MANAGEMENT" title={`Manage ${item?.name ?? record.name}`} description="Archive this record, restore it, or move it to Trash."
        actions={<StageBackButton type="button" disabled={locked} onClick={onClose}>{backLabel}</StageBackButton>} />
      <ErrorNotice error={inventory.error} />
      {inventory.isPending ? <p role="status">Loading record status…</p> : item ? <>
        <div className="record-management-identity"><Badge tone={item.state === 'active' ? 'green' : 'neutral'}>{lifecycleLabel[item.state]}</Badge><span className="muted">{item.kind.replaceAll('-', ' ')}</span><code>{item.id}</code></div>
        <ExperimentLifecycle key={record.recordKey} project={project} recordKey={record.recordKey} state={item.state} name={item.name} onLockChange={onLockChange} />
      </> : !inventory.error ? <p role="status">This record is no longer available in this project.</p> : null}
      {inventory.error ? <button type="button" className="btn btn-secondary" onClick={() => void inventory.refetch()}>Retry loading record</button> : null}
    </div>;
}

export function RecordManageButton({ recordKey, name }: RecordSelection) {
  const open = useContext(ManagementContext);
  return <button type="button" className="btn btn-secondary btn-small" data-record-key={recordKey} aria-label={`Manage ${name}`}
    disabled={!open} onClick={() => open?.({ recordKey, name })}>Manage</button>;
}
