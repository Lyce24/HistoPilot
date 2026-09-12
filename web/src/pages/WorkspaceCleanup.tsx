import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { lifecycle, lifecycleLabel, cleanupPollInterval, cleanupJobActive, cleanupReviewMatches, cleanupApplyRequest } from '../api/lifecycle';
import type { CleanupAction, CleanupItem, CleanupPreview, LifecycleState } from '../api/lifecycle';
import type { Workspace } from '../api/types';
import { useHashParameters } from '../lib/hashRoute';
import { Badge, ErrorNotice, PageHeader, Panel } from '../components/ui';
import './WorkspaceCleanup.css';

const actionLabel: Record<CleanupAction, string> = { archive: 'Archive', trash: 'Move to Trash', restore: 'Restore to Active' };
const recordLabel: Record<CleanupItem['type'], string> = { project: 'Whole project', dataset: 'Dataset', configuration: 'Saved configuration', draft: 'Draft', packing: 'Packing job', extraction: 'Extraction job' };
const kindLabels: Record<string, string> = {
  project: 'Whole project', dataset: 'Dataset', protocol: 'Targets & splits', 'analysis-protocol': 'Targets & splits',
  feature: 'Feature inventory', 'feature-bundle': 'Feature bundle', 'mil-batch': 'Training batch',
  'development-batch': 'Training batch draft', 'mil-experiment': 'Saved experiment inputs',
  'model-experiment': 'Experiment', 'frozen-predictor': 'Predictor', 'predictor-refit': 'Refit training plan', 'model-evaluation': 'Evaluation', 'evaluation-batch': 'Evaluation batch',
  'clinical-analysis': 'Clinical utility report', 'model-interpretation': 'Model interpretation',
  'evaluation-cohort': 'Test cohort', 'dataset-import': 'Import draft', extraction: 'Feature extraction',
  'feature-validate': 'Feature validation', 'feature-validation': 'Feature validation',
  'feature-pack': 'Feature packing', 'feature-packing': 'Feature packing', packing: 'Feature packing',
};
const kindLabel = (kind: string) => kindLabels[kind] ?? kind.replaceAll('-', ' ').replace(/^./, (letter) => letter.toUpperCase());
const retentionNote = 'Delete moves records to recoverable Trash. Source CSV/H5 files, features, packs, model checkpoints and output files stay on disk. This does not free disk space.';
type Review = { preview: CleanupPreview; operationId: string; uncertain: boolean };
const readableError = (reason: unknown) => reason instanceof Error ? reason : new Error('The workspace could not be updated.');
const operation = () => `cleanup:${crypto.randomUUID()}`;

export function filterCleanupItems(items: CleanupItem[], state: LifecycleState, kind: string, search: string) {
  const text = search.trim().toLocaleLowerCase();
  return items.filter((item) => item.state === state && (!kind || item.kind === kind) && (!text || `${item.name} ${kindLabel(item.kind)} ${item.kind} ${item.id}`.toLocaleLowerCase().includes(text)));
}

export function bulkCleanupSelection(selected: string[], visible: CleanupItem[]) {
  const records = visible.filter((item) => item.type !== 'project');
  if (!records.length && visible.length === 1 && visible[0].type === 'project') return [visible[0].key];
  return [...new Set([...selected.filter((key) => !key.startsWith('project:')), ...records.map((item) => item.key)])];
}

function RecordNames({ keys, items }: { keys: string[]; items: CleanupItem[] }) {
  const byKey = new Map(items.map((item) => [item.key, item]));
  return <ul className="cleanup-names">{keys.map((key) => {
    const item = byKey.get(key);
    return <li key={key}><strong>{item?.name ?? key}</strong><span>{item ? `${recordLabel[item.type]} · ${kindLabel(item.kind)}` : 'Record not in current inventory'}</span></li>;
  })}</ul>;
}

export function CleanupReviewPanel({ preview, items, current, acknowledged, busy, uncertain, onAcknowledge, onApply, onInclude, onRefresh }: {
  preview: CleanupPreview; items: CleanupItem[]; current: boolean; acknowledged: boolean; busy: boolean; uncertain: boolean;
  onAcknowledge: (value: boolean) => void; onApply: () => void; onInclude: () => void; onRefresh: () => void;
}) {
  const allowed = current && preview.canApply && !preview.blockers.length && !preview.requiredKeys.some((key) => !preview.keys.includes(key)) && acknowledged;
  return <Panel title={`Review: ${actionLabel[preview.action]}`} subtitle={`${preview.recordCount} selected record${preview.recordCount === 1 ? '' : 's'}. Only the listed selection will change.`}>
    <RecordNames keys={preview.keys} items={items} />
    <p className="cleanup-retention">{retentionNote}</p>
    {preview.keys.some((key) => items.find((item) => item.key === key)?.type === 'project') ? <p className="callout">The whole-project selection changes project visibility only. Its child records keep their current states.</p> : null}
    {preview.note ? <p>{preview.note}</p> : null}
    {preview.blockers.length ? <div className="cleanup-blockers" role="alert"><h3>Resolve these blockers first</h3><ul>{preview.blockers.map((blocker, index) => <li key={`${blocker.code}:${index}`}>{blocker.message}{blocker.keys.length ? <RecordNames keys={blocker.keys} items={items} /> : null}</li>)}</ul></div> : null}
    {preview.requiredKeys.length ? <div className="cleanup-required"><h3>Related records also require attention</h3><RecordNames keys={preview.requiredKeys} items={items} /><p>These records have not been added to your selection.</p><button type="button" className="btn btn-secondary" disabled={busy || uncertain} onClick={onInclude}>Include required records and review again</button></div> : null}
    {uncertain ? <div className="callout callout-warning" role="alert"><p>The confirmation response was lost. The change may already have been applied. Retry sends the exact same reviewed request and operation ID.</p><div className="inline-actions"><button className="btn btn-primary" disabled={busy} onClick={onApply}>Retry this confirmation</button><button className="btn btn-secondary" disabled={busy} onClick={onRefresh}>Refresh records and review again</button></div></div>
      : !current ? <p className="callout callout-warning" role="status">The project or selection changed. Refresh and review again before confirming.</p>
        : preview.canApply && !preview.blockers.length ? <><label className="cleanup-ack"><input type="checkbox" checked={acknowledged} disabled={busy} onChange={(event) => onAcknowledge(event.target.checked)} />I reviewed these records and understand that their files will be retained.</label><button className={`btn ${preview.action === 'trash' ? 'cleanup-trash-button' : 'btn-primary'}`} disabled={busy || !allowed} onClick={onApply}>{busy ? 'Applying…' : `${actionLabel[preview.action]} (${preview.recordCount})`}</button></> : null}
  </Panel>;
}

export default function WorkspaceCleanup({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const client = useQueryClient();
  const inventory = useQuery({ queryKey: ['cleanup', project], queryFn: () => lifecycle.inventory(project), refetchInterval: (query) => cleanupPollInterval(query.state.data) });
  const [state, setState] = useState<LifecycleState>(workspace.project.lifecycleState ?? 'active');
  const [kind, setKind] = useState('');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<string[]>([]);
  const [action, setAction] = useState<CleanupAction>((workspace.project.lifecycleState ?? 'active') === 'active' ? 'archive' : 'restore');
  const [review, setReview] = useState<Review | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState('');
  const cancellationIds = useRef(new Map<string, string>());
  const requestedKey = useHashParameters().get('key');
  const openedKey = useRef<string | null>(null);
  const items = inventory.data?.items ?? [];
  const visible = filterCleanupItems(items, state, kind, search);
  const selectedSet = new Set(selected);
  const editable = !busy && !review?.uncertain;
  const kinds = [...new Set(items.map((item) => item.kind))].sort();
  const allowedActions: CleanupAction[] = state === 'active' ? ['archive', 'trash'] : state === 'archived' ? ['restore', 'trash'] : ['restore'];
  const effectiveAction = allowedActions.includes(action) ? action : allowedActions[0];
  const current = cleanupReviewMatches(review?.preview ?? null, inventory.data, effectiveAction, selected) && !inventory.isError;
  useEffect(() => {
    if (!requestedKey || openedKey.current === requestedKey || !inventory.data) return;
    openedKey.current = requestedKey;
    const item = inventory.data.items.find((item) => item.key === requestedKey);
    if (!item) { setError(new Error('The requested record is not in this project’s cleanup inventory.')); return; }
    setState(item.state); setKind(''); setSearch(''); setSelected([item.key]);
    setAction(item.state === 'active' ? 'archive' : 'restore');
    setReview(null); setAcknowledged(false);
  }, [requestedKey, inventory.data]);
  useEffect(() => {
    for (const item of inventory.data?.items ?? []) {
      if (!cleanupJobActive(item)) cancellationIds.current.delete(item.key);
    }
  }, [inventory.data]);

  function resetReview() { setReview(null); setAcknowledged(false); setError(null); setNotice(''); }
  function select(keys: string[]) { resetReview(); setSelected([...new Set(keys)]); }
  function changeState(next: LifecycleState) { select([]); setState(next); setAction(next === 'active' ? 'archive' : 'restore'); }
  async function refresh() {
    resetReview();
    await inventory.refetch();
  }
  async function requestPreview(keys = selected) {
    setBusy(true); resetReview();
    try {
      const preview = await lifecycle.preview(project, effectiveAction, keys);
      setReview({ preview, operationId: operation(), uncertain: false });
      await inventory.refetch();
    } catch (reason) { setError(readableError(reason)); }
    finally { setBusy(false); }
  }
  async function confirm() {
    if (!review || (!review.uncertain && (!current || !acknowledged)) || busy) return;
    setBusy(true); setError(null);
    try {
      const result = await lifecycle.apply(project, cleanupApplyRequest(review.preview, review.operationId));
      setReview(null); setAcknowledged(false); setSelected([]);
      const next = result.action === 'archive' ? 'archived' : result.action === 'trash' ? 'trashed' : 'active';
      setState(next); setAction(next === 'active' ? 'archive' : 'restore');
      setNotice(`${result.changed.length} record${result.changed.length === 1 ? '' : 's'} ${result.action === 'restore' ? 'restored to Active' : result.action === 'trash' ? 'moved to Trash' : 'archived'}. Files were retained.`);
      await client.invalidateQueries({ predicate: (query) => query.queryKey.includes(project) || query.queryKey[0] === 'projects' });
    } catch (reason) {
      if (!(reason instanceof ApiError)) setReview({ ...review, uncertain: true });
      else { setReview(null); setAcknowledged(false); void inventory.refetch(); }
      setError(readableError(reason));
    } finally { setBusy(false); }
  }
  async function cancel(item: CleanupItem) {
    setBusy(true); resetReview();
    const id = cancellationIds.current.get(item.key) ?? operation();
    cancellationIds.current.set(item.key, id);
    try {
      await lifecycle.cancel(project, item.key, id);
      cancellationIds.current.delete(item.key);
      setNotice(`Cancellation requested for ${item.name}. Status updates automatically; review cleanup again after the job stops.`);
      await inventory.refetch();
    } catch (reason) {
      if (reason instanceof ApiError) cancellationIds.current.delete(item.key);
      setError(readableError(reason));
    }
    finally { setBusy(false); }
  }
  return <div className="clinical-workspace cleanup-workspace">
    <PageHeader eyebrow="PROJECT TOOLS" title="Workspace cleanup" description="Archive work you want to keep, move unwanted records to Trash, or restore earlier work." actions={<button className="btn btn-secondary" disabled={busy} onClick={() => void refresh()}>Refresh records</button>} />
    <p className="callout cleanup-retention">{retentionNote}</p>
    <p className="cleanup-intro">Archived records remain available to existing workflows and are hidden from new selections. Trash is recoverable. Related records and active jobs are checked before every change.</p>
    <ErrorNotice error={error ?? inventory.error} />
    {notice ? <p className="callout" role="status">{notice}</p> : null}
    <section className="card" aria-label="Workspace records">
      <div className="cleanup-tabs" role="group" aria-label="Record state">{(['active', 'archived', 'trashed'] as const).map((value) => <button type="button" key={value} className={state === value ? 'selected' : ''} aria-pressed={state === value} disabled={!editable} onClick={() => changeState(value)}>{lifecycleLabel[value]} <span>{items.filter((item) => item.state === value).length}</span></button>)}</div>
      <div className="cleanup-filters"><label><span>Record kind</span><select className="field" value={kind} onChange={(event) => setKind(event.target.value)}><option value="">All kinds</option>{kinds.map((value) => <option key={value} value={value}>{kindLabel(value)}</option>)}</select></label><label><span>Find records</span><input className="field" type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name, kind or ID" /></label></div>
      {inventory.isPending ? <p className="panel-body" role="status">Loading project records…</p> : <>
        <div className="cleanup-selection"><span>{selected.length} selected{selected.some((key) => !visible.some((item) => item.key === key)) ? ' (including records outside this view)' : ''}</span><div className="inline-actions"><button className="text-button" disabled={!editable || !visible.length} onClick={() => select(bulkCleanupSelection(selected, visible))}>Select visible records</button><button className="text-button" disabled={!editable || !selected.length} onClick={() => select([])}>Clear selection</button></div></div>
        {visible.length ? <ul className="cleanup-records">{visible.map((item) => <li key={item.key} className={selectedSet.has(item.key) ? 'cleanup-selected' : ''}>
          <label className="cleanup-record-select"><input type="checkbox" checked={selectedSet.has(item.key)} disabled={!editable} onChange={(event) => select(event.target.checked ? item.type === 'project' ? [item.key] : [...selected.filter((key) => !key.startsWith('project:')), item.key] : selected.filter((key) => key !== item.key))} /><span><strong>{item.name}</strong><small>{recordLabel[item.type]} · {kindLabel(item.kind)}</small>{item.type === 'project' ? <small>Project visibility only; child records keep their states. Select separately from child records.</small> : null}<span className="cleanup-record-id">{item.id}</span></span></label>
          <div className="cleanup-record-meta"><Badge>{lifecycleLabel[item.state]}</Badge>{item.job ? <Badge tone={cleanupJobActive(item) ? 'warning' : 'neutral'}>{item.job.status}</Badge> : null}<small>{item.usedBy.length} dependent record{item.usedBy.length === 1 ? '' : 's'}</small>{item.job?.cancellable ? <button className="btn btn-secondary btn-small" disabled={!editable} aria-label={`Cancel job: ${item.name}`} onClick={() => void cancel(item)}>Cancel job</button> : null}</div>
        </li>)}</ul> : <p className="panel-body muted">No {lifecycleLabel[state].toLowerCase()} records match this view.</p>}
      </>}
      <div className="cleanup-actions"><label><span>For selected records</span><select className="field" value={effectiveAction} disabled={!editable} onChange={(event) => { resetReview(); setAction(event.target.value as CleanupAction); }}>{allowedActions.map((value) => <option key={value} value={value}>{actionLabel[value]}</option>)}</select></label><button className="btn btn-primary" disabled={!editable || !selected.length || inventory.isError || !inventory.data} onClick={() => void requestPreview()}>Review selected changes</button></div>
    </section>
    {review ? <CleanupReviewPanel preview={review.preview} items={items} current={current} acknowledged={acknowledged} busy={busy} uncertain={review.uncertain} onAcknowledge={setAcknowledged} onApply={() => void confirm()} onRefresh={() => void refresh()} onInclude={() => { const keys = [...new Set([...selected, ...review.preview.requiredKeys])]; select(keys); void requestPreview(keys); }} /> : null}
    {inventory.data?.audit.length ? <details className="card cleanup-audit"><summary>Recent cleanup history</summary><ul>{inventory.data.audit.slice(0, 10).map((entry, index) => <li key={`${entry.at}:${index}`}><time dateTime={entry.at}>{new Date(entry.at).toLocaleString()}</time><span>{entry.action ? actionLabel[entry.action as CleanupAction] ?? entry.action : 'Workspace records updated'}</span></li>)}</ul></details> : null}
  </div>;
}
