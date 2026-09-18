import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { operations } from '../api/operations';
import type { ArchiveAction, ArchiveJob, ArchiveRequest, SourceRegistration } from '../api/operations';
import { lifecycle } from '../api/lifecycle';
import type { Workspace } from '../api/types';
import { Badge, ErrorNotice, PageHeader, Panel } from '../components/ui';
import ServerFolderPicker from '../components/ServerFolderPicker';
import './LocalOperations.css';

const active = new Set(['starting', 'queued', 'running', 'cancelling']);
const readableError = (error: unknown) => error instanceof Error ? error : new Error('The operation could not be completed.');
const bytes = (value: number) => value < 1024 ** 2 ? `${(value / 1024).toFixed(1)} KB` : value < 1024 ** 3 ? `${(value / 1024 ** 2).toFixed(1)} MB` : `${(value / 1024 ** 3).toFixed(2)} GB`;

export async function cancelPipelineJob(project: string, key: string, receipts: Map<string, string>) {
  const operationId = receipts.get(key) ?? `operations-cancel:${crypto.randomUUID()}`;
  receipts.set(key, operationId);
  const result = await lifecycle.cancel(project, key, operationId);
  // Keep an ID only while acknowledgement is uncertain. A later resumed run
  // needs a fresh cancellation receipt even though its record key is unchanged.
  receipts.delete(key);
  return result;
}

function ArchiveReceipt({ job, busy, onAction }: { job: ArchiveJob; busy: boolean; onAction: (job: string, action: 'cancel' | 'retry') => void }) {
  return <article className="operations-receipt">
    <div className="operations-receipt-heading"><strong>{job.action === 'export' ? 'Project export' : job.action === 'verify' ? 'Archive verification' : 'Project restore'}</strong><Badge tone={job.status === 'failed' ? 'danger' : job.status === 'completed' ? 'success' : 'neutral'}>{job.status}</Badge></div>
    <p className="muted">{new Date(job.createdAt).toLocaleString()}</p>
    {job.error ? <p role="alert">{job.error}</p> : null}
    {job.progress && active.has(job.status) ? <div role="status"><p>{job.progress.stage} · {job.progress.completed}/{job.progress.total} files</p><progress aria-label="Archive progress" value={job.progress.completed} max={job.progress.total || 1} /><code className="operations-path">{job.progress.file}</code></div> : null}
    {active.has(job.status) && job.status !== 'cancelling' ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => onAction(job.id, 'cancel')}>Cancel archive operation</button> : ['failed', 'cancelled', 'interrupted'].includes(job.status) ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => onAction(job.id, 'retry')}>Retry saved operation</button> : null}
    {job.result ? <><p>{job.result.verified ? 'Checksums verified' : 'Not verified'} · {job.result.fileCount.toLocaleString()} files · {bytes(job.result.totalBytes)}</p><code className="operations-path">{job.result.destinationPath ?? job.result.archivePath}</code>{job.result.note ? <p className="callout">{job.result.note}</p> : null}{job.result.externalSources?.missingReferences.length ? <p>{job.result.externalSources.missingReferences.length} unavailable external references were recorded at export.</p> : null}</> : null}
    <details><summary>Worker details</summary><p>Session: <code>{job.sessionName}</code></p><p>Reconnect: <code>tmux attach -t {job.sessionName}</code></p><p className="operations-path">Log: {job.logPath}</p></details>
  </article>;
}

function SourceRow({ project, source, onSaved }: { project: string; source: SourceRegistration; onSaved: (note: string) => void }) {
  const [replacement, setReplacement] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  async function relink() {
    setBusy(true); setError(null);
    try { const result = await operations.relink(project, source.id, source.path, replacement.trim()); setReplacement(''); onSaved(result.note); }
    catch (reason) { setError(readableError(reason)); }
    finally { setBusy(false); }
  }
  return <article className="operations-source"><div className="operations-receipt-heading"><strong>{source.name} · {source.role}</strong><Badge tone={source.available ? 'success' : 'warning'}>{source.available ? 'Available' : source.permitted ? 'Missing' : 'Outside configured roots'}</Badge></div><code className="operations-path">{source.path}</code>
    <details><summary>Relink source registration</summary><p>Choose the moved source folder. New imports will use this registration; existing frozen versions keep their recorded paths.</p><label className="operations-field">Replacement folder<input value={replacement} disabled={busy} onChange={(event) => setReplacement(event.target.value)} placeholder="/path/to/moved/source" /></label><div className="inline-actions"><ServerFolderPicker onSelect={setReplacement} label="Choose replacement" /><button type="button" className="btn btn-secondary" disabled={busy || !replacement.trim() || replacement.trim() === source.path} onClick={() => void relink()}>{busy ? 'Saving…' : 'Save source registration'}</button></div><ErrorNotice error={error} /></details>
  </article>;
}

export default function LocalOperations({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const client = useQueryClient();
  const inventory = useQuery({ queryKey: ['operations', project], queryFn: () => operations.inventory(project), refetchInterval: 5000 });
  const sources = useQuery({ queryKey: ['operation-sources', project], queryFn: () => operations.sources(project), staleTime: 30_000 });
  const archives = useQuery({ queryKey: ['operation-archives', project], queryFn: () => operations.archives(project), refetchInterval: (query) => query.state.data?.jobs.some((job) => active.has(job.status)) ? 3000 : 15_000 });
  const [action, setAction] = useState<ArchiveAction>('export');
  const [archivePath, setArchivePath] = useState('');
  const [destination, setDestination] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState('');
  const submissions = useRef(new Map<string, ArchiveRequest>());
  const cancellations = useRef(new Map<string, string>());
  const jobs = inventory.data?.jobs ?? [];
  const activeJobs = jobs.filter((job) => active.has(job.job.status) || job.job.busy);
  async function submit() {
    setBusy(true); setError(null); setNotice('');
    const values = { action, archivePath: archivePath.trim(), ...(action === 'restore' ? { destinationPath: destination.trim() } : {}) };
    const key = JSON.stringify(values);
    const input = submissions.current.get(key) ?? { ...values, operationId: `archive:${crypto.randomUUID()}` };
    submissions.current.set(key, input);
    try {
      const job = await operations.submit(project, input);
      if (!active.has(job.status)) submissions.current.delete(key);
      if (job.status === 'failed' || job.status === 'interrupted') setError(new Error(job.error ?? 'The archive worker could not complete the operation.'));
      else setNotice(job.status === 'completed' ? 'Archive operation completed and verified.' : job.status === 'cancelled' ? 'Archive operation was cancelled.' : 'Archive operation saved. You can leave this page while its worker runs.');
      await client.invalidateQueries({ queryKey: ['operation-archives', project] });
    } catch (reason) { setError(readableError(reason)); void archives.refetch(); }
    finally { setBusy(false); }
  }
  async function cancel(key: string) {
    setBusy(true); setError(null);
    try { await cancelPipelineJob(project, key, cancellations.current); await inventory.refetch(); }
    catch (reason) { setError(readableError(reason)); }
    finally { setBusy(false); }
  }
  function sourceSaved(note: string) {
    setNotice(note);
    void client.invalidateQueries({ queryKey: ['operation-sources', project] });
    void client.invalidateQueries({ queryKey: ['workspace', project] });
  }
  async function archiveAction(job: string, task: 'cancel' | 'retry') {
    setBusy(true); setError(null);
    try { await operations[task](project, job); await archives.refetch(); }
    catch (reason) { setError(readableError(reason)); }
    finally { setBusy(false); }
  }
  return <div className="local-operations">
    <PageHeader eyebrow="Project operations" title="Jobs, backups & sources" description="Follow work across the pipeline, verify project archives, and reconnect moved source folders." actions={<button type="button" className="btn btn-secondary" onClick={() => { void inventory.refetch(); void archives.refetch(); void sources.refetch(); }}>Refresh</button>} />
    <ErrorNotice error={error ?? inventory.error ?? sources.error ?? archives.error} />
    {notice ? <p className="callout" role="status">{notice}</p> : null}
    <Panel title="Unified job queue" subtitle="Extraction, packing, training, inference and interpretation use shared resource reservations.">
      {inventory.isPending ? <p role="status">Loading jobs and reservations…</p> : null}
      {inventory.data ? <p>{activeJobs.length} active jobs in this project · {inventory.data.capacity.cpus} host CPU slots · {inventory.data.capacity.availableRamGb.toFixed(1)} GB RAM currently available</p> : null}
      {jobs.length ? <div className="operations-table-wrap"><table className="operations-table"><thead><tr><th>Job</th><th>Type</th><th>Status</th><th>Action</th></tr></thead><tbody>{jobs.map((job) => <tr key={job.key}><td><a href={`#cleanup?key=${encodeURIComponent(job.key)}`}>{job.name}</a></td><td>{job.kind.replaceAll('-', ' ')}</td><td><Badge>{job.job.status}</Badge>{job.job.waitingReason ? <p>{job.job.waitingReason}</p> : null}</td><td>{job.job.cancellable ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => void cancel(job.key)}>Cancel job</button> : <span className="muted">—</span>}</td></tr>)}</tbody></table></div> : !inventory.isPending && !inventory.isError ? <p className="muted">No jobs yet. Start feature preparation or an experiment to see its progress here.</p> : null}
      {inventory.data?.reservations.length ? <details><summary>Host reservations across all projects ({inventory.data.reservations.length})</summary><ul>{inventory.data.reservations.map((reservation, index) => <li key={`${reservation.runId}:${reservation.gpu}:${index}`}>{reservation.kind ?? 'Compute'} · {reservation.cpus} CPU slots · {reservation.ramGb} GB RAM · {reservation.gpu === null ? 'CPU execution' : `GPU ${reservation.gpu}`}<code className="operations-path">{reservation.runId}</code></li>)}</ul></details> : null}
      <p className="muted">Reservations coordinate requested capacity; they do not impose operating-system memory limits. Older already-running preparation jobs keep their original scheduling.</p>
    </Panel>
    <Panel title="Verified project archives" subtitle="Save the project database, frozen records, review notes, logs and project-contained outputs with checksums.">
      <p className="callout">External slides, feature folders and outputs remain external references. The archive records unavailable references. Export after active jobs finish; restoring preserves the project identity and never overwrites an existing folder.</p>
      <div className="operations-form"><label className="operations-field">Operation<select value={action} disabled={busy} onChange={(event) => setAction(event.target.value as ArchiveAction)}><option value="export">Export project</option><option value="verify">Verify archive</option><option value="restore">Restore archive</option></select></label>
        <label className="operations-field">{action === 'export' ? 'New archive file outside the project' : 'Existing archive file'}<input value={archivePath} disabled={busy} onChange={(event) => setArchivePath(event.target.value)} placeholder="/storage/backups/study.zip" /></label>
        <ServerFolderPicker purpose="storage" selection={action === 'export' ? 'folder' : 'file'} label={action === 'export' ? 'Choose export folder' : 'Choose archive'} onSelect={(path) => setArchivePath(action === 'export' ? `${path}/histopilot-${project.slice(-8)}-${Date.now()}.zip` : path)} />
        {action === 'restore' ? <label className="operations-field">New restore folder<input value={destination} disabled={busy} onChange={(event) => setDestination(event.target.value)} placeholder="/storage/restored-study" /></label> : null}
        <button type="button" className="btn btn-primary" disabled={busy || !archivePath.trim() || (action === 'restore' && !destination.trim()) || (action === 'export' && (inventory.isPending || inventory.isError || activeJobs.length > 0))} onClick={() => void submit()}>{busy ? 'Submitting…' : action === 'export' ? 'Export & verify project' : action === 'verify' ? 'Verify every file' : 'Restore to new folder'}</button>
      </div>
      {archives.isPending ? <p role="status">Loading archive history…</p> : null}
      <div className="operations-receipts">{archives.data?.jobs.map((job) => <ArchiveReceipt key={job.id} job={job} busy={busy} onAction={(id, task) => void archiveAction(id, task)} />)}</div>
    </Panel>
    <Panel title="Source health" subtitle="Check registered folders and absolute references recorded in frozen datasets and configurations.">
      {sources.isPending ? <p role="status">Checking source references…</p> : null}
      {sources.data ? <><p>{sources.data.referenceCount.toLocaleString()} recorded paths checked · {sources.data.missingReferences.length.toLocaleString()} unavailable external references</p>{sources.data.sources.map((source) => <SourceRow key={`${project}:${source.id}:${source.path}`} project={project} source={source} onSaved={sourceSaved} />)}{!sources.data.sources.length ? <p className="muted">No source folders are registered with this project.</p> : null}{sources.data.missingReferences.length ? <details><summary>Unavailable external references</summary><ul>{sources.data.missingReferences.slice(0, 200).map((item) => <li key={item.path}><code className="operations-path">{item.path}</code><span>{item.reason === 'missing' ? 'Missing from this machine' : 'Outside configured roots'}</span></li>)}</ul>{sources.data.missingReferences.length > 200 ? <p>Showing the first 200 paths. The complete inventory is included in exported archives.</p> : null}</details> : null}</> : null}
    </Panel>
  </div>;
}
