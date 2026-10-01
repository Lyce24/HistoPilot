import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { operations } from '../api/operations';
import type { ArchiveAction, ArchiveJob, ArchiveRequest, SourceRegistration } from '../api/operations';
import type { Workspace } from '../api/types';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import ServerFolderPicker from '../components/ServerFolderPicker';
import RunStatusChip from '../components/RunStatusChip';
import LegacyRecordNote, { createdBeforeTaskCenter } from '../components/LegacyRecordNote';
import { taskCenterHref, taskCenterWork } from '../api/taskCenter';
import AgentAccess from '../components/AgentAccess';
import './LocalOperations.css';

const active = new Set(['starting', 'queued', 'running', 'cancelling']);
const readableError = (error: unknown) => error instanceof Error ? error : new Error('The operation could not be completed.');
const bytes = (value: number) => value < 1024 ** 2 ? `${(value / 1024).toFixed(1)} KB` : value < 1024 ** 3 ? `${(value / 1024 ** 2).toFixed(1)} MB` : `${(value / 1024 ** 3).toFixed(2)} GB`;

function ArchiveReceipt({ job, busy, project, onRetry, onSettled }: { job: ArchiveJob; busy: boolean; project: string; onRetry: (job: string) => void; onSettled: () => void }) {
  // Task Center operations show their run status and leave cancel, logs and attempts to it;
  // operations created before the Task Center are read-only receipts.
  const managed = !createdBeforeTaskCenter(job);
  const retry = ['failed', 'cancelled', 'interrupted'].includes(job.status) ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => onRetry(job.id)}>Resume</button> : null;
  if (managed) return <article className="operations-receipt">
    <div className="operations-receipt-heading"><strong>{job.action === 'export' ? 'Project export' : job.action === 'verify' ? 'Archive verification' : 'Project restore'}</strong></div>
    <p className="muted">{new Date(job.createdAt).toLocaleString()}</p>
    <RunStatusChip scope={job.ownerKey ? { owner: job.ownerKey } : { ownerKind: 'archive', ownerId: job.id, project }} variant="chip" primaryAction={retry} notStartedText={job.status} onSettled={onSettled} />
    {job.progress && active.has(job.status) ? <div><p>{job.progress.stage} · {job.progress.completed}/{job.progress.total} files</p><progress aria-label="Archive progress" value={job.progress.completed} max={job.progress.total || 1} /></div> : null}
    {job.result ? <><p>{job.result.verified ? 'Checksums verified' : 'Not verified'} · {job.result.fileCount.toLocaleString()} files · {bytes(job.result.totalBytes)}</p><code className="operations-path">{job.result.destinationPath ?? job.result.archivePath}</code>{job.result.note ? <p className="callout">{job.result.note}</p> : null}</> : null}
  </article>;
  return <article className="operations-receipt">
    <div className="operations-receipt-heading"><strong>{job.action === 'export' ? 'Project export' : job.action === 'verify' ? 'Archive verification' : 'Project restore'}</strong><Badge tone={job.status === 'failed' ? 'orange' : job.status === 'completed' ? 'success' : 'neutral'}>{job.status}</Badge></div>
    <p className="muted">{new Date(job.createdAt).toLocaleString()}</p>
    {job.error ? <p role="alert">{job.error}</p> : null}
    {job.result ? <><p>{job.result.verified ? 'Checksums verified' : 'Not verified'} · {job.result.fileCount.toLocaleString()} files · {bytes(job.result.totalBytes)}</p><code className="operations-path">{job.result.destinationPath ?? job.result.archivePath}</code>{job.result.note ? <p className="callout">{job.result.note}</p> : null}{job.result.externalSources?.missingReferences.length ? <p>{job.result.externalSources.missingReferences.length} unavailable external references were recorded at export.</p> : null}</> : null}
    <LegacyRecordNote />
    <details><summary>Worker details</summary><p className="operations-path">Log: {job.logPath}</p></details>
  </article>;
}

/**
 * Active project jobs never block an export: the service queues it, and its worker waits in the
 * Task Center until the project is idle.
 */
export const archiveSubmitDisabled = (action: ArchiveAction, archivePath: string, destination: string, busy: boolean) =>
  busy || !archivePath.trim() || (action === 'restore' && !destination.trim());

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
  const [action, setAction] = useState<ArchiveAction>('export');
  // The inventory only informs Export, which the Task Center holds until the project's active
  // jobs finish: read it for export, often while jobs run and rarely otherwise.
  const inventory = useQuery({ queryKey: ['operations', project], queryFn: () => operations.inventory(project), enabled: action === 'export', refetchInterval: (query) => query.state.data?.jobs.some((job) => active.has(job.job.status) || job.job.busy) ? 10_000 : 60_000 });
  const sources = useQuery({ queryKey: ['operation-sources', project], queryFn: () => operations.sources(project), staleTime: 30_000 });
  const archives = useQuery({ queryKey: ['operation-archives', project], queryFn: () => operations.archives(project), refetchInterval: (query) => query.state.data?.jobs.some((job) => active.has(job.status)) ? 10_000 : 30_000 });
  const [archivePath, setArchivePath] = useState('');
  const [destination, setDestination] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState('');
  const submissions = useRef(new Map<string, ArchiveRequest>());
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
      else setNotice(job.status === 'completed' ? 'Archive operation completed and verified.' : job.status === 'cancelled' ? 'Archive operation was cancelled.' : job.action === 'export' ? 'Export saved. It starts in the Task Center once this project has no running jobs. You can leave this page.' : 'Archive operation saved. You can leave this page while its worker runs.');
      await client.invalidateQueries({ queryKey: ['operation-archives', project] });
    } catch (reason) { setError(readableError(reason)); void archives.refetch(); }
    finally { setBusy(false); }
  }
  function sourceSaved(note: string) {
    setNotice(note);
    void client.invalidateQueries({ queryKey: ['operation-sources', project] });
    void client.invalidateQueries({ queryKey: ['workspace', project] });
  }
  async function retryArchive(job: string) {
    setBusy(true); setError(null);
    try { await operations.retry(project, job); await archives.refetch(); }
    catch (reason) { setError(readableError(reason)); }
    finally { setBusy(false); }
  }
  return <div className="local-operations">
    <PageHeader eyebrow="Project operations" title="Study backups & sources" description="Verify project archives and reconnect moved source folders." actions={<button type="button" className="btn btn-secondary" onClick={() => { void inventory.refetch(); void archives.refetch(); void sources.refetch(); }}>Refresh</button>} />
    <ErrorNotice error={error ?? inventory.error ?? sources.error ?? archives.error} />
    {notice ? <p className="callout" role="status">{notice}</p> : null}
    <p className="callout operations-task-center" role="note"><Icon name="clock" size={16} /><span>{taskCenterWork(true)}, including the archive operations on this page, run in the <a href={taskCenterHref({ project })}>Task Center</a>, which orders and tracks them for every project on this machine.</span></p>
    <Panel title="Verified project archives" subtitle="Save the project database, frozen records, review notes, logs and project-contained outputs with checksums.">
      <p className="callout">External slides, feature folders and outputs remain external references. The archive records unavailable references. You can submit an export at any time: it waits in the Task Center until this project's running jobs finish. Restoring preserves the project identity and never overwrites an existing folder.</p>
      <div className="operations-form"><label className="operations-field">Operation<select value={action} disabled={busy} onChange={(event) => setAction(event.target.value as ArchiveAction)}><option value="export">Export project</option><option value="verify">Verify archive</option><option value="restore">Restore archive</option></select></label>
        <label className="operations-field">{action === 'export' ? 'New archive file outside the project' : 'Existing archive file'}<input value={archivePath} disabled={busy} onChange={(event) => setArchivePath(event.target.value)} placeholder="/storage/backups/study.zip" /></label>
        <ServerFolderPicker purpose="storage" selection={action === 'export' ? 'folder' : 'file'} label={action === 'export' ? 'Choose export folder' : 'Choose archive'} onSelect={(path) => setArchivePath(action === 'export' ? `${path}/histopilot-${project.slice(-8)}-${Date.now()}.zip` : path)} />
        {action === 'restore' ? <label className="operations-field">New restore folder<input value={destination} disabled={busy} onChange={(event) => setDestination(event.target.value)} placeholder="/storage/restored-study" /></label> : null}
        <button type="button" className="btn btn-primary" disabled={archiveSubmitDisabled(action, archivePath, destination, busy)} onClick={() => void submit()}>{busy ? 'Submitting…' : action === 'export' ? 'Export & verify project' : action === 'verify' ? 'Verify every file' : 'Restore to new folder'}</button>
      </div>
      {action === 'export' ? inventory.isPending ? <p className="muted" role="status">Checking this project's active jobs…</p> : activeJobs.length ? <p className="muted" role="status">{activeJobs.length} project job{activeJobs.length === 1 ? ' is' : 's are'} still active. An export submitted now waits until {activeJobs.length === 1 ? 'it finishes' : 'they finish'}, then starts by itself. Follow them in the <a href={taskCenterHref({ project })}>Task Center</a>.</p> : null : null}
      {archives.isPending ? <p role="status">Loading archive history…</p> : null}
      <div className="operations-receipts">{archives.data?.jobs.map((job) => <ArchiveReceipt key={job.id} job={job} busy={busy} project={project} onRetry={(id) => void retryArchive(id)} onSettled={() => void archives.refetch()} />)}</div>
    </Panel>
    <Panel title="Source health" subtitle="Check registered folders and absolute references recorded in frozen datasets and configurations.">
      {sources.isPending ? <p role="status">Checking source references…</p> : null}
      {sources.data ? <><p>{sources.data.referenceCount.toLocaleString()} recorded paths checked · {sources.data.missingReferences.length.toLocaleString()} unavailable external references</p>{sources.data.sources.map((source) => <SourceRow key={`${project}:${source.id}:${source.path}`} project={project} source={source} onSaved={sourceSaved} />)}{!sources.data.sources.length ? <p className="muted">No source folders are registered with this project.</p> : null}{sources.data.missingReferences.length ? <details><summary>Unavailable external references</summary><ul>{sources.data.missingReferences.slice(0, 200).map((item) => <li key={item.path}><code className="operations-path">{item.path}</code><span>{item.reason === 'missing' ? 'Missing from this machine' : 'Outside configured roots'}</span></li>)}</ul>{sources.data.missingReferences.length > 200 ? <p>Showing the first 200 paths. The complete inventory is included in exported archives.</p> : null}</details> : null}</> : null}
    </Panel>
    {workspace.mode === 'local' ? <AgentAccess project={project} projectName={workspace.project.name} /> : null}
  </div>;
}
