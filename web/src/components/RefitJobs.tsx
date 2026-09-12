import { useState } from 'react';
import { predictors, computeActive, computeStatusLabel, type RefitBuild, type FrozenPredictor } from '../api/predictors';
import { predictorSourceKey } from '../api/predictorBuilds';
import { shortRecordId } from '../lib/recordLabels';
import { cleanupLink } from '../lib/hashRoute';
import { Badge, ErrorNotice } from './ui';
import './RunWorkspace.css';

type Action = 'launch' | 'resume' | 'cancel' | 'publish';
type Submission = { id: string; name: string; operation: string; done: boolean; error?: string };
export default function RefitJobs({ project, builds, published, onOpen, refresh }: { project: string; builds: RefitBuild[]; published: FrozenPredictor[]; onOpen: (id: string) => void; refresh: () => Promise<void> }) {
  const [selected, setSelected] = useState<string[]>([]);
  const [search, setSearch] = useState('');
  const [visibility, setVisibility] = useState('active');
  const [submission, setSubmission] = useState<{ action: Action; items: Submission[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const publishedSources = new Set(published.filter((item) => item.manifest.method === 'refit').map((item) => predictorSourceKey(item.manifest)));
  const isPublished = (item: RefitBuild) => publishedSources.has(predictorSourceKey(item.manifest));
  const visible = builds.filter((item) => (visibility === 'all' || item.lifecycleState === visibility) && `${item.manifest.name} ${item.manifest.trainingSeed} ${item.manifest.splitSeed}`.toLowerCase().includes(search.toLowerCase()));
  const selectable = builds.filter((item) => selected.includes(item.id));
  const eligible = (action: Action) => selectable.filter((item) => action === 'cancel' ? computeActive(item.execution) && !item.execution?.cancellationRequested : item.lifecycleState === 'active' && !isPublished(item) && (action === 'launch' ? !item.execution || item.execution.status === 'not_started' : action === 'publish' ? item.execution?.status === 'completed' : ['failed', 'interrupted', 'cancelled'].includes(item.execution?.status ?? '')));
  const unfinished = submission?.items.some((item) => !item.done) ?? false;
  async function submit(action: Action, retry = false) {
    if (busy) return;
    const plan = retry && submission ? submission : { action, items: eligible(action).map((item) => ({ id: item.id, name: item.manifest.name, operation: crypto.randomUUID(), done: false })) };
    if (!plan.items.length) return;
    setSubmission(plan); setBusy(true); setError(null);
    const items: Submission[] = plan.items.map((item) => ({ ...item }));
    for (const item of items) {
      if (item.done) continue;
      try {
        if (plan.action === 'publish') await predictors.publishRefit(project, item.id, item.operation);
        else await predictors.refitJob(project, item.id, plan.action, item.operation);
        item.done = true; delete item.error;
      } catch (reason) { item.error = reason instanceof Error ? reason.message : 'Request failed.'; }
      setSubmission({ action: plan.action, items: items.map((row) => ({ ...row })) });
    }
    try { await refresh(); } catch { setError(new Error('Actions were submitted. Refresh to see the latest job states.')); }
    finally { setBusy(false); }
  }
  return <div className="run-workspace"><ErrorNotice error={error} />
    <div className="run-toolbar"><label className="label run-search">Search refit jobs<input type="search" className="field" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name or seed" /></label><label className="label">Job visibility<select className="field" value={visibility} onChange={(event) => setVisibility(event.target.value)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label></div>
    <div className="run-selection-bar"><strong>{selectable.length} refits selected</strong><button className="text-button" disabled={busy || unfinished} onClick={() => setSelected(visible.filter((item) => item.lifecycleState === 'active' && !isPublished(item)).map((item) => item.id))}>Select all unpublished refits</button><button className="text-button" disabled={busy || unfinished} onClick={() => setSelected([])}>Clear selection</button></div>
    <div className="inline-actions">{([['launch','Train selected'],['resume','Resume selected'],['cancel','Cancel selected'],['publish','Publish completed']] as const).map(([action,label]) => <button key={action} className={`btn ${action === 'launch' ? 'btn-primary' : 'btn-secondary'}`} disabled={busy || unfinished || !eligible(action).length} onClick={() => void submit(action)}>{label} ({eligible(action).length})</button>)}</div>
    {submission ? <div className="callout" role="status"><p>{submission.items.filter((item) => item.done).length}/{submission.items.length} {submission.action} requests accepted. Job completion is shown below.</p>{submission.items.filter((item) => item.error).map((item) => <p key={item.id}>{item.name}: {item.error}</p>)}{unfinished ? <div className="inline-actions"><button className="btn btn-secondary" disabled={busy} onClick={() => void submit(submission.action, true)}>Retry unfinished requests</button><button className="text-button" disabled={busy} onClick={() => setSubmission(null)}>Close request summary</button></div> : null}</div> : null}
    <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-check"><span className="sr-only">Select</span></th><th className="run-name">Refit plan</th><th>Batch / configuration</th><th className="run-number">Training seed</th><th className="run-number">Split seed</th><th className="run-number">Epochs</th><th>Status</th><th>Actions</th></tr></thead><tbody>{visible.map((item) => <tr key={item.id}><td><input type="checkbox" checked={selected.includes(item.id)} disabled={busy || unfinished} aria-label={`Select refit ${item.manifest.name}`} onChange={(event) => setSelected(event.target.checked ? [...selected,item.id] : selected.filter((id) => id !== item.id))} /></td><td><button className="text-button" onClick={() => onOpen(item.id)}>{item.manifest.name}</button><small>{item.lifecycleState}</small></td><td title={`${item.manifest.batchId} / ${item.manifest.candidateId}`}>{shortRecordId(item.manifest.batchId)}<small>{shortRecordId(item.manifest.candidateId)} · {item.manifest.recipe.model}</small></td><td>{item.manifest.trainingSeed}</td><td>{item.manifest.splitSeed}</td><td>{item.manifest.epochBudget?.epochs ?? '—'}<small>P{item.manifest.epochBudget?.percentile ?? '—'}</small></td><td><Badge>{isPublished(item) ? 'Predictor published' : computeStatusLabel(item.execution)}</Badge></td><td><a href={cleanupLink(item.id)}>Manage record</a></td></tr>)}</tbody></table></div>
    {!visible.length ? <p>No refit jobs in this view. Choose Refit or Both in Build predictors.</p> : null}
  </div>;
}
