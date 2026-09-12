import { shortRecordId } from '../lib/recordLabels';
import { useRef, useState } from 'react';
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { experiments, experimentPollInterval, experimentStage, experimentStageLabel, experimentStatusLabel } from '../api/experiments';
import type { CreateExperimentInput, ModelExperiment } from '../api/experiments';
import type { LifecycleState } from '../api/lifecycle';
import { lifecycleLabel } from '../api/lifecycle';
import { sameJSON } from '../lib/json';
import { Badge, ErrorNotice, PageHeader, Panel } from './ui';
import ExperimentLifecycle from './ExperimentLifecycle';
import './ExperimentRegistry.css';

export function filterExperiments<T extends Pick<ModelExperiment, 'id' | 'name' | 'notes' | 'tags' | 'state' | 'status' | 'createdAt' | 'updatedAt' | 'stage' | 'configurationLocked'>>(items: T[], state: LifecycleState | 'all', status: string, search: string, sort: string) {
  const query = search.trim().toLocaleLowerCase();
  return items.filter((item) => (state === 'all' || item.state === state) && (!status || (['planning', 'running', 'finished'].includes(status) ? experimentStage(item) === status : item.status === status))
    && (!query || [item.name, item.id, item.notes, ...item.tags].join(' ').toLocaleLowerCase().includes(query)))
    .sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name) || a.id.localeCompare(b.id)
      : sort === 'oldest' ? a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id)
        : b.updatedAt.localeCompare(a.updatedAt) || a.id.localeCompare(b.id));
}

function flatten(value: unknown, path = '', result: Record<string, unknown> = {}): Record<string, unknown> {
  if (Array.isArray(value) && value.some((item) => item !== null && typeof item === 'object')) {
    value.forEach((child, index) => flatten(child, `${path}[${index}]`, result));
  } else if (value !== null && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length) {
    for (const [key, child] of Object.entries(value)) flatten(child, path ? `${path}.${key}` : key, result);
  } else result[path] = value;
  return result;
}
export function comparisonSnapshot(item: ModelExperiment, batchId: string): Record<string, unknown> {
  const plan = batchId.startsWith('plan:') && experimentStage(item) === 'planning' ? item.batchPlans?.find((candidate) => candidate.id === batchId.slice(5)) : undefined;
  if (plan) return { ...(item.inputSnapshot ?? {}), inputs: item.inputs, batch: {
    recipe: plan.spec.recipe, mode: plan.spec.mode, grid: plan.spec.grid,
    configurations: plan.spec.configurations, trainingSeeds: plan.spec.trainingSeeds,
    resources: plan.spec.resources,
  } };
  const batch = item.batches.find((candidate) => candidate.id === batchId);
  if (!batch) return { inputs: item.inputs, ...(item.inputSnapshot ?? {}) };
  return { ...(batch.inputSnapshot ?? {}), inputs: batch.manifest.spec.inputs, batch: {
    recipe: batch.manifest.spec.recipe, mode: batch.manifest.spec.mode, grid: batch.manifest.spec.grid,
    configurations: batch.manifest.configurations.map((candidate) => candidate.recipe),
    trainingSeeds: batch.manifest.spec.trainingSeeds, resources: batch.manifest.spec.resources,
    splitPlans: batch.manifest.splitPlans, resolvedInputs: batch.manifest.resolvedInputs,
  } };
}
export function experimentDifferences(values: Record<string, unknown>[], differencesOnly: boolean) {
  const flat = values.map((value) => flatten(value));
  return [...new Set(flat.flatMap((value) => Object.keys(value)))].sort().map((path) => ({
    path, values: flat.map((value) => value[path]),
    different: flat.some((value) => !sameJSON(flat[0]?.[path], value[path])),
  })).filter((row) => !differencesOnly || row.different);
}
const displayValue = (value: unknown) => value === undefined ? 'Not recorded' : value === null ? 'None' : typeof value === 'string' ? value : JSON.stringify(value, null, 2);
export function comparisonFieldLabel(path: string) {
  const labels: Record<string, string> = { batch: 'Batch', configurations: 'Configuration', recipe: 'Training recipe',
    dataset: 'Dataset', protocol: 'Targets & splits', featureBundle: 'Feature bundle', features: 'Features',
    inputs: 'Input selection', resolvedInputs: 'Resolved inputs', contentHash: 'Content hash', spec: 'Settings',
    splitPlans: 'Split plan', learningRate: 'Learning rate', learningRates: 'Learning rates', weightDecay: 'Weight decay',
    trainingSeeds: 'Training seeds', resources: 'Resources', id: 'ID', featureSetId: 'Feature inventory ID',
    bundleId: 'Feature bundle ID', protocolId: 'Protocol ID', featureBundleId: 'Feature bundle ID',
  };
  return path.split('.').map((part) => {
    const match = /^(.*)\[(\d+)\]$/.exec(part);
    const field = match?.[1] ?? part;
    const label = labels[field] ?? field.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, (letter) => letter.toUpperCase());
    return match ? `${label} ${Number(match[2]) + 1}` : label;
  }).join(' / ');
}

export function ExperimentComparison({ items }: { items: ModelExperiment[] }) {
  const [baseline, setBaseline] = useState(items[0]?.id ?? '');
  const [batchIds, setBatchIds] = useState<Record<string, string>>({});
  const [differencesOnly, setDifferencesOnly] = useState(true);
  const ordered = [...items].sort((a, b) => a.id === baseline ? -1 : b.id === baseline ? 1 : 0);
  const chosenBatch = (item: ModelExperiment) => {
    const selected = batchIds[item.id] ?? '';
    return selected.startsWith('plan:') && experimentStage(item) !== 'planning' ? '' : selected;
  };
  const rows = experimentDifferences(ordered.map((item) => comparisonSnapshot(item, chosenBatch(item))), differencesOnly);
  return <Panel title="Compare experiment inputs" subtitle="Compare saved inputs, editable batch recipes or submitted batch snapshots. Training differences include saved recipe fields, configurations, seeds and resource settings.">
    <div className="experiment-compare-controls"><label className="label">Baseline<select className="field" value={items.some((item) => item.id === baseline) ? baseline : items[0]?.id ?? ''} onChange={(event) => setBaseline(event.target.value)}>{items.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.id}</option>)}</select></label><label className="development-check"><input type="checkbox" checked={differencesOnly} onChange={(event) => setDifferencesOnly(event.target.checked)} /> Differences only</label></div>
    <div className="experiment-table-scroll"><table className="experiment-comparison"><thead><tr><th scope="col">Input / setting</th>{ordered.map((item, index) => <th scope="col" key={item.id}><strong>{item.name}</strong>{index === 0 ? <Badge>Baseline</Badge> : null}<small title={item.id}>{shortRecordId(item.id)}</small><label className="label"><span className="sr-only">Snapshot for {item.name} {item.id}</span><select className="field" value={chosenBatch(item)} onChange={(event) => setBatchIds((current) => ({ ...current, [item.id]: event.target.value }))}><option value="">Saved experiment inputs</option>{experimentStage(item) === 'planning' ? item.batchPlans?.map((plan) => <option key={`plan:${plan.id}`} value={`plan:${plan.id}`}>{plan.spec.batchName} · Editable recipe</option>) : null}{item.batches.map((batch) => <option key={batch.id} value={batch.id}>{batch.manifest.spec.batchName} · {lifecycleLabel[batch.state]}</option>)}</select></label></th>)}</tr></thead><tbody>{rows.map((row) => <tr key={row.path} className={row.different ? 'experiment-different' : ''}><th scope="row">{comparisonFieldLabel(row.path)}<small><code>{row.path}</code></small></th>{row.values.map((value, index) => <td key={ordered[index].id}><pre>{displayValue(value)}</pre></td>)}</tr>)}</tbody></table></div>
    {!rows.length ? <p className="muted">No differences in the selected snapshots.</p> : null}
    <p className="muted">“Not recorded” means the older record did not save that field. Saved snapshots remain visible when the original inputs are archived. No current project defaults are substituted.</p>
  </Panel>;
}

export function CreateExperiment({ project, copy, templates = [], onCreated, onClose }: {
  project: string; copy?: ModelExperiment; templates?: Pick<ModelExperiment, 'id' | 'name' | 'notes' | 'tags' | 'state' | 'stage' | 'status' | 'configurationLocked'>[];
  onCreated: (item: ModelExperiment) => void; onClose: () => void;
}) {
  const sources = templates.filter((item) => item.state !== 'trashed');
  const [sourceId, setSourceId] = useState(copy?.id ?? '');
  const [name, setName] = useState(copy ? `${copy.name.slice(0, 100)} copy` : '');
  const [notes, setNotes] = useState(copy?.notes ?? '');
  const [tags, setTags] = useState(copy?.tags.join(', ') ?? '');
  const [busy, setBusy] = useState(false);
  const creating = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const [pending, setPending] = useState<CreateExperimentInput | null>(null);
  const source = sources.find((item) => item.id === sourceId) ?? (copy?.id === sourceId ? copy : undefined);
  function selectSource(id: string) {
    setSourceId(id);
    const chosen = sources.find((item) => item.id === id) ?? (copy?.id === id ? copy : undefined);
    if (chosen) {
      setName(`${chosen.name.slice(0, 100)} copy`);
      setNotes(chosen.notes);
      setTags(chosen.tags.join(', '));
    }
  }
  async function create() {
    if (creating.current || !name.trim()) return;
    creating.current = true;
    setBusy(true); setError(null);
    const input = pending ?? { name: name.trim(), notes: notes.trim(), tags: [...new Set(tags.split(',').map((value) => value.trim()).filter(Boolean))], ...(sourceId ? { sourceExperimentId: sourceId } : {}), operationId: crypto.randomUUID() };
    try { const result = await experiments.create(project, input); setPending(null); onCreated(result); }
    catch (reason) { setPending(reason instanceof ApiError ? null : input); setError(reason instanceof Error ? reason : new Error('The experiment could not be created.')); }
    finally { creating.current = false; setBusy(false); }
  }
  return <Panel title="Create experiment" subtitle="Name your experiment, then adjust its inputs and batches. Everything stays editable until you submit it.">
    <form onSubmit={(event) => { event.preventDefault(); void create(); }}><fieldset disabled={busy || Boolean(pending)} className="experiment-create-fields"><legend className="sr-only">Experiment details</legend>
      <label className="label experiment-template-field">Start from template<select className="field" value={sourceId} onChange={(event) => selectSource(event.target.value)}>
        <option value="">Blank experiment</option>
        {copy && !sources.some((item) => item.id === copy.id) ? <option value={copy.id}>{copy.name} · {shortRecordId(copy.id)}</option> : null}
        {sources.map((item) => <option value={item.id} key={item.id}>{item.name} · {experimentStageLabel[experimentStage(item)]} · {shortRecordId(item.id)}</option>)}
      </select><small>{source ? `Copies the saved inputs and batch recipes from “${source.name}” into an editable plan. Runs and results stay with the source experiment.` : 'Use an existing experiment as a template to reuse its inputs and batch recipes.'}</small></label>
      <label className="label">Experiment name<input autoFocus required className="field" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} /></label>
      <label className="label">Tags<input className="field" value={tags} onChange={(event) => setTags(event.target.value)} placeholder="baseline, abmil, comparison" /><small>Comma-separated labels for filtering and organization.</small></label>
      <label className="label">Notes<textarea className="field" value={notes} maxLength={10000} onChange={(event) => setNotes(event.target.value)} placeholder="What are you testing in this experiment?" /></label>
    </fieldset>
    <ErrorNotice error={error} />{pending ? <p className="callout" role="status">The response was lost. Retry sends the same creation request and cannot create a second record.</p> : null}
    <div className="inline-actions"><button className="btn btn-primary" disabled={busy || !name.trim()} type="submit">{busy ? 'Creating…' : pending ? 'Retry creation' : 'Create & open inputs'}</button><button className="btn btn-secondary" type="button" disabled={busy} onClick={onClose}>Close</button></div></form>
  </Panel>;
}

export function ExperimentMetadata({ project, record }: { project: string; record: ModelExperiment }) {
  const client = useQueryClient();
  const [name, setName] = useState(record.name);
  const [notes, setNotes] = useState(record.notes);
  const [tags, setTags] = useState(record.tags.join(', '));
  const [revision, setRevision] = useState(record.revision);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState('');
  const stale = revision !== record.revision;
  async function save() {
    setBusy(true); setError(null); setNotice('');
    try {
      const saved = await experiments.update(project, record.id, { name: name.trim(), notes, tags: [...new Set(tags.split(',').map((value) => value.trim()).filter(Boolean))], expectedRevision: revision });
      setRevision(saved.revision); client.setQueryData(['model-experiment', project, record.id], saved);
      await client.invalidateQueries({ queryKey: ['model-experiments', project] });
      setNotice('Experiment details saved. Frozen batches retain their original snapshots.');
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Experiment details could not be saved.')); }
    finally { setBusy(false); }
  }
  return <form className="stack" onSubmit={(event) => { event.preventDefault(); void save(); }}>
    {stale ? <p className="callout">The saved experiment changed. <button type="button" className="text-button" onClick={() => { setName(record.name); setNotes(record.notes); setTags(record.tags.join(', ')); setRevision(record.revision); setError(null); }}>Reload saved details</button></p> : null}
    <fieldset className="experiment-create-fields" disabled={busy || stale || record.legacy || record.state !== 'active'}><legend className="sr-only">Edit experiment details</legend>
      <label className="label">Experiment name<input required className="field" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} /></label>
      <label className="label">Tags<input className="field" value={tags} onChange={(event) => setTags(event.target.value)} /><small>Comma-separated labels.</small></label>
      <label className="label">Notes<textarea className="field" value={notes} maxLength={10000} onChange={(event) => setNotes(event.target.value)} /></label>
    </fieldset>
    <ErrorNotice error={error} />{notice ? <p role="status" className="muted">{notice}</p> : null}
    <button className="btn btn-secondary" disabled={busy || stale || record.legacy || record.state !== 'active' || !name.trim()} type="submit">Save experiment details</button>
  </form>;
}

export default function ExperimentRegistry({ project, onOpen }: { project: string; onOpen: (id: string) => void }) {
  const client = useQueryClient();
  const query = useQuery({ queryKey: ['model-experiments', project, 'summary'], queryFn: () => experiments.summaries(project), refetchInterval: (value) => experimentPollInterval(value.state.data) });
  const [state, setState] = useState<LifecycleState | 'all'>('active');
  const [status, setStatus] = useState('');
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState('recent');
  const [selected, setSelected] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);
  const [manage, setManage] = useState('');
  const items = query.data?.items ?? [];
  const runningCount = items.filter((item) => experimentStage(item) === 'running').length;
  const visible = filterExperiments(items, state, status, search, sort);
  const comparisonQueries = useQueries({ queries: selected.map((id) => ({ queryKey: ['model-experiment', project, id], queryFn: () => experiments.get(project, id), refetchInterval: 15000 })) });
  const compared = comparisonQueries.flatMap((value) => value.data ? [value.data] : []);
  const managed = items.find((item) => item.id === manage);
  return <div className="clinical-workspace experiment-registry">
    <PageHeader eyebrow="02 DEVELOP" title="Experiments" description="Every experiment is a record of a development question, its inputs, training batches and results." actions={<button className="btn btn-primary" onClick={() => setCreating(true)}>Create experiment</button>} />

    {creating ? <CreateExperiment project={project} templates={items} onClose={() => setCreating(false)} onCreated={(item) => { client.setQueryData(['model-experiment', project, item.id], item); void client.invalidateQueries({ queryKey: ['model-experiments', project] }); onOpen(item.id); }} /> : null}
    <ErrorNotice error={query.error} />
    <section className="card experiment-list" aria-label="Experiments">
      <div className="experiment-list-heading"><div><h2>Experiments</h2><p className="muted">{items.length} records · {runningCount} {runningCount === 1 ? 'experiment' : 'experiments'} in progress</p></div><button className="btn btn-secondary btn-small" onClick={() => void query.refetch()}>Refresh</button></div>
      <div className="experiment-state-tabs" role="group" aria-label="Experiment state">{(['active', 'archived', 'trashed', 'all'] as const).map((value) => <button key={value} aria-pressed={state === value} className={state === value ? 'selected' : ''} onClick={() => setState(value)}>{value === 'all' ? 'All records' : lifecycleLabel[value]} <span>{items.filter((item) => value === 'all' || item.state === value).length}</span></button>)}</div>
      <div className="experiment-filters"><label className="label">Find experiments<input type="search" className="field" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name, ID, notes or tag" /></label><label className="label">Stage<select className="field" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">All stages</option>{Object.entries(experimentStageLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label className="label">Sort<select className="field" value={sort} onChange={(event) => setSort(event.target.value)}><option value="recent">Recently updated</option><option value="oldest">Created first</option><option value="name">Name</option></select></label></div>
      <div className="experiment-selection"><span>Select 2–4 experiments to compare · {selected.length} selected</span>{selected.length ? <button className="text-button" onClick={() => setSelected([])}>Clear comparison</button> : null}</div>
      {query.isPending ? <p className="panel-body" role="status">Loading experiments…</p> : visible.length ? <div className="experiment-table-scroll"><table className="experiment-record-table"><thead><tr><th scope="col"><span className="sr-only">Compare</span></th><th scope="col">Experiment</th><th scope="col">Stage</th><th scope="col">Inputs</th><th scope="col">Batches / runs</th><th scope="col">Updated</th><th scope="col">Actions</th></tr></thead><tbody>{visible.map((item) => <tr key={item.id}>
        <td><input aria-label={`Compare ${item.name} ${item.id}`} type="checkbox" checked={selected.includes(item.id)} disabled={selected.length >= 4 && !selected.includes(item.id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} /></td>
        <th scope="row"><button className="text-button experiment-name" onClick={() => onOpen(item.id)}>{item.name}</button><small title={item.id}>{shortRecordId(item.id)}</small><div className="experiment-tags">{item.tags.map((tag) => <Badge key={tag}>{tag}</Badge>)}{item.legacy ? <Badge>Legacy record</Badge> : null}</div></th>
        <td><Badge tone={experimentStage(item) === 'running' ? 'warning' : experimentStage(item) === 'finished' ? 'success' : 'neutral'}>{experimentStageLabel[experimentStage(item)]}</Badge>{experimentStage(item) !== 'planning' && experimentStatusLabel(item.status) !== experimentStageLabel[experimentStage(item)] ? <small>{experimentStatusLabel(item.status)}</small> : null}{item.state !== 'active' ? <small>{lifecycleLabel[item.state]}</small> : null}</td>
        <td>{item.inputs ? <><small title={item.inputs.protocolId}>Targets: {shortRecordId(item.inputs.protocolId)}</small><small title={item.inputs.featureBundleId}>Features: {shortRecordId(item.inputs.featureBundleId)}</small></> : <span className="muted">Not configured</span>}</td>
        <td>{item.batches.length + (experimentStage(item) === 'planning' ? item.batchPlans?.length ?? 0 : 0)} batches<small>{item.batches.length ? `${item.batches.reduce((total, batch) => total + batch.manifest.summary.runCount, 0)} planned runs` : item.batchPlans?.length ? 'Editable recipes' : 'No batches yet'}</small></td>

        <td><time dateTime={item.updatedAt}>{new Date(item.updatedAt).toLocaleDateString()}</time></td>
        <td><button className="btn btn-secondary btn-small" aria-expanded={manage === item.id} onClick={() => setManage(manage === item.id ? '' : item.id)}>Manage</button></td>
      </tr>)}</tbody></table></div> : <div className="experiment-empty"><h3>{items.length ? 'No experiments match this view' : 'Create your first experiment'}</h3><p>{items.length ? 'Adjust the stage, search or archive filters to find earlier work.' : 'Start with a name and a question. Add prepared targets, features and training batches after creating the record.'}</p></div>}
    </section>
    {managed ? <Panel title={`Manage: ${managed.name}`}><ExperimentLifecycle key={managed.id} project={project} recordKey={managed.key} state={managed.state} name={managed.name} /></Panel> : null}
    <ErrorNotice error={comparisonQueries.find((value) => value.error)?.error ?? null} />
    {selected.length >= 2 && compared.length !== selected.length ? <p role="status">Loading selected experiment snapshots…</p> : null}
    {compared.length >= 2 && compared.length === selected.length ? <ExperimentComparison items={compared} /> : null}
    <p className="muted">Archive keeps an experiment’s history available. Delete moves records to recoverable Trash; dependencies and running jobs are checked before confirmation. Files stay on disk.</p>
  </div>;
}
