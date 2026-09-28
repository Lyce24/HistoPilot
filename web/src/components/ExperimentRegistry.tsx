import { StageBackButton, StageCreateButton } from './StageActions';
import { shortRecordId } from '../lib/recordLabels';
import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import { flushSync } from 'react-dom';
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { experimentHeadlines } from '../api/experimentResults';
import ExperimentHeadline from './ExperimentHeadline';
import { ApiError } from '../api/client';
import { experiments, experimentExecutionStatus, experimentPollInterval, experimentStatusReason, experimentStage, experimentStageLabel, experimentStatusFilters, experimentStatusLabel, experimentStatusTone } from '../api/experiments';
import type { CreateExperimentInput, ModelExperiment } from '../api/experiments';
import type { LifecycleState } from '../api/lifecycle';
import { lifecycleLabel } from '../api/lifecycle';
import { sameJSON } from '../lib/json';
import { Badge, EmptyState, ErrorNotice, PageHeader, Panel } from './ui';
import { RecordManageButton } from './RecordManagement';
import { StageLibrary, StageLibraryToolbar, StagePage, StageSteps, useStageLibrary } from './StageWorkflow';
import './ExperimentRegistry.css';
import { readSessionDraft, sessionDraftKey, useSessionDraftBackup, writeSessionDraft } from '../lib/sessionDraft';
import { confirmWorkspaceNavigation, useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import { isCreateExperimentDraft, isExperimentMetadataDraft, type CreateExperimentDraft, type ExperimentMetadataBaseline, type ExperimentMetadataDraft } from '../lib/experimentEditorDraft';

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

export function CreateExperiment({ project, copy, templates = [], onCreated, onClose, setup = false }: {
  setup?: boolean;
  project: string; copy?: ModelExperiment; templates?: Pick<ModelExperiment, 'id' | 'name' | 'notes' | 'tags' | 'state' | 'stage' | 'status' | 'configurationLocked'>[];
  onCreated: (item: ModelExperiment) => void; onClose: () => void;
}) {
  const sources = templates.filter((item) => item.state !== 'trashed');
  const recoveryKey = sessionDraftKey(project, copy?.id ?? 'new', 'create');
  const [recovered] = useState(() => readSessionDraft(recoveryKey, isCreateExperimentDraft));
  const [sourceId, setSourceId] = useState(recovered?.sourceId ?? copy?.id ?? '');
  const [name, setName] = useState(recovered?.name ?? (copy ? `${copy.name.slice(0, 100)} copy` : ''));
  const [notes, setNotes] = useState(recovered?.notes ?? copy?.notes ?? '');
  const [tags, setTags] = useState(recovered?.tags ?? copy?.tags.join(', ') ?? '');
  const [busy, setBusy] = useState(false);
  const [complete, setComplete] = useState(false);
  const creating = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const [pending, setPending] = useState<CreateExperimentInput | null>(recovered?.pending ?? null);
  const draft: CreateExperimentDraft | null = !complete && (name || notes || tags || sourceId || pending) ? { version: 1, sourceId, name, notes, tags, pending } : null;
  const backup = useSessionDraftBackup(recoveryKey, draft, isCreateExperimentDraft);
  useWorkspaceNavigationGuard(busy ? 'An experiment creation request is in progress. Its response may arrive after you leave.' : draft && backup.error ? 'Unsaved experiment details cannot be recovered in this browser.' : null);
  function close() { if (!creating.current && confirmWorkspaceNavigation()) onClose(); }
  useStageLibrary(close);
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
    if (creating.current || complete || !name.trim()) return;
    creating.current = true;
    setBusy(true); setError(null);
    const input = pending ?? { name: name.trim(), notes: notes.trim(), tags: [...new Set(tags.split(',').map((value) => value.trim()).filter(Boolean))], ...(sourceId ? { sourceExperimentId: sourceId } : {}), ...(setup ? { setupVersion: 1 as const } : {}), operationId: crypto.randomUUID() };
    // Persist the exact operation before starting I/O so a reload can retry the
    // accepted request rather than creating a second experiment.
    setPending(input);
    writeSessionDraft(recoveryKey, { version: 1, sourceId, name, notes, tags, pending: input } satisfies CreateExperimentDraft);
    try {
      const result = await experiments.create(project, input);
      writeSessionDraft(recoveryKey, null);
      // The acknowledged result deliberately navigates to its new record. Commit
      // guard removal first, so that handoff is not mistaken for leaving a request.
      flushSync(() => { setComplete(true); setPending(null); setBusy(false); });
      onCreated(result);
    }
    catch (reason) {
      // Server/proxy failures can follow acceptance, just like a lost connection.
      const rejected = reason instanceof ApiError && reason.status < 500 && reason.status !== 408;
      setPending(rejected ? null : input);
      setError(reason instanceof Error ? reason : new Error('The experiment could not be created.'));
    }
    finally { creating.current = false; setBusy(false); }
  }
  return <><PageHeader eyebrow={setup ? '03 EXPERIMENTAL SETUP' : '04 EXPERIMENTS'} title={setup ? 'Create experimental setup' : 'Create experiment'} description={setup ? 'Name the setup, then select its data, training design and hyperparameters.' : 'Name your experiment or reuse a saved template, then continue to its inputs.'} actions={<StageBackButton disabled={busy} onClick={close}>Back to {setup ? 'setups' : 'experiments'}</StageBackButton>} />
    <StageSteps label="New experiment steps" current="details" steps={[{ id: 'details', title: 'Experiment details', description: 'Name and optional template' }, { id: 'inputs', title: 'Inputs', description: 'Continue after creating the record', disabled: true }]} onChange={() => {}} />
    <Panel title={setup ? 'Setup details' : 'Experiment details'} subtitle={setup ? 'Prepare inputs and training batches, then freeze the design. Training starts later in Experiments.' : 'Name your experiment, then adjust its inputs and batches. Everything stays editable until you submit it.'}>
    {recovered ? <p className="callout" role="status">Recovered this tab’s unfinished experiment details. {pending ? 'Retry creation to recover the original request.' : 'Review them before creating the experiment.'}</p> : null}
    {draft && backup.error ? <p className="callout callout-warning" role="alert">{backup.error}</p> : null}
    <form onSubmit={(event) => { event.preventDefault(); void create(); }}><fieldset disabled={busy || Boolean(pending)} className="experiment-create-fields"><legend className="sr-only">Experiment details</legend>
      <label className="label experiment-template-field">Start from template<select className="field" value={sourceId} onChange={(event) => selectSource(event.target.value)}>
        <option value="">{setup ? 'Blank setup' : 'Blank experiment'}</option>
        {sourceId && !source ? <option value={sourceId} disabled>Selected template unavailable</option> : null}
        {copy && !sources.some((item) => item.id === copy.id) ? <option value={copy.id}>{copy.name} · {shortRecordId(copy.id)}</option> : null}
        {sources.map((item) => <option value={item.id} key={item.id}>{item.name} · {experimentStageLabel[experimentStage(item)]} · {shortRecordId(item.id)}</option>)}
      </select><small>{source ? `Copies the saved inputs and batch recipes from “${source.name}” into an editable plan. Runs and results stay with the source experiment.` : 'Use an existing experiment as a template to reuse its inputs and batch recipes.'}</small></label>
      <label className="label">{setup ? 'Setup name' : 'Experiment name'}<input autoFocus required className="field" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} /></label>
      <label className="label">Tags<input className="field" value={tags} maxLength={20000} onChange={(event) => setTags(event.target.value)} placeholder="baseline, abmil, comparison" /><small>Comma-separated labels for filtering and organization.</small></label>
      <label className="label">Notes<textarea className="field" value={notes} maxLength={10000} onChange={(event) => setNotes(event.target.value)} placeholder="What are you testing in this experiment?" /></label>
    </fieldset>
    <ErrorNotice error={error} />{pending && !busy ? <p className="callout" role="status">The creation response is not confirmed. Retry sends the same creation request and cannot create a second record.</p> : null}
    <div className="stage-actions"><StageBackButton type="button" disabled={busy} onClick={close}>Back to {setup ? 'setups' : 'experiments'}</StageBackButton><StageCreateButton disabled={busy || !name.trim()} type="submit">{busy ? 'Creating…' : pending ? 'Retry creation' : 'Create & open inputs'}</StageCreateButton></div></form>
  </Panel></>;
}

export function ExperimentMetadata({ project, record }: { project: string; record: ModelExperiment }) {
  const client = useQueryClient();
  const recoveryKey = sessionDraftKey(project, record.id, 'metadata');
  const [recovered] = useState(() => readSessionDraft(recoveryKey, isExperimentMetadataDraft));
  const [name, setName] = useState(recovered?.name ?? record.name);
  const [notes, setNotes] = useState(recovered?.notes ?? record.notes);
  const [tags, setTags] = useState(recovered?.tags ?? record.tags.join(', '));
  const [baseline, setBaseline] = useState<ExperimentMetadataBaseline>(recovered?.baseline ?? record);
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState(recovered ? 'Recovered unsaved experiment details in this tab.' : '');
  const dirty = name !== baseline.name || notes !== baseline.notes || tags !== baseline.tags.join(', ');
  const stale = baseline.revision !== record.revision;
  const readOnly = record.legacy || record.state !== 'active';
  const draft: ExperimentMetadataDraft | null = dirty ? { version: 1, name, notes, tags, baseline: { revision: baseline.revision, name: baseline.name, notes: baseline.notes, tags: baseline.tags } } : null;
  const backup = useSessionDraftBackup(recoveryKey, draft, isExperimentMetadataDraft);
  useWorkspaceNavigationGuard(busy ? 'Experiment details are being saved or reloaded. Wait for the response before leaving.' : dirty && backup.error ? 'Unsaved experiment details cannot be recovered in this browser.' : null);
  function adopt(saved: ModelExperiment) {
    setName(saved.name); setNotes(saved.notes); setTags(saved.tags.join(', ')); setBaseline(saved);
    client.setQueryData(['model-experiment', project, record.id], saved);
  }
  useEffect(() => {
    if (busy || !stale) return;
    // A batch/run update can advance the experiment revision without changing
    // metadata. Such updates do not require discarding independent text edits.
    const unchanged = name === record.name && notes === record.notes && tags === record.tags.join(', ');
    const sameMetadata = baseline.name === record.name && baseline.notes === record.notes && sameJSON(baseline.tags, record.tags);
    if (!dirty || unchanged) { setName(record.name); setNotes(record.notes); setTags(record.tags.join(', ')); setBaseline(record); }
    else if (sameMetadata) setBaseline(record);
  }, [record, busy, stale, dirty, baseline, name, notes, tags]);
  async function save() {
    if (inFlight.current || stale || readOnly || !name.trim() || !dirty) return;
    inFlight.current = true; setBusy(true); setError(null); setNotice('');
    try {
      const saved = await experiments.update(project, record.id, { name: name.trim(), notes, tags: [...new Set(tags.split(',').map((value) => value.trim()).filter(Boolean))], expectedRevision: baseline.revision });
      writeSessionDraft(recoveryKey, null); adopt(saved);
      await client.invalidateQueries({ queryKey: ['model-experiments', project] });
      setNotice('Experiment details saved. Frozen batches retain their original snapshots.');
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('Experiment details could not be saved.'));
      if (reason instanceof ApiError && reason.status === 409) void client.invalidateQueries({ queryKey: ['model-experiment', project, record.id] });
    } finally { inFlight.current = false; setBusy(false); }
  }
  async function reload() {
    if (inFlight.current || dirty && !window.confirm('Replace your unsaved experiment details with the latest saved details?')) return;
    inFlight.current = true; setBusy(true); setError(null);
    try { const saved = await experiments.get(project, record.id); writeSessionDraft(recoveryKey, null); adopt(saved); setNotice('Latest saved experiment details loaded.'); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Saved experiment details could not be loaded.')); }
    finally { inFlight.current = false; setBusy(false); }
  }
  return <form className="stack" onChange={() => setNotice('')} onSubmit={(event) => { event.preventDefault(); void save(); }}>
    {stale ? <p className="callout" role="status">The saved experiment details changed. Your edits are retained. <button type="button" className="text-button" disabled={busy} onClick={() => void reload()}>Reload saved details</button> replaces them with the latest saved values.</p> : null}
    {dirty && backup.error ? <p className="callout callout-warning" role="alert">{backup.error}</p> : null}
    <fieldset className="experiment-create-fields" disabled={busy || stale || readOnly}><legend className="sr-only">Edit experiment details</legend>
      <label className="label">Experiment name<input required className="field" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} /></label>
      <label className="label">Tags<input className="field" value={tags} maxLength={20000} onChange={(event) => setTags(event.target.value)} /><small>Comma-separated labels.</small></label>
      <label className="label">Notes<textarea className="field" value={notes} maxLength={10000} onChange={(event) => setNotes(event.target.value)} /></label>
    </fieldset>
    <ErrorNotice error={error} />{notice ? <p role="status" className="muted">{notice}</p> : null}
    <button className="btn btn-secondary" disabled={busy || stale || readOnly || !name.trim() || !dirty} type="submit">{busy ? 'Working…' : 'Save experiment details'}</button>
    {dirty && !backup.error ? <p className="muted">Unsaved details are kept in this tab. Save to update the experiment.</p> : null}
  </form>;
}

export interface ExperimentLibraryFilters {
  state: LifecycleState | 'all'; status: string; search: string; sort: string; selected: string[];
}
export const newExperimentLibraryFilters = (): ExperimentLibraryFilters => ({ state: 'active', status: '', search: '', sort: 'recent', selected: [] });

export type ExperimentLibraryMode = 'setup' | 'execution';
export function stageExperiments<T extends Pick<ModelExperiment, 'legacy' | 'stage' | 'status' | 'configurationLocked' | 'frozenSetupId' | 'setupVersion'>>(items: T[], mode: ExperimentLibraryMode): T[] {
  return items.filter((item) => mode === 'setup'
    ? !item.legacy && (item.setupVersion === 1 || experimentStage(item) === 'planning')
    : Boolean(item.frozenSetupId) || experimentStage(item) !== 'planning');
}

export default function ExperimentRegistry({ project, onOpen, filters, onFiltersChange, mode }: {
  mode: ExperimentLibraryMode;
  project: string; onOpen: (id: string) => void;
  filters?: ExperimentLibraryFilters; onFiltersChange?: Dispatch<SetStateAction<ExperimentLibraryFilters>>;
}) {
  const client = useQueryClient();
  const query = useQuery({ queryKey: ['model-experiments', project, 'summary'], queryFn: () => experiments.summaries(project), refetchInterval: (value) => experimentPollInterval(value.state.data) });
  const [localFilters, setLocalFilters] = useState(newExperimentLibraryFilters);
  const { state, status, search, sort, selected } = filters ?? localFilters;
  const setFilters = onFiltersChange ?? setLocalFilters;
  const setState = (value: LifecycleState | 'all') => setFilters((current) => ({ ...current, state: value }));
  const setStatus = (value: string) => setFilters((current) => ({ ...current, status: value }));
  const setSearch = (value: string) => setFilters((current) => ({ ...current, search: value }));
  const setSort = (value: string) => setFilters((current) => ({ ...current, sort: value }));
  const setSelected = (value: SetStateAction<string[]>) => setFilters((current) => ({ ...current, selected: typeof value === 'function' ? value(current.selected) : value }));
  const [creating, setCreating] = useState(false);
  const [comparing, setComparing] = useState(false);
  useStageLibrary(() => { if (!creating) setComparing(false); });
  const allItems = query.data?.items ?? [];
  const items = stageExperiments(allItems, mode);
  const title = mode === 'setup' ? 'Experimental Setup' : 'Experiments';
  const visible = filterExperiments(items, state, '', search, sort)
    .filter((item) => !status || (mode === 'execution' ? experimentExecutionStatus(item.status) === status : status === 'frozen' ? Boolean(item.frozenSetupId) : !item.frozenSetupId));
  const comparisonQueries = useQueries({ queries: selected.map((id) => ({ queryKey: ['model-experiment', project, id], queryFn: () => experiments.get(project, id), refetchInterval: 15000 })) });
  // One headline per experiment; folds, seeds and intervals load when an experiment opens.
  const headlines = useQuery({ queryKey: ['experiment-headlines', project], queryFn: () => experimentHeadlines.list(project), enabled: mode !== 'setup',
    refetchInterval: allItems.some((item) => experimentStage(item) === 'running') ? 30000 : false });
  const headlineOf = new Map((headlines.data?.items ?? []).map((row) => [row.experimentId, row.batches]));
  const compared = comparisonQueries.flatMap((value) => value.data ? [value.data] : []);
  return <div className="clinical-workspace experiment-registry">
    {!creating ? <PageHeader eyebrow={mode === 'setup' ? '03 EXPERIMENTAL SETUP' : '04 EXPERIMENTS'} title={comparing ? 'Compare saved designs' : title} description={comparing ? 'Compare saved inputs and training settings.' : mode === 'setup' ? 'Prepare datasets, targets, training splits and hyperparameters. Freeze a setup when the full design is ready.' : 'Run frozen setups and follow queued, active, completed and failed experiments.'} actions={comparing ? <StageBackButton onClick={() => setComparing(false)}>Back to experiments</StageBackButton> : mode === 'execution' ? <a className="btn btn-primary" href="#experimental-setup">Prepare a setup</a> : <StageCreateButton onClick={() => setCreating(true)}>Create setup</StageCreateButton>} /> : null}

    <StagePage pageKey={creating ? 'create' : comparing ? 'comparison' : 'library'}>
    {creating ? <CreateExperiment setup={mode === 'setup'} project={project} templates={allItems} onClose={() => setCreating(false)} onCreated={(item) => { client.setQueryData(['model-experiment', project, item.id], item); void client.invalidateQueries({ queryKey: ['model-experiments', project] }); void client.invalidateQueries({ queryKey: ['scientific', project, 'drafts'] }); onOpen(item.id); }} /> : comparing ? <>
      <ErrorNotice error={comparisonQueries.find((value) => value.error)?.error ?? null} />
      {compared.length !== selected.length ? <p role="status">Loading selected experiment snapshots…</p> : compared.length >= 2 ? <ExperimentComparison items={compared} /> : <p>Select at least two experiments from the library to compare.</p>}
      <div className="stage-actions"><StageBackButton type="button" onClick={() => setComparing(false)}>Back to experiment selection</StageBackButton></div>
    </> : <>
    <ErrorNotice error={query.error} />
    <StageLibrary project={project} title={title}>
      <StageLibraryToolbar search={search} onSearch={setSearch} searchLabel="Search experiments" placeholder="Name, ID, notes or tag" count={query.isPending ? undefined : visible.length} total={items.length}
        onReset={search || status || state !== 'active' || sort !== 'recent' ? () => { setSearch(''); setStatus(''); setState('active'); setSort('recent'); } : undefined}
        actions={<button type="button" className="btn btn-secondary btn-small" disabled={query.isFetching} onClick={() => void query.refetch()}>{query.isFetching ? 'Refreshing…' : 'Refresh'}</button>}>
        <label className="label">State<select className="field" aria-label="Experiment state" value={state} onChange={(event) => setState(event.target.value as LifecycleState | 'all')}>{(['active', 'archived', 'trashed', 'all'] as const).map((value) => <option key={value} value={value}>{value === 'all' ? 'All records' : lifecycleLabel[value]} ({items.filter((item) => value === 'all' || item.state === value).length})</option>)}</select></label>
        <label className="label">Status<select className="field" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">All statuses</option>{Object.entries(mode === 'setup' ? { draft: 'Draft', frozen: 'Frozen' } : experimentStatusFilters).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label className="label">Sort<select className="field" value={sort} onChange={(event) => setSort(event.target.value)}><option value="recent">Last updated</option><option value="oldest">Oldest first</option><option value="name">Name</option></select></label>
      </StageLibraryToolbar>
      {selected.length ? <div className="experiment-selection"><span>{selected.length} selected <span className="muted">· Choose 2–4 to compare</span></span><div className="inline-actions"><button type="button" className="text-button" onClick={() => setSelected([])}>Clear comparison</button><button type="button" className="btn btn-secondary btn-small" disabled={selected.length < 2} onClick={() => setComparing(true)}>Compare selected experiments</button></div></div> : null}
      {query.isPending ? <p className="panel-body" role="status">Loading experiments…</p> : visible.length ? <div className="experiment-table-scroll"><table className="experiment-record-table" aria-label="Saved experiments"><thead><tr><th scope="col"><span className="sr-only">Select 2–4 experiments to compare</span></th><th scope="col">{mode === 'setup' ? 'Setup' : 'Experiment'}</th><th scope="col">Status</th><th scope="col">Training</th>{mode !== 'setup' ? <th scope="col" className="experiment-result-column" title="Out-of-fold result of the best batch: mean ± SD across training seeds">Result</th> : null}<th scope="col">Updated</th><th scope="col">Actions</th></tr></thead><tbody>{visible.map((item) => <tr key={item.id}>
        <td><input aria-label={`Compare ${item.name} ${item.id}`} type="checkbox" checked={selected.includes(item.id)} disabled={selected.length >= 4 && !selected.includes(item.id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} /></td>
        <th scope="row"><button className="text-button stage-record-name experiment-name" title={`Open ${item.name}`} onClick={() => onOpen(item.id)}>{item.name}</button><small title={item.id}>{shortRecordId(item.id)}</small>{item.tags.length || item.legacy ? <div className="experiment-tags">{item.tags.map((tag) => <Badge key={tag}>{tag}</Badge>)}{item.legacy ? <Badge>Legacy record</Badge> : null}</div> : null}</th>
        {/* A setup's badge describes the setup; run outcomes belong to the Experiments page. */}
        <td><Badge tone={mode === 'setup' ? item.frozenSetupId ? 'green' : 'neutral' : experimentStage(item) === 'planning' ? 'neutral' : experimentStatusTone(item.status)}>{mode === 'setup' ? item.frozenSetupId ? 'Frozen' : 'Draft' : experimentStatusLabel(item.status)}</Badge>{(() => { const reason = mode === 'execution' ? experimentStatusReason(item) : null; return reason ? <small title={reason}>{reason}</small> : null; })()}{item.state !== 'active' ? <small>{lifecycleLabel[item.state]}</small> : null}</td>
        <td>{(() => { const count = item.batches.length + (experimentStage(item) === 'planning' ? item.batchPlans?.length ?? 0 : 0); return `${count} ${count === 1 ? 'batch' : 'batches'}`; })()}<small>{item.batches.length ? `${item.batches.reduce((total, batch) => total + batch.manifest.summary.runCount, 0)} planned runs` : item.batchPlans?.length ? item.frozenSetupId ? 'Frozen recipes' : 'Editable recipes' : item.inputs ? 'Inputs selected' : 'Inputs not set'}</small></td>

        {mode !== 'setup' ? <td><ExperimentHeadline batches={headlineOf.get(item.id)} /></td> : null}
        <td><time dateTime={item.updatedAt}>{new Date(item.updatedAt).toLocaleDateString()}</time></td>
        <td><RecordManageButton recordKey={item.key} name={item.name} /></td>
      </tr>)}</tbody></table></div> : <EmptyState
      icon="experiments"
      title={items.length ? 'No records match this view' : mode === 'execution' ? 'No frozen setups ready to run' : 'Create your first setup'}
      description={items.length ? 'Adjust the status, search or archive filters to find earlier work.' : mode === 'execution' ? 'Prepare and freeze an Experimental Setup, then return here to start training.' : 'Start with a name and a question. Add your dataset, target and splits, features, training design and hyperparameters.'}
      action={items.length
        ? <button type="button" className="btn btn-secondary" onClick={() => setFilters(newExperimentLibraryFilters())}>Clear filters</button>
        : mode === 'execution' ? <a className="btn btn-primary" href="#experimental-setup">Open Experimental Setup</a> : <StageCreateButton onClick={() => setCreating(true)}>Create setup</StageCreateButton>}
    />}
    </StageLibrary>
    </>}
    </StagePage>
  </div>;
}
