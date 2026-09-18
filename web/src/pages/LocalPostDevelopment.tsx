import { Fragment, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { predictors, predictorMethodLabel, computeStatusLabel, type FrozenPredictor, type PredictorChoice, type RefitBuild } from '../api/predictors';
import { predictorSourceKey } from '../api/predictorBuilds';
import { ApiError } from '../api/client';
import type { Workspace } from '../api/types';
import { lifecycleLabel, type LifecycleState } from '../api/lifecycle';
import { Badge, ErrorNotice, PageHeader, Panel } from '../components/ui';
import ComputeJobControls from '../components/ComputeJobControls';
import BulkPredictorBuilder from '../components/BulkPredictorBuilder';
import RefitJobs from '../components/RefitJobs';
import EvidenceChain, { evidenceLink } from '../components/EvidenceChain';
import { cleanupLink, useHashParameters } from '../lib/hashRoute';
import { shortRecordId } from '../lib/recordLabels';
import { downloadJSON } from '../lib/download';
import './ModelChains.css';
import '../components/RunWorkspace.css';

export const predictorChoiceKey = (choice: PredictorChoice) => predictorSourceKey(choice);
const experimentLink = (id: string) => `#experiments?experiment=${encodeURIComponent(id)}`;

export default function LocalPredictors({ workspace, historical = false }: { workspace: Workspace; historical?: boolean }) {
  const parameters = useHashParameters();
  const sourceExperiment = parameters.get('experiment') ?? '';
  const sourcePredictor = parameters.get('predictor') ?? '';
  const requestedTab = parameters.get('tab');
  const initialTab = requestedTab === 'library' || requestedTab === 'refits' ? requestedTab : sourcePredictor || historical ? 'library' : 'build';
  return <PredictorWorkspace key={`${workspace.project.id}:${sourceExperiment}:${sourcePredictor}:${initialTab}`} historical={historical} workspace={workspace} sourceExperiment={sourceExperiment} sourcePredictor={sourcePredictor} initialTab={initialTab} />;
}
function PredictorWorkspace({ workspace, sourceExperiment, sourcePredictor, initialTab, historical }: { workspace: Workspace; sourceExperiment: string; sourcePredictor: string; initialTab: 'build' | 'library' | 'refits'; historical: boolean }) {
  const project = workspace.project.id;
  const client = useQueryClient();
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), refetchInterval: 10000 });
  const choices = useQuery({ queryKey: ['predictor-choices', project], queryFn: () => predictors.choices(project), enabled: !historical, refetchInterval: historical ? false : 15000 });
  const refits = useQuery({ queryKey: ['refit-builds', project], queryFn: () => predictors.refits(project), refetchInterval: 5000 });
  const [tab, setTab] = useState<'build' | 'library' | 'refits'>(initialTab);
  const [state, setState] = useState<LifecycleState | 'all'>(sourcePredictor ? 'all' : 'active');
  const [search, setSearch] = useState('');
  const [method, setMethod] = useState('all');
  const [groupBy, setGroupBy] = useState('experiment');
  const [sort, setSort] = useState('seed');
  const [buildId, setBuildId] = useState('');
  async function refresh() { await Promise.all(['predictors', 'predictor-choices', 'refit-builds', 'model-experiments', 'cleanup'].map((key) => client.invalidateQueries({ queryKey: [key, project] }))); }
  const allPredictors = (registry.data?.items ?? []).filter((item) => (!sourceExperiment || item.manifest.experimentId === sourceExperiment) && (!sourcePredictor || item.id === sourcePredictor));
  const visible = allPredictors.filter((item) => (state === 'all' || item.lifecycleState === state) && (method === 'all' || (item.manifest.method ?? 'ensemble') === method) && `${item.manifest.name} ${item.manifest.experimentId} ${item.manifest.trainingSeed} ${item.manifest.splitSeed}`.toLowerCase().includes(search.trim().toLowerCase())).sort((a,b) => sort === 'name' ? a.manifest.name.localeCompare(b.manifest.name) : sort === 'recent' ? b.createdAt.localeCompare(a.createdAt) : a.manifest.experimentId.localeCompare(b.manifest.experimentId) || a.manifest.trainingSeed-b.manifest.trainingSeed || a.manifest.splitSeed-b.manifest.splitSeed);
  const groups = groupBy === 'experiment' ? [...new Map(visible.map((item) => [item.manifest.experimentId, { name: item.manifest.experiment?.name ?? shortRecordId(item.manifest.experimentId), items: visible.filter((row) => row.manifest.experimentId === item.manifest.experimentId) }])).entries()] : [['all', { name: '', items: visible }]] as const;
  const builds = (refits.data?.items ?? []).filter((item) => !sourceExperiment || item.manifest.experimentId === sourceExperiment);
  const build = builds.find((item) => item.id === buildId);
  return <div className="clinical-workspace model-chains run-workspace">
    <PageHeader eyebrow="02 DEVELOP" title={historical ? 'Historical predictors' : 'Build predictors'} description={historical ? 'Inspect existing predictors and recover older refit jobs. New predictor settings and automatic builds are managed inside each experiment.' : 'Build separate ensembles and refits for every experiment, configuration and seed. Review inputs, manage jobs, and compare results in one workspace.'} actions={<a className="btn btn-secondary" href="#evaluation">Run predictors on a test cohort</a>} />
    {historical ? <p className="callout">Create and configure new predictor work in <a href="#experiments">Experiments</a>. This page retains historical predictors and unfinished refit jobs.</p> : null}
    <EvidenceChain current="post-development" experimentId={sourceExperiment} predictorId={sourcePredictor} />
    {sourcePredictor || sourceExperiment ? <p className="callout">{sourcePredictor ? 'Showing the linked predictor.' : 'Showing one experiment.'} <a href="#post-development?tab=library">Show all experiments and predictors</a></p> : null}
    <ErrorNotice error={registry.error ?? (!historical ? choices.error : null) ?? refits.error} />
    <div className="run-kpis"><span><strong>{allPredictors.filter((item) => item.lifecycleState === 'active').length}</strong>published predictors</span><span><strong>{builds.length}</strong>refit plans</span><span><strong>{new Set(allPredictors.map((item) => item.manifest.experimentId)).size}</strong>experiments</span></div>
    <nav className="run-tabs" aria-label="Predictor workspace">{([['build','Build predictors'],['library','Predictor library'],['refits','Refit jobs']] as const).filter(([key]) => !historical || key !== 'build').map(([key,label]) => <button type="button" key={key} aria-pressed={tab===key} className={tab===key?'selected':''} onClick={() => setTab(key)}>{label}</button>)}</nav>
    {tab === 'build' ? <Panel title="Build predictors" subtitle="Select completed seed groups, then choose Ensemble, Refit or Both."><BulkPredictorBuilder project={project} choices={choices.data?.items ?? []} sourceExperiment={sourceExperiment} refresh={async () => { await refresh(); }} /></Panel> : null}
    {tab === 'library' ? <Panel title="Predictor library" subtitle="Every configuration and seed pair keeps its own ensemble and refit. Archived and deleted records remain recoverable.">
      <div className="run-toolbar"><label className="label run-search">Search predictors<input className="field" type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Name, experiment or seed" /></label><label className="label">Predictor visibility<select className="field" value={state} onChange={(event) => setState(event.target.value as typeof state)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label><label className="label">Method<select className="field" value={method} onChange={(event) => setMethod(event.target.value)}><option value="all">All methods</option><option value="ensemble">Ensemble</option><option value="refit">Refit</option></select></label><label className="label">Group by<select className="field" value={groupBy} onChange={(event) => setGroupBy(event.target.value)}><option value="experiment">Experiment</option><option value="none">No grouping</option></select></label><label className="label">Sort<select className="field" value={sort} onChange={(event) => setSort(event.target.value)}><option value="seed">Seed</option><option value="name">Name</option><option value="recent">Recently created</option></select></label></div>
      <div className="run-selection-bar"><strong>{visible.length} predictors</strong><a className="btn btn-primary btn-small" href="#evaluation">Run all predictors on a test cohort</a></div>
      {registry.isPending ? <p role="status">Loading predictors…</p> : !visible.length ? <p>No published predictors in this view.</p> : <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-name">Predictor</th><th>Experiment / batch</th><th>Method / seeds</th><th className="run-number">Weights</th><th className="run-actions">Actions</th></tr></thead><tbody>{groups.map(([key,group]) => <Fragment key={key}>{group.name ? <tr className="run-group"><th colSpan={5}>{group.name} · {group.items.length} predictors</th></tr> : null}{group.items.map((item) => <PredictorRow key={item.id} predictor={item} />)}</Fragment>)}</tbody></table></div>}
    </Panel> : null}
    {tab === 'refits' ? <><Panel title="Refit jobs" subtitle="Select several seed-specific plans to train, resume, cancel or publish together."><RefitJobs project={project} builds={builds} published={allPredictors} onOpen={setBuildId} refresh={refresh} /></Panel>{build ? <Panel title={build.manifest.name}><RefitTraining key={build.id} project={project} build={build} existing={allPredictors.find((item) => predictorSourceKey(item.manifest) === predictorSourceKey(build.manifest) && item.manifest.method === 'refit')} refresh={refresh} /></Panel> : null}</> : null}
  </div>;
}

function PredictorEvidence({ manifest }: { manifest: FrozenPredictor['manifest'] | RefitBuild['manifest'] }) {
  const checkpoints = manifest.sourceCheckpoints ?? manifest.checkpoints;
  return <><p>{checkpoints.length} verified fold checkpoints. Inputs retain the experiment, batch, settings, target, features and checkpoint checksums.</p>
    {manifest.epochBudget ? <div className="callout"><strong>Refit budget: {manifest.epochBudget.epochs} epochs</strong><p>P{manifest.epochBudget.percentile} of selected fold checkpoint epochs [{manifest.epochBudget.foldBestEpochs.map((fold) => fold.bestEpoch).join(', ')}], rounded up. {manifest.trainingSlideCount} development slides{manifest.trainingPatientCount !== undefined ? ` from ${manifest.trainingPatientCount} patients` : ''}.</p></div> : null}
    <details><summary>Checkpoint evidence</summary><ul className="chain-evidence">{checkpoints.map((checkpoint) => <li key={checkpoint.runId}><strong>{checkpoint.runId}</strong><code>{checkpoint.path}</code><small>SHA256: {checkpoint.sha256}</small></li>)}</ul></details></>;
}
export function RefitTraining({ project, build, existing, refresh }: { project: string; build: RefitBuild; existing?: FrozenPredictor; refresh: () => Promise<void> }) {
  const [operation, setOperation] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [published, setPublished] = useState<FrozenPredictor | null>(null);
  const ready = published ?? existing;
  async function publish() {
    if (busy) return;
    const id = operation ?? crypto.randomUUID(); setOperation(id); setBusy(true); setError(null);
    try { setPublished(await predictors.publishRefit(project, build.id, id)); setOperation(null); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Publication failed.')); if (reason instanceof ApiError) setOperation(null); }
    finally { setBusy(false); }
    await refresh();
  }
  return <div className="refit-detail"><PredictorEvidence manifest={build.manifest} />{build.manifest.resources ? <p className="muted">Resources: {build.manifest.resources.gpuIds.length ? `GPU ${build.manifest.resources.gpuIds.join(', ')}` : 'CPU'} · {build.manifest.resources.cpuThreadsPerRun} CPU threads · {build.manifest.resources.dataLoaderWorkers} data workers · {build.manifest.resources.ramGbPerRun} GB RAM reservation.</p> : null}<ComputeJobControls project={project} id={build.id} kind="refit" initial={build.execution} readOnly={build.lifecycleState !== 'active' || Boolean(ready)} readOnlyReason={ready ? 'This configuration and seed already have a refit predictor. Use or restore that predictor.' : undefined} />
    <ErrorNotice error={error} />
    {ready ? <p className={`callout ${ready.lifecycleState !== 'trashed' ? 'science-success' : ''}`}>{ready.manifest.refitId === build.id ? 'This plan’s refit predictor is published.' : 'This configuration and seed already have a refit predictor from another plan.'} {ready.lifecycleState === 'trashed' ? <a href={cleanupLink(ready.id)}>Restore {ready.manifest.name} from Trash</a> : <a href={`#evaluation?predictor=${encodeURIComponent(ready.id)}`}>Evaluate {ready.manifest.name}</a>}</p> : build.execution?.status === 'completed' && build.lifecycleState === 'active' ? <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void publish()}>{busy ? 'Verifying checkpoint…' : operation ? 'Retry publication' : 'Publish refit predictor'}</button> : null}
    <p><a href={cleanupLink(build.id)}>{build.lifecycleState === 'active' ? 'Archive / delete refit plan' : 'Restore / manage refit plan'}</a></p>
  </div>;
}
function PredictorRow({ predictor: item }: { predictor: FrozenPredictor }) {
  const manifest = item.manifest;
  return <tr><td><strong>{manifest.name}</strong><small><Badge>{lifecycleLabel[item.lifecycleState]}</Badge></small></td><td><a href={experimentLink(manifest.experimentId)} title={manifest.experimentId}>{manifest.experiment?.name ?? shortRecordId(manifest.experimentId)}</a><small title={manifest.batchId}>{shortRecordId(manifest.batchId)}</small></td><td>{predictorMethodLabel(manifest.method)}<small>{manifest.recipe.model} · training {manifest.trainingSeed} / split {manifest.splitSeed}</small>{manifest.epochBudget ? <small>{manifest.epochBudget.epochs} epochs · P{manifest.epochBudget.percentile}</small> : null}</td><td>{manifest.checkpoints.length} checkpoint{manifest.checkpoints.length === 1 ? '' : 's'}</td><td><div className="inline-actions">{item.lifecycleState !== 'trashed' ? <><a className="btn btn-secondary btn-small" href={`#evaluation?predictor=${encodeURIComponent(item.id)}`}>Evaluate</a><a className="btn btn-secondary btn-small" href={evidenceLink('interpretation', { experimentId: manifest.experimentId, predictorId: item.id })}>Interpret slides</a></> : null}<a className="text-link" href={cleanupLink(item.id)}>{item.lifecycleState === 'active' ? 'Archive / delete' : 'Restore / manage'}</a><button type="button" className="text-button" onClick={() => downloadJSON(`${manifest.name}.json`, item)}>Export</button></div></td></tr>;
}
