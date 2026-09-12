import { useEffect, useState } from 'react';
import { splitModeLabel, taskLabel, unitLabel } from '../lib/labels';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { type Configuration, type ProtocolSpec } from '../api/scientific';
import { experiments, experimentStatusLabel, type ModelExperiment } from '../api/experiments';
import { lifecycleLabel } from '../api/lifecycle';
import ExperimentRegistry, { CreateExperiment, ExperimentMetadata } from '../components/ExperimentRegistry';
import ExperimentLifecycle from '../components/ExperimentLifecycle';
import { bundles, type FeatureBundle } from '../api/bundles';
import { mil, type LoadingPolicy, type MILExperimentPreview, type MILExperimentSpec } from '../api/mil';
import { sameJSON } from '../lib/json';
import { configurationVersionLabel, versionLabelText } from '../lib/versionLabels';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import { Findings, SavedNotice, useConfigurations } from '../components/ScientificUI';
import DevelopmentBatches, { developmentTabs, type DevelopmentTab } from '../components/DevelopmentBatches';
import SetupContext from '../components/SetupContext';
import { protocolBundleCompatible } from '../lib/roadmap';
import { preparationLink, usePreparationContext, type PreparationContext } from '../lib/preparationRoute';
import PreparationNotice from '../components/PreparationNotice';
import './LocalExperiments.css';

const initialSpec = (): MILExperimentSpec => ({
  protocolId: '', featureBundleId: '', loadingPolicy: 'auto', packArtifactId: null,
});

/** Suggest a unique compatible pair, retaining explicit prepared inputs when the other choice is ambiguous. */
export function suggestedExperimentInputs(protocols: Configuration[], featureBundles: FeatureBundle[], context: PreparationContext = {}): MILExperimentSpec {
  const pairs = protocols.filter((protocol) => (!context.protocolId || protocol.id === context.protocolId) && (!context.datasetId || protocol.manifest.datasetId === context.datasetId)).flatMap((protocol) => featureBundles
    .filter((bundle) => (!context.bundleId || bundle.id === context.bundleId) && bundle.current && !bundle.findings.some((finding) => finding.severity === 'error') && protocolBundleCompatible(protocol, bundle))
    .map((bundle) => ({ protocolId: protocol.id, featureBundleId: bundle.id })));
  if (pairs.length === 1) return { ...initialSpec(), ...pairs[0] };
  return pairs.length > 1 ? { ...initialSpec(), protocolId: context.protocolId ?? '', featureBundleId: context.bundleId ?? '' } : initialSpec();
}

const loadingOptions: { value: LoadingPolicy; title: string; description: string }[] = [
  { value: 'auto', title: 'Auto', description: 'Use the original files for a bundle without packs. Automatically use a single pack that preserves source precision. Choose explicitly when several packs are included or the pack changes precision.' },
  { value: 'native', title: 'Original files', description: 'Read each slide from its original feature file.' },
  { value: 'mmap', title: 'Packed mmap', description: 'Read selected rows from a verified pack. The operating system caches pages as memory allows; the whole pack need not fit in RAM.' },
];

export function LoadingOptions({ value, hasPacks, onChange }: {
  value: LoadingPolicy; hasPacks: boolean; onChange: (value: LoadingPolicy) => void;
}) {
  return <fieldset className="mil-loading-options">
    <legend>Feature loading</legend>
    {loadingOptions.map((option) => <label key={option.value} className={`mil-loading-option${value === option.value ? ' is-selected' : ''}`}>
      <input type="radio" name="mil-loading-policy" value={option.value} checked={value === option.value} disabled={option.value === 'mmap' && !hasPacks} onChange={() => onChange(option.value)} />
      <span><strong>{option.title}</strong><small>{option.description}</small></span>
    </label>)}
  </fieldset>;
}

export function ExperimentDetail({ workspace: w, record, initialTab = 'setup', onBack, onOpen, context = {} }: { workspace: Workspace; record: ModelExperiment; initialTab?: DevelopmentTab; onBack: () => void; onOpen: (id: string) => void; context?: PreparationContext }) {
  const project = w.project.id;
  const protocols = useConfigurations(project, 'protocol');
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const client = useQueryClient();
  const [selectedSpec, setSpec] = useState<MILExperimentSpec | null>(record.inputs);
  const [baseRevision, setBaseRevision] = useState(record.revision);
  const [copying, setCopying] = useState(false);
  const refresh = () => client.invalidateQueries({ predicate: (query) => query.queryKey.includes(project) });
  const spec = selectedSpec ?? (record.legacy || record.state !== 'active' ? initialSpec() : suggestedExperimentInputs(protocols.data?.configurations ?? [], featureBundles.data?.items ?? [], context));
  const name = record.name;
  const readOnly = record.legacy || record.state !== 'active';
  const [preview, setPreview] = useState<MILExperimentPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const [tab, setTab] = useState<DevelopmentTab>(initialTab);
  useEffect(() => setTab(initialTab), [initialTab]);
  function changeTab(next: DevelopmentTab) {
    setTab(next);
    if (typeof window !== 'undefined') window.history.replaceState(null, '', preparationLink('experiments', context, { experiment: record.id, tab: next === 'setup' ? 'inputs' : next }));
  }
  const protocol = protocols.data?.configurations.find((item) => item.id === spec.protocolId);
  const protocolSpec = protocol?.manifest.spec as ProtocolSpec | undefined;
  const bundle = featureBundles.data?.items.find((item) => item.id === spec.featureBundleId);
  const packs = bundle?.manifest.packs ?? [];
  const packIds = bundle?.manifest.spec.packArtifactIds ?? [];
  const dirty = !sameJSON(record.inputs, spec);
  const stale = baseRevision !== record.revision;
  const ready = Boolean(name.trim() && spec.protocolId && spec.featureBundleId);
  const loadingNeedsChoice = spec.loadingPolicy !== 'auto' || Boolean(spec.packArtifactId) || packs.length > 1 || Boolean(preview && !preview.canPlan);

  function edit(update: Partial<MILExperimentSpec>) {
    setSpec((current) => ({ ...(current ?? spec), ...update }));
    setPreview(null);
    setMessage('');
    setError(null);
  }
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setMessage('');
    try { await action(); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('The MIL plan could not be saved.')); }
    finally { setBusy(false); }
  }
  async function reviewAndSave(save: boolean) {
    setPreview(null);
    const result = await mil.preview(project, spec);
    setSpec(spec);
    setPreview(result);
    if (!save || !result.canPlan) return result.canPlan;
    const saved = await experiments.update(project, record.id, { name: record.name, notes: record.notes, tags: record.tags, inputs: spec, expectedRevision: baseRevision });
    setBaseRevision(saved.revision);
    client.setQueryData(['model-experiment', project, record.id], saved);
    setMessage('Experiment inputs saved. Continue to Batches to configure training.');
    await refresh();
    return true;
  }

  return <div className="clinical-workspace mil-workspace">
    <button className="text-button" onClick={onBack}>← All experiments</button>
    <PageHeader eyebrow="02 DEVELOP · EXPERIMENT" title={name} description={record.notes || 'Manage the inputs, batches, runs and results for this experiment.'}
      actions={<button className="btn btn-secondary" onClick={() => setCopying(true)}>Create from these inputs</button>} />
    <div className="experiment-detail-heading"><Badge>{experimentStatusLabel(record.status)}</Badge><Badge>{lifecycleLabel[record.state]}</Badge>{record.tags.map((tag) => <Badge key={tag}>{tag}</Badge>)}<span className="experiment-detail-id">{record.id} · revision {record.revision}</span></div>
    {record.legacy ? <p className="callout">This legacy record retains its original batch or draft identity. Create a new experiment from its inputs to organize new work; historical runs stay here.</p> : null}
    {record.state !== 'active' ? <p className="callout">This experiment is {lifecycleLabel[record.state].toLowerCase()}. Its saved history remains visible. Restore it to Active before editing or creating batches.</p> : null}
    {copying ? <CreateExperiment project={project} copy={record} onClose={() => setCopying(false)} onCreated={(item) => { client.setQueryData(['model-experiment', project, item.id], item); void refresh(); onOpen(item.id); }} /> : null}
    <details className="setup-details"><summary>Manage experiment</summary><ExperimentMetadata project={project} record={record} /><ExperimentLifecycle key={record.id} project={project} recordKey={record.key} state={record.state} name={name} /></details>
    <nav className="development-tabs" role="tablist" aria-label="Experiments">
      {developmentTabs.map((item) => <button key={item.id} id={`development-tab-${item.id}`} type="button" role="tab" aria-selected={tab === item.id} aria-controls="development-panel" className="btn btn-secondary btn-small" onClick={() => changeTab(item.id)}>{item.label}</button>)}
    </nav>
    <div role="tabpanel" id="development-panel" aria-labelledby={`development-tab-${tab}`}>
    <div hidden={tab !== 'setup'}>
    <SetupContext input="Saved development targets and verified features" output="A reproducible training batch">Targets and splits are already defined. Training seeds and model settings belong to each batch.</SetupContext>
    {context.bundleId ? <p className="muted">Prepared inputs are suggested only for experiments without saved inputs. Existing experiment inputs are retained. Review the selected protocol and feature bundle below.</p> : null}
    <ErrorNotice error={error ?? protocols.error ?? featureBundles.error} />
    {stale ? <p className="callout">This experiment changed since you opened its inputs. Reload the saved inputs before editing. <button className="text-button" onClick={() => { setSpec(record.inputs); setBaseRevision(record.revision); setPreview(null); }}>Reload saved inputs</button></p> : null}
    {readOnly ? <p className="muted">Inputs shown below are retained history; current input choices may hide archived versions.</p> : null}
    <SavedNotice>{message}</SavedNotice>
    <fieldset className="mil-plan-fields" disabled={busy || readOnly || stale}>
      <legend className="sr-only">MIL experiment plan</legend>
      <Panel title="Choose training inputs" subtitle="A single compatible pair is selected automatically. Review the versions below.">
        <div className="stack">
          <label className="label">Development protocol
            <select className="field" value={spec.protocolId} onChange={(event) => {
              const id = event.target.value;
              const selected = protocols.data?.configurations.find((item) => item.id === id);
              const compatible = bundle && selected && protocolBundleCompatible(selected, bundle);
              edit({ protocolId: id, ...(!compatible ? { featureBundleId: '', packArtifactId: null } : {}) });
            }}>
              <option value="">{protocols.isPending ? 'Loading targets…' : 'Choose saved targets and splits'}</option>
              {spec.protocolId && !protocol ? <option value={spec.protocolId} disabled>Retained protocol (archived or unavailable to new selections)</option> : null}
              {(protocols.data?.configurations ?? []).map((item) => <option key={item.id} value={item.id}>{configurationVersionLabel(item)}</option>)}
            </select>
          </label>
          <label className="label">Feature bundle
            <select className="field" value={spec.featureBundleId} disabled={!protocol} onChange={(event) => edit({ featureBundleId: event.target.value, packArtifactId: null })}>
              <option value="">{featureBundles.isPending ? 'Loading features…' : 'Choose verified features'}</option>
              {spec.featureBundleId && !bundle ? <option value={spec.featureBundleId} disabled>Retained feature bundle (archived or unavailable to new selections)</option> : null}
              {(featureBundles.data?.items ?? []).map((item) => {
                const incompatible = Boolean(protocol && !protocolBundleCompatible(protocol, item));
                return <option key={item.id} value={item.id} disabled={item.current === false || incompatible}>{versionLabelText(item, 'Feature bundle')} · {item.manifest.summary.packCount} pack(s){item.current === false ? ' · needs verification' : incompatible ? ' · different protocol inputs' : ''}</option>;
              })}
            </select>
          </label>
          {!protocol || !bundle ? <p className="callout">Prepare missing inputs in <a href="#cohort">Targets &amp; splits</a> or <a href="#features">Features</a>, then return here.</p> : null}
          {protocolSpec ? <div className="mil-inherited-protocol"><strong>Inherited development protocol</strong><div className="mil-protocol-summary"><Badge>{taskLabel(protocolSpec.target.task)}</Badge><Badge>Target: {protocolSpec.target.field}</Badge><Badge>{unitLabel(protocolSpec.target.unit)} predictions</Badge><Badge>{splitModeLabel(protocolSpec.split.mode)}</Badge><Badge>Split seeds: {protocolSpec.split.seeds.join(', ')}</Badge></div><p className="muted">Targets, labels, and development memberships are fixed by this protocol. Training seeds and optimizer settings belong to each batch. <a href="#cohort">Open protocol</a></p></div> : null}
          {bundle ? <div className="mil-bundle-summary"><Badge>{bundle.manifest.summary.slideCount.toLocaleString()} slides</Badge><Badge>{bundle.manifest.summary.patchCount.toLocaleString()} patches</Badge><Badge>{bundle.manifest.summary.dimensions ?? '?'} dimensions</Badge><Badge>{Array.isArray(bundle.manifest.summary.dtype) ? bundle.manifest.summary.dtype.join(', ') : bundle.manifest.summary.dtype ?? 'Unknown dtype'}</Badge></div> : null}
          {bundle?.findings?.length ? <Findings findings={bundle.findings} /> : null}
          {protocolSpec?.featurePackId ? <p className="callout">This older protocol pins pack <code>{protocolSpec.featurePackId}</code>. The selected bundle must include that pack, and this plan must retain its loading source.</p> : null}
        </div>
      </Panel>
      <details className="setup-details" open={loadingNeedsChoice}>
        <summary>Advanced: feature loading <span className="muted">· {spec.loadingPolicy === 'auto' ? 'Automatic' : spec.loadingPolicy === 'mmap' ? 'Packed mmap' : 'Original files'}</span></summary>
        <LoadingOptions value={spec.loadingPolicy} hasPacks={packIds.length > 0} onChange={(loadingPolicy) => edit({ loadingPolicy, ...(loadingPolicy === 'native' ? { packArtifactId: null } : {}) })} />
        {spec.loadingPolicy !== 'native' && packIds.length > 0 ? <label className="label mil-pack-select">Pack in this bundle
          <select className="field" value={spec.packArtifactId ?? ''} onChange={(event) => edit({ packArtifactId: event.target.value || null })}>
            <option value="">{spec.loadingPolicy === 'mmap' ? 'Choose a pack' : packIds.length === 1 ? 'Auto-select when source precision is preserved' : 'Choose a pack — multiple versions are included'}</option>
            {spec.packArtifactId && !packIds.includes(spec.packArtifactId) ? <option value={spec.packArtifactId} disabled>Saved pack is not in this bundle</option> : null}
            {packIds.map((id) => {
              const pack = packs.find((item) => item.id === id);
              return <option key={id} value={id}>{pack ? `${pack.outputDtype} · ${pack.outputPath}` : id}</option>;
            })}
          </select>
        </label> : null}
        <p className="muted mil-memory-note">Packed mmap can read a pack larger than available RAM. Full RAM/GPU preloading and resource-based tuning are not enabled.</p>
      </details>
    </fieldset>
    <Panel title="Check & continue" subtitle="Check slide coverage and compatibility before choosing training settings.">
      {preview ? <div className="mil-plan-review">
        <Findings findings={preview.findings} />
        {preview.canPlan ? <p className="science-success">Resolved source: <strong>{preview.resolvedLoadingPolicy === 'mmap' ? 'Packed mmap' : 'Original feature files'}</strong>{preview.packArtifactId ? <> · <code>{preview.packArtifactId}</code></> : null}</p> : null}
      </div> : <p className="muted">Review checks the current files and the immutable bundle references.</p>}
      <div className="inline-actions">
        <button type="button" className="btn btn-primary" disabled={busy || !ready || readOnly || stale} onClick={() => void run(async () => { if (await reviewAndSave(true)) changeTab('batches'); })}>{busy ? 'Checking inputs…' : 'Check inputs & continue'}<Icon name="arrow" size={16} /></button>
        <button type="button" className="text-button" disabled={busy || !ready || !dirty || readOnly || stale} onClick={() => void run(async () => { await reviewAndSave(true); })}>Save inputs for later</button>
      </div>
      <p className="muted">Save the experiment inputs before creating a batch. Each frozen batch preserves its exact input and training snapshot. Training starts only when you choose Launch batch.</p>
    </Panel>
    </div>
    <div hidden={tab === 'setup'}><DevelopmentBatches project={project} inputs={record.inputs ?? initialSpec()} experimentName={name} experimentId={record.id} experimentRevision={record.revision} ownedBatches={record.batches} ownedDrafts={record.drafts} executionImplemented={record.executionImplemented === true} readOnly={readOnly} tab={tab === 'setup' ? 'batches' : tab} onOpenSetup={() => changeTab('setup')} onRestoreInputs={(inputs) => { if (!sameJSON(record.inputs, inputs)) { setSpec(inputs); setPreview(null); changeTab('setup'); setMessage('This batch uses different inputs. Review and save them before creating the copy.'); } }} /></div>
    </div>
    <details className="setup-details"><summary>Exact input history &amp; configuration snapshots</summary>
      <p className="muted">These saved values belong to this experiment and its batches. Archived inputs remain inspectable; current project defaults are never substituted.</p>
      <h3>Saved experiment inputs</h3><pre className="experiment-snapshot">{JSON.stringify(record.inputSnapshot ?? record.inputs, null, 2)}</pre>
      {record.batches.map((batch) => <details key={batch.id}><summary>{batch.manifest.spec.batchName} · {batch.id} · {lifecycleLabel[batch.state]}</summary><pre className="experiment-snapshot">{JSON.stringify({ inputSnapshot: batch.inputSnapshot, spec: batch.manifest.spec, configurations: batch.manifest.configurations, splitPlans: batch.manifest.splitPlans }, null, 2)}</pre></details>)}
      {record.legacy && record.drafts.map((draft) => <pre key={draft.id} className="experiment-snapshot">{JSON.stringify(draft, null, 2)}</pre>)}
    </details>
    <p className="muted">After reviewing results, continue to <a href={`#post-development?experiment=${encodeURIComponent(record.id)}`}>Build predictors for this experiment</a>{record.predictorId ? <> · {record.predictors?.length ?? (Object.keys(record.predictorIds ?? {}).length || 1)} published predictor(s)</> : null}.</p>
  </div>;
}

export function experimentRoute(hash: string, fallback: DevelopmentTab = 'setup') {
  const params = new URLSearchParams(hash.split('?')[1] ?? '');
  const value = params.get('tab');
  const tab = value === 'inputs' ? 'setup' : developmentTabs.some((item) => item.id === value) ? value as DevelopmentTab : fallback;
  return { id: params.get('experiment') ?? '', tab };
}
export default function LocalExperiments({ workspace, initialTab = 'setup' }: { workspace: Workspace; initialTab?: DevelopmentTab }) {
  const context = usePreparationContext();
  const [route, setRoute] = useState(() => experimentRoute(typeof window === 'undefined' ? '' : window.location.hash, initialTab));
  useEffect(() => {
    const update = () => setRoute(experimentRoute(window.location.hash, initialTab));
    window.addEventListener('hashchange', update);
    window.addEventListener('popstate', update);
    return () => { window.removeEventListener('hashchange', update); window.removeEventListener('popstate', update); };
  }, [initialTab]);
  const detail = useQuery({ queryKey: ['model-experiment', workspace.project.id, route.id], queryFn: () => experiments.get(workspace.project.id, route.id), enabled: Boolean(route.id), refetchInterval: 5000 });
  function open(id: string) { window.location.hash = preparationLink('experiments', context, id ? { experiment: id, tab: 'inputs' } : {}); setRoute({ id, tab: 'setup' }); }
  if (!route.id) return <><PreparationNotice context={context} /><ExperimentRegistry project={workspace.project.id} onOpen={open} /></>;
  if (!detail.data) return <div className="clinical-workspace"><button className="text-button" onClick={() => open('')}>← All experiments</button><ErrorNotice error={detail.error} />{detail.isPending ? <p role="status">Loading experiment…</p> : null}</div>;
  return <><PreparationNotice context={context} /><ErrorNotice error={detail.error} /><ExperimentDetail key={`${route.id}:${context.datasetId ?? ''}:${context.protocolId ?? ''}:${context.bundleId ?? ''}`} workspace={workspace} record={detail.data} initialTab={route.tab} onBack={() => open('')} onOpen={open} context={context} /></>;
}
