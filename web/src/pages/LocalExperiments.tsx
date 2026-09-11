import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { scientific, type ProtocolSpec, type ScientificDraft } from '../api/scientific';
import { bundles } from '../api/bundles';
import { mil, type LoadingPolicy, type MILExperimentPreview, type MILExperimentSpec } from '../api/mil';
import { sameJSON } from '../lib/json';
import { configurationVersionLabel, versionLabelText } from '../lib/versionLabels';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import { DraftSelect, Findings, SavedNotice, useConfigurations, useDrafts, useRefreshScientific } from '../components/ScientificUI';
import { Settings } from './LocalWorkspace';
import './LocalExperiments.css';

const initialSpec = (): MILExperimentSpec => ({
  protocolId: '', featureBundleId: '', loadingPolicy: 'auto', packArtifactId: null,
});

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

export default function LocalExperiments({ workspace: w }: { workspace: Workspace }) {
  const project = w.project.id;
  const protocols = useConfigurations(project, 'protocol');
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const drafts = useDrafts(project);
  const refresh = useRefreshScientific(project);
  const [spec, setSpec] = useState<MILExperimentSpec>(initialSpec);
  const [name, setName] = useState(`${w.project.name} MIL plan`);
  const [draft, setDraft] = useState<ScientificDraft<MILExperimentSpec> | null>(null);
  const [preview, setPreview] = useState<MILExperimentPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const protocol = protocols.data?.configurations.find((item) => item.id === spec.protocolId);
  const protocolSpec = protocol?.manifest.spec as ProtocolSpec | undefined;
  const bundle = featureBundles.data?.items.find((item) => item.id === spec.featureBundleId);
  const packs = bundle?.manifest.packs ?? [];
  const packIds = bundle?.manifest.spec.packArtifactIds ?? [];
  const savedDrafts = (drafts.data?.drafts ?? []).filter((item) => item.payload.type === 'mil-experiment');
  const dirty = !draft || draft.name !== name.trim() || !sameJSON(draft.payload.spec, spec);
  const ready = Boolean(name.trim() && spec.protocolId && spec.featureBundleId);

  function edit(update: Partial<MILExperimentSpec>) {
    setSpec((current) => ({ ...current, ...update }));
    setPreview(null);
    setMessage('');
    setError(null);
  }
  function reset() {
    setSpec(initialSpec());
    setName(`${w.project.name} MIL plan`);
    setDraft(null);
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
    setPreview(result);
    if (!save || !result.canPlan) return;
    const saved = await scientific.saveDraft(project, {
      kind: 'experiment', name: name.trim(), payload: { type: 'mil-experiment', spec },
    }, draft?.status === 'editable' ? draft : undefined);
    setDraft(saved);
    setName(saved.name);
    setMessage('MIL plan saved on the server. No training job was started.');
    await refresh();
  }

  return <div className="clinical-workspace mil-workspace">
    <PageHeader eyebrow="04 / EXPERIMENT DESIGN" title="MIL experiments"
      description="Choose a frozen protocol and feature bundle, then decide how this experiment will read its features."
      actions={<button type="button" className="btn btn-secondary" disabled={busy} onClick={reset}><Icon name="plus" /> New MIL plan</button>} />
    <div className="science-toolbar">
      <DraftSelect drafts={savedDrafts} value={draft?.id ?? ''} disabled={busy} onChange={(id) => {
        if (!id) { reset(); return; }
        void run(async () => {
          const saved = await scientific.draft<MILExperimentSpec>(project, id);
          if (saved.payload.type !== 'mil-experiment') throw new Error('This draft is not a MIL experiment plan.');
          setDraft(saved); setName(saved.name); setSpec(saved.payload.spec); setPreview(null);
        });
      }} />
      {draft ? <Badge>{draft.status} · revision {draft.revision}{dirty ? ' · unsaved changes' : ''}</Badge> : <Badge>New plan</Badge>}
    </div>
    <ErrorNotice error={error ?? protocols.error ?? featureBundles.error ?? drafts.error} />
    <SavedNotice>{message}</SavedNotice>
    <fieldset className="mil-plan-fields" disabled={busy}>
      <legend className="sr-only">MIL experiment plan</legend>
      <Panel title="1. Experiment inputs" subtitle="Frozen versions keep the data and feature contents explicit.">
        <div className="stack">
          <label className="label">Plan name<input className="field" value={name} maxLength={120} onChange={(event) => { setName(event.target.value); setMessage(''); }} /></label>
          <label className="label">Frozen protocol
            <select className="field" value={spec.protocolId} onChange={(event) => {
              const id = event.target.value;
              const selected = protocols.data?.configurations.find((item) => item.id === id);
              const selectedSpec = selected?.manifest.spec as ProtocolSpec | undefined;
              const compatible = bundle && selected && bundle.manifest.datasetId === selected.manifest.datasetId && (!selectedSpec?.featureSetId || selectedSpec.featureSetId === bundle.manifest.spec.featureSetId);
              edit({ protocolId: id, ...(!compatible ? { featureBundleId: '', packArtifactId: null } : {}) });
            }}>
              <option value="">{protocols.isPending ? 'Loading protocols…' : 'Choose a frozen protocol'}</option>
              {spec.protocolId && !protocol ? <option value={spec.protocolId} disabled>Saved protocol unavailable</option> : null}
              {(protocols.data?.configurations ?? []).map((item) => <option key={item.id} value={item.id}>{configurationVersionLabel(item)}</option>)}
            </select>
          </label>
          <label className="label">Frozen feature bundle
            <select className="field" value={spec.featureBundleId} disabled={!protocol} onChange={(event) => edit({ featureBundleId: event.target.value, packArtifactId: null })}>
              <option value="">{featureBundles.isPending ? 'Loading feature bundles…' : 'Choose a frozen feature bundle'}</option>
              {spec.featureBundleId && !bundle ? <option value={spec.featureBundleId} disabled>Saved bundle unavailable</option> : null}
              {(featureBundles.data?.items ?? []).map((item) => {
                const incompatible = Boolean(protocol && (item.manifest.datasetId !== protocol.manifest.datasetId || (protocolSpec?.featureSetId && protocolSpec.featureSetId !== item.manifest.spec.featureSetId)));
                return <option key={item.id} value={item.id} disabled={item.current === false || incompatible}>{versionLabelText(item, 'Feature bundle')} · {item.manifest.summary.packCount} pack(s){item.current === false ? ' · needs verification' : incompatible ? ' · different protocol inputs' : ''}</option>;
              })}
            </select>
          </label>
          <p className="muted">Prepare and freeze bundles in <a href="#features">PFM &amp; features</a>. Set targets and patient assignments in <a href="#cohort">Target &amp; split</a>.</p>
          {bundle ? <div className="mil-bundle-summary"><Badge>{bundle.manifest.summary.slideCount.toLocaleString()} slides</Badge><Badge>{bundle.manifest.summary.patchCount.toLocaleString()} patches</Badge><Badge>{bundle.manifest.summary.dimensions ?? '?'} dimensions</Badge><Badge>{Array.isArray(bundle.manifest.summary.dtype) ? bundle.manifest.summary.dtype.join(', ') : bundle.manifest.summary.dtype ?? 'Unknown dtype'}</Badge></div> : null}
          {bundle?.findings?.length ? <Findings findings={bundle.findings} /> : null}
          {protocolSpec?.featurePackId ? <p className="callout">This older protocol pins pack <code>{protocolSpec.featurePackId}</code>. The selected bundle must include that pack, and this plan must retain its loading source.</p> : null}
        </div>
      </Panel>
      <Panel title="2. Loading policy" subtitle="These settings belong to this MIL plan; the frozen feature bundle stays unchanged.">
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
      </Panel>
    </fieldset>
    <Panel title="3. Review & save" subtitle="Check that the protocol, bundle, and loading source still match before saving.">
      {preview ? <div className="mil-plan-review">
        <Findings findings={preview.findings} />
        {preview.canPlan ? <p className="science-success">Resolved source: <strong>{preview.resolvedLoadingPolicy === 'mmap' ? 'Packed mmap' : 'Original feature files'}</strong>{preview.packArtifactId ? <> · <code>{preview.packArtifactId}</code></> : null}</p> : null}
      </div> : <p className="muted">Review checks the current files and the immutable bundle references.</p>}
      <div className="inline-actions">
        <button type="button" className="btn btn-secondary" disabled={busy || !ready} onClick={() => void run(() => reviewAndSave(false))}>Review inputs</button>
        <button type="button" className="btn btn-primary" disabled={busy || !ready || !preview?.canPlan || !dirty} onClick={() => void run(() => reviewAndSave(true))}><Icon name="check" />{busy ? 'Checking…' : 'Save MIL plan'}</button>
      </div>
      <p className="callout mil-execution-note">MIL training is not implemented in this local workflow yet. Saving persists the plan and its selected versions; it does not launch a job.</p>
    </Panel>
    <details className="mil-initial-settings"><summary>Initial project preferences</summary><Settings workspace={w} /></details>
  </div>;
}
