import { useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { scientific } from '../api/scientific';
import type { Configuration, FeaturePreview, FeatureSpec } from '../api/scientific';
import { Badge, EmptyState, ErrorNotice, Icon, Metric, PageHeader, Panel } from '../components/ui';
import {
  DatasetSelect,
  Findings,
  SavedNotice,
  useConfigurations,
  useDatasets,
  useRefreshScientific,
} from '../components/ScientificUI';
import ServerFolderPicker from '../components/ServerFolderPicker';
import TridentExtraction from '../components/TridentExtraction';
import FeatureFolderExamples from '../components/FeatureFolderExamples';
import VersionLabelEditor from '../components/VersionLabelEditor';
import FeatureBundlePreparation from '../components/FeatureBundlePreparation';
import FeatureBundleLibrary from '../components/FeatureBundleLibrary';
import SetupContext from '../components/SetupContext';
import { bundles } from '../api/bundles';
import { configurationVersionLabel, datasetVersionLabel } from '../lib/versionLabels';
import { preparationLink, usePreparationContext, type PreparationContext } from '../lib/preparationRoute';
import PreparationNotice from '../components/PreparationNotice';
import './LocalFeatures.css';
export default function LocalFeatures({ workspace: w }: { workspace: Workspace }) {
  const context = usePreparationContext();
  return <FeaturesWorkspace key={`${context.datasetId ?? ''}:${context.protocolId ?? ''}`} workspace={w} context={context} />;
}
function FeaturesWorkspace({ workspace: w, context }: { workspace: Workspace; context: PreparationContext }) {
  const project = w.project.id;
  const datasets = useDatasets(project);
  const configurations = useConfigurations(project, 'feature');
  const frozenBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const refresh = useRefreshScientific(project);
  const [spec, setSpec] = useState<FeatureSpec>({
    datasetId: context.datasetId ?? w.dataset.id,
    path: w.sources.find((source) => source.role === 'features')?.path ?? '',
    encoderId: undefined, fileSuffix: '.h5', idSuffix: '', recursive: false, layout: 'auto',
  });
  const [view, setView] = useState<'bundles' | 'library' | 'add' | null>(null);
  const [mode, setMode] = useState<'extract' | 'attach'>('attach');
  const [preview, setPreview] = useState<FeaturePreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const [selected, setSelected] = useState('');
  const [selectedBundle, setSelectedBundle] = useState('');
  const [bundleSeed, setBundleSeed] = useState<{ featureId: string; packIds: string[]; revision: number } | null>(null);
  const workflowRef = useRef<HTMLDivElement>(null);
  const scopedDatasetId = context.datasetId ? spec.datasetId : undefined;
  const versions = (configurations.data?.configurations ?? []).filter((item) => !scopedDatasetId || item.manifest.datasetId === scopedDatasetId);
  const bundleItems = (frozenBundles.data?.items ?? []).filter((item) => !scopedDatasetId || item.manifest.datasetId === scopedDatasetId);
  const activeView = view ?? (bundleItems.length ? 'bundles' : versions.length ? 'library' : 'add');
  const configuration = versions.find((item) => item.id === selected) ?? versions[0];
  const selectedId = configuration?.id ?? '';
  const dataset = datasets.data?.datasets.find((item) => item.id === configuration?.manifest.datasetId);
  function focusWorkflow() {
    requestAnimationFrame(() => {
      workflowRef.current?.focus({ preventScroll: true });
      workflowRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
  function edit(update: Partial<FeatureSpec>) {
    setSpec((current) => ({
      ...current,
      ...(('datasetId' in update || 'path' in update || 'encoderId' in update || 'coordinatesPath' in update || 'layout' in update || 'fileSuffix' in update || 'idSuffix' in update || 'recursive' in update)
        && !('sourceExtractionJobId' in update) ? { sourceExtractionJobId: undefined } : {}),
      ...update,
    }));
    setPreview(null);
    setMessage('');
    setError(null);
  }
  async function run(action: () => Promise<void>) {
    setBusy(true); setError(null); setMessage('');
    try { await action(); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Feature inspection failed.')); }
    finally { setBusy(false); }
  }
  function selectVersion(identity: string) {
    setView('library');
    if (identity === selectedId) return;
    setSelected(identity); setBundleSeed(null); setMessage('');
  }
  function prepareBundle(featureId?: string, packIds: string[] = []) {
    const sourceId = featureId ?? configuration?.id;
    if (sourceId) {
      setSelected(sourceId);
      setBundleSeed((current) => ({ featureId: sourceId, packIds, revision: (current?.revision ?? 0) + 1 }));
      setView('library');
    } else setView('add');
    setMessage(''); focusWorkflow();
  }
  async function registerSource() {
    if (!preview?.canFreeze) return;
    await run(async () => {
      const existing = versions.find((item) => item.manifest.previewHash === preview.previewHash);
      const source = existing ?? await scientific.featureFreeze(project, spec, preview.previewHash,
        { tag: `Source ${(preview.layout?.encoderId || spec.encoderId || 'features').slice(0, 30)} ${preview.previewHash.slice(0, 12)}`, note: 'Feature source registered for bundle preparation.' },
        `bundle-source:${preview.previewHash}`);
      await refresh();
      setPreview(null); prepareBundle(source.id);
      setMessage('Feature source ready. Choose whether to include packs, then review and freeze your bundle.');
    });
  }
  return (
    <>
      <PageHeader
        eyebrow="MODEL INPUTS · FEATURES"
        title="Prepare slide features"
        description="Choose existing features or extract them, check slide coverage, then freeze a feature bundle."
      />
      <SetupContext input="A frozen dataset and slide features or images" output="A verified feature bundle for model development">
        A bundle saves the feature files and any optional packs together. Packing is optional; full feature validation is required either way.
      </SetupContext>
      <PreparationNotice context={context} />
      {context.datasetId ? <p className="muted">Feature sources and bundles shown here belong to the selected dataset. <a href="#features">Show all project features</a></p> : null}
      <nav className="pfm-navigation" aria-label="Feature sections">
        <button type="button" disabled={busy} aria-pressed={activeView === 'bundles'} className={activeView === 'bundles' ? 'is-selected' : ''} onClick={() => setView('bundles')}>
          <Icon name="lock" /> Frozen bundles <span>{bundleItems.length}</span>
        </button>
        <button type="button" disabled={busy} aria-pressed={activeView !== 'bundles'} className={activeView !== 'bundles' ? 'is-selected' : ''} onClick={() => setView(versions.length ? 'library' : 'add')}>
          <Icon name="plus" /> Prepare a bundle
        </button>
      </nav>
      <ErrorNotice error={error ?? datasets.error ?? configurations.error ?? frozenBundles.error} />
      <SavedNotice>{message}</SavedNotice>
      <div ref={workflowRef} className="pfm-workspace" tabIndex={-1}>
      {configurations.isPending || frozenBundles.isPending ? <p className="muted" role="status">Loading feature sources and bundles…</p> : <>
      <section hidden={activeView !== 'bundles'} className="pfm-content" aria-label="Frozen feature bundles">
        <FeatureBundleLibrary project={project} items={bundleItems} features={versions} selectedId={selectedBundle} onSelect={setSelectedBundle} onPrepare={prepareBundle} context={context} />
      </section>
      {activeView !== 'bundles' ? (
        <ol className="pfm-workflow-steps" aria-label="Add features workflow">
          <li aria-current={activeView === 'add' && !preview ? 'step' : undefined} className={activeView === 'add' && !preview ? 'is-current' : 'is-complete'}><span>{preview || activeView === 'library' ? <Icon name="check" size={14} /> : '1'}</span><div><strong>Choose features</strong><small>Use a folder or extract with a PFM</small></div></li>
          <li aria-current={activeView === 'add' && preview ? 'step' : undefined} className={activeView === 'library' ? 'is-complete' : preview ? 'is-current' : ''}><span>{activeView === 'library' ? <Icon name="check" size={14} /> : '2'}</span><div><strong>Review coverage</strong><small>Check the source against your dataset</small></div></li>
          <li aria-current={activeView === 'library' ? 'step' : undefined} className={activeView === 'library' ? 'is-current' : ''}><span>3</span><div><strong>Validate &amp; freeze</strong><small>Packing is optional</small></div></li>
        </ol>
      ) : null}
      <section hidden={activeView !== 'add'} className="pfm-content" aria-label="Add features">
        {preview ? <button type="button" className="text-button pfm-back" disabled={busy} onClick={() => { setPreview(null); focusWorkflow(); }}>← Back to feature settings</button> : null}
        <div hidden={Boolean(preview)} className="pfm-content">
          <div className="pfm-modes" role="group" aria-label="Where will the features come from?">
            <button type="button" disabled={busy} className={`pfm-mode ${mode === 'attach' ? 'is-selected' : ''}`} aria-pressed={mode === 'attach'} onClick={() => setMode('attach')}>
              <Icon name="folder" size={25} /><span><strong>Use existing features</strong><small>Choose a folder of extracted slide features</small></span>
            </button>
            <button type="button" disabled={busy} className={`pfm-mode ${mode === 'extract' ? 'is-selected' : ''}`} aria-pressed={mode === 'extract'} onClick={() => setMode('extract')}>
              <Icon name="features" size={25} /><span><strong>Extract with a PFM</strong><small>Generate features from whole-slide images</small></span>
            </button>
          </div>
          <div className="pfm-content" hidden={mode !== 'extract'}>
            <TridentExtraction workspace={context.datasetId ? { ...w, dataset: { ...w.dataset, id: context.datasetId } } : w} datasets={datasets.data?.datasets ?? []} onAttach={(input) => {
              edit({ ...input, layout: 'auto', fileSuffix: '.h5', recursive: false, idSuffix: '', coordinatesPath: undefined });
              setMode('attach'); setView('add'); focusWorkflow();
            }} />
          </div>
          <div className="pfm-content" hidden={mode !== 'attach'}>
            {spec.sourceExtractionJobId ? <p className="callout" role="status"><strong>Extraction outputs selected.</strong> Inspect their coverage, then choose packing and freeze a bundle. Extraction settings and validation evidence will be linked automatically.</p> : null}
      <fieldset className="science-fieldset" disabled={busy}>
        <Panel
          title="Choose your existing features"
          subtitle="Choose the dataset and folder to check. The next step lets you skip packing, bind an existing pack, or create one."
        >
          <div className="stack">
            <div className="science-grid-two">
              <DatasetSelect
                versions={datasets.data?.datasets ?? []}
                value={spec.datasetId}
                onChange={(datasetId) => edit({ datasetId })}
              />
              <label className="label">
                Feature directory or TRIDENT job root
                <input
                  className="field mono"
                  value={spec.path}
                  placeholder="/path/to/trident_output or /path/to/features_<encoder>"
                  onChange={(event) => edit({ path: event.target.value })}
                />
              </label>
              <div className="science-field-actions">
                <ServerFolderPicker
                  label="Browse feature folders"
                  purpose="storage"
                  onSelect={(path) => edit({ path })}
                />
              </div>
            </div>
            <p className="muted">Features are matched to Slide_ID. TRIDENT folders and encoder metadata are detected automatically; source files stay in place.</p>
            <details className="pfm-source-details">
              <summary>Advanced file matching and encoder settings{spec.encoderId ? ` · ${spec.encoderId}` : ''}</summary>
              <div className="science-grid-two">
                <label className="label">
                  Encoder identity (optional)
                  <input className="field" value={spec.encoderId ?? ''} placeholder="Auto-detect, e.g. uni_v1" onChange={(event) => edit({ encoderId: event.target.value || undefined })} />
                  <small>Choose an encoder when a TRIDENT root contains several.</small>
                </label>
                <label className="label">
                  Folder layout
                  <select className="field" value={spec.layout ?? 'auto'} onChange={(event) => edit({ layout: event.target.value as FeatureSpec['layout'] })}>
                    <option value="auto">Auto-detect</option>
                    <option value="trident">TRIDENT output structure</option>
                    <option value="flat">Single feature directory</option>
                  </select>
                </label>
                <div className="pfm-coordinate-picker">
                  <label className="label">
                    Coordinate directory (optional)
                    <input className="field mono" value={spec.coordinatesPath ?? ''} placeholder="Auto-detect embedded coords or sibling patches/" onChange={(event) => edit({ coordinatesPath: event.target.value || undefined })} />
                  </label>
                  <ServerFolderPicker label="Browse coordinates" purpose="storage" initialPath={spec.coordinatesPath || spec.path || undefined} onSelect={(coordinatesPath) => edit({ coordinatesPath })} />
                </div>
                <label className="label">
                  Feature filename extension
                  <select className="field" value={spec.fileSuffix} onChange={(event) => edit({ fileSuffix: event.target.value as FeatureSpec['fileSuffix'] })}>
                    <option value=".h5">.h5 (TRIDENT default)</option>
                    <option value=".hdf5">.hdf5</option>
                  </select>
                </label>
                <label className="label">
                  Suffix to remove from filename stem (optional)
                  <input className="field mono" value={spec.idSuffix} placeholder="e.g. _features" onChange={(event) => edit({ idSuffix: event.target.value })} />
                </label>
                <label className="science-check">
                  <input type="checkbox" checked={spec.recursive} onChange={(event) => edit({ recursive: event.target.checked })} />
                  <span>Include subfolders<small>For a single encoder stored across nested directories.</small></span>
                </label>
              </div>
              <FeatureFolderExamples />
            </details>
            {!spec.datasetId || !spec.path.trim() ? <p className="muted">Choose a frozen dataset and a feature folder to inspect coverage.</p> : null}
            <button
              type="button"
              className="btn btn-primary science-fit"
              disabled={!spec.datasetId || !spec.path.trim()}
              onClick={() =>
                void run(async () => {
                  setPreview(await scientific.featurePreview(project, spec));
                  focusWorkflow();
                })
              }
            >
              {busy ? 'Inspecting features…' : 'Inspect & review features'} <Icon name="arrow" />
            </button>
          </div>
        </Panel>
      </fieldset>
          </div>
        </div>
      {preview ? (
        <Panel
          title="Review feature coverage"
          subtitle="Check dataset coverage and file structure here. Full content verification follows during bundle preparation."
          actions={
            <Badge tone={preview.canFreeze ? 'green' : 'orange'}>
              {preview.canFreeze ? 'Coverage ready' : 'Resolve blocking findings'}
            </Badge>
          }
        >
          <div className="science-metrics">
            <Metric
              label="Matched slides"
              value={preview.summary.matchedSlides}
              note={`${preview.summary.slideCount} dataset slides`}
            />
            <Metric
              label="Missing features"
              value={preview.summary.missingSlides}
              note={`${preview.summary.orphanFiles} files have no matching slide`}
            />
            <Metric
              label="Feature dimension"
              value={preview.summary.dimensions ?? 'Unresolved'}
              note={preview.layout?.encoderId ? `Encoder: ${preview.layout.encoderId}` : 'Read from array headers'}
            />
            <Metric
              label="Patch count"
              value={preview.summary.patchCount.toLocaleString()}
              note="Header totals only"
            />
          </div>
          <Findings findings={preview.findings} />
          {preview.layout ? (
            <div className="trident-layout">
              <div><Icon name="folder" size={18} /><strong>{preview.layout.kind === 'trident' ? 'TRIDENT structure detected' : 'Feature directory'}</strong></div>
              <dl>
                <div><dt>Feature directory</dt><dd className="mono">{preview.layout.featureDirectory}</dd></div>
                <div><dt>Coordinates</dt><dd className="mono">{preview.layout.coordinatesDirectory ?? 'Read from feature files when present'}</dd></div>
              </dl>
            </div>
          ) : null}
          <details>
            <summary>Inspect matched file headers</summary>
            <FeatureFiles files={preview.files} />
          </details>
          <div className="science-savebar">
            <p>
              Save this inspected feature source, choose whether to include packs, then name and freeze the complete bundle.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || !preview.canFreeze}
              onClick={() => void registerSource()}
            >
              {busy ? 'Saving source…' : 'Save source & prepare bundle'} <Icon name="arrow" />
            </button>
          </div>
        </Panel>
      ) : null}

      </section>
      <section hidden={activeView !== 'library'} className="pfm-content" aria-label="Prepare feature bundle">
        {!versions.length ? <Panel title="Prepare a feature bundle" subtitle="Start with an existing feature folder or extract with a PFM.">
          <EmptyState title="Choose a feature source first" description="Review your features, choose optional packing, then freeze them together as a bundle." />
          <button type="button" className="btn btn-primary science-fit" onClick={() => { setView('add'); focusWorkflow(); }}><Icon name="plus" /> Add features</button>
        </Panel> : <div className="pfm-library">
          <label className="label pfm-version-select">Feature source
            <select className="field" value={selectedId} onChange={(event) => selectVersion(event.target.value)}>
              {versions.map((item) => <option key={item.id} value={item.id}>{configurationVersionLabel(item)} · {featureEncoder(item)}</option>)}
            </select>
            <small>Choose an inspected source to prepare its bundle.</small>
          </label>
          {configuration ? <div className="pfm-version-detail">
            <Panel title={configurationVersionLabel(configuration)} subtitle="Decide which packs, if any, will be frozen with these features." actions={<button type="button" className="btn btn-secondary btn-small" onClick={() => { setView('add'); focusWorkflow(); }}><Icon name="plus" /> Add feature source</button>}>
              <dl className="pfm-version-facts">
                <div><dt>Dataset</dt><dd>{dataset ? datasetVersionLabel(dataset) : `Dataset · ${configuration.manifest.datasetId.slice(-8)}`}</dd></div>
                <div><dt>Encoder</dt><dd>{featureEncoder(configuration)}</dd></div>
                <div><dt>Source folder</dt><dd className="mono">{configuration.manifest.layout?.featureDirectory ?? (configuration.manifest.spec as FeatureSpec).path}</dd></div>
              </dl>
              {configuration.versionLabel?.note ? <p className="pfm-version-note">{configuration.versionLabel.note}</p> : null}
              <FeatureBundlePreparation key={`${configuration.id}:${bundleSeed?.revision ?? 0}`} project={project} configuration={configuration} configurations={versions} onSelectVersion={selectVersion}
                initialPackIds={bundleSeed?.featureId === configuration.id ? bundleSeed.packIds : []}
                onFrozen={(bundle) => {
                  setSelectedBundle(bundle.id);
                  setView('bundles');
                  setMessage(`Bundle “${bundle.versionLabel?.tag || 'Feature bundle'}” frozen.`);
                  window.location.hash = preparationLink('experiments', { datasetId: bundle.manifest.datasetId, bundleId: bundle.id, protocolId: bundle.manifest.datasetId === context.datasetId ? context.protocolId : undefined, saved: 'bundle' });
                  window.scrollTo({ top: 0 });
                }} />
            </Panel>
            <details className="pfm-version-details" key={configuration.id}>
              <summary>Source details &amp; inspection</summary>
              <div className="stack">
                <VersionLabelEditor project={project} resourceType="configuration" resource={configuration} tagLabel="Feature source tag" />
                <details><summary>Source files &amp; inspection findings</summary><Findings findings={configuration.manifest.findings ?? []} /><FeatureFiles files={configuration.manifest.files ?? []} /></details>
                <details><summary>Configuration provenance</summary><pre className="code-block">{JSON.stringify(configuration, null, 2)}</pre></details>
                <button type="button" className="btn btn-secondary science-fit" onClick={() => { edit(configuration.manifest.spec as FeatureSpec); setMode('attach'); setView('add'); focusWorkflow(); }}>Inspect these settings as a new version <Icon name="arrow" /></button>
              </div>
            </details>
          </div> : null}
        </div>}
      </section>
      </>}
      </div>
    </>
  );
}
function featureEncoder(configuration: Configuration) {
  return configuration.manifest.layout?.encoderId ?? (configuration.manifest.spec as FeatureSpec).encoderId ?? 'Encoder unspecified';
}
function FeatureFiles({ files }: { files: FeaturePreview['files'] }) {
  return (
    <div className="table-wrap pfm-feature-files">
      <table>
        <thead>
          <tr>
            <th>Slide_ID</th>
            <th>Patches</th>
            <th>Dimensions</th>
            <th>Type</th>
            <th>Coordinates</th>
            <th>Path</th>
          </tr>
        </thead>
        <tbody>
          {files.map((file) => (
            <tr key={file.slideId}>
              <td className="mono">{file.slideId}</td>
              <td>{file.patchCount.toLocaleString()}</td>
              <td>{file.dimensions}</td>
              <td>{file.dtype}</td>
              <td>
                {file.coordinateSource === 'embedded' ? 'Embedded' : file.coordinateSource === 'trident-patches' ? 'TRIDENT patches' : 'Unspecified'}
                {file.coordinateSpace === 'level0_pixels' ? <div className="pfm-coordinate-note">Level 0 pixels</div> : null}
              </td>
              <td className="mono local-path">{file.path}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
