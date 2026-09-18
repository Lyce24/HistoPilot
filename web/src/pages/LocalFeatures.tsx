import { StageBackButton, StageContinueButton, StageCreateButton } from '../components/StageActions';
import { useState } from 'react';
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
import SlideListField from '../components/SlideListField';
import { slideListReady } from '../api/slideLists';
import { editFeatureSource } from '../lib/featureSource';
import FeatureFolderExamples from '../components/FeatureFolderExamples';
import VersionLabelEditor from '../components/VersionLabelEditor';
import FeatureBundlePreparation from '../components/FeatureBundlePreparation';
import FeatureBundleLibrary, { type FeatureBundleLibraryFilters } from '../components/FeatureBundleLibrary';
import SetupContext from '../components/SetupContext';
import { StagePage, StageSteps, useStageLibrary } from '../components/StageWorkflow';
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
    datasetId: context.datasetId ?? null,
    path: w.sources.find((source) => source.role === 'features')?.path ?? '',
    encoderId: undefined, fileSuffix: '.h5', idSuffix: '', recursive: true, layout: 'auto',
  });
  const [view, setView] = useState<'bundles' | 'detail' | 'source' | 'library' | 'add'>('bundles');
  const [started, setStarted] = useState(false);
  const [bundleStarted, setBundleStarted] = useState(false);
  const [extractionStarted, setExtractionStarted] = useState(false);
  const [sourceStep, setSourceStep] = useState<'settings' | 'coverage'>('settings');
  const [bundlePage, setBundlePage] = useState<'packing' | 'review'>('packing');
  const [bundleReviewReady, setBundleReviewReady] = useState(false);
  const [mode, setMode] = useState<'extract' | 'attach'>('attach');
  const [preview, setPreview] = useState<FeaturePreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [bundleBusy, setBundleBusy] = useState(false);
  useStageLibrary(() => { if (!busy && !bundleBusy) { setView('bundles'); setSelectedBundle(''); } });
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const [selected, setSelected] = useState('');
  const [selectedBundle, setSelectedBundle] = useState('');
  const [libraryFilters, setLibraryFilters] = useState<FeatureBundleLibraryFilters>({ search: '', status: '', sort: 'recent' });
  const [bundleSeed, setBundleSeed] = useState<{ featureId: string; packIds: string[]; revision: number } | null>(null);
  const versions = configurations.data?.configurations ?? [];
  const bundleItems = frozenBundles.data?.items ?? [];
  const activeView = view;
  const configuration = versions.find((item) => item.id === selected) ?? versions[0];
  const selectedId = configuration?.id ?? '';
  const dataset = datasets.data?.datasets.find((item) => item.id === configuration?.manifest.datasetId);
  function startPreparation() {
    setStarted(true); setView(versions.length ? 'source' : 'add'); setSourceStep('settings'); setMessage('');
  }
  function addSource() {
    setStarted(true); setView('add'); setSourceStep('settings'); setMessage('');
  }
  function edit(update: Partial<FeatureSpec>) {
    setSpec((current) => editFeatureSource(current, update));
    setPreview(null); setSourceStep('settings');
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
    setView('library'); setBundlePage('packing'); setStarted(true); setBundleStarted(true);
    if (identity === selectedId) return;
    setBundleReviewReady(false);
    setSelected(identity); setBundleSeed(null); setMessage('');
  }
  function prepareBundle(featureId?: string, packIds: string[] = []) {
    const sourceId = featureId;
    if (sourceId) {
      setSelected(sourceId);
      setBundleSeed((current) => ({ featureId: sourceId, packIds, revision: (current?.revision ?? 0) + 1 }));
      setView('library'); setBundlePage('packing'); setBundleReviewReady(false); setStarted(true); setBundleStarted(true);
    } else startPreparation();
    setMessage('');
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
    <div className="clinical-workspace feature-workspace">
      <PageHeader
        eyebrow="01 PREPARE"
        title={activeView === 'bundles' ? 'Slide features' : 'Prepare slide features'}
        description={activeView === 'bundles' ? 'Open a feature bundle or create one for your experiments.' : 'Choose existing features or extract them, check slide coverage, then freeze a feature bundle.'}
        actions={activeView === 'bundles' ? <StageCreateButton type="button" disabled={busy || bundleBusy} onClick={() => prepareBundle()}>Create feature bundle</StageCreateButton> : <StageBackButton disabled={busy || bundleBusy} onClick={() => { setView('bundles'); setSelectedBundle(''); }}>Back to feature bundles</StageBackButton>}
      />
      {activeView !== 'bundles' ? <SetupContext input="Slide images or extracted features; a dataset is optional" output="A verified feature bundle for model development">
        A bundle saves verified features and optional packs for reuse. In Targets & splits, choose a dataset and bundle; their shared slides form the training cohort.
      </SetupContext> : null}
      <PreparationNotice context={context} />
      {context.datasetId ? <p className="muted">All project bundles are available. The selected dataset is an optional slide filter when adding features.</p> : null}
      <ErrorNotice error={error ?? datasets.error ?? configurations.error ?? frozenBundles.error} />
      <SavedNotice>{message}</SavedNotice>
      {activeView !== 'bundles' && activeView !== 'detail' && !(activeView === 'add' && mode === 'extract') ? <StageSteps label="Feature bundle steps" current={activeView === 'library' ? bundlePage : activeView === 'add' && sourceStep === 'coverage' ? 'coverage' : 'source'} disabled={busy || bundleBusy}
        steps={[
          { id: 'source', title: 'Choose features', description: 'Reuse, attach, or extract a source' },
          { id: 'coverage', title: 'Review coverage', description: 'Check newly attached features', disabled: !preview },
          { id: 'packing', title: 'Validate & choose packs', description: 'Packing is optional', disabled: !bundleStarted || !configuration },
          { id: 'review', title: 'Review & freeze', description: 'Save a verified bundle', disabled: !bundleReviewReady },
        ]}
        onChange={(step) => {
          if (step === 'source') setView('source');
          else if (step === 'coverage' && preview) { setSourceStep('coverage'); setView('add'); }
          else if (step === 'packing' && configuration && bundleStarted) { setBundlePage('packing'); setView('library'); }
          else if (step === 'review' && bundleReviewReady) { setBundlePage('review'); setView('library'); }
        }} /> : null}
      <StagePage pageKey={`${activeView}:${sourceStep}:${bundlePage}:${selectedBundle}`} className="pfm-workspace">
      {configurations.isPending || frozenBundles.isPending ? <p className="muted" role="status">Loading feature sources and bundles…</p> : <>
      {activeView === 'bundles' || activeView === 'detail' ? <section className="pfm-content" aria-label="Frozen feature bundles">
        <FeatureBundleLibrary project={project} items={bundleItems} features={versions} selectedId={activeView === 'detail' ? selectedBundle : ''} onSelect={(id) => { setSelectedBundle(id); setView('detail'); }} onPrepare={prepareBundle} context={context}
          filters={libraryFilters} onFiltersChange={setLibraryFilters}
          onRefresh={() => { void Promise.all([frozenBundles.refetch(), configurations.refetch()]); }} refreshBusy={frozenBundles.isFetching || configurations.isFetching} />
      </section> : null}
      {activeView === 'source' ? <Panel title="Choose a feature source" subtitle="Reuse an inspected source, or attach and inspect a new source for this bundle." actions={<StageCreateButton type="button" onClick={addSource}>Add feature source</StageCreateButton>}>
        {versions.length ? <div className="pfm-source-list">{versions.map((item) => <button type="button" className="pfm-version-card" key={item.id} onClick={() => prepareBundle(item.id)}>
          <strong>{configurationVersionLabel(item)}</strong><span>{featureEncoder(item)}</span><small className="mono">{item.manifest.layout?.featureDirectory ?? (item.manifest.spec as FeatureSpec).path}</small><span>Use this source <Icon name="arrow" size={14} /></span>
        </button>)}</div> : <EmptyState title="No inspected feature sources yet" description="Add features from a folder or extract them with a pathology foundation model." />}
      </Panel> : null}
      {started ? <>
      <section hidden={activeView !== 'add'} className="pfm-content" aria-label="Add features">
        {sourceStep === 'coverage' ? <StageBackButton type="button" className="pfm-back" disabled={busy} onClick={() => setSourceStep('settings')}>Back to feature settings</StageBackButton> : null}
        <div hidden={sourceStep !== 'settings'} className="pfm-content">
          <div className="pfm-modes" role="group" aria-label="Where will the features come from?">
            <button type="button" disabled={busy} className={`pfm-mode ${mode === 'attach' ? 'is-selected' : ''}`} aria-pressed={mode === 'attach'} onClick={() => setMode('attach')}>
              <Icon name="folder" size={25} /><span><strong>Use existing features</strong><small>Choose a folder of extracted slide features</small></span>
            </button>
            <button type="button" disabled={busy} className={`pfm-mode ${mode === 'extract' ? 'is-selected' : ''}`} aria-pressed={mode === 'extract'} onClick={() => { setExtractionStarted(true); setMode('extract'); }}>
              <Icon name="features" size={25} /><span><strong>Extract with a PFM</strong><small>Generate features from whole-slide images</small></span>
            </button>
          </div>
          <div className="pfm-content" hidden={mode !== 'extract'}>
            {extractionStarted ? <TridentExtraction workspace={w} initialDatasetId={context.datasetId} datasets={datasets.data?.datasets ?? []} onAttach={(input) => {
              edit({ ...input, slideList: null, slideListPath: null, layout: 'auto', fileSuffix: '.h5', recursive: false, idSuffix: '', coordinatesPath: undefined });
              setMode('attach'); setView('add'); setSourceStep('settings');
            }} /> : null}
          </div>
          <div className="pfm-content" hidden={mode !== 'attach'}>
            {spec.sourceExtractionJobId ? <p className="callout" role="status"><strong>Extraction outputs selected.</strong> Inspect their coverage, then {spec.featureKind === 'slide' ? 'validate and freeze a slide-embedding bundle' : 'choose packing and freeze a bundle'}. Extraction settings and validation evidence will be linked automatically.</p> : null}
      <fieldset className="science-fieldset" disabled={busy}>
        <Panel
          title="Choose your existing features"
          subtitle="Choose a feature folder. Keep all eligible slides, or narrow them with a slide list and an optional dataset filter."
        >
          <div className="stack">
            <label className="label">Feature contents<select className="field" value={spec.featureKind ?? 'patch'} onChange={(event) => edit({ featureKind: event.target.value as FeatureSpec['featureKind'] })}><option value="patch">Patch features with locations</option><option value="slide">One embedding per slide</option></select><small>{spec.featureKind === 'slide' ? 'A slide encoder produces one vector per slide. Slide models use these vectors; patch locations, patch attention and packing are unavailable.' : 'One feature vector per image patch, with matching coordinates for slide inspection.'}</small></label>
            <div className="science-grid-two">
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
              <SlideListField
                value={spec.slideList ?? (spec.slideListPath ? { path: spec.slideListPath } : null)}
                onChange={(slideList) => edit({ slideList, slideListPath: null })}
              />
              <DatasetSelect
                label="Dataset filter (optional)"
                versions={datasets.data?.datasets ?? []}
                value={spec.datasetId ?? ''}
                allowEmpty
                emptyLabel="No dataset filter"
                onChange={(datasetId) => edit({ datasetId: datasetId || null })}
              />
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
                {spec.featureKind !== 'slide' ? <div className="pfm-coordinate-picker">
                  <label className="label">
                    Coordinate directory (optional)
                    <input className="field mono" value={spec.coordinatesPath ?? ''} placeholder="Auto-detect embedded coords or sibling patches/" onChange={(event) => edit({ coordinatesPath: event.target.value || undefined })} />
                  </label>
                  <ServerFolderPicker label="Browse coordinates" purpose="storage" initialPath={spec.coordinatesPath || spec.path || undefined} onSelect={(coordinatesPath) => edit({ coordinatesPath })} />
                </div> : null}
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
            {!spec.path.trim() ? <p className="muted">Choose a feature folder to inspect. Its files are the initial selection; a slide list narrows them, and a dataset narrows whatever that produced.</p> : null}
            <StageContinueButton
              type="button"
              className="science-fit"
              disabled={!spec.path.trim() || !slideListReady(spec.slideList)}
              onClick={() =>
                void run(async () => {
                  setPreview(await scientific.featurePreview(project, spec));
                  setSourceStep('coverage');
                })
              }
            >
              {busy ? 'Inspecting features…' : 'Inspect & review features'}
            </StageContinueButton>
          </div>
        </Panel>
      </fieldset>
          </div>
        </div>
      {preview && sourceStep === 'coverage' ? (
        <Panel
          title="Review feature coverage"
          subtitle="Check selected slides and file structure. Full content verification follows during bundle preparation."
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
              note={`${preview.summary.slideCount} ${preview.summary.scope === 'dataset' ? 'dataset' : preview.summary.scope === 'list' ? 'listed' : 'eligible'} slides`}
            />
            <Metric
              label="Missing features"
              value={preview.summary.missingSlides}
              note={`${preview.summary.orphanFiles} files excluded by selection`}
            />
            <Metric
              label="Feature dimension"
              value={preview.summary.dimensions ?? 'Unresolved'}
              note={preview.layout?.encoderId ? `Encoder: ${preview.layout.encoderId}` : 'Read from array headers'}
            />
            <Metric
              label={spec.featureKind === 'slide' ? 'Slide embeddings' : 'Patch count'}
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
                <div><dt>Coordinates</dt><dd className="mono">{spec.featureKind === 'slide' ? 'Not applicable — one vector per slide' : preview.layout.coordinatesDirectory ?? 'Read from feature files when present'}</dd></div>
              </dl>
            </div>
          ) : null}
          <details>
            <summary>Inspect matched file headers</summary>
            <FeatureFiles files={preview.files} slideFeatures={spec.featureKind === 'slide'} />
          </details>
          <div className="science-savebar">
            <p>
              Continue with these features, validate their contents, and choose optional packs for the bundle.
            </p>
            <StageContinueButton
              type="button"
              disabled={busy || !preview.canFreeze}
              onClick={() => void registerSource()}
            >
              {busy ? 'Preparing source…' : 'Continue to validation & packs'}
            </StageContinueButton>
          </div>
        </Panel>
      ) : null}

      </section>
      {bundleStarted ? <section hidden={activeView !== 'library'} className="pfm-content" aria-label="Prepare feature bundle">
        {!versions.length ? <Panel title="Prepare a feature bundle" subtitle="Start with an existing feature folder or extract with a PFM.">
          <EmptyState title="Choose a feature source first" description="Review your features, choose optional packing, then freeze them together as a bundle." />
          <StageCreateButton className="science-fit" onClick={addSource}>Add feature source</StageCreateButton>
        </Panel> : <div className="pfm-library">
          <label className="label pfm-version-select" hidden={bundlePage === 'review'}>Feature source
            <select className="field" value={selectedId} disabled={bundleBusy} onChange={(event) => selectVersion(event.target.value)}>
              {versions.map((item) => <option key={item.id} value={item.id}>{configurationVersionLabel(item)} · {featureEncoder(item)}</option>)}
            </select>
            <small>Choose an inspected source to prepare its bundle.</small>
          </label>
          {configuration ? <div className="pfm-version-detail">
            <Panel title={bundlePage === 'review' ? 'Review feature bundle' : configurationVersionLabel(configuration)} subtitle={bundlePage === 'review' ? `Source: ${configurationVersionLabel(configuration)}. Review the verified contents before freezing.` : 'Decide which packs, if any, will be frozen with these features.'} actions={bundlePage !== 'review' ? <StageCreateButton type="button" size="small" tone="secondary" disabled={bundleBusy} onClick={addSource}>Add feature source</StageCreateButton> : undefined}>
              <dl className="pfm-version-facts" hidden={bundlePage === 'review'}>
                <div><dt>Dataset selection</dt><dd>{dataset ? datasetVersionLabel(dataset) : configuration.manifest.datasetId ? `Dataset · ${configuration.manifest.datasetId.slice(-8)}` : 'All selected slides; no dataset filter'}</dd></div>
                <div><dt>Encoder</dt><dd>{featureEncoder(configuration)}</dd></div>
                <div><dt>Source folder</dt><dd className="mono">{configuration.manifest.layout?.featureDirectory ?? (configuration.manifest.spec as FeatureSpec).path}</dd></div>
              </dl>
              {configuration.versionLabel?.note ? <p className="pfm-version-note">{configuration.versionLabel.note}</p> : null}
              <FeatureBundlePreparation onBusyChange={setBundleBusy} page={bundlePage} onPageChange={setBundlePage} onReviewReadyChange={setBundleReviewReady} showSteps={false} key={`${configuration.id}:${bundleSeed?.revision ?? 0}`} project={project} configuration={configuration} configurations={versions} onSelectVersion={selectVersion}
                initialPackIds={bundleSeed?.featureId === configuration.id ? bundleSeed.packIds : []}
                onFrozen={(bundle) => {
                  setSelectedBundle(bundle.id);
                  setView('detail');
                  setMessage(`Bundle “${bundle.versionLabel?.tag || 'Feature bundle'}” frozen.`);
                  window.location.hash = preparationLink('cohort', { datasetId: context.datasetId, bundleId: bundle.id, protocolId: context.protocolId, saved: 'bundle' });
                  window.scrollTo({ top: 0 });
                }} />
            </Panel>
            <details className="pfm-version-details" key={configuration.id}>
              <summary>Source details &amp; inspection</summary>
              <div className="stack">
                <VersionLabelEditor project={project} resourceType="configuration" resource={configuration} tagLabel="Feature source tag" />
                <details><summary>Source files &amp; inspection findings</summary><Findings findings={configuration.manifest.findings ?? []} /><FeatureFiles files={configuration.manifest.files ?? []} slideFeatures={(configuration.manifest.spec as FeatureSpec).featureKind === 'slide'} /></details>
                <details><summary>Configuration provenance</summary><pre className="code-block">{JSON.stringify(configuration, null, 2)}</pre></details>
                <StageContinueButton type="button" tone="secondary" className="science-fit" disabled={bundleBusy} onClick={() => { edit(configuration.manifest.spec as FeatureSpec); setMode('attach'); setView('add'); setSourceStep('settings'); }}>Inspect these settings as a new version</StageContinueButton>
              </div>
            </details>
          </div> : null}
        </div>}
      </section> : null}
      </> : null}
      </>}
      </StagePage>
    </div>
  );
}
function featureEncoder(configuration: Configuration) {
  return configuration.manifest.layout?.encoderId ?? (configuration.manifest.spec as FeatureSpec).encoderId ?? 'Encoder unspecified';
}
function FeatureFiles({ files, slideFeatures = false }: { files: FeaturePreview['files']; slideFeatures?: boolean }) {
  return (
    <div className="table-wrap pfm-feature-files">
      <table>
        <thead>
          <tr>
            <th>Slide_ID</th>
            <th>{slideFeatures ? 'Slide embeddings' : 'Patches'}</th>
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
                {slideFeatures ? 'Not applicable' : file.coordinateSource === 'embedded' ? 'Embedded' : file.coordinateSource === 'trident-patches' ? 'TRIDENT patches' : 'Unspecified'}
                {!slideFeatures && file.coordinateSpace === 'level0_pixels' ? <div className="pfm-coordinate-note">Level 0 pixels</div> : null}
              </td>
              <td className="mono local-path">{file.path}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
