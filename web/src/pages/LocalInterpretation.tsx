import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { interpretations, type GallerySlide, type GallerySource, type Interpretation, type InterpretationDatasetSource, type InterpretationSource, type VisualizeSelection } from '../api/interpretation';
import { computeActive, predictorMethodLabel, predictors, type FrozenPredictor } from '../api/predictors';
import type { Workspace } from '../api/types';
import { ErrorNotice, PageHeader, Panel } from '../components/ui';
import { StageBackButton, StageContinueButton, StageSteps } from '../components/StageWorkflow';
import SlideGalleryCard from '../components/SlideGalleryCard';
import InterpretationResources from '../components/InterpretationResources';
import InterpretationReview from '../components/InterpretationReview';
import { useInterpretationBatch } from '../components/useInterpretationBatch';
import RunStatusChip from '../components/RunStatusChip';
import { defaultRepresentation, representationCompatible, sourceCompatible, validInterpretationResources } from '../lib/interpretationGallery';
import { adoptSavedInterpretation, completedSlideResult, replaceWizardContext, persistWizardDraft, restoreWizardDraft, wizardResources, wizardRoute, wizardStage, type InterpretationStage, type InterpretationWizardDraft } from '../lib/interpretationWizard';
import { useHashParameters } from '../lib/hashRoute';
import { modelLabel, supportsAttention } from '../lib/modelCapabilities';
import './ModelChains.css';
import './ClinicalInsights.css';
import './InterpretationGallery.css';
import './InterpretationWizard.css';
import '../components/InterpretationReview.css';
import { stageEyebrow } from '../lib/roadmap';

const MAX_SLIDES = 128;
const PAGE_SIZE = 24;
const dtypeLabel = (value: unknown) => Array.isArray(value) ? `mixed (${value.join(', ')})` : typeof value === 'string' ? value : 'unknown dtype';

export default function LocalInterpretation({ workspace }: { workspace: Workspace }) {
  return <InterpretationWorkspace key={workspace.project.id} workspace={workspace} />;
}
function InterpretationWorkspace({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const parameters = useHashParameters();
  const stage = wizardStage(parameters);
  const [draft, setDraft] = useState<InterpretationWizardDraft>(() => restoreWizardDraft(project, parameters));
  const contextSignature = JSON.stringify(['predictor', 'search'].map((key) => parameters.get(key) ?? ''));
  const previousContext = useRef(contextSignature);
  useEffect(() => {
    if (previousContext.current === contextSignature) return;
    previousContext.current = contextSignature;
    if (draft.batch?.pending) return;
    if ((parameters.get('predictor') ?? '') !== draft.predictorId || (parameters.has('search') && parameters.get('search') !== draft.search)) setDraft(restoreWizardDraft(project, parameters));
  }, [contextSignature, parameters, project, draft.predictorId, draft.search, draft.batch?.pending]);
  useEffect(() => persistWizardDraft(project, draft), [project, draft]);
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project) });
  const sources = useQuery({ queryKey: ['interpretation-sources', project], queryFn: () => interpretations.sources(project) });
  const datasets = useQuery({ queryKey: ['interpretation-datasets', project], queryFn: () => interpretations.datasets(project), retry: false });
  const records = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project), enabled: stage === 'select' });
  const linkedId = parameters.get('interpretation') ?? '';
  const linkedRecord = useQuery({ queryKey: ['interpretation', project, linkedId], queryFn: () => interpretations.get(project, linkedId), enabled: Boolean(linkedId) });
  const batch = useInterpretationBatch(project, draft, setDraft);
  const predictor = registry.data?.items.find((item) => item.id === draft.predictorId && item.lifecycleState !== 'trashed');
  const contract = predictor?.manifest.inputs.features;
  const bundleId = draft.bundleId ?? contract?.bundle.id ?? '';
  const source = sources.data?.items.find((item) => item.id === bundleId);
  const preferredPack = predictor?.manifest.inputs.resolvedLoading?.packArtifactId ?? predictor?.manifest.inputs.loading.packArtifactId;
  const packId = draft.packChoice ?? defaultRepresentation(source, contract?.dtype, preferredPack);
  const pack = source?.packs.find((item) => item.id === packId);
  const sourceMatches = Boolean(source && sourceCompatible(source, contract?.encoderId, contract?.dimensions, contract?.dtype));
  const representationMatches = packId !== null && representationCompatible(packId ? pack?.outputDtype : source?.dtype, contract?.dtype);
  const resources = wizardResources(draft.resources);
  const geometryValid = (!draft.resources.width && !draft.resources.height) || [draft.resources.width, draft.resources.height].every((value) => value.trim() && Number.isFinite(Number(value)) && Number(value) > 0 && Number(value) <= 1000000);
  const resourcesValid = validInterpretationResources(resources) && geometryValid;
  // Slides come from a chosen frozen dataset; features from the bundle. A service without the
  // dataset listing falls back to the bundle's own dataset folder.
  const datasetItems = datasets.data?.items ?? [];
  const usableDataset = (item: InterpretationDatasetSource) => Boolean(item.slideFolder) && item.slideFolderFinding?.severity !== 'error';
  const automaticDataset = source?.datasetId && datasetItems.some((item) => item.datasetId === source.datasetId) ? source.datasetId
    : datasetItems.filter(usableDataset).length === 1 ? datasetItems.find(usableDataset)!.datasetId : '';
  const datasetId = draft.datasetId ?? automaticDataset;
  const slideSource: InterpretationDatasetSource | undefined = datasets.data
    ? datasetItems.find((item) => item.datasetId === datasetId)
    : source?.slideFolder ? { datasetId: source.datasetId ?? '', datasetName: source.datasetName ?? 'Frozen dataset', slideFolder: source.slideFolder, slideFolderSource: source.slideFolderSource ?? null, slideFolderFinding: null, slideCount: source.slideCount } : undefined;
  const slideFolder = slideSource && usableDataset(slideSource) ? slideSource.slideFolder! : '';
  const ready = Boolean(predictor && supportsAttention(predictor.manifest.recipe.model, predictor.manifest.recipe.inputMode) && sourceMatches && representationMatches && slideFolder && !registry.isError && !sources.isError);
  const gallerySource: GallerySource = { predictorId: draft.predictorId, featureBundleId: bundleId, packArtifactId: packId || null, slideFolder };
  const classOrder = predictor?.manifest.target?.classes ?? [];
  function navigate(next: InterpretationStage, studyId?: string, slideId?: string, replace = false) {
    const route = wizardRoute(parameters, next, draft, studyId, slideId);
    window.history[replace ? 'replaceState' : 'pushState'](window.history.state, '', route);
    window.dispatchEvent(new Event('hashchange'));
  }
  useEffect(() => {
    const record = linkedRecord.data;
    if (draft.batch?.pending || !linkedId || !record || draft.batch?.items.some((item) => item.interpretationId === linkedId)) return;
    const next = adoptSavedInterpretation(draft, record);
    setDraft(next);
    replaceWizardContext(project, next, parameters, stage, linkedId, parameters.get('slide') ?? undefined);
  }, [linkedId, linkedRecord.data, draft.batch]);
  function changeSource(patch: Partial<InterpretationWizardDraft>) {
    if (batch.locked) return;
    const next = { ...draft, ...patch, selected: [], search: '', offset: 0, batch: null };
    setDraft(next);
    replaceWizardContext(project, next, parameters, 'select');
  }
  function selectSlides(slides: GallerySlide[]) { if (batch.locked) return; setDraft((current) => ({ ...current, selected: slides, batch: null })); }
  function selection(paths: string[], original = draft.batch?.request): VisualizeSelection {
    if (original) return { ...original, slidePaths: paths, resources };
    return { ...gallerySource, slidePaths: paths, resources, patchWidthLevel0: draft.resources.width ? Number(draft.resources.width) : undefined, patchHeightLevel0: draft.resources.height ? Number(draft.resources.height) : undefined };
  }
  function startReview() {
    if (!ready || !draft.selected.length || !resourcesValid || batch.locked) return;
    if (!batch.ready) batch.start(selection(draft.selected.map((slide) => slide.slidePath), null), draft.selected);
    navigate('review');
  }
  const failedPaths = (draft.batch?.selected ?? []).filter((slide) => { const state = batch.states.get(slide.slidePath); return !state?.execution || (!computeActive(state.execution) && state.execution.status !== 'completed'); }).map((slide) => slide.slidePath);
  function retryFailed() {
    if (!draft.batch?.request || !resourcesValid || batch.locked || !failedPaths.length) return;
    batch.retrySlides(selection(failedPaths));
  }
  function openSaved(record: Interpretation) { if (batch.locked) return; navigate('review', record.id, record.manifest.slides[0]?.slideId); }
  function openReviewSlide(slide: GallerySlide) {
    const item = draft.batch?.items.find((entry) => entry.slidePath === slide.slidePath);
    navigate('review', item?.interpretationId ?? undefined, slide.slideId, true);
  }
  const readyCount = draft.batch ? draft.batch.selected.filter((slide) => completedSlideResult(slide, batch.states.get(slide.slidePath))).length : 0;
  const steps = [
    { id: 'model', title: 'Model & features', description: predictor ? predictor.manifest.name : 'Load trained weights', complete: ready },
    { id: 'slides', title: 'Slides', description: draft.selected.length ? `${draft.selected.length} selected` : 'Choose slides', complete: draft.selected.length > 0, disabled: !ready },
    { id: 'review', title: 'Attention review', description: draft.batch ? `${readyCount} / ${draft.batch.selected.length} ready` : 'Attention & predictions', complete: batch.ready, disabled: !draft.batch },
  ];
  const stepper = <StageSteps label="Interpretation steps" current={stage === 'review' ? 'review' : ready ? 'slides' : 'model'} steps={steps} disabled={batch.locked && stage !== 'review'} onChange={(id) => navigate(id === 'review' ? 'review' : 'select')} />;

  if (stage === 'review') return <div className="clinical-workspace model-chains clinical-insights interpretation-wizard">
    <PageHeader eyebrow={stageEyebrow('interpretation')} title="Attention review" description={predictor ? `${predictor.manifest.name} · ${modelLabel(predictor.manifest.recipe.model)} · ${predictorMethodLabel(predictor.manifest.method)}. Select a slide to see where the model attends, its highest-attention patches and its predicted label.` : 'Select a slide to see where the model attends, its highest-attention patches and its predicted label.'} actions={<StageBackButton onClick={() => navigate('select')}>Back to slide selection</StageBackButton>} />
    {stepper}
    <ErrorNotice error={linkedRecord.error ?? (draft.batch?.error ? new Error(draft.batch.error) : null)} />
    {draft.batch?.uncertain ? <p className="callout" role="alert">The attention request response was lost, so some jobs may already exist. <button className="btn btn-secondary" disabled={batch.busy} onClick={batch.retryExact}>Retry the same attention request</button></p> : null}
    {draft.batch ? <InterpretationReview project={project} selected={draft.batch.selected} items={draft.batch.items} states={batch.states} records={batch.records} classOrder={classOrder} activeSlideId={parameters.get('slide') ?? undefined} busy={batch.busy} canControlJobs={!draft.batch.request} onSelect={openReviewSlide}
      actions={<><RunStatusChip scope={batch.statusScope} variant="chip" label="Attention jobs" hideWhenNotStarted onSettled={batch.refresh} /><button className="btn btn-secondary btn-small" onClick={batch.refresh}>Refresh status</button>{draft.batch.request && failedPaths.length && !batch.busy ? <button className="btn btn-secondary btn-small" disabled={batch.locked || !resourcesValid} onClick={retryFailed}>Resume {failedPaths.length} missing or failed</button> : null}</>} />
      : <Panel title="Choose slides first"><p>No attention batch is open.</p><button className="btn btn-primary" onClick={() => navigate('select')}>Choose slides</button></Panel>}
    {draft.batch?.request && failedPaths.length && !batch.busy ? <InterpretationResources value={draft.resources} geometryReadOnly disabled={batch.locked} onChange={(value) => setDraft((current) => ({ ...current, resources: value }))} /> : null}
  </div>;

  return <div className="clinical-workspace model-chains clinical-insights interpretation-wizard">
    <PageHeader eyebrow={stageEyebrow('interpretation')} title="Model interpretation" description="Load trained model weights with a dataset and its features, choose slides, then review attention overlays, the highest-attention patches and each slide's predicted label. No evaluation, inference or clinical results are needed." />
    {stepper}
    <ErrorNotice error={registry.error ?? sources.error ?? records.error} />
    <Panel title="Load model weights and features" subtitle="Slides come from the chosen frozen dataset. Attention runs the model weights on the feature bundle's features or its frozen pack.">
      <fieldset className="interpretation-sources" disabled={batch.locked}><legend className="sr-only">Model weights and features</legend>
        <label className="label">Trained model (weights)<select className="field" value={draft.predictorId} onChange={(event) => changeSource({ predictorId: event.target.value, bundleId: null, packChoice: null })}><option value="">Choose a frozen ABMIL or nnMIL predictor</option>{draft.predictorId && !predictor ? <option value={draft.predictorId} disabled>Linked predictor unavailable</option> : null}{(registry.data?.items ?? []).filter((item) => item.lifecycleState === 'active' || item.id === draft.predictorId && item.lifecycleState === 'archived').map((item) => <option key={item.id} value={item.id} disabled={!supportsAttention(item.manifest.recipe.model, item.manifest.recipe.inputMode)}>{item.manifest.name} · {modelLabel(item.manifest.recipe.model)} · {predictorMethodLabel(item.manifest.method)}{!supportsAttention(item.manifest.recipe.model, item.manifest.recipe.inputMode) ? ' · attention unsupported' : ''}</option>)}</select></label>
        <label className="label">Dataset (slides)<select className="field" value={datasetId} disabled={datasets.isPending && !datasets.isError} onChange={(event) => changeSource({ datasetId: event.target.value || null })}><option value="">{datasets.isError ? 'Use the feature bundle’s dataset' : 'Choose a frozen dataset'}</option>{datasetItems.map((item) => <option key={item.datasetId} value={item.datasetId} disabled={!usableDataset(item)}>{item.datasetName}{item.slideCount != null ? ` · ${item.slideCount.toLocaleString()} slides` : ''}{!usableDataset(item) ? ' · slide folder unavailable' : ''}</option>)}</select></label>
        <label className="label">Feature bundle<select className="field" value={bundleId} disabled={!predictor || sources.isPending} onChange={(event) => changeSource({ bundleId: event.target.value, packChoice: null })}><option value="">Choose a frozen feature bundle</option>{bundleId && !source ? <option value={bundleId} disabled>Predictor’s feature bundle unavailable</option> : null}{(sources.data?.items ?? []).map((item) => <option key={item.id} value={item.id} disabled={!sourceCompatible(item, contract?.encoderId, contract?.dimensions, contract?.dtype)}>{item.name} · {item.slideCount.toLocaleString()} slides{!sourceCompatible(item, contract?.encoderId, contract?.dimensions, contract?.dtype) ? ' · incompatible or needs verification' : ''}</option>)}</select></label>
        <label className="label">Feature representation<select className="field" value={packId ?? '__unavailable__'} disabled={!source} onChange={(event) => changeSource({ packChoice: event.target.value })}>{packId === null ? <option value="__unavailable__" disabled>Choose a compatible feature representation</option> : null}<option value="" disabled={!representationCompatible(source?.dtype, contract?.dtype)}>Original slide features{source ? ` · ${dtypeLabel(source.dtype)}${!representationCompatible(source.dtype, contract?.dtype) ? ` · requires ${dtypeLabel(contract?.dtype)}` : ''}` : ''}</option>{source?.packs.map((item) => <option key={item.id} value={item.id} disabled={!representationCompatible(item.outputDtype, contract?.dtype)}>{item.name} · {item.outputDtype}{!representationCompatible(item.outputDtype, contract?.dtype) ? ` · requires ${dtypeLabel(contract?.dtype)}` : ''}</option>)}</select></label>
      </fieldset>
      {predictor ? <LoadedInputs predictor={predictor} slides={slideSource} slideFolder={slideFolder} source={source} pack={pack} sourceMatches={sourceMatches} representationMatches={representationMatches} /> : null}
      {predictor && !slideFolder ? <p className="callout">{slideSource?.slideFolderFinding?.message ?? (datasets.data ? 'Choose the frozen dataset whose slides you want to interpret.' : datasets.isError ? 'This running HistoPilot service predates dataset selection. Restart it to choose the dataset whose slides you want to interpret.' : source?.slideFolderFinding?.message ?? 'The frozen dataset does not identify a usable slide location. Correct its slide mapping in Datasets.')}</p> : null}
      {source && !representationMatches ? <p className="callout">This model requires {dtypeLabel(contract?.dtype)} features. Choose a compatible original source or frozen pack.</p> : null}
      <InterpretationResources value={draft.resources} disabled={batch.locked} onChange={(value) => setDraft((current) => ({ ...current, resources: value, batch: value.width !== current.resources.width || value.height !== current.resources.height ? null : current.batch }))} />
      {!resourcesValid ? <p className="callout">Use valid CPU/GPU resources, positive finite memory and either both positive footprint dimensions or neither.</p> : null}
    </Panel>
    {ready ? <GalleryWorkspace project={project} source={gallerySource} selected={draft.selected} search={draft.search} offset={draft.offset} locked={batch.locked} canContinue={resourcesValid} onSelection={selectSlides} onSearch={(search) => setDraft((current) => ({ ...current, search, offset: 0 }))} onPage={(offset) => setDraft((current) => ({ ...current, offset }))} onContinue={startReview} /> : <Panel title="Choose slides"><p className="muted">Load a compatible model and feature bundle to browse its dataset slides.</p></Panel>}
    {batch.locked ? <p className="callout">Your attention request is still being resolved. <button className="text-button" onClick={() => navigate('review')}>Return to attention review</button></p> : draft.batch ? <p className="callout">An attention batch of {draft.batch.selected.length} slides is open ({readyCount} ready). <button className="text-button" onClick={() => navigate('review')}>Return to attention review</button></p> : null}
    <details className="interpretation-history"><summary>Open a saved attention study</summary>{(records.data?.items ?? []).filter((item) => item.lifecycleState !== 'trashed').map((item) => <article className="report-card" key={item.id}><button className="text-button" disabled={batch.locked} onClick={() => openSaved(item)}>{item.manifest.name}</button><small>{item.manifest.slides.length} slides · {predictorMethodLabel(item.manifest.method)}</small></article>)}</details>
  </div>;
}

/** What the attention jobs will load: exact model weights, the dataset's slides and matching features. */
function LoadedInputs({ predictor, slides, slideFolder, source, pack, sourceMatches, representationMatches }: { predictor: FrozenPredictor; slides?: InterpretationDatasetSource; slideFolder: string; source?: InterpretationSource; pack?: InterpretationSource['packs'][number]; sourceMatches: boolean; representationMatches: boolean }) {
  const manifest = predictor.manifest;
  const checkpoints = manifest.checkpoints?.length ?? 0;
  const classes = manifest.target?.classes ?? [];
  const featuresReady = Boolean(source && sourceMatches && representationMatches);
  const row = (ok: boolean, title: string, detail: string) => <li className={ok ? 'is-ready' : 'is-missing'}><span aria-hidden="true">{ok ? '✓' : '!'}</span><div><strong>{title}<span className="sr-only">{ok ? ' ready' : ' not ready'}</span></strong><small>{detail}</small></div></li>;
  return <ul className="interpretation-loaded" aria-label="Loaded inputs">
    {row(true, 'Model weights', `${modelLabel(manifest.recipe.model)} · ${manifest.method === 'refit' ? 'refit model, 1 checkpoint' : `fold ensemble, ${checkpoints} checkpoints`}${classes.length ? ` · predicts ${classes.join(' / ')}` : ''}`)}
    {row(Boolean(slideFolder), 'Dataset slides', slides ? `${slides.datasetName}${slides.slideCount != null ? ` · ${slides.slideCount.toLocaleString()} slides` : ''}${slideFolder ? ` · ${slideFolder}` : ''}` : 'Choose the dataset whose slides you want to interpret')}
    {row(featuresReady, 'Features', `${source ? `${source.encoderId ?? 'unknown encoder'} · ${source.dimensions ?? '?'}-d · ${pack ? `pack ${pack.name} (${pack.outputDtype})` : `original feature files (${dtypeLabel(source.dtype)})`} · ` : ''}Predictor requires ${dtypeLabel(manifest.inputs.features?.dtype)}`)}
  </ul>;
}

export function GalleryWorkspace({ project, source, selected, search, offset, locked, canContinue, onSelection, onSearch, onPage, onContinue }: { project: string; source: GallerySource; selected: GallerySlide[]; search: string; offset: number; locked: boolean; canContinue: boolean; onSelection: (slides: GallerySlide[]) => void; onSearch: (search: string) => void; onPage: (offset: number) => void; onContinue: () => void }) {
  const [settledSearch, setSettledSearch] = useState(search.trim());
  useEffect(() => { const timer = setTimeout(() => setSettledSearch(search.trim()), 250); return () => clearTimeout(timer); }, [search]);
  const gallery = useQuery({ queryKey: ['interpretation-gallery', project, source, settledSearch, offset], queryFn: ({ signal }) => interpretations.gallery(project, source, settledSearch, offset, signal), staleTime: 15000 });
  const current = search.trim() === settledSearch && !gallery.isFetching && !gallery.isError;
  const paths = new Set(selected.map((slide) => slide.slidePath));
  const addable = current ? (gallery.data?.items ?? []).filter((slide) => slide.available && !paths.has(slide.slidePath)) : [];
  function toggle(slide: GallerySlide, checked: boolean) { if (!checked) onSelection(selected.filter((item) => item.slidePath !== slide.slidePath)); else if (slide.available && !paths.has(slide.slidePath) && selected.length < MAX_SLIDES) onSelection([...selected, slide]); }
  return <Panel title="Choose slides" subtitle="Click thumbnails or checkboxes to select slides. Attention starts when you continue; finished slides open for review right away.">
    <div className="slide-gallery-tools"><label className="label">Search slides<input type="search" className="field" maxLength={200} value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Search slide name or relative path" /></label><StageContinueButton disabled={!selected.length || !canContinue || locked} onClick={onContinue}>Compute &amp; review {selected.length} selected</StageContinueButton><button className="btn btn-secondary" disabled={!addable.length || selected.length >= MAX_SLIDES || locked} onClick={() => onSelection([...selected, ...addable.slice(0, MAX_SLIDES - selected.length)])}>Select all on this page</button><button className="btn btn-secondary" disabled={!selected.length || locked} onClick={() => onSelection([])}>Clear selection</button></div>
    <div className="slide-gallery-count">{gallery.data ? `${gallery.data.total.toLocaleString()} matching slides` : 'Reading dataset slides'} · {selected.length} selected across all searches and pages{selected.length >= MAX_SLIDES ? ' · batch limit reached' : ''}.</div>
    {selected.length ? <details><summary>Selected slides ({selected.length})</summary><ul>{selected.map((slide) => <li key={slide.slidePath}>{slide.name} <button className="text-button" disabled={locked} onClick={() => toggle(slide, false)}>Remove from selection</button></li>)}</ul></details> : null}
    <ErrorNotice error={gallery.error} />
    {gallery.data?.warnings.length ? <ul className="callout">{gallery.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
    {!current && !gallery.isError ? <p role="status">Searching and matching slide features…</p> : null}
    {current && !gallery.data?.items.length ? <p className="callout">No slides match{settledSearch ? ` “${settledSearch}”` : ' this dataset'}. Your selection is preserved.</p> : null}
    <div className="slide-gallery-grid">{current ? gallery.data?.items.map((slide) => <SlideGalleryCard key={slide.slidePath} project={project} slide={slide} checked={paths.has(slide.slidePath)} disabled={locked} selectionDisabled={!paths.has(slide.slidePath) && selected.length >= MAX_SLIDES} onToggle={(checked) => toggle(slide, checked)} />) : null}</div>
    {gallery.data && (offset > 0 || gallery.data.hasMore) ? <div className="slide-gallery-pagination"><button className="btn btn-secondary" disabled={!current || offset === 0} onClick={() => onPage(Math.max(0, offset - PAGE_SIZE))}>Previous slides</button><span>Page {Math.floor(offset / PAGE_SIZE) + 1} / {Math.max(1, Math.ceil(gallery.data.total / PAGE_SIZE))}</span><button className="btn btn-secondary" disabled={!current || !gallery.data.hasMore} onClick={() => onPage(offset + PAGE_SIZE)}>Next slides</button></div> : null}
  </Panel>;
}
