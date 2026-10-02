import { Fragment, useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { interpretations, poolingFunctions, poolingLabel, type GallerySlide, type GallerySource, type Interpretation, type InterpretationDatasetSource, type InterpretationSource, type PoolingFunction, type VisualizeSelection } from '../api/interpretation';
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
import { adoptSavedInterpretation, adoptSavedStudy, completedSlideResult, replaceWizardContext, persistWizardDraft, restoreWizardDraft, wizardResources, wizardRoute, wizardStage, MAX_POOLED_MODELS, type InterpretationStage, type InterpretationWizardDraft } from '../lib/interpretationWizard';
import { useHashParameters } from '../lib/hashRoute';
import { modelLabel, supportsAttention } from '../lib/modelCapabilities';
import { experimentRoleLabel, objectiveRankLabel, objectives as objectiveRecords, type ExperimentFiling, type Objective } from '../api/objectives';
import { sectionByObjective } from '../lib/experimentLibrary';
import { interpretationStudies } from '../lib/interpretationStudies';
import { inObjectiveScope, useObjectiveScope } from '../lib/objectiveScope';
import { predictorOptionGroups } from '../lib/predictorGroups';
import { ObjectiveScopeNotice } from '../components/ObjectiveScope';
import { Badge } from '../components/ui';
import './ModelChains.css';
import './ClinicalInsights.css';
import './InterpretationGallery.css';
import './InterpretationWizard.css';
import '../components/InterpretationReview.css';
import { stageEyebrow } from '../lib/roadmap';

const MAX_SLIDES = 128;
const PAGE_SIZE = 24;
const dtypeLabel = (value: unknown) => Array.isArray(value) ? `mixed (${value.join(', ')})` : typeof value === 'string' ? value : 'unknown dtype';
/** JSON with object keys sorted, so frozen targets compare by content. */
const canonical = (value: unknown) => JSON.stringify(value, (_key, inner: unknown) => inner && typeof inner === 'object' && !Array.isArray(inner) ? Object.fromEntries(Object.entries(inner).sort(([a], [b]) => a.localeCompare(b))) : inner);
/** A model can pool with the primary when it predicts the same classes from the same features. */
function poolCompatible(primary: FrozenPredictor, item: FrozenPredictor) {
  const a = primary.manifest.inputs?.features, b = item.manifest.inputs?.features;
  return supportsAttention(item.manifest.recipe.model, item.manifest.recipe.inputMode) && item.manifest.recipe.inputMode !== 'multimodal'
    && canonical(item.manifest.target) === canonical(primary.manifest.target)
    && Boolean(a?.encoderId) && a?.encoderId === b?.encoderId && a?.dimensions === b?.dimensions && canonical(a?.dtype) === canonical(b?.dtype);
}

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
  // Predictors and studies follow the project's objectives, and its objective focus.
  const objectiveList = useQuery({ queryKey: ['objectives', project], queryFn: () => objectiveRecords.list(project), staleTime: 30000 });
  const objectiveItems = objectiveList.data?.items ?? [];
  const filings = objectiveList.data?.filings ?? {};
  const [objectiveScope, setObjectiveScope] = useObjectiveScope(objectiveItems);
  const sources = useQuery({ queryKey: ['interpretation-sources', project], queryFn: () => interpretations.sources(project) });
  const datasets = useQuery({ queryKey: ['interpretation-datasets', project], queryFn: () => interpretations.datasets(project), retry: false });
  const records = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project), enabled: stage === 'select' });
  const linkedId = parameters.get('interpretation') ?? '';
  const linkedRecord = useQuery({ queryKey: ['interpretation', project, linkedId], queryFn: () => interpretations.get(project, linkedId), enabled: Boolean(linkedId) });
  const batch = useInterpretationBatch(project, draft, setDraft);
  const predictor = registry.data?.items.find((item) => item.id === draft.predictorId && item.lifecycleState !== 'trashed');
  // Models whose attention can pool with the primary model's, in the same objective focus.
  const registryItems = registry.data?.items ?? [];
  const poolable = Boolean(predictor && predictor.manifest.recipe.inputMode !== 'multimodal');
  const poolCandidates = predictor && poolable ? registryItems.filter((item) => item.id !== predictor.id && poolCompatible(predictor, item)
    && (item.lifecycleState === 'active' || item.lifecycleState === 'archived' && draft.pooledPredictorIds.includes(item.id))
    && (draft.pooledPredictorIds.includes(item.id) || inObjectiveScope(objectiveScope, filings[item.manifest.experimentId]?.objectiveId))) : [];
  const pooledModels = draft.pooledPredictorIds.map((id) => poolCandidates.find((item) => item.id === id)).filter((item): item is FrozenPredictor => Boolean(item));
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
  function changePool(patch: Partial<Pick<InterpretationWizardDraft, 'pooledPredictorIds' | 'pooling'>>) {
    if (batch.locked) return;
    setDraft((current) => ({ ...current, ...patch, batch: null }));
  }
  function selection(paths: string[], original = draft.batch?.request): VisualizeSelection {
    if (original) return { ...original, slidePaths: paths, resources };
    const pooled = pooledModels.length ? { predictorIds: [draft.predictorId, ...pooledModels.map((item) => item.id)], pooling: draft.pooling } : {};
    return { ...gallerySource, ...pooled, slidePaths: paths, resources, patchWidthLevel0: draft.resources.width ? Number(draft.resources.width) : undefined, patchHeightLevel0: draft.resources.height ? Number(draft.resources.height) : undefined };
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
  /** Reopen every slide of a saved study, then show its first. */
  function openStudy(records: Interpretation[]) {
    if (batch.locked || !records.length) return;
    const next = adoptSavedStudy(draft, records);
    if (next === draft) { openSaved(records[0]); return; }
    setDraft(next);
    persistWizardDraft(project, next);
    window.history.pushState(window.history.state, '', wizardRoute(parameters, 'review', next, records[0].id, records[0].manifest.slides[0]?.slideId));
    window.dispatchEvent(new Event('hashchange'));
  }
  function openReviewSlide(slide: GallerySlide) {
    const item = draft.batch?.items.find((entry) => entry.slidePath === slide.slidePath);
    navigate('review', item?.interpretationId ?? undefined, slide.slideId, true);
  }
  const readyCount = draft.batch ? draft.batch.selected.filter((slide) => completedSlideResult(slide, batch.states.get(slide.slidePath))).length : 0;
  const steps = [
    { id: 'model', title: 'Model & features', description: predictor ? `${predictor.manifest.name}${pooledModels.length ? ` + ${pooledModels.length} pooled (${poolingLabel(draft.pooling).toLocaleLowerCase()})` : ''}` : 'Load trained weights', complete: ready },
    { id: 'slides', title: 'Slides', description: draft.selected.length ? `${draft.selected.length} selected` : 'Choose slides', complete: draft.selected.length > 0, disabled: !ready },
    { id: 'review', title: 'Attention review', description: draft.batch ? `${readyCount} / ${draft.batch.selected.length} ready` : 'Attention & predictions', complete: batch.ready, disabled: !draft.batch },
  ];
  const stepper = <StageSteps label="Interpretation steps" current={stage === 'review' ? 'review' : ready ? 'slides' : 'model'} steps={steps} disabled={batch.locked && stage !== 'review'} onChange={(id) => navigate(id === 'review' ? 'review' : 'select')} />;

  if (stage === 'review') return <div className="clinical-workspace model-chains clinical-insights interpretation-wizard">
    <PageHeader eyebrow={stageEyebrow('interpretation')} title="Attention review" description={predictor && draft.pooledPredictorIds.length ? `Pooled attention (${poolingLabel(draft.pooling).toLocaleLowerCase()}) of ${draft.pooledPredictorIds.length + 1} models, led by ${predictor.manifest.name}. Each slide's map pools the models' maps patch by patch; choose a single model under Attention map to compare.` : predictor ? `${predictor.manifest.name} · ${modelLabel(predictor.manifest.recipe.model)} · ${predictorMethodLabel(predictor.manifest.method)}. Select a slide to see where the model attends, its highest-attention patches and its predicted label.` : 'Select a slide to see where the model attends, its highest-attention patches and its predicted label.'} actions={<StageBackButton onClick={() => navigate('select')}>Back to slide selection</StageBackButton>} />
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
        <label className="label">Trained model (weights)<select className="field" value={draft.predictorId} onChange={(event) => changeSource({ predictorId: event.target.value, bundleId: null, packChoice: null, pooledPredictorIds: [] })}><option value="">Choose a frozen ABMIL or nnMIL predictor</option>{draft.predictorId && !predictor ? <option value={draft.predictorId} disabled>Linked predictor unavailable</option> : null}{predictorOptionGroups((registry.data?.items ?? []).filter((item) => (item.lifecycleState === 'active' || item.id === draft.predictorId && item.lifecycleState === 'archived') && (item.id === draft.predictorId || inObjectiveScope(objectiveScope, filings[item.manifest.experimentId]?.objectiveId))), filings, objectiveItems).map((group) => <optgroup key={group.key} label={group.label}>{group.items.map((item) => <option key={item.id} value={item.id} disabled={!supportsAttention(item.manifest.recipe.model, item.manifest.recipe.inputMode)}>{item.manifest.name} · {modelLabel(item.manifest.recipe.model)} · {predictorMethodLabel(item.manifest.method)}{!supportsAttention(item.manifest.recipe.model, item.manifest.recipe.inputMode) ? ' · attention unsupported' : ''}</option>)}</optgroup>)}</select>{objectiveScope ? <small>Showing the models of the focused objective. <button type="button" className="text-button" onClick={() => setObjectiveScope('')}>Show all objectives</button></small> : null}</label>
        <label className="label">Dataset (slides)<select className="field" value={datasetId} disabled={datasets.isPending && !datasets.isError} onChange={(event) => changeSource({ datasetId: event.target.value || null })}><option value="">{datasets.isError ? 'Use the feature bundle’s dataset' : 'Choose a frozen dataset'}</option>{datasetItems.map((item) => <option key={item.datasetId} value={item.datasetId} disabled={!usableDataset(item)}>{item.datasetName}{item.slideCount != null ? ` · ${item.slideCount.toLocaleString()} slides` : ''}{!usableDataset(item) ? ' · slide folder unavailable' : ''}</option>)}</select></label>
        <label className="label">Feature bundle<select className="field" value={bundleId} disabled={!predictor || sources.isPending} onChange={(event) => changeSource({ bundleId: event.target.value, packChoice: null })}><option value="">Choose a frozen feature bundle</option>{bundleId && !source ? <option value={bundleId} disabled>Predictor’s feature bundle unavailable</option> : null}{(sources.data?.items ?? []).map((item) => <option key={item.id} value={item.id} disabled={!sourceCompatible(item, contract?.encoderId, contract?.dimensions, contract?.dtype)}>{item.name} · {item.slideCount.toLocaleString()} slides{!sourceCompatible(item, contract?.encoderId, contract?.dimensions, contract?.dtype) ? ' · incompatible or needs verification' : ''}</option>)}</select></label>
        <label className="label">Feature representation<select className="field" value={packId ?? '__unavailable__'} disabled={!source} onChange={(event) => changeSource({ packChoice: event.target.value })}>{packId === null ? <option value="__unavailable__" disabled>Choose a compatible feature representation</option> : null}<option value="" disabled={!representationCompatible(source?.dtype, contract?.dtype)}>Original slide features{source ? ` · ${dtypeLabel(source.dtype)}${!representationCompatible(source.dtype, contract?.dtype) ? ` · requires ${dtypeLabel(contract?.dtype)}` : ''}` : ''}</option>{source?.packs.map((item) => <option key={item.id} value={item.id} disabled={!representationCompatible(item.outputDtype, contract?.dtype)}>{item.name} · {item.outputDtype}{!representationCompatible(item.outputDtype, contract?.dtype) ? ` · requires ${dtypeLabel(contract?.dtype)}` : ''}</option>)}</select></label>
        {predictor && poolable ? <PoolChooser candidates={poolCandidates} selectedIds={pooledModels.map((item) => item.id)} pooling={draft.pooling} classes={predictor.manifest.target?.classes ?? []} filings={filings} objectives={objectiveItems} onChange={changePool} /> : null}
      </fieldset>
      {predictor ? <LoadedInputs predictor={predictor} pooled={pooledModels} pooling={draft.pooling} slides={slideSource} slideFolder={slideFolder} source={source} pack={pack} sourceMatches={sourceMatches} representationMatches={representationMatches} /> : null}
      {predictor && !slideFolder ? <p className="callout">{slideSource?.slideFolderFinding?.message ?? (datasets.data ? 'Choose the frozen dataset whose slides you want to interpret.' : datasets.isError ? 'This running HistoPilot service predates dataset selection. Restart it to choose the dataset whose slides you want to interpret.' : source?.slideFolderFinding?.message ?? 'The frozen dataset does not identify a usable slide location. Correct its slide mapping in Datasets.')}</p> : null}
      {source && !representationMatches ? <p className="callout">This model requires {dtypeLabel(contract?.dtype)} features. Choose a compatible original source or frozen pack.</p> : null}
      <InterpretationResources value={draft.resources} disabled={batch.locked} onChange={(value) => setDraft((current) => ({ ...current, resources: value, batch: value.width !== current.resources.width || value.height !== current.resources.height ? null : current.batch }))} />
      {!resourcesValid ? <p className="callout">Use valid CPU/GPU resources, positive finite memory and either both positive footprint dimensions or neither.</p> : null}
    </Panel>
    {ready ? <GalleryWorkspace project={project} source={gallerySource} selected={draft.selected} search={draft.search} offset={draft.offset} locked={batch.locked} canContinue={resourcesValid} onSelection={selectSlides} onSearch={(search) => setDraft((current) => ({ ...current, search, offset: 0 }))} onPage={(offset) => setDraft((current) => ({ ...current, offset }))} onContinue={startReview} /> : <Panel title="Choose slides"><p className="muted">Load a compatible model and feature bundle to browse its dataset slides.</p></Panel>}
    {batch.locked ? <p className="callout">Your attention request is still being resolved. <button className="text-button" onClick={() => navigate('review')}>Return to attention review</button></p> : draft.batch ? <p className="callout">An attention batch of {draft.batch.selected.length} slides is open ({readyCount} ready). <button className="text-button" onClick={() => navigate('review')}>Return to attention review</button></p> : null}
    <SavedStudies records={(records.data?.items ?? []).filter((item) => item.lifecycleState !== 'trashed')} loading={records.isPending} predictors={registry.data?.items ?? []} objectives={objectiveItems} filings={filings} objectiveScope={objectiveScope} onClearScope={() => setObjectiveScope('')} disabled={batch.locked} onOpen={openStudy} />
  </div>;
}

/**
 * More models whose attention pools with the primary model's. Each model's map is computed as
 * usual (an ensemble's as its member mean); the study's map combines them patch by patch.
 */
function PoolChooser({ candidates, selectedIds, pooling, classes, filings, objectives, onChange }: {
  candidates: FrozenPredictor[]; selectedIds: string[]; pooling: PoolingFunction; classes: readonly string[];
  filings: Record<string, ExperimentFiling>; objectives: readonly Objective[];
  onChange: (patch: Partial<Pick<InterpretationWizardDraft, 'pooledPredictorIds' | 'pooling'>>) => void;
}) {
  const chosen = new Set(selectedIds);
  const full = selectedIds.length >= MAX_POOLED_MODELS - 1;
  return <details className="interpretation-pool" open={selectedIds.length > 0 || undefined}>
    <summary>Pool attention across more models <span className="muted">{selectedIds.length ? `${selectedIds.length + 1} models · ${poolingLabel(pooling)}` : 'optional'}</span></summary>
    <p className="muted">Each selected model's attention map is computed as usual, then the maps are pooled patch by patch into one map. The pooled prediction is the mean of the models' probabilities. Listed: models that predict {classes.length ? classes.join(' / ') : 'the same classes'} from the same features.</p>
    <label className="label">Pooling function<select className="field" value={pooling} onChange={(event) => onChange({ pooling: event.target.value as PoolingFunction })}>{poolingFunctions.map((item) => <option key={item.value} value={item.value}>{item.label} — {item.help}</option>)}</select></label>
    {candidates.length ? <div className="interpretation-pool-list" role="group" aria-label="Models to pool with the primary model">
      {predictorOptionGroups(candidates, filings, objectives).map((group) => <fieldset key={group.key}><legend>{group.label}</legend>
        {group.items.map((item) => <label key={item.id} className="interpretation-pool-model"><input type="checkbox" checked={chosen.has(item.id)} disabled={!chosen.has(item.id) && full}
          onChange={(event) => onChange({ pooledPredictorIds: event.target.checked ? [...selectedIds, item.id] : selectedIds.filter((id) => id !== item.id) })} />
          <span>{item.manifest.name} · {modelLabel(item.manifest.recipe.model)} · {predictorMethodLabel(item.manifest.method)}</span></label>)}
      </fieldset>)}
    </div> : <p className="muted">No other model predicts the same classes from the same features.</p>}
    {selectedIds.length ? <p className="interpretation-pool-actions">{full ? <span className="muted">At most {MAX_POOLED_MODELS} models pool together. </span> : null}<button type="button" className="text-button" onClick={() => onChange({ pooledPredictorIds: [] })}>Clear pooled models</button></p> : null}
  </details>;
}

/** What the attention jobs will load: exact model weights, the dataset's slides and matching features. */
function LoadedInputs({ predictor, pooled, pooling, slides, slideFolder, source, pack, sourceMatches, representationMatches }: { predictor: FrozenPredictor; pooled: FrozenPredictor[]; pooling: PoolingFunction; slides?: InterpretationDatasetSource; slideFolder: string; source?: InterpretationSource; pack?: InterpretationSource['packs'][number]; sourceMatches: boolean; representationMatches: boolean }) {
  const manifest = predictor.manifest;
  const checkpoints = manifest.checkpoints?.length ?? 0;
  const classes = manifest.target?.classes ?? [];
  const featuresReady = Boolean(source && sourceMatches && representationMatches);
  const row = (ok: boolean, title: string, detail: string) => <li className={ok ? 'is-ready' : 'is-missing'}><span aria-hidden="true">{ok ? '✓' : '!'}</span><div><strong>{title}<span className="sr-only">{ok ? ' ready' : ' not ready'}</span></strong><small>{detail}</small></div></li>;
  return <ul className="interpretation-loaded" aria-label="Loaded inputs">
    {row(true, 'Model weights', `${modelLabel(manifest.recipe.model)} · ${manifest.method === 'refit' ? 'refit model, 1 checkpoint' : `${manifest.method === 'seed_ensemble' ? 'seed ensemble' : 'fold ensemble'}, ${checkpoints} checkpoints`}${classes.length ? ` · predicts ${classes.join(' / ')}` : ''}`)}
    {row(Boolean(slideFolder), 'Dataset slides', slides ? `${slides.datasetName}${slides.slideCount != null ? ` · ${slides.slideCount.toLocaleString()} slides` : ''}${slideFolder ? ` · ${slideFolder}` : ''}` : 'Choose the dataset whose slides you want to interpret')}
    {row(featuresReady, 'Features', `${source ? `${source.encoderId ?? 'unknown encoder'} · ${source.dimensions ?? '?'}-d · ${pack ? `pack ${pack.name} (${pack.outputDtype})` : `original feature files (${dtypeLabel(source.dtype)})`} · ` : ''}Predictor requires ${dtypeLabel(manifest.inputs.features?.dtype)}`)}
    {pooled.length ? <li className="is-ready is-pooled"><span aria-hidden="true">✓</span><div><strong>Pooled attention<span className="sr-only"> ready</span></strong><small>{`${poolingLabel(pooling)} of ${pooled.length + 1} models (${[predictor, ...pooled].reduce((total, item) => total + (item.manifest.checkpoints?.length ?? 0), 0)} checkpoints): ${[predictor, ...pooled].map((item) => `${item.manifest.name} · ${modelLabel(item.manifest.recipe.model)}`).join('; ')}`}</small></div></li> : null}
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

/**
 * Saved attention studies, by the objective of the experiment whose model they used. A study
 * is the slides interpreted together, though each slide is saved as its own record.
 */
function SavedStudies({ records, loading, predictors: models, objectives, filings, objectiveScope, onClearScope, disabled, onOpen }: {
  records: Interpretation[]; loading: boolean; predictors: FrozenPredictor[]; objectives: readonly Objective[];
  filings: Record<string, ExperimentFiling>; objectiveScope: string; onClearScope: () => void; disabled: boolean; onOpen: (records: Interpretation[]) => void;
}) {
  const studies = interpretationStudies(records);
  const visible = studies.filter((study) => inObjectiveScope(objectiveScope, filings[study.experimentId]?.objectiveId));
  const model = new Map(models.map((item) => [item.id, item]));
  const sections = objectives.length ? sectionByObjective(visible, (study) => filings[study.experimentId]?.objectiveId ?? null, objectives) : [{ key: 'all', objective: null, items: visible }];
  return <Panel title="Saved attention studies" subtitle="Open a study to review its slides again. Each study is listed with the model it used and the experiment that model came from.">
    <ObjectiveScopeNotice objectives={objectives} scope={objectiveScope} hidden={studies.length - visible.length} noun={studies.length - visible.length === 1 ? 'study' : 'studies'} onClear={onClearScope} />
    {loading ? <p role="status">Loading saved studies…</p> : !visible.length ? <p className="muted">{studies.length ? 'No saved studies in this focus.' : 'No attention studies yet. Choose a model and slides above to start one.'}</p>
      : <div className="table-wrap"><table className="chain-table interpretation-studies"><thead><tr><th scope="col">Study</th><th scope="col">Model</th><th scope="col">Experiment</th><th scope="col">Slides</th><th scope="col">Actions</th></tr></thead><tbody>
        {sections.map((section) => <Fragment key={section.key}>
          {objectives.length ? <tr className="run-group run-objective-group"><th colSpan={5}>{section.objective?.name ?? 'No objective'}{section.objective ? <Badge tone={section.objective.rank === 'primary' ? 'purple' : 'neutral'}>{objectiveRankLabel[section.objective.rank]}</Badge> : null}<small>{section.items.length} {section.items.length === 1 ? 'study' : 'studies'}</small></th></tr> : null}
          {section.items.map((study) => { const source = model.get(study.predictorId)?.manifest; const filing = filings[study.experimentId]; const first = study.records[0].manifest; const pooledStudy = first.method === 'pooled'; return <tr key={study.key}>
            <th scope="row">{new Date(study.createdAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}<small>{study.records.length === 1 ? study.records[0].manifest.name : `${study.records[0].manifest.name} and ${study.records.length - 1} more`}</small></th>
            <td>{pooledStudy ? `Pooled · ${poolingLabel(first.pooling)} of ${first.memberCount} models` : source ? `${source.name} · ${modelLabel(source.recipe.model)}` : 'Predictor unavailable'}<small>{pooledStudy ? (first.predictors ?? []).map((item) => item.name ?? item.id).join(' · ') : source ? predictorMethodLabel(source.method) : null}</small></td>
            <td>{source?.experiment?.name ?? study.experimentId}{filing?.role ? <small><span className={`experiment-role experiment-role-${filing.role}`}>{experimentRoleLabel[filing.role]}</span>{filing.seriesMember ? ` · ${filing.series} › ${filing.seriesMember}` : ''}</small> : null}</td>
            <td>{study.slides.toLocaleString()}</td>
            <td><button type="button" className="btn btn-secondary btn-small" disabled={disabled} onClick={() => onOpen(study.records)}>Open study</button></td>
          </tr>; })}
        </Fragment>)}
      </tbody></table></div>}
  </Panel>;
}
