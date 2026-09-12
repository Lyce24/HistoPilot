import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { interpretations, type GallerySlide, type GallerySource, type Interpretation, type VisualizeSelection } from '../api/interpretation';
import { clinicalAnalyses } from '../api/clinicalUtility';
import { computeActive, computeStatusLabel, modelEvaluations, predictorMethodLabel, predictors } from '../api/predictors';
import type { Workspace } from '../api/types';
import { ErrorNotice, PageHeader, Panel } from '../components/ui';
import EvidenceChain from '../components/EvidenceChain';
import SlideGalleryCard from '../components/SlideGalleryCard';
import AttentionPreviewCard from '../components/AttentionPreviewCard';
import AttentionSlideViewer from '../components/AttentionSlideViewer';
import ComputeJobControls from '../components/ComputeJobControls';
import InterpretationResources from '../components/InterpretationResources';
import { useInterpretationBatch } from '../components/useInterpretationBatch';
import { defaultRepresentation, representationCompatible, sourceCompatible, validInterpretationResources } from '../lib/interpretationGallery';
import { adoptSavedInterpretation, replaceWizardContext, persistWizardDraft, restoreWizardDraft, wizardResources, wizardRoute, wizardStage, type InterpretationStage, type InterpretationWizardDraft } from '../lib/interpretationWizard';
import { useHashParameters } from '../lib/hashRoute';
import './ModelChains.css';
import './ClinicalInsights.css';
import './InterpretationGallery.css';
import './InterpretationWizard.css';

export default function LocalInterpretation({ workspace }: { workspace: Workspace }) {
  return <InterpretationWorkspace key={workspace.project.id} workspace={workspace} />;
}
function InterpretationWorkspace({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const parameters = useHashParameters();
  const stage = wizardStage(parameters);
  const [draft, setDraft] = useState<InterpretationWizardDraft>(() => restoreWizardDraft(project, parameters));
  const contextSignature = ['predictor', 'evaluation', 'clinical'].map((key) => parameters.get(key) ?? '').join('|');
  const previousContext = useRef(contextSignature);
  useEffect(() => {
    if (previousContext.current === contextSignature) return;
    previousContext.current = contextSignature;
    if (draft.batch?.pending) return;
    if ((parameters.get('predictor') ?? '') !== draft.predictorId || (parameters.get('evaluation') ?? '') !== draft.evaluationId || (parameters.get('clinical') ?? '') !== draft.clinicalId) setDraft(restoreWizardDraft(project, parameters));
  }, [contextSignature, parameters, project, draft.predictorId, draft.evaluationId, draft.clinicalId, draft.batch?.pending]);
  useEffect(() => persistWizardDraft(project, draft), [project, draft]);
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project) });
  const sources = useQuery({ queryKey: ['interpretation-sources', project], queryFn: () => interpretations.sources(project) });
  const evaluations = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project) });
  const reports = useQuery({ queryKey: ['clinical-analyses', project], queryFn: () => clinicalAnalyses.list(project) });
  const records = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project), enabled: stage === 'select' });
  const linkedId = parameters.get('interpretation') ?? '';
  const linkedRecord = useQuery({ queryKey: ['interpretation', project, linkedId], queryFn: () => interpretations.get(project, linkedId), enabled: Boolean(linkedId) });
  const linkedClinical = useQuery({ queryKey: ['clinical-analysis', project, draft.clinicalId], queryFn: () => clinicalAnalyses.get(project, draft.clinicalId), enabled: Boolean(draft.clinicalId) && !(reports.data?.items ?? []).some((item) => item.id === draft.clinicalId) });
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
  const reportItems = [...(reports.data?.items ?? []), ...(linkedClinical.data && !(reports.data?.items ?? []).some((item) => item.id === linkedClinical.data.id) ? [linkedClinical.data] : [])];
  const matchingEvaluations = (evaluations.data?.items ?? []).filter((item) => item.manifest.predictorId === draft.predictorId && item.lifecycleState !== 'trashed' && item.execution?.status === 'completed');
  const matchingReports = reportItems.filter((item) => item.manifest.predictorId === draft.predictorId && (!draft.evaluationId || item.manifest.evaluationId === draft.evaluationId) && item.lifecycleState !== 'trashed');
  const contextValid = (!draft.evaluationId || matchingEvaluations.some((item) => item.id === draft.evaluationId)) && (!draft.clinicalId || matchingReports.some((item) => item.id === draft.clinicalId));
  const ready = Boolean(predictor && predictor.manifest.recipe.model.toLowerCase() === 'abmil' && sourceMatches && representationMatches && source?.slideFolder && contextValid && !registry.isError && !sources.isError);
  const gallerySource: GallerySource = { predictorId: draft.predictorId, featureBundleId: bundleId, packArtifactId: packId || null, slideFolder: source?.slideFolder ?? '' };
  const dtypeLabel = (value: unknown) => Array.isArray(value) ? `mixed (${value.join(', ')})` : typeof value === 'string' ? value : 'unknown dtype';
  function navigate(next: InterpretationStage, studyId?: string, slideId?: string, replace = false) {
    const route = wizardRoute(parameters, next, draft, studyId, slideId);
    window.history[replace ? 'replaceState' : 'pushState'](window.history.state, '', route);
    window.dispatchEvent(new Event('hashchange'));
  }
  useEffect(() => {
    if (stage === 'compute' && batch.ready && !batch.locked) navigate('results', undefined, undefined, true);
  });
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
    return { ...gallerySource, slidePaths: paths, resources, evaluationId: draft.evaluationId || null, clinicalAnalysisId: draft.clinicalId || null, patchWidthLevel0: draft.resources.width ? Number(draft.resources.width) : undefined, patchHeightLevel0: draft.resources.height ? Number(draft.resources.height) : undefined };
  }
  function continueToCompute() {
    if (!ready || !draft.selected.length || !resourcesValid || batch.locked) return;
    if (batch.ready) { navigate('results'); return; }
    batch.start(selection(draft.selected.map((slide) => slide.slidePath), null), draft.selected);
    navigate('compute');
  }
  function retryFailed() {
    if (!draft.batch?.request || !resourcesValid || batch.locked) return;
    const paths = draft.batch.selected.filter((slide) => { const state = batch.states.get(slide.slidePath); return !state?.execution || (!computeActive(state.execution) && state.execution.status !== 'completed'); }).map((slide) => slide.slidePath);
    if (paths.length) batch.retrySlides(selection(paths));
  }
  function openSaved(record: Interpretation) { if (batch.locked) return; navigate(record.execution?.status === 'completed' ? 'viewer' : 'compute', record.id, record.manifest.slides[0]?.slideId); }
  const activeRecord = linkedId ? batch.records.get(linkedId) ?? linkedRecord.data : undefined;
  const activeSlideId = parameters.get('slide') ?? activeRecord?.manifest.slides[0]?.slideId;
  const activeSlide = activeRecord?.manifest.slides.find((slide) => slide.slideId === activeSlideId);
  const activeState = activeSlide ? batch.states.get(activeSlide.slidePath) : undefined;
  const activeResult = !activeState?.error && activeState?.execution?.status === 'completed' ? activeState.execution.result?.slides?.find((slide) => slide.slideId === activeSlideId) : undefined;
  if (stage === 'viewer') return <div className="clinical-workspace model-chains clinical-insights interpretation-wizard interpretation-wizard-viewer">
    <PageHeader eyebrow="MODEL INTERPRETATION" title={`Attention ${activeSlide?.slideId ?? ''}`.trim()} description={activeRecord ? `${predictorMethodLabel(activeRecord.manifest.method)} · ${activeRecord.manifest.name}` : 'Loading the selected slide.'} actions={<button className="btn btn-secondary" onClick={() => navigate('results')}>Back to results</button>} />
    <ErrorNotice error={linkedRecord.error ?? activeState?.error ?? null} />
    {activeRecord && activeSlide && activeResult ? <AttentionSlideViewer project={project} record={activeRecord} slide={activeSlide} result={activeResult} /> : <Panel title="Attention is not ready"><p>{linkedRecord.isPending || !draft.batch ? 'Loading this saved slide and its evidence…' : 'This slide does not have verified completed attention yet.'}</p><button className="btn btn-secondary" onClick={() => navigate('compute', linkedId || undefined, activeSlideId)}>View computation progress</button></Panel>}
  </div>;
  if (stage === 'results') return <div className="clinical-workspace model-chains clinical-insights interpretation-wizard">
    <PageHeader eyebrow="MODEL INTERPRETATION" title="Attention results" description="Choose one of your selected slides to open its attention viewer." actions={<button className="btn btn-secondary" onClick={() => navigate('select')}>Back to slide selection</button>} />
    {batch.ready && draft.batch ? <><p className="callout science-success">All {draft.batch.selected.length} selected slides are ready.{draft.batch.items.some((item) => item.reused) ? ` ${draft.batch.items.filter((item) => item.reused).length} matching studies were reused.` : ''}</p><div className="slide-gallery-grid">{draft.batch.selected.map((slide) => { const item = draft.batch!.items.find((entry) => entry.slidePath === slide.slidePath)!; return <AttentionPreviewCard key={slide.slidePath} project={project} item={item} initial={item.interpretationId ? batch.records.get(item.interpretationId) : undefined} disabled={false} onOpen={(record) => navigate('viewer', record.id, slide.slideId)} />; })}</div></> : <Panel title="Selected attention is not ready"><p>Results open when every selected slide has completed and its evidence can be verified.</p><button className="btn btn-primary" onClick={() => navigate(draft.batch ? 'compute' : 'select')}>{draft.batch ? 'View computation progress' : 'Choose slides'}</button></Panel>}
  </div>;
  if (stage === 'compute') return <div className="clinical-workspace model-chains clinical-insights interpretation-wizard">
    <PageHeader eyebrow="MODEL INTERPRETATION" title="Prepare attention" description="Completed studies are skipped, active jobs are reused, and only missing attention is computed." actions={<button className="btn btn-secondary" onClick={() => navigate('select')}>Back to slide selection</button>} />
    {draft.batch ? <><p className="callout" role="status">{draft.batch.selected.filter((slide) => batch.states.get(slide.slidePath)?.execution?.status === 'completed').length} / {draft.batch.selected.length} selected slides completed.{batch.busy ? ' Checking existing attention and starting missing jobs…' : ' This page opens the results when all selected slides are ready.'}</p><button className="btn btn-secondary" onClick={batch.refresh}>Refresh computation status</button><ErrorNotice error={draft.batch.error ? new Error(draft.batch.error) : null} />
      {draft.batch.uncertain ? <p className="callout" role="alert">The request response is uncertain. Some jobs may already exist. <button className="btn btn-secondary" disabled={batch.busy} onClick={batch.retryExact}>Retry the same visualization request</button></p> : null}
      <div className="interpretation-progress-list">{draft.batch.selected.map((slide) => { const state = batch.states.get(slide.slidePath); const execution = state?.execution; const error = state?.error?.message ?? (!computeActive(execution) && execution?.status !== 'completed' ? state?.item?.error?.message ?? execution?.error : null); return <article className="interpretation-progress-row" key={slide.slidePath}><div><strong>{slide.name}</strong><small>{state?.item?.reused && execution?.status === 'completed' ? 'Already computed · skipped' : state?.item?.reused && computeActive(execution) ? 'Existing job reused' : execution ? computeStatusLabel(execution) : batch.busy ? 'Checking existing attention…' : state?.item?.error ? 'Could not start' : 'Awaiting computation'}{execution?.progress?.completedPairs !== undefined ? ` · ${execution.progress.completedPairs} / ${execution.progress.totalPairs} checkpoint passes` : ''}</small>{error ? <p className="slide-gallery-error">{error}</p> : null}{state?.item?.interpretationId && !draft.batch?.request && execution?.status !== 'completed' ? <ComputeJobControls project={project} id={state.item.interpretationId} kind="interpretation" initial={execution} /> : null}</div></article>; })}</div>
      {draft.batch.request ? <><InterpretationResources value={draft.resources} geometryReadOnly disabled={batch.locked} onChange={(resources) => setDraft((current) => ({ ...current, resources }))} /><button className="btn btn-secondary" disabled={batch.locked || !resourcesValid || !draft.batch.selected.some((slide) => { const job = batch.states.get(slide.slidePath)?.execution; return !job || (!computeActive(job) && job.status !== 'completed'); })} onClick={retryFailed}>Retry missing or failed attention</button></> : null}
    </> : <Panel title="Choose slides first"><p>No selected attention batch is available.</p><button className="btn btn-primary" onClick={() => navigate('select')}>Choose slides</button></Panel>}
  </div>;
  return <div className="clinical-workspace model-chains clinical-insights interpretation-wizard">
    <PageHeader eyebrow="04 CLINICAL INSIGHTS" title="Model interpretation" description="Choose a model and shared features, then select slides. Continue to compute missing attention and review your results." />
    <EvidenceChain current="interpretation" experimentId={predictor?.manifest.experimentId} predictorId={draft.predictorId} evaluationId={draft.evaluationId} clinicalAnalysisId={draft.clinicalId} />
    <ErrorNotice error={registry.error ?? sources.error ?? evaluations.error ?? reports.error ?? records.error ?? linkedClinical.error} />
    <Panel title="Choose model and shared features" subtitle="Slides are loaded automatically from the dataset attached to the feature bundle.">
      <fieldset className="interpretation-sources" disabled={batch.locked}><legend className="sr-only">Model and shared features</legend>
        <label className="label">Predictor<select className="field" value={draft.predictorId} onChange={(event) => changeSource({ predictorId: event.target.value, bundleId: null, packChoice: null, evaluationId: '', clinicalId: '' })}><option value="">Choose a frozen ABMIL predictor</option>{draft.predictorId && !predictor ? <option value={draft.predictorId} disabled>Linked predictor unavailable</option> : null}{(registry.data?.items ?? []).filter((item) => item.lifecycleState === 'active' || item.id === draft.predictorId && item.lifecycleState === 'archived').map((item) => <option key={item.id} value={item.id} disabled={item.manifest.recipe.model.toLowerCase() !== 'abmil'}>{item.manifest.name} · {predictorMethodLabel(item.manifest.method)}{item.manifest.recipe.model.toLowerCase() !== 'abmil' ? ' · attention unsupported' : ''}</option>)}</select></label>
        <label className="label">Feature bundle<select className="field" value={bundleId} disabled={!predictor || sources.isPending} onChange={(event) => changeSource({ bundleId: event.target.value, packChoice: null })}><option value="">Choose a frozen feature bundle</option>{bundleId && !source ? <option value={bundleId} disabled>Predictor’s feature bundle unavailable</option> : null}{(sources.data?.items ?? []).map((item) => <option key={item.id} value={item.id} disabled={!sourceCompatible(item, contract?.encoderId, contract?.dimensions, contract?.dtype)}>{item.name} · {item.slideCount.toLocaleString()} slides{!sourceCompatible(item, contract?.encoderId, contract?.dimensions, contract?.dtype) ? ' · incompatible or needs verification' : ''}</option>)}</select></label>
        <label className="label">Feature representation<select className="field" value={packId ?? '__unavailable__'} disabled={!source} onChange={(event) => changeSource({ packChoice: event.target.value })}>{packId === null ? <option value="__unavailable__" disabled>Choose a compatible feature representation</option> : null}<option value="" disabled={!representationCompatible(source?.dtype, contract?.dtype)}>Original slide features{source ? ` · ${dtypeLabel(source.dtype)}${!representationCompatible(source.dtype, contract?.dtype) ? ` · requires ${dtypeLabel(contract?.dtype)}` : ''}` : ''}</option>{source?.packs.map((item) => <option key={item.id} value={item.id} disabled={!representationCompatible(item.outputDtype, contract?.dtype)}>{item.name} · {item.outputDtype}{!representationCompatible(item.outputDtype, contract?.dtype) ? ` · requires ${dtypeLabel(contract?.dtype)}` : ''}</option>)}</select></label>
      </fieldset>
      {source ? <div className="interpretation-source-notes"><span>Dataset: {source.datasetName ?? source.datasetId ?? 'Frozen dataset'}</span><span>Encoder: {source.encoderId ?? 'unknown'}</span><span>Predictor requires {dtypeLabel(contract?.dtype)}</span><span>{pack ? `Packed features: ${pack.name}` : 'Original feature files'}</span></div> : null}
      {source?.slideFolderFinding || source && !source.slideFolder ? <p className="callout">{source.slideFolderFinding?.message ?? 'The frozen dataset does not identify a usable slide location. Correct its slide mapping in Data preparation and freeze an updated feature bundle.'}</p> : null}
      {source && !representationMatches ? <p className="callout">This predictor requires {dtypeLabel(contract?.dtype)} features. Choose a compatible original source or frozen pack.</p> : null}
      <InterpretationResources value={draft.resources} disabled={batch.locked} onChange={(resources) => setDraft((current) => ({ ...current, resources, batch: resources.width !== current.resources.width || resources.height !== current.resources.height ? null : current.batch }))} />
      <details><summary>Evaluation and clinical evidence links</summary><fieldset className="chain-fields" disabled={batch.locked}><legend className="sr-only">Evidence links</legend><label className="label">Linked evaluation (optional)<select className="field" value={draft.evaluationId} onChange={(event) => changeSource({ evaluationId: event.target.value, clinicalId: '' })}><option value="">Standalone slide interpretation</option>{draft.evaluationId && !matchingEvaluations.some((item) => item.id === draft.evaluationId) ? <option value={draft.evaluationId} disabled>Linked evaluation unavailable or incompatible</option> : null}{matchingEvaluations.map((item) => <option key={item.id} value={item.id}>{item.manifest.name}</option>)}</select></label><label className="label">Linked clinical report (optional)<select className="field" value={draft.clinicalId} onChange={(event) => { const report = matchingReports.find((item) => item.id === event.target.value); changeSource({ clinicalId: event.target.value, evaluationId: report?.manifest.evaluationId ?? draft.evaluationId }); }}><option value="">No clinical report</option>{draft.clinicalId && !matchingReports.some((item) => item.id === draft.clinicalId) ? <option value={draft.clinicalId} disabled>Linked report unavailable or incompatible</option> : null}{matchingReports.map((item) => <option key={item.id} value={item.id}>{item.manifest.name}</option>)}</select></label></fieldset></details>
      {!resourcesValid ? <p className="callout">Use valid CPU/GPU resources, positive finite memory and either both positive footprint dimensions or neither.</p> : null}
      {!contextValid ? <p className="callout">Choose compatible evaluation and clinical evidence, or clear their optional links.</p> : null}
    </Panel>
    {ready ? <GalleryWorkspace project={project} source={gallerySource} selected={draft.selected} search={draft.search} offset={draft.offset} locked={batch.locked} canContinue={resourcesValid} onSelection={selectSlides} onSearch={(search) => setDraft((current) => ({ ...current, search, offset: 0 }))} onPage={(offset) => setDraft((current) => ({ ...current, offset }))} onContinue={continueToCompute} /> : <Panel title="Choose slides"><p className="muted">Choose a compatible model and feature bundle to browse its dataset slides.</p></Panel>}
    {batch.locked ? <p className="callout">Your attention request is still being resolved. <button className="text-button" onClick={() => navigate('compute')}>Return to computation</button></p> : null}
    <details className="interpretation-history"><summary>Open a saved attention study</summary>{(records.data?.items ?? []).filter((item) => item.lifecycleState !== 'trashed').map((item) => <article className="report-card" key={item.id}><button className="text-button" disabled={batch.locked} onClick={() => openSaved(item)}>{item.manifest.name}</button><small>{item.manifest.slides.length} slides · {predictorMethodLabel(item.manifest.method)}</small></article>)}</details>
  </div>;
}

export function GalleryWorkspace({ project, source, selected, search, offset, locked, canContinue, onSelection, onSearch, onPage, onContinue }: { project: string; source: GallerySource; selected: GallerySlide[]; search: string; offset: number; locked: boolean; canContinue: boolean; onSelection: (slides: GallerySlide[]) => void; onSearch: (search: string) => void; onPage: (offset: number) => void; onContinue: () => void }) {
  const [settledSearch, setSettledSearch] = useState(search.trim());
  useEffect(() => { const timer = setTimeout(() => setSettledSearch(search.trim()), 250); return () => clearTimeout(timer); }, [search]);
  const gallery = useQuery({ queryKey: ['interpretation-gallery', project, source, settledSearch, offset], queryFn: ({ signal }) => interpretations.gallery(project, source, settledSearch, offset, signal), staleTime: 15000 });
  const current = search.trim() === settledSearch && !gallery.isFetching && !gallery.isError;
  const paths = new Set(selected.map((slide) => slide.slidePath));
  function toggle(slide: GallerySlide, checked: boolean) { if (!checked) onSelection(selected.filter((item) => item.slidePath !== slide.slidePath)); else if (slide.available && !paths.has(slide.slidePath) && selected.length < 128) onSelection([...selected, slide]); }
  return <Panel title="Choose slides" subtitle="Click thumbnails or checkboxes to select slides. No computation starts until you continue.">
    <div className="slide-gallery-tools"><label className="label">Search slides<input type="search" className="field" maxLength={200} value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Search slide name or relative path" /></label><button className="btn btn-primary" disabled={!selected.length || !canContinue || locked} onClick={onContinue}>Continue with {selected.length} selected</button><button className="btn btn-secondary" disabled={!selected.length || locked} onClick={() => onSelection([])}>Clear selection</button></div>
    <div className="slide-gallery-count">{gallery.data ? `${gallery.data.total.toLocaleString()} matching slides` : 'Reading dataset slides'} · {selected.length} selected across all searches and pages{selected.length >= 128 ? ' · batch limit reached' : ''}.</div>
    {selected.length ? <details><summary>Selected slides ({selected.length})</summary><ul>{selected.map((slide) => <li key={slide.slidePath}>{slide.name} <button className="text-button" disabled={locked} onClick={() => toggle(slide, false)}>Remove from selection</button></li>)}</ul></details> : null}
    <ErrorNotice error={gallery.error} />
    {gallery.data?.warnings.length ? <ul className="callout">{gallery.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
    {!current && !gallery.isError ? <p role="status">Searching and matching slide features…</p> : null}
    {current && !gallery.data?.items.length ? <p className="callout">No slides match{settledSearch ? ` “${settledSearch}”` : ' this dataset'}. Your selection is preserved.</p> : null}
    <div className="slide-gallery-grid">{current ? gallery.data?.items.map((slide) => <SlideGalleryCard key={slide.slidePath} project={project} slide={slide} checked={paths.has(slide.slidePath)} disabled={locked} selectionDisabled={!paths.has(slide.slidePath) && selected.length >= 128} onToggle={(checked) => toggle(slide, checked)} />) : null}</div>
    {gallery.data && (offset > 0 || gallery.data.hasMore) ? <div className="slide-gallery-pagination"><button className="btn btn-secondary" disabled={!current || offset === 0} onClick={() => onPage(Math.max(0, offset - 24))}>Previous slides</button><span>Page {Math.floor(offset / 24) + 1} / {Math.max(1, Math.ceil(gallery.data.total / 24))}</span><button className="btn btn-secondary" disabled={!current || !gallery.data.hasMore} onClick={() => onPage(offset + 24)}>Next slides</button></div> : null}
  </Panel>;
}
