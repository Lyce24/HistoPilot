import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { evaluation } from '../api/evaluation';
import { experiments, experimentPollInterval } from '../api/experiments';
import { modelEvaluations, predictors, predictorMethodLabel, computeActive, computePollInterval, type ModelEvaluation, type EvaluationSelection } from '../api/predictors';
import type { Workspace } from '../api/types';
import { EmptyState, ErrorNotice, PageHeader, Panel } from '../components/ui';
import { Findings } from '../components/ScientificUI';
import PublicationConfirmation from '../components/PublicationConfirmation';
import { useReviewedPublication } from '../components/useReviewedPublication';
import { useHashParameters } from '../lib/hashRoute';
import { versionLabelText } from '../lib/versionLabels';
import ComputeJobControls from '../components/ComputeJobControls';
import { shortRecordId } from '../lib/recordLabels';
import { downloadJSON } from '../lib/download';
import BulkEvaluationRunner, { EvaluationBatchStatus } from '../components/BulkEvaluationRunner';
import { bulkEvaluations, bulkEvaluationActive, bulkEvaluationPollInterval } from '../api/bulkEvaluations';
import { StageCreateButton, StageBackButton, StageContinueButton, StageLibrary, StageLibraryToolbar, StageRecordManageButton, StagePage, StageSteps, useStageLibrary } from '../components/StageWorkflow';
import EvaluationInputSettings, { initialEvaluationInputs, evaluationExecutionSelection, EvaluationCoverageSummary, type EvaluationExecutionInputs } from '../components/EvaluationInputSettings';
import { reportEditorValidity } from '../components/NumericField';
import PatientAnalysisResults from '../components/PatientAnalysisResults';
import CaseReviewWorkspace from '../components/CaseReviewWorkspace';
import type { CaseQuery } from '../api/caseReview';
import EvaluationResultsTable, { newEvaluationRecordFilters } from '../components/EvaluationResultsTable';
import EvidenceChain, { evidenceLink } from '../components/EvidenceChain';
import { groupPredictors, predictorConfigurationLabel, experimentPredictorLink } from '../lib/predictorGroups';
import './ModelChains.css';

export default function LocalModelEvaluation({ workspace }: { workspace: Workspace }) {
  const parameters = useHashParameters();
  const linkedExperiment = parameters.get('experiment') ?? '';
  const linkedPredictor = parameters.get('predictor') ?? '';
  const linkedCohort = parameters.get('cohort') ?? '';
  const linkedEvaluation = parameters.get('evaluation') ?? '';
  return <EvaluationWorkspace key={`${workspace.project.id}:${linkedPredictor}:${linkedCohort}:${linkedEvaluation}:${linkedExperiment}`} workspace={workspace} linkedPredictor={linkedPredictor} linkedCohort={linkedCohort} linkedEvaluation={linkedEvaluation} linkedExperiment={linkedExperiment} />;
}

function EvaluationWorkspace({ workspace, linkedPredictor, linkedCohort, linkedEvaluation, linkedExperiment }: { workspace: Workspace; linkedPredictor: string; linkedCohort: string; linkedEvaluation: string; linkedExperiment: string }) {
  const project = workspace.project.id;
  const client = useQueryClient();
  // Every list refreshes quickly while its own jobs are running and slowly once
  // they finish, so an open results page does not keep re-reading saved records.
  const batches = useQuery({ queryKey: ['evaluation-batches', project], queryFn: () => bulkEvaluations.list(project, true), refetchIntervalInBackground: false, refetchInterval: (query) => bulkEvaluationPollInterval(query.state.data) });
  const running = Boolean(batches.data?.items.some(bulkEvaluationActive));
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), refetchIntervalInBackground: false, refetchInterval: () => running ? 5000 : 60000 });
  const experimentRegistry = useQuery({ queryKey: ['model-experiment-summaries', project], queryFn: () => experiments.summaries(project), refetchIntervalInBackground: false, refetchInterval: (query) => experimentPollInterval(query.state.data) });
  const cohorts = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project) });
  const records = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const [view, setView] = useState<'library' | 'setup' | 'detail' | 'batch' | 'comparison'>(linkedEvaluation ? 'detail' : linkedPredictor || linkedCohort || linkedExperiment ? 'setup' : 'library');
  const [setupStarted, setSetupStarted] = useState(Boolean(linkedPredictor || linkedCohort || linkedExperiment));
  const [setupKey, setSetupKey] = useState(0);
  const [libraryTab, setLibraryTab] = useState<'records' | 'batches'>('records');
  const [recordFilters, setRecordFilters] = useState(newEvaluationRecordFilters);
  const [batchSearch, setBatchSearch] = useState('');
  const [batchState, setBatchState] = useState('active');
  const [batchStatus, setBatchStatus] = useState('all');
  const [batchSort, setBatchSort] = useState('recent');
  const [batchId, setBatchId] = useState('');
  const [experimentIds, setExperimentIds] = useState<string[]>(() => linkedExperiment ? [linkedExperiment] : []);
  const experimentId = experimentIds.length === 1 ? experimentIds[0] : '';
  const setExperimentId = (id: string) => setExperimentIds(id ? [id] : []);
  const [predictorId, setPredictorId] = useState<string | null>(null);
  const chosenPredictor = predictorId ?? linkedPredictor;
  const [cohortId, setCohortId] = useState(linkedCohort);
  const [name, setName] = useState('');
  const [executionInputs, setExecutionInputs] = useState<EvaluationExecutionInputs | null>(null);
  const editor = useRef<HTMLFieldSetElement>(null);
  const [bulkLocked, setBulkLocked] = useState(false);
  const [bulkOpened, setBulkOpened] = useState(!linkedPredictor);
  const [mode, setMode] = useState<'batch' | 'single'>(linkedPredictor ? 'single' : 'batch');
  const [selectedRecord, setSelectedRecord] = useState(linkedEvaluation);
  const publication = useReviewedPublication(
    (selection: EvaluationSelection) => modelEvaluations.preview(project, selection),
    (selection, hash, operation) => modelEvaluations.save(project, selection, hash, operation),
    (preview) => preview.canSave && !preview.findings.some((finding) => finding.severity === 'error'),
    async (record) => { setSelectedRecord(record.id); setView('detail'); await Promise.all([client.invalidateQueries({ queryKey: ['model-evaluations', project] }), client.invalidateQueries({ queryKey: ['cleanup', project] })]); },
  );
  const predictor = registry.data?.items.find((item) => item.id === chosenPredictor && item.lifecycleState !== 'trashed');
  const cohort = cohorts.data?.items.find((item) => item.id === cohortId);
  const availableCohorts = cohorts.data?.items ?? [];
  const inputs = executionInputs ?? initialEvaluationInputs(cohort);
  const cohortReady = Boolean(cohort);
  const canReview = predictor?.lifecycleState === 'active' && cohortReady && name.trim() && !registry.isError && !cohorts.isError;
  const sources = groupPredictors((registry.data?.items ?? []).filter((item) => item.lifecycleState === 'active' || (item.id === chosenPredictor && item.lifecycleState === 'archived')));
  const choices = sources.filter((item) => !experimentId || item.id === experimentId);
  const visible = (records.data?.items ?? []).filter((item) => (!experimentIds.length || experimentIds.includes(item.manifest.experimentId)) && (mode !== 'single' || !chosenPredictor || item.manifest.predictorId === chosenPredictor));
  const detail = (records.data?.items ?? []).find((item) => item.id === selectedRecord) ?? (publication.saved?.id === selectedRecord ? publication.saved : undefined);
  const singlePage = publication.review ? 'review' : 'inputs';
  const locked = bulkLocked || publication.locked;
  function openRecord(id: string) { if (locked) return; setSelectedRecord(id); setView('detail'); void records.refetch(); }
  function openLibrary() { if (!locked) setView('library'); }
  useStageLibrary(openLibrary);
  function create() {
    if (locked) return;
    publication.reset(); setSelectedRecord(''); setBulkOpened(true); setMode('batch'); setPredictorId('');
    setExperimentIds(linkedExperiment ? [linkedExperiment] : []); setCohortId(linkedCohort); setName(''); setExecutionInputs(null);
    setSetupStarted(true); setSetupKey((key) => key + 1); setView('setup');
  }
  const predictorName = (id: string) => registry.data?.items.find((item) => item.id === id)?.manifest.name ?? id;
  const cohortName = (id: string) => { const item = cohorts.data?.items.find((item) => item.id === id); return item ? versionLabelText(item, 'Test cohort') : id; };
  const batchRows = batches.data?.items ?? [];
  const visibleBatches = batchRows.filter((item) => (batchState === 'all' || (item.lifecycleState ?? 'active') === batchState)
    && (batchStatus === 'all' || item.status === batchStatus)
    && `${item.name ?? ''} ${item.id} ${cohortName(item.cohortId)} ${item.items.map((source) => source.predictorName ?? source.predictorId).join(' ')}`.toLowerCase().includes(batchSearch.trim().toLowerCase()))
    .sort((a, b) => (batchSort === 'name' ? (a.name ?? a.id).localeCompare(b.name ?? b.id) : batchSort === 'oldest' ? (a.createdAt ?? '').localeCompare(b.createdAt ?? '') : (b.createdAt ?? '').localeCompare(a.createdAt ?? '')) || a.id.localeCompare(b.id));
  function resetBatchFilters() { setBatchSearch(''); setBatchState('active'); setBatchStatus('all'); setBatchSort('recent'); }
  const libraryActions = <><button type="button" className="btn btn-secondary btn-small" disabled={records.isFetching || batches.isFetching} onClick={() => void Promise.all([records.refetch(), batches.refetch(), cohorts.refetch(), registry.refetch(), experimentRegistry.refetch()])}>Refresh</button>{setupStarted ? <button type="button" className="btn btn-secondary btn-small" onClick={() => setView('setup')}>Resume evaluation setup</button> : null}<button type="button" className="btn btn-secondary btn-small" onClick={() => setView('comparison')}>Compare methods</button>{experimentIds.length || chosenPredictor ? <button type="button" className="text-button" onClick={() => { setExperimentIds([]); setPredictorId(''); publication.reset(); }}>Show all experiments and predictors</button> : null}</>;
  return <div className="clinical-workspace model-chains">
    <PageHeader eyebrow="03 EVALUATE" title={view === 'library' ? 'Model evaluations' : view === 'detail' ? detail?.manifest.name ?? 'Evaluation results' : view === 'batch' ? 'Evaluation batch' : view === 'comparison' ? 'Compare evaluation methods' : 'Evaluate models'} description={view === 'library' ? 'Open evaluation results or create an evaluation for your models.' : 'Select development models and a frozen test cohort, review compatibility, then run evaluation.'} actions={<div className="inline-actions">{view === 'library' ? <StageCreateButton onClick={create}>Create evaluation</StageCreateButton> : <StageBackButton disabled={locked} onClick={openLibrary}>Back to evaluations</StageBackButton>}{view !== 'library' ? <a className="btn btn-secondary" href="#test-data">Test cohorts</a> : null}</div>} />
    {view !== 'library' ? <EvidenceChain current="evaluation" experimentId={detail?.manifest.experimentId ?? predictor?.manifest.experimentId ?? experimentId} predictorId={detail?.manifest.predictorId ?? chosenPredictor} evaluationId={selectedRecord} /> : null}
    <ErrorNotice error={publication.error ?? records.error ?? registry.error ?? cohorts.error ?? experimentRegistry.error ?? batches.error} />
    <StagePage pageKey={`${view}:${mode}:${view === 'setup' ? singlePage : selectedRecord || batchId}`}>
    {setupStarted ? <div hidden={view !== 'setup'}>
    <p className="callout">Choose one or more experiments to evaluate. Each ready ensemble or refit predictor keeps its own results. Unlabeled slides receive predictions; metrics use labeled records only.</p>
    <nav className="run-tabs" aria-label="Evaluation setup"><button className={mode === 'batch' ? 'selected' : ''} disabled={locked} onClick={() => { setBulkOpened(true); setMode('batch'); }}>Evaluate experiments</button><button className={mode === 'single' ? 'selected' : ''} disabled={locked} onClick={() => setMode('single')}>Advanced: single predictor plan</button></nav>
    {bulkOpened ? <div hidden={mode !== 'batch'}><Panel title="Evaluate selected experiments" subtitle="Choose experiments and methods to compare on one test cohort."><BulkEvaluationRunner key={setupKey} project={project} predictors={registry.data?.items ?? []} experiments={experimentRegistry.data?.items ?? []} experimentsLoading={experimentRegistry.isPending} cohorts={cohorts.data?.items ?? []} linkedCohort={linkedCohort} experimentIds={experimentIds} onExperimentsChange={setExperimentIds} onLockChange={setBulkLocked} onOpenEvaluation={openRecord} /></Panel></div> : null}
    {mode === 'single' ? <>
    <StageSteps label="Single evaluation steps" current={singlePage} disabled={publication.locked} steps={[{ id: 'inputs', title: 'Evaluation inputs', description: 'Predictor, test cohort and features' }, { id: 'review', title: 'Review and save', description: 'Verify compatibility and coverage', disabled: !publication.review }]} onChange={(next) => { if (next === 'inputs') publication.reset(); }} />
    {!publication.review ? <Panel title="Select evaluation inputs" subtitle="Each plan uses one frozen predictor and one test cohort. The same predictor can have several evaluations.">
      <fieldset ref={editor} className="chain-fields" disabled={publication.locked} onChange={() => publication.reset()}>
        <legend className="sr-only">Evaluation inputs</legend>
        <label className="label">Source experiment<select className="field" value={experimentId} onChange={(event) => { setExperimentId(event.target.value); setPredictorId(''); setSelectedRecord(''); }}><option value="">All experiments</option>{experimentId && !sources.some((item) => item.id === experimentId) ? <option value={experimentId}>Linked experiment · no ready predictors</option> : null}{sources.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.items.length} predictors</option>)}</select></label>
        <label className="label">Predictor<select className="field" value={chosenPredictor} onChange={(event) => { setPredictorId(event.target.value); setSelectedRecord(''); }}><option value="">Choose a ready predictor</option>{chosenPredictor && !predictor ? <option value={chosenPredictor} disabled>Linked predictor unavailable</option> : null}{choices.map((group) => <optgroup key={group.id} label={group.name}>{group.items.map((item) => <option key={item.id} value={item.id}>{predictorConfigurationLabel(item.manifest)} · Train {item.manifest.trainingSeed} / split {item.manifest.splitSeed} · {predictorMethodLabel(item.manifest.method)}{item.lifecycleState === 'archived' ? ' · archived' : ''}</option>)}</optgroup>)}</select></label>
        <label className="label">Test cohort<select className="field" value={cohortId} disabled={!predictor} onChange={(event) => { setCohortId(event.target.value); setExecutionInputs(null); }}><option value="">Choose a frozen test cohort</option>{cohortId && !cohort ? <option value={cohortId} disabled>Selected test cohort unavailable</option> : null}{availableCohorts.map((item) => <option key={item.id} value={item.id}>{versionLabelText(item, 'Test cohort')} · {item.manifest.summary.includedSlides} slides{!item.current ? ' · needs verification' : ''}</option>)}</select></label>
        <label className="label">Evaluation name<input className="field" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} placeholder="For example: External validation" /></label>
        <div className="chain-wide"><EvaluationInputSettings project={project} cohort={cohort} value={inputs} onChange={(value) => { setExecutionInputs(value); publication.reset(); }} disabled={publication.locked} /></div>
      </fieldset>
      {predictor ? <p className="muted">Source: <a href={experimentPredictorLink(predictor.manifest.experimentId, predictor.id)}>{predictor.manifest.experiment?.name ?? shortRecordId(predictor.manifest.experimentId)}</a> · {predictorMethodLabel(predictor.manifest.method)} · {predictor.manifest.checkpoints.length} checkpoints · class target {predictor.manifest.target.field}</p> : <p>Choose a ready predictor from <a href="#experiments">Experiments</a>. Predictors appear automatically when the selected ensemble or refit work finishes. Batches using Skip have no predictors; use one as a template to choose a different policy.</p>}
      {predictor?.lifecycleState === 'archived' ? <p className="callout">This predictor is archived. Restore it to Active using <StageRecordManageButton type="configuration" id={predictor.id} name={predictor.manifest.name} /> before creating another evaluation; existing results remain available in the evaluation library.</p> : null}
      {!availableCohorts.length ? <p className="callout">Create and freeze a cohort in <a href="#test-data">Test cohorts</a>. Its prediction targets and feature readiness are checked here for each model.</p> : null}
      <div className="stage-actions"><StageContinueButton disabled={publication.locked || !canReview} onClick={() => { if (canReview && reportEditorValidity(editor.current)) void publication.preview({ predictorId: chosenPredictor, cohortId, name: name.trim(), ...evaluationExecutionSelection(inputs) }); }}>Review evaluation</StageContinueButton></div>
    </Panel> : null}
    {publication.review ? <Panel title="Review evaluation inputs">
      <Findings findings={publication.review.preview.findings} />
      <EvaluationCoverageSummary manifest={publication.review.preview.manifest} />
      <p><strong>{publication.review.selection.name}</strong>: {predictorName(publication.review.selection.predictorId)} → {cohortName(publication.review.selection.cohortId)}</p>
      <p>The review checks frozen weights, target encoding, development provenance, feature compatibility and test-slide coverage.</p>
      <div className="stage-actions"><StageBackButton disabled={publication.locked} onClick={publication.reset}>Back to evaluation inputs</StageBackButton></div>
      {publication.review.preview.canSave ? <PublicationConfirmation busy={publication.busy} uncertain={publication.review.uncertain} acknowledged={publication.acknowledged} onAcknowledge={publication.setAcknowledged} onConfirm={() => void publication.publish()} label="Save evaluation plan" /> : null}
    </Panel> : null}
    </> : null}
    </div> : null}
    {view === 'library' ? <StageLibrary project={project} title="Saved evaluations">
      {batchRows.length || libraryTab === 'batches' ? <div className="stage-library-tabs" role="group" aria-label="Evaluation records"><button type="button" className={libraryTab === 'records' ? 'selected' : ''} aria-pressed={libraryTab === 'records'} onClick={() => setLibraryTab('records')}>Evaluations <span>{visible.length}</span></button><button type="button" className={libraryTab === 'batches' ? 'selected' : ''} aria-pressed={libraryTab === 'batches'} onClick={() => setLibraryTab('batches')}>Batches <span>{batchRows.length}</span></button></div> : null}
      {libraryTab === 'records' ? <EvaluationResultsTable project={project} view="records" records={visible} predictors={registry.data?.items ?? []} experiments={experimentRegistry.data?.items ?? []} cohorts={cohorts.data?.items ?? []} loading={records.isPending} onOpen={openRecord} filters={recordFilters} onFiltersChange={setRecordFilters} actions={libraryActions} onCreate={create} /> : <>
        <StageLibraryToolbar search={batchSearch} onSearch={setBatchSearch} searchLabel="Search evaluation batches" placeholder="Name, ID, cohort or predictor" count={batches.isPending ? undefined : visibleBatches.length} total={batchRows.length} actions={libraryActions}
          onReset={batchSearch || batchState !== 'active' || batchStatus !== 'all' || batchSort !== 'recent' ? resetBatchFilters : undefined}>
          <label className="label">State<select className="field" aria-label="Evaluation batch state" value={batchState} onChange={(event) => setBatchState(event.target.value)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label>
          <label className="label">Status<select className="field" aria-label="Evaluation batch status" value={batchStatus} onChange={(event) => setBatchStatus(event.target.value)}><option value="all">All statuses</option>{[...new Set(batchRows.map((item) => item.status))].sort().map((status) => <option key={status} value={status}>{status.replaceAll('_', ' ')}</option>)}</select></label>
          <label className="label">Sort<select className="field" aria-label="Sort evaluation batches" value={batchSort} onChange={(event) => setBatchSort(event.target.value)}><option value="recent">Newest first</option><option value="oldest">Oldest first</option><option value="name">Name</option></select></label>
        </StageLibraryToolbar>
        {batches.isPending ? <p role="status">Loading evaluation batches…</p> : visibleBatches.length ? <div className="table-wrap"><table className="chain-table"><thead><tr><th scope="col">Batch</th><th scope="col">Status</th><th scope="col">Test cohort</th><th scope="col">Predictors</th><th scope="col">Actions</th></tr></thead><tbody>{visibleBatches.map((batch) => <tr key={batch.id}><th scope="row">{batch.lifecycleState === 'trashed' ? batch.name ?? shortRecordId(batch.id) : <button type="button" className="text-button stage-record-name" onClick={() => { setBatchId(batch.id); setView('batch'); }}>{batch.name ?? shortRecordId(batch.id)}</button>}{batch.lifecycleState === 'archived' || batch.lifecycleState === 'trashed' ? <small>{batch.lifecycleState === 'trashed' ? 'Trash' : 'Archived'}</small> : null}</th><td>{batch.status.replaceAll('_', ' ')}</td><td>{cohortName(batch.cohortId)}</td><td>{batch.items.length}</td><td><StageRecordManageButton type="configuration" id={batch.id} name={batch.name ?? batch.id} /></td></tr>)}</tbody></table></div> : <EmptyState title={batchRows.length ? 'No matching evaluation batches' : 'No evaluation batches yet'} description={batchRows.length ? 'Try another search or clear the filters.' : 'Create an evaluation to run several model predictors together.'} />}
      </>}
    </StageLibrary> : null}
    {view === 'comparison' ? <Panel title="Ensemble and refit comparison" subtitle="Compare matched results on the same test cohort and scoring unit."><EvaluationResultsTable project={project} view="comparison" filters={recordFilters} onFiltersChange={setRecordFilters} records={visible} predictors={registry.data?.items ?? []} experiments={experimentRegistry.data?.items ?? []} cohorts={cohorts.data?.items ?? []} loading={records.isPending} onOpen={openRecord} /></Panel> : null}
    {view === 'detail' ? <>{detail ? <EvaluationDetail key={detail.id} project={project} record={detail} comparisons={records.data?.items ?? []} /> : <Panel title="Evaluation results"><p role="status">{records.isPending ? 'Loading evaluation…' : records.isError ? 'The evaluation could not be loaded. Retry to check this record.' : 'This evaluation is unavailable. Return to the evaluation library to choose a saved record.'}</p>{records.isError ? <button className="btn btn-secondary" onClick={() => void records.refetch()}>Retry loading evaluation</button> : null}</Panel>}</> : null}
    {view === 'batch' ? <><Panel title="Evaluation batch"><EvaluationBatchStatus project={project} id={batchId} onOpen={openRecord} /></Panel></> : null}
    </StagePage>
  </div>;
}

function EvaluationDetail({ project, record, comparisons }: { project: string; record: ModelEvaluation; comparisons: ModelEvaluation[] }) {
  const [error, setError] = useState<Error | null>(null);
  const [downloading, setDownloading] = useState(false);
  const trashed = record.lifecycleState === 'trashed';
  const shouldPoll = !trashed || computeActive(record.execution);
  const execution = useQuery({ queryKey: ['compute-job', project, 'evaluation', record.id], queryFn: () => modelEvaluations.execution(project, record.id), initialData: record.execution, enabled: shouldPoll, refetchInterval: (query) => shouldPoll && computeActive(query.state.data) ? 3000 : false });
  const current = shouldPoll ? execution.data : record.execution;
  const result = !execution.isError && current?.status === 'completed' ? current.result : null;
  const metrics = result?.metrics;
  const [unit, setUnit] = useState<'selected' | 'slide' | 'patient'>('selected');
  const [caseFilter, setCaseFilter] = useState<Partial<CaseQuery> | null>(null);
  const selected = metrics?.[unit];
  async function download(filename: 'slide-predictions.csv' | 'patient-predictions.csv' | 'metrics.json') {
    if (trashed || !result || downloading) return;
    setError(null); setDownloading(true);
    try { await modelEvaluations.download(project, record.id, filename); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Download failed.')); }
    finally { setDownloading(false); }
  }
  return <Panel title={record.manifest.name} subtitle="Evaluation run and results">
    <ComputeJobControls project={project} id={record.id} kind="evaluation" initial={record.execution} readOnly={record.lifecycleState !== 'active'} />
    <ErrorNotice error={error} />
    {result ? <>
      {trashed ? <p className="callout">These retained results belong to an evaluation in Trash. Restore it using <StageRecordManageButton type="configuration" id={record.id} name={record.manifest.name} /> to download its files.</p> : null}
      <label className="label">Metrics by prediction unit<select className="field" value={unit} onChange={(event) => setUnit(event.target.value as typeof unit)}><option value="selected">Target unit ({metrics?.unit ?? 'configured'})</option><option value="slide">Slide</option><option value="patient">Patient</option></select></label>
      {metrics?.positiveClass && typeof metrics.decisionThreshold === 'number' ? <p className="muted">Predict {metrics.positiveClass} when its probability is at least {metrics.decisionThreshold}. Ranking metrics use the original probabilities.</p> : null}
      {typeof selected?.unlabeledCount === 'number' && selected.unlabeledCount > 0 ? <p className="muted">{selected.unlabeledCount} unlabeled {unit === 'selected' ? metrics?.unit : unit} records have predictions and are excluded from these metrics.</p> : null}
      {selected?.available ? <><p>{selected.count} labeled {unit === 'selected' ? metrics?.unit : unit} records. Classes: {metrics?.classOrder.join(', ')}.</p><div className="chain-metrics">{(['accuracy', 'balancedAccuracy', 'macroF1', 'auroc', 'auprc', 'loss'] as const).map((key) => <div key={key}><strong>{typeof selected[key] === 'number' ? selected[key].toFixed(4) : 'Unavailable'}</strong><span>{{ accuracy: 'Accuracy', balancedAccuracy: 'Balanced accuracy', macroF1: 'Macro F1', auroc: 'AUROC', auprc: 'AUPRC', loss: 'Log loss' }[key]}</span></div>)}</div>
      {selected.missingClasses?.length ? <p className="callout">Classes absent from labeled test records: {selected.missingClasses.join(', ')}. Some metrics cannot be estimated.</p> : null}
      {selected.confusionMatrix ? <details><summary>Confusion matrix</summary><p>Rows: actual class. Columns: predicted class. Select a count to review its cases.</p><div className="table-wrap"><table className="chain-table"><thead><tr><th>Actual / predicted</th>{metrics?.classOrder.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{selected.confusionMatrix.map((row, index) => <tr key={index}><th scope="row">{metrics?.classOrder[index]}</th>{row.map((value, column) => <td key={column}><button type="button" className="text-button" disabled={trashed || !value} aria-label={`Review ${value} cases: actual ${metrics?.classOrder[index]}, predicted ${metrics?.classOrder[column]}`} onClick={() => setCaseFilter({ unit, actualClass: index, predictedClass: column })}>{value}</button></td>)}</tr>)}</tbody></table></div></details> : null}</> : <p className="callout">{selected?.reason ?? 'No labeled records are available for these metrics.'} Predictions are still available.</p>}
      <PatientAnalysisResults value={metrics?.patientAnalysis} />
      {!trashed ? <><div className="inline-actions"><button type="button" className="btn btn-secondary" onClick={() => setCaseFilter({ unit, outcome: 'all' })}>Review cases and model errors</button>{caseFilter ? <button type="button" className="text-button" onClick={() => setCaseFilter(null)}>Close case review</button> : null}</div>{caseFilter ? <CaseReviewWorkspace key={JSON.stringify(caseFilter)} project={project} evaluation={record} comparisons={comparisons} initial={caseFilter} /> : null}</> : null}
      <div className="inline-actions"><button className="btn btn-secondary" disabled={downloading || trashed} onClick={() => void download('slide-predictions.csv')}>Download slide predictions</button><button className="btn btn-secondary" disabled={downloading || trashed} onClick={() => void download('patient-predictions.csv')}>Download patient predictions</button><button className="btn btn-secondary" disabled={downloading || trashed} onClick={() => void download('metrics.json')}>Download metrics</button></div>
      {!trashed ? <div className="inline-actions"><StageContinueButton href={evidenceLink('clinical-utility', { experimentId: record.manifest.experimentId, predictorId: record.manifest.predictorId, evaluationId: record.id })}>Continue to clinical utility</StageContinueButton><a className="btn btn-secondary" href={evidenceLink('interpretation', { experimentId: record.manifest.experimentId, predictorId: record.manifest.predictorId, evaluationId: record.id })}>Inspect slide attention</a></div> : null}
    </> : <p>{execution.isError ? 'Results cannot be verified. Resolve the execution error before viewing or downloading results.' : 'No completed results yet. Run this evaluation to generate predictions and metrics.'}</p>}
    <div className="inline-actions"><button className="btn btn-secondary" type="button" onClick={() => downloadJSON(`${record.manifest.name}.json`, { ...record, execution: current })}>Export evaluation record</button></div>
    <details><summary>Saved inputs, settings and results</summary><pre className="chain-details">{JSON.stringify({ manifest: record.manifest, execution: current }, null, 2)}</pre></details>
  </Panel>;
}
