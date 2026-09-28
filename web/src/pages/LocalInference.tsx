import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { evaluation } from '../api/evaluation';
import { experiments, experimentPollInterval } from '../api/experiments';
import { computePollInterval, computeStatusLabel, modelEvaluations, predictorMethodLabel, predictors, type ModelEvaluation } from '../api/predictors';
import { bulkEvaluations, bulkEvaluationActive, bulkEvaluationPollInterval } from '../api/bulkEvaluations';
import type { Workspace } from '../api/types';
import { lifecycleLabel } from '../api/lifecycle';
import BulkEvaluationRunner, { EvaluationBatchStatus } from '../components/BulkEvaluationRunner';
import EvidenceChain from '../components/EvidenceChain';
import { InferenceRunDetail } from '../components/InferenceResults';
import { StageBackButton, StageCreateButton, StageLibrary, StageLibraryToolbar, StagePage, StageRecordManageButton, useStageLibrary } from '../components/StageWorkflow';
import { Badge, EmptyState, ErrorNotice, PageHeader, Panel } from '../components/ui';
import { useHashParameters } from '../lib/hashRoute';
import { isInferenceBatch, isInferenceCohort, isInferenceRun, isUnitSummary, percent, predictedMix, unanimousShare } from '../lib/inference';
import { predictorConfigurationLabel, experimentPredictorLink } from '../lib/predictorGroups';
import { shortRecordId } from '../lib/recordLabels';
import { versionLabelText } from '../lib/versionLabels';
import './ModelChains.css';
import '../components/RunWorkspace.css';
import '../components/InferenceResults.css';

export default function LocalInference({ workspace }: { workspace: Workspace }) {
  const parameters = useHashParameters();
  const linked = { experiment: parameters.get('experiment') ?? '', predictor: parameters.get('predictor') ?? '', cohort: parameters.get('cohort') ?? '', evaluation: parameters.get('evaluation') ?? '', batch: parameters.get('batch') ?? '' };
  return <InferenceWorkspace key={`${workspace.project.id}:${JSON.stringify(linked)}`} workspace={workspace} linked={linked} />;
}

/** One-line prediction summary for tables, from the frozen worker summary. */
export function runSummary(record: ModelEvaluation) {
  const summary = record.execution?.result?.summary;
  const selected = summary && isUnitSummary(summary.selected) ? summary.selected : null;
  return { unit: summary?.unit ?? null, selected, count: typeof record.execution?.result?.slideCount === 'number' ? record.execution.result.slideCount as number : null };
}

function InferenceWorkspace({ workspace, linked }: { workspace: Workspace; linked: { experiment: string; predictor: string; cohort: string; evaluation: string; batch?: string } }) {
  const project = workspace.project.id;
  // Query keys are shared with Evaluate models so both pages stay consistent.
  const batches = useQuery({ queryKey: ['evaluation-batches', project], queryFn: () => bulkEvaluations.list(project, true), refetchIntervalInBackground: false, refetchInterval: (query) => bulkEvaluationPollInterval(query.state.data) });
  const running = Boolean(batches.data?.items.some(bulkEvaluationActive));
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), refetchIntervalInBackground: false, refetchInterval: () => running ? 30000 : 120000 });
  const experimentRegistry = useQuery({ queryKey: ['model-experiment-summaries', project], queryFn: () => experiments.summaries(project), refetchIntervalInBackground: false, refetchInterval: (query) => experimentPollInterval(query.state.data) });
  const cohorts = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project) });
  const records = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const [view, setView] = useState<'library' | 'setup' | 'detail' | 'batch'>(linked.evaluation ? 'detail' : linked.batch ? 'batch' : linked.cohort || linked.predictor || linked.experiment ? 'setup' : 'library');
  const [setupKey, setSetupKey] = useState(0);
  const [selectedRecord, setSelectedRecord] = useState(linked.evaluation);
  const [batchId, setBatchId] = useState(linked.batch ?? '');
  const [tab, setTab] = useState<'runs' | 'batches'>('runs');
  const [search, setSearch] = useState('');
  const [state, setState] = useState('active');
  const [cohortFilter, setCohortFilter] = useState('');
  const [chosenExperimentIds, setExperimentIds] = useState<string[] | null>(() => linked.experiment ? [linked.experiment] : null);
  const linkedPredictor = registry.data?.items.find((item) => item.id === linked.predictor);
  const experimentIds = chosenExperimentIds ?? (linkedPredictor ? [linkedPredictor.manifest.experimentId] : []);
  const [locked, setLocked] = useState(false);
  const inferenceCohorts = (cohorts.data?.items ?? []).filter(isInferenceCohort);
  const cohortIds = new Set(inferenceCohorts.map((item) => item.id));
  const runs = (records.data?.items ?? []).filter(isInferenceRun);
  const runIds = new Set(runs.map((item) => item.id));
  const batchRows = (batches.data?.items ?? []).filter((item) => isInferenceBatch(item, runIds, cohortIds));
  const detail = runs.find((item) => item.id === selectedRecord);
  const predictorById = new Map((registry.data?.items ?? []).map((item) => [item.id, item.manifest]));
  const cohortName = (id: string) => { const item = cohorts.data?.items.find((cohort) => cohort.id === id); return item ? versionLabelText(item, 'Inference cohort') : shortRecordId(id); };
  const visible = runs.filter((item) => (state === 'all' || item.lifecycleState === state) && (!cohortFilter || item.manifest.cohortId === cohortFilter)
    && `${item.manifest.name} ${item.id} ${cohortName(item.manifest.cohortId)} ${predictorById.get(item.manifest.predictorId)?.name ?? ''}`.toLowerCase().includes(search.trim().toLowerCase()))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt) || a.id.localeCompare(b.id));
  function openRecord(id: string) { if (locked) return; setSelectedRecord(id); setView('detail'); void records.refetch(); }
  function openLibrary() { if (!locked) setView('library'); }
  useStageLibrary(openLibrary);
  function create() { if (locked) return; setSelectedRecord(''); setSetupKey((key) => key + 1); setView('setup'); }
  const title = view === 'library' ? 'Run inference' : view === 'detail' ? detail?.manifest.name ?? 'Inference results' : view === 'batch' ? 'Inference batch' : 'Run inference';
  return <div className="clinical-workspace model-chains inference-workspace">
    <PageHeader eyebrow="05 EVALUATE · INFERENCE" title={title}
      description={view === 'library' ? 'Predict unlabeled slides with ready predictors. Results show what each model predicts, how confident it is and where it attends: no labels and no performance metrics.' : view === 'detail' ? 'Review saved predictions, compare model outputs and inspect attention on the original slides.' : view === 'batch' ? 'Monitor prediction jobs and open each predictor’s results.' : 'Select development models and an unlabeled inference cohort, review compatibility, then run predictions.'}
      actions={<div className="inline-actions">{view === 'library' ? <StageCreateButton onClick={create}>Run inference</StageCreateButton> : <StageBackButton disabled={locked} onClick={openLibrary}>Back to inference runs</StageBackButton>}<a className="btn btn-secondary" href="#test-data?purpose=inference">Create inference cohort</a></div>} />
    {view !== 'library' ? <EvidenceChain current="inference" experimentId={detail?.manifest.experimentId ?? (experimentIds.length === 1 ? experimentIds[0] : undefined)} predictorId={detail?.manifest.predictorId ?? (linked.predictor || undefined)} evaluationId={selectedRecord || undefined} /> : null}
    <ErrorNotice error={records.error ?? registry.error ?? cohorts.error ?? experimentRegistry.error ?? batches.error} />
    <StagePage pageKey={`${view}:${selectedRecord || batchId || setupKey}`}>
    {view === 'setup' ? <Panel title="Predict unlabeled slides" subtitle="Choose experiments and methods, then one inference cohort. Each ready ensemble or refit predictor keeps its own predictions.">
      <p className="callout">Inference reads no labels. Review checks each predictor&rsquo;s features and development overlap: development slides are never predicted, and patients seen in development are allowed only for slide-level predictors and flagged in every result.{inferenceCohorts.length ? '' : ' Start by creating an inference cohort from your unlabeled slides.'}</p>
      {linked.predictor && !registry.isPending && !linkedPredictor ? <p className="callout">The linked predictor is unavailable. Choose an experiment and a ready predictor below.</p> : null}
      <BulkEvaluationRunner key={setupKey} mode="inference" project={project} predictors={registry.data?.items ?? []} experiments={experimentRegistry.data?.items ?? []} experimentsLoading={experimentRegistry.isPending} cohorts={inferenceCohorts} linkedCohort={linked.cohort} linkedPredictor={linked.predictor} experimentIds={experimentIds} onExperimentsChange={setExperimentIds} onLockChange={setLocked} onOpenEvaluation={openRecord} />
    </Panel> : null}
    {view === 'library' ? <StageLibrary project={project} title="Inference runs">
      {batchRows.length || tab === 'batches' ? <div className="stage-library-tabs" role="group" aria-label="Inference records"><button type="button" className={tab === 'runs' ? 'selected' : ''} aria-pressed={tab === 'runs'} onClick={() => setTab('runs')}>Runs <span>{runs.length}</span></button><button type="button" className={tab === 'batches' ? 'selected' : ''} aria-pressed={tab === 'batches'} onClick={() => setTab('batches')}>Batches <span>{batchRows.length}</span></button></div> : null}
      {tab === 'runs' ? <>
        <StageLibraryToolbar search={search} onSearch={setSearch} searchLabel="Search inference runs" placeholder="Name, ID, predictor or cohort" count={records.isPending ? undefined : visible.length} total={runs.length}
          actions={<button type="button" className="btn btn-secondary btn-small" disabled={records.isFetching} onClick={() => void Promise.all([records.refetch(), batches.refetch(), cohorts.refetch(), registry.refetch()])}>Refresh</button>}
          onReset={search || state !== 'active' || cohortFilter ? () => { setSearch(''); setState('active'); setCohortFilter(''); } : undefined}>
          <label className="label">State<select className="field" aria-label="Inference run visibility" value={state} onChange={(event) => setState(event.target.value)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label>
          <label className="label">Inference cohort<select className="field" aria-label="Inference cohort filter" value={cohortFilter} onChange={(event) => setCohortFilter(event.target.value)}><option value="">All cohorts</option>{[...new Set(runs.map((item) => item.manifest.cohortId))].map((id) => <option key={id} value={id}>{cohortName(id)}</option>)}</select></label>
        </StageLibraryToolbar>
        {records.isPending ? <p role="status">Loading inference runs…</p> : visible.length ? <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-name">Run</th><th>Predictor</th><th>Inference cohort</th><th>Status</th><th className="run-number">Cases</th><th>Predicted classes</th><th className="run-number">Unanimous</th><th>Actions</th></tr></thead><tbody>{visible.map((item) => {
          const source = predictorById.get(item.manifest.predictorId);
          const { selected, unit, count } = runSummary(item);
          return <tr key={item.id}><td><button type="button" className="text-button stage-record-name" onClick={() => openRecord(item.id)}>{item.manifest.name}</button>{item.manifest.purpose === 'review' ? <small>Earlier slide-review run</small> : null}</td>
            <td>{source ? <><a href={experimentPredictorLink(source.experimentId, item.manifest.predictorId)}>{source.experiment?.name ?? shortRecordId(source.experimentId)}</a><small>{predictorConfigurationLabel(source)} · {predictorMethodLabel(source.method)}</small></> : shortRecordId(item.manifest.predictorId)}</td>
            <td>{cohortName(item.manifest.cohortId)}</td>
            <td><Badge>{computeStatusLabel(item.execution)}</Badge>{item.lifecycleState !== 'active' ? <small>{lifecycleLabel[item.lifecycleState]}</small> : null}</td>
            <td>{selected ? `${selected.count.toLocaleString()} ${unit}s` : count !== null ? `${count.toLocaleString()} slides` : '—'}</td>
            <td className="inference-library-mix">{predictedMix(selected)}</td>
            <td>{percent(unanimousShare(selected), 0)}</td>
            <td><StageRecordManageButton type="configuration" id={item.id} name={item.manifest.name} /></td></tr>;
        })}</tbody></table></div> : <EmptyState icon="inference" title={runs.length ? 'No matching inference runs' : 'No inference runs yet'}
          description={runs.length ? 'Try another search or clear the filters.' : inferenceCohorts.length ? 'Run ready predictors on an inference cohort to get predictions without labels.' : 'Create an inference cohort from your unlabeled slides, then run predictors on it.'}
          action={runs.length ? undefined : inferenceCohorts.length ? <StageCreateButton onClick={create}>Run inference</StageCreateButton> : <a className="btn btn-primary" href="#test-data?purpose=inference">Create inference cohort</a>} />}
      </> : <>{batchRows.length ? <div className="table-wrap"><table className="chain-table"><thead><tr><th scope="col">Batch</th><th scope="col">Status</th><th scope="col">Inference cohort</th><th scope="col">Predictors</th><th scope="col">Actions</th></tr></thead><tbody>{batchRows.map((batch) => <tr key={batch.id}><th scope="row"><button type="button" className="text-button stage-record-name" onClick={() => { setBatchId(batch.id); setView('batch'); }}>{batch.name ?? shortRecordId(batch.id)}</button>{batch.lifecycleState === 'archived' || batch.lifecycleState === 'trashed' ? <small>{batch.lifecycleState === 'trashed' ? 'Trash' : 'Archived'}</small> : null}</th><td>{batch.status.replaceAll('_', ' ')}</td><td>{cohortName(batch.cohortId)}</td><td>{batch.items.length}</td><td><StageRecordManageButton type="configuration" id={batch.id} name={batch.name ?? batch.id} /></td></tr>)}</tbody></table></div> : <EmptyState title="No inference batches yet" description="Batches appear when several predictors run on one inference cohort." />}</>}
    </StageLibrary> : null}
    {view === 'detail' ? detail ? <InferenceRunDetail key={detail.id} project={project} record={detail} runs={runs} />
      : <Panel title="Inference results"><p role="status">{records.isPending ? 'Loading inference run…' : records.isError ? 'The run could not be loaded. Retry to check this record.' : (records.data?.items ?? []).some((item) => item.id === selectedRecord) ? <>This record is a labeled evaluation. <a href={`#evaluation?evaluation=${encodeURIComponent(selectedRecord)}`}>Open it in Evaluate models</a>.</> : 'This inference run is unavailable. Return to the library to choose a saved run.'}</p>{records.isError ? <button className="btn btn-secondary" onClick={() => void records.refetch()}>Retry loading run</button> : null}</Panel> : null}
    {view === 'batch' ? <Panel title="Inference batch"><EvaluationBatchStatus project={project} id={batchId} onOpen={openRecord} kind="inference" /></Panel> : null}
    </StagePage>
  </div>;
}
