import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { CaseQuery } from '../api/caseReview';
import type { EvaluationCohort } from '../api/evaluation';
import type { ModelExperimentSummary } from '../api/experiments';
import { inferenceRuns, type InferenceSummary } from '../api/inference';
import { runPerformance, type GroupMetrics } from '../api/performance';
import { computeActive, hasPatientPredictions, modelEvaluations, type ComputeExecution, type EvaluationMetrics, type FrozenPredictor, type ModelEvaluation } from '../api/predictors';
import { referenceFits, references, type ReferenceArtifact, type ReferenceStandard } from '../api/references';
import { formatStatistic } from '../api/statistics';
import { batchNameLookup, cohortKindText, cohortName, modelDescription, runLabeled } from '../lib/applyModels';
import type { RunTab } from '../lib/applyRoutes';
import { downloadJSON } from '../lib/download';
import { comparableRuns } from '../lib/inference';
import { experimentPredictorLink } from '../lib/predictorGroups';
import { shortRecordId } from '../lib/recordLabels';
import CaseReviewWorkspace from './CaseReviewWorkspace';
import { RunClinicalUtility } from './ClinicalUtility';
import ComputeJobControls, { computeExecutionQuery } from './ComputeJobControls';
import { evidenceLink } from './EvidenceChain';
import { ComparisonMatrix, CompositionTable } from './InferenceCharts';
import { InferenceOverview } from './InferenceResults';
import PairedPatientComparison from './PairedPatientComparison';
import PatientAnalysisResults from './PatientAnalysisResults';
import ReferenceStandardEditor from './ReferenceStandardEditor';
import RunAgreement from './RunAgreement';
import RunRecalibration from './RunRecalibration';
import { StageRecordManageButton, StageViews } from './StageWorkflow';
import { ErrorNotice } from './ui';
import './InferenceResults.css';
import './ApplyRunDetail.css';

type Unit = 'selected' | 'slide' | 'patient';
type Review = (query: Partial<CaseQuery>) => void;
const TAB_TITLES: Record<RunTab, string> = { performance: 'Performance', agreement: 'Agreement', predictions: 'Predictions', cases: 'Cases', compare: 'Compare' };
const METRIC_LABELS = { accuracy: 'Accuracy', balancedAccuracy: 'Balanced accuracy', macroF1: 'Macro F1', auroc: 'AUROC', auprc: 'AUPRC', loss: 'Log loss' } as const;
// Groups this small are listed, but their estimates are too unstable to read as a difference.
const SMALL_GROUP = 10;

/** A run opens on what it is for: performance when it has labels to score against, predictions otherwise. */
export function runTabs(scored: boolean): RunTab[] {
  return scored ? ['performance', 'agreement', 'predictions', 'cases', 'compare'] : ['predictions', 'cases', 'compare'];
}

/** Labels a run is scored against: its cohort's own (`id` null) or a reference standard. */
export interface LabelSource { id: string | null; name: string; standard?: ReferenceStandard }

/**
 * What a run can be scored against: its cohort's frozen labels when it has any, then each
 * reference standard of the cohort with the run's classes, by name. An archived reference is
 * offered only when the link names it; it stays valid provenance.
 */
export function labelSources(record: ModelEvaluation, standards: readonly ReferenceStandard[], linked?: string): LabelSource[] {
  const classes = record.manifest.target?.classes ?? [];
  const fitting = standards
    .filter((item) => referenceFits(item, record.manifest.cohortId, classes) && ((item.lifecycleState ?? 'active') === 'active' || item.id === linked))
    .sort((a, b) => a.manifest.name < b.manifest.name ? -1 : a.manifest.name > b.manifest.name ? 1 : a.id < b.id ? -1 : 1);
  return [
    ...runLabeled(record) ? [{ id: null, name: 'Cohort labels' }] : [],
    ...fitting.map((item) => ({ id: item.id, name: item.manifest.name, standard: item })),
  ];
}

/** The labels a run is scored against: the named reference when it fits, else the first source. */
export const labelSource = (sources: LabelSource[], reference?: string | null) => sources.find((item) => item.id === (reference ?? null)) ?? sources[0];

/**
 * One predictor applied to one cohort. Every run has its predictions, cases and comparisons.
 * A run with labels to score against, its labeled cohort's or a reference standard added
 * later, adds performance (with subgroups and clinical utility) and agreement with every
 * label source. Nothing on these views changes the saved predictions.
 */
export default function ApplyRunDetail({ project, record, runs, predictors, experiments, cohorts, tab, onTabChange, reference, onReferenceChange, initialAnalysis }: {
  project: string; record: ModelEvaluation; runs: ModelEvaluation[]; predictors: FrozenPredictor[];
  experiments: ModelExperimentSummary[]; cohorts: EvaluationCohort[];
  tab?: RunTab; onTabChange: (tab: RunTab) => void;
  /** The reference standard to score against; absent means the run's default labels. */
  reference?: string; onReferenceChange: (reference: string | undefined, tab?: RunTab) => void;
  initialAnalysis?: string;
}) {
  const trashed = record.lifecycleState === 'trashed';
  const shouldPoll = !trashed || computeActive(record.execution);
  // Shared with the job controls; read on mount, since the list may predate a Task Center change.
  const execution = useQuery(computeExecutionQuery(project, 'evaluation', record.id, record.execution, shouldPoll));
  const current = shouldPoll ? execution.data : record.execution;
  const completed = !execution.isError && current?.status === 'completed';
  const labeled = runLabeled(record);
  const cohort = cohorts.find((item) => item.id === record.manifest.cohortId);
  const standards = useQuery({ queryKey: ['reference-standards', project, record.manifest.cohortId], queryFn: () => references.list(project, record.manifest.cohortId), staleTime: 60000 });
  const sources = labelSources(record, standards.data?.items ?? [], reference);
  // A link naming a reference that does not fit falls back to the run's default labels.
  const source = labelSource(sources, reference);
  const unavailableReference = Boolean(reference && standards.isSuccess && source?.id !== reference);
  // Only a run without labels of its own, or a link naming a reference, waits for the references.
  const waiting = standards.isPending && (!labeled || Boolean(reference));
  const scored = Boolean(source);
  const tabs = runTabs(scored);
  const shown = tab && tabs.includes(tab) ? tab : tabs[0];
  const [review, setReview] = useState<{ key: number; query: Partial<CaseQuery> }>({ key: 0, query: {} });
  const [adding, setAdding] = useState(false);
  const openCases: Review = (query) => { setReview((value) => ({ key: value.key + 1, query })); onTabChange('cases'); };
  const predictor = predictors.find((item) => item.id === record.manifest.predictorId)?.manifest;
  const batchName = batchNameLookup(experiments);
  const classes = record.manifest.target?.classes ?? [];
  const canAdd = Boolean(cohort && classes.length >= 2 && record.lifecycleState === 'active');
  return <div className="apply-run">
    <dl className="run-context">
      <div><dt>Model</dt><dd>{predictor ? <><a href={experimentPredictorLink(record.manifest.experimentId, record.manifest.predictorId)}>{predictor.experiment?.name ?? shortRecordId(record.manifest.experimentId)}</a><small>{modelDescription(predictor, batchName)}</small></> : 'Predictor unavailable'}</dd></div>
      <div><dt>Cohort</dt><dd>{cohortName(cohort, record.manifest.cohortId)}<small>{cohort ? `${cohortKindText(cohort, sources.filter((item) => item.standard).length)} · ${cohort.manifest.summary.includedSlides.toLocaleString()} slides` : labeled ? 'Labeled · scored' : 'Unlabeled · predictions only'}</small></dd></div>
      <div className="run-labels"><dt id={`run-labels-${record.id}`}>Scored against</dt><dd>
        {waiting ? <span role="status">Loading label sources…</span>
          : sources.length ? <select className="field" aria-labelledby={`run-labels-${record.id}`} value={source?.id ?? ''} onChange={(event) => onReferenceChange(event.target.value || undefined)}>{sources.map((item) => <option key={item.id ?? 'cohort'} value={item.id ?? ''}>{item.name}{item.standard && item.standard.lifecycleState === 'archived' ? ' (archived)' : ''}</option>)}</select>
          : 'No labels · predictions only'}
        <small>{waiting ? null : source?.standard ? `Reference standard · ${source.standard.manifest.summary.labeledSlides.toLocaleString()} of ${source.standard.manifest.summary.slides.toLocaleString()} slides labeled` : source ? 'The labels frozen with the cohort' : 'Add a reference standard when labels arrive.'}
          {canAdd && !adding ? <> · <button type="button" className="text-button" onClick={() => setAdding(true)}>Add reference standard</button></> : null}</small>
      </dd></div>
    </dl>
    <ErrorNotice error={standards.error} />
    {unavailableReference ? <p className="callout">The linked reference standard does not label this run&rsquo;s cohort with its classes, or is in Trash. {source ? `Showing ${source.name} instead.` : ''}</p> : null}
    {adding && cohort ? <ReferenceStandardEditor project={project} cohort={cohort} classes={classes} onCancel={() => setAdding(false)} onSaved={(saved) => { setAdding(false); onReferenceChange(saved.id, 'performance'); }} /> : null}
    <ComputeJobControls project={project} id={record.id} kind="evaluation" initial={record.execution} readOnly={record.lifecycleState !== 'active'} />
    {trashed ? <p className="callout">This run is in Trash; its saved predictions are retained. Restore it using <StageRecordManageButton type="configuration" id={record.id} name={record.manifest.name} /> to analyze or download them.</p> : null}
    {completed && !trashed && !waiting ? <>
      <StageViews label="Run views" views={tabs.map((id) => ({ id, title: TAB_TITLES[id] }))} current={shown} onChange={onTabChange} idPrefix="run-tab-" />
      <div className="run-view" role="region" aria-labelledby={`run-tab-${shown}`}>
        {shown === 'performance' && source ? <PerformanceView key={source.id ?? 'cohort'} project={project} record={record} current={current} source={source} onReview={openCases} initialAnalysis={initialAnalysis} /> : null}
        {shown === 'agreement' ? <RunAgreement project={project} evaluationId={record.id} patient={patientUnavailable(record, current?.result)} targetUnit={current?.result?.metrics?.unit ?? current?.result?.summary?.unit} onReview={(id, query) => { onReferenceChange(id ?? undefined, 'cases'); setReview((value) => ({ key: value.key + 1, query })); }} /> : null}
        {shown === 'predictions' ? <PredictionsView project={project} record={record} current={current} labeled={scored} onReview={openCases} /> : null}
        {shown === 'cases' ? <CaseReviewWorkspace key={`${review.key}:${source?.id ?? ''}`} project={project} evaluation={record} comparisons={runs} initial={review.query} reference={source?.standard ? { id: source.standard.id, name: source.name } : undefined} /> : null}
        {shown === 'compare' ? <CompareView project={project} record={{ ...record, execution: current ?? undefined }} runs={runs} source={source} onReview={openCases} /> : null}
      </div>
      <div className="inline-actions run-actions">
        <a className="btn btn-secondary" href={evidenceLink('interpretation', { experimentId: record.manifest.experimentId, predictorId: record.manifest.predictorId, evaluationId: record.id })}>Open in Model interpretation</a>
        <button className="btn btn-secondary" type="button" onClick={() => downloadJSON(`${record.manifest.name}.json`, { ...record, execution: current })}>Export run record</button>
      </div>
    </> : !completed ? <p>{execution.isError ? 'Results cannot be verified. Resolve the execution error before viewing them.' : 'No completed predictions yet. Run this job to predict every slide in its cohort.'}</p> : null}
    <details><summary>Saved inputs, settings and results</summary><pre className="chain-details">{JSON.stringify({ manifest: record.manifest, execution: current }, null, 2)}</pre></details>
  </div>;
}

function useDownload() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  async function run(action: () => Promise<void>) {
    if (busy) return;
    setBusy(true); setError(null);
    try { await action(); } catch (reason) { setError(reason instanceof Error ? reason : new Error('Download failed.')); } finally { setBusy(false); }
  }
  return { busy, error, run };
}

/** Why a patient view is unavailable for this run, or null when it has one. */
function patientUnavailable(record: ModelEvaluation, result: ComputeExecution['result']) {
  if (record.manifest.splitUnit === 'slide') return 'Patient analysis is disabled for slide-level experiments.';
  const summary = result?.summary?.patient;
  if (summary && 'available' in summary && summary.available === false) return summary.reason;
  const metrics = result?.metrics?.patient;
  return metrics && metrics.available === false ? metrics.reason ?? 'Patient predictions are unavailable.' : null;
}

function UnitSelect({ unit, onChange, patient, label = 'Prediction unit', targetUnit }: { unit: Unit; onChange: (unit: Unit) => void; patient: string | null; label?: string; targetUnit?: string }) {
  return <label className="label">{label}<select className="field" value={unit} onChange={(event) => onChange(event.target.value as Unit)}>
    <option value="selected">Target unit ({targetUnit ?? 'configured'})</option><option value="slide">Slide</option>
    <option value="patient" disabled={Boolean(patient)}>Patient{patient ? ' · unavailable' : ''}</option>
  </select></label>;
}

function PredictionsView({ project, record, current, labeled, onReview }: { project: string; record: ModelEvaluation; current?: ComputeExecution | null; labeled: boolean; onReview: Review }) {
  const client = useQueryClient();
  const [unit, setUnit] = useState<Unit>('selected');
  const [attribute, setAttribute] = useState('');
  const download = useDownload();
  const summary = useQuery({ queryKey: ['inference-summary', project, record.id, unit, attribute], queryFn: ({ signal }) => inferenceRuns.summary(project, record.id, { unit, attribute }, signal), staleTime: 60000 });
  const data = summary.isError ? undefined : summary.data;
  const attributes = data?.attributes ?? client.getQueryData<InferenceSummary>(['inference-summary', project, record.id, 'selected', ''])?.attributes ?? [];
  const evidence = current?.result?.summary?.memberProbabilities;
  const memberNote = evidence === 'omitted_for_size' ? 'Member probabilities were omitted for this large cohort to keep predictions within size limits'
    : evidence === 'single_model' ? 'Single model: a refit has no fold members' : undefined;
  const patient = patientUnavailable(record, current?.result);
  const overlap = record.manifest.overlap;
  return <div className="run-tab-body">
    <p className="callout inference-note">{labeled ? 'These views describe the predictions without reading any label; Performance scores them.' : 'Predictions only: no labels are read and no performance metrics are computed. Agreement and confidence describe the model, not its accuracy.'}{overlap?.patientsComparable && overlap.patientIds.length ? ` ${overlap.patientIds.length.toLocaleString()} patients in this cohort also contributed development slides; their new slides are flagged throughout${labeled ? ' and left out of every metric' : ''}.` : ''}{overlap && !overlap.patientsComparable ? ' Patient identifiers use a separate naming system, so shared patients cannot be flagged.' : ''}</p>
    <ErrorNotice error={summary.error ?? download.error} />
    <div className="inference-controls" role="group" aria-label="Prediction scope">
      <UnitSelect unit={unit} onChange={setUnit} patient={patient} targetUnit={data?.unit ?? current?.result?.summary?.unit ?? current?.result?.metrics?.unit} />
      <label className="label">Break down by<select className="field" value={attribute} onChange={(event) => setAttribute(event.target.value)}><option value="">No attribute</option>{attributes.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
    </div>
    {summary.isError ? <button type="button" className="btn btn-secondary" onClick={() => void summary.refetch()}>Retry prediction summary</button> : null}
    {summary.isPending ? <p role="status">Verifying saved predictions and summarizing…</p> : null}
    {data ? <div className="inference-results" style={{ opacity: summary.isFetching ? 0.6 : 1 }}>
      <InferenceOverview summary={data} onReview={(query) => onReview({ unit, ...query })} memberNote={memberNote} />
      {data.breakdown ? <CompositionTable caption={`Predicted class by ${data.breakdown.label}`} groupLabel={data.breakdown.label} unit={data.unit} classes={data.classOrder}
        rows={[...data.breakdown.rows, ...(data.breakdown.other ? [{ value: `Other (${data.breakdown.otherValues} values)`, ...data.breakdown.other }] : [])]} /> : null}
      <p className="muted">Predictions SHA-256 {data.source.predictionsSha256.slice(0, 16)}… · frozen {data.positiveClass ? `${data.positiveClass} threshold ${data.decisionThreshold}` : 'highest-probability decision'}{record.manifest.splitUnit === 'slide' ? '' : ` · patient aggregation ${data.patientAggregation.replace('_', ' ')}`}.</p>
    </div> : null}
    <div className="inference-section"><h3>Downloads</h3>
      <div className="inference-downloads">
        <button type="button" className="btn btn-primary" disabled={download.busy || !data || summary.isFetching} onClick={() => void download.run(() => inferenceRuns.export(project, record.id, unit))}>Predictions with metadata (CSV)</button>
        <button type="button" className="btn btn-secondary" disabled={download.busy} onClick={() => void download.run(() => modelEvaluations.download(project, record.id, 'predictions.json'))}>Checksummed predictions (JSON)</button>
      </div>
      <p className="muted">The CSV has one row per {(data?.unit ?? 'slide') === 'patient' ? 'patient' : 'slide'} with the predicted class, per-class probabilities, confidence, margin, fold-member agreement, the development-patient flag, frozen dataset attributes and the predictions checksum. It reads no labels.</p>
    </div>
  </div>;
}

/** A run's scores against a reference standard; the cohort's own come with the run. */
function useReferenceScores(project: string, evaluationId: string, referenceId: string | null) {
  return useQuery({ queryKey: ['reference-scores', project, evaluationId, referenceId], queryFn: ({ signal }) => references.scores(project, evaluationId, referenceId, signal), enabled: Boolean(referenceId), staleTime: 300000 });
}

function PerformanceView({ project, record, current, source, onReview, initialAnalysis }: { project: string; record: ModelEvaluation; current?: ComputeExecution | null; source: LabelSource; onReview: Review; initialAnalysis?: string }) {
  const [unit, setUnit] = useState<Unit>('selected');
  const download = useDownload();
  const result = current?.result;
  const referenceId = source.id;
  const scores = useReferenceScores(project, record.id, referenceId);
  const metrics: EvaluationMetrics | undefined = referenceId ? scores.isError ? undefined : scores.data : result?.metrics;
  const selected = metrics?.[unit];
  const patient = patientUnavailable(record, result);
  const unitName = unit === 'selected' ? metrics?.unit : unit;
  // The frozen data dictionary, from the prediction summary that Predictions shares.
  const summary = useQuery({ queryKey: ['inference-summary', project, record.id, 'selected', ''], queryFn: ({ signal }) => inferenceRuns.summary(project, record.id, { unit: 'selected' }, signal), staleTime: 60000 });
  if (referenceId && scores.isPending) return <div className="run-tab-body"><p role="status">Scoring the saved predictions against {source.name}…</p></div>;
  if (!metrics) return <div className="run-tab-body">
    <ErrorNotice error={scores.error} />
    {scores.isError ? <button type="button" className="btn btn-secondary" onClick={() => void scores.refetch()}>Retry scoring</button>
      : <p className="callout" role="alert">{result?.metricsError ? `The predictions are saved, but they could not be scored: ${result.metricsError.message}` : 'Scores for this run are unavailable.'}</p>}
  </div>;
  // Files scored against a reference carry its name, so they are never mistaken for the cohort's.
  const fetchFile = (file: ReferenceArtifact) => download.run(() => referenceId ? references.download(project, record.id, referenceId, file, `${source.name} ${file}`) : modelEvaluations.download(project, record.id, file));
  return <div className="run-tab-body">
    <ErrorNotice error={download.error} />
    <div className="inference-controls" role="group" aria-label="Scoring unit"><UnitSelect unit={unit} onChange={setUnit} patient={patient} label="Metrics by prediction unit" targetUnit={metrics.unit} /></div>
    {metrics.positiveClass && typeof metrics.decisionThreshold === 'number' ? <p className="muted">Predict {metrics.positiveClass} when its probability is at least {metrics.decisionThreshold}. Ranking metrics use the original probabilities.</p> : null}
    {referenceId ? <p className="muted">Scored against the reference standard {source.name}, labels added to the cohort after it was frozen. The saved predictions are unchanged.</p>
      : metrics.scoredBy ? <p className="muted">Predicted without reading labels, then scored against the cohort&rsquo;s frozen labels.</p> : null}
    {metrics.conflictingPatients ? <p className="muted">{metrics.conflictingPatients.toLocaleString()} {metrics.conflictingPatients === 1 ? 'patient has' : 'patients have'} slides this reference labels differently; patient scores leave {metrics.conflictingPatients === 1 ? 'it' : 'them'} unlabeled.</p> : null}
    {typeof selected?.unlabeledCount === 'number' && selected.unlabeledCount > 0 ? <p className="muted">{selected.unlabeledCount.toLocaleString()} unlabeled {unitName} records have predictions and are not scored.</p> : null}
    {metrics.developmentExcluded ? <p className="callout">{metrics.developmentExcluded.slides.toLocaleString()} {metrics.developmentExcluded.slides === 1 ? 'slide' : 'slides'} from {metrics.developmentExcluded.patients.toLocaleString()} development {metrics.developmentExcluded.patients === 1 ? 'patient was' : 'patients were'} predicted but left out of every metric, because they are not independent of this predictor&rsquo;s development.</p> : null}
    {selected?.available ? <>
      <p>{selected.count?.toLocaleString()} labeled {unitName} records. Classes: {metrics.classOrder.join(', ')}.</p>
      <div className="chain-metrics">{(Object.keys(METRIC_LABELS) as (keyof typeof METRIC_LABELS)[]).map((key) => <div key={key}><strong>{formatStatistic(selected[key], key === 'auroc' || key === 'auprc' ? selected.confidenceIntervals?.[key] : undefined)}</strong><span>{METRIC_LABELS[key]}{key === 'auroc' || key === 'auprc' ? (selected.confidenceIntervals?.[key] ? ' (95% CI)' : '') : ''}</span></div>)}</div>
      {selected.missingClasses?.length ? <p className="callout">Classes absent from the labeled records: {selected.missingClasses.join(', ')}. Some metrics cannot be estimated.</p> : null}
      {selected.confusionMatrix ? <figure className="run-figure"><figcaption>Confusion matrix</figcaption><p className="muted">Rows: actual class. Columns: predicted class. Select a count to review its cases.</p><div className="table-wrap"><table className="chain-table"><thead><tr><th>Actual / predicted</th>{metrics.classOrder.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{selected.confusionMatrix.map((row, index) => <tr key={index}><th scope="row">{metrics.classOrder[index]}</th>{row.map((value, column) => <td key={column}><button type="button" className="text-button" disabled={!value} aria-label={`Review ${value} cases: actual ${metrics.classOrder[index]}, predicted ${metrics.classOrder[column]}`} onClick={() => onReview({ unit, outcome: 'all', actualClass: index, predictedClass: column })}>{value}</button></td>)}</tr>)}</tbody></table></div></figure> : null}
      <div className="inline-actions"><button type="button" className="btn btn-secondary" onClick={() => onReview({ unit, outcome: 'error' })}>Review errors</button></div>
    </> : <p className="callout">{selected?.reason ?? 'No labeled records are available for these metrics.'} Predictions are still available.</p>}
    <PatientAnalysisResults value={metrics.patientAnalysis} />
    <SubgroupPerformance project={project} record={record} unit={unit} referenceId={referenceId} attributes={summary.data?.attributes ?? []} />
    <RunRecalibration project={project} record={record} unit={unit} referenceId={referenceId} />
    <RunClinicalUtility project={project} record={record} metrics={metrics} reference={source.standard ? { id: source.standard.id, name: source.name } : undefined} initialAnalysis={initialAnalysis} />
    <div className="inference-section"><h3>Downloads</h3>
      <div className="inference-downloads">
        <button type="button" className="btn btn-secondary" disabled={download.busy} onClick={() => void fetchFile('slide-predictions.csv')}>Scored slide table (CSV)</button>
        {hasPatientPredictions(result) || (referenceId && !patient) ? <button type="button" className="btn btn-secondary" disabled={download.busy} onClick={() => void fetchFile('patient-predictions.csv')}>Scored patient table (CSV)</button> : null}
        <button type="button" className="btn btn-secondary" disabled={download.busy} onClick={() => void fetchFile('metrics.json')}>Metrics (JSON)</button>
      </div>
      <p className="muted">The tables carry each record&rsquo;s label{referenceId ? ` from ${source.name}` : ''}, prediction and class probabilities, and whether it counts toward the metrics.</p>
    </div>
  </div>;
}

/** Metrics within each value of a frozen attribute, over exactly the records the run scores. */
function SubgroupPerformance({ project, record, unit, referenceId, attributes }: { project: string; record: ModelEvaluation; unit: Unit; referenceId: string | null; attributes: { key: string; label: string }[] }) {
  const [attribute, setAttribute] = useState('');
  const query = useQuery({ queryKey: ['run-performance', project, record.id, unit, attribute, referenceId], queryFn: ({ signal }) => runPerformance.breakdown(project, record.id, unit, attribute, referenceId, signal), enabled: Boolean(attribute), staleTime: 60000 });
  const data = query.isError ? undefined : query.data;
  const row = (value: string, metrics: GroupMetrics) => {
    const labeled = metrics.count ?? 0;
    return <tr key={value} className={labeled < SMALL_GROUP ? 'is-muted' : undefined}><th scope="row">{value}</th><td>{labeled.toLocaleString()}{metrics.unlabeledCount ? <small>+{metrics.unlabeledCount.toLocaleString()} unlabeled</small> : null}</td>
      {metrics.available ? (['auroc', 'auprc', 'balancedAccuracy', 'accuracy'] as const).map((key) => <td key={key}>{formatStatistic(metrics[key])}</td>) : <td colSpan={4}>{metrics.reason ?? 'Unavailable'}</td>}</tr>;
  };
  return <section className="run-section" aria-labelledby={`subgroups-${record.id}`}>
    <div className="run-section-heading"><h3 id={`subgroups-${record.id}`}>Performance by subgroup</h3>
      <label className="label">Attribute<select className="field" value={attribute} onChange={(event) => setAttribute(event.target.value)}><option value="">Choose an attribute</option>{attributes.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label></div>
    <p className="muted">The same metrics within each value of a frozen dataset attribute, such as site or scanner. Descriptive only: small groups are unstable, and comparing many groups finds gaps by chance.</p>
    <ErrorNotice error={query.error} />
    {query.isFetching && !data ? <p role="status">Scoring each group…</p> : null}
    {data ? <div className="table-wrap"><table className="chain-table subgroup-table"><thead><tr><th scope="col">{data.label}</th><th scope="col">Labeled {data.unit}s</th><th scope="col">AUROC</th><th scope="col">AUPRC</th><th scope="col">Balanced accuracy</th><th scope="col">Accuracy</th></tr></thead>
      <tbody>{data.rows.map((item) => row(item.value, item))}{data.other ? row(`Other (${data.otherValues} values)`, data.other) : null}</tbody>
      <tfoot>{row('All scored records', data.overall)}</tfoot></table>
      <p className="muted">Rows with fewer than {SMALL_GROUP} labeled {data.unit}s are dimmed.{data.developmentExcluded ? ` ${data.developmentExcluded.toLocaleString()} records from development patients are left out, as in every metric.` : ''}</p></div> : null}
  </section>;
}

function CompareView({ project, record, runs, source, onReview }: { project: string; record: ModelEvaluation; runs: ModelEvaluation[]; source?: LabelSource; onReview: Review }) {
  const options = comparableRuns(record, runs);
  const [otherId, setOtherId] = useState(options[0]?.id ?? '');
  const [unit, setUnit] = useState<Unit>('selected');
  const other = options.find((item) => item.id === otherId);
  const comparison = useQuery({ queryKey: ['inference-summary', project, record.id, unit, '', otherId], queryFn: ({ signal }) => inferenceRuns.summary(project, record.id, { unit, comparisonId: otherId }, signal), enabled: Boolean(other), staleTime: 60000 });
  const patient = patientUnavailable(record, record.execution?.result);
  // Both runs are scored against the same labels: the cohort's, or the chosen reference.
  const referenceId = source?.id ?? null;
  const mineScores = useReferenceScores(project, record.id, referenceId);
  const theirScores = useReferenceScores(project, other?.id ?? '', other ? referenceId : null);
  const scores = (run: ModelEvaluation | undefined, reference: typeof mineScores) => referenceId ? reference.data?.selected : run && runLabeled(run) ? run.execution?.result?.metrics?.selected : undefined;
  const mine = scores(record, mineScores), theirs = scores(other, theirScores);
  const bothScored = Boolean(source && other && mine?.available && theirs?.available);
  if (!options.length) return <div className="run-tab-body"><p className="muted">No other completed run on this cohort predicts the same target yet. Apply another predictor to this cohort to compare them case by case.</p></div>;
  return <div className="run-tab-body">
    <div className="inference-controls" role="group" aria-label="Comparison">
      <label className="label">Compare with<select className="field" value={other?.id ?? ''} onChange={(event) => setOtherId(event.target.value)}>{options.map((item) => <option key={item.id} value={item.id}>{item.manifest.name}</option>)}</select></label>
      <UnitSelect unit={unit} onChange={setUnit} patient={patient} targetUnit={record.execution?.result?.metrics?.unit ?? record.execution?.result?.summary?.unit} />
    </div>
    {bothScored ? <div className="table-wrap"><table className="chain-table"><caption className="muted">Scored against {source?.name}</caption><thead><tr><th scope="col">Metric (target unit)</th><th scope="col">This run</th><th scope="col">{other?.manifest.name}</th></tr></thead><tbody>{(['auroc', 'auprc', 'balancedAccuracy', 'accuracy'] as const).map((key) => <tr key={key}><th scope="row">{METRIC_LABELS[key]}</th><td>{formatStatistic(mine?.[key], key === 'auroc' || key === 'auprc' ? mine?.confidenceIntervals?.[key] : undefined)}</td><td>{formatStatistic(theirs?.[key], key === 'auroc' || key === 'auprc' ? theirs?.confidenceIntervals?.[key] : undefined)}</td></tr>)}</tbody></table></div> : null}
    <ErrorNotice error={comparison.error} />
    {comparison.isPending && other ? <p role="status">Pairing predictions with the other run…</p> : null}
    {!comparison.isError && comparison.data?.comparison ? <>
      <ComparisonMatrix comparison={comparison.data.comparison} classes={comparison.data.classOrder} name={record.manifest.name} />
      {comparison.data.comparison.disagreements ? <p><button type="button" className="text-button" onClick={() => onReview({ unit, comparisonId: otherId, outcome: 'disagreement' })}>Review the {comparison.data.comparison.disagreements.toLocaleString()} cases where they disagree</button></p> : null}
    </> : null}
    {bothScored && !referenceId && other && record.execution?.result?.metrics?.patient?.available && other.execution?.result?.metrics?.patient?.available ? <PairedPatientComparison project={project} records={[record, other]} /> : null}
  </div>;
}
