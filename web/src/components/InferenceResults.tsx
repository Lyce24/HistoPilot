import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { inferenceRuns, type InferenceSummary } from '../api/inference';
import { computeActive, modelEvaluations, type ModelEvaluation } from '../api/predictors';
import type { CaseQuery } from '../api/caseReview';
import { downloadJSON } from '../lib/download';
import { comparableRuns, countBelow, decimal, percent } from '../lib/inference';
import ComputeJobControls, { computeExecutionQuery } from './ComputeJobControls';
import CaseReviewWorkspace from './CaseReviewWorkspace';
import { evidenceLink } from './EvidenceChain';
import { AgreementBars, ComparisonMatrix, CompositionTable, Histogram, PredictedClassBars, ThresholdTable } from './InferenceCharts';
import { StageRecordManageButton } from './StageWorkflow';
import { ErrorNotice, Panel } from './ui';
import './InferenceResults.css';

const MARGIN_CUTOFFS = [0.1, 0.2, 0.3, 0.4];
const MARGIN_NOTE = 'Distance from the frozen decision boundary: top class minus runner-up, or for binary targets the distance from the threshold relative to the room on that side. Values near zero are borderline.';

/** Everything shown here is computed from verified predictions; no label is ever read. */
export function InferenceOverview({ summary, onReview, memberNote }: { summary: InferenceSummary; onReview?: (query: Partial<CaseQuery>) => void; memberNote?: string }) {
  const [cutoff, setCutoff] = useState(0.2);
  const unit = summary.unit;
  const low = countBelow(summary.margin, cutoff);
  const development = summary.development;
  return <div className="inference-results">
    <div className="inference-tiles" aria-label="Prediction summary">
      <div><span>{unit === 'patient' ? 'Patients' : 'Slides'} predicted</span><strong>{summary.count.toLocaleString()}</strong><small>{unit === 'slide' && summary.patients ? `${summary.patients.toLocaleString()} patient groups` : 'Frozen cohort membership'}</small></div>
      <div><span>Median confidence</span><strong>{percent(summary.confidence.quantiles.median)}</strong><small>IQR {percent(summary.confidence.quantiles.p25, 0)}–{percent(summary.confidence.quantiles.p75, 0)}</small></div>
      <div><span>Margin below <select className="inference-inline-select" aria-label="Decision margin cutoff" value={cutoff} onChange={(event) => setCutoff(Number(event.target.value))}>{MARGIN_CUTOFFS.map((value) => <option key={value} value={value}>{value}</option>)}</select></span><strong>{low.toLocaleString()}</strong><small>{percent(summary.count ? low / summary.count : null)} of {unit}s near a decision boundary{onReview && low ? <> · <button type="button" className="text-button" onClick={() => onReview({ sort: 'margin_asc', maxMargin: cutoff })}>review</button></> : null}</small></div>
      {summary.ensemble ? <div><span>Fold members disagree</span><strong>{summary.ensemble.disagreements.toLocaleString()}</strong><small>{percent(summary.ensemble.records ? summary.ensemble.disagreements / summary.ensemble.records : null)} · {summary.ensemble.memberCount} members{onReview && summary.ensemble.disagreements ? <> · <button type="button" className="text-button" onClick={() => onReview({ sort: 'agreement_asc', memberDisagreement: true })}>review</button></> : null}</small></div>
        : <div><span>Fold member agreement</span><strong>—</strong><small>{memberNote ?? 'Member probabilities were not recorded for this run'}</small></div>}
      {development.comparable ? <div><span>From development patients</span><strong>{development.records.toLocaleString()}</strong><small>{development.patients.toLocaleString()} shared patients · new slides only{development.unknown ? ` · ${development.unknown.toLocaleString()} unverifiable` : ''}{onReview && development.records ? <> · <button type="button" className="text-button" onClick={() => onReview({ developmentPatients: 'shared' })}>review</button></> : null}</small></div> : null}
    </div>
    <div className="inference-grid-2">
      <PredictedClassBars predicted={summary.predicted} unit={unit} />
      {summary.ensemble ? <AgreementBars ensemble={summary.ensemble} unit={unit} /> : <figure className="inference-figure"><figcaption>Decision margin</figcaption><p className="muted">{MARGIN_NOTE} Median {decimal(summary.margin.quantiles.median, 2)}; {percent(summary.count ? countBelow(summary.margin, 0.2) / summary.count : null)} of {unit}s have a margin below 0.2.</p></figure>}
    </div>
    <div className="inference-grid-2">
      <Histogram title="Confidence by predicted class" description={`Probability of the predicted class for each ${unit}. Low values are uncertain predictions worth reviewing first.`} edges={summary.confidence.edges}
        series={summary.classOrder.map((label) => ({ label, counts: summary.confidence.counts[label] ?? [] }))} xLabel="Predicted-class probability" />
      {summary.binary ? <Histogram title={`Probability of ${summary.binary.positiveClass}`} description="Positive-class probability for every case, with the frozen decision threshold." edges={summary.binary.positiveProbability.edges}
        series={[{ label: summary.binary.positiveClass, counts: summary.binary.positiveProbability.counts }]} xLabel={`P(${summary.binary.positiveClass})`}
        colorOffset={summary.classOrder.indexOf(summary.binary.positiveClass)} marker={{ value: summary.binary.threshold, label: `threshold ${summary.binary.threshold}` }} />
        : <Histogram title="Decision margin by predicted class" description={MARGIN_NOTE} edges={summary.margin.edges}
          series={summary.classOrder.map((label) => ({ label, counts: summary.margin.counts[label] ?? [] }))} xLabel="Decision margin" />}
    </div>
    {summary.binary ? <ThresholdTable binary={summary.binary} total={summary.count} /> : null}
    {development.comparable && development.records ? <CompositionTable caption="Predictions for new patients and patients seen in development" groupLabel="Patients" unit={unit} classes={summary.classOrder}
      rows={[{ value: 'New patients', counts: development.new, count: Object.values(development.new).reduce((sum, value) => sum + value, 0) }, { value: 'Seen in development', counts: development.shared, count: development.records }]} /> : null}
  </div>;
}

export function InferenceRunDetail({ project, record, runs }: { project: string; record: ModelEvaluation; runs: ModelEvaluation[] }) {
  const client = useQueryClient();
  const trashed = record.lifecycleState === 'trashed';
  const shouldPoll = !trashed || computeActive(record.execution);
  // Shared with the controls below; read on mount, since the list may predate a change made in the Task Center.
  const execution = useQuery(computeExecutionQuery(project, 'evaluation', record.id, record.execution, shouldPoll));
  const current = shouldPoll ? execution.data : record.execution;
  const completed = !execution.isError && current?.status === 'completed';
  const [unit, setUnit] = useState<'selected' | 'slide' | 'patient'>('selected');
  const [attribute, setAttribute] = useState('');
  const [comparisonId, setComparisonId] = useState('');
  const [review, setReview] = useState<{ key: number; query: Partial<CaseQuery> }>({ key: 0, query: {} });
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const summary = useQuery({ queryKey: ['inference-summary', project, record.id, unit, attribute], queryFn: ({ signal }) => inferenceRuns.summary(project, record.id, { unit, attribute }, signal), enabled: completed && !trashed, staleTime: 60000 });
  // A failed comparison must never replace the run's own summary.
  const comparison = useQuery({ queryKey: ['inference-summary', project, record.id, unit, '', comparisonId], queryFn: ({ signal }) => inferenceRuns.summary(project, record.id, { unit, comparisonId }, signal), enabled: completed && !trashed && Boolean(comparisonId), staleTime: 60000 });
  const comparable = comparableRuns(record, runs);
  const memberEvidence = current?.result?.summary?.memberProbabilities;
  const memberNote = memberEvidence === 'omitted_for_size' ? 'Member probabilities were omitted for this large cohort to keep predictions within size limits' : memberEvidence === 'single_model' ? 'Single model: a refit has no fold members' : undefined;
  const overlap = record.manifest.overlap;
  const data = summary.isError ? undefined : summary.data;
  const attributes = data?.attributes ?? client.getQueryData<InferenceSummary>(['inference-summary', project, record.id, 'selected', ''])?.attributes ?? [];
  const patientSummary = current?.result?.summary?.patient;
  const patientUnavailable = patientSummary && 'available' in patientSummary && patientSummary.available === false ? patientSummary.reason : null;
  function openReview(query: Partial<CaseQuery>) { setReview((currentReview) => ({ key: currentReview.key + 1, query: { unit, ...query } })); window.setTimeout(() => document.getElementById(`review-${record.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0); }
  async function download(action: () => Promise<void>) {
    if (downloading) return;
    setError(null); setDownloading(true);
    try { await action(); } catch (reason) { setError(reason instanceof Error ? reason : new Error('Download failed.')); } finally { setDownloading(false); }
  }
  return <Panel title={record.manifest.name} subtitle="Predictions for an unlabeled inference cohort">
    <ComputeJobControls project={project} id={record.id} kind="evaluation" inference initial={record.execution} readOnly={record.lifecycleState !== 'active'} />
    <p className="callout inference-note">Predictions only: no labels are read and no performance metrics are computed. Agreement and confidence describe the model, not its accuracy.{overlap?.patientsComparable && overlap.patientIds.length ? ` ${overlap.patientIds.length.toLocaleString()} patients in this cohort also contributed development slides; their new slides are flagged throughout.` : ''}{overlap && !overlap.patientsComparable ? ' Patient identifiers use a separate naming system, so shared patients cannot be flagged.' : ''}</p>
    <ErrorNotice error={error ?? summary.error} />
    {trashed ? <p className="callout">These retained predictions belong to a run in Trash. Restore it using <StageRecordManageButton type="configuration" id={record.id} name={record.manifest.name} /> to analyze or download them.</p> : null}
    {completed && !trashed ? <>
      <div className="inference-controls" role="group" aria-label="Analysis scope">
        <label className="label">Prediction unit<select className="field" value={unit} onChange={(event) => setUnit(event.target.value as typeof unit)}><option value="selected">Target unit ({String(current?.result?.summary?.unit ?? 'configured')})</option><option value="slide">Slide</option><option value="patient" disabled={Boolean(patientUnavailable)}>Patient{patientUnavailable ? ' · unavailable' : ''}</option></select></label>
        <label className="label">Break down by<select className="field" value={attribute} onChange={(event) => setAttribute(event.target.value)}><option value="">No attribute</option>{attributes.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
        <label className="label">Compare with<select className="field" value={comparisonId} onChange={(event) => setComparisonId(event.target.value)}><option value="">No other run</option>{comparable.map((item) => <option key={item.id} value={item.id}>{item.manifest.name}</option>)}</select></label>
      </div>
      {patientUnavailable ? <p className="muted">Patient predictions unavailable: {patientUnavailable}</p> : null}
      {summary.isError ? <button type="button" className="btn btn-secondary" onClick={() => void summary.refetch()}>Retry prediction summary</button> : null}
      {summary.isPending ? <p role="status">Verifying saved predictions and summarizing…</p> : null}
      {data ? <div className="inference-results" style={{ opacity: summary.isFetching ? 0.6 : 1 }}>
        <InferenceOverview summary={data} onReview={openReview} memberNote={memberNote} />
        {data.breakdown ? <CompositionTable caption={`Predicted class by ${data.breakdown.label}`} groupLabel={data.breakdown.label} unit={data.unit} classes={data.classOrder}
          rows={[...data.breakdown.rows, ...(data.breakdown.other ? [{ value: `Other (${data.breakdown.otherValues} values)`, ...data.breakdown.other }] : [])]} /> : null}
        {comparisonId ? <ErrorNotice error={comparison.error} /> : null}
        {comparisonId && comparison.isPending ? <p role="status">Pairing predictions with the other run…</p> : null}
        {comparisonId && !comparison.isError && comparison.data?.comparison ? <><ComparisonMatrix comparison={comparison.data.comparison} classes={comparison.data.classOrder} name={record.manifest.name} />{comparison.data.comparison.disagreements ? <p><button type="button" className="text-button" onClick={() => openReview({ comparisonId, outcome: 'disagreement' })}>Review the {comparison.data.comparison.disagreements.toLocaleString()} disagreements</button></p> : null}</> : null}
        <p className="muted">Predictions SHA-256 {data.source.predictionsSha256.slice(0, 16)}… · frozen {data.positiveClass ? `${data.positiveClass} threshold ${data.decisionThreshold}` : 'highest-probability decision'} · patient aggregation {data.patientAggregation.replace('_', ' ')}.</p>
      </div> : null}
      <div className="inference-section" id={`review-${record.id}`}>
        <CaseReviewWorkspace key={review.key} project={project} evaluation={record} comparisons={comparable} initial={review.query} />
      </div>
      <div className="inference-section"><h3>Downloads</h3>
        <div className="inference-downloads">
          <button type="button" className="btn btn-primary" disabled={downloading || !data || summary.isFetching} onClick={() => void download(() => inferenceRuns.export(project, record.id, unit))}>Download predictions with metadata (CSV)</button>
          <button type="button" className="btn btn-secondary" disabled={downloading} onClick={() => void download(() => modelEvaluations.download(project, record.id, 'slide-predictions.csv'))}>Raw slide predictions</button>
          <button type="button" className="btn btn-secondary" disabled={downloading || Boolean(patientUnavailable)} title={patientUnavailable ?? undefined} onClick={() => void download(() => modelEvaluations.download(project, record.id, 'patient-predictions.csv'))}>Raw patient predictions</button>
          <button type="button" className="btn btn-secondary" disabled={downloading} onClick={() => void download(() => modelEvaluations.download(project, record.id, 'predictions.json'))}>Checksummed predictions (JSON)</button>
        </div>
        <p className="muted">The CSV has one row per {(data?.unit ?? (unit === 'selected' ? current?.result?.summary?.unit : unit)) === 'patient' ? 'patient' : 'slide'} with the predicted class, per-class probabilities, confidence, margin, fold-member agreement, the development-patient flag, frozen dataset attributes and the predictions checksum.</p>
      </div>
      <div className="inline-actions"><a className="btn btn-secondary" href={evidenceLink('interpretation', { experimentId: record.manifest.experimentId, predictorId: record.manifest.predictorId, evaluationId: record.id })}>Open in Model interpretation</a></div>
    </> : !completed ? <p>{execution.isError ? 'Results cannot be verified. Resolve the execution error before viewing predictions.' : 'No completed predictions yet. Run this inference job to predict every slide in its cohort.'}</p> : null}
    <div className="inline-actions"><button className="btn btn-secondary" type="button" onClick={() => downloadJSON(`${record.manifest.name}.json`, { ...record, execution: current })}>Export run record</button></div>
    <details><summary>Saved inputs, settings and results</summary><pre className="chain-details">{JSON.stringify({ manifest: record.manifest, execution: current }, null, 2)}</pre></details>
  </Panel>;
}
