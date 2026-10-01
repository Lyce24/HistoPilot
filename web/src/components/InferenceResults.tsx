import { useState } from 'react';
import type { InferenceSummary, InferenceUnitSummary } from '../api/inference';
import type { CaseQuery } from '../api/caseReview';
import { countBelow, decimal, percent } from '../lib/inference';
import { AgreementBars, CompositionTable, Histogram, PredictedClassBars, ThresholdTable } from './InferenceCharts';
import './InferenceResults.css';

const MARGIN_CUTOFFS = [0.1, 0.2, 0.3, 0.4];

/**
 * What the decision margin measures for this target (`decision_margin` in inference_summary.py).
 * Binary margins run from the frozen threshold, not from 0.5.
 */
export function marginNote(summary: Pick<InferenceSummary, 'binary'>) {
  const binary = summary.binary;
  return binary
    ? `How far P(${binary.positiveClass}) lies from the frozen threshold ${binary.threshold}, as a share of the room on that side: 0 is on the threshold, 1 is a probability of 0 or 1. Values near zero are borderline.`
    : 'Probability of the predicted class minus the runner-up. Values near zero are borderline.';
}

export type DistributionView = 'probability' | 'margin';

/**
 * Binary targets switch between the positive-class probability and the decision margin; other
 * targets have only the margin. Both come from the same frozen predictions.
 */
export function DistributionFigure({ summary, view, onViewChange }: { summary: InferenceUnitSummary & Pick<InferenceSummary, 'classOrder'>; view: DistributionView; onViewChange: (view: DistributionView) => void }) {
  const binary = summary.binary;
  const labels = { probability: binary ? `P(${binary.positiveClass})` : '', margin: 'Decision margin' };
  const toggle = binary ? <div className="inference-view-toggle" role="group" aria-label="Distribution to show">{(['probability', 'margin'] as const).map((item) => <button type="button" key={item} aria-pressed={view === item} onClick={() => onViewChange(item)}>{labels[item]}</button>)}</div> : undefined;
  if (binary && view === 'probability') return <Histogram title={`Probability of ${binary.positiveClass}`} description="Positive-class probability for every case, with the frozen decision threshold." edges={binary.positiveProbability.edges} controls={toggle}
    series={[{ label: binary.positiveClass, counts: binary.positiveProbability.counts }]} xLabel={labels.probability}
    colorOffset={summary.classOrder.indexOf(binary.positiveClass)} marker={{ value: binary.threshold, label: `threshold ${binary.threshold}` }} />;
  return <Histogram title="Decision margin by predicted class" description={marginNote(summary)} edges={summary.margin.edges} controls={toggle}
    series={summary.classOrder.map((label) => ({ label, counts: summary.margin.counts[label] ?? [] }))} xLabel="Decision margin" />;
}

/** Everything shown here is computed from verified predictions; no label is ever read. */
export function InferenceOverview({ summary, onReview, memberNote }: { summary: InferenceSummary; onReview?: (query: Partial<CaseQuery>) => void; memberNote?: string }) {
  const [cutoff, setCutoff] = useState(0.2);
  const [binaryView, setBinaryView] = useState<DistributionView>('probability');
  const unit = summary.unit;
  const note = marginNote(summary);
  const low = countBelow(summary.margin, cutoff);
  const development = summary.development;
  return <div className="inference-results">
    <div className="inference-tiles" aria-label="Prediction summary">
      <div><span>{unit === 'patient' ? 'Patients' : 'Slides'} predicted</span><strong>{summary.count.toLocaleString()}</strong><small>{unit === 'slide' && summary.patients ? `${summary.patients.toLocaleString()} patient groups` : 'Frozen cohort membership'}</small></div>
      <div><span>Median confidence</span><strong>{percent(summary.confidence.quantiles.median)}</strong><small>IQR {percent(summary.confidence.quantiles.p25, 0)}–{percent(summary.confidence.quantiles.p75, 0)}</small></div>
      <div title={note}><span>Margin below <select className="inference-inline-select" aria-label="Decision margin cutoff" value={cutoff} onChange={(event) => setCutoff(Number(event.target.value))}>{MARGIN_CUTOFFS.map((value) => <option key={value} value={value}>{value}</option>)}</select></span><strong>{low.toLocaleString()}</strong><small>{percent(summary.count ? low / summary.count : null)} of {unit}s near a decision boundary{onReview && low ? <> · <button type="button" className="text-button" onClick={() => onReview({ sort: 'margin_asc', maxMargin: cutoff })}>review</button></> : null}</small></div>
      {summary.ensemble ? <div><span>Fold members disagree</span><strong>{summary.ensemble.disagreements.toLocaleString()}</strong><small>{percent(summary.ensemble.records ? summary.ensemble.disagreements / summary.ensemble.records : null)} · {summary.ensemble.memberCount} members{onReview && summary.ensemble.disagreements ? <> · <button type="button" className="text-button" onClick={() => onReview({ sort: 'agreement_asc', memberDisagreement: true })}>review</button></> : null}</small></div>
        : <div><span>Fold member agreement</span><strong>—</strong><small>{memberNote ?? 'Member probabilities were not recorded for this run'}</small></div>}
      {development.comparable ? <div><span>From development patients</span><strong>{development.records.toLocaleString()}</strong><small>{development.patients.toLocaleString()} shared patients · new slides only{development.unknown ? ` · ${development.unknown.toLocaleString()} unverifiable` : ''}{onReview && development.records ? <> · <button type="button" className="text-button" onClick={() => onReview({ developmentPatients: 'shared' })}>review</button></> : null}</small></div> : null}
    </div>
    <div className="inference-grid-2">
      <PredictedClassBars predicted={summary.predicted} unit={unit} />
      {summary.ensemble ? <AgreementBars ensemble={summary.ensemble} unit={unit} /> : <figure className="inference-figure"><figcaption>Decision margin</figcaption><p className="muted">{note} Median {decimal(summary.margin.quantiles.median, 2)}; {percent(summary.count ? countBelow(summary.margin, 0.2) / summary.count : null)} of {unit}s have a margin below 0.2.</p></figure>}
    </div>
    <div className="inference-grid-2">
      <Histogram title="Confidence by predicted class" description={`Probability of the predicted class for each ${unit}. Low values are uncertain predictions worth reviewing first.`} edges={summary.confidence.edges}
        series={summary.classOrder.map((label) => ({ label, counts: summary.confidence.counts[label] ?? [] }))} xLabel="Predicted-class probability" />
      <DistributionFigure summary={summary} view={binaryView} onViewChange={setBinaryView} />
    </div>
    {summary.binary ? <ThresholdTable binary={summary.binary} total={summary.count} /> : null}
    {development.comparable && development.records ? <CompositionTable caption="Predictions for new patients and patients seen in development" groupLabel="Patients" unit={unit} classes={summary.classOrder}
      rows={[{ value: 'New patients', counts: development.new, count: Object.values(development.new).reduce((sum, value) => sum + value, 0) }, { value: 'Seen in development', counts: development.shared, count: development.records }]} /> : null}
  </div>;
}
