import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { clinicalAnalyses, type ClinicalAnalysis, type ClinicalArtifact, type ClinicalReport, type ClinicalSelection } from '../api/clinicalUtility';
import type { EvaluationMetrics, ModelEvaluation } from '../api/predictors';
import { finiteNumber, formatStatistic } from '../lib/evidenceCharts';
import { Findings } from './ScientificUI';
import PublicationConfirmation from './PublicationConfirmation';
import { useReviewedPublication } from './useReviewedPublication';
import CurveChart from './CurveChart';
import { StageRecordManageButton } from './StageWorkflow';
import { ErrorNotice } from './ui';
import '../pages/ClinicalInsights.css';

const DEFAULTS = { name: 'Clinical utility analysis', unit: 'selected' as ClinicalSelection['unit'], threshold: '', bins: '10', thresholdMin: '0.01', thresholdMax: '0.99', thresholdSteps: '99' };

/**
 * Clinical utility of one scored run: calibration, operating points and decision curves from
 * its saved predictions, with outcomes from the labels the run is scored against (its cohort's
 * or a reference standard's). Analyses are saved records of their own; this section lists the
 * run's analyses for those labels, opens one, and creates new ones. Nothing here refits,
 * recalibrates or changes the frozen threshold.
 */
export function RunClinicalUtility({ project, record, metrics, reference, initialAnalysis = '' }: {
  project: string; record: ModelEvaluation; metrics?: EvaluationMetrics;
  /** The reference standard supplying outcomes; absent for the cohort's own labels. */
  reference?: { id: string; name: string }; initialAnalysis?: string;
}) {
  const client = useQueryClient();
  const saved = useQuery({ queryKey: ['clinical-analyses', project, 'all'], queryFn: () => clinicalAnalyses.list(project, true) });
  const referenceId = reference?.id ?? null;
  const analyses = (saved.data?.items ?? []).filter((item) => item.manifest.evaluationId === record.id && item.lifecycleState !== 'trashed' && (item.manifest.selection.referenceId ?? null) === referenceId)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt) || a.id.localeCompare(b.id));
  const [openId, setOpenId] = useState(initialAnalysis);
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState(DEFAULTS);
  const [positiveClass, setPositiveClass] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<Error | null>(null);
  const publication = useReviewedPublication(
    (selection: ClinicalSelection) => clinicalAnalyses.preview(project, selection),
    (selection, hash, operation) => clinicalAnalyses.save(project, selection, hash, operation),
    (preview) => preview.canSave && !preview.findings.some((item) => item.severity === 'error'),
    async (created) => { setOpenId(created.id); setEditing(false); await client.invalidateQueries({ queryKey: ['clinical-analyses', project] }); },
  );
  const classes = metrics?.classOrder ?? [];
  const chosenClass = classes.length === 2 ? metrics?.positiveClass ?? '' : positiveClass;
  const open: ClinicalAnalysis | undefined = analyses.find((item) => item.id === openId) ?? (publication.saved?.id === openId ? publication.saved : undefined) ?? (editing ? undefined : analyses[0]);
  const numericValid = [form.bins, form.thresholdMin, form.thresholdMax, form.thresholdSteps].every((value) => value.trim() && finiteNumber(Number(value)))
    && Number.isInteger(Number(form.bins)) && Number(form.bins) >= 2 && Number(form.bins) <= 50
    && Number(form.thresholdMin) > 0 && Number(form.thresholdMax) < 1 && Number(form.thresholdMin) < Number(form.thresholdMax)
    && Number.isInteger(Number(form.thresholdSteps)) && Number(form.thresholdSteps) >= 2 && Number(form.thresholdSteps) <= 501
    && (!form.threshold.trim() || (finiteNumber(Number(form.threshold)) && Number(form.threshold) > 0 && Number(form.threshold) < 1));
  const canAnalyze = Boolean(form.name.trim() && chosenClass && numericValid && record.lifecycleState === 'active');
  const update = (change: Partial<typeof DEFAULTS>) => { setForm((current) => ({ ...current, ...change })); publication.reset(); };
  function start(copyOf?: ClinicalAnalysis) {
    if (publication.locked) return;
    publication.reset(); setEditing(true); setOpenId('');
    const selection = copyOf?.manifest.selection;
    setForm(selection ? { name: `${selection.name.slice(0, 110)} copy`, unit: selection.unit, threshold: selection.threshold == null ? '' : String(selection.threshold), bins: String(selection.bins), thresholdMin: String(selection.thresholdMin), thresholdMax: String(selection.thresholdMax), thresholdSteps: String(selection.thresholdSteps) } : DEFAULTS);
    setPositiveClass(selection?.positiveClass ?? '');
  }
  async function download(filename: ClinicalArtifact) {
    if (!open || downloading) return;
    setDownloading(true); setDownloadError(null);
    try { await clinicalAnalyses.download(project, open.id, filename); }
    catch (reason) { setDownloadError(reason instanceof Error ? reason : new Error('Report download failed.')); }
    finally { setDownloading(false); }
  }
  const report = publication.review?.preview.manifest?.report ?? (!editing ? open?.manifest.report : undefined);
  return <section className="run-section clinical-insights" aria-labelledby={`clinical-${record.id}`}>
    <div className="run-section-heading"><h3 id={`clinical-${record.id}`}>Clinical utility</h3>
      {!editing ? <button type="button" className="btn btn-secondary btn-small" disabled={publication.locked || record.lifecycleState !== 'active'} onClick={() => start()}>New analysis</button> : null}</div>
    <p className="muted">Calibration, operating characteristics and decision curves from this run&rsquo;s saved predictions{reference ? `, with outcomes from ${reference.name}` : ''}. An exploratory threshold never changes the frozen predictor or its metrics.</p>
    <ErrorNotice error={publication.error ?? saved.error ?? downloadError} />
    {analyses.length > 1 && !editing ? <label className="label">Saved analysis<select className="field" value={open?.id ?? ''} onChange={(event) => { publication.reset(); setOpenId(event.target.value); }}>{analyses.map((item) => <option key={item.id} value={item.id}>{item.manifest.name} · {item.manifest.report.positiveClass} · {item.manifest.report.unit}</option>)}</select></label> : null}
    {editing && !publication.review ? <fieldset className="chain-fields" disabled={publication.locked}><legend className="sr-only">Clinical analysis settings</legend>
      <label className="label">Report name<input className="field" value={form.name} maxLength={120} onChange={(event) => update({ name: event.target.value })} /></label>
      <label className="label">Prediction unit<select className="field" value={form.unit} onChange={(event) => update({ unit: event.target.value as ClinicalSelection['unit'] })}><option value="selected">Frozen target unit ({metrics?.unit ?? 'configured'})</option><option value="patient">Patient</option><option value="slide">Slide</option></select></label>
      <label className="label">Positive outcome<select className="field" value={chosenClass} disabled={classes.length === 2} onChange={(event) => { setPositiveClass(event.target.value); publication.reset(); }}><option value="">Choose the outcome of interest</option>{classes.map((value) => <option key={value} value={value}>{value}{classes.length > 2 ? ' versus all other classes' : ' · frozen positive class'}</option>)}</select></label>
      <label className="label">Operating threshold<input className="field" type="number" min="0.001" max="0.999" step="any" placeholder={`Frozen (${metrics?.decisionThreshold ?? 0.5})`} value={form.threshold} onChange={(event) => update({ threshold: event.target.value })} /><span className="muted">Leave blank to use the frozen threshold.</span></label>
      <details className="chain-wide"><summary>Curve range and calibration bins</summary><div className="chain-fields threshold-fields">
        <label className="label">Minimum threshold<input className="field" type="number" min="0.001" max="0.998" step="any" value={form.thresholdMin} onChange={(event) => update({ thresholdMin: event.target.value })} /></label>
        <label className="label">Maximum threshold<input className="field" type="number" min="0.002" max="0.999" step="any" value={form.thresholdMax} onChange={(event) => update({ thresholdMax: event.target.value })} /></label>
        <label className="label">Threshold points<input className="field" type="number" min="2" max="501" value={form.thresholdSteps} onChange={(event) => update({ thresholdSteps: event.target.value })} /></label>
        <label className="label">Calibration bins<input className="field" type="number" min="2" max="50" value={form.bins} onChange={(event) => update({ bins: event.target.value })} /></label>
      </div><p className="muted">Use a clinically plausible threshold range. Decision curves depend on outcome prevalence and the relative harm of false positives; comparing curves alone does not establish clinical benefit.</p></details>
      {!numericValid ? <p className="callout chain-wide" role="status">Use thresholds strictly between 0 and 1, an increasing range, 2–501 curve points and 2–50 calibration bins.</p> : null}
      <div className="inline-actions chain-wide">
        <button type="button" className="btn btn-primary" disabled={!canAnalyze || publication.locked} onClick={() => { if (canAnalyze) void publication.preview({ evaluationId: record.id, name: form.name.trim(), unit: form.unit, positiveClass: chosenClass, threshold: form.threshold.trim() ? Number(form.threshold) : null, bins: Number(form.bins), thresholdMin: Number(form.thresholdMin), thresholdMax: Number(form.thresholdMax), thresholdSteps: Number(form.thresholdSteps), ...(referenceId ? { referenceId } : {}) }); }}>{publication.busy ? 'Calculating…' : 'Analyze clinical utility'}</button>
        <button type="button" className="text-button" disabled={publication.locked} onClick={() => { setEditing(false); publication.reset(); }}>Cancel</button>
      </div>
    </fieldset> : null}
    {publication.review ? <Findings findings={publication.review.preview.findings} /> : null}
    {report ? <>
      <ClinicalReportView report={report} />
      {publication.review ? <div className="inline-actions"><button type="button" className="text-button" disabled={publication.locked} onClick={() => publication.reset()}>Back to analysis settings</button></div> : null}
      {publication.review?.preview.canSave ? <PublicationConfirmation busy={publication.busy} uncertain={publication.review.uncertain} acknowledged={publication.acknowledged} onAcknowledge={publication.setAcknowledged} onConfirm={() => void publication.publish()} label="Save clinical utility report" /> : null}
      {!publication.review && open ? <div className="inline-actions">
        {(['report.json', 'operating-curves.csv', 'calibration.csv', 'roc.csv', 'precision-recall.csv'] as const).map((file) => <button key={file} type="button" className="btn btn-secondary btn-small" disabled={downloading} onClick={() => void download(file)}>{CLINICAL_DOWNLOADS[file]}</button>)}
        <button type="button" className="text-button" onClick={() => start(open)}>Copy into a new analysis</button>
        <StageRecordManageButton type="configuration" id={open.id} name={open.manifest.name} />
      </div> : null}
    </> : !editing ? <p className="muted">{saved.isPending ? 'Loading saved analyses…' : `No clinical utility analysis for this run${reference ? ` against ${reference.name}` : ''} yet.`}</p> : null}
  </section>;
}

const CLINICAL_DOWNLOADS: Record<ClinicalArtifact, string> = {
  'report.json': 'Report (JSON)', 'operating-curves.csv': 'Operating curves', 'calibration.csv': 'Calibration',
  'roc.csv': 'ROC curve', 'precision-recall.csv': 'Precision–recall curve',
};

/** Where the report's threshold came from. Earlier multiclass reports marked the frozen threshold
 * as an override even when nothing was overridden, so an unchanged value reads as frozen. */
export function clinicalThresholdNote(r: Pick<ClinicalReport, 'thresholdSource' | 'decisionThreshold' | 'frozenDecisionThreshold' | 'multiclass' | 'positiveClass'>) {
  if (r.thresholdSource === 'descriptive_override' && r.decisionThreshold !== r.frozenDecisionThreshold) return `Exploratory threshold; the frozen evaluation still uses ${formatStatistic(r.frozenDecisionThreshold)}.`;
  return r.multiclass
    ? `This is the frozen evaluation threshold, applied to ${r.positiveClass} versus the other classes; the evaluation's own decisions pick the most probable class.`
    : 'This is the frozen evaluation threshold.';
}

export function ClinicalReportView({ report }: { report: ClinicalReport }) {
  const r = report, point = r.operatingPoint;
  const [fullDecisionRange, setFullDecisionRange] = useState(false);
  const curve = (key: keyof typeof point) => r.operatingCurve.map((item) => ({ x: item.threshold, y: item[key] }));
  return <>
    <p className="evidence-counts"><strong>{r.counts.labeled} labeled {r.unit} records</strong> · {r.counts.positive} positive · {r.counts.negative} negative · {r.counts.unlabeled} unlabeled excluded. Positive outcome: <strong>{r.positiveClass}</strong>{r.multiclass ? ' versus all other classes' : ''}.</p>
    <p className="muted">Probability ≥ {formatStatistic(r.decisionThreshold)} predicts the positive outcome. {clinicalThresholdNote(r)} These descriptive estimates do not establish clinical validity or an optimal threshold.</p>
    {r.warnings.length ? <ul className="callout">{r.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
    {r.curveSampling?.downsampled ? <p className="muted">ROC and precision–recall curves show {r.curveSampling.returnedPoints} representative cutoffs from {r.curveSampling.distinctScores} distinct scores. Summary statistics use all predictions.</p> : null}
    <div className="chain-metrics">{([
      ['Brier score', r.metrics.brierScore, 'Mean squared probability error; lower is better.'],
      ['Brier reference', r.metrics.brierReference, 'Constant prediction at observed cohort prevalence.'],
      ['Brier skill score', r.metrics.brierSkillScore, 'Relative to this descriptive prevalence reference.'],
      ['AUROC', r.metrics.rocAuc, 'Ranking across all score cutoffs.'],
      ['Average precision', r.metrics.averagePrecision, 'Precision–recall summary; depends on prevalence.'],
      ['Outcome prevalence', r.metrics.prevalence, 'Positive fraction of labeled records.'],
      ['Mean predicted risk', r.metrics.meanPredictedRisk ?? null, 'Average predicted positive-outcome probability.'],
      ['Observed / expected ratio', r.metrics.observedExpectedRatio ?? null, 'Observed positive count divided by the sum of predicted risks.'],
      ['Calibration gap', r.metrics.calibrationGap ?? null, 'Observed prevalence minus mean predicted risk.'],
      ['Log loss', r.metrics.logLoss, 'Probability error penalizing confident mistakes.'],
      ['Calibration error (ECE)', r.metrics.ece, 'Count-weighted error across the selected bins.'],
      ['Maximum calibration error', r.metrics.mce, 'Largest observed bin error.'],
      ...(r.multiclass ? [['Multiclass Brier score', r.metrics.multiclassBrierScore, 'Squared error summed across all classes.']] : []),
    ] as [string, number | null, string][]).map(([label, value, explanation]) => <div key={label}><strong>{formatStatistic(value)}</strong><span>{label}</span><span className="stat-help">{explanation}</span></div>)}</div>
    <div className="evidence-chart-grid">
      <CurveChart title="Calibration" description="Observed positive fraction versus mean predicted risk in each nonempty bin. The diagonal represents perfect calibration." xLabel="Mean predicted probability" yLabel="Observed positive fraction" yRange={[0, 1]} referenceDiagonal series={[{ label: 'Evaluation', points: r.calibration.map((item) => ({ x: item.meanPredicted, y: item.observedFraction })) }]} />
      <CurveChart title="Receiver operating characteristic" description="Sensitivity versus false-positive rate across score cutoffs. The diagonal is chance discrimination." xLabel="False-positive rate (1 − specificity)" yLabel="Sensitivity" yRange={[0, 1]} referenceDiagonal series={[{ label: 'Predictor', points: r.rocCurve.map((item) => ({ x: item.falsePositiveRate, y: item.truePositiveRate })) }]} />
      <CurveChart title="Precision–recall curve" description="Positive predictive value versus sensitivity. The reference is the observed positive outcome prevalence." xLabel="Recall (sensitivity)" yLabel="Precision (PPV)" yRange={[0, 1]} series={[{ label: 'Predictor', points: r.precisionRecallCurve.map((item) => ({ x: item.recall, y: item.precision })) }, { label: 'Prevalence', dashed: true, points: [{ x: 0, y: r.metrics.prevalence }, { x: 1, y: r.metrics.prevalence }] }]} />
      <CurveChart title="Operating characteristics" description="The tradeoff between finding positive outcomes and avoiding false positives as the decision threshold changes." xLabel="Decision threshold" yLabel="Fraction" yRange={[0, 1]} series={[{ label: 'Sensitivity', points: curve('sensitivity') }, { label: 'Specificity', points: curve('specificity') }, { label: 'PPV', points: curve('ppv'), dashed: true }, { label: 'NPV', points: curve('npv'), dashed: true }]} />
      <div><CurveChart title="Decision curve: net benefit" description={`True positives minus threshold-weighted false positives per record. ${fullDecisionRange ? 'Full observed vertical range.' : 'Focused view: values below −0.10 are clipped; full values remain in the table and exports.'}`} xLabel="Decision threshold" yLabel="Net benefit per record" yRange={fullDecisionRange ? undefined : [-.1, Math.max(.1, r.metrics.prevalence * 1.1)]} series={[{ label: 'Predictor', points: curve('netBenefit') }, { label: 'Treat all', dashed: true, points: curve('treatAllNetBenefit') }, { label: 'Treat none', dashed: true, points: curve('treatNoneNetBenefit') }]} /><label className="checkbox-label"><input type="checkbox" checked={fullDecisionRange} onChange={(event) => setFullDecisionRange(event.target.checked)} /> Show full net-benefit range</label></div>
      <CurveChart title="Potential clinical impact" description="How many records would be flagged and how many flagged records actually have the positive outcome, per 100 labeled records." xLabel="Decision threshold" yLabel="Records per 100" yRange={[0, 100]} series={[{ label: 'Flagged positive', points: curve('highRiskPer100') }, { label: 'True positives', points: curve('truePositivePer100') }, { label: 'False positives', points: curve('falsePositivePer100'), dashed: true }, { label: 'Missed positives', points: curve('missedPositivePer100'), dashed: true }]} />
      <CurveChart title="Net interventions avoided" description="Equivalent net reduction in interventions per 100 records compared with treating everyone, under the threshold’s harm tradeoff." xLabel="Decision threshold" yLabel="Net interventions avoided per 100" series={[{ label: 'Predictor versus treat all', points: curve('netInterventionsAvoidedPer100') }]} />
    </div>
    <h3>At threshold {formatStatistic(r.decisionThreshold)}</h3>
    <div className="table-wrap"><table className="chain-table"><thead><tr><th>Statistic</th><th>Value</th><th>Statistic</th><th>Value</th></tr></thead><tbody>{([
      ['Sensitivity', point.sensitivity, 'Specificity', point.specificity], ['Positive predictive value', point.ppv, 'Negative predictive value', point.npv],
      ['Positive likelihood ratio', point.positiveLikelihoodRatio, 'Negative likelihood ratio', point.negativeLikelihoodRatio],
      ['Accuracy', point.accuracy, 'Balanced accuracy', point.balancedAccuracy], ['F1 score', point.f1, 'Net benefit', point.netBenefit],
      ['Standardized net benefit', point.standardizedNetBenefit, 'Net interventions avoided per 100', point.netInterventionsAvoidedPer100],
    ] as [string, number | null, string, number | null][]).map(([a, av, b, bv]) => <tr key={a}><th scope="row">{a}</th><td>{formatStatistic(av)}</td><th scope="row">{b}</th><td>{formatStatistic(bv)}</td></tr>)}</tbody></table></div>
    <p>True positives: <strong>{point.tp}</strong> · False positives: <strong>{point.fp}</strong> · True negatives: <strong>{point.tn}</strong> · False negatives: <strong>{point.fn}</strong>. Undefined statistics are shown as unavailable.</p>
    {r.uncertainty ? <><p className="muted">{r.uncertainty.method === 'wilson_95' ? '95% Wilson confidence intervals for proportions. ' : 'Confidence intervals unavailable. '}{r.uncertainty.reason}</p>{r.uncertainty.method === 'wilson_95' ? <p>{(['sensitivity', 'specificity', 'ppv', 'npv'] as const).map((key) => { const interval = r.uncertainty?.operatingPoint[key]; return <span key={key}>{key.toUpperCase()}: {interval ? `${formatStatistic(interval.lower)}–${formatStatistic(interval.upper)}` : 'Unavailable'} · </span>; })}</p> : null}</> : null}
    <details><summary>Calibration bin counts and observed outcomes</summary><div className="table-wrap"><table className="chain-table"><thead><tr><th>Probability bin</th><th>Records</th><th>Mean probability</th><th>Observed positive fraction</th><th>Absolute error</th></tr></thead><tbody>{r.calibration.map((item, index) => <tr key={index}><th scope="row">{formatStatistic(item.lower, 2)}–{formatStatistic(item.upper, 2)}</th><td>{item.count}</td><td>{formatStatistic(item.meanPredicted)}</td><td>{formatStatistic(item.observedFraction)}</td><td>{formatStatistic(item.absoluteError)}</td></tr>)}</tbody></table></div></details>
    <details><summary>Methods and references</summary><dl>{Object.entries(r.definitions).map(([key, definition]) => <div key={key}><dt>{key}</dt><dd>{definition}</dd></div>)}</dl><ul>{r.sources.filter((source) => /^https?:\/\//i.test(source.url)).map((source) => <li key={source.url}><a href={source.url} target="_blank" rel="noreferrer">{source.title}</a></li>)}</ul></details>
  </>;
}
