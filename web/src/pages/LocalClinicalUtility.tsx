import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { clinicalAnalyses, type ClinicalArtifact, type ClinicalReport, type ClinicalSelection } from '../api/clinicalUtility';
import { modelEvaluations, predictors } from '../api/predictors';
import type { Workspace } from '../api/types';
import { ErrorNotice, PageHeader, Panel } from '../components/ui';
import { Findings } from '../components/ScientificUI';
import PublicationConfirmation from '../components/PublicationConfirmation';
import { useReviewedPublication } from '../components/useReviewedPublication';
import EvidenceChain, { evidenceLink } from '../components/EvidenceChain';
import CurveChart from '../components/CurveChart';
import { finiteNumber, formatStatistic } from '../lib/evidenceCharts';
import { useHashParameters } from '../lib/hashRoute';
import './ModelChains.css';
import './ClinicalInsights.css';

export default function LocalClinicalUtility({ workspace }: { workspace: Workspace }) {
  const parameters = useHashParameters();
  return <ClinicalWorkspace key={`${workspace.project.id}:${parameters}`} workspace={workspace} linkedEvaluation={parameters.get('evaluation') ?? ''} linkedPredictor={parameters.get('predictor') ?? ''} linkedReport={parameters.get('clinical') ?? ''} />;
}
function ClinicalWorkspace({ workspace, linkedEvaluation, linkedPredictor, linkedReport }: { workspace: Workspace; linkedEvaluation: string; linkedPredictor: string; linkedReport: string }) {
  const project = workspace.project.id;
  const client = useQueryClient();
  const evaluations = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project) });
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project) });
  const saved = useQuery({ queryKey: ['clinical-analyses', project], queryFn: () => clinicalAnalyses.list(project) });
  const linked = useQuery({ queryKey: ['clinical-analysis', project, linkedReport], queryFn: () => clinicalAnalyses.get(project, linkedReport), enabled: Boolean(linkedReport) });
  const [evaluationId, setEvaluationId] = useState(linkedEvaluation);
  const [savedId, setSavedId] = useState(linkedReport);
  const [name, setName] = useState('Clinical utility analysis');
  const [unit, setUnit] = useState<ClinicalSelection['unit']>('selected');
  const [positiveClass, setPositiveClass] = useState('');
  const [threshold, setThreshold] = useState('');
  const [bins, setBins] = useState('10');
  const [thresholdMin, setThresholdMin] = useState('0.01');
  const [thresholdMax, setThresholdMax] = useState('0.99');
  const [thresholdSteps, setThresholdSteps] = useState('99');
  const [downloadError, setDownloadError] = useState<Error | null>(null);
  const [downloading, setDownloading] = useState(false);
  const publication = useReviewedPublication(
    (selection: ClinicalSelection) => clinicalAnalyses.preview(project, selection),
    (selection, hash, operation) => clinicalAnalyses.save(project, selection, hash, operation),
    (preview) => preview.canSave && !preview.findings.some((item) => item.severity === 'error'),
    async (record) => { setSavedId(record.id); await client.invalidateQueries({ queryKey: ['clinical-analyses', project] }); },
  );
  const selected = evaluations.data?.items.find((item) => item.id === evaluationId);
  const sourcePredictor = registry.data?.items.find((item) => item.id === selected?.manifest.predictorId);
  const classes = selected?.execution?.result?.metrics?.classOrder ?? sourcePredictor?.manifest.target.classes ?? [];
  const frozenPositive = selected?.execution?.result?.metrics?.positiveClass ?? sourcePredictor?.manifest.target.positiveClass ?? '';
  const chosenClass = classes.length === 2 ? frozenPositive : positiveClass;
  const available = (evaluations.data?.items ?? []).filter((item) => item.lifecycleState !== 'trashed' && item.execution?.status === 'completed' && (!linkedPredictor || item.manifest.predictorId === linkedPredictor));
  const savedRecord = saved.data?.items.find((item) => item.id === savedId && item.lifecycleState !== 'trashed') ?? (publication.saved?.id === savedId ? publication.saved : null) ?? (linked.data?.id === savedId ? linked.data : null);
  const savedSelection = savedRecord?.manifest.selection;
  useEffect(() => {
    if (!savedSelection) return;
    setEvaluationId(savedSelection.evaluationId); setName(savedSelection.name); setUnit(savedSelection.unit); setPositiveClass(savedSelection.positiveClass ?? '');
    setThreshold(savedSelection.threshold == null ? '' : String(savedSelection.threshold)); setBins(String(savedSelection.bins)); setThresholdMin(String(savedSelection.thresholdMin)); setThresholdMax(String(savedSelection.thresholdMax)); setThresholdSteps(String(savedSelection.thresholdSteps));
  }, [savedSelection]);
  const manifest = publication.review?.preview.manifest ?? savedRecord?.manifest;
  const report = manifest?.report;
  const context = { experimentId: manifest?.experimentId ?? selected?.manifest.experimentId, predictorId: manifest?.predictorId ?? selected?.manifest.predictorId ?? linkedPredictor, evaluationId: manifest?.evaluationId ?? evaluationId, clinicalAnalysisId: savedRecord?.id };
  const numericValid = [bins, thresholdMin, thresholdMax, thresholdSteps].every((value) => value.trim() && finiteNumber(Number(value))) && Number.isInteger(Number(bins)) && Number(bins) >= 2 && Number(bins) <= 50 && Number(thresholdMin) > 0 && Number(thresholdMax) < 1 && Number(thresholdMin) < Number(thresholdMax) && Number.isInteger(Number(thresholdSteps)) && Number(thresholdSteps) >= 2 && Number(thresholdSteps) <= 501 && (!threshold.trim() || (finiteNumber(Number(threshold)) && Number(threshold) > 0 && Number(threshold) < 1));
  const canAnalyze = available.some((item) => item.id === evaluationId) && name.trim() && chosenClass && numericValid && !evaluations.isError && !registry.isError;
  function resetReview() { publication.reset(); setSavedId(''); setDownloadError(null); }
  async function download(filename: ClinicalArtifact) {
    if (!savedRecord || downloading) return;
    setDownloading(true); setDownloadError(null);
    try { await clinicalAnalyses.download(project, savedRecord.id, filename); }
    catch (reason) { setDownloadError(reason instanceof Error ? reason : new Error('Report download failed.')); }
    finally { setDownloading(false); }
  }
  return <div className="clinical-workspace model-chains clinical-insights">
    <PageHeader eyebrow="04 CLINICAL INSIGHTS" title="Clinical utility" description="Assess probability quality, operating tradeoffs and potential clinical benefit using completed model evaluations." />
    <EvidenceChain current="clinical-utility" {...context} />
    <ErrorNotice error={publication.error ?? evaluations.error ?? registry.error ?? saved.error ?? linked.error ?? downloadError} />
    <Panel title="Choose evaluation evidence" subtitle="Analyze saved test predictions. An exploratory operating threshold does not change the frozen evaluation or predictor.">
      {evaluations.isPending ? <p role="status">Loading completed evaluations…</p> : !available.length ? <p className="callout">No completed evaluations are available{linkedPredictor ? ' for this predictor' : ''}. <a href={evidenceLink('evaluation', { predictorId: linkedPredictor })}>Evaluate a predictor</a> to generate test predictions first.</p> : null}
      <fieldset className="chain-fields" disabled={publication.locked} onChange={resetReview}><legend className="sr-only">Clinical analysis settings</legend>
        <label className="label chain-wide">Completed evaluation<select className="field" value={evaluationId} onChange={(event) => { setEvaluationId(event.target.value); setPositiveClass(''); setThreshold(''); }}><option value="">Choose a completed evaluation</option>{evaluationId && !available.some((item) => item.id === evaluationId) ? <option value={evaluationId} disabled>Linked evaluation is unavailable or incomplete</option> : null}{available.map((item) => <option key={item.id} value={item.id}>{item.manifest.name}{item.lifecycleState === 'archived' ? ' · archived' : ''}</option>)}</select></label>
        <label className="label">Prediction unit<select className="field" value={unit} onChange={(event) => setUnit(event.target.value as ClinicalSelection['unit'])}><option value="selected">Frozen target unit ({selected?.execution?.result?.metrics?.unit ?? 'configured'})</option><option value="patient">Patient</option><option value="slide">Slide</option></select></label>
        <label className="label">Positive outcome<select className="field" value={chosenClass} disabled={classes.length === 2} onChange={(event) => setPositiveClass(event.target.value)}><option value="">Choose the outcome of interest</option>{classes.map((value) => <option key={value} value={value}>{value}{classes.length > 2 ? ' versus all other classes' : ' · frozen positive class'}</option>)}</select></label>
        <label className="label">Report name<input className="field" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} /></label>
        <label className="label">Operating threshold<input className="field" type="number" min="0.001" max="0.999" step="any" placeholder={`Frozen evaluation (${selected?.execution?.result?.metrics?.decisionThreshold ?? .5})`} value={threshold} onChange={(event) => setThreshold(event.target.value)} /><span className="muted">Leave blank to use the frozen threshold.</span></label>
      </fieldset>
      <details><summary>Curve range and calibration bins</summary><fieldset className="chain-fields threshold-fields" disabled={publication.locked} onChange={resetReview}><legend className="sr-only">Curve settings</legend>
        <label className="label">Minimum threshold<input className="field" type="number" min="0.001" max="0.998" step="any" value={thresholdMin} onChange={(event) => setThresholdMin(event.target.value)} /></label>
        <label className="label">Maximum threshold<input className="field" type="number" min="0.002" max="0.999" step="any" value={thresholdMax} onChange={(event) => setThresholdMax(event.target.value)} /></label>
        <label className="label">Threshold points<input className="field" type="number" min="2" max="501" value={thresholdSteps} onChange={(event) => setThresholdSteps(event.target.value)} /></label>
        <label className="label">Calibration bins<input className="field" type="number" min="2" max="50" value={bins} onChange={(event) => setBins(event.target.value)} /></label>
      </fieldset><p className="muted">Use a clinically plausible threshold range. Decision curves depend on outcome prevalence and the relative harm of false positives; comparing curves alone does not establish clinical benefit.</p></details>
      {!numericValid ? <p className="callout" role="status">Use thresholds strictly between 0 and 1, an increasing range, 2–501 curve points and 2–50 calibration bins.</p> : null}
      <button className="btn btn-primary" disabled={!canAnalyze || publication.locked} onClick={() => { if (canAnalyze) { setSavedId(''); void publication.preview({ evaluationId, name: name.trim(), unit, positiveClass: chosenClass, threshold: threshold.trim() ? Number(threshold) : null, bins: Number(bins), thresholdMin: Number(thresholdMin), thresholdMax: Number(thresholdMax), thresholdSteps: Number(thresholdSteps) }); } }}>{publication.busy ? 'Calculating…' : 'Analyze clinical utility'}</button>
    </Panel>
    {publication.review ? <Findings findings={publication.review.preview.findings} /> : null}
    {report ? <Panel title={manifest?.name ?? 'Clinical utility report'} subtitle={savedRecord ? 'Saved analysis · immutable source predictions and settings' : 'Review the analysis before saving its evidence and settings'}>
      <ClinicalReportView report={report} />
      {publication.review?.preview.canSave ? <PublicationConfirmation busy={publication.busy} uncertain={publication.review.uncertain} acknowledged={publication.acknowledged} onAcknowledge={publication.setAcknowledged} onConfirm={() => void publication.publish()} onReset={publication.reset} label="Save clinical utility report" /> : null}
      {savedRecord ? <><p className="science-success" role="status">Saved report: {savedRecord.manifest.name}</p><div className="inline-actions"><button className="btn btn-secondary" disabled={downloading} onClick={() => void download('report.json')}>Download report JSON</button><button className="btn btn-secondary" disabled={downloading} onClick={() => void download('operating-curves.csv')}>Download operating curves</button><button className="btn btn-secondary" disabled={downloading} onClick={() => void download('calibration.csv')}>Download calibration</button><button className="btn btn-secondary" disabled={downloading} onClick={() => void download('roc.csv')}>Download ROC curve</button><button className="btn btn-secondary" disabled={downloading} onClick={() => void download('precision-recall.csv')}>Download precision–recall curve</button><a className="btn btn-primary" href={evidenceLink('interpretation', context)}>Next: interpret this predictor</a></div></> : null}
    </Panel> : null}
    <Panel title="Saved clinical analyses" subtitle="Open a report to trace its evaluation, predictor and experiment.">{saved.isPending ? <p role="status">Loading saved reports…</p> : !(saved.data?.items ?? []).filter((item) => item.lifecycleState !== 'trashed').length ? <p className="muted">No saved clinical analyses yet.</p> : (saved.data?.items ?? []).filter((item) => item.lifecycleState !== 'trashed' && (!linkedPredictor || item.manifest.predictorId === linkedPredictor)).map((item) => <article className="report-card" key={item.id}><button className="text-button" disabled={publication.locked} aria-pressed={savedId === item.id} onClick={() => { publication.reset(); setSavedId(item.id); setEvaluationId(item.manifest.evaluationId); }}>{item.manifest.name}</button><small>{item.manifest.report.counts.labeled} labeled {item.manifest.report.unit} records · {item.manifest.report.positiveClass} · threshold {formatStatistic(item.manifest.report.decisionThreshold)}{item.lifecycleState === 'archived' ? ' · archived' : ''}</small><a href={evidenceLink('interpretation', { experimentId: item.manifest.experimentId, predictorId: item.manifest.predictorId, evaluationId: item.manifest.evaluationId, clinicalAnalysisId: item.id })}>Interpret predictor</a></article>)}</Panel>
  </div>;
}

export function ClinicalReportView({ report }: { report: ClinicalReport }) {
  const r = report, point = r.operatingPoint;
  const [fullDecisionRange, setFullDecisionRange] = useState(false);
  const curve = (key: keyof typeof point) => r.operatingCurve.map((item) => ({ x: item.threshold, y: item[key] }));
  return <>
    <p className="evidence-counts"><strong>{r.counts.labeled} labeled {r.unit} records</strong> · {r.counts.positive} positive · {r.counts.negative} negative · {r.counts.unlabeled} unlabeled excluded. Positive outcome: <strong>{r.positiveClass}</strong>{r.multiclass ? ' versus all other classes' : ''}.</p>
    <p className="muted">Probability ≥ {formatStatistic(r.decisionThreshold)} predicts the positive outcome. {r.thresholdSource === 'descriptive_override' ? `Exploratory threshold; the frozen evaluation still uses ${formatStatistic(r.frozenDecisionThreshold)}.` : 'This is the frozen evaluation threshold.'} These descriptive estimates do not establish clinical validity or an optimal threshold.</p>
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
