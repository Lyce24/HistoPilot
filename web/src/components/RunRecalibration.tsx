import { useQuery } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import type { ModelEvaluation } from '../api/predictors';
import { recalibration, type CalibrationMetrics, type Recalibration } from '../api/recalibration';
import { formatStatistic } from '../api/statistics';
import { downloadJSON } from '../lib/download';
import CurveChart from './CurveChart';
import { ErrorNotice } from './ui';
import './ApplyRunDetail.css';

/** A run that cannot be recalibrated, now or at all, reads as a note rather than an error. */
const EXPECTED_CODES = new Set(['RECALIBRATION_UNAVAILABLE', 'RECALIBRATION_REQUIRES_LABELS']);

const ROWS: [keyof CalibrationMetrics, string, string][] = [
  ['brierScore', 'Brier score', 'Mean squared error of the risks; lower is better.'],
  ['logLoss', 'Log loss', 'Penalizes confident mistakes; lower is better.'],
  ['ece', 'Calibration error (ECE)', 'Gap between risk and outcome across 10 equal-width bins.'],
  ['calibrationSlope', 'Calibration slope', 'Below 1: risks too extreme; above 1: too cautious.'],
  ['calibrationIntercept', 'Calibration intercept', 'Offset of the outcome against logit(risk); 0 when calibrated.'],
  ['observedExpectedRatio', 'Observed / expected', 'Observed outcomes divided by the sum of risks.'],
];

/** The recalibration map in words: its form, fitted values and the units it was fitted on. */
export function recalibrationMap(report: Pick<Recalibration, 'method' | 'parameters' | 'development' | 'unit'>) {
  const units = `${report.development.units.toLocaleString()} development ${report.unit}s${report.development.seedGroups > 1 ? ` (${report.development.seedGroups} seed groups averaged)` : ''}`;
  if ('temperature' in report.parameters) return `Temperature scaling, T = ${formatStatistic(report.parameters.temperature)}, fitted on ${units}.`;
  const { slope, intercept } = report.parameters;
  return `Platt scaling, logit p′ = ${formatStatistic(slope)} · logit p ${intercept < 0 ? '−' : '+'} ${formatStatistic(Math.abs(intercept))}, fitted on ${units}.`;
}

/**
 * Calibration of a scored run as predicted and after recalibration. The map is fitted on
 * the predictor's out-of-fold development predictions, never on this cohort, and only
 * rescales risks: decisions, thresholds and ranking metrics keep the original probabilities.
 */
export default function RunRecalibration({ project, record, unit, referenceId }: {
  project: string; record: ModelEvaluation; unit: 'selected' | 'slide' | 'patient'; referenceId: string | null;
}) {
  const query = useQuery({
    queryKey: ['run-recalibration', project, record.id, unit, referenceId],
    queryFn: ({ signal }) => recalibration.get(project, record.id, unit, referenceId, signal),
    staleTime: 300000,
  });
  const data = query.isError ? undefined : query.data;
  const unavailable = query.error instanceof ApiError && EXPECTED_CODES.has(query.error.code ?? '') ? query.error.message : null;
  const binary = data?.method === 'platt';
  const risk = binary ? `P(${data?.positiveClass})` : 'Confidence';
  return <section className="run-section" aria-labelledby={`recalibration-${record.id}`}>
    <div className="run-section-heading"><h3 id={`recalibration-${record.id}`}>Recalibration</h3>
      {data ? <button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${record.manifest.name} recalibration.json`, data)}>Recalibration (JSON)</button> : null}</div>
    <p className="muted">How well predicted risks match outcomes, as predicted and after a map fitted on this predictor&rsquo;s out-of-fold development predictions, never on this cohort. Recalibration only rescales risks: decisions, the frozen threshold and ranking metrics keep the original probabilities.</p>
    <ErrorNotice error={unavailable ? null : query.error} />
    {unavailable ? <p className="muted" role="status">{unavailable}</p> : null}
    {query.isPending ? <p role="status">Fitting the map on development predictions…</p> : null}
    {data ? <>
      <p className="muted">{recalibrationMap(data)} Scored on {data.cohort.units.toLocaleString()} labeled {data.unit}s{data.reference ? ` from ${data.reference.name}` : ''}{data.developmentExcluded ? `; ${data.developmentExcluded.toLocaleString()} slides of development patients left out` : ''}.</p>
      <div className="table-wrap"><table className="chain-table recalibration-table"><thead><tr><th scope="col">{binary ? `Risk of ${data.positiveClass}` : 'Confidence of the predicted class'}</th><th scope="col">As predicted</th><th scope="col">Recalibrated</th></tr></thead>
        <tbody>{ROWS.map(([key, label, help]) => <tr key={key}><th scope="row">{label}<small>{help}</small></th><td>{formatStatistic(data.cohort.original[key] as number | null)}</td><td>{formatStatistic(data.cohort.recalibrated[key] as number | null)}</td></tr>)}</tbody></table></div>
      <CurveChart title="Reliability" description={`Observed ${binary ? `fraction of ${data.positiveClass}` : 'accuracy'} against mean ${binary ? 'risk' : 'confidence'} in each nonempty bin, as predicted and recalibrated. The diagonal is perfect calibration.`}
        xLabel={`Mean ${risk}`} yLabel={binary ? `Observed fraction of ${data.positiveClass}` : 'Observed accuracy'} yRange={[0, 1]} referenceDiagonal
        series={[
          { label: 'As predicted', points: data.cohort.original.bins.map((bin) => ({ x: bin.meanPredicted, y: bin.observedFraction })) },
          { label: 'Recalibrated', dashed: true, points: data.cohort.recalibrated.bins.map((bin) => ({ x: bin.meanPredicted, y: bin.observedFraction })) },
        ]} />
    </> : null}
  </section>;
}
