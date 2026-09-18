import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { modelEvaluations, type ModelEvaluation } from '../api/predictors';
import { formatStatistic } from '../api/statistics';
import { downloadJSON } from '../lib/download';
import { BootstrapDetails } from './PatientAnalysisResults';
import { ErrorNotice } from './ui';

export function comparablePatientEvaluations(left: ModelEvaluation, right: ModelEvaluation) {
  const a = left.execution?.result?.metrics, b = right.execution?.result?.metrics;
  return left.id !== right.id && left.manifest.cohortId === right.manifest.cohortId && a && b
    && JSON.stringify([a.classOrder, a.positiveClass, a.patientAggregation]) === JSON.stringify([b.classOrder, b.positiveClass, b.patientAggregation]);
}

export default function PairedPatientComparison({ project, records }: { project: string; records: ModelEvaluation[] }) {
  const completed = records.filter((row) => row.execution?.status === 'completed' && row.execution.result?.metrics?.patient?.available);
  const [leftId, setLeftId] = useState('');
  const [rightId, setRightId] = useState('');
  const left = completed.find((row) => row.id === leftId) ?? completed[0];
  const compatible = left ? completed.filter((row) => comparablePatientEvaluations(left, row)) : [];
  const right = compatible.find((row) => row.id === rightId) ?? compatible[0];
  const query = useQuery({ queryKey: ['paired-patient-comparison', project, left?.id, left?.contentHash, right?.id, right?.contentHash],
    queryFn: ({ signal }) => modelEvaluations.compare(project, left!.id, right!.id, signal), enabled: Boolean(left && right), staleTime: Infinity, retry: false });
  const result = query.data?.statistics;
  return <section className="evaluation-comparison" aria-label="Paired patient comparison"><h3>Paired patient comparison</h3>
    <p>Compare two fixed predictors on exactly the same patients and outcomes. Both models use the same patient bootstrap draws. Differences below are left minus right. Freeze model choices using development evidence before external evaluation.</p>
    {!left || !right ? <p>Complete two evaluations on the same frozen cohort with matching patient aggregation and class encoding to compare their predictions.</p> : null}
    {left ? <div className="chain-fields"><label className="label">Left evaluation<select className="field" value={left.id} onChange={(event) => setLeftId(event.target.value)}>{completed.map((row) => <option key={row.id} value={row.id}>{row.manifest.name}</option>)}</select></label><label className="label">Right evaluation<select className="field" value={right?.id ?? ''} onChange={(event) => setRightId(event.target.value)}>{!right ? <option value="">No compatible completed evaluation</option> : null}{compatible.map((row) => <option key={row.id} value={row.id}>{row.manifest.name}</option>)}</select></label></div> : null}
    <ErrorNotice error={query.error} />
    {query.isFetching ? <p role="status">Resampling matched patients…</p> : null}
    {result ? <><div className="table-wrap"><table className="chain-table"><thead><tr><th>Patient metric</th><th>Left</th><th>Right</th><th>Left − right (95% CI)</th></tr></thead><tbody>{(['auroc', 'auprc'] as const).map((metric) => <tr key={metric}><th scope="row">{metric.toUpperCase()}</th><td>{formatStatistic(result.estimates?.[metric]?.left)}</td><td>{formatStatistic(result.estimates?.[metric]?.right)}</td><td>{formatStatistic(result.estimates?.[metric]?.difference, result.intervals?.[metric])}</td></tr>)}</tbody></table></div><BootstrapDetails value={result} /><button type="button" className="btn btn-secondary" onClick={() => downloadJSON('paired-patient-comparison.json', query.data)}>Export paired comparison</button></> : null}
  </section>;
}
