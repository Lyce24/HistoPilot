import { formatStatistic, type BootstrapResult, type PatientAnalysis } from '../api/statistics';
import { downloadJSON } from '../lib/download';

export function BootstrapDetails({ value }: { value: Omit<BootstrapResult, 'estimates'> }) {
  return <>
    {value.patientCount != null ? <p>{value.patientCount} labeled patients · {value.resamples} bootstrap resamples · seed {value.seed} · {value.validResamples ?? 0} valid resamples{value.excludedResamples ? ` · ${value.excludedResamples} without all classes excluded` : ''}.</p> : null}
    {!value.available ? <p className="callout">95% confidence intervals unavailable: {value.reason ?? 'Insufficient patient evidence.'}</p> : null}
    {value.degenerate ? <p className="muted">An empirical interval has zero width. This can occur with perfect rankings or identical predictions and does not establish certainty in a new population.</p> : null}
    {value.note ? <p className="muted">{value.note}</p> : null}
  </>;
}

export default function PatientAnalysisResults({ value }: { value?: PatientAnalysis }) {
  if (!value) return null;
  const sensitivity = value.oneSlidePerPatient;
  return <section aria-label="Patient uncertainty and sensitivity"><h4>Patient AUROC and AUPRC · 95% confidence intervals</h4>
    <div className="table-wrap"><table className="chain-table"><thead><tr><th>Analysis</th><th>AUROC (95% CI)</th><th>AUPRC (95% CI)</th></tr></thead><tbody>
      <tr><th scope="row">All eligible slides, aggregated per patient</th>{(['auroc', 'auprc'] as const).map((metric) => <td key={metric}>{formatStatistic(value.uncertainty.estimates?.[metric], value.uncertainty.intervals?.[metric])}</td>)}</tr>
      <tr><th scope="row">One slide per patient</th>{(['auroc', 'auprc'] as const).map((metric) => <td key={metric}>{formatStatistic(sensitivity.metrics?.[metric], sensitivity.uncertainty.intervals?.[metric])}</td>)}</tr>
    </tbody></table></div>
    <BootstrapDetails value={value.uncertainty} />
    <details><summary>One-slide sensitivity analysis</summary><p>One eligible slide is chosen per patient using a hash of patient identity, slide identity and seed {value.policy.oneSlideSeed}. Labels and predictions do not affect selection. The analysis uses the same patient outcomes.</p><BootstrapDetails value={sensitivity.uncertainty} /><button type="button" className="btn btn-secondary" onClick={() => downloadJSON('patient-analysis.json', value)}>Export analysis and selected slide IDs</button></details>
  </section>;
}
