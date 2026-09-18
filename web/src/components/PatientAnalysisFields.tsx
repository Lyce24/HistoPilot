import type { TrainingRecipe } from '../api/development';
import { defaultPatientAnalysis } from '../api/statistics';
import NumericField from './NumericField';

export default function PatientAnalysisFields({ value, onChange }: { value: TrainingRecipe; onChange: (recipe: TrainingRecipe) => void }) {
  const analysis = value.analysis;
  return <details className="batch-settings-details setup-details"><summary>Patient analysis &amp; external threshold</summary>
    <p>Report patient AUROC and AUPRC with 95% confidence intervals. One slide per patient is selected by a seeded identity hash, without labels or predictions. These settings are frozen with the training recipe.</p>
    {analysis ? <div className="development-fields">
      <NumericField label="Patient bootstrap resamples" value={analysis.bootstrapResamples} min={200} max={10000} onChange={(bootstrapResamples) => onChange({ ...value, analysis: { ...analysis, bootstrapResamples } })} />
      <NumericField label="Patient bootstrap seed" value={analysis.bootstrapSeed} min={0} max={2 ** 32 - 1} onChange={(bootstrapSeed) => onChange({ ...value, analysis: { ...analysis, bootstrapSeed } })} />
      <NumericField label="One-slide sensitivity seed" value={analysis.oneSlideSeed} min={0} max={2 ** 32 - 1} onChange={(oneSlideSeed) => onChange({ ...value, analysis: { ...analysis, oneSlideSeed } })} />
    </div> : <p>This historical recipe has no patient analysis policy. <button type="button" className="text-button" onClick={() => onChange({ ...value, analysis: defaultPatientAnalysis(), decisionThreshold: value.decisionThreshold ?? 0.5 })}>Enable patient analysis defaults</button></p>}
    {value.decisionThreshold != null ? <NumericField label="Frozen binary decision threshold" value={value.decisionThreshold} integer={false} min={0} minExclusive max={1} maxExclusive onChange={(decisionThreshold) => onChange({ ...value, decisionThreshold })} /> : <p>No threshold was frozen with this historical recipe.</p>}
    <small>Choose thresholds using development evidence. AUROC and AUPRC do not depend on this threshold. Bootstrap analysis runs after fitting; it does not slow each training epoch.</small>
  </details>;
}
