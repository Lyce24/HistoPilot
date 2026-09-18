import type { FrozenPredictor, ModelEvaluation } from '../api/predictors';
import { compareEvaluationMethods, evaluationMetric, pairedEvaluationMean } from '../lib/evaluationComparison';
import { experimentPredictorLink, predictorConfigurationLabel } from '../lib/predictorGroups';
import { shortRecordId } from '../lib/recordLabels';

const format = (value: number | null) => value === null ? '—' : value.toFixed(3);
export default function EvaluationMethodComparison({ records, predictors, cohortName, batchName, onOpen }: {
  records: ModelEvaluation[]; predictors: FrozenPredictor[]; cohortName: (id: string) => string;
  batchName: (experimentId: string, batchId: string) => string; onOpen: (id: string) => void;
}) {
  const comparisons = compareEvaluationMethods(records, predictors);
  return <section className="evaluation-comparison" aria-labelledby="method-comparison-title"><h3 id="method-comparison-title">Ensemble vs refit</h3>
    <p className="muted">Compare the same experiment, batch, configuration, training seed and split seed on the same test cohort and scoring unit. Means use matched pairs only, with the latest completed evaluation for each method. Unmatched results are counted separately.</p>
    <p className="muted">Freeze the strategy using development evidence before external evaluation. These method means are descriptive; use the paired patient comparison for confidence intervals on model differences.</p>
    {!comparisons.length ? <p>Completed evaluations with available scores will appear here. Evaluate both methods from selected experiments to form matched comparisons.</p> : comparisons.map((group) => {
      const auroc = pairedEvaluationMean(group.pairs, 'auroc'), accuracy = pairedEvaluationMean(group.pairs, 'accuracy');
      return <div key={group.key} className="evaluation-comparison-context" data-cohort={group.cohortId} data-unit={group.unit}>
        <h4>{cohortName(group.cohortId)} · {group.unit} scoring</h4>
        <p className="muted">{group.labeledCount ?? 'Unknown number of'} labeled records · classes {group.classOrder.join(', ')}{group.positiveClass ? ` · positive class ${group.positiveClass}` : ''}{group.threshold !== null ? ` · threshold ${group.threshold}` : ''}{group.unit === 'patient' ? ` · ${group.patientAggregation} aggregation` : ''}</p>
        <p><strong>{group.pairs.length}</strong> matched source {group.pairs.length === 1 ? 'pair' : 'pairs'} · {group.ensembleOnly} ensemble-only · {group.refitOnly} refit-only</p>
        {group.pairs.length ? <><div className="run-table-scroll"><table className="run-table evaluation-comparison-means"><thead><tr><th>Method</th><th>Mean AUROC ({auroc.count} matched {auroc.count === 1 ? 'pair' : 'pairs'})</th><th>Mean accuracy ({accuracy.count} matched {accuracy.count === 1 ? 'pair' : 'pairs'})</th></tr></thead><tbody><tr><th scope="row">Ensemble</th><td>{format(auroc.ensemble)}</td><td>{format(accuracy.ensemble)}</td></tr><tr><th scope="row">Refit</th><td>{format(auroc.refit)}</td><td>{format(accuracy.refit)}</td></tr></tbody></table></div>
          <details><summary>View {group.pairs.length} matched source {group.pairs.length === 1 ? 'pair' : 'pairs'}</summary><div className="run-table-scroll"><table className="run-table"><thead><tr><th>Experiment / batch / configuration</th><th>Training / split seed</th><th>Ensemble AUROC / accuracy</th><th>Refit AUROC / accuracy</th></tr></thead><tbody>{group.pairs.map((pair) => <tr key={pair.key}><td><a href={experimentPredictorLink(pair.source.experimentId)}>{pair.source.experiment?.name ?? shortRecordId(pair.source.experimentId)}</a><small>{batchName(pair.source.experimentId, pair.source.batchId)} · {predictorConfigurationLabel(pair.source)}</small></td><td>{pair.source.trainingSeed} / {pair.source.splitSeed}</td><td><button className="text-button" onClick={() => onOpen(pair.ensemble!.id)}>{format(evaluationMetric(pair.ensemble!, 'auroc'))} / {format(evaluationMetric(pair.ensemble!, 'accuracy'))}</button></td><td><button className="text-button" onClick={() => onOpen(pair.refit!.id)}>{format(evaluationMetric(pair.refit!, 'auroc'))} / {format(evaluationMetric(pair.refit!, 'accuracy'))}</button></td></tr>)}</tbody></table></div></details></> : <p className="muted">No matching ensemble/refit pair in this scoring context. Unmatched results are excluded from method means.</p>}
      </div>;
    })}
  </section>;
}
