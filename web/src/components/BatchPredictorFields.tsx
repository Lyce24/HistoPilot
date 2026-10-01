import { useId, useState } from 'react';
import type { ExperimentPredictorPolicy } from '../api/experiments';
import { includesRefit, predictorPolicyLabel } from '../lib/experimentPredictors';
import { assessmentPlanNoun } from '../lib/split';
import NumericField from './NumericField';
import './ExperimentPredictors.css';

const choices: { method: ExperimentPredictorPolicy['method']; title: string; description: string }[] = [
  { method: 'ensemble', title: 'Ensemble', description: 'Average predictions from trained fold checkpoints. No extra model training.' },
  { method: 'refit', title: 'Refit', description: 'Train a new model on all development data after its folds finish.' },
  { method: 'both', title: 'Ensemble + refit', description: 'Create both methods to compare on the same cohort in Apply models. Adds refit training.' },
  { method: 'skip', title: 'Cross-validation only', description: 'Finish with development results. No predictor is created for Apply models.' },
];

export function batchPredictorLabel(policy: ExperimentPredictorPolicy) {
  return `${predictorPolicyLabel(policy)}${includesRefit(policy) ? policy.refitPercentile === null ? ' · Refit epoch budget not set' : ` · P${policy.refitPercentile} refit epochs` : ''}`;
}

/** Predictor settings are saved atomically with the surrounding batch form. */
export default function BatchPredictorFields({ value, onChange, configurationCount, trainingSeedCount, splitSeedCount, foldCount, split }: {
  value: ExperimentPredictorPolicy; onChange: (value: ExperimentPredictorPolicy) => void;
  configurationCount: number | null; trainingSeedCount: number | null; splitSeedCount?: number; foldCount?: number;
  /** The experiment's training design, which names its assessment runs. */
  split?: { mode?: string } | null;
}) {
  const noun = assessmentPlanNoun(split);
  const radioId = useId();
  const [lastPercentile, setLastPercentile] = useState(value.refitPercentile ?? 50);
  const [custom, setCustom] = useState(value.refitPercentile != null && ![50, 75, 90, 100].includes(value.refitPercentile));
  const refit = includesRefit(value);
  const knownGroups = configurationCount === null || trainingSeedCount === null ? null : configurationCount * trainingSeedCount;
  const groups = knownGroups === null ? null : knownGroups * (splitSeedCount ?? 1);
  const methods = value.method === 'skip' ? 0 : value.method === 'both' ? 2 : 1;
  function percentile(next: number) {
    setLastPercentile(next);
    onChange({ ...value, refitPercentile: next });
  }
  return <section className="batch-editor-section batch-predictor-settings" aria-label="Configure predictors">
    <div className="batch-section-heading"><h3>Configure predictors</h3><p>Choose what this batch creates after its folds finish. Other batches can use different choices.</p></div>
    <fieldset className="experiment-predictor-fields"><legend className="sr-only">Predictor choices for this batch</legend>
      <div className="experiment-predictor-options">{choices.map((choice) => <label className={`experiment-predictor-option${value.method === choice.method ? ' selected' : ''}`} key={choice.method}>
        <input type="radio" name={radioId} value={choice.method} checked={value.method === choice.method} onChange={() => onChange({ method: choice.method, refitPercentile: choice.method === 'refit' || choice.method === 'both' ? lastPercentile : null })} />
        <span><strong>{choice.title}</strong><small>{choice.description}</small></span>
      </label>)}</div>
      {refit ? <div className="experiment-refit-budget">
        <label className="label">Refit epoch budget<select className="field" value={custom ? 'custom' : String(value.refitPercentile ?? lastPercentile)} onChange={(event) => { setCustom(event.target.value === 'custom'); if (event.target.value !== 'custom') percentile(Number(event.target.value)); }}>
          <option value="50">P50 · median</option><option value="75">P75 · 75th percentile</option><option value="90">P90 · 90th percentile</option><option value="100">P100 · longest fold</option><option value="custom">Custom percentile</option>
        </select></label>
        {custom ? <NumericField label="Custom percentile (1–100)" value={value.refitPercentile ?? lastPercentile} integer={false} min={1} max={100} onChange={percentile} /> : null}
        <p>{foldCount === 1 ? 'With one assessment run per group, every percentile gives the same budget: the checkpoint epoch that run selected.' : `Each refit uses the selected percentile of the checkpoint epochs its group’s ${noun.many} selected, rounded up.`} Checkpoints follow the batch’s evaluation policy. It trains on all training data for that budget, with early stopping disabled. The testing set stays out of training.</p>
      </div> : null}
    </fieldset>
    {groups !== null ? <div className="experiment-predictor-count" aria-live="polite">
      <strong>{(groups * methods).toLocaleString()} predictors {splitSeedCount === undefined ? 'per split seed' : 'planned for this batch'}</strong>
      <span>{configurationCount} configuration{configurationCount === 1 ? '' : 's'} × {trainingSeedCount} training seed{trainingSeedCount === 1 ? '' : 's'}{splitSeedCount === undefined ? '' : ` × ${splitSeedCount} split seed${splitSeedCount === 1 ? '' : 's'}`} × {methods} predictor method{methods === 1 ? '' : 's'}</span>
      {splitSeedCount !== undefined && foldCount !== undefined ? <span>{groups.toLocaleString()} {groups === 1 ? 'group' : 'groups'} · {(groups * foldCount).toLocaleString()} {foldCount === 1 ? 'assessment' : noun.one} {groups * foldCount === 1 ? 'run' : 'runs'}</span> : <span>The total also depends on the split seeds saved in Inputs. Check batch to confirm the frozen groups.</span>}
      {refit ? <span>{groups.toLocaleString()} additional refit training run{groups === 1 ? '' : 's'}{splitSeedCount === undefined ? ' per split seed' : ''} after cross-validation.</span> : null}
    </div> : <p className="muted">Complete the parameter values and training seeds to see the planned predictor count.</p>}
    <p className="muted">{value.method === 'skip' ? 'This batch finishes after cross-validation. To create predictors later, copy it into a new experiment and change this choice.' : 'Predictors are created automatically. Refits add training work to the Task Center queue after the source folds finish.'}</p>
  </section>;
}
