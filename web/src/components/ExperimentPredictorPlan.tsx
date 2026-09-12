import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { experiments, type ExperimentPredictorPolicy, type ModelExperiment } from '../api/experiments';
import type { ProtocolSpec } from '../api/scientific';
import { sameJSON } from '../lib/json';
import { defaultPredictorPolicy, experimentPredictorCount, includesRefit, predictorPolicyLabel } from '../lib/experimentPredictors';
import { ErrorNotice, Panel } from './ui';
import { parseNumericField } from './NumericField';
import './ExperimentPredictors.css';

const choices: { method: ExperimentPredictorPolicy['method']; description: string }[] = [
  { method: 'skip', description: 'Run cross-validation only. No predictors will be created.' },
  { method: 'refit', description: 'Train one model on all development data using an epoch budget from the folds.' },
  { method: 'ensemble', description: 'Keep all fold checkpoints together and average their prediction probabilities.' },
  { method: 'both', description: 'Create an ensemble and a full-development refit for each complete k-fold group.' },
];

export default function ExperimentPredictorPlan({ project, record, protocol, readOnly, batchDirty, onDirtyChange }: {
  project: string; record: ModelExperiment; protocol?: ProtocolSpec; readOnly: boolean; batchDirty: boolean; onDirtyChange: (dirty: boolean) => void;
}) {
  const client = useQueryClient();
  const saved = record.predictorPolicy ?? defaultPredictorPolicy();
  const [base, setBase] = useState(saved);
  const [draft, setDraft] = useState(saved);
  const [percentileText, setPercentileText] = useState(String(saved.refitPercentile ?? 50));
  const [custom, setCustom] = useState(saved.refitPercentile != null && ![50, 75, 90, 100].includes(saved.refitPercentile));
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const stale = !sameJSON(base, saved);
  const percentile = Number(percentileText);
  const invalid = includesRefit(draft) && Boolean(parseNumericField(percentileText, { label: 'Percentile', integer: false, min: 1, max: 100 }).error);
  const editedPolicy = { ...draft, refitPercentile: includesRefit(draft) ? percentile : null };
  const dirty = !readOnly && (invalid || !sameJSON(editedPolicy, saved));
  const policy = readOnly ? saved : editedPolicy;
  const count = experimentPredictorCount(record, policy, protocol);
  useEffect(() => { onDirtyChange(dirty || (!readOnly && stale) || busy); }, [dirty, stale, readOnly, busy, onDirtyChange]);

  function reset() {
    setDraft(saved); setBase(saved); setPercentileText(String(saved.refitPercentile ?? 50));
    setCustom(saved.refitPercentile != null && ![50, 75, 90, 100].includes(saved.refitPercentile)); setError(null); setMessage('');
  }
  async function save() {
    if (inFlight.current || readOnly || stale || invalid || batchDirty || !dirty) return;
    inFlight.current = true; setBusy(true); setError(null); setMessage('');
    try {
      const updated = await experiments.update(project, record.id, { name: record.name, notes: record.notes, tags: record.tags, expectedRevision: record.revision, predictorPolicy: editedPolicy });
      const next = updated.predictorPolicy ?? editedPolicy;
      setDraft(next); setBase(next);
      client.setQueryData(['model-experiment', project, record.id], updated);
      setMessage('Predictor choices saved. They will be frozen when you submit this experiment.');
      void client.invalidateQueries({ queryKey: ['model-experiments', project] });
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('Could not save predictor choices.'));
      void client.invalidateQueries({ queryKey: ['model-experiment', project, record.id] });
    } finally { inFlight.current = false; setBusy(false); }
  }

  return <Panel title="Predictors from this experiment" subtitle="Choose once for all batches. Each configuration, training seed and split seed defines one k-fold group.">
    {readOnly && !record.predictorPolicy ? <p className="callout">This historical experiment has no automatic predictor plan. Its existing predictors remain available in Results.</p> : <>
      <fieldset className="experiment-predictor-fields" disabled={readOnly || busy || stale}>
        <legend className="sr-only">Predictor choices</legend>
        <div className="experiment-predictor-options">{choices.map((choice) => <label className={`experiment-predictor-option${policy.method === choice.method ? ' selected' : ''}`} key={choice.method}>
          <input type="radio" name={`predictor-policy-${record.id}`} value={choice.method} checked={policy.method === choice.method} onChange={() => { setDraft({ method: choice.method, refitPercentile: ['refit', 'both'].includes(choice.method) ? percentile : null }); setMessage(''); }} />
          <span><strong>{predictorPolicyLabel({ method: choice.method, refitPercentile: null })}</strong><small>{choice.description}</small></span>
        </label>)}</div>
        {includesRefit(policy) ? <div className="experiment-refit-budget">
          <label className="label">Refit epoch budget
            <select className="field" value={custom ? 'custom' : String(readOnly ? saved.refitPercentile : percentile)} onChange={(event) => { setCustom(event.target.value === 'custom'); if (event.target.value !== 'custom') setPercentileText(event.target.value); setMessage(''); }}>
              <option value="50">P50 · median</option><option value="75">P75 · 75th percentile</option><option value="90">P90 · 90th percentile</option><option value="100">P100 · longest fold</option><option value="custom">Custom percentile</option>
            </select>
          </label>
          {custom ? <label className="label">Custom percentile (1–100)<input className="field" inputMode="decimal" value={readOnly ? String(saved.refitPercentile ?? 50) : percentileText} aria-invalid={invalid || undefined} onChange={(event) => { setPercentileText(event.target.value); setMessage(''); }} /></label> : null}
          <p>The selected percentile of the folds’ best checkpoint epochs is rounded up to a whole epoch. Each refit trains on all development data for that budget, with early stopping disabled. Test cohorts are kept out of training.</p>
          {invalid && !readOnly ? <p role="alert" className="development-field-error">Enter a percentile from 1 to 100 before saving.</p> : null}
        </div> : null}
      </fieldset>
      {count ? <div className="experiment-predictor-count" aria-live="polite"><strong>{count.total.toLocaleString()} predictors planned</strong><span>{count.groups.toLocaleString()} k-fold groups · {count.foldRuns.toLocaleString()} fold runs</span><span>{count.ensembles.toLocaleString()} ensembles + {count.refits.toLocaleString()} refits</span></div> : <p className="muted">The predictor count appears after saving k-fold inputs and batches.</p>}
      {policy.method === 'skip' ? <p className="muted">The experiment finishes after cross-validation. To create predictors later, copy this experiment as a new plan and choose a predictor method.</p> : <p className="muted">Predictors are created automatically after their source folds finish. Refits add training jobs using each batch’s saved resources; the experiment stays Running until predictor work finishes.</p>}
      {readOnly ? <p className="muted">Predictor choices are frozen with this experiment.</p> : <>
        {stale ? <p className="callout">Predictor choices changed since you opened this plan. <button className="text-button" onClick={reset} disabled={busy}>Reload predictor choices</button></p> : null}
        {batchDirty && dirty ? <p className="muted">Save or discard batch edits before saving predictor choices.</p> : null}
        <div className="inline-actions"><button className="btn btn-secondary" disabled={busy || stale || invalid || batchDirty || !dirty} onClick={() => void save()}>{busy ? 'Saving predictor choices…' : 'Save predictor choices'}</button>{dirty ? <button className="text-button" disabled={busy} onClick={reset}>Discard predictor edits</button> : null}</div>
      </>}
      <ErrorNotice error={error} />{message ? <p className="science-success" role="status">{message}</p> : null}
    </>}
  </Panel>;
}
