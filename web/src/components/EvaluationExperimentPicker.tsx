import { useState } from 'react';
import type { EvaluationExperimentOption } from '../lib/evaluationSelection';
import { experimentPredictorLink } from '../lib/predictorGroups';
import { shortRecordId } from '../lib/recordLabels';

export default function EvaluationExperimentPicker({ options, selected, onChange, disabled, loading }: {
  options: EvaluationExperimentOption[]; selected: string[]; onChange: (ids: string[]) => void; disabled: boolean; loading?: boolean;
}) {
  const [search, setSearch] = useState('');
  const visible = options.filter((item) => `${item.name} ${item.id} ${item.description}`.toLowerCase().includes(search.trim().toLowerCase()));
  const hidden = selected.filter((id) => !visible.some((item) => item.id === id)).length;
  return <section className="evaluation-step" aria-labelledby="evaluation-experiments-title">
    <h3 id="evaluation-experiments-title">1. Select experiments</h3>
    <p className="muted">Choose the experiments to evaluate. Only their ready predictors are included; training and predictor generation continue in Experiments.</p>
    <div className="run-toolbar"><label className="label run-search">Find experiments<input type="search" className="field" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Experiment name, ID or stage" /></label><button className="btn btn-secondary" disabled={disabled || !selected.length} onClick={() => onChange([])}>Clear experiments</button></div>
    <div className="evaluation-experiment-list" role="group" aria-label="Experiments to evaluate">{visible.map((item) => <div key={item.id} className={`evaluation-experiment-option${selected.includes(item.id) ? ' selected' : ''}`}>
      <label><input type="checkbox" aria-label={`Evaluate experiment ${item.name} (${item.id})`} checked={selected.includes(item.id)} disabled={disabled} onChange={(event) => onChange(event.target.checked ? [...selected, item.id] : selected.filter((id) => id !== item.id))} /><span><strong>{item.name}</strong><small>{item.description}</small><small className="muted">{shortRecordId(item.id)}</small></span></label>
      <a href={experimentPredictorLink(item.id)}>View experiment</a>
    </div>)}</div>
    {loading ? <p role="status">Loading experiment plans…</p> : !visible.length ? <p>No experiments match this search. <a href="#experiments">Open Experiments</a> to create a plan.</p> : null}
    <p className="muted">{selected.length} {selected.length === 1 ? 'experiment' : 'experiments'} selected{hidden ? ` · ${hidden} hidden by search and still included` : ''}.</p>
  </section>;
}
