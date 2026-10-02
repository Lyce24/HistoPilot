import { Fragment, useState } from 'react';
import { experimentRoleLabel, objectiveRankLabel, type Objective } from '../api/objectives';
import { groupExperiments, memberSummary, type LibraryEntry } from '../lib/experimentLibrary';
import type { EvaluationExperimentOption } from '../lib/evaluationSelection';
import { inObjectiveScope, UNFILED_SCOPE } from '../lib/objectiveScope';
import { experimentPredictorLink } from '../lib/predictorGroups';
import { shortRecordId } from '../lib/recordLabels';
import { Badge } from './ui';

/**
 * The experiments whose ready predictors to apply. With objectives, the list follows the
 * project's structure: objectives in rank order, roles in order within each, and a series
 * (such as every held-out cohort of leave-one-cohort-out) under one heading that can be
 * selected in one step. The project-wide focus keeps one objective's experiments.
 */
export default function EvaluationExperimentPicker({ options, selected, onChange, disabled, loading, objectives = [], scope = '' }: {
  options: EvaluationExperimentOption[]; selected: string[]; onChange: (ids: string[]) => void; disabled: boolean; loading?: boolean;
  objectives?: readonly Objective[]; scope?: string;
}) {
  const [search, setSearch] = useState('');
  const query = search.trim().toLowerCase();
  const visible = options.filter((item) => (selected.includes(item.id) || inObjectiveScope(scope, item.objectiveId))
    && `${item.name} ${item.id} ${item.description} ${item.series ?? ''} ${item.seriesMember ?? ''} ${item.role ?? ''}`.toLowerCase().includes(query));
  const hidden = selected.filter((id) => !visible.some((item) => item.id === id)).length;
  const toggle = (id: string, checked: boolean) => onChange(checked ? [...selected, id] : selected.filter((value) => value !== id));
  /** Select every experiment of a heading that has a ready predictor, keeping the rest selected. */
  const selectAll = (items: EvaluationExperimentOption[]) => onChange([...new Set([...selected, ...items.filter((item) => item.ready > 0).map((item) => item.id)])]);
  const readyIn = (items: EvaluationExperimentOption[]) => items.filter((item) => item.ready > 0 && !selected.includes(item.id)).length;
  const option = (item: EvaluationExperimentOption, member = false) => <div key={item.id} className={`evaluation-experiment-option${selected.includes(item.id) ? ' selected' : ''}${member ? ' series-member' : ''}`}>
    <label><input type="checkbox" aria-label={`Apply experiment ${item.name} (${item.id})`} checked={selected.includes(item.id)} disabled={disabled} onChange={(event) => toggle(item.id, event.target.checked)} /><span><strong>{member && item.seriesMember ? item.seriesMember : item.name}</strong>{!member && item.role && objectives.length ? <span className={`experiment-role experiment-role-${item.role}`}>{experimentRoleLabel[item.role]}</span> : null}<small>{member && item.seriesMember ? `${item.name} · ` : ''}{item.description}</small><small className="muted">{shortRecordId(item.id)}</small></span></label>
    <a href={experimentPredictorLink(item.id)}>View experiment</a>
  </div>;
  const entry = (row: LibraryEntry<EvaluationExperimentOption>) => row.kind === 'experiment' ? option(row.item) : <div key={row.key} className="evaluation-experiment-series" role="group" aria-label={`Series ${row.series}`}>
    <div className="evaluation-experiment-heading">{row.role ? <span className={`experiment-role experiment-role-${row.role}`}>{experimentRoleLabel[row.role]}</span> : null}<strong>{row.series}</strong><small>{row.items.length} experiments · {memberSummary(row.items)}</small>{readyIn(row.items) ? <button type="button" className="text-button" disabled={disabled} onClick={() => selectAll(row.items)}>Select series</button> : null}</div>
    {row.items.map((item) => option(item, true))}
  </div>;
  const scopedObjectives = scope === UNFILED_SCOPE ? [] : scope ? objectives.filter((item) => item.id === scope) : objectives;
  const groups = objectives.length ? groupExperiments(visible, scopedObjectives, { showEmpty: false }) : null;
  return <section className="evaluation-step" aria-labelledby="evaluation-experiments-title">
    <h3 id="evaluation-experiments-title">1. Select experiments</h3>
    <p className="muted">Choose the experiments whose models to apply. Only their ready predictors are included; training and predictor building continue in Experiments.</p>
    <div className="run-toolbar"><label className="label run-search">Find experiments<input type="search" className="field" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Experiment, series, role, ID or stage" /></label><button className="btn btn-secondary" disabled={disabled || !selected.length} onClick={() => onChange([])}>Clear experiments</button></div>
    <div className="evaluation-experiment-list" role="group" aria-label="Experiments to apply">{groups
      ? groups.map((group) => { const items = [...group.entries, ...group.tuning].flatMap((row) => row.kind === 'series' ? row.items : [row.item]); return <Fragment key={group.key}>
        <div className="evaluation-objective-heading"><strong>{group.objective?.name ?? 'No objective'}</strong>{group.objective ? <Badge tone={group.objective.rank === 'primary' ? 'purple' : 'neutral'}>{objectiveRankLabel[group.objective.rank]}</Badge> : null}<small>{items.length} {items.length === 1 ? 'experiment' : 'experiments'}</small>{readyIn(items) ? <button type="button" className="text-button" disabled={disabled} onClick={() => selectAll(items)}>Select all with ready predictors</button> : null}</div>
        {[...group.entries, ...group.tuning].map(entry)}
      </Fragment>; })
      : visible.map((item) => option(item))}</div>
    {loading ? <p role="status">Loading experiment plans…</p> : !visible.length ? <p>No experiments match this search. <a href="#experiments">Open Experiments</a> to create a plan.</p> : null}
    <p className="muted">{selected.length} {selected.length === 1 ? 'experiment' : 'experiments'} selected{hidden ? ` · ${hidden} hidden by search and still included` : ''}.</p>
  </section>;
}
