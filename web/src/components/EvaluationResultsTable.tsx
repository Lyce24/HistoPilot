import { Fragment, useState } from 'react';
import { computeStatusLabel, predictorMethodLabel, type ModelEvaluation, type FrozenPredictor } from '../api/predictors';
import type { ModelExperimentSummary } from '../api/experiments';
import type { EvaluationCohort } from '../api/evaluation';
import { lifecycleLabel } from '../api/lifecycle';
import { cleanupLink } from '../lib/hashRoute';
import { versionLabelText } from '../lib/versionLabels';
import { shortRecordId } from '../lib/recordLabels';
import { evaluationMetric, evaluationUnit } from '../lib/evaluationComparison';
import { predictorConfigurationLabel, experimentPredictorLink } from '../lib/predictorGroups';
import EvaluationMethodComparison from './EvaluationMethodComparison';
import { Badge } from './ui';
import './RunWorkspace.css';

export default function EvaluationResultsTable({ records, predictors, experiments = [], cohorts, loading, onOpen }: {
  records: ModelEvaluation[]; predictors: FrozenPredictor[]; experiments?: ModelExperimentSummary[];
  cohorts: EvaluationCohort[]; loading: boolean; onOpen: (id: string) => void;
}) {
  const [search, setSearch] = useState('');
  const [state, setState] = useState('active');
  const [method, setMethod] = useState('all');
  const [cohortId, setCohortId] = useState('');
  const [sort, setSort] = useState('recent');
  const predictorById = new Map(predictors.map((item) => [item.id, item]));
  const cohortById = new Map(cohorts.map((item) => [item.id, versionLabelText(item, 'Test cohort')]));
  const batchById = new Map(experiments.flatMap((item) => item.batches.map((batch) => [JSON.stringify([item.id, batch.id]), batch.name] as const)));
  const experimentName = (item: ModelEvaluation) => predictorById.get(item.manifest.predictorId)?.manifest.experiment?.name ?? shortRecordId(item.manifest.experimentId);
  const cohortName = (id: string) => cohortById.get(id) ?? shortRecordId(id);
  const batchName = (experimentId: string, batchId: string) => batchById.get(JSON.stringify([experimentId, batchId])) ?? `Batch ${shortRecordId(batchId)}`;
  const format = (value: number | null) => value === null ? '—' : value.toFixed(3);
  const retained = records.filter((item) => (state === 'all' || item.lifecycleState === state) && (!cohortId || item.manifest.cohortId === cohortId));
  // Detail filters never change the paired method means above the table.
  const visible = retained.filter((item) => (method === 'all' || (predictorById.get(item.manifest.predictorId)?.manifest.method ?? 'ensemble') === method)
    && `${item.manifest.name} ${predictorById.get(item.manifest.predictorId)?.manifest.name ?? ''} ${experimentName(item)} ${predictorById.get(item.manifest.predictorId)?.manifest.candidateId ?? ''} ${cohortName(item.manifest.cohortId)}`.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => sort === 'name' ? a.manifest.name.localeCompare(b.manifest.name) : b.createdAt.localeCompare(a.createdAt));
  const groups = [...visible.reduce((groups, item) => {
    const unit = evaluationUnit(item), key = JSON.stringify([item.manifest.cohortId, unit]);
    const group = groups.get(key) ?? { name: `${cohortName(item.manifest.cohortId)} · ${unit === 'unavailable' ? 'scores unavailable' : `${unit} scoring`}`, items: [] };
    group.items.push(item); groups.set(key, group); return groups;
  }, new Map<string, { name: string; items: ModelEvaluation[] }>()).entries()];
  return <div className="run-workspace">
    <div className="run-toolbar"><label className="label">Evaluation visibility<select className="field" value={state} onChange={(event) => setState(event.target.value)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label><label className="label">Test cohort filter<select className="field" value={cohortId} onChange={(event) => setCohortId(event.target.value)}><option value="">All cohorts (compared separately)</option>{[...new Set(records.map((item) => item.manifest.cohortId))].map((id) => <option key={id} value={id}>{cohortName(id)}</option>)}</select></label></div>
    {loading ? <p role="status">Loading evaluations…</p> : <EvaluationMethodComparison records={retained} predictors={predictors} cohortName={cohortName} batchName={batchName} onOpen={onOpen} />}
    <h3>Individual evaluations</h3><p className="muted">Row search and method filters affect this table only; method comparisons above keep all matched results in the selected cohort and visibility.</p>
    <div className="run-toolbar"><label className="label run-search">Search evaluations<input className="field" type="search" value={search} onChange={(event) => setSearch(event.target.value)} /></label><label className="label">Evaluation method<select className="field" value={method} onChange={(event) => setMethod(event.target.value)}><option value="all">All methods</option><option value="ensemble">Ensemble</option><option value="refit">Refit</option></select></label><label className="label">Sort evaluations<select className="field" value={sort} onChange={(event) => setSort(event.target.value)}><option value="recent">Recently created</option><option value="name">Name</option></select></label></div>
    <div className="run-kpis"><span><strong>{visible.length}</strong>evaluations</span><span><strong>{visible.filter((item) => item.execution?.status === 'completed').length}</strong>completed</span></div>
    {!loading && !visible.length ? <p>No evaluation records in this view.</p> : visible.length ? <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-name">Evaluation / predictor</th><th>Experiment / batch / configuration</th><th>Method / seeds</th><th>Test cohort / scoring unit</th><th>Status</th><th className="run-number">Accuracy</th><th className="run-number">AUROC</th><th>Actions</th></tr></thead><tbody>{groups.map(([id, group]) => <Fragment key={id}><tr className="run-group"><th colSpan={8}>{group.name}</th></tr>{group.items.map((item) => {
      const source = predictorById.get(item.manifest.predictorId)?.manifest;
      return <tr key={item.id}><td><button className="text-button" onClick={() => onOpen(item.id)}>{item.manifest.name}</button><small>{source?.name ?? shortRecordId(item.manifest.predictorId)}</small></td><td><a href={experimentPredictorLink(item.manifest.experimentId, item.manifest.predictorId)}>{experimentName(item)}</a>{source ? <><small>{batchName(source.experimentId, source.batchId)}</small><small title={source.candidateId}>{predictorConfigurationLabel(source)}</small></> : null}</td><td>{source ? predictorMethodLabel(source.method) : 'Unavailable'}<small>Train {source?.trainingSeed ?? '—'} / split {source?.splitSeed ?? '—'}</small></td><td>{cohortName(item.manifest.cohortId)}<small>{evaluationUnit(item) === 'unavailable' ? 'Scores unavailable' : `${evaluationUnit(item)} scoring`}</small></td><td><Badge>{computeStatusLabel(item.execution)}</Badge><small>{lifecycleLabel[item.lifecycleState]}</small></td><td>{format(evaluationMetric(item, 'accuracy'))}</td><td>{format(evaluationMetric(item, 'auroc'))}</td><td><a href={cleanupLink(item.id)}>{item.lifecycleState === 'active' ? 'Archive / delete' : 'Restore / manage'}</a></td></tr>;
    })}</Fragment>)}</tbody></table></div> : null}
  </div>;
}
