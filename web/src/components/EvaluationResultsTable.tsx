import { Fragment, useState, type ReactNode } from 'react';
import { computeStatusLabel, predictorMethodLabel, type ModelEvaluation, type FrozenPredictor } from '../api/predictors';
import type { ModelExperimentSummary } from '../api/experiments';
import type { EvaluationCohort } from '../api/evaluation';
import { lifecycleLabel } from '../api/lifecycle';
import { versionLabelText } from '../lib/versionLabels';
import { shortRecordId } from '../lib/recordLabels';
import { evaluationMetric, evaluationUnit } from '../lib/evaluationComparison';
import { predictorConfigurationLabel, experimentPredictorLink } from '../lib/predictorGroups';
import EvaluationMethodComparison from './EvaluationMethodComparison';
import { Badge, EmptyState } from './ui';
import { StageLibraryToolbar, StageRecordManageButton } from './StageWorkflow';
import './RunWorkspace.css';

export interface EvaluationRecordFilters { search: string; state: string; method: string; cohortId: string; sort: string }
export const newEvaluationRecordFilters = (): EvaluationRecordFilters => ({ search: '', state: 'active', method: 'all', cohortId: '', sort: 'recent' });

export default function EvaluationResultsTable({ records, predictors, experiments = [], cohorts, loading, onOpen, view = 'all', filters, onFiltersChange, actions }: {
  records: ModelEvaluation[]; predictors: FrozenPredictor[]; experiments?: ModelExperimentSummary[];
  cohorts: EvaluationCohort[]; loading: boolean; onOpen: (id: string) => void; view?: 'all' | 'records' | 'comparison';
  filters?: EvaluationRecordFilters; onFiltersChange?: (value: EvaluationRecordFilters) => void; actions?: ReactNode;
}) {
  const [localFilters, setLocalFilters] = useState(newEvaluationRecordFilters);
  const currentFilters = filters ?? localFilters;
  const { search, state, method, cohortId, sort } = currentFilters;
  const updateFilters = (update: Partial<EvaluationRecordFilters>) => (onFiltersChange ?? setLocalFilters)({ ...currentFilters, ...update });
  const resetFilters = () => (onFiltersChange ?? setLocalFilters)(newEvaluationRecordFilters());
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
    && `${item.manifest.name} ${item.id} ${item.manifest.predictorId} ${predictorById.get(item.manifest.predictorId)?.manifest.name ?? ''} ${experimentName(item)} ${predictorById.get(item.manifest.predictorId)?.manifest.candidateId ?? ''} ${cohortName(item.manifest.cohortId)}`.toLowerCase().includes(search.trim().toLowerCase()))
    .sort((a, b) => (sort === 'name' ? a.manifest.name.localeCompare(b.manifest.name) : sort === 'oldest' ? a.createdAt.localeCompare(b.createdAt) : b.createdAt.localeCompare(a.createdAt)) || a.id.localeCompare(b.id));
  const groups = [...visible.reduce((groups, item) => {
    const unit = evaluationUnit(item), key = JSON.stringify([item.manifest.cohortId, unit]);
    const group = groups.get(key) ?? { name: `${cohortName(item.manifest.cohortId)} · ${unit === 'unavailable' ? 'scores unavailable' : `${unit} scoring`}`, items: [] };
    group.items.push(item); groups.set(key, group); return groups;
  }, new Map<string, { name: string; items: ModelEvaluation[] }>()).entries()];
  const stateFilter = <label className="label">State<select className="field" aria-label="Evaluation visibility" value={state} onChange={(event) => updateFilters({ state: event.target.value })}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label>;
  const cohortFilter = <label className="label">Test cohort<select className="field" aria-label="Test cohort filter" value={cohortId} onChange={(event) => updateFilters({ cohortId: event.target.value })}><option value="">All cohorts</option>{[...new Set(records.map((item) => item.manifest.cohortId))].map((id) => <option key={id} value={id}>{cohortName(id)}</option>)}</select></label>;
  return <div className="run-workspace evaluation-record-library">
    {view === 'comparison' ? <div className="run-toolbar">{stateFilter}{cohortFilter}</div> : <StageLibraryToolbar search={search} onSearch={(value) => updateFilters({ search: value })} searchLabel="Search evaluations" placeholder="Name, ID, experiment or cohort" count={loading ? undefined : visible.length} total={records.length} actions={actions}
      onReset={search || state !== 'active' || method !== 'all' || cohortId || sort !== 'recent' ? resetFilters : undefined}>
      {stateFilter}{cohortFilter}
      <label className="label">Method<select className="field" aria-label="Evaluation method" value={method} onChange={(event) => updateFilters({ method: event.target.value })}><option value="all">All methods</option><option value="ensemble">Ensemble</option><option value="refit">Refit</option></select></label>
      <label className="label">Sort<select className="field" aria-label="Sort evaluations" value={sort} onChange={(event) => updateFilters({ sort: event.target.value })}><option value="recent">Newest first</option><option value="oldest">Oldest first</option><option value="name">Name</option></select></label>
    </StageLibraryToolbar>}
    {loading ? <p role="status">Loading evaluations…</p> : null}
    {view !== 'records' && !loading ? <EvaluationMethodComparison records={retained} predictors={predictors} cohortName={cohortName} batchName={batchName} onOpen={onOpen} /> : null}
    {view !== 'comparison' ? <>
    {!loading && !visible.length ? <EmptyState title={records.length ? "No matching evaluations" : "No evaluations yet"} description={records.length ? "Try another search or clear the filters." : "Create an evaluation to test your saved models."} /> : visible.length ? <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-name">Evaluation</th><th>Experiment</th><th>Method</th><th>Test cohort</th><th>Status</th><th className="run-number">Accuracy</th><th className="run-number">AUROC</th><th>Actions</th></tr></thead><tbody>{groups.map(([id, group]) => <Fragment key={id}>{groups.length > 1 ? <tr className="run-group"><th colSpan={8}>{group.name}</th></tr> : null}{group.items.map((item) => {
      const source = predictorById.get(item.manifest.predictorId)?.manifest;
      return <tr key={item.id}><td><button type="button" className="text-button stage-record-name" title={`Open ${item.manifest.name} · Predictor: ${source?.name ?? item.manifest.predictorId}`} onClick={() => onOpen(item.id)}>{item.manifest.name}</button></td><td><a href={experimentPredictorLink(item.manifest.experimentId, item.manifest.predictorId)}>{experimentName(item)}</a>{source ? <><small>{batchName(source.experimentId, source.batchId)}</small><small title={source.candidateId}>{predictorConfigurationLabel(source)}</small></> : null}</td><td>{source ? predictorMethodLabel(source.method) : 'Unavailable'}<small>Train {source?.trainingSeed ?? '—'} / split {source?.splitSeed ?? '—'}</small></td><td>{cohortName(item.manifest.cohortId)}<small>{evaluationUnit(item) === 'unavailable' ? 'Scores unavailable' : `${evaluationUnit(item)} scoring`}</small></td><td><Badge>{computeStatusLabel(item.execution)}</Badge>{item.lifecycleState !== 'active' ? <small>{lifecycleLabel[item.lifecycleState]}</small> : null}</td><td>{format(evaluationMetric(item, 'accuracy'))}</td><td>{format(evaluationMetric(item, 'auroc'))}</td><td><StageRecordManageButton type="configuration" id={item.id} name={item.manifest.name} /></td></tr>;
    })}</Fragment>)}</tbody></table></div> : null}
    </> : null}
  </div>;
}
