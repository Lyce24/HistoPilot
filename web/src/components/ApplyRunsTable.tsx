import { Fragment, useState, type ReactNode } from 'react';
import type { EvaluationCohort } from '../api/evaluation';
import type { ModelExperimentSummary } from '../api/experiments';
import { lifecycleLabel } from '../api/lifecycle';
import { computeStatusLabel, type FrozenPredictor, type ModelEvaluation } from '../api/predictors';
import { referenceFits, type ReferenceStandard } from '../api/references';
import { formatStatistic } from '../api/statistics';
import { batchNameLookup, cohortKindText, cohortLabeled, cohortName, modelDescription, runLabeled, unlabeledCohortText } from '../lib/applyModels';
import { isUnitSummary, percent, predictedMix, unanimousShare } from '../lib/inference';
import { experimentPredictorLink } from '../lib/predictorGroups';
import { shortRecordId } from '../lib/recordLabels';
import { StageCreateButton } from './StageActions';
import { StageLibraryToolbar, StageRecordManageButton } from './StageWorkflow';
import { Badge, EmptyState } from './ui';
import './RunWorkspace.css';

/** `experimentId` and `predictorId` scope the list to one source, as a link from it asks. */
export interface RunFilters { search: string; state: string; cohortId: string; kind: string; method: string; sort: string; experimentId: string; predictorId: string }
export const newRunFilters = (scope: { experiment?: string; predictor?: string } = {}): RunFilters => ({ search: '', state: 'active', cohortId: '', kind: 'all', method: 'all', sort: 'recent', experimentId: scope.experiment ?? '', predictorId: scope.predictor ?? '' });

/** What a run found: scored metrics on a labeled cohort, the predicted mix on an unlabeled one. */
export function RunResult({ run }: { run: ModelEvaluation }) {
  const result = run.execution?.status === 'completed' ? run.execution.result : null;
  if (!result) return <>—</>;
  if (runLabeled(run)) {
    const selected = result.metrics?.selected;
    if (!selected?.available) return <>{result.metricsError ? 'Not scored' : '—'}{result.metricsError ? <small>{result.metricsError.message}</small> : null}</>;
    return <>AUROC {formatStatistic(selected.auroc, selected.confidenceIntervals?.auroc)}<small>Accuracy {formatStatistic(selected.accuracy)} · {selected.count?.toLocaleString()} {result.metrics?.unit}s scored</small></>;
  }
  const summary = result.summary;
  const selected = summary && isUnitSummary(summary.selected) ? summary.selected : null;
  return <>{predictedMix(selected)}{unanimousShare(selected) !== null ? <small>{percent(unanimousShare(selected), 0)} of {summary?.unit}s unanimous</small> : null}</>;
}

/** Every run in one table: the model it applied, the cohort, and what it found. */
export default function ApplyRunsTable({ runs, predictors, experiments, cohorts, references = [], loading, onOpen, onCreate, filters, onFiltersChange, actions }: {
  runs: ModelEvaluation[]; predictors: FrozenPredictor[]; experiments: ModelExperimentSummary[]; cohorts: EvaluationCohort[];
  /** The project's reference standards: labels added to cohorts after they were frozen. */
  references?: ReferenceStandard[];
  loading: boolean; onOpen: (id: string) => void; onCreate: () => void;
  filters?: RunFilters; onFiltersChange?: (filters: RunFilters) => void; actions?: ReactNode;
}) {
  const [localFilters, setLocalFilters] = useState(newRunFilters);
  const current = filters ?? localFilters;
  const update = (change: Partial<RunFilters>) => (onFiltersChange ?? setLocalFilters)({ ...current, ...change });
  const reset = () => (onFiltersChange ?? setLocalFilters)(newRunFilters());
  const predictorById = new Map(predictors.map((item) => [item.id, item.manifest]));
  const cohortById = new Map(cohorts.map((item) => [item.id, item]));
  const batchName = batchNameLookup(experiments);
  const nameOf = (id: string) => cohortName(cohortById.get(id), id);
  const activeReferences = references.filter((item) => (item.lifecycleState ?? 'active') === 'active');
  // A cohort's references, or those that label one run's classes.
  const referenceCount = (cohortId: string, classes?: readonly string[]) => activeReferences.filter((item) => item.manifest.cohortId === cohortId && (!classes || referenceFits(item, cohortId, classes))).length;
  const experimentName = (run: ModelEvaluation) => predictorById.get(run.manifest.predictorId)?.experiment?.name ?? shortRecordId(run.manifest.experimentId);
  const visible = runs.filter((run) => {
    const source = predictorById.get(run.manifest.predictorId);
    return (!current.experimentId || run.manifest.experimentId === current.experimentId)
      && (!current.predictorId || run.manifest.predictorId === current.predictorId)
      && (current.state === 'all' || run.lifecycleState === current.state)
      && (!current.cohortId || run.manifest.cohortId === current.cohortId)
      && (current.kind === 'all' || (current.kind === 'labeled') === runLabeled(run))
      && (current.method === 'all' || (source?.method ?? 'ensemble') === current.method)
      && `${run.manifest.name} ${run.id} ${experimentName(run)} ${source ? modelDescription(source, batchName) : ''} ${nameOf(run.manifest.cohortId)}`.toLowerCase().includes(current.search.trim().toLowerCase());
  }).sort((a, b) => (current.sort === 'name' ? a.manifest.name.localeCompare(b.manifest.name) : current.sort === 'oldest' ? a.createdAt.localeCompare(b.createdAt) : b.createdAt.localeCompare(a.createdAt)) || a.id.localeCompare(b.id));
  const groups = [...visible.reduce((map, run) => {
    const group = map.get(run.manifest.cohortId) ?? [];
    group.push(run); map.set(run.manifest.cohortId, group); return map;
  }, new Map<string, ModelEvaluation[]>()).entries()];
  const filtered = Boolean(current.search || current.state !== 'active' || current.cohortId || current.kind !== 'all' || current.method !== 'all' || current.sort !== 'recent' || current.experimentId || current.predictorId);
  const scoped = current.predictorId ? predictorById.get(current.predictorId) : undefined;
  const scope = current.predictorId ? scoped ? `${scoped.experiment?.name ?? shortRecordId(scoped.experimentId)} · ${modelDescription(scoped, batchName)}` : `predictor ${shortRecordId(current.predictorId)}`
    : current.experimentId ? experiments.find((item) => item.id === current.experimentId)?.name ?? `experiment ${shortRecordId(current.experimentId)}` : '';
  return <div className="run-workspace apply-runs">
    <StageLibraryToolbar search={current.search} onSearch={(search) => update({ search })} searchLabel="Search runs" placeholder="Name, experiment, model or cohort" count={loading ? undefined : visible.length} total={runs.length} actions={actions} onReset={filtered ? reset : undefined}>
      <label className="label">State<select className="field" aria-label="Run state" value={current.state} onChange={(event) => update({ state: event.target.value })}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label>
      <label className="label">Cohort<select className="field" aria-label="Cohort filter" value={current.cohortId} onChange={(event) => update({ cohortId: event.target.value })}><option value="">All cohorts</option>{[...new Set(runs.map((run) => run.manifest.cohortId))].map((id) => <option key={id} value={id}>{nameOf(id)}</option>)}</select></label>
      <label className="label">Labels<select className="field" aria-label="Cohort labels" value={current.kind} onChange={(event) => update({ kind: event.target.value })}><option value="all">Labeled and unlabeled cohorts</option><option value="labeled">Labeled cohort</option><option value="unlabeled">Unlabeled cohort</option></select></label>
      <label className="label">Method<select className="field" aria-label="Predictor method" value={current.method} onChange={(event) => update({ method: event.target.value })}><option value="all">All methods</option><option value="seed_ensemble">Seed ensemble</option><option value="ensemble">Fold ensemble</option><option value="refit">Refit</option></select></label>
      <label className="label">Sort<select className="field" aria-label="Sort runs" value={current.sort} onChange={(event) => update({ sort: event.target.value })}><option value="recent">Newest first</option><option value="oldest">Oldest first</option><option value="name">Name</option></select></label>
    </StageLibraryToolbar>
    {scope ? <p className="callout apply-runs-scope">Showing the runs of {scope}. <button type="button" className="text-button" onClick={() => update({ experimentId: '', predictorId: '' })}>Show all runs</button></p> : null}
    {loading ? <p role="status">Loading runs…</p> : !visible.length ? <EmptyState icon="evaluation"
      title={runs.length ? 'No matching runs' : 'No models applied yet'}
      description={runs.length ? 'Try another search or clear the filters.' : 'Apply ready predictors to a cohort: labeled cohorts are scored, unlabeled ones get predictions only.'}
      action={runs.length ? <button type="button" className="btn btn-secondary" onClick={reset}>Clear filters</button> : <StageCreateButton onClick={onCreate}>Apply predictors</StageCreateButton>} />
      : <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-name">Run</th><th>Model</th><th>Cohort</th><th>Status</th><th>Result</th><th>Actions</th></tr></thead><tbody>
        {groups.map(([cohortId, items]) => { const cohort = cohortById.get(cohortId); return <Fragment key={cohortId}>
          {groups.length > 1 ? <tr className="run-group"><th colSpan={6}>{nameOf(cohortId)}{cohort ? ` · ${cohortKindText(cohort, referenceCount(cohortId))}` : ''}</th></tr> : null}
          {items.map((run) => { const source = predictorById.get(run.manifest.predictorId); return <tr key={run.id}>
            <td><button type="button" className="text-button stage-record-name" onClick={() => onOpen(run.id)}>{run.manifest.name}</button>{run.manifest.purpose === 'review' ? <small>Earlier slide-review run</small> : null}</td>
            <td><a href={experimentPredictorLink(run.manifest.experimentId, run.manifest.predictorId)}>{experimentName(run)}</a><small>{source ? modelDescription(source, batchName) : 'Predictor unavailable'}</small></td>
            <td>{nameOf(run.manifest.cohortId)}<small>{(cohort ? cohortLabeled(cohort) : runLabeled(run)) ? 'Labeled · scored' : unlabeledCohortText(referenceCount(run.manifest.cohortId, run.manifest.target?.classes))}</small></td>
            <td><Badge>{computeStatusLabel(run.execution)}</Badge>{run.lifecycleState !== 'active' ? <small>{lifecycleLabel[run.lifecycleState]}</small> : null}</td>
            <td><RunResult run={run} /></td>
            <td><StageRecordManageButton type="configuration" id={run.id} name={run.manifest.name} /></td>
          </tr>; })}
        </Fragment>; })}
      </tbody></table></div>}
  </div>;
}
