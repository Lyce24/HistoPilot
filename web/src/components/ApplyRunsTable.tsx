import { Fragment, useState, type ReactNode } from 'react';
import type { EvaluationCohort } from '../api/evaluation';
import type { ModelExperimentSummary } from '../api/experiments';
import { lifecycleLabel } from '../api/lifecycle';
import { computeStatusLabel, type FrozenPredictor, type ModelEvaluation } from '../api/predictors';
import { referenceFits, type ReferenceStandard } from '../api/references';
import { formatStatistic } from '../api/statistics';
import { batchNameLookup, cohortKindText, cohortLabeled, cohortName, modelDescription, runLabeled, unlabeledCohortText } from '../lib/applyModels';
import { isUnitSummary, percent, predictedMix, unanimousShare } from '../lib/inference';
import { experimentRoleLabel, experimentRoles, objectiveRankLabel, type ExperimentFiling, type Objective } from '../api/objectives';
import { sectionByObjective } from '../lib/experimentLibrary';
import { inObjectiveScope } from '../lib/objectiveScope';
import { experimentPredictorLink } from '../lib/predictorGroups';
import { shortRecordId } from '../lib/recordLabels';
import { ObjectiveScopeNotice } from './ObjectiveScope';
import { StageCreateButton } from './StageActions';
import { StageLibraryToolbar, StageRecordManageButton } from './StageWorkflow';
import { Badge, EmptyState, LoadFailed } from './ui';
import './RunWorkspace.css';

/** `experimentId` and `predictorId` scope the list to one source, as a link from it asks.
 * `group` sections the runs by the objective their experiment answers, or by cohort. */
export interface RunFilters { search: string; state: string; cohortId: string; kind: string; method: string; sort: string; experimentId: string; predictorId: string; group?: 'objective' | 'cohort' }
export const newRunFilters = (scope: { experiment?: string; predictor?: string } = {}): RunFilters => ({ search: '', state: 'active', cohortId: '', kind: 'all', method: 'all', sort: 'recent', experimentId: scope.experiment ?? '', predictorId: scope.predictor ?? '', group: 'objective' });
const roleRank = (filing?: ExperimentFiling) => filing?.role ? experimentRoles.indexOf(filing.role) : experimentRoles.length;

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
export default function ApplyRunsTable({ runs, predictors, experiments, cohorts, references = [], objectives = [], filings = {}, objectiveScope = '', onClearScope, loading, error = null, onRetry, onOpen, onCreate, filters, onFiltersChange, actions }: {
  runs: ModelEvaluation[]; predictors: FrozenPredictor[]; experiments: ModelExperimentSummary[]; cohorts: EvaluationCohort[];
  /** The project's reference standards: labels added to cohorts after they were frozen. */
  references?: ReferenceStandard[];
  /** The project's objectives and each experiment's filing, to section runs by objective. */
  objectives?: readonly Objective[]; filings?: Record<string, ExperimentFiling>;
  /** The project-wide objective focus (lib/objectiveScope). */
  objectiveScope?: string; onClearScope?: () => void;
  loading: boolean; onOpen: (id: string) => void; onCreate: () => void;
  /** Why the runs could not be read, if they could not, and how to read them again. */
  error?: Error | null; onRetry?: () => void;
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
  const experimentName = (run: ModelEvaluation) => predictorById.get(run.manifest.predictorId)?.experiment?.name ?? experiments.find((item) => item.id === run.manifest.experimentId)?.name ?? shortRecordId(run.manifest.experimentId);
  const filingOf = (run: ModelEvaluation) => filings[run.manifest.experimentId];
  const matching = runs.filter((run) => {
    const source = predictorById.get(run.manifest.predictorId);
    return (!current.experimentId || run.manifest.experimentId === current.experimentId)
      && (!current.predictorId || run.manifest.predictorId === current.predictorId)
      && (current.state === 'all' || run.lifecycleState === current.state)
      && (!current.cohortId || run.manifest.cohortId === current.cohortId)
      && (current.kind === 'all' || (current.kind === 'labeled') === runLabeled(run))
      && (current.method === 'all' || (source?.method ?? 'ensemble') === current.method)
      && `${run.manifest.name} ${run.id} ${experimentName(run)} ${source ? modelDescription(source, batchName) : ''} ${nameOf(run.manifest.cohortId)}`.toLowerCase().includes(current.search.trim().toLowerCase());
  }).sort((a, b) => (current.sort === 'name' ? a.manifest.name.localeCompare(b.manifest.name) : current.sort === 'oldest' ? a.createdAt.localeCompare(b.createdAt) : b.createdAt.localeCompare(a.createdAt)) || a.id.localeCompare(b.id));
  const visible = matching.filter((run) => inObjectiveScope(objectiveScope, filingOf(run)?.objectiveId));
  const byObjective = (current.group ?? 'objective') === 'objective' && objectives.length > 0;
  // By objective: a section per objective, then one block per experiment in role order.
  // By cohort: a block per cohort, as the runs of one testing set read together.
  const sections: { key: string; header: ReactNode; blocks: { key: string; header: ReactNode; runs: ModelEvaluation[] }[] }[] = byObjective
    ? sectionByObjective(visible, (run) => filingOf(run)?.objectiveId ?? null, objectives).map((section) => {
      const experimentsOfSection = [...new Set(section.items.map((run) => run.manifest.experimentId))].sort((a, b) => {
        const left = filings[a], right = filings[b];
        return roleRank(left) - roleRank(right) || (left?.series ?? '').localeCompare(right?.series ?? '') || (left?.seriesMember ?? '').localeCompare(right?.seriesMember ?? '', undefined, { numeric: true }) || a.localeCompare(b);
      });
      return {
        key: section.key,
        header: <>{section.objective?.name ?? 'No objective'}{section.objective ? <Badge tone={section.objective.rank === 'primary' ? 'purple' : 'neutral'}>{objectiveRankLabel[section.objective.rank]}</Badge> : null}<small>{section.items.length} {section.items.length === 1 ? 'run' : 'runs'}</small></>,
        blocks: experimentsOfSection.map((experiment) => {
          const filing = filings[experiment];
          const items = section.items.filter((run) => run.manifest.experimentId === experiment);
          return { key: experiment, runs: items, header: <>{filing?.role ? <span className={`experiment-role experiment-role-${filing.role}`}>{experimentRoleLabel[filing.role]}</span> : null}<span>{experimentName(items[0])}</span>{filing?.series ? <small>{filing.series}{filing.seriesMember ? ` › ${filing.seriesMember}` : ''}</small> : null}<small>{items.length} {items.length === 1 ? 'run' : 'runs'}</small></> };
        }),
      };
    })
    : [...visible.reduce((map, run) => {
      const group = map.get(run.manifest.cohortId) ?? [];
      group.push(run); map.set(run.manifest.cohortId, group); return map;
    }, new Map<string, ModelEvaluation[]>()).entries()].map(([cohortId, items], _index, all) => {
      const cohort = cohortById.get(cohortId);
      return { key: cohortId, header: all.length > 1 ? <>{nameOf(cohortId)}{cohort ? ` · ${cohortKindText(cohort, referenceCount(cohortId))}` : ''}</> : null, blocks: [{ key: cohortId, header: null, runs: items }] };
    });
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
      {objectives.length ? <label className="label">Group by<select className="field" aria-label="Group runs by" value={current.group ?? 'objective'} onChange={(event) => update({ group: event.target.value as RunFilters['group'] })}><option value="objective">Objective and experiment</option><option value="cohort">Cohort</option></select></label> : null}
    </StageLibraryToolbar>
    {onClearScope ? <ObjectiveScopeNotice objectives={objectives} scope={objectiveScope} hidden={matching.length - visible.length} noun={matching.length - visible.length === 1 ? 'run' : 'runs'} onClear={onClearScope} /> : null}
    {scope ? <p className="callout apply-runs-scope">Showing the runs of {scope}. <button type="button" className="text-button" onClick={() => update({ experimentId: '', predictorId: '' })}>Show all runs</button></p> : null}
    {loading ? <p role="status">Loading runs…</p> : error && !runs.length ? <LoadFailed what="Runs" error={error} onRetry={onRetry} /> : !visible.length ? <EmptyState icon="evaluation"
      title={runs.length ? 'No matching runs' : 'No models applied yet'}
      description={runs.length ? 'Try another search or clear the filters.' : 'Apply ready predictors to a cohort: labeled cohorts are scored, unlabeled ones get predictions only.'}
      action={runs.length ? <button type="button" className="btn btn-secondary" onClick={reset}>Clear filters</button> : <StageCreateButton onClick={onCreate}>Apply predictors</StageCreateButton>} />
      : <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-name">Run</th><th>Model</th><th>Cohort</th><th>Status</th><th>Result</th><th>Actions</th></tr></thead><tbody>
        {sections.map((section) => <Fragment key={section.key}>
          {section.header ? <tr className={`run-group${byObjective ? ' run-objective-group' : ''}`}><th colSpan={6}>{section.header}</th></tr> : null}
          {section.blocks.map((block) => <Fragment key={block.key}>
          {block.header ? <tr className="run-subgroup"><th colSpan={6}>{block.header}</th></tr> : null}
          {block.runs.map((run) => { const source = predictorById.get(run.manifest.predictorId); const cohort = cohortById.get(run.manifest.cohortId); return <tr key={run.id}>
            <td><button type="button" className="text-button stage-record-name" onClick={() => onOpen(run.id)}>{run.manifest.name}</button>{run.manifest.purpose === 'review' ? <small>Earlier slide-review run</small> : null}</td>
            <td><a href={experimentPredictorLink(run.manifest.experimentId, run.manifest.predictorId)}>{experimentName(run)}</a><small>{source ? modelDescription(source, batchName) : 'Predictor unavailable'}</small></td>
            <td>{nameOf(run.manifest.cohortId)}<small>{(cohort ? cohortLabeled(cohort) : runLabeled(run)) ? 'Labeled · scored' : unlabeledCohortText(referenceCount(run.manifest.cohortId, run.manifest.target?.classes))}</small></td>
            <td><Badge>{computeStatusLabel(run.execution)}</Badge>{run.lifecycleState !== 'active' ? <small>{lifecycleLabel[run.lifecycleState]}</small> : null}</td>
            <td><RunResult run={run} /></td>
            <td><StageRecordManageButton type="configuration" id={run.id} name={run.manifest.name} /></td>
          </tr>; })}
          </Fragment>)}
        </Fragment>)}
      </tbody></table></div>}
  </div>;
}
