import { useState } from 'react';
import type { EvaluationCohort } from '../api/evaluation';
import type { ModelExperimentSummary } from '../api/experiments';
import type { FrozenPredictor, ModelEvaluation } from '../api/predictors';
import { batchNameLookup, cohortName, runLabeled } from '../lib/applyModels';
import EvaluationMethodComparison from './EvaluationMethodComparison';
import PairedPatientComparison from './PairedPatientComparison';
import './RunWorkspace.css';

/**
 * Methods compared across scored runs: fold ensemble against refit on matched sources, and
 * paired patient-level differences. Unlabeled runs have no scores, so they never appear here.
 */
export default function ApplyMethods({ project, runs, predictors, experiments, cohorts, loading, onOpen }: {
  project: string; runs: ModelEvaluation[]; predictors: FrozenPredictor[]; experiments: ModelExperimentSummary[];
  cohorts: EvaluationCohort[]; loading: boolean; onOpen: (id: string) => void;
}) {
  const [state, setState] = useState('active');
  const [cohortId, setCohortId] = useState('');
  const scored = runs.filter(runLabeled);
  const retained = scored.filter((item) => (state === 'all' || item.lifecycleState === state) && (!cohortId || item.manifest.cohortId === cohortId));
  const cohortById = new Map(cohorts.map((item) => [item.id, item]));
  const nameOf = (id: string) => cohortName(cohortById.get(id), id);
  const batchName = batchNameLookup(experiments);
  return <div className="run-workspace apply-methods">
    <div className="run-toolbar">
      <label className="label">State<select className="field" aria-label="Run state" value={state} onChange={(event) => setState(event.target.value)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label>
      <label className="label">Cohort<select className="field" aria-label="Labeled cohort filter" value={cohortId} onChange={(event) => setCohortId(event.target.value)}><option value="">All labeled cohorts</option>{[...new Set(scored.map((item) => item.manifest.cohortId))].map((id) => <option key={id} value={id}>{nameOf(id)}</option>)}</select></label>
    </div>
    <p className="muted">Runs on labeled cohorts, scored against the cohort&rsquo;s own labels. Runs on unlabeled cohorts are not compared here, including runs scored against a reference standard.</p>
    {loading ? <p role="status">Loading runs…</p> : <>
      <PairedPatientComparison project={project} records={retained} />
      <EvaluationMethodComparison records={retained} predictors={predictors} cohortName={nameOf} batchName={(_, batchId) => batchName(batchId)} onOpen={onOpen} />
    </>}
  </div>;
}
