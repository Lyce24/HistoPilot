import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { EvaluationCohort } from '../api/evaluation';
import { modelEvaluations, type ModelEvaluation } from '../api/predictors';
import { referenceFits, references, type ReferenceStandard } from '../api/references';
import { applyHref } from '../lib/applyRoutes';
import ReferenceStandardEditor from './ReferenceStandardEditor';
import { StageRecordManageButton } from './StageWorkflow';
import { Badge, ErrorNotice, Panel } from './ui';

const RUNS_LISTED = 3;

/**
 * The class sets a reference of this cohort can use: its own target's, then those of the runs
 * on it. A reference only scores runs whose classes it maps to.
 */
export function referenceClassSets(cohort: EvaluationCohort, runs: readonly ModelEvaluation[]) {
  const sets = new Map<string, string[]>();
  const add = (classes?: readonly string[]) => {
    const key = classes && classes.length >= 2 ? [...classes].sort().join('\u0000') : '';
    if (key && !sets.has(key)) sets.set(key, [...classes!]);
  };
  add(cohort.manifest.target?.classes);
  for (const run of runs) if (run.manifest.cohortId === cohort.id && run.lifecycleState !== 'trashed') add(run.manifest.target?.classes);
  return [...sets.values()];
}

/**
 * Labels attached to one frozen cohort after it was frozen, such as each reader's grades and
 * their consensus, with the runs each one scores.
 */
export default function CohortReferences({ project, cohort }: { project: string; cohort: EvaluationCohort }) {
  const standards = useQuery({ queryKey: ['reference-standards', project, cohort.id], queryFn: () => references.list(project, cohort.id) });
  const runs = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project) });
  const [adding, setAdding] = useState(false);
  const [classIndex, setClassIndex] = useState(0);
  const cohortRuns = (runs.data?.items ?? []).filter((run) => run.manifest.cohortId === cohort.id && run.lifecycleState !== 'trashed' && run.execution?.status === 'completed');
  const classSets = referenceClassSets(cohort, runs.data?.items ?? []);
  const classes = classSets[Math.min(classIndex, classSets.length - 1)] ?? [];
  const listed = (standards.data?.items ?? []).filter((item) => item.lifecycleState !== 'trashed')
    .sort((a, b) => a.manifest.name < b.manifest.name ? -1 : a.manifest.name > b.manifest.name ? 1 : 0);
  const scores = (standard: ReferenceStandard) => cohortRuns.filter((run) => referenceFits(standard, cohort.id, run.manifest.target?.classes ?? []));
  return <Panel title="Reference standards" subtitle="Labels added to this cohort after it was frozen, such as each reader’s grades and their consensus. Runs on the cohort with the same classes can be scored against each one; the cohort itself never changes."
    actions={!adding ? <button type="button" className="btn btn-secondary btn-small" disabled={!classSets.length} onClick={() => setAdding(true)}>Add reference standard</button> : undefined}>
    <ErrorNotice error={standards.error ?? runs.error} />
    {!classSets.length && !runs.isPending ? <p className="muted">Apply a predictor to this cohort first: a reference standard takes the classes of the runs it scores.</p> : null}
    {adding ? <>
      {classSets.length > 1 ? <label className="label">Classes<select className="field" value={classIndex} onChange={(event) => setClassIndex(Number(event.target.value))}>{classSets.map((item, index) => <option key={item.join('\u0000')} value={index}>{item.join(', ')}</option>)}</select></label> : null}
      <ReferenceStandardEditor key={classes.join('\u0000')} project={project} cohort={cohort} classes={classes} onCancel={() => setAdding(false)} onSaved={() => setAdding(false)} />
    </> : null}
    {standards.isPending ? <p role="status">Loading reference standards…</p> : listed.length ? <div className="table-wrap"><table className="chain-table"><thead><tr><th scope="col">Reference</th><th scope="col">Labeled slides</th><th scope="col">Classes</th><th scope="col">Scores runs</th><th scope="col">Actions</th></tr></thead><tbody>
      {listed.map((item) => {
        const fitting = scores(item);
        return <tr key={item.id}><th scope="row">{item.manifest.name}{item.lifecycleState === 'archived' ? <> <Badge>Archived</Badge></> : null}<small>Column {item.manifest.field}</small></th>
          <td>{item.manifest.summary.labeledSlides.toLocaleString()} of {item.manifest.summary.slides.toLocaleString()}</td>
          <td>{item.manifest.classes.map((name) => `${name} ${(item.manifest.summary.classCounts[name] ?? 0).toLocaleString()}`).join(' · ')}</td>
          <td>{fitting.length ? <>{fitting.slice(0, RUNS_LISTED).map((run, index) => <span key={run.id}>{index ? ', ' : ''}<a href={applyHref({ run: run.id, tab: 'performance', reference: item.id })}>{run.manifest.name}</a></span>)}{fitting.length > RUNS_LISTED ? ` and ${(fitting.length - RUNS_LISTED).toLocaleString()} more` : ''}</> : <span className="muted">No completed run with these classes</span>}</td>
          <td><StageRecordManageButton type="configuration" id={item.id} name={item.manifest.name} /></td></tr>;
      })}
    </tbody></table></div> : !adding ? <p className="muted">No reference standard for this cohort yet.</p> : null}
  </Panel>;
}
