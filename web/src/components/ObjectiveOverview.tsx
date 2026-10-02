import { useQuery } from '@tanstack/react-query';
import { interpretations } from '../api/interpretation';
import { experimentRoleLabel, experimentRoles, objectiveRankLabel, objectives as objectiveRecords, type ExperimentRole } from '../api/objectives';
import { modelEvaluations, predictors } from '../api/predictors';
import { interpretationStudies } from '../lib/interpretationStudies';
import { setObjectiveScope, UNFILED_SCOPE } from '../lib/objectiveScope';
import { Badge } from './ui';
import './ObjectiveScope.css';

const plural = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString()} ${count === 1 ? one : many}`;

/**
 * The project by objective: what each one holds at every stage after Experiments, and the
 * way into each module focused on it. The lists come from the queries the roadmap already
 * keeps, so this adds one small request.
 */
export default function ObjectiveOverview({ project }: { project: string }) {
  const listed = useQuery({ queryKey: ['objectives', project], queryFn: () => objectiveRecords.list(project), staleTime: 30000 });
  const models = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project) });
  const runs = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project) });
  const studies = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project) });
  const objectives = listed.data?.items ?? [];
  if (!objectives.length) return null;
  const filings = listed.data?.filings ?? {};
  const objectiveOf = (experimentId: string) => filings[experimentId]?.objectiveId ?? null;
  const active = <T extends { lifecycleState?: string }>(items: readonly T[] | undefined) => (items ?? []).filter((item) => (item.lifecycleState ?? 'active') === 'active');
  const readyPredictors = active(models.data?.items);
  const activeRuns = active(runs.data?.items);
  const savedStudies = interpretationStudies(active(studies.data?.items));
  const open = (scope: string, module: string) => { setObjectiveScope(scope); window.location.hash = module; };
  // Work from experiments under no objective; experiments never filed are not in `filings`,
  // so the work they led to is what can be counted here.
  const unfiled = [
    plural(readyPredictors.filter((item) => !objectiveOf(item.manifest.experimentId)).length, 'ready predictor'),
    plural(activeRuns.filter((item) => !objectiveOf(item.manifest.experimentId)).length, 'run'),
    plural(savedStudies.filter((study) => !objectiveOf(study.experimentId)).length, 'study', 'studies'),
  ].filter((text) => !text.startsWith('0 '));
  return <section className="objective-overview" aria-labelledby="objective-overview-title">
    <div className="objective-overview-heading"><h2 id="objective-overview-title">Objectives</h2><a href="#experiments">Manage in Experiments</a></div>
    <div className="objective-overview-grid">
      {objectives.map((objective) => {
        const experiments = Object.entries(filings).filter(([, filing]) => filing.objectiveId === objective.id);
        const roles = experimentRoles.map((role) => [role, experiments.filter(([, filing]) => filing.role === role).length] as [ExperimentRole, number]).filter(([, count]) => count);
        const predictorCount = readyPredictors.filter((item) => objectiveOf(item.manifest.experimentId) === objective.id).length;
        const runCount = activeRuns.filter((item) => objectiveOf(item.manifest.experimentId) === objective.id).length;
        const studyCount = savedStudies.filter((study) => objectiveOf(study.experimentId) === objective.id).length;
        return <article key={objective.id} className="objective-card">
          <header><strong>{objective.name}</strong><Badge tone={objective.rank === 'primary' ? 'purple' : 'neutral'}>{objectiveRankLabel[objective.rank]}</Badge></header>
          {objective.description ? <p className="muted">{objective.description}</p> : null}
          <dl>
            <div><dt>Experiments</dt><dd>{plural(experiments.length, 'experiment')}{roles.length ? <small>{roles.map(([role, count]) => `${count} ${experimentRoleLabel[role].toLowerCase()}`).join(' · ')}</small> : null}</dd></div>
            <div><dt>Predictors</dt><dd>{plural(predictorCount, 'ready predictor')}</dd></div>
            <div><dt>Apply models</dt><dd>{plural(runCount, 'run')}</dd></div>
            <div><dt>Interpretation</dt><dd>{plural(studyCount, 'study', 'studies')}</dd></div>
          </dl>
          <div className="objective-card-actions inline-actions">
            <button type="button" className="text-button" onClick={() => open(objective.id, 'experiments')}>Experiments</button>
            <button type="button" className="text-button" onClick={() => open(objective.id, 'apply')}>Apply models</button>
            <button type="button" className="text-button" onClick={() => open(objective.id, 'interpretation')}>Interpretation</button>
          </div>
        </article>;
      })}
    </div>
    {unfiled.length ? <p className="muted objective-overview-unfiled">{unfiled.join(' · ')} {unfiled.length === 1 && unfiled[0].startsWith('1 ') ? 'comes' : 'come'} from experiments under no objective. <button type="button" className="text-button" onClick={() => open(UNFILED_SCOPE, 'experiments')}>File them in Experiments</button></p> : null}
  </section>;
}
