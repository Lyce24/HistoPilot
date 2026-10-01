import type { ModelExperimentSummary } from '../api/experiments';
import { experimentStage, experimentStageLabel } from '../api/experiments';
import type { FrozenPredictor } from '../api/predictors';
import { groupPredictors } from './predictorGroups';
import { shortRecordId } from './recordLabels';

/** `both` is each seed group's fold ensemble and refit; seed ensembles are chosen on their own. */
export type EvaluationMethod = 'both' | 'ensemble' | 'refit' | 'seed_ensemble';
export interface EvaluationExperimentOption {
  id: string; name: string; description: string; ensemble: number; refit: number; seedEnsemble: number; ready: number;
}

/** Include plans with no outputs, without presenting pending work as ready weights. */
export function evaluationExperiments(records: ModelExperimentSummary[], predictors: FrozenPredictor[], selectedIds: string[] = []): EvaluationExperimentOption[] {
  const ready = new Map(groupPredictors(predictors.filter((item) => item.lifecycleState === 'active')).map((group) => [group.id, group]));
  const retained = records.filter((item) => item.state !== 'trashed');
  const byId = new Map(retained.map((item) => [item.id, item]));
  return [...new Set([...byId.keys(), ...ready.keys(), ...selectedIds])].map((id) => {
    const record = byId.get(id), group = ready.get(id);
    const count = (method: string) => group?.items.filter((item) => (item.manifest.method ?? 'ensemble') === method).length ?? 0;
    const ensemble = count('ensemble'), refit = count('refit'), seedEnsemble = count('seed_ensemble');
    const total = ensemble + refit + seedEnsemble;
    const state = record ? `${experimentStageLabel[experimentStage(record)]}${record.state === 'archived' ? ' · archived' : ''}` : group ? 'Retained predictors' : 'Experiment unavailable';
    return { id, name: record?.name ?? group?.name ?? shortRecordId(id), ensemble, refit, seedEnsemble, ready: total,
      description: `${state} · ${total ? `${ensemble} ensemble / ${refit} refit${seedEnsemble ? ` / ${seedEnsemble} seed ensemble` : ''} ready` : 'No ready predictors'}` };
  }).sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
}

/** An empty experiment selection always means no new evaluations. */
export function experimentPredictors(predictors: FrozenPredictor[], experimentIds: string[], method: EvaluationMethod) {
  const sources = new Set(experimentIds);
  return predictors.filter((item) => item.lifecycleState === 'active' && sources.has(item.manifest.experimentId)
    && (method === 'both' ? item.manifest.method !== 'seed_ensemble' : (item.manifest.method ?? 'ensemble') === method));
}

/**
 * The method Apply models proposes. A linked predictor keeps its own; otherwise seed ensembles
 * when any is ready, since they are the deployable form of the Results headline; otherwise
 * each seed group's fold ensemble and refit.
 */
export function defaultEvaluationMethod(predictors: FrozenPredictor[], experimentIds: string[], linkedPredictor = ''): EvaluationMethod {
  if (linkedPredictor) return (predictors.find((item) => item.id === linkedPredictor)?.manifest.method ?? 'ensemble') as EvaluationMethod;
  const sources = new Set(experimentIds);
  return predictors.some((item) => item.lifecycleState === 'active' && sources.has(item.manifest.experimentId) && (item.manifest.method ?? 'ensemble') === 'seed_ensemble')
    ? 'seed_ensemble' : 'both';
}
