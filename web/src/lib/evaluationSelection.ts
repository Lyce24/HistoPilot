import type { ModelExperimentSummary } from '../api/experiments';
import { experimentStage, experimentStageLabel } from '../api/experiments';
import type { FrozenPredictor } from '../api/predictors';
import { groupPredictors } from './predictorGroups';
import { shortRecordId } from './recordLabels';

export type EvaluationMethod = 'both' | 'ensemble' | 'refit';
export interface EvaluationExperimentOption {
  id: string; name: string; description: string; ensemble: number; refit: number; ready: number;
}

/** Include plans with no outputs, without presenting pending work as ready weights. */
export function evaluationExperiments(records: ModelExperimentSummary[], predictors: FrozenPredictor[], selectedIds: string[] = []): EvaluationExperimentOption[] {
  const ready = new Map(groupPredictors(predictors.filter((item) => item.lifecycleState === 'active')).map((group) => [group.id, group]));
  const retained = records.filter((item) => item.state !== 'trashed');
  const byId = new Map(retained.map((item) => [item.id, item]));
  return [...new Set([...byId.keys(), ...ready.keys(), ...selectedIds])].map((id) => {
    const record = byId.get(id), group = ready.get(id);
    const ensemble = group?.items.filter((item) => item.manifest.method !== 'refit').length ?? 0;
    const refit = group?.items.filter((item) => item.manifest.method === 'refit').length ?? 0;
    const state = record ? `${experimentStageLabel[experimentStage(record)]}${record.state === 'archived' ? ' · archived' : ''}` : group ? 'Retained predictors' : 'Experiment unavailable';
    return { id, name: record?.name ?? group?.name ?? shortRecordId(id), ensemble, refit, ready: ensemble + refit,
      description: `${state} · ${ensemble + refit ? `${ensemble} ensemble / ${refit} refit ready` : 'No ready predictors'}` };
  }).sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
}

/** An empty experiment selection always means no new evaluations. */
export function experimentPredictors(predictors: FrozenPredictor[], experimentIds: string[], method: EvaluationMethod) {
  const sources = new Set(experimentIds);
  return predictors.filter((item) => item.lifecycleState === 'active' && sources.has(item.manifest.experimentId)
    && (method === 'both' || (item.manifest.method ?? 'ensemble') === method));
}
