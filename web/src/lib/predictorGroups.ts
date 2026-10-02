import { experimentRoleLabel, experimentRoles, type ExperimentFiling, type Objective } from '../api/objectives';
import { predictorSeedLabel, type FrozenPredictor, type PredictorManifest } from '../api/predictors';
import { shortRecordId } from './recordLabels';

export function predictorConfigurationLabel(source: Pick<PredictorManifest, 'candidateId'> & { candidateNumber?: number }) {
  return `Configuration ${source.candidateNumber ?? shortRecordId(source.candidateId)}`;
}

export function predictorExperimentName(item: FrozenPredictor) {
  return item.manifest.experiment?.name ?? shortRecordId(item.manifest.experimentId ?? '');
}

export function predictorMatches(item: FrozenPredictor, experimentId: string, method: string, search: string) {
  const source = item.manifest;
  return (!experimentId || source.experimentId === experimentId)
    && (method === 'all' || (source.method ?? 'ensemble') === method)
    && [source.name, predictorExperimentName(item), source.experimentId, source.batchId, source.candidateId, predictorConfigurationLabel(source), predictorSeedLabel(source)].join(' ').toLocaleLowerCase().includes(search.trim().toLocaleLowerCase());
}

/** Group by stable experiment identity; keep every configuration, seed pair and method distinct. */
export function groupPredictors(items: FrozenPredictor[]) {
  const groups = new Map<string, { id: string; name: string; items: FrozenPredictor[] }>();
  for (const item of items) {
    const id = item.manifest.experimentId ?? '';
    const group = groups.get(id) ?? { id, name: predictorExperimentName(item), items: [] };
    group.items.push(item); groups.set(id, group);
  }
  return [...groups.values()].sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id)).map((group) => ({
    ...group, items: group.items.sort((a, b) => (a.manifest.batchId ?? '').localeCompare(b.manifest.batchId ?? '')
      || (a.manifest.candidateId ?? '').localeCompare(b.manifest.candidateId ?? '', undefined, { numeric: true })
      // A seed ensemble (no single seed) leads its configuration's per-seed predictors.
      || (a.manifest.trainingSeed ?? -1) - (b.manifest.trainingSeed ?? -1) || (a.manifest.splitSeed ?? -1) - (b.manifest.splitSeed ?? -1)
      || (a.manifest.method ?? 'ensemble').localeCompare(b.manifest.method ?? 'ensemble') || a.id.localeCompare(b.id)),
  }));
}

export function experimentPredictorLink(experimentId: string, predictorId?: string) {
  const query = new URLSearchParams({ experiment: experimentId, tab: 'predictors' });
  if (predictorId) query.set('predictor', predictorId);
  return `#experiments?${query}`;
}

/**
 * Predictors for a picker, one group per experiment, ordered as the project is organized:
 * objectives in their order, roles in order within each, then experiment names. Each
 * group's label names the objective and the experiment, so a flat `<select>` reads like the
 * library ("Aim 2 · Transport › A2 LOCO cptac · Generalization › cptac").
 */
export function predictorOptionGroups(items: readonly FrozenPredictor[], filings: Record<string, ExperimentFiling>, objectives: readonly Objective[]) {
  const order = new Map(objectives.map((objective, index) => [objective.id, index]));
  const names = new Map(objectives.map((objective) => [objective.id, objective.name]));
  const groups = new Map<string, { key: string; label: string; sort: [number, number, string]; items: FrozenPredictor[] }>();
  for (const group of groupPredictors([...items])) {
    const filing = filings[group.id];
    const objective = filing?.objectiveId && order.has(filing.objectiveId) ? filing.objectiveId : null;
    const role = filing?.role ? ` · ${experimentRoleLabel[filing.role]}` : '';
    const member = filing?.seriesMember ? ` › ${filing.seriesMember}` : '';
    groups.set(group.id, {
      key: group.id,
      label: objectives.length ? `${objective ? names.get(objective) : 'No objective'} › ${group.name}${role}${member}` : group.name,
      sort: [objective ? order.get(objective)! : objectives.length, filing?.role ? experimentRoles.indexOf(filing.role) : experimentRoles.length, `${filing?.series ?? ''} ${filing?.seriesMember ?? ''} ${group.name}`],
      items: group.items,
    });
  }
  return [...groups.values()].sort((a, b) => a.sort[0] - b.sort[0] || a.sort[1] - b.sort[1] || a.sort[2].localeCompare(b.sort[2], undefined, { numeric: true }) || a.key.localeCompare(b.key));
}
