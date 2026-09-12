import type { FrozenPredictor, PredictorManifest } from '../api/predictors';
import { shortRecordId } from './recordLabels';

export function predictorConfigurationLabel(source: Pick<PredictorManifest, 'candidateId'> & { candidateNumber?: number }) {
  return `Configuration ${source.candidateNumber ?? shortRecordId(source.candidateId)}`;
}

export function predictorExperimentName(item: FrozenPredictor) {
  return item.manifest.experiment?.name ?? shortRecordId(item.manifest.experimentId);
}

export function predictorMatches(item: FrozenPredictor, experimentId: string, method: string, search: string) {
  const source = item.manifest;
  return (!experimentId || source.experimentId === experimentId)
    && (method === 'all' || (source.method ?? 'ensemble') === method)
    && [source.name, predictorExperimentName(item), source.experimentId, source.batchId, source.candidateId, predictorConfigurationLabel(source), source.trainingSeed, source.splitSeed].join(' ').toLocaleLowerCase().includes(search.trim().toLocaleLowerCase());
}

/** Group by stable experiment identity; keep every configuration, seed pair and method distinct. */
export function groupPredictors(items: FrozenPredictor[]) {
  const groups = new Map<string, { id: string; name: string; items: FrozenPredictor[] }>();
  for (const item of items) {
    const id = item.manifest.experimentId;
    const group = groups.get(id) ?? { id, name: predictorExperimentName(item), items: [] };
    group.items.push(item); groups.set(id, group);
  }
  return [...groups.values()].sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id)).map((group) => ({
    ...group, items: group.items.sort((a, b) => a.manifest.batchId.localeCompare(b.manifest.batchId)
      || a.manifest.candidateId.localeCompare(b.manifest.candidateId, undefined, { numeric: true })
      || a.manifest.trainingSeed - b.manifest.trainingSeed || a.manifest.splitSeed - b.manifest.splitSeed
      || (a.manifest.method ?? 'ensemble').localeCompare(b.manifest.method ?? 'ensemble') || a.id.localeCompare(b.id)),
  }));
}

export function experimentPredictorLink(experimentId: string, predictorId?: string) {
  const query = new URLSearchParams({ experiment: experimentId, tab: 'predictors' });
  if (predictorId) query.set('predictor', predictorId);
  return `#experiments?${query}`;
}
