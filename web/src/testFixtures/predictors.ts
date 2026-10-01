import type { FrozenPredictor } from '../api/predictors';
import { defaultRecipe } from '../api/development';

export function fixturePredictor(configuration: number, seed: number, method: 'ensemble' | 'refit', experimentId = 'study'): FrozenPredictor {
  return { id: `${experimentId}-${configuration}-${seed}-${method}`, contentHash: 'fixture', createdAt: '', lifecycleState: 'active', manifest: {
    kind: 'frozen-predictor', experimentId, experiment: { id: experimentId, name: 'Same study name' }, name: `Config ${configuration} seed ${seed} ${method}`, batchId: 'batch', candidateId: `candidate-${configuration}`, trainingSeed: seed, splitSeed: 42, method,
    checkpoints: Array.from({ length: method === 'ensemble' ? 5 : 1 }, (_, i) => ({ runId: `fold-${i}`, path: '/offline/weights', sha256: 'fixture', bytes: 1 })), runIds: ['1', '2', '3', '4', '5'], aggregation: method === 'ensemble' ? 'mean_probability' : 'single_model', recipe: defaultRecipe(), target: { field: 'target', task: 'binary_classification', unit: 'patient', classes: ['a', 'b'], labels: {}, missing: 'block', unmapped: 'block', positiveClass: 'b' }, inputs: { protocol: { id: 'protocol', contentHash: '' }, features: { bundle: { id: 'features', contentHash: '' } }, loading: { protocolId: 'protocol', featureBundleId: 'features', loadingPolicy: 'native', packArtifactId: null } },
  } };
}

/** Every seed group of one configuration pooled into one predictor: no single seed. */
export function fixtureSeedEnsemble(configuration: number, experimentId = 'study', trainingSeeds = [11, 22, 33]): FrozenPredictor {
  const base = fixturePredictor(configuration, trainingSeeds[0], 'ensemble', experimentId);
  const { trainingSeed: _seed, splitSeed: _split, ...rest } = base.manifest;
  return { ...base, id: `${experimentId}-${configuration}-seed-ensemble`, manifest: {
    ...rest, method: 'seed_ensemble', name: `Config ${configuration} seed ensemble`, trainingSeeds, splitSeeds: [42],
    seedGroups: trainingSeeds.map((seed) => ({ trainingSeed: seed, splitSeed: 42, runIds: [] })),
    checkpoints: trainingSeeds.flatMap((seed) => Array.from({ length: 5 }, (_, fold) => ({ runId: `seed-${seed}-fold-${fold}`, path: '/offline/weights', sha256: 'fixture', bytes: 1 }))),
  } };
}
