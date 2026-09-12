import type { FrozenPredictor } from '../api/predictors';
import { defaultRecipe } from '../api/development';

export function fixturePredictor(configuration: number, seed: number, method: 'ensemble' | 'refit', experimentId = 'study'): FrozenPredictor {
  return { id: `${experimentId}-${configuration}-${seed}-${method}`, contentHash: 'fixture', createdAt: '', lifecycleState: 'active', manifest: {
    kind: 'frozen-predictor', experimentId, experiment: { id: experimentId, name: 'Same study name' }, name: `Config ${configuration} seed ${seed} ${method}`, batchId: 'batch', candidateId: `candidate-${configuration}`, trainingSeed: seed, splitSeed: 42, method,
    checkpoints: Array.from({ length: method === 'ensemble' ? 5 : 1 }, (_, i) => ({ runId: `fold-${i}`, path: '/offline/weights', sha256: 'fixture', bytes: 1 })), runIds: ['1', '2', '3', '4', '5'], aggregation: method === 'ensemble' ? 'mean_probability' : 'single_model', recipe: defaultRecipe(), target: { field: 'target', task: 'binary_classification', unit: 'patient', classes: ['a', 'b'], labels: {}, missing: 'block', unmapped: 'block', positiveClass: 'b' }, inputs: { protocol: { id: 'protocol', contentHash: '' }, features: { bundle: { id: 'features', contentHash: '' } }, loading: { protocolId: 'protocol', featureBundleId: 'features', loadingPolicy: 'native', packArtifactId: null } },
  } };
}
