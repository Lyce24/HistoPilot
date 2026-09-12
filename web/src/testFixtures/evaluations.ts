import type { ModelExperimentSummary } from '../api/experiments';
import type { FrozenPredictor, ModelEvaluation } from '../api/predictors';

export function fixtureExperiment(id: string, stage: 'planning' | 'running' | 'finished' = 'finished'): ModelExperimentSummary {
  return { id, key: id, name: id === 'study' ? 'Three-seed ABMIL comparison' : id === 'other' ? 'Independent study' : id, notes: '', tags: [], revision: 1, state: 'active', status: stage === 'finished' ? 'completed' : stage, legacy: false, createdAt: '', updatedAt: '', inputs: null, batches: [], drafts: [], predictorId: null, stage, batchPlans: [] };
}
export function fixtureEvaluation(source: FrozenPredictor, auroc: number | null, accuracy: number | null, cohortId = 'cohort', unit: 'slide' | 'patient' = 'patient', id = `${source.id}-${cohortId}-${unit}`): ModelEvaluation {
  const scores = { available: true, count: unit === 'patient' ? 120 : 180, auroc, accuracy };
  return { id, createdAt: '2026-09-12T12:00:00Z', contentHash: 'fixture', lifecycleState: 'active',
    manifest: { kind: 'model-evaluation', predictorId: source.id, experimentId: source.manifest.experimentId, cohortId, name: `${source.manifest.name} evaluation`, status: 'planned' },
    execution: { status: 'completed', result: { metrics: { unit, classOrder: ['a', 'b'], positiveClass: 'b', decisionThreshold: 0.5, patientAggregation: 'mean', selected: { ...scores }, patient: { ...scores }, slide: { ...scores } } } } };
}
