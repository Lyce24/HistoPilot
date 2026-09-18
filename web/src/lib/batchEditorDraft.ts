import { defaultRecipe, defaultResources, experimentalRecipeDefaults, type DevelopmentBatchSpec, type ResourcePolicy, type TrainingRecipe } from '../api/development';
import type { ExperimentPredictorPolicy } from '../api/experiments';
import type { MILExperimentSpec } from '../api/mil';
import type { NumericDrafts } from '../components/NumericFieldDrafts';

/** An editing recovery copy, never a reviewed or submitted scientific record. */
export interface BatchEditorDraft {
  version: 1;
  editorRevision: number;
  inputs: MILExperimentSpec;
  workingPlan: string | null;
  name: string;
  editorOpen: boolean;
  batchPage: number;
  templateId: string;
  predictorPolicy: ExperimentPredictorPolicy;
  selectionMetric?: DevelopmentBatchSpec['selectionMetric'];
  candidateSelection?: DevelopmentBatchSpec['candidateSelection'];
  recipe: TrainingRecipe;
  resources: ResourcePolicy;
  mode: DevelopmentBatchSpec['mode'];
  rows: { id: number; recipe: TrainingRecipe }[];
  explicitInitialized: boolean;
  seeds: string;
  lrs: string;
  wds: string;
  epochs: string;
  gpus: string;
  notes: string;
  numericDrafts: NumericDrafts;
}

const object = (value: unknown): value is Record<string, unknown> => Boolean(value && typeof value === 'object' && !Array.isArray(value));
const text = (value: unknown, max = 20000): value is string => typeof value === 'string' && value.length <= max;
const number = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const integer = (value: unknown): value is number => number(value) && Number.isSafeInteger(value) && value >= 0;
function recipe(value: unknown): value is TrainingRecipe {
  if (!object(value)) return false;
  const base = Object.entries(defaultRecipe()).every(([key, sample]) => {
    if (key === 'decisionThreshold') return value[key] == null || number(value[key]);
    if (key === 'analysis') {
      const item = value[key];
      return item == null || object(item) && item.version === 1 && item.confidenceLevel === 0.95
        && integer(item.bootstrapResamples) && item.bootstrapResamples >= 200 && item.bootstrapResamples <= 10000
        && integer(item.bootstrapSeed) && item.bootstrapSeed <= 2 ** 32 - 1
        && integer(item.oneSlideSeed) && item.oneSlideSeed <= 2 ** 32 - 1;
    }
    if (key === 'bagSize' && value[key] === null) return true;
    return typeof sample === 'number' ? number(value[key]) : typeof value[key] === typeof sample;
  });
  if (!base) return false;
  const choices: Record<string, string[]> = {
    lossType: ['ce', 'bce', 'focal'], classWeighting: ['none', 'inverse_prevalence'],
    patientAggregation: ['mean_probabilities', 'mean_logits'], ensembleAggregation: ['mean_probability', 'mean_logit'],
    samplingStrategy: ['slide_uniform', 'patient_natural', 'cohort_balanced', 'cohort_label_balanced'],
    bagSizeMode: ['fixed', 'training_median'], nnmilWindowAggregation: ['mean_logits', 'mean_probabilities'],
    nnmilBatchSampler: ['patient_weighted', 'class_balanced', 'auc_stratified'], nnmilCheckpointSelection: ['best_validation', 'latest'],
    weightDecayPolicy: ['all', 'weights_only'], lrScheduleInterval: ['epoch', 'step'],
  };
  return Object.entries(experimentalRecipeDefaults).every(([key, sample]) => {
    const item = value[key];
    if (item === undefined) return true; // Historical recovery copies predate these controls.
    if (key === 'classWeights') return item === null || Array.isArray(item) && item.length >= 2 && item.length <= 50 && item.every(number);
    if (key === 'adamBetas') return Array.isArray(item) && item.length === 2 && item.every(number);
    if (choices[key]) return typeof item === 'string' && choices[key].includes(item);
    if (sample === null) return item === null || number(item);
    if (typeof sample === 'number') return number(item);
    if (typeof sample === 'string') return text(item, 128);
    return typeof item === typeof sample;
  });
}
function resources(value: unknown): value is ResourcePolicy {
  return object(value) && Array.isArray(value.gpuIds) && value.gpuIds.length <= 128 && value.gpuIds.every(integer)
    && Object.keys(defaultResources()).every((key) => key === 'gpuIds' || number(value[key]));
}

/** Reject malformed browser storage before it can become component state. Scientific
 * ranges are still validated in the editor and by the service before publication. */
export function isBatchEditorDraft(value: unknown): value is BatchEditorDraft {
  if (!object(value) || value.version !== 1 || !integer(value.editorRevision) || !object(value.inputs)
    || !text(value.inputs.protocolId, 256) || !text(value.inputs.featureBundleId, 256)
    || !['auto', 'native', 'mmap'].includes(String(value.inputs.loadingPolicy))
    || !(value.inputs.packArtifactId === null || text(value.inputs.packArtifactId, 256))
    || !(value.workingPlan === null || text(value.workingPlan, 256))
    || !text(value.name, 80) || !text(value.templateId, 80) || !text(value.notes, 2000)
    || typeof value.editorOpen !== 'boolean' || !integer(value.batchPage) || value.batchPage < 1 || value.batchPage > 4
    || !['single', 'grid', 'explicit'].includes(String(value.mode)) || typeof value.explicitInitialized !== 'boolean'
    || !recipe(value.recipe) || !resources(value.resources) || !object(value.predictorPolicy)
    || (value.selectionMetric != null && !['validation_auroc', 'validation_loss', 'validation_accuracy'].includes(String(value.selectionMetric)))
    || (value.candidateSelection != null && !['best_validation', 'all'].includes(String(value.candidateSelection)))
    || !['skip', 'ensemble', 'refit', 'both'].includes(String(value.predictorPolicy.method))
    || !(value.predictorPolicy.refitPercentile === null || number(value.predictorPolicy.refitPercentile))
    || !Array.isArray(value.rows) || value.rows.length < 1 || value.rows.length > 512
    || !value.rows.every((row) => object(row) && integer(row.id) && recipe(row.recipe))
    || new Set(value.rows.map((row) => (row as { id: number }).id)).size !== value.rows.length
    || !['seeds', 'lrs', 'wds', 'epochs', 'gpus'].every((key) => text(value[key]))
    || !object(value.numericDrafts) || Object.keys(value.numericDrafts).length > 20000) return false;
  return Object.entries(value.numericDrafts).every(([key, draft]) => text(key, 256) && object(draft) && number(draft.source) && text(draft.text, 1000));
}
