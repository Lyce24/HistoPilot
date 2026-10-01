import type { EvaluationCohort } from '../api/evaluation';
import type { ModelExperiment, ModelExperimentSummary } from '../api/experiments';
import { predictorMethodLabel, predictorSeedLabel, type FrozenPredictor, type ModelEvaluation, type SeedEnsembleChoice, type SeedEnsembleSelection } from '../api/predictors';
import type { Configuration } from '../api/scientific';
import { isInferenceCohort, isInferenceRun } from './inference';
import { predictorConfigurationLabel } from './predictorGroups';
import { shortRecordId } from './recordLabels';
import { versionLabelText } from './versionLabels';

/** A labeled cohort scores its labeled slides; an unlabeled one gets predictions only. */
export const cohortLabeled = (cohort: Pick<EvaluationCohort, 'manifest'>) => !isInferenceCohort(cohort);
export const runLabeled = (run: Pick<ModelEvaluation, 'manifest'>) => !isInferenceRun(run);

export function cohortName(cohort: EvaluationCohort | undefined, id: string) {
  return cohort ? versionLabelText(cohort, 'Cohort') : shortRecordId(id);
}

/** An unlabeled cohort predicts only, unless reference standards added later score its runs. */
export function unlabeledCohortText(referenceCount = 0) {
  return referenceCount ? `Unlabeled · ${referenceCount} reference ${referenceCount === 1 ? 'standard' : 'standards'}` : 'Unlabeled · predictions only';
}

/** "Labeled · 76 of 80 slides scored", "Unlabeled · 2 reference standards" or "Unlabeled · predictions only". */
export function cohortKindText(cohort: Pick<EvaluationCohort, 'manifest'>, referenceCount = 0) {
  const summary = cohort.manifest.summary;
  if (!cohortLabeled(cohort)) return unlabeledCohortText(referenceCount);
  return summary.labeledSlides < summary.includedSlides
    ? `Labeled · ${summary.labeledSlides.toLocaleString()} of ${summary.includedSlides.toLocaleString()} slides scored`
    : 'Labeled · scored';
}

/**
 * The testing set a predictor's development reserved: the cohort Targets & splits derived for
 * the protocol it was trained on. Undefined when its protocol has none.
 */
export function reservedTestingCohort(protocolIds: readonly string[], protocols: readonly Configuration[], cohorts: readonly EvaluationCohort[]) {
  const splits = new Set(protocols.filter((item) => protocolIds.includes(item.id)).map((item) => {
    const manifest = item.manifest as { sourceTargetSplit?: { id: string }; spec?: { sourceTargetSplitId?: string } };
    return manifest.sourceTargetSplit?.id ?? manifest.spec?.sourceTargetSplitId;
  }).filter(Boolean));
  return cohorts.find((cohort) => cohort.current !== false && splits.has(cohort.manifest.spec.sourceTargetSplitId ?? ''));
}

export function batchNameLookup(experiments: readonly ModelExperimentSummary[]) {
  const names = new Map(experiments.flatMap((item) => item.batches.map((batch) => [batch.id, batch.name] as const)));
  return (batchId: string) => names.get(batchId) ?? `Batch ${shortRecordId(batchId)}`;
}

/** "ABMIL baseline · Configuration 1 · Seed ensemble · 3 training × 1 split seed · 15 models". */
export function modelDescription(predictor: FrozenPredictor['manifest'], batchName: (batchId: string) => string) {
  return [batchName(predictor.batchId), predictorConfigurationLabel(predictor), predictorMethodLabel(predictor.method), predictorSeedLabel(predictor)].join(' · ');
}

/** The request that builds a configuration's seed ensemble, named after its experiment. */
export const seedEnsembleSelection = (record: Pick<ModelExperiment, 'name'>, choice: SeedEnsembleChoice): SeedEnsembleSelection => ({
  experimentId: choice.experimentId, batchId: choice.batchId, candidateId: choice.candidateId,
  name: `${record.name} · configuration ${choice.candidateNumber} · seed ensemble`.slice(0, 120),
});

export type ConfigurationChoice =
  | { action: 'apply'; predictorId: string; detail: string }
  | { action: 'build-seed-ensemble'; choice: SeedEnsembleChoice; selection: SeedEnsembleSelection; detail: string }
  | { action: 'choose' | 'none'; predictorIds: string[]; detail: string };

/**
 * What "Apply this configuration" does: apply its built seed ensemble, build one from the
 * verified fold checkpoints (nothing trains), apply its single fold ensemble, or choose among
 * the fold ensembles of its seed groups.
 */
export function configurationChoice(choices: readonly SeedEnsembleChoice[], registry: readonly FrozenPredictor[], record: Pick<ModelExperiment, 'id' | 'name'>, batchId: string, candidateId: string): ConfigurationChoice {
  const choice = choices.find((item) => item.batchId === batchId && item.candidateId === candidateId && item.seedGroups > 1);
  const folds = registry.filter((item) => item.lifecycleState === 'active' && item.manifest.experimentId === record.id
    && item.manifest.batchId === batchId && item.manifest.candidateId === candidateId && (item.manifest.method ?? 'ensemble') === 'ensemble');
  const seedsText = choice ? `${choice.trainingSeeds.length} training × ${choice.splitSeeds.length} split ${choice.splitSeeds.length === 1 ? 'seed' : 'seeds'} · ${choice.members} fold models` : '';
  const existing = choice?.existingPredictorId && registry.some((item) => item.id === choice.existingPredictorId && item.lifecycleState === 'active') ? choice.existingPredictorId : null;
  if (existing) return { action: 'apply', predictorId: existing, detail: `Its seed ensemble is built: ${seedsText}.` };
  if (choice?.eligible) return { action: 'build-seed-ensemble', choice, selection: seedEnsembleSelection(record, choice), detail: `Its seed ensemble (${seedsText}) is built from the verified fold checkpoints; nothing trains.` };
  if (!choice && folds.length === 1) return { action: 'apply', predictorId: folds[0].id, detail: 'Its fold ensemble is ready.' };
  return { action: folds.length ? 'choose' : 'none', predictorIds: folds.map((item) => item.id),
    detail: choice?.reason ?? (folds.length ? `${folds.length} fold ensembles are ready, one per seed group; choose among them in Apply models.` : 'No predictor is ready for this configuration yet.') };
}
