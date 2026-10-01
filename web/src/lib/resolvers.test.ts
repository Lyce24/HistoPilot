import { describe, expect, it } from 'vitest';
import type { EvaluationCohort } from '../api/evaluation';
import type { FrozenPredictor, ModelEvaluation, SeedEnsembleChoice } from '../api/predictors';
import type { ReferenceStandard } from '../api/references';
import type { Configuration, FeatureSpec } from '../api/scientific';
import type { ModelExperimentSummary } from '../api/experiments';
import type { TrainingRecipe } from '../api/development';
import type { ExtractionJob } from '../api/trident';
import { labelSource, labelSources } from '../components/ApplyRunDetail';
import { ablationArms, withArmModel } from '../components/ControlledComparison';
import { evaluationExecutionSelection, initialEvaluationInputs } from '../components/EvaluationInputSettings';
import { suggestedClass } from '../components/ReferenceStandardEditor';
import { batchNameLookup, cohortLabeled, configurationChoice, modelDescription, reservedTestingCohort } from './applyModels';
import { defaultEvaluationMethod, experimentPredictors, type EvaluationMethod } from './evaluationSelection';
import { csv, resultRows } from './experimentResults';
import { editFeatureSource, extractionFeatureInput, extractionFeatureUpdate } from './featureSource';
import { inferTargetSettings } from './protocol';
import cases from './resolverCases.json';

// The same cases run against histopilot/resolvers.py in tests/test_resolvers.py, so the CLI,
// agents and this browser propose the same defaults. Undefined fields compare as JSON null.
const plain = (value: unknown) => JSON.parse(JSON.stringify(value ?? null, (_key, item) => item === undefined ? null : item));
const predictors = cases.predictors as unknown as FrozenPredictor[];

describe('resolvers shared with the service', () => {
  it('chooses the predictors of a method', () => {
    for (const item of cases.experimentPredictors) {
      expect(experimentPredictors(predictors, item.experimentIds, item.method as EvaluationMethod).map((predictor) => predictor.id)).toEqual(item.expected);
    }
  });

  it('describes a predictor as the Runs table does', () => {
    const shared = cases.modelDescription;
    const batchName = batchNameLookup(shared.experiments as unknown as ModelExperimentSummary[]);
    for (const item of shared.cases) {
      expect(modelDescription(item.manifest as unknown as FrozenPredictor['manifest'], batchName)).toBe(item.expected);
    }
  });

  it('proposes the method Apply models opens with', () => {
    for (const item of cases.defaultMethod) {
      expect(defaultEvaluationMethod(predictors, item.experimentIds, item.linkedPredictor)).toBe(item.expected);
    }
  });

  it('finds the testing cohort development reserved', () => {
    const protocols = cases.protocols as unknown as Configuration[];
    const cohorts = cases.cohorts as unknown as EvaluationCohort[];
    for (const item of cases.reservedTestingCohort) {
      expect(reservedTestingCohort(item.protocolIds, protocols, cohorts)?.id ?? null).toBe(item.expected);
    }
  });

  it('starts from the cohort’s inference settings', () => {
    for (const item of cases.initialInputs) {
      const cohort = (item.cohort ?? undefined) as unknown as EvaluationCohort | undefined;
      const inputs = initialEvaluationInputs(cohort);
      expect(plain(inputs)).toEqual(item.expected.inputs);
      expect(plain(evaluationExecutionSelection(inputs))).toEqual(item.expected.selection);
      expect(cohortLabeled(cohort ?? { manifest: { spec: {} } } as unknown as EvaluationCohort)).toBe(item.expected.labeled);
    }
  });

  it('decides what Apply this configuration does', () => {
    const choices = cases.seedEnsembleChoices as unknown as SeedEnsembleChoice[];
    const registry = cases.configurationPredictors as unknown as FrozenPredictor[];
    for (const item of cases.configurationChoice) {
      const plan = { ...configurationChoice(choices, registry, item.experiment, item.batchId, item.candidateId) } as Record<string, unknown>;
      delete plan.choice;
      expect(plain(plan)).toEqual(item.expected);
    }
  });

  it('infers a target from a field’s values', () => {
    for (const item of cases.inferTarget) {
      expect(plain(inferTargetSettings(item.values, item.truncated))).toEqual(item.expected);
    }
  });

  it('maps reference values to classes', () => {
    for (const item of cases.suggestedClass) expect(suggestedClass(item.value, item.classes)).toBe(item.expected);
  });

  it('lists label sources and picks the default', () => {
    const standards = cases.standards as unknown as ReferenceStandard[];
    for (const item of cases.labelSources) {
      const sources = labelSources(item.run as unknown as ModelEvaluation, standards, item.linked ?? undefined);
      expect(sources.map(({ id, name }) => ({ id, name }))).toEqual(item.expected);
      const chosen = sources.length ? labelSource(sources, item.reference) : undefined;
      expect(chosen ? chosen.id : 'none').toBe(item.expectedSource);
    }
  });

  it('builds comparison arms', () => {
    for (const item of cases.withArmModel) {
      expect(plain(withArmModel(item.base as unknown as TrainingRecipe, item.model))).toEqual(item.expected);
    }
    for (const item of cases.ablationArms) {
      const arms = ablationArms(item.base as unknown as TrainingRecipe, item.models, item.inputs as ('image' | 'clinical' | 'multimodal')[]);
      expect(plain(arms)).toEqual(item.expected);
    }
  });

  it('writes the folds-and-seeds CSV byte for byte', () => {
    for (const item of cases.resultsCsv) expect(csv(resultRows(item.results as unknown as Parameters<typeof resultRows>[0]))).toBe(item.expected);
  });

  it('turns an extraction into a features spec', () => {
    const spec = cases.featureSpec as unknown as FeatureSpec;
    for (const item of cases.featureSpecFromExtraction) {
      const input = extractionFeatureInput(item.job as unknown as ExtractionJob);
      expect(plain(input ? editFeatureSource(spec, extractionFeatureUpdate(input)) : null)).toEqual(item.expected);
    }
  });
});
