import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { defaultRecipe, nnmilRecipe, type NnMILPlanningRow } from '../api/development';
import { BatchPlanSettings, ConfigurationTable, RecipeFields, RecipeSummary, batchTemplate, batchTemplates } from './DevelopmentBatches';
import { applyNnMILPaperOptimizer, NnMILPlanning } from './NnMILRecipeFields';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };

describe('editable nnMIL template', () => {
  it('preserves the patient protocol and requested optimizer defaults while configuring the nnMIL method', () => {
    const spec = batchTemplate('nnmil', inputs, 'KRAS study');
    expect(batchTemplates.some((template) => template.id === 'nnmil')).toBe(true);
    expect(spec).toMatchObject({ selectionMetric: 'validation_auroc', candidateSelection: 'best_validation', trainingSeeds: [42] });
    expect(spec.recipe).toMatchObject({ model: 'nnmil', learningRate: 3e-4, weightDecay: 1e-4, attentionDim: 256,
      dropout: 0.25, batchSize: 32, bagSizeMode: 'training_median', bagSizeFraction: 0.5,
      maxEpochs: 100, patience: 10, minEpochs: 1, lrScheduler: 'cosine', warmupEpochs: 5,
      evalBagSize: null, evalBatchSize: 1, nnmilFeatureSampling: true, nnmilWindowStrideDivisor: 4,
      nnmilWindowAggregation: 'mean_logits', nnmilBatchSampler: 'patient_weighted', nnmilCheckpointSelection: 'best_validation',
      patientAggregation: 'mean_probabilities', ensembleAggregation: 'mean_probability', checkpointMetric: 'validation_auroc' });
    expect(spec.recipe.analysis).toEqual(defaultRecipe().analysis);
    expect(defaultRecipe().model).toBe('abmil');
  });

  it('offers editable method settings and manual patch limits without irrelevant projection fields', () => {
    const html = renderToStaticMarkup(<RecipeFields value={nnmilRecipe()} onChange={() => {}} />);
    for (const label of ['Calculate from fitting-fold median', 'Fraction of median patch count', 'Sample patches per bag', 'Use whole bag for training', 'Attention dimensions', 'Use feature sampling and windowed testing', 'Feature-window stride divisor', 'Feature-window shuffle seed', 'Feature-window aggregation', 'Class-balanced batches', 'AUC-stratified batches', 'Latest completed epoch', 'Apply paper optimizer and schedule', 'Cosine schedule interval', 'Weight decay applies to']) expect(html).toContain(label);
    expect(html).not.toContain('>Embedding dimensions');
    expect(html).not.toContain('>Fully connected layers');
    expect(html).toMatch(/<input[^>]*disabled=""[^>]*\/>Grow the training bag/);
    const fixed = renderToStaticMarkup(<RecipeFields value={{ ...nnmilRecipe(), bagSizeMode: 'fixed', bagSize: 2048 }} onChange={() => {}} />);
    expect(fixed).toContain('Patches per bag');
    expect(fixed).not.toContain('Fraction of median patch count');
    expect(fixed).not.toMatch(/<input[^>]*disabled=""[^>]*\/>Grow the training bag/);
  });

  it('applies paper optimization explicitly and preserves customized model and patient settings', () => {
    const recipe = { ...nnmilRecipe(), learningRate: 1e-4, weightDecay: 5e-3, attentionDim: 128, nnmilWindowStrideDivisor: 2, bagSizeMode: 'fixed' as const, bagSize: 3000, maxEpochs: 80, headLearningRate: 0.01, aggregatorLearningRate: 0.002 };
    expect(applyNnMILPaperOptimizer(recipe)).toEqual({ ...recipe, optimizer: 'adamw', learningRate: 3e-4, weightDecay: 1e-4,
      weightDecayPolicy: 'weights_only', lrScheduler: 'cosine', lrScheduleInterval: 'step', warmupEpochs: 5, finalLrFraction: 0,
      aggregatorLearningRate: null, headLearningRate: null, adamBetas: [0.9, 0.999], adamEps: 1e-8 });
    expect(recipe.learningRate).toBe(1e-4);
    const short = renderToStaticMarkup(<RecipeFields value={{ ...nnmilRecipe(), maxEpochs: 5, warmupEpochs: 0 }} onChange={() => {}} />);
    expect(short).toMatch(/<button[^>]*disabled=""[^>]*>Apply paper optimizer and schedule/);
  });

  it('makes sampler overrides and checkpoint choices unambiguous', () => {
    const html = renderToStaticMarkup(<RecipeFields value={{ ...nnmilRecipe(), nnmilBatchSampler: 'class_balanced', nnmilCheckpointSelection: 'latest' }} onChange={() => {}} />);
    expect(html).toMatch(/Training record sampling<select[^>]*disabled=""/);
    expect(html).toMatch(/<input[^>]*disabled=""[^>]*\/>Class-weighted sampling/);
    expect(html).toContain('Early-stopping validation monitor');
    expect(html).toContain('changes patient exposure');
  });

  it('explains that disabling feature sampling also uses one full-dimensional testing view', () => {
    const html = renderToStaticMarkup(<RecipeFields value={{ ...nnmilRecipe(), nnmilFeatureSampling: false }} onChange={() => {}} />);
    expect(html).toContain('training and testing use one view with all feature coordinates');
    expect(html).toMatch(/Feature-window aggregation<select[^>]*disabled=""/);
  });

  it('reviews the symbolic auto-bag rule and exact overrides in saved settings', () => {
    const spec = batchTemplate('nnmil', inputs, 'Study');
    spec.recipe = { ...spec.recipe, nnmilWindowAggregation: 'mean_probabilities', nnmilWindowShuffle: false,
      nnmilCheckpointSelection: 'latest', nnmilBatchSampler: 'auc_stratified', bagSizeFraction: 0.75 };
    const html = renderToStaticMarkup(<BatchPlanSettings spec={spec} />);
    for (const text of ['floor(0.75 × fitting-fold median)', 'Full-dimensional pooling', 'Average window probabilities', 'Original coordinate order', 'Latest completed epoch', 'AUC-stratified batches']) expect(html).toContain(text);
    expect(renderToStaticMarkup(<RecipeSummary recipe={spec.recipe} />)).toContain('0.75 × fitting-fold median patches');
    expect(renderToStaticMarkup(<ConfigurationTable batch={{ configurations: [{ id: 'c1', number: 1, recipe: spec.recipe }] }} />)).toContain('0.75 × fitting-fold median');
  });

  it('shows fitting-fold counts and resolved limits from the backend preview', () => {
    const row: NnMILPlanningRow = { candidateId: 'c1', splitPlanId: 'fold-1', trainingSlideCount: 100, trainingPatientCount: 80,
      medianPatchCount: 11583, bagSize: 5791, featureDimension: 1024, windowCount: 13,
      minPatchCount: 1000, maxPatchCount: 50000, paddedSlides: 12, truncatedSlides: 88,
      patchCountP05: 2000, patchCountP95: 30000, fingerprint: 'fingerprint', inputMemoryMiB: 723.875 };
    const html = renderToStaticMarkup(<NnMILPlanning rows={[row]} />);
    for (const text of ['c1 / fold-1', '80 / 100', '11,583', '5,791', '1,024 / 13', '12 / 88', 'Validation, assessment and external cohorts are excluded']) expect(html).toContain(text);
    expect(html).toContain('Input tensors (MiB)');
    expect(html).toContain('723.9');
    expect(renderToStaticMarkup(<NnMILPlanning rows={[]} />)).toBe('');
  });
});
