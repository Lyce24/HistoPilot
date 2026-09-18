import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { defaultRecipe, oceanPathRecipe, withRecipeDefaults, type TrainingRecipe } from '../api/development';
import { BatchPlanSettings, RecipeFields, RecipeSummary, batchTemplate } from './DevelopmentBatches';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const render = (recipe: TrainingRecipe) => renderToStaticMarkup(<RecipeFields value={recipe} classes={['Wild type', 'Mutant']} onChange={() => {}} />);

describe('OceanPath recipe controls', () => {
  it('changes new defaults while preserving omitted historical optimizer values', () => {
    expect(defaultRecipe()).toMatchObject({ learningRate: 0.0003, weightDecay: 0.0001, maxEpochs: 40, patience: 8 });
    const { learningRate: _lr, weightDecay: _wd, ...legacy } = defaultRecipe();
    expect(withRecipeDefaults(legacy as TrainingRecipe)).toMatchObject({ learningRate: 0.0003, weightDecay: 0.0001, lossType: 'ce', patientAggregation: 'mean_probabilities', ensembleAggregation: 'mean_probability' });
    expect(withRecipeDefaults({ ...defaultRecipe(), learningRate: 0.001, weightDecay: 0 })).toMatchObject({ learningRate: 0.001, weightDecay: 0 });
    expect(withRecipeDefaults({ ...defaultRecipe(), learningRate: 0.0001, weightDecay: 0.005 })).toMatchObject({ learningRate: 0.0001, weightDecay: 0.005 });
    expect(batchTemplate('blank', inputs, 'Study').grid).toEqual({ learningRates: [0.0003], weightDecays: [0.0001], maxEpochs: [40] });
  });

  it('provides complete standard and KRAS presets without inventing a fallback budget', () => {
    expect(oceanPathRecipe('standard')).toMatchObject({ learningRate: 0.0003, weightDecay: 0.0001, maxEpochs: 20, minEpochs: 10, patience: 5, lrScheduler: 'cosine', finalLrFraction: 0.01, gradientClipNorm: 1, checkpointMetric: 'validation_auroc', patientAggregation: 'mean_probabilities', ensembleAggregation: 'mean_probability', bagSize: null });
    const kras = batchTemplate('oceanpath-kras', inputs, 'Study');
    expect(kras.recipe).toMatchObject({ lossType: 'bce', classWeighting: 'none', weightDecay: 0.01, minEpochs: 0, maxEpochs: 40, patience: 8, finalLrFraction: 0.001, bagSize: 4096 });
    expect(kras.recipe.minValidationPositives).toBeUndefined();
    expect(kras.recipe.fixedEpochBudget).toBeUndefined();
    expect(render(oceanPathRecipe('standard'))).toContain('Average member logits, then convert to probabilities');
  });

  it('offers supported pooling models, objective choices, samplers and evaluation controls', () => {
    const html = render(defaultRecipe());
    for (const label of ['Mean pooling MIL', 'Max pooling MIL', 'Binary cross entropy', 'Focal loss', 'Balanced cohorts', 'Balanced cohorts and labels', 'Class-weighted sampling', 'Instance dropout', 'Feature noise standard deviation', 'Validation &amp; assessment loading', 'Use a separate evaluation batch size', 'Reduce on validation plateau', 'Step decay', 'Adam epsilon', 'Epoch selection policy']) expect(html).toContain(label);
    const pooling = render({ ...defaultRecipe(), model: 'mean_pool' });
    expect(pooling).not.toContain('>Attention dimensions');
    expect(pooling).not.toContain('Gated attention');
    expect(pooling).toContain('Embedding dimensions');
  });

  it('shows class weights in target order and only offers implemented objective controls', () => {
    const weighted = render({ ...defaultRecipe(), classWeights: [1, 1] });
    expect(weighted).toContain('Loss weight · Wild type');
    expect(weighted).toContain('Loss weight · Mutant');
    expect(render({ ...defaultRecipe(), lossType: 'bce' })).not.toContain('Label smoothing');
    expect(render({ ...defaultRecipe(), lossType: 'focal', focalGamma: 1.5 })).toContain('Focal gamma');
    const multiclass = renderToStaticMarkup(<RecipeFields value={defaultRecipe()} classes={['A', 'B', 'C']} onChange={() => {}} />);
    expect(multiclass).toContain('<option value="bce" disabled="">');
  });

  it('makes nondefault experimental choices visible in saved review summaries', () => {
    const spec = batchTemplate('oceanpath-kras', inputs, 'Study');
    spec.recipe = { ...spec.recipe, classWeighting: 'inverse_prevalence', patientAggregation: 'mean_logits', ensembleAggregation: 'mean_logit', samplingStrategy: 'cohort_balanced', cohortColumn: 'site', bagCurriculum: true, bagCurriculumStart: 512, bagCurriculumEnd: 8000, bagCurriculumWarmupEpochs: 5, evalBagSize: 2048, evalBatchSize: 4, minValidationPositives: 5, fixedEpochBudget: 20, lrScheduler: 'step', lrStepSize: 3, lrGamma: 0.8, headLearningRate: 0.002 };
    const html = renderToStaticMarkup(<BatchPlanSettings spec={spec} />);
    for (const label of ['Binary cross entropy', 'Inverse training-fold prevalence weights', 'Average logits, then convert to probabilities', 'Average member logits, then convert to probabilities', 'Balanced cohorts · Column site', '512 → 8000 patches over 5 epochs', '2048 patches maximum', '20 epochs when validation has fewer than 5 positives', 'Every 3 epochs · Factor 0.8', 'Classifier 0.002']) expect(html).toContain(label);
    const compact = renderToStaticMarkup(<RecipeSummary recipe={spec.recipe} />);
    expect(compact).toContain('Binary cross entropy');
    expect(compact).toContain('Patient logit averaging');
  });

  it('keeps optional fallback and bag curriculum controls explicit', () => {
    expect(render(defaultRecipe())).not.toContain('>Fixed epoch budget');
    const html = render({ ...defaultRecipe(), minValidationPositives: 5, fixedEpochBudget: 20, bagCurriculum: true });
    expect(html).toContain('Fixed epoch budget');
    expect(html).toContain('Minimum validation positives');
    expect(html).toContain('Starting patches per bag');
    expect(html).toContain('Final patches per bag');
    expect(html).toContain('Bag curriculum warmup epochs');
    expect(render({ ...defaultRecipe(), classWeightedSampling: true })).not.toContain('Target positive prevalence');
    expect(render({ ...defaultRecipe(), samplingStrategy: 'cohort_label_balanced' })).toContain('Target positive prevalence');
  });
});
