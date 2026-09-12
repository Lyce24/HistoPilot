import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import BatchPredictorFields, { batchPredictorLabel } from './BatchPredictorFields';
import { defaultRecipe, withRecipeDefaults, type TrainingRecipe } from '../api/development';
import type { ExperimentPredictorPolicy } from '../api/experiments';

function render(value: ExperimentPredictorPolicy, splitSeedCount?: number) {
  return renderToStaticMarkup(<BatchPredictorFields value={value} onChange={() => {}} configurationCount={15} trainingSeedCount={3} splitSeedCount={splitSeedCount} foldCount={5} />);
}

describe('predictor choices within a batch', () => {
  it('creates ninety predictors for fifteen configurations, three seeds and both methods, irrespective of five folds', () => {
    const html = render({ method: 'both', refitPercentile: 75 }, 1);
    expect(html).toContain('90 predictors planned for this batch');
    expect(html).toContain('225 fold runs');
    expect(html).toContain('P75 · 75th percentile');
    expect(html).toContain('Other batches can use different choices.');
    expect(html).not.toContain('<button');
    expect(html).not.toContain('Save predictor choices');
  });

  it('includes multiple split seeds and labels incomplete protocol information without inventing a total', () => {
    expect(render({ method: 'ensemble', refitPercentile: null }, 2)).toContain('90 predictors planned for this batch');
    const unknown = render({ method: 'both', refitPercentile: 75 });
    expect(unknown).toContain('90 predictors per split seed');
    expect(unknown).not.toContain('90 predictors planned for this batch');
    expect(unknown).not.toContain('225 fold runs');
  });

  it('keeps Skip free of refit controls and represents custom percentiles explicitly', () => {
    const skipped = render({ method: 'skip', refitPercentile: null }, 1);
    expect(skipped).toContain('0 predictors planned for this batch');
    expect(skipped).not.toContain('Refit epoch budget');
    const custom = render({ method: 'refit', refitPercentile: 82.5 }, 1);
    expect(custom).toContain('Custom percentile (1–100)');
    expect(custom).toContain('value="82.5"');
    expect(batchPredictorLabel({ method: 'refit', refitPercentile: 82.5 })).toBe('Refit · P82.5 refit epochs');
  });

  it('uses the new training defaults while retaining explicit and omitted historical recipe budgets', () => {
    expect(defaultRecipe()).toMatchObject({ maxEpochs: 40, patience: 8 });
    expect(withRecipeDefaults({ ...defaultRecipe(), maxEpochs: 120, patience: 25 })).toMatchObject({ maxEpochs: 120, patience: 25 });
    const { maxEpochs: _epochs, patience: _patience, ...legacy } = defaultRecipe();
    expect(withRecipeDefaults(legacy as TrainingRecipe)).toMatchObject({ maxEpochs: 100, patience: 15 });
  });
});
