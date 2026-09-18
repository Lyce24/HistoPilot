import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { defaultRecipe, withRecipeDefaults, type TrainingRecipe } from '../api/development';
import { defaultPatientAnalysis, type PatientAnalysis } from '../api/statistics';
import { initialEvaluationInputs } from './EvaluationInputSettings';
import { batchTemplate } from './DevelopmentBatches';
import PatientAnalysisResults from './PatientAnalysisResults';
import { plannedBatchPredictorCount } from '../lib/experimentPredictors';
import type { ProtocolSpec } from '../api/scientific';

describe('patient analysis defaults and reporting', () => {
  it('freezes the patient setup and builds only the validation winner by default', () => {
    const inputs = { protocolId: 'p', featureBundleId: 'b', loadingPolicy: 'native' as const, packArtifactId: null };
    const spec = batchTemplate('learning-rate', inputs, 'Study');
    expect(spec).toMatchObject({ selectionMetric: 'validation_auroc', candidateSelection: 'best_validation',
      recipe: { checkpointMetric: 'validation_auroc', analysis: defaultPatientAnalysis(), decisionThreshold: 0.5 } });
    const protocol = { split: { mode: 'kfold', folds: 5, seeds: [42] } } as ProtocolSpec;
    expect(plannedBatchPredictorCount(spec, protocol)).toEqual({ groups: 1, foldRuns: 15, ensembles: 1, refits: 0, total: 1 });
    expect(initialEvaluationInputs().inference).toMatchObject({ patientAggregation: 'predictor', decisionThreshold: 'predictor' });
  });

  it('preserves historical recipe omissions when opening or recovering settings', () => {
    const { analysis: _a, decisionThreshold: _t, checkpointMetric: _m, ...old } = defaultRecipe();
    expect(withRecipeDefaults(old as TrainingRecipe)).toMatchObject({ analysis: null, decisionThreshold: null, checkpointMetric: 'validation_loss' });
  });

  it('reports patient intervals, sensitivity, excluded draws and uncertainty limits', () => {
    const uncertainty = { available: true, patientCount: 20, resamples: 2000, validResamples: 1999, excludedResamples: 1, seed: 42,
      estimates: { auroc: 0.8, auprc: 0.7 }, intervals: { auroc: { lower: 0.6, upper: 0.9 }, auprc: { lower: 0.5, upper: 0.9 } }, note: 'Fixed predictions; excludes model-selection uncertainty.' };
    const value: PatientAnalysis = { policy: defaultPatientAnalysis(), uncertainty,
      oneSlidePerPatient: { metrics: { auroc: 0.75, auprc: 0.65 }, uncertainty, slideIds: ['s1'], seed: 42 } };
    const html = renderToStaticMarkup(<PatientAnalysisResults value={value} />);
    for (const text of ['0.800 (0.600–0.900)', 'AUPRC (95% CI)', '20 labeled patients', '1999 valid resamples',
      'One slide per patient', 'Labels and predictions do not affect selection', 'excludes model-selection uncertainty', 'Export analysis and selected slide IDs']) expect(html).toContain(text);
    expect(renderToStaticMarkup(<PatientAnalysisResults value={{ ...value, uncertainty: { available: false, reason: 'Too few patients.' } }} />)).toContain('Too few patients.');
  });
});
