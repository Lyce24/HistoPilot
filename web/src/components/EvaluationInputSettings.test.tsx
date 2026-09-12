import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { EvaluationCohort } from '../api/evaluation';
import type { FeatureBundle } from '../api/bundles';
import type { ModelEvaluation } from '../api/predictors';
import EvaluationInputSettings, { EvaluationCoverageSummary, evaluationExecutionSelection, initialEvaluationInputs } from './EvaluationInputSettings';

const cohort: EvaluationCohort = {
  id: 'independent', projectId: 'p', createdAt: '', contentHash: 'cohort-hash', current: true,
  manifest: { kind: 'evaluation-cohort', datasetId: 'test-data',
    spec: { datasetId: 'test-data', target: null, eligibility: [], patientIdentifiers: 'shared', inference: initialEvaluationInputs().inference },
    target: null, summary: { includedSlides: 6, includedPatients: 3, excludedSlides: 2, labeledSlides: 0, classCounts: {}, developmentSlideOverlap: 0, developmentPatientOverlap: 0 }, findings: [] },
};
const bundle = (id: string, packId: string) => ({ id, current: true, contentHash: id, createdAt: '', findings: [], manifest: { kind: 'feature-bundle', datasetId: 'different-dataset-binding', summary: { slideCount: 8, dimensions: 1024, dtype: 'float32', patchCount: 80, packCount: 1 }, spec: { featureSetId: `${id}-features`, packArtifactIds: [packId] }, packs: [{ id: packId, outputDtype: 'float32' }] } }) as unknown as FeatureBundle;

function render(items: FeatureBundle[], value = initialEvaluationInputs(cohort)) {
  const client = new QueryClient();
  client.setQueryData(['feature-bundles', 'p'], { items });
  try { return renderToStaticMarkup(<QueryClientProvider client={client}><EvaluationInputSettings project="p" cohort={cohort} value={value} onChange={() => {}} /></QueryClientProvider>); }
  finally { client.clear(); }
}

describe('evaluation execution inputs', () => {
  it('allows reviewing an independent cohort before extraction and leaves automatic resolution to the server', () => {
    const inputs = initialEvaluationInputs(cohort);
    expect(evaluationExecutionSelection(inputs)).not.toHaveProperty('featureBundleId');
    expect(inputs.inference).toMatchObject({ loadingPolicy: 'per_slide', numWorkers: 0, decisionThreshold: 0.5 });
    const html = render([]);
    expect(html).toContain('Test features and inference');
    expect(html).toContain('No frozen feature bundles are available');
    expect(html).toContain('The test cohort is already saved');
    expect(html).toContain('prediction task and class encoding');
  });

  it('offers feature inventories across dataset bindings and limits explicit packs to the chosen bundle', () => {
    const inputs = initialEvaluationInputs();
    inputs.featureBundleId = 'bundle-one';
    inputs.inference = { ...inputs.inference, loadingPolicy: 'packed', packArtifactId: 'pack-one' };
    const html = render([bundle('bundle-one', 'pack-one'), bundle('bundle-two', 'pack-two')], inputs);
    expect(html).toContain('value="bundle-one"');
    expect(html).toContain('value="bundle-two"');
    expect(html).toContain('value="pack-one"');
    expect(html).not.toContain('value="pack-two"');
    expect(evaluationExecutionSelection(inputs)).toMatchObject({ featureBundleId: 'bundle-one', inference: { packArtifactId: 'pack-one' } });
  });

  it('preserves historical cohort loading choices as evaluation defaults without mutating the cohort', () => {
    const legacy = { ...cohort, manifest: { ...cohort.manifest, spec: { ...cohort.manifest.spec, featureBundleId: 'legacy-features', inference: { ...cohort.manifest.spec.inference, loadingPolicy: 'packed' as const, packArtifactId: 'legacy-pack', decisionThreshold: 0 } } } };
    const inputs = initialEvaluationInputs(legacy);
    expect(inputs).toMatchObject({ featureBundleId: 'legacy-features', inference: { loadingPolicy: 'packed', packArtifactId: 'legacy-pack', decisionThreshold: 0 } });
    inputs.inference.decisionThreshold = 0.7;
    expect(legacy.manifest.spec.inference.decisionThreshold).toBe(0);
  });

  it('shows exact reviewed feature and pack coverage, including incomparable patient identifiers', () => {
    const manifest: ModelEvaluation['manifest'] = { kind: 'model-evaluation', experimentId: 'study', status: 'planned', name: 'Review', predictorId: 'predictor', cohortId: 'cohort', coverage: { selectedSlideIds: ['s1', 's2'], featureSlideCount: 100, missingFeatureSlideIds: [], missingPackSlideIds: [], packChecked: true }, overlap: { slideIds: [], patientIds: [], patientsComparable: false } };
    const html = renderToStaticMarkup(<EvaluationCoverageSummary manifest={manifest} />);
    expect(html).toContain('2 / 2');
    expect(html).toContain('selected slides have extracted features');
    expect(html).toContain('selected slides are in the feature pack');
    expect(html).toContain('patient overlap cannot be checked by ID');
    expect(html).not.toContain('100 / 2');
  });
});
