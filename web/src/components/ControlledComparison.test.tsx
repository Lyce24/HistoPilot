import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { defaultRecipe, nnmilRecipe, withRecipeDefaults, type DevelopmentBatchSpec, type TrainingRecipe } from '../api/development';
import { defaultPredictorPolicy } from '../lib/experimentPredictors';
import { isBatchEditorDraft } from '../lib/batchEditorDraft';
import { AblationArms, ComparisonReview, ComparisonSettings, ablationArms, armLabel, comparisonFromArms, comparisonSummary, withArmModel } from './ControlledComparison';
import { BatchPlanSettings, BatchPredictorSummary, batchSpecification, batchTemplate, type BatchEditorValues } from './DevelopmentBatches';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const text = (html: string) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
/** A tuned ABMIL recipe with clinical fields: every optimisation choice must reach every image arm. */
const base: TrainingRecipe = { ...withRecipeDefaults(defaultRecipe()), inputMode: 'multimodal', clinicalFields: [{ field: 'age', kind: 'numeric' }],
  learningRate: 1e-4, weightDecay: 0.01, maxEpochs: 30, batchSize: 1, bagSize: 2048, lrScheduler: 'cosine', warmupEpochs: 2 };
const shared = ['learningRate', 'weightDecay', 'maxEpochs', 'batchSize', 'bagSize', 'bagSizeMode', 'lrScheduler', 'warmupEpochs', 'optimizer', 'dropout', 'embedDim', 'numFcLayers', 'patience', 'checkpointMetric'] as const;
function editorValues(changes: Partial<BatchEditorValues> = {}): BatchEditorValues {
  return { experimentId: 'experiment', experimentRevision: 3, experimentName: 'Study', name: ' Ablation ', inputs, recipe: base, mode: 'single',
    lrs: '0.0001', wds: '0.01', epochs: '30', rows: [{ id: 0, recipe: base }], seeds: '42, 43, 44', notes: '', predictorPolicy: defaultPredictorPolicy(),
    selectionMetric: 'validation_auroc', candidateSelection: 'best_validation', comparison: null, ...changes };
}

describe('ablation arms', () => {
  it('builds every ticked model × input with the base recipe first as the reference', () => {
    const arms = ablationArms(base, ['abmil', 'nnmil'], ['multimodal', 'image', 'clinical']);
    expect(arms.map(armLabel)).toEqual(['ABMIL · clinical + image', 'ABMIL · image', 'nnMIL · image', 'nnMIL · clinical + image', 'clinical only']);
    expect(arms[0]).toBe(base);
    // Image arms read no clinical field; clinical arms keep the same typed fields.
    expect(arms[1].clinicalFields).toEqual([]);
    expect(arms[3].clinicalFields).toEqual(base.clinicalFields);
    expect(arms[4]).toMatchObject({ inputMode: 'clinical', model: 'abmil', clinicalFields: base.clinicalFields });
    // One clinical-only arm, whatever the number of models: it reads no image.
    expect(ablationArms(base, ['abmil', 'nnmil', 'mean_pool'], ['multimodal', 'clinical']).filter((arm) => arm.inputMode === 'clinical')).toHaveLength(1);
  });

  it('copies only the options a new model owns, never another template’s optimisation', () => {
    const [, nnmil] = ablationArms({ ...base, inputMode: 'image', clinicalFields: [] }, ['abmil', 'nnmil'], ['image']);
    expect(nnmil).toMatchObject({ model: 'nnmil', attentionDim: 256, gatedAttention: true, nnmilWindowSeedFromTraining: true });
    for (const key of shared) expect(nnmil[key], key).toEqual(base[key]);
    // nnMIL's template batches 32 slides for 100 epochs from fitting-fold median bags; none of that is imported.
    expect(nnmil.batchSize).not.toBe(nnmilRecipe().batchSize);
    expect(nnmil.bagSizeMode).toBe('fixed');
    // From an nnMIL base, other models drop nnMIL-only choices the service would refuse, and keep its training settings.
    const tuned = { ...nnmilRecipe(), nnmilBatchSampler: 'class_balanced' as const, nnmilCheckpointSelection: 'latest' as const };
    const abmil = withArmModel(tuned, 'abmil');
    expect(abmil).toMatchObject({ model: 'abmil', attentionDim: 384, nnmilBatchSampler: 'patient_weighted', nnmilCheckpointSelection: 'best_validation', nnmilWindowSeedFromTraining: false });
    for (const key of ['batchSize', 'bagSize', 'bagSizeMode', 'bagSizeFraction', 'maxEpochs', 'lrScheduler', 'warmupEpochs', 'evalBatchSize'] as const) expect(abmil[key], key).toEqual(tuned[key]);
    expect(withArmModel(tuned, 'nnmil')).toBe(tuned);
  });

  it('declares a comparison with the first arm as reference, all built, and saves it in the spec', () => {
    let id = 10;
    const arms = ablationArms(base, ['abmil', 'nnmil'], ['multimodal', 'image']);
    const plan = comparisonFromArms(arms, () => id++);
    expect(plan).toMatchObject({ mode: 'explicit', candidateSelection: 'all', comparison: { referenceRow: 10, primaryMetric: 'auroc' } });
    expect(plan.rows.map((row) => row.id)).toEqual([10, 11, 12, 13]);
    const spec = batchSpecification(editorValues({ mode: plan.mode, rows: plan.rows, comparison: plan.comparison, candidateSelection: 'best_validation' }));
    expect(spec.comparison).toEqual({ reference: 1, primaryMetric: 'auroc' });
    expect(spec.candidateSelection).toBe('all');
    expect(spec.configurations).toEqual(arms);
    expect(spec).toMatchObject({ mode: 'explicit', batchName: 'Ablation', trainingSeeds: [42, 43, 44] });
  });

  it('renders the helper with the base model and inputs locked in and a live arm count', () => {
    const html = text(renderToStaticMarkup(<AblationArms recipe={base} featureKind="patch" onCreate={() => {}} />));
    expect(html).toContain('Ablation arms (optional)');
    expect(html).toContain('ABMIL (this configuration)');
    expect(html).toContain('Clinical + image (this configuration)');
    expect(html).not.toContain('Slide-embedding');
    expect(html).toContain('1 configuration 1. ABMIL · clinical + image (reference)');
    expect(html).toContain('Tick at least one more model or input.');
    const imageOnly = renderToStaticMarkup(<AblationArms recipe={defaultRecipe()} featureKind="patch" onCreate={() => {}} />);
    expect(imageOnly).toContain('Select clinical fields under Model inputs to add clinical arms.');
    expect(imageOnly).toMatch(/<input type="checkbox" disabled=""\/>Clinical only/);
  });
});

describe('controlled comparison settings', () => {
  const rows = ablationArms(base, ['abmil', 'nnmil'], ['multimodal', 'image']).map((recipe, id) => ({ id, recipe }));

  it('keeps the reference by row, falls back to the first, and omits the comparison outside custom configurations', () => {
    const values = editorValues({ mode: 'explicit', rows, comparison: { referenceRow: 2, primaryMetric: 'balancedAccuracy' } });
    expect(batchSpecification(values).comparison).toEqual({ reference: 3, primaryMetric: 'balancedAccuracy' });
    expect(batchSpecification({ ...values, rows: rows.filter((row) => row.id !== 2) }).comparison).toEqual({ reference: 1, primaryMetric: 'balancedAccuracy' });
    const single = batchSpecification({ ...values, mode: 'single' });
    expect(single).not.toHaveProperty('comparison');
    expect(single.candidateSelection).toBe('best_validation');
    expect(batchSpecification(editorValues({ mode: 'explicit', rows }))).not.toHaveProperty('comparison');
  });

  it('explains the one-factor rule and offers the reference and primary metric', () => {
    const html = text(renderToStaticMarkup(<ComparisonSettings rows={rows} value={{ referenceRow: 0, primaryMetric: 'auroc' }} onChange={() => {}} />));
    expect(html).toContain('Run as a controlled comparison');
    expect(html).toContain('Configurations may differ only in the model and its inputs; if any other setting differs, the batch review blocks the batch.');
    expect(html).toContain('Configuration 3 · nnMIL · image');
    expect(html).toContain('AUROC (default)');
    const tooMany = Array.from({ length: 9 }, (_, id) => ({ id, recipe: base }));
    expect(text(renderToStaticMarkup(<ComparisonSettings rows={tooMany} value={{ referenceRow: 0, primaryMetric: 'auroc' }} onChange={() => {}} />))).toContain('needs 2 to 8 configurations; this batch has 9.');
    expect(renderToStaticMarkup(<ComparisonSettings rows={rows} value={null} onChange={() => {}} />)).not.toContain('Reference configuration');
  });

  it('summarizes the comparison in the batch review and saved settings', () => {
    const spec: DevelopmentBatchSpec = { ...batchTemplate('baseline', inputs, 'Study'), mode: 'explicit', candidateSelection: 'all',
      configurations: rows.map((row) => row.recipe), comparison: { reference: 1, primaryMetric: 'auroc' } };
    expect(comparisonSummary(spec)).toBe('Controlled comparison · reference: configuration 1 (ABMIL · clinical + image) · primary metric AUROC');
    expect(text(renderToStaticMarkup(<BatchPredictorSummary spec={spec} />))).toContain('Controlled comparison · reference: configuration 1 (ABMIL · clinical + image) · primary metric AUROC');
    const saved = text(renderToStaticMarkup(<BatchPlanSettings spec={spec} />));
    expect(saved).toContain('Controlled comparison Reference: configuration 1 (ABMIL · clinical + image) · Primary metric AUROC');
    expect(saved).toContain('Configuration 1 (reference)');
    expect(saved).toContain('Configuration selection: none.');
    expect(renderToStaticMarkup(<BatchPlanSettings spec={spec} />)).toContain('&quot;comparison&quot;: {');
  });

  it('puts the review’s comparison findings first, in the editor’s words', () => {
    const findings = [
      { severity: 'error' as const, code: 'COMPARISON_CONFOUNDED', message: 'Arm 2 differs from the reference arm in batchSize, maxEpochs, nnmilFoo. A comparison may change only the model and its inputs; align these settings so the difference measures one factor.' },
      { severity: 'warning' as const, code: 'CLINICAL_FIELD_SEPARATES_LABELS', message: 'age alone separates the development labels almost perfectly.' },
    ];
    const html = text(renderToStaticMarkup(<ComparisonReview findings={findings} />));
    expect(html).toContain('The controlled comparison cannot run yet.');
    expect(html).toContain('Arm 2 differs from the reference arm in batch size, maximum epochs, nnmil foo.');
    expect(html).toContain('COMPARISON_CONFOUNDED');
    expect(html).not.toContain('separates the development labels');
    expect(text(renderToStaticMarkup(<ComparisonReview findings={findings.slice(1)} />))).toContain('the comparison is controlled');
  });

  it('recovers an unsaved comparison and rejects a malformed one', () => {
    const draft = { version: 1, editorRevision: 1, inputs, workingPlan: null, name: 'Ablation', editorOpen: true, batchPage: 2, templateId: 'blank',
      predictorPolicy: defaultPredictorPolicy(), selectionMetric: 'validation_auroc', candidateSelection: 'all', recipe: base, mode: 'explicit',
      rows, explicitInitialized: true, seeds: '42', lrs: '0.0001', wds: '0.01', epochs: '30', notes: '', numericDrafts: {} };
    expect(isBatchEditorDraft({ ...draft, comparison: { referenceRow: 1, primaryMetric: 'macroF1' } })).toBe(true);
    expect(isBatchEditorDraft(draft)).toBe(true);
    expect(isBatchEditorDraft({ ...draft, comparison: { referenceRow: 1, primaryMetric: 'loss' } })).toBe(false);
  });
});
