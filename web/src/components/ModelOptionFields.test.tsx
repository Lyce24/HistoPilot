import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { defaultRecipe, withRecipeDefaults, type TrainingRecipe } from '../api/development';
import { MODEL_OPTIONS, MODELS, modelOptions, readsField, sharedFieldEnabled, withoutForeignOptions } from '../lib/modelCapabilities';
import { isBatchEditorDraft, type BatchEditorDraft } from '../lib/batchEditorDraft';
import { RecipeFields, recipeArchitectureSummary } from './DevelopmentBatches';
import { ExperimentalRecipeSummary } from './ExperimentalRecipeFields';
import { ModelOptionFields, modelOptionSummary, optionEnabled } from './ModelOptionFields';
import { withArmModel } from './ControlledComparison';

const recipe = (model: string, values: Partial<TrainingRecipe> = {}): TrainingRecipe => ({ ...defaultRecipe(), model, ...values });
const NEW_MODELS = ['clam_sb', 'clam_mb', 'transmil', 'dsmil', 'dtfd', 'acmil'];

describe('architecture options', () => {
  it('offers every new architecture in the image-model choice', () => {
    const html = renderToStaticMarkup(<RecipeFields value={defaultRecipe()} onChange={() => {}} featureKind="patch" />);
    for (const label of ['CLAM-SB', 'CLAM-MB', 'TransMIL', 'DSMIL', 'DTFD-MIL', 'ACMIL']) expect(html).toContain(`>${label}</option>`);
    expect(NEW_MODELS.every((name) => MODELS.some((item) => item.name === name && item.options.length))).toBe(true);
  });

  it('edits TransMIL layers, heads and landmarks with the published defaults', () => {
    const html = renderToStaticMarkup(<RecipeFields value={recipe('transmil')} onChange={() => {}} />);
    expect(html).toContain('TransMIL settings');
    for (const label of ['Transformer layers', 'Attention heads', 'Nyström landmarks', 'Pseudo-inverse iterations', 'Attention dropout', 'Pyramid position encoding (PPEG)', 'Value residual convolution']) expect(html).toContain(label);
    expect(html).toContain('value="2"');
    expect(html).toContain('Shao et al., NeurIPS 2021');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Restore TransMIL defaults/);
    // TransMIL reads the projection, not ABMIL's attention width or its dropout.
    expect(html).toContain('Embedding dimensions');
    expect(html).toContain('Fully connected layers');
    expect(html).not.toContain('>Attention dimensions<');
    expect(html).not.toMatch(/>Dropout<input/);
    expect(html).not.toContain('Gated attention</label>');
  });

  it('shows CLAM clustering choices, its own attention width and a reset once changed', () => {
    const html = renderToStaticMarkup(<ModelOptionFields value={recipe('clam_sb', { clamBagWeight: 0.5 })} onChange={() => {}} />);
    for (const label of ['Attention dimensions', 'Patches per instance class (B)', 'Slide loss weight', 'Smooth top-1 SVM', 'Cross entropy', 'Mutually exclusive subtypes']) expect(html).toContain(label);
    expect(html).toContain('extra training losses use this recipe’s loss');
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Restore CLAM-SB defaults/);
    expect(renderToStaticMarkup(<RecipeFields value={recipe('clam_sb')} onChange={() => {}} />)).toMatch(/>Dropout<input/);
  });

  it('disables settings another option switches off', () => {
    const dtfd = MODEL_OPTIONS.find((option) => option.key === 'dtfdDistillPatches')!;
    expect(optionEnabled(recipe('dtfd'), dtfd)).toBe(false);
    expect(optionEnabled(recipe('dtfd', { dtfdDistill: 'maxmins' }), dtfd)).toBe(true);
    const html = renderToStaticMarkup(<ModelOptionFields value={recipe('dtfd')} onChange={() => {}} />);
    expect(html).toMatch(/Patches distilled per pseudo-bag<input[^>]*disabled=""/);
    const value = MODEL_OPTIONS.find((option) => option.key === 'dsmilValueDropout')!;
    expect(optionEnabled(recipe('dsmil', { dsmilPassingValue: true }), value)).toBe(true);
  });

  it('renders nothing for architectures without options or for clinical-only recipes', () => {
    expect(renderToStaticMarkup(<ModelOptionFields value={defaultRecipe()} onChange={() => {}} />)).toBe('');
    expect(renderToStaticMarkup(<ModelOptionFields value={recipe('transmil', { inputMode: 'clinical' })} onChange={() => {}} />)).toBe('');
    expect(modelOptions('abmil')).toEqual([]);
  });

  it('summarizes the shape and every changed option, and reviews all of them', () => {
    expect(recipeArchitectureSummary(recipe('transmil', { transmilDropout: 0.2 })))
      .toBe('TransMIL · 512 embedding · 1 fully connected layer · Transformer layers 2 · Attention heads 8 · Nyström landmarks 256 · Attention dropout 0.2');
    expect(modelOptionSummary(recipe('clam_mb'), 'all')).toContain('Instance loss Smooth top-1 SVM');
    expect(modelOptionSummary(recipe('transmil', { transmilLayers: 4 }), 'changed')).toEqual(['Transformer layers 4']);
    expect(modelOptionSummary(recipe('transmil'), 'changed')).toEqual([]);
    const review = renderToStaticMarkup(<dl><ExperimentalRecipeSummary recipe={recipe('acmil')} /></dl>);
    expect(review).toContain('ACMIL settings');
    expect(review).toContain('Attention branches 5 · Top patches eligible for masking 10 · Masked fraction 0.6');
    expect(review).toContain('own loss terms');
  });

  it('keeps the shared architecture fields of existing models', () => {
    expect(['embedDim', 'attentionDim', 'numFcLayers', 'gatedAttention', 'dropout', 'inputDropout'].every((field) => readsField('abmil', field as never))).toBe(true);
    expect(readsField('nnmil', 'embedDim')).toBe(false);
    expect(readsField('slide_linear', 'dropout')).toBe(false);
    expect(readsField('unlisted-model', 'attentionDim')).toBe(false);
    expect(readsField('unlisted-model', 'numFcLayers')).toBe(true);
  });

  it('returns options of the previous architecture to their defaults when the model changes', () => {
    const tuned = recipe('transmil', { transmilLayers: 4, learningRate: 0.001 });
    const clam = withoutForeignOptions({ ...tuned, model: 'clam_sb' }, 'clam_sb');
    expect(clam.transmilLayers).toBe(2);
    expect(clam.learningRate).toBe(0.001);
    expect('clamBagWeight' in clam).toBe(false);
    expect(withoutForeignOptions(tuned, 'transmil')).toEqual(tuned);
    expect(withArmModel(tuned, 'dsmil').transmilLayers).toBe(2);
    // The defaults every editor starts from include each option once.
    expect(withRecipeDefaults(defaultRecipe()).acmilBranches).toBe(5);
  });

  it('keeps DSMIL as published and enables its input-layer settings only on request', () => {
    const published = renderToStaticMarkup(<RecipeFields value={recipe('dsmil')} onChange={() => {}} />);
    expect(published).toMatch(/Embedding dimensions<input[^>]*disabled=""/);
    expect(published).toMatch(/Fully connected layers<input[^>]*disabled=""/);
    expect(published).toContain('reads the patch features directly');
    expect(sharedFieldEnabled(recipe('dsmil'), 'embedDim')).toBe(false);
    expect(recipeArchitectureSummary(recipe('dsmil'))).toBe('DSMIL · Patch features read directly · Shared input layer off · Query dimensions 128');
    const projected = renderToStaticMarkup(<RecipeFields value={recipe('dsmil', { dsmilProjection: true })} onChange={() => {}} />);
    expect(projected).not.toMatch(/Embedding dimensions<input[^>]*disabled=""/);
    expect(recipeArchitectureSummary(recipe('dsmil', { dsmilProjection: true }))).toContain('DSMIL · 512 embedding · 1 fully connected layer');
  });

  it('offers the published DTFD-MIL and CLAM switches and hides layers they do not read', () => {
    const dtfd = renderToStaticMarkup(<ModelOptionFields value={recipe('dtfd')} onChange={() => {}} />);
    for (const label of ['Test-time pseudo-bag splits', 'Residual blocks']) expect(dtfd).toContain(label);
    expect(renderToStaticMarkup(<ModelOptionFields value={recipe('clam_sb')} onChange={() => {}} />)).toContain('Attention dropout follows Dropout');
    for (const model of ['dtfd', 'acmil']) {
      const architecture = renderToStaticMarkup(<RecipeFields value={recipe(model)} onChange={() => {}} />);
      expect(architecture).toContain('Embedding dimensions');
      expect(architecture).not.toContain('Fully connected layers');
    }
  });

  it('accepts recovery copies with valid architecture choices only', () => {
    const draft = (value: TrainingRecipe): BatchEditorDraft => ({
      version: 1, editorRevision: 1, inputs: { protocolId: 'p', featureBundleId: 'b', loadingPolicy: 'native', packArtifactId: null },
      workingPlan: null, name: 'Draft', editorOpen: true, batchPage: 2, templateId: 'blank',
      predictorPolicy: { method: 'ensemble', refitPercentile: null }, recipe: value, mode: 'single',
      rows: [{ id: 0, recipe: value }], explicitInitialized: false, seeds: '42', lrs: '', wds: '', epochs: '', notes: '', numericDrafts: {},
    });
    expect(isBatchEditorDraft(draft(recipe('dtfd', { dtfdDistill: 'maxmins', dtfdPseudoBags: 8 })))).toBe(true);
    expect(isBatchEditorDraft(draft(recipe('dtfd', { dtfdDistill: 'median' as never })))).toBe(false);
    expect(isBatchEditorDraft(draft(recipe('transmil', { transmilLayers: '2' as never })))).toBe(false);
    expect(isBatchEditorDraft(draft(recipe('transmil', { transmilPositionEncoding: 'yes' as never })))).toBe(false);
  });
});
