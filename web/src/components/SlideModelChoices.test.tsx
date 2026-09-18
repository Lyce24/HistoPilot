/**
 * The model choice follows the service's catalog and the bundle's feature kind.
 *
 * A bundle of slide-encoder embeddings has no patch bag to attend over, so the
 * patch architectures must not be offered for it, and the reverse.
 */
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { defaultRecipe } from '../api/development';
import { MODELS, modelsForFeatureKind, supportsAttention } from '../lib/modelCapabilities';
import { BatchPlanSettings, RecipeFields, RecipeSummary, batchTemplate } from './DevelopmentBatches';

const render = (featureKind?: 'patch' | 'slide') =>
  renderToStaticMarkup(
    <RecipeFields value={defaultRecipe()} onChange={() => {}} featureKind={featureKind} />,
  );

describe('model choices', () => {
  it('offers every architecture while no bundle is resolved', () => {
    const html = render();
    for (const item of MODELS) expect(html).toContain(`value="${item.name}"`);
  });

  it('offers only slide probes for a slide-embedding bundle', () => {
    // The default recipe still names ABMIL, so it stays visible as the saved but
    // unreadable choice rather than disappearing without explanation.
    const html = render('slide');
    expect(html).toContain('value="slide_linear"');
    expect(html).toContain('value="slide_mlp"');
    expect(html).toContain('<option value="abmil" disabled="" selected="">ABMIL (unavailable)');
    expect(html).not.toContain('value="nnmil"');
    expect(html).toContain('one embedding per slide');
    expect(html).toContain('Choose a compatible image model before checking the batch.');
  });

  it('offers only patch architectures for a patch bundle', () => {
    const html = render('patch');
    expect(html).toContain('value="abmil"');
    expect(html).not.toContain('value="slide_linear"');
    expect(html).not.toContain('one embedding per slide');
  });

  it('keeps a saved but unavailable architecture visible and disabled', () => {
    const html = renderToStaticMarkup(
      <RecipeFields
        value={{ ...defaultRecipe(), model: 'slide_linear' }}
        onChange={() => {}}
        featureKind="patch"
      />,
    );
    expect(html).toContain('(unavailable)');
  });

  it('splits the catalog by what each architecture reads', () => {
    expect(modelsForFeatureKind('slide').map((item) => item.name)).toEqual([
      'slide_linear',
      'slide_mlp',
    ]);
    expect(modelsForFeatureKind('patch').map((item) => item.name)).toContain('abmil');
  });

  it('reports no attention for a slide embedding', () => {
    expect(supportsAttention('slide_linear')).toBe(false);
    expect(supportsAttention('abmil')).toBe(true);
    expect(supportsAttention('abmil', 'clinical')).toBe(false);
  });

  it.each(['slide_linear', 'slide_mlp'])('omits patch-only settings and false pooling descriptions for %s', (model) => {
    const recipe = { ...defaultRecipe(), model };
    const html = renderToStaticMarkup(<RecipeFields value={recipe} onChange={() => {}} featureKind="slide" />);
    for (const label of ['Training bag', 'Patches per bag', 'Instance dropout', 'Grow the training bag', 'Limit validation and assessment patches', 'Fully connected layers', 'Gradient checkpointing']) expect(html).not.toContain(label);
    expect(html).toContain('Training record sampling');
    expect(html).toContain('Use a separate evaluation batch size');
    expect(html.includes('Hidden layer dimensions')).toBe(model === 'slide_mlp');
    const summary = renderToStaticMarkup(<RecipeSummary recipe={recipe} />);
    expect(summary).toContain('One embedding per slide');
    expect(summary).not.toContain('patches / bag');
    const spec = { ...batchTemplate('blank', { protocolId: 'p', featureBundleId: 'b', loadingPolicy: 'auto', packArtifactId: null }, 'Study'), recipe };
    const saved = renderToStaticMarkup(<BatchPlanSettings spec={spec} />);
    expect(saved).toContain(model === 'slide_linear' ? 'Linear read-out' : 'One hidden layer');
    expect(saved).not.toContain('Mean pooling');
    expect(saved).not.toContain('<dt>Training bag</dt>');
  });

  it('keeps clinical-only settings usable with a slide-embedding cohort', () => {
    const html = renderToStaticMarkup(<RecipeFields value={{ ...defaultRecipe(), inputMode: 'clinical', clinicalFields: [{ field: 'age', kind: 'numeric' }] }} onChange={() => {}} clinicalFields={['age']} featureKind="slide" />);
    expect(html).toContain('Clinical-only logistic baseline. Image-model choices do not apply.');
    expect(html).not.toContain('(unavailable)');
    expect(html).not.toContain('Image model<select');
    expect(html).toContain('Training record sampling');
  });
});
