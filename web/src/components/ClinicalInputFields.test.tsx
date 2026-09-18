import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { defaultRecipe } from '../api/development';
import { ClinicalInputFields, matchedInputRecipes } from './ClinicalInputFields';

describe('clinical input comparisons', () => {
  it('creates all three arms with the same typed clinical fields and optimization settings', () => {
    const value = { ...defaultRecipe(), inputMode: 'multimodal' as const,
      clinicalFields: [{ field: 'age', kind: 'numeric' as const }], learningRate: .005 };
    const arms = matchedInputRecipes(value);
    expect(arms.map((arm) => arm.inputMode)).toEqual(['image', 'clinical', 'multimodal']);
    expect(arms[0].clinicalFields).toEqual([]);
    expect(arms[1].clinicalFields).toEqual(value.clinicalFields);
    expect(arms[2].clinicalFields).toEqual(value.clinicalFields);
    expect(arms.every((arm) => arm.learningRate === .005)).toBe(true);
    expect(value.inputMode).toBe('multimodal');
  });

  it('explains the common cohort and requires declared fields', () => {
    const html = renderToStaticMarkup(<ClinicalInputFields value={{ ...defaultRecipe(), inputMode: 'clinical' }} onChange={() => {}} />);
    expect(html).toContain('Select at least one clinical field');
    expect(html).toContain('Clinical-only fitting does not read image embeddings');
    expect(html).toContain('Patients missing required image features');
    expect(html).toContain('Targets &amp; splits');
  });
});
