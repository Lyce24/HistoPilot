import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { defaultRecipe, nnmilRecipe, type ClinicalFieldChoice } from '../api/development';
import { ClinicalInputFields, matchedInputRecipes, toggleClinicalField } from './ClinicalInputFields';

const choices: ClinicalFieldChoice[] = [
  { field: 'age', owner: 'patient', type: 'integer', kind: 'numeric', excluded: null },
  { field: 'site', owner: 'slide', type: 'categorical', kind: 'categorical', excluded: null },
  { field: 'status', owner: 'patient', type: 'categorical', kind: 'categorical', excluded: 'It is the prediction target.' },
  { field: 'cohort', owner: 'slide', type: 'text', kind: 'categorical', excluded: 'It defines the training and testing split.' },
];
const clinical = (changes = {}) => ({ ...defaultRecipe(), inputMode: 'clinical' as const, ...changes });
const text = (html: string) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

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

  it('turns nnMIL-only choices off for a clinical-only arm, which the service requires', () => {
    const [, arm] = matchedInputRecipes({ ...nnmilRecipe(), inputMode: 'multimodal', clinicalFields: [{ field: 'age', kind: 'numeric' }], nnmilBatchSampler: 'class_balanced' });
    expect(arm).toMatchObject({ model: 'abmil', inputMode: 'clinical', nnmilWindowSeedFromTraining: false, nnmilBatchSampler: 'patient_weighted', bagSizeMode: 'fixed' });
  });

  it('explains the common cohort and requires a selected field', () => {
    const html = renderToStaticMarkup(<ClinicalInputFields value={clinical()} onChange={() => {}} choices={{ fields: choices }} />);
    expect(html).toContain('Select at least one clinical field');
    expect(html).toContain('Clinical-only fitting does not read image embeddings');
    expect(html).toContain('Patients missing required image features');
    // The old dead end asked for inputs that new target/split versions cannot declare.
    expect(html).not.toContain('Targets &amp; splits');
    expect(html).not.toContain('Declare extra spreadsheet inputs');
  });

  it('lists the dataset’s fields with owner and type, and shows refused fields disabled with the reason', () => {
    const html = renderToStaticMarkup(<ClinicalInputFields value={clinical({ clinicalFields: [{ field: 'age', kind: 'numeric' }] })} onChange={() => {}} choices={{ fields: choices }} />);
    const plain = text(html);
    expect(plain).toContain('Clinical fields in the frozen dataset');
    expect(plain).toContain('age Patient attribute · whole number');
    expect(plain).toContain('site Slide attribute · category');
    expect(plain).toContain('status Patient attribute · category Not available: It is the prediction target.');
    expect(plain).toContain('cohort Slide attribute · text Not available: It defines the training and testing split.');
    expect(html).toMatch(/<input type="checkbox" checked=""\/><span><strong>age</);
    expect(html).toMatch(/<input type="checkbox" disabled=""\/><span><strong>status</);
    expect(html).toMatch(/<input type="checkbox" disabled=""\/><span><strong>cohort</);
    expect(html).toMatch(/<input type="checkbox"\/><span><strong>site</);
    expect(html).toContain('Use age as<select');
    // Usable fields come first; refused ones follow.
    expect(plain.indexOf('site Slide')).toBeLessThan(plain.indexOf('status Patient'));
  });

  it('starts a chosen field with the kind the dataset suggests, and clears it again', () => {
    const chosen = toggleClinicalField(clinical(), choices[0], true);
    expect(chosen.clinicalFields).toEqual([{ field: 'age', kind: 'numeric' }]);
    const both = toggleClinicalField(chosen, choices[1], true);
    expect(both.clinicalFields).toEqual([{ field: 'age', kind: 'numeric' }, { field: 'site', kind: 'categorical' }]);
    expect(toggleClinicalField(both, choices[0], false).clinicalFields).toEqual([{ field: 'site', kind: 'categorical' }]);
  });

  it('flags saved fields the split now refuses or no longer offers', () => {
    const html = text(renderToStaticMarkup(<ClinicalInputFields value={clinical({ clinicalFields: [{ field: 'status', kind: 'categorical' }, { field: 'stage', kind: 'numeric' }] })} onChange={() => {}} choices={{ fields: choices }} />));
    expect(html).toContain('status, stage cannot be used with this experiment’s frozen split. It is the prediction target.');
    expect(html).toContain('Remove unavailable fields');
  });

  it('shows loading and error states instead of an empty list', () => {
    expect(renderToStaticMarkup(<ClinicalInputFields value={clinical()} onChange={() => {}} choices={{ loading: true }} />)).toContain('Loading the dataset’s clinical fields…');
    const failed = text(renderToStaticMarkup(<ClinicalInputFields value={clinical()} onChange={() => {}} choices={{ error: new Error('Dataset unavailable.'), onRetry: () => {} }} />));
    expect(failed).toContain('The dataset’s clinical fields could not be loaded. Dataset unavailable.');
    expect(failed).toContain('Try again');
    expect(renderToStaticMarkup(<ClinicalInputFields value={clinical()} onChange={() => {}} choices={{ fields: [] }} />)).toContain('no numeric or categorical columns');
    expect(renderToStaticMarkup(<ClinicalInputFields value={clinical()} onChange={() => {}} />)).toContain('once this experiment’s targets and splits are chosen');
    // Image-only recipes do not ask for clinical fields at all.
    expect(renderToStaticMarkup(<ClinicalInputFields value={defaultRecipe()} onChange={() => {}} choices={{ loading: true }} />)).not.toContain('Loading');
  });
});
