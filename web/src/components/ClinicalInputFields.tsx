import type { TrainingRecipe } from '../api/development';

export const inputModeLabel = (mode?: TrainingRecipe['inputMode']) => ({
  image: 'Image only', clinical: 'Clinical only', multimodal: 'Clinical + image',
}[mode ?? 'image']);

export function ClinicalInputFields({ value, onChange, availableFields = [] }: {
  value: TrainingRecipe; onChange: (value: TrainingRecipe) => void; availableFields?: string[];
}) {
  const fields = value.clinicalFields ?? [];
  const mode = value.inputMode ?? 'image';
  return <section className="batch-editor-section" aria-label="Model inputs">
    <h3>Model inputs</h3>
    <label className="label">Prediction inputs<select className="field" value={mode} onChange={(event) => {
      const inputMode = event.target.value as TrainingRecipe['inputMode'];
      onChange({ ...value, inputMode, clinicalFields: inputMode === 'image' ? [] : fields,
        ...(inputMode === 'clinical' ? { model: 'abmil', bagSizeMode: 'fixed', bagCurriculum: false,
          nnmilBatchSampler: 'patient_weighted', nnmilCheckpointSelection: 'best_validation' } as const : {}) });
    }}><option value="image">Image only</option><option value="clinical">Clinical only — logistic baseline</option><option value="multimodal">Clinical + image — additive logit fusion</option></select></label>
    {mode !== 'image' ? <>
      <p>Choose variables available when the prediction will be made. Numeric values use training-patient median imputation and scaling; categorical values use training categories plus missing and unseen-category indicators. Every fold saves its own transform.</p>
      {availableFields.length ? <fieldset><legend>Declared clinical fields</legend>{availableFields.map((field) => {
        const selected = fields.find((item) => item.field === field);
        return <div className="development-fields" key={field}>
          <label className="development-check"><input type="checkbox" checked={Boolean(selected)} onChange={(event) => onChange({ ...value,
            clinicalFields: event.target.checked ? [...fields, { field, kind: 'categorical' }] : fields.filter((item) => item.field !== field),
          })} />{field}</label>
          {selected ? <label className="label">{field} type<select className="field" value={selected.kind} onChange={(event) => onChange({ ...value,
            clinicalFields: fields.map((item) => item.field === field ? { ...item, kind: event.target.value as 'numeric' | 'categorical' } : item),
          })}><option value="numeric">Numeric</option><option value="categorical">Categorical</option></select></label> : null}
        </div>;
      })}</fieldset> : <p className="callout">Declare extra spreadsheet inputs in Targets &amp; splits, freeze that protocol, and select it in this experiment.</p>}
      {fields.some((item) => !availableFields.includes(item.field)) ? <p className="callout">This recipe contains fields absent from the current protocol. Remove them or select the protocol that declares them. <button type="button" className="text-button" onClick={() => onChange({ ...value, clinicalFields: fields.filter((item) => availableFields.includes(item.field)) })}>Remove unavailable fields</button></p> : null}
      {!fields.length ? <p className="callout">Select at least one clinical field before checking this batch.</p> : null}
    </> : null}
    <p className="muted">All input comparisons use the same frozen patients, splits and feature-covered cohort. Patients missing required image features must be resolved in the common cohort before any arm runs. Clinical-only fitting does not read image embeddings or use image signal.</p>
  </section>;
}

export function matchedInputRecipes(value: TrainingRecipe): TrainingRecipe[] {
  if (!value.clinicalFields?.length) throw new Error('Select clinical fields before creating matched arms.');
  return [
    { ...value, inputMode: 'image', clinicalFields: [] },
    { ...value, inputMode: 'clinical', model: 'abmil', bagSizeMode: 'fixed', bagCurriculum: false,
      nnmilBatchSampler: 'patient_weighted', nnmilCheckpointSelection: 'best_validation' },
    { ...value, inputMode: 'multimodal' },
  ];
}
