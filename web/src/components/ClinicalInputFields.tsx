import { coupling } from '../lib/templates';
import type { ClinicalFieldChoice, TrainingRecipe } from '../api/development';

export const inputModeLabel = (mode?: TrainingRecipe['inputMode']) => ({
  image: 'Image only', clinical: 'Clinical only', multimodal: 'Clinical + image',
}[mode ?? 'image']);

/** The dataset columns this experiment's frozen split offers, and the state of loading them. */
export interface ClinicalChoices {
  fields?: ClinicalFieldChoice[]; loading?: boolean; error?: Error | null; onRetry?: () => void;
}

const typeLabels: Record<string, string> = {
  integer: 'whole number', decimal: 'decimal number', boolean: 'yes / no', categorical: 'category',
  ordered_categorical: 'ordered category', text: 'text',
};
export const clinicalFieldDescription = (choice: ClinicalFieldChoice) =>
  `${choice.owner === 'patient' ? 'Patient attribute' : 'Slide attribute'} · ${typeLabels[choice.type] ?? choice.type}`;

/** Selects or clears one field; a newly selected field starts with the kind the dataset suggests. */
export function toggleClinicalField(value: TrainingRecipe, choice: ClinicalFieldChoice, selected: boolean): TrainingRecipe {
  const fields = value.clinicalFields ?? [];
  return { ...value, clinicalFields: selected
    ? [...fields.filter((item) => item.field !== choice.field), { field: choice.field, kind: choice.kind }]
    : fields.filter((item) => item.field !== choice.field) };
}

/** A clinical-only arm fits a logistic model on the clinical fields; image settings and nnMIL options do not apply. */
export const clinicalOnlyRecipe = (value: TrainingRecipe): TrainingRecipe => ({ ...value, ...coupling.clinicalOnly });

export function ClinicalInputFields({ value, onChange, choices }: {
  value: TrainingRecipe; onChange: (value: TrainingRecipe) => void; choices?: ClinicalChoices;
}) {
  const fields = value.clinicalFields ?? [];
  const mode = value.inputMode ?? 'image';
  const offered = choices?.fields;
  const byName = new Map(offered?.map((choice) => [choice.field, choice]));
  // Fields a saved recipe names that this split no longer offers, or now refuses.
  const unavailable = offered ? fields.filter((item) => !byName.get(item.field) || byName.get(item.field)?.excluded) : [];
  const ordered = offered ? [...offered.filter((choice) => !choice.excluded), ...offered.filter((choice) => choice.excluded)] : [];
  return <section className="batch-editor-section" aria-label="Model inputs">
    <h3>Model inputs</h3>
    <label className="label">Prediction inputs<select className="field" value={mode} onChange={(event) => {
      const inputMode = event.target.value as TrainingRecipe['inputMode'];
      const next = { ...value, inputMode, clinicalFields: inputMode === 'image' ? [] : fields };
      onChange(inputMode === 'clinical' ? clinicalOnlyRecipe(next) : next);
    }}><option value="image">Image only</option><option value="clinical">Clinical only — logistic baseline</option><option value="multimodal">Clinical + image — additive logit fusion</option></select></label>
    {mode !== 'image' ? <>
      <p>Choose variables that are known when the prediction is made. Missing numeric values are filled with the training data’s median and scaled on training data only; categorical values use the training categories plus indicators for missing and unseen values. Every fold fits and saves its own transform.</p>
      {choices?.loading ? <p className="muted" role="status">Loading the dataset’s clinical fields…</p>
        : choices?.error ? <div className="callout callout-warning" role="alert"><p>The dataset’s clinical fields could not be loaded. {choices.error.message}</p>{choices.onRetry ? <button type="button" className="text-button" onClick={choices.onRetry}>Try again</button> : null}</div>
          : !offered ? <p className="callout">Clinical fields are listed once this experiment’s targets and splits are chosen in Inputs.</p>
            : !offered.length ? <p className="callout">The frozen dataset has no numeric or categorical columns to use as clinical inputs.</p>
              : <fieldset className="clinical-field-list"><legend>Clinical fields in the frozen dataset</legend>
                {ordered.map((choice) => {
                  const selected = fields.find((item) => item.field === choice.field);
                  const blocked = Boolean(choice.excluded);
                  return <div className={`clinical-field-row${blocked ? ' is-excluded' : ''}`} key={choice.field}>
                    <label className="development-check"><input type="checkbox" checked={Boolean(selected)} disabled={blocked && !selected}
                      onChange={(event) => onChange(toggleClinicalField(value, choice, event.target.checked))} />
                      <span><strong>{choice.field}</strong><small>{clinicalFieldDescription(choice)}</small></span></label>
                    {selected && !blocked ? <label className="label">Use {choice.field} as<select className="field" value={selected.kind} onChange={(event) => onChange({ ...value,
                      clinicalFields: fields.map((item) => item.field === choice.field ? { ...item, kind: event.target.value as 'numeric' | 'categorical' } : item),
                    })}><option value="numeric">Numeric</option><option value="categorical">Categorical</option></select></label> : null}
                    {blocked ? <small className="clinical-field-reason">Not available: {choice.excluded}</small> : null}
                  </div>;
                })}
              </fieldset>}
      {unavailable.length ? <p className="callout callout-warning" role="alert">{unavailable.map((item) => item.field).join(', ')} cannot be used with this experiment’s frozen split. {unavailable.map((item) => byName.get(item.field)?.excluded).filter(Boolean).join(' ')} <button type="button" className="text-button" onClick={() => onChange({ ...value, clinicalFields: fields.filter((item) => !unavailable.includes(item)) })}>Remove unavailable fields</button></p> : null}
      {!fields.length ? <p className="callout">Select at least one clinical field before checking this batch.</p> : null}
    </> : null}
    <p className="muted">All input comparisons use the same frozen patients, splits and feature-covered cohort. Patients missing required image features must be resolved in the common cohort before any arm runs. Clinical-only fitting does not read image embeddings or use image signal.</p>
  </section>;
}

export function matchedInputRecipes(value: TrainingRecipe): TrainingRecipe[] {
  if (!value.clinicalFields?.length) throw new Error('Select clinical fields before creating matched arms.');
  return [
    { ...value, inputMode: 'image', clinicalFields: [] },
    clinicalOnlyRecipe(value),
    { ...value, inputMode: 'multimodal' },
  ];
}
