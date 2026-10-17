import type { TrainingRecipe } from '../api/development';
import { modelOptions, modelSpec, type ModelOption } from '../lib/modelCapabilities';
import NumericField from './NumericField';

type RecipeProps = { value: TrainingRecipe; onChange: (recipe: TrainingRecipe) => void };
type OptionValue = number | boolean | string;

/** The recipe's value of one option, or its default when the recipe leaves it out. */
export function optionValue(recipe: TrainingRecipe, option: ModelOption): OptionValue {
  return (recipe as unknown as Record<string, OptionValue | undefined>)[option.key] ?? option.default;
}

/** "Smooth top-1 SVM", "on", or the number as entered. */
export function optionValueLabel(option: ModelOption, value: OptionValue): string {
  if (option.kind === 'choice') return option.choices.find((choice) => choice.value === value)?.label ?? String(value);
  if (option.kind === 'boolean') return value ? 'on' : 'off';
  return String(value);
}

export const optionEnabled = (recipe: TrainingRecipe, option: ModelOption): boolean => {
  if (!option.enabledWhen) return true;
  const { key, values } = option.enabledWhen;
  const controller = modelOptions(recipe.model).find((item) => item.key === key);
  return controller ? values.includes(optionValue(recipe, controller)) : true;
};

/**
 * "Transformer layers 2 · Attention heads 8": the options that shape the model, then any
 * other option changed from its default. `changed` lists only changed options, and `all`
 * every option, for saved settings.
 */
export function modelOptionSummary(recipe: TrainingRecipe, which: 'shape' | 'changed' | 'all' = 'shape'): string[] {
  return modelOptions(recipe.model)
    .filter((option) => which === 'all' || (which === 'shape' && option.memory) || optionValue(recipe, option) !== option.default)
    .map((option) => `${option.label} ${optionValueLabel(option, optionValue(recipe, option))}`);
}

function OptionField({ option, value, disabled, onChange }: { option: ModelOption; value: OptionValue; disabled: boolean; onChange: (value: OptionValue) => void }) {
  const control = option.kind === 'boolean'
    ? <label className="development-check"><input type="checkbox" checked={Boolean(value)} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />{option.label}</label>
    : option.kind === 'choice'
      ? <label className="label">{option.label}<select className="field" value={String(value)} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {option.choices.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
      </select></label>
      : <NumericField label={option.label} value={Number(value)} integer={option.kind === 'integer'} disabled={disabled}
        min={option.minimum ?? undefined} max={option.maximum ?? undefined} maxExclusive={option.maximumExclusive} onChange={onChange} />;
  return <div className="batch-check-with-hint">{control}<small>{option.help}</small></div>;
}

/** The chosen architecture's own settings, with the published method's defaults. */
export function ModelOptionFields({ value, onChange }: RecipeProps) {
  const spec = modelSpec(value.model);
  if (!spec?.options.length || value.inputMode === 'clinical') return null;
  const customized = spec.options.some((option) => optionValue(value, option) !== option.default);
  const defaults = Object.fromEntries(spec.options.map((option) => [option.key, option.default]));
  return <section className="batch-editor-section" aria-label={`${spec.label} settings`}>
    <div className="batch-section-heading"><h3>{spec.label} settings</h3><p>{spec.summary} Defaults follow the published method{spec.reference ? ` (${spec.reference})` : ''}.{spec.auxiliaryLoss ? ' Its extra training losses use this recipe’s loss, class weights and patient weights.' : ''}</p></div>
    <div className="development-fields">
      {spec.options.map((option) => <OptionField key={option.key} option={option} value={optionValue(value, option)} disabled={!optionEnabled(value, option)}
        onChange={(next) => onChange({ ...value, [option.key]: next })} />)}
    </div>
    <div className="inline-actions"><button type="button" className="btn btn-secondary" disabled={!customized} onClick={() => onChange({ ...value, ...defaults })}>Restore {spec.label} defaults</button></div>
  </section>;
}
