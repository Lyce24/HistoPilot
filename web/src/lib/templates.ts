/**
 * Server-owned starters and presets for every spec the browser builds.
 *
 * `templates.json` is written from `histopilot/templates.py` by
 * `scripts/export_templates.py` and checked against it by `tests/test_templates.py`.
 * The CLI and agent tools read the same values from the service, so a spec started in
 * any of them begins from the same science.
 */
import type { TrainingRecipe } from '../api/development';
import data from './templates.json';

export const templates = data;

/** A fresh copy that callers may change freely. */
export const fromTemplate = <T>(value: unknown): T => JSON.parse(JSON.stringify(value)) as T;

type RecipePreset = { base: ('default' | 'experimental')[]; values: Partial<TrainingRecipe> };

/** A recipe preset: its base recipes merged in order, then its own values. */
export function recipePreset(name: string): TrainingRecipe {
  const preset = (templates.recipes.presets as Record<string, RecipePreset>)[name];
  const bases = preset.base.map((base) => base === 'default' ? templates.recipes.default : templates.recipes.experimentalDefaults);
  return fromTemplate<TrainingRecipe>(Object.assign({}, ...bases, preset.values));
}

export type Coupling = Partial<TrainingRecipe>;
export const coupling = templates.coupling as unknown as {
  modelSelect: { nnmil: Coupling; other: Coupling; slide: Coupling };
  comparisonArm: { nnmil: Coupling; nnmilOff: Coupling };
  clinicalOnly: Coupling;
  nonCosineSchedule: Coupling;
};
