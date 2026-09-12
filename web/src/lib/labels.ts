/** Human-readable names for enum values that must never reach the screen raw. */
const TASKS: Record<string, string> = {
  binary_classification: 'Binary classification',
  multiclass_classification: 'Multiclass classification',
};
const UNITS: Record<string, string> = { patient: 'Patient-level', slide: 'Slide-level' };
const SPLIT_MODES: Record<string, string> = {
  kfold: 'K-fold cross-validation',
  monte_carlo: 'Repeated random splits',
  held_out: 'Single held-out validation',
  holdout: 'Single held-out validation',
  nested: 'Nested cross-validation',
  nested_kfold: 'Nested cross-validation',
  lodo: 'Leave-one-domain-out',
  leave_one_domain_out: 'Leave-one-domain-out',
  imported: 'Imported split assignments',
  rules: 'Rule-based assignments',
};
const fallback = (value: string) => value.replace(/[_-]+/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
export const taskLabel = (value: string) => TASKS[value] ?? fallback(value);
export const unitLabel = (value: string) => UNITS[value] ?? fallback(value);
export const splitModeLabel = (value: string) => SPLIT_MODES[value] ?? fallback(value);
