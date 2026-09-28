import type { ProtocolSpec } from '../api/scientific';

type Split = ProtocolSpec['split'];

export const DEFAULT_VALIDATION_FRACTION = 0.15;
/** The service's limit on split seeds per training design (`SplitSpec.seeds`). */
export const MAX_SPLIT_SEEDS = 10;
const MAX_SEED = 2 ** 32 - 1;

/** Why comma-separated split seeds cannot be saved, or '' when they can. */
export function splitSeedsError(text: string): string {
  const entries = text.split(',').map((value) => value.trim());
  if (entries.some((value) => !/^\d+$/.test(value) || Number(value) > MAX_SEED)) return `Enter whole-number seeds from 0 to ${MAX_SEED}, separated by commas.`;
  if (new Set(entries.map(Number)).size !== entries.length) return 'Each split seed must be different.';
  if (entries.length > MAX_SPLIT_SEEDS) return `Enter at most ${MAX_SPLIT_SEEDS} split seeds; you entered ${entries.length}.`;
  return '';
}

const validFraction = (value: number) =>
  Number.isFinite(value) && value > 0 && value < 1;
const validInteger = (value: number, min: number, max: number) =>
  Number.isInteger(value) && value >= min && value <= max;

/**
 * Unused invalid controls must not block a development (version 4) strategy after their
 * inputs disappear.
 */
export function changeSplitStrategy(split: Split, mode: Split['mode']): Partial<Split> {
  const next: Partial<Split> = {
    mode,
    rules: { train: [], val: [], test: [] },
    imported: undefined,
    heldOutSource: 'fractions',
    domainField: undefined,
    domainPolicy: 'all',
    heldOutDomains: [],
  };
  if (mode !== 'kfold' && !validInteger(split.folds, 2, 10)) next.folds = 5;
  if (mode !== 'nested_kfold') {
    if (split.outerFolds !== undefined && !validInteger(split.outerFolds, 2, 10))
      next.outerFolds = 5;
    if (split.innerFolds !== undefined && !validInteger(split.innerFolds, 2, 10))
      next.innerFolds = 3;
  }
  if (
    mode !== 'monte_carlo' &&
    split.repeats !== undefined &&
    !validInteger(split.repeats, 1, 100)
  )
    next.repeats = 5;
  const showsTestFraction = mode === 'monte_carlo' || mode === 'held_out';
  if (
    !showsTestFraction &&
    split.testFraction !== undefined &&
    !validFraction(split.testFraction)
  )
    next.testFraction = 0.2;
  if (
    split.pools?.validationSource === 'fixed' &&
    split.validationFraction !== undefined &&
    !validFraction(split.validationFraction)
  )
    next.validationFraction = DEFAULT_VALIDATION_FRACTION;
  return next;
}
