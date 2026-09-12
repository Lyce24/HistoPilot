import type { ProtocolSpec } from '../api/scientific';

type Split = ProtocolSpec['split'];
type Pools = NonNullable<Split['pools']>;

export const DEFAULT_VALIDATION_FRACTION = 0.15;
export const validationFractionDefault = (version: Split['version']) =>
  (version ?? 1) >= 3 ? DEFAULT_VALIDATION_FRACTION : 0.2;

export const validFraction = (value: number) =>
  Number.isFinite(value) && value > 0 && value < 1;
const validInteger = (value: number, min: number, max: number) =>
  Number.isInteger(value) && value >= min && value <= max;

/** Unused invalid controls must not block a strategy after their inputs disappear. */
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
  const showsTestFraction =
    mode === 'monte_carlo' || (mode === 'held_out' && split.version !== 3);
  if (
    !showsTestFraction &&
    split.testFraction !== undefined &&
    !validFraction(split.testFraction)
  )
    next.testFraction = 0.2;
  if (
    (split.version ?? 1) >= 3 &&
    split.pools?.validationSource === 'fixed' &&
    split.validationFraction !== undefined &&
    !validFraction(split.validationFraction)
  )
    next.validationFraction = DEFAULT_VALIDATION_FRACTION;
  return next;
}

/** Return a single update so changing validation source cannot discard its fraction fix. */
export function changeValidationSource(
  pools: Pools,
  source: Pools['validationSource'],
  fraction: number,
) {
  return {
    pools: {
      validationSource: source,
      rules: { ...pools.rules, val: [] },
    } satisfies Partial<Pools>,
    validationFraction:
      source === 'fixed' && !validFraction(fraction) ? DEFAULT_VALIDATION_FRACTION : undefined,
  };
}

/** Keep the predefined-column editor mounted while removing references to the previous dataset. */
export function resetImportedForDataset(split: Split): Split['imported'] {
  return split.mode === 'imported' ||
    (split.version === 2 && split.mode === 'held_out' && split.heldOutSource === 'imported')
    ? { partitionLabels: {}, foldLabels: {}, testFoldLabels: [] }
    : undefined;
}

/** Predefined or rule-based held-out sets hide the random test percentage. */
export function changeHeldOutSource(
  split: Split,
  source: Split['heldOutSource'],
): Partial<Split> {
  return {
    heldOutSource: source,
    rules: { train: [], val: [], test: [] },
    imported:
      source === 'imported'
        ? { partitionLabels: {}, foldLabels: {}, testFoldLabels: [] }
        : undefined,
    ...(source !== 'fractions' &&
    split.testFraction !== undefined &&
    !validFraction(split.testFraction)
      ? { testFraction: 0.2 }
      : {}),
  };
}
