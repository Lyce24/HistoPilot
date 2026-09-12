import type { ProtocolSpec } from '../api/scientific';
import { DEFAULT_VALIDATION_FRACTION } from './split';

export function newDevelopmentSplit(seeds: number[] = [42], folds = 5): ProtocolSpec['split'] {
  return {
    version: 4,
    pools: {
      source: 'rules',
      trainSelection: 'remaining',
      validationSource: 'training_fraction',
      rules: { train: [], val: [], test: [] },
    },
    mode: 'kfold',
    folds,
    seeds,
    stratify: true,
    validationFraction: DEFAULT_VALIDATION_FRACTION,
    testFraction: 0.2,
    repeats: 5,
    outerFolds: 5,
    innerFolds: 3,
    domainPolicy: 'all',
    heldOutDomains: [],
    heldOutSource: 'fractions',
    ratios: { train: 0.8, val: 0.2, test: 0 },
    rules: { train: [], val: [], test: [] },
  };
}

/** Retain an existing positive-class choice only while it is still a class. */
export function preservePositiveClass(
  current: string | undefined,
  classes: string[],
): string | undefined {
  return current !== undefined && classes.includes(current) ? current : undefined;
}

/** Initial suggestions use complete, non-missing source values without changing their spelling. */
export function inferTargetSettings(values: (string | null)[], truncated = false) {
  const classes = truncated
    ? []
    : [
        ...new Set(
          values.filter((value): value is string => value !== null && value.trim() !== ''),
        ),
      ];
  const task: '' | 'binary_classification' | 'multiclass_classification' =
    classes.length === 2
      ? 'binary_classification'
      : classes.length > 2
        ? 'multiclass_classification'
        : '';
  return {
    task,
    classes,
    labels: Object.fromEntries(classes.map((value) => [value, value])),
    // Value ordering is not evidence of which clinical outcome is positive.
    positiveClass: undefined as string | undefined,
  };
}
