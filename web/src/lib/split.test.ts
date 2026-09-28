import { describe, expect, it } from 'vitest';
import type { ProtocolSpec } from '../api/scientific';
import { changeSplitStrategy, MAX_SPLIT_SEEDS, splitSeedsError } from './split';

const spec = (): ProtocolSpec['split'] => ({
  version: 4,
  mode: 'monte_carlo',
  seeds: [42],
  folds: 5,
  repeats: 5,
  outerFolds: 5,
  innerFolds: 3,
  testFraction: 0.2,
  validationFraction: 0.2,
  rules: { train: [], val: [], test: [] },
  ratios: { train: 0.8, val: 0.2, test: 0 },
  pools: {
    source: 'rules',
    trainSelection: 'remaining',
    validationSource: 'training_fraction',
    rules: { train: [], val: [], test: [] },
  },
});

describe('changing strategies after editing numeric settings', () => {
  it('repairs an invalid Monte Carlo assessment fraction and repeats when their controls disappear', () => {
    const before = { ...spec(), testFraction: 0, repeats: 0 };
    const after = { ...before, ...changeSplitStrategy(before, 'kfold') };
    expect(after).toMatchObject({ mode: 'kfold', testFraction: 0.2, repeats: 5 });
    expect(after.pools).toEqual(before.pools);
  });

  it('repairs invalid hidden nested and K-fold settings when switching to held-out', () => {
    const before = {
      ...spec(),
      mode: 'nested_kfold' as const,
      folds: 11,
      outerFolds: 0,
      innerFolds: 2.5,
      testFraction: Infinity,
    };
    expect({ ...before, ...changeSplitStrategy(before, 'held_out') }).toMatchObject({
      folds: 5,
      outerFolds: 5,
      innerFolds: 3,
      // A development holdout shows its assessment percentage, so the user corrects it.
      testFraction: Infinity,
    });
  });

  it('preserves valid hidden choices, including backend-valid fractions above the input hint', () => {
    const before = { ...spec(), repeats: 42, outerFolds: 7, innerFolds: 4, testFraction: 0.95 };
    expect({ ...before, ...changeSplitStrategy(before, 'kfold') }).toMatchObject({
      repeats: 42,
      outerFolds: 7,
      innerFolds: 4,
      testFraction: 0.95,
    });
  });

  it('leaves invalid visible controls available for the user to correct', () => {
    const before = { ...spec(), testFraction: 0, repeats: 0, validationFraction: 0 };
    expect({ ...before, ...changeSplitStrategy(before, 'monte_carlo') }).toMatchObject({
      testFraction: 0,
      repeats: 0,
      validationFraction: 0,
    });
    expect({ ...before, ...changeSplitStrategy(before, 'held_out') }).toMatchObject({
      testFraction: 0,
    });
  });
});

describe('split seed validation', () => {
  it('accepts up to ten distinct whole-number seeds, as the service does', () => {
    expect(MAX_SPLIT_SEEDS).toBe(10);
    expect(splitSeedsError('42')).toBe('');
    expect(splitSeedsError('0, 1, 2, 3, 4, 5, 6, 7, 8, 4294967295')).toBe('');
    expect(splitSeedsError('0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10')).toBe('Enter at most 10 split seeds; you entered 11.');
    expect(splitSeedsError('42, 42')).toBe('Each split seed must be different.');
    for (const text of ['', '42,', '1.5', '-1', '4294967296', 'x']) expect(splitSeedsError(text)).toContain('whole-number seeds');
  });
});
