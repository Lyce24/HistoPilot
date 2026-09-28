import { describe, expect, it } from 'vitest';
import type { ProtocolSpec } from '../api/scientific';
import { changeSplitStrategy } from './split';

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
