import { describe, expect, it } from 'vitest';
import type { ProtocolSpec } from '../api/scientific';
import {
  changeHeldOutSource,
  changeSplitStrategy,
  changeValidationSource,
  resetImportedForDataset,
} from './split';

const spec = (): ProtocolSpec['split'] => ({
  version: 3,
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
      testFraction: 0.2,
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
    expect({
      ...before,
      ...changeSplitStrategy({ ...before, version: 2 }, 'held_out'),
    }).toMatchObject({ testFraction: 0 });
  });
});

describe('fixed validation source', () => {
  it('returns the source and hidden fraction correction together', () => {
    const pools = spec().pools!;
    expect(changeValidationSource(pools, 'fixed', 0)).toEqual({
      pools: { validationSource: 'fixed', rules: pools.rules },
      validationFraction: 0.15,
    });
  });

  it('retains a valid fraction and leaves an invalid visible fraction editable', () => {
    expect(
      changeValidationSource(spec().pools!, 'fixed', 0.35).validationFraction,
    ).toBeUndefined();
    expect(
      changeValidationSource(spec().pools!, 'training_fraction', 0).validationFraction,
    ).toBeUndefined();
  });
});

describe('predefined columns after changing datasets', () => {
  it('keeps legacy and version 2 imported editors available with empty mappings', () => {
    const empty = { partitionLabels: {}, foldLabels: {}, testFoldLabels: [] };
    expect(resetImportedForDataset({ ...spec(), version: 1, mode: 'imported' })).toEqual(empty);
    expect(
      resetImportedForDataset({
        ...spec(),
        version: 2,
        mode: 'held_out',
        heldOutSource: 'imported',
      }),
    ).toEqual(empty);
  });

  it('does not add unused imported settings to version 3 or generated strategies', () => {
    expect(resetImportedForDataset(spec())).toBeUndefined();
    expect(resetImportedForDataset({ ...spec(), version: 2, mode: 'kfold' })).toBeUndefined();
  });
});

it('repairs a hidden random test fraction when an older held-out design switches to predefined sets', () => {
  const before = { ...spec(), version: 2 as const, mode: 'held_out' as const, testFraction: 0 };
  expect(changeHeldOutSource(before, 'imported')).toMatchObject({
    testFraction: 0.2,
    heldOutSource: 'imported',
    imported: { partitionLabels: {} },
  });
  expect(changeHeldOutSource({ ...before, testFraction: 0.3 }, 'rules')).not.toHaveProperty(
    'testFraction',
  );
});
