import { describe, expect, it } from 'vitest';
import { inferTargetSettings, newDevelopmentSplit, preservePositiveClass } from './protocol';

describe('new development protocols', () => {
  it('starts with all eligible training records and no reserved population', () => {
    expect(newDevelopmentSplit([9, 27], 3)).toMatchObject({
      version: 4, mode: 'kfold', folds: 3, seeds: [9, 27],
      pools: {
        source: 'rules', trainSelection: 'remaining', validationSource: 'training_fraction',
        rules: { train: [], val: [], test: [] },
      },
    });
  });

  it('creates independent drafts so selections cannot leak across experiments', () => {
    const first = newDevelopmentSplit();
    first.pools!.rules.train.push({ field: 'cohort', op: 'eq', value: 'development' });
    expect(newDevelopmentSplit().pools!.rules.train).toEqual([]);
  });
});

describe('positive class when suggesting source-value labels', () => {
  it('preserves high when discovered values reorder the original low/high classes', () => {
    expect(preservePositiveClass('high', ['high', 'low'])).toBe('high');
    expect(preservePositiveClass('low', ['low', 'high'])).toBe('low');
  });

  it('requires an explicit choice if the previous positive class is absent or unset', () => {
    expect(preservePositiveClass('high', ['0', '1'])).toBeUndefined();
    expect(preservePositiveClass(undefined, ['high', 'low'])).toBeUndefined();
  });
});

describe('target suggestions from observed values', () => {
  it('starts unconfigured without complete values', () => {
    expect(inferTargetSettings([])).toEqual({
      task: '',
      classes: [],
      labels: {},
      positiveClass: undefined,
    });
    expect(inferTargetSettings(['A', 'B'], true)).toEqual({
      task: '',
      classes: [],
      labels: {},
      positiveClass: undefined,
    });
  });

  it('infers binary classes but requires an explicit positive outcome', () => {
    expect(inferTargetSettings([null, '0', '1', '0', '', ' '])).toEqual({
      task: 'binary_classification',
      classes: ['0', '1'],
      labels: { '0': '0', '1': '1' },
      positiveClass: undefined,
    });
  });

  it('infers multiclass classification with no positive class', () => {
    expect(inferTargetSettings(['group C', 'group A', 'group B'])).toEqual({
      task: 'multiclass_classification',
      classes: ['group C', 'group A', 'group B'],
      labels: { 'group C': 'group C', 'group A': 'group A', 'group B': 'group B' },
      positiveClass: undefined,
    });
  });

  it('retains a single source class but requires a task and another class', () => {
    expect(inferTargetSettings([null, 'positive', 'positive'])).toEqual({
      task: '',
      classes: ['positive'],
      labels: { positive: 'positive' },
      positiveClass: undefined,
    });
  });

  it('preserves spelling and distinct source values', () => {
    expect(inferTargetSettings(['High', 'high'])).toEqual({
      task: 'binary_classification',
      classes: ['High', 'high'],
      labels: { High: 'High', high: 'high' },
      positiveClass: undefined,
    });
  });

  it('never infers the positive outcome from frequency or source ordering', () => {
    for (const values of [['negative', 'positive'], ['positive', 'negative'], ['1', '0']]) {
      expect(inferTargetSettings(values).positiveClass).toBeUndefined();
    }
  });
});
