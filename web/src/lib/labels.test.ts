import { describe, expect, it } from 'vitest';
import { splitModeLabel, taskLabel, unitLabel } from './labels';

describe('enum labels', () => {
  it('names known values and never returns raw snake_case', () => {
    expect(taskLabel('binary_classification')).toBe('Binary classification');
    expect(unitLabel('patient')).toBe('Patient-level');
    expect(splitModeLabel('kfold')).toBe('K-fold cross-validation');
    expect(splitModeLabel('nested_kfold')).toBe('Nested cross-validation');
    expect(splitModeLabel('leave_one_domain_out')).toBe('Leave-one-domain-out');
    expect(taskLabel('ordinal_regression')).toBe('Ordinal regression');
  });
});
