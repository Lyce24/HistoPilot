import { describe, expect, it } from 'vitest';
import { validateBatchNumberList } from './BatchNumberList';

describe('batch parameter lists', () => {
  it('accepts decimal scientific notation and seed zero without accepting partial numeric text', () => {
    const rates = { label: 'Learning rates', integer: false, min: 0, minExclusive: true };
    expect(validateBatchNumberList('1e-4, 0.0003, .001', rates)).toBe('');
    for (const text of ['', '1e-', '1e-4,', '0x10', 'Infinity', '0', '-1']) expect(validateBatchNumberList(text, rates)).not.toBe('');
    expect(validateBatchNumberList('0, 42, 4294967295', { label: 'Training seeds', min: 0, max: 2 ** 32 - 1 })).toBe('');
  });

  it('rejects duplicate values, fractional or out-of-range seeds and excessive epoch limits', () => {
    const seeds = { label: 'Training seeds', min: 0, max: 2 ** 32 - 1 };
    for (const text of ['42, 42', '1.2', '4294967296', '-1']) expect(validateBatchNumberList(text, seeds)).not.toBe('');
    expect(validateBatchNumberList('1e-4, 0.0001', { label: 'Learning rates', integer: false, min: 0, minExclusive: true })).toContain('distinct');
    expect(validateBatchNumberList('50, 100001', { label: 'Maximum epochs', min: 1, max: 100000 })).toContain('at most 100000');
  });

  it('keeps an empty GPU list as CPU mode while enforcing GPU IDs and list limits', () => {
    const gpu = { label: 'Allowed GPU IDs', min: 0, max: 127, maxItems: 128, allowEmpty: true };
    expect(validateBatchNumberList('', gpu)).toBe('');
    expect(validateBatchNumberList('0, 127', gpu)).toBe('');
    expect(validateBatchNumberList('128', gpu)).not.toBe('');
    expect(validateBatchNumberList('0, 1, 2', { ...gpu, maxItems: 2 })).toContain('at most 2 values');
  });
});
