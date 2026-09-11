import { describe, expect, it } from 'vitest';
import { categoryLabel } from './datasetValues';

describe('literal and missing category labels', () => {
  it('distinguishes null, empty strings, whitespace and literal missing markers', () => {
    const values = [null, '', ' ', '(missing)', '(missing value)', '"(missing value)"'];
    expect(new Set(values.map(categoryLabel)).size).toBe(values.length);
    for (const value of values.filter((item) => item !== null)) {
      expect(JSON.parse(categoryLabel(value))).toBe(value);
    }
  });
});
