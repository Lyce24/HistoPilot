import { describe, expect, it } from 'vitest';
import { sameJSON } from './json';

describe('saved scientific draft comparison', () => {
  it('keeps a saved draft clean when storage reorders nested JSON object keys', () => {
    const edited = {
      source: { path: '/allowed/bladder.csv', sheet: 'Sheet1' },
      attributes: [{ sourceColumn: 'WHO 2022', key: 'grade', owner: 'slide' }],
      recursive: true,
    };
    const saved = {
      attributes: [{ key: 'grade', owner: 'slide', sourceColumn: 'WHO 2022' }],
      recursive: true,
      source: { sheet: 'Sheet1', path: '/allowed/bladder.csv' },
    };
    expect(sameJSON(edited, saved)).toBe(true);
  });

  it('treats omitted optional fields exactly like JSON transport does', () => {
    expect(
      sameJSON(
        { source: { path: '/allowed/bladder.csv', sheet: undefined }, patientIdColumn: undefined },
        { source: { path: '/allowed/bladder.csv' } },
      ),
    ).toBe(true);
    expect(sameJSON({ patientIdColumn: null }, {})).toBe(false);
  });

  it('marks changed nested scientific values and reordered arrays as dirty', () => {
    const saved = { split: { seeds: [7, 42], rules: { test: [{ field: 'grade', value: '2' }] } } };
    expect(
      sameJSON(saved, {
        split: { seeds: [7, 42], rules: { test: [{ field: 'grade', value: '3' }] } },
      }),
    ).toBe(false);
    expect(
      sameJSON(saved, {
        split: { seeds: [42, 7], rules: { test: [{ field: 'grade', value: '2' }] } },
      }),
    ).toBe(false);
  });
});
