import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

describe('roadmap phase layout', () => {
  it('keeps Prepare unchanged and places Develop and Evaluate pairs side by side in their own rows', () => {
    const styles = readFileSync(new URL('../roadmap.css', import.meta.url), 'utf8');
    const positions = Object.fromEntries(Array.from(styles.matchAll(/\.node-([\w-]+)\s*\{\s*grid-area:\s*(\d+)\s*\/\s*(\d+);/g), ([, id, row, column]) => [id, [Number(row), Number(column)]]));
    expect(positions).toMatchObject({
      dataset: [1, 2], cohort: [2, 1], features: [2, 3],
      experiments: [3, 2], 'post-development': [3, 3],
      'test-data': [4, 2], evaluation: [4, 3],
      'clinical-utility': [5, 2], interpretation: [5, 3],
    });
  });
});
