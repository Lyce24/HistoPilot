import { describe, expect, it } from 'vitest';
import { fitRegion, patchRegion } from './slideGeometry';
import type { SlideRegion } from '../api/interpretation';

function includes(outer: SlideRegion, inner: SlideRegion) {
  expect(outer.x).toBeLessThanOrEqual(inner.x);
  expect(outer.y).toBeLessThanOrEqual(inner.y);
  expect(outer.x + outer.width).toBeGreaterThanOrEqual(inner.x + inner.width);
  expect(outer.y + outer.height).toBeGreaterThanOrEqual(inner.y + inner.height);
}
describe('fitting patch coverage to the actual attention canvas', () => {
  const width = 116736, height = 101376;
  const coverage = { x: 35000, y: 25000, width: 20000, height: 30000 };
  it('fills the desktop canvas while retaining every covered patch and context', () => {
    const view = fitRegion(coverage, width, height, 820, 650, .08);
    includes(view, coverage);
    expect(view.width / view.height).toBeCloseTo(820 / 650);
    expect(view.width).toBeLessThan(width / 2);
    expect(view.height).toBeCloseTo(34800);
    expect(view.x + view.width / 2).toBeCloseTo(45000);
    expect(view.y + view.height / 2).toBeCloseTo(40000);
  });
  it('uses the phone canvas aspect rather than the original glass slide aspect', () => {
    const view = fitRegion(coverage, width, height, 360, 550, .08);
    includes(view, coverage);
    expect(view.width / view.height).toBeCloseTo(360 / 550);
    expect(view.width).toBeCloseTo(23200);
  });
  it('keeps edge and full-slide coverage inside the slide without trimming covered tissue', () => {
    const edge = { x: width - 2500, y: height - 1500, width: 2500, height: 1500 };
    const view = fitRegion(edge, width, height, 820, 650, .08);
    includes(view, edge);
    expect(view.x + view.width).toBe(width);
    expect(view.y + view.height).toBe(height);
    const whole = { x: 0, y: 0, width, height };
    expect(fitRegion(whole, width, height, 820, 650, .08)).toEqual(whole);
  });
  it('bounds invalid viewport measurements, oversized padding and tiny images', () => {
    const view = fitRegion(coverage, width, height, NaN, 0, Infinity);
    includes(view, coverage);
    expect(view.width / view.height).toBeCloseTo(width / height);
    expect(fitRegion({ x: 0, y: 0, width: 1, height: 1 }, 1, 1, 360, 550, .5)).toEqual({ x: 0, y: 0, width: 1, height: 1 });
    expect(fitRegion({ x: NaN, y: NaN, width: Infinity, height: NaN }, width, height, 0, 0)).toEqual({ x: 0, y: 0, width, height });
  });
  it('centers a chosen patch with the canvas aspect and preserves the legacy default', () => {
    const patch = { index: 1, x: 400, y: 300, weight: .5, percentile: .9 };
    expect(patchRegion(patch, 50, 100, 1000, 1000)).toEqual({ x: 225, y: 150, width: 400, height: 400 });
    const wide = patchRegion(patch, 50, 100, 1000, 1000, 2);
    expect(wide).toEqual({ x: 25, y: 150, width: 800, height: 400 });
    includes(wide, { x: patch.x, y: patch.y, width: 50, height: 100 });
  });
});
