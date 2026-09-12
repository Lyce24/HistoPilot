import { describe, expect, it } from 'vitest';
import { chartRange, curveSegments, formatStatistic } from './evidenceCharts';
import { attentionAt, attentionColor, boundedRegion, integerRegion, zoomRegion } from './slideGeometry';
import { evidenceLink } from '../components/EvidenceChain';

describe('clinical curve representation', () => {
  it('preserves unavailable denominators as gaps and excludes non-finite coordinates', () => {
    expect(curveSegments([{ x: 0, y: 1 }, { x: .2, y: null }, { x: .4, y: .8 }, { x: Infinity, y: .7 }, { x: .8, y: NaN }, { x: 1, y: 0 }])).toEqual([[{ x: 0, y: 1 }], [{ x: .4, y: .8 }], [{ x: 1, y: 0 }]]);
    expect(formatStatistic(Infinity)).toBe('Unavailable');
    expect(formatStatistic(null)).toBe('Unavailable');
    expect(formatStatistic(0)).toBe('0.000');
  });
  it('keeps negative net benefit visible and guards empty or degenerate chart ranges', () => {
    const range = chartRange([-.4, .2, null, Infinity]);
    expect(range[0]).toBeLessThan(-.4); expect(range[1]).toBeGreaterThan(.2);
    expect(chartRange([], [0, 0])).toEqual([0, 1]);
    expect(chartRange([99], [0, 1])).toEqual([0, 1]);
  });
  it('retains exact experiment, predictor, evaluation and clinical identities in onward links', () => {
    const link = evidenceLink('interpretation', { experimentId: 'exp 1', predictorId: 'refit/a', evaluationId: 'eval&1', clinicalAnalysisId: 'clinical#2' });
    const params = new URLSearchParams(link.split('?')[1]);
    expect(params.get('experiment')).toBe('exp 1'); expect(params.get('predictor')).toBe('refit/a'); expect(params.get('evaluation')).toBe('eval&1'); expect(params.get('clinical')).toBe('clinical#2');
  });
});
describe('whole-slide geometry and attention inspection', () => {
  it('bounds panning and rounds viewports outward without reading beyond the slide', () => {
    expect(boundedRegion({ x: -30, y: 900, width: 200, height: 200 }, 1000, 1000)).toEqual({ x: 0, y: 800, width: 200, height: 200 });
    expect(integerRegion({ x: 799.7, y: 800.2, width: 200.3, height: 199.8 }, 1000, 1000)).toEqual({ x: 799, y: 800, width: 201, height: 200 });
    expect(boundedRegion({ x: NaN, y: Infinity, width: -1, height: NaN }, 1000, 500)).toEqual({ x: 0, y: 0, width: 1, height: 500 });
  });
  it('zooms around the center with a useful minimum viewport and a full-slide maximum', () => {
    expect(zoomRegion({ x: 0, y: 0, width: 1000, height: 500 }, 2, 1000, 500)).toEqual({ x: 250, y: 125, width: 500, height: 250 });
    expect(zoomRegion({ x: 1, y: 1, width: 10, height: 10 }, .001, 1000, 500)).toEqual({ x: 0, y: 0, width: 1000, height: 500 });
  });
  it('uses level-0 patch footprints and exclusive far boundaries for precise hit testing', () => {
    const patches = [{ index: 0, x: 100, y: 200, weight: .1, percentile: .2 }, { index: 1, x: 200, y: 200, weight: .3, percentile: .9 }];
    expect(attentionAt(patches, 199, 200, 100, 200)?.index).toBe(0);
    expect(attentionAt(patches, 200, 200, 100, 200)?.index).toBe(1);
    expect(attentionAt(patches, 200, 400, 100, 200)).toBeNull();
    expect(attentionColor(5)).toBe(attentionColor(1));
    expect(attentionColor(NaN)).toBe(attentionColor(0));
  });
  it('inspects the strongest visible overlapping patch independently of feature row order', () => {
    const strongest = { index: 1, x: 100, y: 100, weight: .7, percentile: .9 };
    const weaker = { index: 2, x: 150, y: 100, weight: .1, percentile: .2 };
    expect(attentionAt([strongest, weaker], 175, 125, 100, 100)).toBe(strongest);
    expect(attentionAt([weaker, strongest], 175, 125, 100, 100)).toBe(strongest);
    expect(attentionAt([strongest, weaker], 225, 125, 100, 100, .5)).toBeNull();
  });
});
