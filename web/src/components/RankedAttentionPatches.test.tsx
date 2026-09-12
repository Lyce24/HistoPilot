import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider, type QueryObserverOptions } from '@tanstack/react-query';
import type { RankedAttentionPatch } from '../api/interpretation';
import RankedAttentionPatches, { SelectedAttentionPatch } from './RankedAttentionPatches';
import RankedPatchMarkers from './RankedPatchMarkers';
import { markerScale, patchIntersectsRegion, patchRegion } from '../lib/slideGeometry';

const patches: RankedAttentionPatch[] = [
  { index: 3, x: 150.5, y: 90.25, weight: .2, percentile: .99, rank: 1 },
  { index: 8, x: 150.5, y: 90.25, weight: .2, percentile: .99, rank: 2 },
];
const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
describe('ranked patch geometry and accessibility', () => {
  it('keeps exact fractional footprints and distinct equal-weight boxes at duplicate coordinates', () => {
    const html = renderToStaticMarkup(<svg><RankedPatchMarkers patches={patches} patchWidth={31.5} patchHeight={47.25} scale={2} selectedIndex={8} onSelect={() => {}} /></svg>);
    expect(html.match(/role="button"/g)).toHaveLength(2);
    expect(html).toContain('aria-label="Inspect rank 1, patch 3"');
    expect(html).toContain('aria-label="Inspect rank 2, patch 8"');
    expect(html).toContain('x="150.5" y="90.25" width="31.5" height="47.25"');
    expect(html).toContain('vector-effect="non-scaling-stroke"');
    expect(html).toContain('aria-pressed="true"');
    expect(html.match(/tabindex="0"/g)).toHaveLength(2);
  });
  it('centers selected patches with tissue context and clamps at all slide boundaries', () => {
    const centered = patchRegion({ ...patches[0], x: 400, y: 300 }, 50, 100, 1000, 1000);
    expect(centered).toEqual({ x: 225, y: 150, width: 400, height: 400 });
    const edge = patchRegion({ ...patches[0], x: 990, y: 990 }, 31.5, 47.25, 1000, 1000);
    expect(edge.x + edge.width).toBe(1000); expect(edge.y + edge.height).toBe(1000);
    expect(edge.width).toBe(189);
    expect(patchRegion({ ...patches[0], x: 0, y: 0 }, 2000, 2000, 1000, 500)).toEqual({ x: 0, y: 0, width: 1000, height: 500 });
  });
  it('keeps number pins screen-sized through zoom/letterboxing and excludes invisible keyboard targets', () => {
    expect(markerScale({ x: 0, y: 0, width: 2000, height: 1000 }, 1000, 1000)).toBe(2);
    expect(markerScale({ x: 0, y: 0, width: 1000, height: 500 }, 1000, 1000)).toBe(1);
    expect(markerScale({ x: 0, y: 0, width: 1000, height: 500 }, 0, 0)).toBe(1);
    expect(patchIntersectsRegion(patches[0], 31.5, 47.25, { x: 180, y: 90, width: 100, height: 100 })).toBe(true);
    expect(patchIntersectsRegion(patches[0], 31.5, 47.25, { x: 182, y: 90, width: 100, height: 100 })).toBe(false);
  });
});
describe('ranked original slide crops', () => {
  function render(selected: RankedAttentionPatch | null = null) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }); clients.push(client);
    const context = { project: 'p', interpretationId: 'study', slideId: 'S1', member: '1' };
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><SelectedAttentionPatch context={context} patch={selected} rank={selected?.rank} patchWidth={31.5} patchHeight={47.25} onShowLocation={() => {}} /><RankedAttentionPatches context={context} patches={patches} total={200001} selectedPatch={selected} onSelect={() => {}} /></QueryClientProvider>);
    return { html, client };
  }
  it('uses global totals and retains separate equal-weight contact cards with clear original-image labels', () => {
    const { html } = render();
    expect(html).toContain('2 highest-weight patches from all 200,001 slide patches');
    expect(html).toContain('Equal weights are ordered by patch index');
    expect(html).toContain('equal-weight ranks do not imply different importance');
    expect(html).toContain('aria-label="View rank 1 patch"'); expect(html).toContain('aria-label="View rank 2 patch"');
    expect(html).toContain('aria-label="Selected patch crop"');
    expect(html).not.toContain('Loading selected patch');
  });
  it('only enables the selected crop before contact cards enter the viewport and scopes cache to member and patch', () => {
    const { html, client } = render(patches[1]);
    expect(html).toContain('Rank 2 · Patch 8'); expect(html).toContain('Original slide image without heatmap colors');
    const selected = client.getQueryCache().find({ queryKey: ['interpretation-patch-crop', 'p', 'study', 'S1', '1', 8] });
    const unselected = client.getQueryCache().find({ queryKey: ['interpretation-patch-crop', 'p', 'study', 'S1', '1', 3] });
    expect(selected).toBeDefined(); expect(unselected).toBeDefined();
    // Both card observers are visibility-gated; selected detail owns a separate enabled observer for its shared key.
    expect((unselected?.options as QueryObserverOptions).enabled).toBe(false);
    expect(html).toContain('31.5 × 47.25 level-0 pixels');
  });
});
