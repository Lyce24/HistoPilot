import type { AttentionPatch, SlideRegion } from '../api/interpretation';
import { finiteNumber } from './evidenceCharts';
export function boundedRegion(region: SlideRegion, width: number, height: number): SlideRegion {
  const w = Math.max(1, Math.min(width, finiteNumber(region.width) ? region.width : width));
  const h = Math.max(1, Math.min(height, finiteNumber(region.height) ? region.height : height));
  return { x: Math.max(0, Math.min(width - w, finiteNumber(region.x) ? region.x : 0)), y: Math.max(0, Math.min(height - h, finiteNumber(region.y) ? region.y : 0)), width: w, height: h };
}
/** Fit the entire supplied region with context, using the actual canvas aspect. */
export function fitRegion(region: SlideRegion, slideWidth: number, slideHeight: number, viewportWidth: number, viewportHeight: number, padding = 0): SlideRegion {
  const source = boundedRegion(region, slideWidth, slideHeight);
  const aspect = finiteNumber(viewportWidth) && finiteNumber(viewportHeight) && viewportWidth > 0 && viewportHeight > 0 ? viewportWidth / viewportHeight : slideWidth / slideHeight;
  const margin = finiteNumber(padding) ? Math.max(0, Math.min(.5, padding)) : 0;
  const paddedWidth = source.width * (1 + margin * 2), paddedHeight = source.height * (1 + margin * 2);
  const width = Math.min(slideWidth, Math.max(paddedWidth, paddedHeight * aspect));
  const height = Math.min(slideHeight, Math.max(paddedHeight, paddedWidth / aspect));
  return boundedRegion({ x: source.x + source.width / 2 - width / 2, y: source.y + source.height / 2 - height / 2, width, height }, slideWidth, slideHeight);
}
export function zoomRegion(region: SlideRegion, factor: number, width: number, height: number): SlideRegion {
  if (!finiteNumber(factor) || factor <= 0) return boundedRegion(region, width, height);
  const nextWidth = Math.min(width, Math.max(Math.min(32, width), region.width / factor));
  const nextHeight = Math.min(height, Math.max(Math.min(32, height), region.height / factor));
  return boundedRegion({ x: region.x + (region.width - nextWidth) / 2, y: region.y + (region.height - nextHeight) / 2, width: nextWidth, height: nextHeight }, width, height);
}
/** A viewport is rounded outward so its edge patches and source pixels are retained. */
export function integerRegion(region: SlideRegion, width: number, height: number): SlideRegion {
  const value = boundedRegion(region, width, height), x = Math.floor(value.x), y = Math.floor(value.y);
  return { x, y, width: Math.min(width - x, Math.ceil(value.x + value.width) - x), height: Math.min(height - y, Math.ceil(value.y + value.height) - y) };
}
export function attentionColor(percentile: number) { return `hsl(${230 * (1 - Math.max(0, Math.min(1, finiteNumber(percentile) ? percentile : 0)))} 85% 50%)`; }
export function attentionAt(patches: AttentionPatch[], x: number, y: number, patchWidth: number, patchHeight: number, minimumPercentile = 0) {
  let hit: AttentionPatch | null = null;
  for (const patch of patches) if (patch.percentile >= minimumPercentile && x >= patch.x && y >= patch.y && x < patch.x + patchWidth && y < patch.y + patchHeight && (!hit || patch.weight > hit.weight)) hit = patch;
  return hit;
}
/** Center the actual patch footprint with surrounding tissue, retaining the slide viewport aspect. */
export function patchRegion(patch: AttentionPatch, patchWidth: number, patchHeight: number, slideWidth: number, slideHeight: number, viewportAspect = slideWidth / slideHeight): SlideRegion {
  const aspect = finiteNumber(viewportAspect) && viewportAspect > 0 ? viewportAspect : slideWidth / slideHeight;
  const width = Math.min(slideWidth, Math.max(patchWidth * 4, patchHeight * 4 * aspect, Math.min(32, slideWidth)));
  const height = Math.min(slideHeight, width / aspect);
  return boundedRegion({ x: patch.x + patchWidth / 2 - width / 2, y: patch.y + patchHeight / 2 - height / 2, width, height }, slideWidth, slideHeight);
}
/** Return slide-space units for a fixed-size marker after the SVG viewBox transform. */
export function markerScale(view: SlideRegion, viewportWidth: number, viewportHeight: number) {
  if (!finiteNumber(viewportWidth) || !finiteNumber(viewportHeight) || viewportWidth <= 0 || viewportHeight <= 0) return 1;
  return Math.max(view.width / viewportWidth, view.height / viewportHeight);
}
export function patchIntersectsRegion(patch: AttentionPatch, width: number, height: number, region: SlideRegion) {
  return patch.x < region.x + region.width && patch.x + width > region.x && patch.y < region.y + region.height && patch.y + height > region.y;
}
