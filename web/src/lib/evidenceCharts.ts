export interface CurvePoint { x: number | null; y: number | null }
export type Range = readonly [number, number];
export function finiteNumber(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value); }
export function formatStatistic(value: unknown, digits = 3) { return finiteNumber(value) ? value.toFixed(digits) : 'Unavailable'; }
/** Missing denominators split the line; never connect across an unavailable statistic. */
export function curveSegments(points: CurvePoint[]): { x: number; y: number }[][] {
  const segments: { x: number; y: number }[][] = [];
  let segment: { x: number; y: number }[] = [];
  for (const point of points) {
    if (!finiteNumber(point.x) || !finiteNumber(point.y)) { if (segment.length) segments.push(segment); segment = []; }
    else segment.push({ x: point.x, y: point.y });
  }
  if (segment.length) segments.push(segment);
  return segments;
}
export function chartRange(values: (number | null)[], requested?: Range): Range {
  if (requested && finiteNumber(requested[0]) && finiteNumber(requested[1]) && requested[1] > requested[0]) return requested;
  let minimum = 0, maximum = 0;
  for (const value of values) if (finiteNumber(value)) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
  if (minimum === maximum) return [minimum, minimum + 1];
  const pad = (maximum - minimum) * .05;
  return [minimum < 0 ? minimum - pad : 0, maximum + pad];
}
