import type { AttentionMap, AttentionPatch, SlideRegion } from '../api/interpretation';
import { attentionColor } from './slideGeometry';

const WORK_BUDGET_MS = 4;
const SORT_RUN = 512;
const orderedMaps = new WeakMap<AttentionPatch[], AttentionPatch[]>();

function checkAborted(signal: AbortSignal) {
  if (signal.aborted) throw signal.reason ?? new DOMException('Attention rendering cancelled.', 'AbortError');
}
function yieldForInput(signal: AbortSignal): Promise<void> {
  checkAborted(signal);
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(signal.reason ?? new DOMException('Attention rendering cancelled.', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, 0);
    signal.addEventListener('abort', abort, { once: true });
  });
}

/** Stable ascending weights preserve the highest-weight color where patches overlap. */
export async function orderedAttentionPatches(patches: AttentionPatch[], signal: AbortSignal): Promise<AttentionPatch[]> {
  checkAborted(signal);
  const cached = orderedMaps.get(patches);
  if (cached) return cached;
  let deadline = performance.now() + WORK_BUDGET_MS;
  let source: AttentionPatch[] = [];
  // Native sorting stays bounded to small runs; merging also yields during dense maps.
  for (let start = 0; start < patches.length; start += SORT_RUN) {
    const run = patches.slice(start, start + SORT_RUN).sort((a, b) => a.weight - b.weight);
    for (const patch of run) source.push(patch);
    if (performance.now() >= deadline) { await yieldForInput(signal); deadline = performance.now() + WORK_BUDGET_MS; }
  }
  let destination = new Array<AttentionPatch>(source.length);
  for (let width = SORT_RUN; width < source.length; width *= 2) {
    for (let start = 0; start < source.length; start += width * 2) {
      const middle = Math.min(source.length, start + width), end = Math.min(source.length, start + width * 2);
      let left = start, right = middle;
      for (let index = start; index < end; index++) {
        destination[index] = right >= end || (left < middle && source[left].weight <= source[right].weight) ? source[left++] : source[right++];
        if ((index & 255) === 0 && performance.now() >= deadline) { await yieldForInput(signal); deadline = performance.now() + WORK_BUDGET_MS; }
      }
    }
    [source, destination] = [destination, source];
  }
  checkAborted(signal);
  orderedMaps.set(patches, source);
  return source;
}

/** Yield dense raster work between short batches so wheel and pointer frames can paint. */
export async function paintAttentionMap(
  context: Pick<CanvasRenderingContext2D, 'fillStyle' | 'fillRect'>,
  map: Pick<AttentionMap, 'patches' | 'patchWidthLevel0' | 'patchHeightLevel0'>,
  region: SlideRegion, width: number, height: number, minimum: number, signal: AbortSignal,
): Promise<void> {
  const ordered = await orderedAttentionPatches(map.patches, signal);
  checkAborted(signal);
  const scaleX = width / region.width, scaleY = height / region.height;
  let deadline = performance.now() + WORK_BUDGET_MS;
  for (let index = 0; index < ordered.length; index++) {
    const patch = ordered[index];
    if (patch.percentile >= minimum) {
      context.fillStyle = attentionColor(patch.percentile);
      context.fillRect((patch.x - region.x) * scaleX, (patch.y - region.y) * scaleY, map.patchWidthLevel0 * scaleX, map.patchHeightLevel0 * scaleY);
    }
    if ((index & 127) === 0 && performance.now() >= deadline) { await yieldForInput(signal); deadline = performance.now() + WORK_BUDGET_MS; }
  }
  checkAborted(signal);
}
