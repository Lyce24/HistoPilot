import type { MorphologyRegion, QualityEvidence } from '../api/morphology';

export type QualityOverlayGeometry = Pick<QualityEvidence, 'width' | 'height' | 'patchWidth' | 'patchHeight' | 'patches' | 'tissueContours'>;
export interface QualityOverlayOptions { coverage: boolean; contours: boolean }
const WORK_BUDGET_MS = 4;
export const QUALITY_OVERLAY_MAX_SIDE = 2048;

/** The last painted footprint wins, matching the previous SVG rectangle stack. */
export function hitTestQualityPatch(quality: QualityOverlayGeometry, x: number, y: number): number | undefined {
  if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > quality.width || y > quality.height) return undefined;
  const known = quality.patchWidth != null && quality.patchWidth > 0 && quality.patchHeight != null && quality.patchHeight > 0;
  const radius = Math.max(quality.width, quality.height) / 470;
  for (let index = quality.patches.length - 1; index >= 0; index--) {
    const patch = quality.patches[index];
    if (known ? x >= patch.x && y >= patch.y && x <= Math.min(quality.width, patch.x + quality.patchWidth!) && y <= Math.min(quality.height, patch.y + quality.patchHeight!) : Math.hypot(x - patch.x, y - patch.y) <= radius) return patch.patchIndex;
  }
  return undefined;
}

/** Fit actual screen pixels, including SVG letterboxing, inside a bounded bitmap. */
export function qualityOverlaySize(region: MorphologyRegion, viewportWidth: number, viewportHeight: number, ratio = 1) {
  if (!(region.width > 0 && region.height > 0)) throw new Error('The overlay region must have positive dimensions.');
  const cssScale = Math.min(Math.max(1, viewportWidth) / region.width, Math.max(1, viewportHeight) / region.height);
  const scale = Math.min(cssScale * Math.min(2, Math.max(1, ratio)), QUALITY_OVERLAY_MAX_SIDE / region.width, QUALITY_OVERLAY_MAX_SIDE / region.height);
  return { width: Math.max(1, Math.round(region.width * scale)), height: Math.max(1, Math.round(region.height * scale)), strokePixels: scale / cssScale };
}

function checkAborted(signal: AbortSignal) {
  if (signal.aborted) throw signal.reason ?? new DOMException('QC overlay rendering cancelled.', 'AbortError');
}
function yieldForInput(signal: AbortSignal): Promise<void> {
  checkAborted(signal);
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(signal.reason ?? new DOMException('QC overlay rendering cancelled.', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, 0);
    signal.addEventListener('abort', abort, { once: true });
  });
}

type OverlayContext = Pick<CanvasRenderingContext2D, 'fillStyle' | 'strokeStyle' | 'lineWidth' | 'beginPath' | 'moveTo' | 'lineTo' | 'closePath' | 'fill' | 'stroke' | 'fillRect' | 'strokeRect' | 'arc'>;

/** Rasterize recorded geometry in level-0 space, yielding before dense work blocks input. */
export async function paintQualityOverlay(context: OverlayContext, quality: QualityOverlayGeometry, options: QualityOverlayOptions, region: MorphologyRegion, width: number, height: number, signal: AbortSignal, strokePixels = 1): Promise<void> {
  checkAborted(signal);
  const scaleX = width / region.width, scaleY = height / region.height;
  let deadline = performance.now() + WORK_BUDGET_MS, operations = 0;
  const checkpoint = () => (++operations & 63) === 0 && performance.now() >= deadline;
  context.lineWidth = strokePixels;
  if (options.contours) {
    context.fillStyle = '#27ae6040'; context.strokeStyle = '#1e7543';
    for (const polygon of quality.tissueContours) {
      context.beginPath();
      for (const ring of polygon) {
        let started = false;
        for (const point of ring) {
          if (Number.isFinite(point[0]) && Number.isFinite(point[1])) {
            const x = (point[0] - region.x) * scaleX, y = (point[1] - region.y) * scaleY;
            if (started) context.lineTo(x, y); else { context.moveTo(x, y); started = true; }
          }
          if (checkpoint()) { await yieldForInput(signal); deadline = performance.now() + WORK_BUDGET_MS; }
        }
        if (started) context.closePath();
      }
      context.fill('evenodd'); context.stroke();
      if (performance.now() >= deadline) { await yieldForInput(signal); deadline = performance.now() + WORK_BUDGET_MS; }
    }
  }
  if (options.coverage) {
    const known = quality.patchWidth != null && quality.patchWidth > 0 && quality.patchHeight != null && quality.patchHeight > 0;
    const radius = Math.max(quality.width, quality.height) / 470;
    context.fillStyle = known ? '#285c7624' : '#285c76'; context.strokeStyle = '#285c76';
    for (const patch of quality.patches) {
      const patchWidth = known ? Math.min(quality.patchWidth!, quality.width - patch.x) : radius * 2;
      const patchHeight = known ? Math.min(quality.patchHeight!, quality.height - patch.y) : radius * 2;
      const left = known ? patch.x : patch.x - radius, top = known ? patch.y : patch.y - radius;
      if (Number.isFinite(left) && Number.isFinite(top) && patchWidth > 0 && patchHeight > 0 && left <= region.x + region.width && top <= region.y + region.height && left + patchWidth >= region.x && top + patchHeight >= region.y) {
        if (known) {
          const x = (left - region.x) * scaleX, y = (top - region.y) * scaleY, w = patchWidth * scaleX, h = patchHeight * scaleY;
          context.fillRect(x, y, w, h); context.strokeRect(x, y, w, h);
        } else {
          context.beginPath(); context.arc((patch.x - region.x) * scaleX, (patch.y - region.y) * scaleY, radius * Math.min(scaleX, scaleY), 0, Math.PI * 2); context.fill();
        }
      }
      if (checkpoint()) { await yieldForInput(signal); deadline = performance.now() + WORK_BUDGET_MS; }
    }
  }
  checkAborted(signal);
}

/** Bound asynchronous browser encoding/decoding and release abandoned work immediately. */
export function encodeQualityOverlay(canvas: Pick<HTMLCanvasElement, 'toBlob' | 'width' | 'height'>, signal: AbortSignal, timeoutMs = 10000): Promise<string> {
  return new Promise((resolve, reject) => {
    let finished = false, url: string | undefined, image: HTMLImageElement | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (error?: unknown) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer); signal.removeEventListener('abort', abort);
      image?.removeAttribute('src'); canvas.width = 0; canvas.height = 0;
      if (error !== undefined) { if (url) URL.revokeObjectURL(url); reject(error); }
      else resolve(url!);
    };
    const abort = () => finish(signal.reason ?? new DOMException('QC overlay rendering cancelled.', 'AbortError'));
    if (signal.aborted) { abort(); return; }
    signal.addEventListener('abort', abort, { once: true });
    timer = setTimeout(() => finish(new Error('QC overlay preparation took too long. Reload the slide to try again.')), timeoutMs);
    try {
      canvas.toBlob((blob) => {
        if (finished) return;
        if (!blob) { finish(new Error('QC overlay image could not be prepared.')); return; }
        try {
          url = URL.createObjectURL(blob); image = new Image(); image.src = url;
          void image.decode().then(() => finish(), finish);
        } catch (error) { finish(error); }
      }, 'image/png');
    } catch (error) { finish(error); }
  });
}
