import { afterEach, describe, expect, it, vi } from 'vitest';
import { encodeQualityOverlay, hitTestQualityPatch, paintQualityOverlay, qualityOverlaySize, type QualityOverlayGeometry } from './qualityOverlay';

const geometry = (values: Partial<QualityOverlayGeometry> = {}): QualityOverlayGeometry => ({
  width: 100, height: 80, patchWidth: 20, patchHeight: 15,
  patches: [{ patchIndex: 3, x: 10, y: 20 }, { patchIndex: 9, x: 20, y: 25 }, { patchIndex: 11, x: 95, y: 75 }], tissueContours: [], ...values,
});
function context() {
  return { fillStyle: '', strokeStyle: '', lineWidth: 1, beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), closePath: vi.fn(), fill: vi.fn(), stroke: vi.fn(), fillRect: vi.fn(), strokeRect: vi.fn(), arc: vi.fn() };
}
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('QC raster geometry and patch interaction', () => {
  it('selects the topmost overlapping patch and clips edge footprints to the slide', () => {
    const quality = geometry();
    expect(hitTestQualityPatch(quality, 25, 30)).toBe(9);
    expect(hitTestQualityPatch(quality, 15, 22)).toBe(3);
    expect(hitTestQualityPatch(quality, 99, 79)).toBe(11);
    expect(hitTestQualityPatch(quality, 105, 79)).toBeUndefined();
    expect(hitTestQualityPatch(quality, 40, 50)).toBeUndefined();
    expect(hitTestQualityPatch(quality, NaN, 30)).toBeUndefined();
  });
  it('hit-tests unknown-footprint origin circles without inventing rectangles', () => {
    const quality = geometry({ patchWidth: null, patchHeight: null });
    expect(hitTestQualityPatch(quality, 10.1, 20)).toBe(3);
    expect(hitTestQualityPatch(quality, 11, 20)).toBeUndefined();
  });
  it('bounds raster memory for huge slides, high DPR and extreme aspect ratios', () => {
    expect(qualityOverlaySize({ x: 0, y: 0, width: 60000, height: 40000 }, 1200, 1000, 2)).toEqual({ width: 2048, height: 1365, strokePixels: 2048 / 1200 });
    expect(qualityOverlaySize({ x: 0, y: 0, width: 1e8, height: 1 }, 4000, 4000, 8)).toEqual({ width: 2048, height: 1, strokePixels: 2048 / 4000 });
    expect(() => qualityOverlaySize({ x: 0, y: 0, width: 0, height: 10 }, 100, 100)).toThrow(/positive/);
  });
  it('keeps level-0 placement, recorded paint order, clipped edges and viewport culling', async () => {
    const canvas = context();
    await paintQualityOverlay(canvas, geometry(), { coverage: true, contours: false }, { x: 20, y: 25, width: 80, height: 55 }, 160, 110, new AbortController().signal, 2);
    expect(canvas.fillRect.mock.calls).toEqual([[-20, -10, 40, 30], [0, 0, 40, 30], [150, 100, 10, 10]]);
    expect(canvas.strokeRect.mock.calls).toEqual(canvas.fillRect.mock.calls);
    expect(canvas.lineWidth).toBe(2); expect(canvas.fillStyle).toBe('#285c7624');
    const clipped = context();
    await paintQualityOverlay(clipped, geometry(), { coverage: true, contours: false }, { x: 70, y: 60, width: 30, height: 20 }, 90, 60, new AbortController().signal);
    expect(clipped.fillRect.mock.calls).toEqual([[75, 45, 15, 15]]);
  });
  it('keeps contour holes in a single even-odd path and paints contours before coverage', async () => {
    const canvas = context();
    const quality = geometry({ tissueContours: [[[[0, 0], [100, 0], [100, 80], [0, 80]], [[30, 30], [50, 30], [50, 50], [30, 50]]]] });
    await paintQualityOverlay(canvas, quality, { coverage: true, contours: true }, { x: 10, y: 20, width: 100, height: 80 }, 200, 160, new AbortController().signal);
    expect(canvas.beginPath).toHaveBeenCalledTimes(1); expect(canvas.closePath).toHaveBeenCalledTimes(2);
    expect(canvas.moveTo.mock.calls).toEqual([[-20, -40], [40, 20]]);
    expect(canvas.fill).toHaveBeenCalledExactlyOnceWith('evenodd');
    expect(canvas.fill.mock.invocationCallOrder[0]).toBeLessThan(canvas.fillRect.mock.invocationCallOrder[0]);
  });
  it('does no geometry work when both overlay types are hidden', async () => {
    const canvas = context();
    await paintQualityOverlay(canvas, geometry(), { coverage: false, contours: false }, { x: 0, y: 0, width: 100, height: 80 }, 100, 80, new AbortController().signal);
    expect(canvas.beginPath).not.toHaveBeenCalled(); expect(canvas.fillRect).not.toHaveBeenCalled();
  });
  it('yields within a dense contour and cancels obsolete paths before filling', async () => {
    vi.useFakeTimers(); let time = 0; vi.stubGlobal('performance', { now: () => time += 5 });
    const quality = geometry({ tissueContours: [[Array.from({ length: 50000 }, (_, index) => [index % 100, index % 80])]] });
    const controller = new AbortController(), canvas = context();
    const pending = paintQualityOverlay(canvas, quality, { coverage: true, contours: true }, { x: 0, y: 0, width: 100, height: 80 }, 200, 160, controller.signal);
    const stopped = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(vi.getTimerCount()).toBe(1); expect(canvas.lineTo.mock.calls.length).toBeLessThanOrEqual(63);
    controller.abort(); await stopped;
    expect(canvas.fill).not.toHaveBeenCalled(); expect(canvas.fillRect).not.toHaveBeenCalled(); expect(vi.getTimerCount()).toBe(0);
  });
  it('yields dense coverage painting and cancels without processing remaining patches', async () => {
    vi.useFakeTimers(); let time = 0; vi.stubGlobal('performance', { now: () => time += 5 });
    const quality = geometry({ patches: Array.from({ length: 4096 }, (_, patchIndex) => ({ patchIndex, x: patchIndex % 80, y: patchIndex % 60 })) });
    const controller = new AbortController(), canvas = context();
    const pending = paintQualityOverlay(canvas, quality, { coverage: true, contours: false }, { x: 0, y: 0, width: 100, height: 80 }, 200, 160, controller.signal);
    const stopped = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(vi.getTimerCount()).toBe(1); expect(canvas.fillRect.mock.calls.length).toBe(64);
    controller.abort(); await stopped; expect(canvas.fillRect.mock.calls.length).toBe(64);
  });
});

describe('QC overlay browser resource lifetime', () => {
  it('aborts a stalled encode immediately and ignores its late callback', async () => {
    const createObjectURL = vi.fn(); vi.stubGlobal('URL', { createObjectURL, revokeObjectURL: vi.fn() });
    let complete: BlobCallback | undefined;
    const canvas = { width: 2048, height: 1365, toBlob: (callback: BlobCallback) => { complete = callback; } };
    const controller = new AbortController();
    const pending = encodeQualityOverlay(canvas, controller.signal), stopped = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort(); await stopped;
    expect([canvas.width, canvas.height]).toEqual([0, 0]);
    complete?.(new Blob(['late'])); expect(createObjectURL).not.toHaveBeenCalled();
  });
  it('clears a stalled image decode and its URL immediately on cancellation', async () => {
    const revokeObjectURL = vi.fn(), removeAttribute = vi.fn();
    vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:overlay'), revokeObjectURL });
    let complete: (() => void) | undefined;
    vi.stubGlobal('Image', class { src = ''; removeAttribute = removeAttribute; decode() { return new Promise<void>((resolve) => { complete = resolve; }); } });
    const canvas = { width: 2048, height: 1365, toBlob: (callback: BlobCallback) => callback(new Blob(['pixels'])) };
    const controller = new AbortController();
    const pending = encodeQualityOverlay(canvas, controller.signal), stopped = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort(); await stopped;
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:overlay');
    expect(removeAttribute).toHaveBeenCalledExactlyOnceWith('src'); expect([canvas.width, canvas.height]).toEqual([0, 0]);
    complete?.(); await Promise.resolve(); expect(revokeObjectURL).toHaveBeenCalledTimes(1);
  });
  it('reports a bounded deadline and releases resources if encoding never returns', async () => {
    vi.useFakeTimers();
    const canvas = { width: 2048, height: 1365, toBlob: vi.fn() };
    const pending = encodeQualityOverlay(canvas, new AbortController().signal, 25), stopped = expect(pending).rejects.toThrow(/took too long/);
    await vi.advanceTimersByTimeAsync(25); await stopped;
    expect([canvas.width, canvas.height]).toEqual([0, 0]); expect(vi.getTimerCount()).toBe(0);
  });
  it('keeps successful image URLs for the displayed layer while releasing the staging canvas and decoder', async () => {
    vi.useFakeTimers();
    const revokeObjectURL = vi.fn(), removeAttribute = vi.fn();
    vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:overlay'), revokeObjectURL });
    vi.stubGlobal('Image', class { src = ''; removeAttribute = removeAttribute; decode() { return Promise.resolve(); } });
    const canvas = { width: 2048, height: 1365, toBlob: (callback: BlobCallback) => callback(new Blob(['pixels'])) };
    await expect(encodeQualityOverlay(canvas, new AbortController().signal)).resolves.toBe('blob:overlay');
    expect(revokeObjectURL).not.toHaveBeenCalled(); expect(removeAttribute).toHaveBeenCalledExactlyOnceWith('src');
    expect([canvas.width, canvas.height]).toEqual([0, 0]); expect(vi.getTimerCount()).toBe(0);
  });
});
