import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AttentionPatch } from '../api/interpretation';
import { attentionColor } from './slideGeometry';
import { orderedAttentionPatches, paintAttentionMap } from './attentionRaster';

const patch = (index: number): AttentionPatch => ({ index, x: index * 4, y: index * 3, weight: (index * 7919) % 97, percentile: index % 10 / 10 });
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('cooperative attention rasterization', () => {
  it('preserves stable weight ordering and caches it without mutating query data', async () => {
    const patches = Array.from({ length: 5000 }, (_, index) => patch(index));
    const original = [...patches]; const signal = new AbortController().signal;
    const ordered = await orderedAttentionPatches(patches, signal);
    expect(ordered).toEqual([...patches].sort((a, b) => a.weight - b.weight));
    expect(patches).toEqual(original); expect(await orderedAttentionPatches(patches, signal)).toBe(ordered);
  });
  it('keeps original color, filtering, overlap order and exact level-0 coordinates', async () => {
    const patches = Array.from({ length: 12 }, (_, index) => patch(index));
    const commands: unknown[] = [];
    const context = { fillStyle: '', fillRect(x: number, y: number, width: number, height: number) { commands.push([this.fillStyle, x, y, width, height]); } };
    await paintAttentionMap(context, { patches, patchWidthLevel0: 8, patchHeightLevel0: 6 }, { x: 10, y: 20, width: 100, height: 50 }, 200, 100, .5, new AbortController().signal);
    expect(commands).toEqual([...patches].sort((a, b) => a.weight - b.weight).filter((value) => value.percentile >= .5).map((value) => [attentionColor(value.percentile), (value.x - 10) * 2, (value.y - 20) * 2, 16, 12]));
  });
  it('yields dense sorting to input and cancels obsolete work before it reaches the canvas', async () => {
    vi.useFakeTimers(); let time = 0; vi.stubGlobal('performance', { now: () => time += 5 });
    const controller = new AbortController(); const patches = Array.from({ length: 5000 }, (_, index) => patch(index));
    const context = { fillStyle: '', fillRect: vi.fn() };
    const pending = paintAttentionMap(context, { patches, patchWidthLevel0: 8, patchHeightLevel0: 6 }, { x: 0, y: 0, width: 100, height: 50 }, 200, 100, 0, controller.signal);
    const stopped = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(vi.getTimerCount()).toBe(1); controller.abort(); await stopped;
    expect(context.fillRect).not.toHaveBeenCalled(); expect(vi.getTimerCount()).toBe(0);
    vi.unstubAllGlobals(); vi.useRealTimers();
    expect(await orderedAttentionPatches(patches, new AbortController().signal)).toHaveLength(5000);
  });
  it('interrupts a dense draw after a short batch even when its sort is cached', async () => {
    const patches = Array.from({ length: 1000 }, (_, index) => patch(index));
    await orderedAttentionPatches(patches, new AbortController().signal);
    vi.useFakeTimers(); let time = 0; vi.stubGlobal('performance', { now: () => time += 5 });
    const controller = new AbortController(); const context = { fillStyle: '', fillRect: vi.fn() };
    const pending = paintAttentionMap(context, { patches, patchWidthLevel0: 8, patchHeightLevel0: 6 }, { x: 0, y: 0, width: 100, height: 50 }, 200, 100, 0, controller.signal);
    const stopped = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    await Promise.resolve(); expect(vi.getTimerCount()).toBe(1); controller.abort(); await stopped;
    expect(context.fillRect.mock.calls.length).toBeGreaterThan(0); expect(context.fillRect.mock.calls.length).toBeLessThanOrEqual(129);
  });
});
