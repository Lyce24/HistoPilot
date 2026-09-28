import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SlideRegion } from '../api/interpretation';
import { slidePointAt, zoomRegionAt } from './slideGeometry';
import { bindSlideZoom, createFramePublisher, normalizedZoomDelta, type SlideZoomOptions } from './useSlideZoom';

const full = { x: 0, y: 0, width: 1000, height: 500 };
const box = { left: 10, top: 20, width: 1000, height: 1000 };

describe('pointer-anchored slide zoom', () => {
  it('accounts for centered SVG letterboxing and ignores points outside the image', () => {
    expect(slidePointAt(full, box, 260, 395)).toEqual({ x: 250, y: 125 });
    expect(slidePointAt(full, box, 260, 100)).toBeNull();
    expect(slidePointAt(full, { ...box, width: 0 }, 260, 395)).toBeNull();
    expect(slidePointAt(full, box, NaN, 395)).toBeNull();
    const portrait = { x: 100, y: 200, width: 200, height: 400 };
    expect(slidePointAt(portrait, { left: 0, top: 0, width: 800, height: 400 }, 350, 100)).toEqual({ x: 150, y: 300 });
    expect(slidePointAt(portrait, { left: 0, top: 0, width: 800, height: 400 }, 100, 100)).toBeNull();
  });
  it('keeps the tissue under the pointer fixed through zoom and its inverse', () => {
    const anchor = { x: 250, y: 125 };
    const next = zoomRegionAt(full, 2, 1000, 500, anchor);
    expect(next).toEqual({ x: 125, y: 62.5, width: 500, height: 250 });
    expect(slidePointAt(next, box, 260, 395)).toEqual(anchor);
    expect(zoomRegionAt(next, .5, 1000, 500, anchor)).toEqual(full);
  });
  it('preserves aspect at extreme zoom limits and remains inside slide edges', () => {
    const close = zoomRegionAt(full, 1e30, 1000, 500, { x: 990, y: 495 });
    expect(close.width).toBe(64); expect(close.height).toBe(32);
    const tinySelection = { x: 100, y: 100, width: 8, height: 16 };
    expect(zoomRegionAt(tinySelection, 2, 1000, 500, { x: 104, y: 108 })).toEqual(tinySelection);
    const far = zoomRegionAt({ x: 100, y: 100, width: 100, height: 200 }, 1e-30, 1000, 500, { x: 190, y: 290 });
    expect(far.width / far.height).toBe(.5); expect(far.width).toBe(250); expect(far.height).toBe(500);
    expect(far.x).toBeGreaterThanOrEqual(0); expect(far.y).toBe(0); expect(far.x + far.width).toBeLessThanOrEqual(1000);
    expect(zoomRegionAt(full, NaN, 1000, 500, { x: 0, y: 0 })).toEqual(full);
    expect(zoomRegionAt({ x: 0, y: 0, width: 1, height: 1 }, 2, 1, 1, { x: .5, y: .5 })).toEqual({ x: 0, y: 0, width: 1, height: 1 });
  });
  it('normalizes fine trackpad, wheel-line and page deltas with bounded jumps', () => {
    expect(normalizedZoomDelta(.25, 0, 500)).toBe(.25); expect(normalizedZoomDelta(3, 1, 500)).toBe(48);
    expect(normalizedZoomDelta(-1, 2, 500)).toBe(-240); expect(normalizedZoomDelta(1e10, 0, 500)).toBe(240);
    expect(normalizedZoomDelta(NaN, 0, 500)).toBe(0); expect(normalizedZoomDelta(Infinity, 0, 500)).toBe(0);
  });
});

let frames: Map<number, FrameRequestCallback>;
let frameId: number;
let frameTime: number;
beforeEach(() => {
  frames = new Map(); frameId = 0; frameTime = 0;
  vi.stubGlobal('performance', { now: () => frameTime });
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { frames.set(++frameId, callback); return frameId; });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });
});
afterEach(() => vi.unstubAllGlobals());
function flushFrames(elapsed = 1000 / 60) { frameTime += elapsed; const pending = [...frames.values()]; frames.clear(); pending.forEach((callback) => callback(frameTime)); }
function finishFrames() { for (let index = 0; frames.size && index < 20; index++) flushFrames(); expect(frames.size).toBe(0); }
function setup(initial: SlideRegion = full) {
  const target = new EventTarget();
  const element = Object.assign(new EventTarget(), { parentElement: target, getBoundingClientRect: () => box }) as unknown as SVGSVGElement;
  const onChange = vi.fn<(view: SlideRegion) => void>();
  let options: SlideZoomOptions = { view: initial, width: 1000, height: 500, onChange };
  const binding = bindSlideZoom(element, () => options);
  const event = (name: string, data: object) => { const value = Object.assign(new Event(name, { cancelable: true }), data); target.dispatchEvent(value); return value; };
  const wheel = (data: object = {}) => event('wheel', { ctrlKey: true, deltaY: -50, deltaMode: 0, clientX: 260, clientY: 395, ...data });
  return { onChange, event, wheel, binding, update: (next: Partial<SlideZoomOptions>) => { options = { ...options, ...next }; binding.sync(); } };
}

describe('slide-scoped native gesture handling', () => {
  it.each([false, true])('captures wheel inside the slide window with ctrlKey=%s', (ctrlKey) => {
    const test = setup();
    expect(test.wheel({ ctrlKey }).defaultPrevented).toBe(true); finishFrames();
    expect(test.onChange).toHaveBeenCalledTimes(ctrlKey ? 1 : 8);
    const final = test.onChange.mock.calls.at(-1)![0];
    expect(final.width).toBeCloseTo(1000 / Math.exp(.2));
    expect(slidePointAt(final, box, 260, 395)).toEqual({ x: 250, y: 125 });
    test.binding.dispose();
  });
  it('cancels competing pointer work before scheduling a wheel frame', () => {
    const test = setup(); const pan = vi.fn<(view: SlideRegion) => void>(); const pending = createFramePublisher(pan);
    pending.schedule({ ...full, x: 10 }); test.update({ onInteraction: pending.cancel }); test.wheel();
    expect(frames.size).toBe(1); flushFrames(); expect(pan).not.toHaveBeenCalled();
    expect(test.onChange).toHaveBeenCalledTimes(1); test.binding.dispose();
  });
  it('eases coarse mouse notches across frames without moving the tissue under the cursor', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames();
    const first = test.onChange.mock.calls[0][0];
    expect(first.width).toBeLessThan(full.width); expect(first.width).toBeGreaterThan(full.width / Math.exp(.4));
    test.update({ view: first }); test.wheel({ ctrlKey: false, deltaY: -100 }); finishFrames();
    expect(test.onChange.mock.calls.at(-1)![0].width).toBeCloseTo(full.width / Math.exp(.8));
    for (const [view] of test.onChange.mock.calls) {
      const point = slidePointAt(view, box, 260, 395)!;
      expect(point.x).toBeCloseTo(250); expect(point.y).toBeCloseTo(125);
    }
    expect(test.onChange.mock.calls.length).toBeLessThanOrEqual(9); test.binding.dispose();
  });
  it('keeps fine trackpad deltas direct and respects reduced motion for coarse notches', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -2 }); flushFrames();
    expect(frames.size).toBe(0); expect(test.onChange.mock.calls[0][0].width).toBeCloseTo(full.width / Math.exp(.008));
    vi.stubGlobal('matchMedia', () => ({ matches: true }));
    test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames();
    expect(frames.size).toBe(0); expect(test.onChange.mock.calls[1][0].width).toBeCloseTo(full.width / Math.exp(.408));
    test.binding.dispose();
  });
  it('latches easing through wheel deceleration instead of jumping to the target', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames();
    const first = test.onChange.mock.calls.at(-1)![0]; test.update({ view: first });
    test.wheel({ ctrlKey: false, deltaY: -2 }); flushFrames();
    const second = test.onChange.mock.calls.at(-1)![0];
    expect(second.width).toBeLessThan(first.width); expect(second.width).toBeGreaterThan(full.width / Math.exp(.408));
    expect(frames.size).toBe(1); finishFrames();
    expect(test.onChange.mock.calls.at(-1)![0].width).toBeCloseTo(full.width / Math.exp(.408)); test.binding.dispose();
  });
  it('keeps a precise gesture direct while it accelerates and reassesses after idle', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -2 }); flushFrames();
    test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames(); expect(frames.size).toBe(0);
    expect(test.onChange.mock.calls.at(-1)![0].width).toBeCloseTo(full.width / Math.exp(.408));
    frameTime += 181; test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames();
    expect(frames.size).toBe(1); finishFrames(); test.binding.dispose();
  });
  it('treats fractional trackpad input as precise and pinch as a separate gesture', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -70.5 }); flushFrames();
    expect(frames.size).toBe(0); expect(test.onChange.mock.calls.at(-1)![0].width).toBeCloseTo(full.width / Math.exp(.282));
    frameTime += 181; test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames(); expect(frames.size).toBe(1);
    test.wheel({ ctrlKey: true, deltaY: -2 }); flushFrames(); expect(frames.size).toBe(0);
    expect(test.onChange.mock.calls.at(-1)![0].width).toBeCloseTo(full.width / Math.exp(.69)); test.binding.dispose();
  });
  it('stops wheel motion before pointer-down records the pan view and does not restore an older scale', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -120 }); flushFrames();
    const visible = test.onChange.mock.calls.at(-1)![0]; test.update({ view: visible });
    const pressed = test.event('pointerdown', { button: 0, isPrimary: true, pointerType: 'mouse' });
    expect(pressed.defaultPrevented).toBe(false); expect(frames.size).toBe(0);
    const calls = test.onChange.mock.calls.length; flushFrames(); expect(test.onChange).toHaveBeenCalledTimes(calls);
    const panned = { ...visible, x: visible.x + 10 }; test.update({ view: panned }); flushFrames();
    expect(test.onChange.mock.calls.at(-1)![0].width).toBe(visible.width);
    test.wheel({ ctrlKey: false, deltaY: -2 }); flushFrames();
    expect(frames.size).toBe(0); expect(test.onChange.mock.calls.at(-1)![0].width).toBeCloseTo(panned.width / Math.exp(.008));
    test.binding.dispose();
  });
  it('overwrites a published but uncommitted frame when pointer-down takes over', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -120 }); flushFrames();
    // React still displays the original options.view until its pending state commits.
    expect(test.onChange.mock.calls.at(-1)![0].width).toBeLessThan(full.width);
    test.event('pointerdown', { button: 0, isPrimary: true });
    expect(test.onChange.mock.calls.at(-1)![0]).toBe(full); expect(frames.size).toBe(0);
    finishFrames(); test.binding.dispose();
  });
  it('does not interrupt for secondary pointers and removes the capture listener on disposal', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames();
    test.event('pointerdown', { button: 2, isPrimary: true }); expect(frames.size).toBe(1);
    test.event('pointerdown', { button: 0, isPrimary: false }); expect(frames.size).toBe(1);
    test.binding.dispose(); const calls = test.onChange.mock.calls.length;
    test.event('pointerdown', { button: 0, isPrimary: true }); expect(test.onChange).toHaveBeenCalledTimes(calls);
  });
  it('cancels an active zoom animation when fit or a selected region replaces the camera', () => {
    const test = setup(); test.wheel({ ctrlKey: false, deltaY: -100 }); flushFrames();
    expect(frames.size).toBe(1); test.update({ view: { ...full } }); finishFrames();
    expect(test.onChange).toHaveBeenCalledTimes(1); test.binding.dispose();
  });
  it('accumulates every gesture in a burst but renders only once per animation frame', () => {
    const test = setup(); test.wheel(); test.wheel(); test.wheel();
    expect(frames.size).toBe(1); expect(test.onChange).not.toHaveBeenCalled(); flushFrames();
    expect(test.onChange).toHaveBeenCalledTimes(1); expect(test.onChange.mock.calls[0][0].width).toBeCloseTo(1000 / Math.exp(.6));
    // The next event can arrive before React commits the preceding callback.
    test.wheel(); test.update({ view: test.onChange.mock.calls[0][0] }); flushFrames();
    expect(test.onChange.mock.calls[1][0].width).toBeCloseTo(1000 / Math.exp(.8)); test.binding.dispose();
  });
  it('discards queued frames after reset, including reset to identical coordinates', () => {
    const test = setup(); test.wheel(); test.update({ view: { ...full } });
    expect(frames.size).toBe(0); flushFrames(); expect(test.onChange).not.toHaveBeenCalled();
    test.update({ view: { x: 200, y: 100, width: 400, height: 200 } }); test.wheel();
    test.update({ view: { x: 300, y: 150, width: 200, height: 100 } }); flushFrames();
    expect(test.onChange).not.toHaveBeenCalled(); test.binding.dispose();
  });
  it('cancels pending work when disabled, resized or unmounted and removes listeners', () => {
    const test = setup(); test.wheel(); test.update({ disabled: true });
    expect(test.wheel().defaultPrevented).toBe(true); flushFrames(); expect(test.onChange).not.toHaveBeenCalled();
    test.update({ disabled: false }); test.wheel(); test.update({ width: 2000 }); flushFrames();
    expect(test.onChange).not.toHaveBeenCalled(); test.wheel(); test.binding.dispose(); flushFrames();
    expect(test.onChange).not.toHaveBeenCalled(); expect(test.wheel().defaultPrevented).toBe(false);
    expect(test.event('gesturestart', { scale: 1 }).defaultPrevented).toBe(false);
  });
  it('zooms from letterbox margins around the view center', () => {
    const test = setup();
    expect(test.wheel({ ctrlKey: false, clientY: 100 }).defaultPrevented).toBe(true); finishFrames();
    const next = test.onChange.mock.calls.at(-1)![0];
    expect(next.width).toBeCloseTo(1000 / Math.exp(.2));
    expect(next.x + next.width / 2).toBeCloseTo(500);
    expect(next.y + next.height / 2).toBeCloseTo(250);
    test.binding.dispose();
  });
  it('ignores invalid wheel input without generating requests', () => {
    const test = setup(); test.wheel({ deltaY: NaN }); test.wheel({ deltaY: 0 });
    flushFrames(); expect(test.onChange).not.toHaveBeenCalled(); test.binding.dispose();
  });
  it('uses Safari cumulative scale once and suppresses duplicate wheel events', () => {
    const test = setup(); expect(test.event('gesturestart', { scale: 1 }).defaultPrevented).toBe(true);
    test.event('gesturechange', { scale: 1.5, clientX: 260, clientY: 395 }); test.wheel({ deltaY: -100 });
    test.event('gesturechange', { scale: 2, clientX: 260, clientY: 395 }); test.event('gestureend', {}); test.wheel({ deltaY: -100 }); flushFrames();
    expect(test.onChange).toHaveBeenCalledTimes(1);
    expect(test.onChange.mock.calls[0][0]).toEqual({ x: 125, y: 62.5, width: 500, height: 250 }); test.binding.dispose();
  });
  it('ignores Safari changes that started outside the slide and invalid gesture scales', () => {
    const test = setup(); expect(test.event('gesturechange', { scale: 2 }).defaultPrevented).toBe(false);
    test.event('gesturestart', { scale: 1 }); test.event('gesturechange', { scale: NaN }); test.event('gesturechange', { scale: 0 });
    flushFrames(); expect(test.onChange).not.toHaveBeenCalled(); test.event('gesturechange', { scale: 2 }); flushFrames();
    expect(test.onChange.mock.calls[0][0]).toEqual({ x: 250, y: 125, width: 500, height: 250 }); test.binding.dispose();
  });
});

describe('frame-coalesced panning', () => {
  it('publishes the latest pointer position once per frame and keeps the final release position', () => {
    const publish = vi.fn<(value: number) => void>(); const frame = createFramePublisher(publish);
    for (let position = 1; position <= 100; position++) frame.schedule(position);
    expect(frames.size).toBe(1); expect(publish).not.toHaveBeenCalled(); flushFrames();
    expect(publish.mock.calls).toEqual([[100]]);
    frame.schedule(101); frame.schedule(102); frame.flush();
    expect(publish.mock.calls).toEqual([[100], [102]]); expect(frames.size).toBe(0);
  });
  it('discards stale pan positions after a zoom, fit, source switch or unmount', () => {
    const publish = vi.fn<(value: number) => void>(); const frame = createFramePublisher(publish);
    frame.schedule(1); frame.cancel(); flushFrames(); frame.flush(); expect(publish).not.toHaveBeenCalled();
    frame.schedule(2); flushFrames(); expect(publish.mock.calls).toEqual([[2]]);
  });
});
