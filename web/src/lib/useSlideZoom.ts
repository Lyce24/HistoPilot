import { useCallback, useLayoutEffect, useMemo, useRef } from 'react';
import type { SlideRegion } from '../api/interpretation';
import { slidePointAt, zoomRegionAt } from './slideGeometry';

export interface SlideZoomOptions {
  view: SlideRegion;
  width: number;
  height: number;
  onChange: (next: SlideRegion) => void;
  onInteraction?: () => void;
  disabled?: boolean;
}
interface SafariGestureEvent extends Event { scale: number; clientX?: number; clientY?: number }

/** Browsers report wheel distances as pixels, text lines, or viewport pages. */
export function normalizedZoomDelta(deltaY: number, deltaMode: number, viewportHeight: number): number {
  if (!Number.isFinite(deltaY)) return 0;
  const page = Number.isFinite(viewportHeight) && viewportHeight > 0 ? viewportHeight : 800;
  return Math.max(-240, Math.min(240, deltaY * (deltaMode === 1 ? 16 : deltaMode === 2 ? page : 1)));
}
const sameView = (a: SlideRegion, b: SlideRegion) => a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

/** Native listeners stay inside the slide window, including its floating controls. */
export function bindSlideZoom(element: SVGSVGElement, getOptions: () => SlideZoomOptions) {
  const target = element.parentElement ?? element;
  let observed = getOptions();
  let working = observed.view;
  let rendered = working;
  let easing = 0;
  let previousFrame: number | null = null;
  let emitted: SlideRegion | null = null;
  let frame: number | null = null;
  let dirty = false;
  let disposed = false;
  let gestureScale: number | null = null;
  let gestureFinishedAt = -Infinity;
  let wheelMode: 'direct' | 'eased' | null = null;
  let wheelAt = -Infinity;
  let wheelPinch = false;

  function cancelFrame() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    dirty = false;
    easing = 0;
    previousFrame = null;
  }
  function resetWheel() { wheelMode = null; wheelAt = -Infinity; }
  function sync() {
    const next = getOptions();
    // A new externally selected region wins over an animation frame still waiting to run.
    const resized = next.width !== observed.width || next.height !== observed.height;
    const replaced = next.view !== observed.view && next.view !== emitted;
    if (resized || replaced || next.disabled) {
      cancelFrame(); resetWheel();
      working = rendered = next.view;
      emitted = null;
    }
    observed = next;
    return next;
  }
  function flush(timestamp: number) {
    frame = null;
    const options = sync();
    if (disposed || options.disabled || !dirty) return;
    dirty = false;
    if (easing > 0) {
      const elapsed = previousFrame === null ? 1000 / 60 : Math.max(1, Math.min(64, timestamp - previousFrame));
      easing = Math.max(0, easing - elapsed);
      const amount = easing === 0 ? 1 : 1 - Math.exp(-elapsed / 25);
      rendered = amount === 1 ? working : { x: rendered.x + (working.x - rendered.x) * amount, y: rendered.y + (working.y - rendered.y) * amount, width: rendered.width + (working.width - rendered.width) * amount, height: rendered.height + (working.height - rendered.height) * amount };
    } else rendered = working;
    previousFrame = timestamp;
    emitted = rendered;
    options.onChange(rendered);
    if (!sameView(rendered, working)) { dirty = true; frame = requestAnimationFrame(flush); }
    else previousFrame = null;
  }
  function zoom(factor: number, clientX: number, clientY: number, animate = false, bounds = element.getBoundingClientRect()) {
    const options = sync();
    if (disposed || options.disabled || !Number.isFinite(factor) || factor <= 0 || options.width <= 0 || options.height <= 0) return;
    options.onInteraction?.();
    // Empty letterbox margins still belong to the viewer; zoom around its center there.
    const anchor = slidePointAt(working, bounds, clientX, clientY)
      ?? { x: working.x + working.width / 2, y: working.y + working.height / 2 };
    const next = zoomRegionAt(working, factor, options.width, options.height, anchor);
    if (sameView(next, working)) return;
    working = next;
    // Smooth coarse mouse notches for a short, bounded tail. Fine trackpad and pinch
    // deltas follow the fingers directly, and reduced-motion users get no easing.
    easing = animate ? 120 : 0;
    dirty = true;
    if (frame === null) frame = requestAnimationFrame(flush);
  }
  function pointerDown(nativeEvent: Event) {
    const event = nativeEvent as PointerEvent;
    if (event.button !== 0 || event.isPrimary === false) return;
    const options = sync();
    // Capture runs before React's drag handler reads its view and screen transform.
    // Stop at the committed view, including when a published frame has not committed.
    const pending = frame !== null || dirty || (emitted !== null && emitted !== options.view);
    cancelFrame(); resetWheel();
    working = rendered = options.view;
    emitted = null;
    if (pending) options.onChange(options.view);
  }
  function wheel(nativeEvent: Event) {
    const event = nativeEvent as WheelEvent;
    // Ordinary wheel and trackpad pinch both zoom here. Outside this window, the page scrolls.
    // Consume wheel even at zoom limits or while drawing so the page never jumps underneath it.
    event.preventDefault();
    if (sync().disabled) return;
    if (gestureScale !== null || Date.now() - gestureFinishedAt < 100) return;
    const bounds = element.getBoundingClientRect();
    const delta = normalizedZoomDelta(event.deltaY, event.deltaMode, bounds.height);
    if (delta === 0) return;
    const now = performance.now();
    if (wheelMode === null || now - wheelAt > 180 || wheelPinch !== event.ctrlKey) {
      // Fractional pixels are precise input. Latch the motion style so a trackpad's
      // accelerating/decelerating deltas do not alternate between jumps and easing.
      const coarse = event.deltaMode !== 0 || (Number.isInteger(event.deltaY) && Math.abs(event.deltaY) >= 40);
      wheelMode = !event.ctrlKey && coarse ? 'eased' : 'direct';
    }
    wheelAt = now; wheelPinch = event.ctrlKey;
    const reducedMotion = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    zoom(Math.exp(-delta * .004), event.clientX, event.clientY, wheelMode === 'eased' && !reducedMotion, bounds);
  }
  function gestureStart(event: Event) {
    event.preventDefault();
    if (sync().disabled) return;
    resetWheel();
    const scale = (event as SafariGestureEvent).scale;
    gestureScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  }
  function gestureChange(event: Event) {
    if (sync().disabled || gestureScale === null) return;
    event.preventDefault();
    const gesture = event as SafariGestureEvent;
    if (!Number.isFinite(gesture.scale) || gesture.scale <= 0) return;
    const box = element.getBoundingClientRect();
    const clientX = Number.isFinite(gesture.clientX) ? gesture.clientX! : box.left + box.width / 2;
    const clientY = Number.isFinite(gesture.clientY) ? gesture.clientY! : box.top + box.height / 2;
    zoom(gesture.scale / gestureScale, clientX, clientY);
    gestureScale = gesture.scale;
  }
  function gestureEnd(event: Event) {
    if (gestureScale === null) return;
    event.preventDefault();
    gestureScale = null;
    gestureFinishedAt = Date.now();
  }
  const listenerOptions = { passive: false };
  const captureOptions = { capture: true };
  target.addEventListener('pointerdown', pointerDown, captureOptions);
  target.addEventListener('wheel', wheel, listenerOptions);
  target.addEventListener('gesturestart', gestureStart, listenerOptions);
  target.addEventListener('gesturechange', gestureChange, listenerOptions);
  target.addEventListener('gestureend', gestureEnd, listenerOptions);
  return {
    sync,
    dispose() {
      disposed = true;
      cancelFrame();
      target.removeEventListener('pointerdown', pointerDown, captureOptions);
      target.removeEventListener('wheel', wheel);
      target.removeEventListener('gesturestart', gestureStart);
      target.removeEventListener('gesturechange', gestureChange);
      target.removeEventListener('gestureend', gestureEnd);
    },
  };
}

/** Stable callback ref also handles viewers whose SVG appears after its image loads. */
export function useSlideZoom(options: SlideZoomOptions) {
  const latest = useRef(options);
  const binding = useRef<ReturnType<typeof bindSlideZoom> | null>(null);
  useLayoutEffect(() => { latest.current = options; binding.current?.sync(); });
  return useCallback((element: SVGSVGElement | null) => {
    binding.current?.dispose();
    binding.current = element ? bindSlideZoom(element, () => latest.current) : null;
  }, []);
}

/** Keep the latest pointer position, with at most one camera update per paint. */
export function createFramePublisher<T>(publish: (value: T) => void) {
  let frame: number | null = null;
  let pending: { value: T } | null = null;
  function cancel() { if (frame !== null) cancelAnimationFrame(frame); frame = null; pending = null; }
  function flush() { const latest = pending; cancel(); if (latest) publish(latest.value); }
  return { schedule(value: T) { pending = { value }; if (frame === null) frame = requestAnimationFrame(flush); }, flush, cancel };
}
export function useSlideFrame<T>(publish: (value: T) => void) {
  const latest = useRef(publish);
  const publisher = useMemo(() => createFramePublisher<T>((value) => latest.current(value)), []);
  useLayoutEffect(() => { latest.current = publish; });
  useLayoutEffect(() => () => publisher.cancel(), [publisher]);
  return publisher;
}
