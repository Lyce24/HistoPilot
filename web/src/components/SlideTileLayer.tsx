import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type { SlideRegion } from '../api/interpretation';
import { planSlidePreparation, planSlidePrefetch, planSlideTiles, selectSlideTiles, SlideTileStore, type TileSnapshot } from '../lib/slideTiles';

export interface SlideTileStatus { loading: boolean; error: Error | null; preparing: boolean; prepared: number; preparationTotal: number }
interface Props {
  sourceKey: string; width: number; height: number; view: SlideRegion; enabled: boolean;
  fetchRegion: (region: SlideRegion, signal: AbortSignal, maxSize: number) => Promise<Blob>;
  onStatus: (status: SlideTileStatus) => void;
  retry: number;
}
const empty: TileSnapshot = { tiles: [], cachedTiles: [], loading: false, error: null, preparing: true, prepared: 0, preparationTotal: 0 };

async function decodeTile(blob: Blob, signal: AbortSignal) {
  const url = URL.createObjectURL(blob), image = new Image();
  image.src = url;
  let abort: () => void = () => {};
  try {
    await Promise.race([image.decode(), new Promise<never>((_, reject) => {
      abort = () => { image.src = ''; reject(new DOMException('Slide decoding cancelled', 'AbortError')); };
      if (signal.aborted) abort(); else signal.addEventListener('abort', abort, { once: true });
    })]);
    return { url, width: image.naturalWidth, height: image.naturalHeight };
  } catch (error) { URL.revokeObjectURL(url); throw error; }
  finally { signal.removeEventListener('abort', abort); }
}

/** Image tiles share the parent SVG's level-0 coordinates with every overlay. */
export default memo(function SlideTileLayer(props: Props) {
  const group = useRef<SVGGElement>(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0, ratio: 1 });
  const [rendered, setRendered] = useState<{ sourceKey: string; snapshot: TileSnapshot }>({ sourceKey: props.sourceKey, snapshot: empty });
  const latest = useRef(props);
  const store = useRef<SlideTileStore | null>(null);
  const scheduled = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastUpdate = useRef(0);
  const anticipation = useRef<ReturnType<typeof setTimeout> | null>(null);
  const camera = useRef({ ...props, viewport });
  useEffect(() => { latest.current = props; camera.current = { ...props, viewport }; });

  useEffect(() => {
    const svg = group.current?.ownerSVGElement;
    if (!svg) return;
    const measure = () => {
      const box = svg.getBoundingClientRect(), next = { width: box.width, height: box.height, ratio: Math.min(2, window.devicePixelRatio || 1) };
      setViewport((previous) => previous.width === next.width && previous.height === next.height && previous.ratio === next.ratio ? previous : next);
    };
    measure();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(svg); window.addEventListener('resize', measure);
    return () => { observer?.disconnect(); window.removeEventListener('resize', measure); };
  }, []);

  useEffect(() => {
    const sourceKey = props.sourceKey;
    const active = new SlideTileStore({
      fetch: (tile, signal) => latest.current.fetchRegion(tile.region, signal, tile.maxSize),
      decode: decodeTile, release: (url) => URL.revokeObjectURL(url),
      changed: () => setRendered({ sourceKey, snapshot: active.snapshot() }),
    });
    store.current = active; lastUpdate.current = 0;
    active.prepare(planSlidePreparation(props.width, props.height));
    return () => {
      if (scheduled.current !== null) clearTimeout(scheduled.current);
      if (anticipation.current !== null) clearTimeout(anticipation.current);
      anticipation.current = null; scheduled.current = null; active.dispose();
      if (store.current === active) store.current = null;
    };
  }, [props.sourceKey, props.width, props.height]);

  useEffect(() => {
    if (anticipation.current !== null) clearTimeout(anticipation.current);
    anticipation.current = setTimeout(() => {
      anticipation.current = null;
      const current = camera.current;
      if (current.enabled) store.current?.prefetch(planSlidePrefetch(current.view, current.width, current.height, current.viewport.width, current.viewport.height, current.viewport.ratio));
    }, 200);
    if (scheduled.current !== null) return;
    // This is a throttle, not a trailing debounce: continuing gestures still
    // request fresh detail, without making a native read for every wheel event.
    scheduled.current = setTimeout(() => {
      scheduled.current = null; lastUpdate.current = Date.now();
      const current = camera.current;
      store.current?.update(planSlideTiles(current.view, current.width, current.height, current.viewport.width, current.viewport.height, current.viewport.ratio), current.view);
    }, Math.max(0, 80 - (Date.now() - lastUpdate.current)));
  }, [props.view, props.width, props.height, props.enabled, props.sourceKey, viewport]);

  useEffect(() => { if (props.retry > 0) store.current?.retry(); }, [props.retry]);
  const snapshot = rendered.sourceKey === props.sourceKey ? rendered.snapshot : empty;
  useEffect(() => { props.onStatus({ loading: snapshot.loading, error: snapshot.error, preparing: snapshot.preparing, prepared: snapshot.prepared, preparationTotal: snapshot.preparationTotal }); }, [props.onStatus, snapshot.loading, snapshot.error, snapshot.preparing, snapshot.prepared, snapshot.preparationTotal]);
  const visible = useMemo(() => selectSlideTiles(snapshot.cachedTiles, props.view,
    planSlideTiles(props.view, props.width, props.height, viewport.width, viewport.height, viewport.ratio)),
  [snapshot.cachedTiles, props.view, props.width, props.height, viewport]);
  const images = useMemo(() => visible.map((tile) => <image key={tile.key} data-slide-detail="true" data-slide-tile={tile.key} data-tile-level={tile.level} data-tile-source={props.sourceKey} href={tile.url} {...tile.region} preserveAspectRatio="none" />), [visible, props.sourceKey]);
  return <g ref={group} data-slide-tile-layer="true" data-prepared-tiles={snapshot.prepared} data-preparation-total={snapshot.preparationTotal} data-preparing={snapshot.preparing} pointerEvents="none">
    {images}
  </g>;
});
