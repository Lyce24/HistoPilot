import { memo, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { MorphologyRegion, QualityEvidence } from '../api/morphology';
import { encodeQualityOverlay, paintQualityOverlay, qualityOverlaySize, type QualityOverlayOptions } from '../lib/qualityOverlay';

export interface QualityOverlayStatus { loading: boolean; error: Error | null }
interface Props {
  quality: QualityEvidence; view: MorphologyRegion; coverage: boolean; contours: boolean;
  onStatus?: (status: QualityOverlayStatus) => void;
}
interface Identity extends QualityOverlayOptions { quality: QualityEvidence }
interface Raster { identity: Identity; url: string; region: MorphologyRegion }
interface Request { identity: Identity; region: MorphologyRegion; width: number; height: number; strokePixels: number }

async function renderOverlay(request: Request, signal: AbortSignal): Promise<Raster> {
  const canvas = document.createElement('canvas');
  canvas.width = request.width; canvas.height = request.height;
  const context = canvas.getContext('2d');
  if (!context) { canvas.width = 0; canvas.height = 0; throw new Error('QC overlays could not be drawn in this browser.'); }
  let url: string | undefined;
  try {
    await paintQualityOverlay(context, request.identity.quality, request.identity, request.region, request.width, request.height, signal, request.strokePixels);
    url = await encodeQualityOverlay(canvas, signal);
    if (signal.aborted) throw signal.reason;
    return { identity: request.identity, region: request.region, url };
  } catch (error) {
    if (url) URL.revokeObjectURL(url);
    throw error;
  } finally { canvas.width = 0; canvas.height = 0; }
}

/** Two bounded cached images replace dense SVG geometry; selected shapes stay vector. */
export default memo(function QualityOverlayLayer({ quality, view, coverage, contours, onStatus }: Props) {
  const group = useRef<SVGGElement>(null), clipId = useId();
  const identity = useMemo<Identity>(() => ({ quality, coverage, contours }), [quality, quality.sourceFingerprint, coverage, contours]);
  const full = useMemo(() => ({ x: 0, y: 0, width: quality.width, height: quality.height }), [quality.width, quality.height]);
  const [viewport, setViewport] = useState({ width: 0, height: 0, ratio: 1 });
  const [overview, setOverview] = useState<Raster | null>(null), [detail, setDetail] = useState<Raster | null>(null);
  const [request, setRequest] = useState<Request | null>(null);
  const [overviewWork, setOverviewWork] = useState<{ identity: Identity; loading: boolean; error: Error | null }>({ identity, loading: true, error: null });
  const [detailWork, setDetailWork] = useState<{ identity: Identity; loading: boolean; error: Error | null }>({ identity, loading: false, error: null });
  const camera = useRef({ identity, view, viewport });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null), lastUpdate = useRef(0);
  const enabled = coverage || contours;
  useEffect(() => { camera.current = { identity, view, viewport }; });
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
  useEffect(() => { setOverview(null); setDetail(null); setRequest(null); }, [identity]);
  useEffect(() => () => { if (overview) URL.revokeObjectURL(overview.url); }, [overview]);
  useEffect(() => () => { if (detail) URL.revokeObjectURL(detail.url); }, [detail]);

  useEffect(() => {
    if (!enabled || viewport.width <= 0 || viewport.height <= 0) return;
    const controller = new AbortController();
    setOverviewWork({ identity, loading: true, error: null });
    // Prepare enough pixels for the next zoom, while keeping either side <=2048.
    const size = qualityOverlaySize(full, viewport.width * 2, viewport.height * 2, viewport.ratio);
    // Strokes represent one CSS pixel at the fitted overview, not its preparation scale.
    size.strokePixels *= 2;
    void renderOverlay({ identity, region: full, ...size }, controller.signal).then((raster) => {
      if (controller.signal.aborted) { URL.revokeObjectURL(raster.url); return; }
      setOverview(raster); setOverviewWork({ identity, loading: false, error: null });
    }, (error: unknown) => { if (!controller.signal.aborted) setOverviewWork({ identity, loading: false, error: error instanceof Error ? error : new Error(String(error)) }); });
    return () => controller.abort();
  }, [identity, enabled, full, viewport]);

  useEffect(() => {
    if (timer.current !== null) return;
    // Throttle rather than debounce: continuous navigation can still sharpen.
    timer.current = setTimeout(() => {
      timer.current = null; lastUpdate.current = Date.now();
      const current = camera.current;
      const atFull = current.view.x === 0 && current.view.y === 0 && current.view.width === current.identity.quality.width && current.view.height === current.identity.quality.height;
      if ((!current.identity.coverage && !current.identity.contours) || atFull || current.viewport.width <= 0 || current.viewport.height <= 0) { setRequest(null); return; }
      setRequest({ identity: current.identity, region: current.view, ...qualityOverlaySize(current.view, current.viewport.width, current.viewport.height, current.viewport.ratio) });
    }, Math.max(0, 120 - (Date.now() - lastUpdate.current)));
  }, [identity, view, viewport]);
  useEffect(() => () => { if (timer.current !== null) clearTimeout(timer.current); timer.current = null; }, []);
  useEffect(() => {
    if (!request || request.identity !== identity || overview?.identity !== identity) { setDetailWork({ identity, loading: false, error: null }); return; }
    const controller = new AbortController();
    setDetailWork({ identity, loading: true, error: null });
    void renderOverlay(request, controller.signal).then((raster) => {
      if (controller.signal.aborted) { URL.revokeObjectURL(raster.url); return; }
      setDetail(raster); setDetailWork({ identity, loading: false, error: null });
    }, (error: unknown) => { if (!controller.signal.aborted) setDetailWork({ identity, loading: false, error: error instanceof Error ? error : new Error(String(error)) }); });
    return () => controller.abort();
  }, [request, identity, overview?.identity]);

  const base = enabled && overview?.identity === identity ? overview : null;
  const atFull = view.x === 0 && view.y === 0 && view.width === full.width && view.height === full.height;
  const sharp = enabled && !atFull && detail?.identity === identity ? detail : null;
  const loading = enabled && (overviewWork.identity !== identity || overviewWork.loading || (detailWork.identity === identity && detailWork.loading));
  const error = enabled ? overviewWork.identity === identity && overviewWork.error || detailWork.identity === identity && detailWork.error || null : null;
  useEffect(() => { onStatus?.({ loading, error }); }, [onStatus, loading, error]);
  const exclusion = sharp ? `M0,0H${full.width}V${full.height}H0Z M${sharp.region.x},${sharp.region.y}h${sharp.region.width}v${sharp.region.height}h-${sharp.region.width}Z` : '';
  return <g ref={group} data-quality-overlay-layer="true" data-overlay-loading={loading} pointerEvents="none">
    {base && sharp ? <defs><clipPath id={clipId} clipPathUnits="userSpaceOnUse"><path d={exclusion} clipRule="evenodd" /></clipPath></defs> : null}
    {base ? <image data-quality-overlay="overview" href={base.url} {...base.region} clipPath={sharp ? `url(#${clipId})` : undefined} preserveAspectRatio="none" /> : null}
    {sharp ? <image data-quality-overlay="detail" href={sharp.url} {...sharp.region} preserveAspectRatio="none" /> : null}
  </g>;
});
