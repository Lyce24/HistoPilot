import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type PointerEvent } from 'react';
import { morphology, type MorphologyRegion, type QualityEvidence } from '../api/morphology';
import { boundedRegion, zoomRegionAt } from '../lib/slideGeometry';
import { useSlideFrame, useSlideZoom } from '../lib/useSlideZoom';
import { hitTestQualityPatch } from '../lib/qualityOverlay';
import { isSlideSourceError } from '../lib/slideTiles';
import QualityOverlayLayer, { type QualityOverlayStatus } from './QualityOverlayLayer';
import SlideTileLayer, { type SlideTileStatus } from './SlideTileLayer';
import { ErrorNotice } from './ui';

interface Props {
  project: string; datasetId: string; slideId: string; quality: QualityEvidence;
  overviewURL?: string; patchIndex?: number; selectedRegion?: MorphologyRegion;
  onPatch: (index: number) => void; onRegion: (region: MorphologyRegion) => void;
  onSourceError?: (error: Error) => void; onReload?: () => void;
}
interface Drag {
  id: number; clientX: number; clientY: number; scaleX: number; scaleY: number;
  view: MorphologyRegion; start: { x: number; y: number }; drawing: boolean; moved: boolean;
}

export default function QualitySlideCanvas({ project, datasetId, slideId, quality, overviewURL, patchIndex, selectedRegion, onPatch, onRegion, onSourceError, onReload }: Props) {
  const navigationId = useId();
  const full = useMemo(() => ({ x: 0, y: 0, width: quality.width, height: quality.height }), [quality.width, quality.height]);
  const [view, setView] = useState(full);
  const panFrame = useSlideFrame(setView);
  const [detail, setDetail] = useState<SlideTileStatus>({ loading: false, error: null, preparing: true, prepared: 0, preparationTotal: 0 });
  const [overlay, setOverlay] = useState<QualityOverlayStatus>({ loading: true, error: null });
  useEffect(() => { if (detail.error && isSlideSourceError(detail.error)) onSourceError?.(detail.error); }, [detail.error, onSourceError]);
  const [detailRetry, setDetailRetry] = useState(0);
  const [coverage, setCoverage] = useState(true);
  const [contours, setContours] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const [draftRegion, setDraftRegion] = useState<MorphologyRegion>();
  const [panning, setPanning] = useState(false);
  const drag = useRef<Drag | null>(null);
  const ignorePatchClick = useRef(false);
  const atFull = view.x === 0 && view.y === 0 && view.width === full.width && view.height === full.height;
  const fetchDetail = useCallback((region: MorphologyRegion, signal: AbortSignal, maxSize: number) => morphology.image(project, datasetId, slideId, signal, region, maxSize, quality.sourceFingerprint), [project, datasetId, slideId, quality.sourceFingerprint]);
  const zoomRef = useSlideZoom({ view, width: full.width, height: full.height, disabled: drawing, onInteraction: () => {
    panFrame.cancel(); drag.current = null; ignorePatchClick.current = true; setPanning(false);
  }, onChange: setView });
  function choosePatch(event: MouseEvent<SVGSVGElement>) {
    if (drawing || !coverage || ignorePatchClick.current) return;
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    const index = hitTestQualityPatch(quality, point.x, point.y);
    if (index !== undefined) onPatch(index);
  }
  function point(event: PointerEvent<SVGSVGElement>) {
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return null;
    const value = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return { x: Math.max(view.x, Math.min(view.x + view.width, value.x)), y: Math.max(view.y, Math.min(view.y + view.height, value.y)) };
  }
  function rectangle(start: { x: number; y: number }, end: { x: number; y: number }) {
    return { x: Math.floor(Math.min(start.x, end.x)), y: Math.floor(Math.min(start.y, end.y)), width: Math.floor(Math.abs(end.x - start.x)), height: Math.floor(Math.abs(end.y - start.y)) };
  }
  function pointerDown(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0 || !event.isPrimary || (event.pointerType === 'touch' && !drawing)) return;
    const start = point(event), matrix = event.currentTarget.getScreenCTM();
    if (!start || !matrix) return;
    event.currentTarget.focus({ preventScroll: true });
    ignorePatchClick.current = drawing;
    drag.current = { id: event.pointerId, clientX: event.clientX, clientY: event.clientY, scaleX: matrix.a, scaleY: matrix.d, view, start, drawing, moved: false };
    if (drawing) event.currentTarget.setPointerCapture(event.pointerId);
  }
  function pointerMove(event: PointerEvent<SVGSVGElement>) {
    const active = drag.current;
    if (!active || active.id !== event.pointerId) return;
    if ((event.buttons & 1) === 0) { finish(event, true); return; }
    if (active.drawing) { const end = point(event); if (end) setDraftRegion(rectangle(active.start, end)); return; }
    if (!active.moved && Math.hypot(event.clientX - active.clientX, event.clientY - active.clientY) < 4) return;
    active.moved = true; ignorePatchClick.current = true; setPanning(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    panFrame.schedule(boundedRegion({ ...active.view, x: active.view.x - (event.clientX - active.clientX) / active.scaleX, y: active.view.y - (event.clientY - active.clientY) / active.scaleY }, full.width, full.height));
  }
  function finish(event: PointerEvent<SVGSVGElement>, cancelled = false) {
    const active = drag.current;
    if (active?.id !== event.pointerId) return;
    if (active.drawing && !cancelled) {
      const end = point(event);
      if (end) { const region = rectangle(active.start, end); if (region.width >= 1 && region.height >= 1) { onRegion(region); setDrawing(false); } }
    }
    if (cancelled) panFrame.cancel(); else panFrame.flush();
    drag.current = null; setDraftRegion(undefined); setPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  function zoom(factor: number) { panFrame.cancel(); drag.current = null; setPanning(false); setView((value) => zoomRegionAt(value, factor, full.width, full.height, { x: value.x + value.width / 2, y: value.y + value.height / 2 })); }
  function fit() { panFrame.cancel(); drag.current = null; setDraftRegion(undefined); setPanning(false); setView(full); }
  function keyDown(event: KeyboardEvent<SVGSVGElement>) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const delta = { ArrowLeft: [-.2, 0], ArrowRight: [.2, 0], ArrowUp: [0, -.2], ArrowDown: [0, .2] }[event.key];
    if (delta && !drawing) { event.preventDefault(); panFrame.cancel(); drag.current = null; setPanning(false); setView((value) => boundedRegion({ ...value, x: value.x + value.width * delta[0], y: value.y + value.height * delta[1] }, full.width, full.height)); }
    else if (!drawing && ['+', '=', '-'].includes(event.key)) { event.preventDefault(); zoom(event.key === '-' ? .5 : 2); }
    else if (event.key === 'Home') { event.preventDefault(); fit(); }
    else if (event.key === 'Escape') { panFrame.cancel(); drag.current = null; setDraftRegion(undefined); setDrawing(false); setPanning(false); }
  }
  const highlight = draftRegion ?? selectedRegion;
  const selectedPatch = coverage ? quality.patches.find((row) => row.patchIndex === patchIndex) : undefined;
  const preparing = detail.preparing || (atFull && overlay.loading);

  return <>
    <div className="inline-actions">
      {quality.patchCount != null ? <label className="checkbox-label"><input type="checkbox" checked={coverage} onChange={(event) => setCoverage(event.target.checked)} />Patch coverage</label> : null}
      {quality.tissueContours.length ? <label className="checkbox-label"><input type="checkbox" checked={contours} onChange={(event) => setContours(event.target.checked)} />Recorded tissue contours</label> : null}
      <button type="button" className="btn btn-secondary" aria-pressed={drawing} onClick={() => { panFrame.cancel(); drag.current = null; setDraftRegion(undefined); setDrawing(!drawing); }}>{drawing ? 'Cancel region selection' : 'Draw review region'}</button>
      <button type="button" className="btn btn-secondary" disabled={!selectedRegion || drawing} onClick={() => { panFrame.cancel(); drag.current = null; setPanning(false); if (selectedRegion) setView(boundedRegion(selectedRegion, full.width, full.height)); }}>Zoom to selection</button>
      <button type="button" className="btn btn-secondary" disabled={atFull} onClick={fit}>Fit slide</button>
    </div>
    {drawing ? <p role="status">Drag a rectangle on the image. Its level-0 coordinates can be saved with the review. Press Escape to cancel.</p> : null}
    <div className={`morphology-image-scroll${drawing ? ' is-drawing' : ''}${panning ? ' is-panning' : ''}`}>
      <svg ref={zoomRef} style={{ touchAction: drawing ? 'none' : 'pan-y pinch-zoom' }} viewBox={`${view.x} ${view.y} ${view.width} ${view.height}`} preserveAspectRatio="xMidYMid meet" tabIndex={0} role="group" aria-label={`Exact slide ${slideId}${quality.patchCount != null ? ' with patch coverage' : ''}`} aria-describedby={navigationId} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={(event) => finish(event)} onPointerCancel={(event) => finish(event, true)} onLostPointerCapture={(event) => finish(event, true)} onKeyDown={keyDown} onClick={choosePatch}>
        <title>{`${slideId}: scroll or pinch to zoom; drag to pan; + / − to zoom; arrows to pan; Home to fit slide.`}</title>
        <rect x="0" y="0" width={full.width} height={full.height} fill="#eee9e2" />
        {overviewURL ? <image href={overviewURL} x="0" y="0" width={full.width} height={full.height} preserveAspectRatio="none" /> : null}
        <SlideTileLayer sourceKey={`morphology:${project}:${datasetId}:${slideId}:${quality.sourceFingerprint ?? 'legacy'}`} width={full.width} height={full.height} view={view} enabled={!atFull} fetchRegion={fetchDetail} onStatus={setDetail} retry={detailRetry} />
        <QualityOverlayLayer quality={quality} view={view} coverage={coverage} contours={contours} onStatus={setOverlay} />
        {selectedPatch ? quality.patchWidth && quality.patchHeight ? <rect data-selected-patch={selectedPatch.patchIndex} x={selectedPatch.x} y={selectedPatch.y} width={Math.min(quality.patchWidth, quality.width - selectedPatch.x)} height={Math.min(quality.patchHeight, quality.height - selectedPatch.y)} fill="#f2a90080" stroke="#a44900" strokeWidth="1" vectorEffect="non-scaling-stroke" pointerEvents="none" /> : <circle data-selected-patch={selectedPatch.patchIndex} cx={selectedPatch.x} cy={selectedPatch.y} r={Math.max(quality.width, quality.height) / 470} fill="#f2a90080" stroke="#a44900" vectorEffect="non-scaling-stroke" pointerEvents="none" /> : null}
        {highlight ? <rect {...highlight} fill="none" stroke="#a44900" strokeWidth="3" vectorEffect="non-scaling-stroke" pointerEvents="none" /> : null}
      </svg>
      <div className="morphology-zoom-controls" aria-label="Slide zoom controls"><button type="button" aria-label="Zoom out" disabled={atFull || drawing} onClick={() => zoom(.5)}>−</button><span>{Math.max(full.width / view.width, full.height / view.height).toFixed(1)}× view</span><button type="button" aria-label="Zoom in" disabled={drawing} onClick={() => zoom(2)}>+</button></div>
      {preparing || (detail.loading && !atFull) || overlay.loading ? <span className="morphology-image-status" role="status">{detail.preparing ? `Preparing slide · ${detail.preparationTotal ? Math.round(detail.prepared / detail.preparationTotal * 100) : 0}%` : overlay.loading ? 'Preparing QC overlay…' : 'Loading detail…'}</span> : null}
    </div>
    <p className="muted" role="status">{preparing ? 'Preparing slide for smooth zooming. This may take a moment.' : detail.error ? 'Some slide detail could not be prepared. The overview remains available.' : overlay.error ? 'The slide is available, but its QC overlay could not be prepared.' : 'Slide ready · Nearby zoom levels are cached; finer detail loads as needed.'}</p>
    <p id={navigationId} className="muted morphology-navigation-hint">Scroll or pinch to zoom · Drag to pan · + / − to zoom · Arrow keys to pan · Home to fit slide</p>
    {overlay.error ? <div className="callout" role="status"><ErrorNotice error={overlay.error} />{onReload ? <button type="button" className="btn btn-secondary" onClick={onReload}>Reload slide</button> : null}</div> : null}
    {detail.error && !isSlideSourceError(detail.error) ? <div className="callout" role="status">Sharper detail could not be loaded. The slide overview remains available.<ErrorNotice error={detail.error} /><button type="button" className="btn btn-secondary" disabled={detail.loading} onClick={() => setDetailRetry((value) => value + 1)}>Retry slide detail</button></div> : null}
  </>;
}
