import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { interpretations, type AttentionPatch, type Interpretation, type InterpretationSlide, type InterpretationSlideResult, type RankedAttentionPatch, type SlideRegion } from '../api/interpretation';
import { ErrorNotice } from './ui';
import { attentionAt, attentionColor, boundedRegion, fitRegion, integerRegion, markerScale, patchIntersectsRegion, patchRegion, zoomRegion } from '../lib/slideGeometry';
import { formatStatistic } from '../lib/evidenceCharts';
import RankedAttentionPatches, { SelectedAttentionPatch } from './RankedAttentionPatches';
import RankedPatchMarkers from './RankedPatchMarkers';
import './RankedAttentionPatches.css';

const MAX_VISIBLE_PATCHES = 100000;
function useBlobImage(blob?: Blob) {
  const [image, setImage] = useState<{ blob: Blob; url: string } | null>(null);
  useEffect(() => {
    if (!blob) return;
    const url = URL.createObjectURL(blob); setImage({ blob, url });
    return () => URL.revokeObjectURL(url);
  }, [blob]);
  return image && image.blob === blob ? image.url : undefined;
}
interface ViewerProps { project: string; record: Interpretation; slide: InterpretationSlide; result: InterpretationSlideResult }
export default function AttentionSlideViewer(props: ViewerProps) {
  return <AttentionSlideWorkspace key={`${props.project}:${props.record.id}:${props.slide.slideId}`} {...props} />;
}
function AttentionSlideWorkspace({ project, record, slide, result }: ViewerProps) {
  const full = { x: 0, y: 0, width: slide.width, height: slide.height };
  const [view, setView] = useState<SlideRegion>(full);
  const [settledView, setSettledView] = useState<SlideRegion>(full);
  const [member, setMember] = useState('mean');
  const [opacity, setOpacity] = useState(.55);
  const [minimum, setMinimum] = useState(0);
  const [topCount, setTopCount] = useState<10 | 20>(10);
  const [showRankedBoxes, setShowRankedBoxes] = useState(true);
  const [railOpen, setRailOpen] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });
  const [selection, setSelection] = useState<{ patch: AttentionPatch; member: string; source: 'map' | 'rank' } | null>(null);
  const [heatmap, setHeatmap] = useState<{ url: string; signature: string } | null>(null);
  const [downloadError, setDownloadError] = useState<Error | null>(null);
  const drag = useRef<{ x: number; y: number; scaleX: number; scaleY: number; view: SlideRegion; moved: boolean } | null>(null);
  const svg = useRef<SVGSVGElement>(null);
  const workspaceRef = useRef<HTMLElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const userMoved = useRef(false);
  const initialFit = useRef(false);
  useEffect(() => {
    if (!svg.current) return;
    const element = svg.current;
    const measure = () => { const box = element.getBoundingClientRect(); setViewportSize({ width: box.width, height: box.height }); };
    measure();
    if (typeof ResizeObserver === 'undefined') { window.addEventListener('resize', measure); return () => window.removeEventListener('resize', measure); }
    const observer = new ResizeObserver(measure); observer.observe(element); return () => observer.disconnect();
  }, []);
  useEffect(() => { const timer = setTimeout(() => setSettledView(integerRegion(view, slide.width, slide.height)), 250); return () => clearTimeout(timer); }, [view, slide.width, slide.height]);
  useEffect(() => {
    if (!expanded) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden'; workspaceRef.current?.focus();
    const handleKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); setExpanded(false); return; }
      if (event.key !== 'Tab') return;
      const focusable = Array.from(workspaceRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input, select, summary, [tabindex="0"]') ?? []).filter((element) => element.getClientRects().length > 0);
      const first = focusable[0], last = focusable.at(-1);
      if (event.shiftKey && (document.activeElement === first || document.activeElement === workspaceRef.current)) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', handleKey);
    return () => { document.body.style.overflow = overflow; document.removeEventListener('keydown', handleKey); previousFocus?.focus(); };
  }, [expanded]);
  const thumbnail = useQuery({ queryKey: ['interpretation-thumbnail', project, record.id, slide.slideId], queryFn: ({ signal }) => interpretations.thumbnail(project, record.id, slide.slideId, signal), staleTime: Infinity });
  const zoomed = settledView.width < slide.width || settledView.height < slide.height;
  const region = useQuery({ queryKey: ['interpretation-region', project, record.id, slide.slideId, settledView], queryFn: ({ signal }) => interpretations.region(project, record.id, slide.slideId, settledView, signal), enabled: zoomed, staleTime: 60000, gcTime: 60000 });
  const attention = useQuery({ queryKey: ['interpretation-attention', project, record.id, slide.slideId, member, settledView], queryFn: async ({ signal }) => {
    const first = await interpretations.attention(project, record.id, slide.slideId, member, settledView, 0, 10000, signal);
    const patches = [...first.patches];
    while (patches.length < first.total && patches.length < MAX_VISIBLE_PATCHES) {
      const page = await interpretations.attention(project, record.id, slide.slideId, member, settledView, patches.length, Math.min(10000, MAX_VISIBLE_PATCHES - patches.length), signal);
      if (!page.patches.length) break;
      patches.push(...page.patches);
    }
    return { ...first, patches };
  }, staleTime: 60000, gcTime: 60000 });
  const top = useQuery({ queryKey: ['interpretation-top-attention', project, record.id, slide.slideId, member, topCount], queryFn: ({ signal }) => interpretations.topAttention(project, record.id, slide.slideId, member, topCount, signal), staleTime: Infinity, gcTime: 60000 });
  const ranked = !top.isError ? top.data?.patches ?? [] : [];
  const coverage = !top.isError ? top.data?.coordinateBounds : null;
  useEffect(() => {
    if (!coverage || initialFit.current || userMoved.current || viewportSize.width <= 0 || viewportSize.height <= 0) return;
    initialFit.current = true;
    setView(fitRegion(coverage, slide.width, slide.height, viewportSize.width, viewportSize.height, .08));
  }, [coverage, slide.width, slide.height, viewportSize.width, viewportSize.height]);
  const manualSelection = selection?.member === member ? selection : null;
  const selectedPatch = manualSelection?.patch ?? ranked[0] ?? null;
  const verifiedSelectedPatch = !top.isError && !(manualSelection?.source === 'map' && attention.isError) ? selectedPatch : null;
  const selectedRank = verifiedSelectedPatch ? ranked.find((patch) => patch.index === verifiedSelectedPatch.index)?.rank : undefined;
  const cropContext = { project, interpretationId: record.id, slideId: slide.slideId, member };
  const rankMarkerScale = markerScale(view, viewportSize.width, viewportSize.height);
  const thumbnailURL = useBlobImage(thumbnail.data);
  const regionURL = useBlobImage(zoomed ? region.data : undefined);
  const signature = `${project}:${record.id}:${slide.slideId}:${member}:${JSON.stringify(settledView)}:${minimum}`;
  useEffect(() => {
    if (!attention.data) return;
    const canvas = document.createElement('canvas');
    const scale = Math.min(2048 / settledView.width, 2048 / settledView.height);
    canvas.width = Math.max(1, Math.ceil(settledView.width * scale)); canvas.height = Math.max(1, Math.ceil(settledView.height * scale));
    const scaleX = canvas.width / settledView.width, scaleY = canvas.height / settledView.height;
    const context = canvas.getContext('2d'); if (!context) return;
    const map = attention.data;
    for (const patch of [...map.patches].sort((a, b) => a.weight - b.weight)) {
      if (patch.percentile < minimum) continue;
      context.fillStyle = attentionColor(patch.percentile);
      context.fillRect((patch.x - settledView.x) * scaleX, (patch.y - settledView.y) * scaleY, map.patchWidthLevel0 * scaleX, map.patchHeightLevel0 * scaleY);
    }
    let alive = true, objectURL: string | undefined;
    canvas.toBlob((blob) => { if (!alive || !blob) return; objectURL = URL.createObjectURL(blob); setHeatmap({ url: objectURL, signature }); });
    return () => { alive = false; if (objectURL) URL.revokeObjectURL(objectURL); };
  }, [attention.data, settledView, minimum, signature]);
  function moveView(next: SlideRegion) { userMoved.current = true; setView(next); }
  function fitCoverage() { moveView(coverage ? fitRegion(coverage, slide.width, slide.height, viewportSize.width, viewportSize.height, .08) : full); }
  function zoom(factor: number) { userMoved.current = true; setView((value) => zoomRegion(value, factor, slide.width, slide.height)); }
  function pointAt(event: PointerEvent<SVGSVGElement>) {
    const transform = svg.current?.getScreenCTM();
    if (!transform || !svg.current) return null;
    const point = svg.current.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
    return point.matrixTransform(transform.inverse());
  }
  function pointerDown(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    const transform = svg.current?.getScreenCTM(); if (!transform) return;
    svg.current?.setPointerCapture(event.pointerId); drag.current = { x: event.clientX, y: event.clientY, scaleX: transform.a, scaleY: transform.d, view, moved: false };
  }
  function pointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!drag.current) return;
    if (Math.hypot(event.clientX - drag.current.x, event.clientY - drag.current.y) < 4 && !drag.current.moved) return;
    drag.current.moved = true; userMoved.current = true;
    const dx = (event.clientX - drag.current.x) / drag.current.scaleX, dy = (event.clientY - drag.current.y) / drag.current.scaleY;
    setView(boundedRegion({ ...drag.current.view, x: drag.current.view.x - dx, y: drag.current.view.y - dy }, slide.width, slide.height));
  }
  function pointerUp(event: PointerEvent<SVGSVGElement>) {
    const point = pointAt(event);
    if (drag.current && !drag.current.moved && point && attention.data && !attention.isError && !top.isError) {
      const patch = attentionAt(attention.data.patches, point.x, point.y, attention.data.patchWidthLevel0, attention.data.patchHeightLevel0, minimum);
      if (patch) { setSelection({ patch, member, source: 'map' }); setRailOpen(true); }
    }
    drag.current = null;
    if (svg.current?.hasPointerCapture(event.pointerId)) svg.current.releasePointerCapture(event.pointerId);
  }
  function keyDown(event: KeyboardEvent<SVGSVGElement>) {
    const delta = { ArrowLeft: [-.2, 0], ArrowRight: [.2, 0], ArrowUp: [0, -.2], ArrowDown: [0, .2] }[event.key];
    if (delta) { event.preventDefault(); userMoved.current = true; setView((value) => boundedRegion({ ...value, x: value.x + value.width * delta[0], y: value.y + value.height * delta[1] }, slide.width, slide.height)); }
    else if (event.key === '+' || event.key === '=' || event.key === '-') { event.preventDefault(); zoom(event.key === '-' ? .5 : 2); }
    else if (event.key === 'Home') { event.preventDefault(); fitCoverage(); }
  }
  function centerPatch(patch: AttentionPatch) {
    moveView(patchRegion(patch, slide.patchWidthLevel0, slide.patchHeightLevel0, slide.width, slide.height, viewportSize.width / viewportSize.height));
  }
  function inspectRankedPatch(patch: RankedAttentionPatch) {
    setSelection({ patch, member, source: 'rank' }); setRailOpen(true); centerPatch(patch);
    if (!expanded && viewportSize.width < 600) viewerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  async function download() { setDownloadError(null); try { await interpretations.download(project, record.id, result.attentionArtifact); } catch (error) { setDownloadError(error instanceof Error ? error : new Error('Attention download failed.')); } }
  const zoomLabel = `${formatStatistic(Math.max(slide.width / view.width, slide.height / view.height), 1)}× view`;
  return <section ref={workspaceRef} className={`attention-workspace ${expanded ? 'is-expanded' : ''}`} role={expanded ? 'dialog' : 'region'} aria-modal={expanded ? true : undefined} aria-label={`Attention workspace for ${slide.slideId}`} tabIndex={expanded ? -1 : undefined}>
    <div className="attention-controls-primary">
      <label className="attention-map-select">Attention map<select className="field" value={member} onChange={(event) => { setMember(event.target.value); setSelection(null); }}><option value="mean">{record.manifest.method === 'ensemble' ? 'Ensemble mean' : 'Refit model'}</option>{record.manifest.memberCount > 1 ? result.members.map((item) => <option key={item.index} value={String(item.index)}>Member {item.index + 1}</option>) : null}</select></label>
      <div className="attention-fit-actions"><button className="btn btn-secondary" disabled={!coverage} onClick={fitCoverage}>Fit patch coverage</button><button className="btn btn-secondary" onClick={() => moveView(full)}>Fit slide</button></div>
      <div className="attention-layout-actions"><button className="btn btn-secondary" aria-expanded={railOpen} onClick={() => setRailOpen((open) => !open)}>{railOpen ? 'Hide patches' : 'Show patches'}</button><button className="btn btn-secondary" aria-pressed={expanded} onClick={() => setExpanded((value) => !value)}>{expanded ? 'Exit expanded view' : 'Expand view'}</button></div>
    </div>
    <div className="attention-controls-display">
      <label className="attention-opacity">Overlay <input aria-label="Overlay opacity" type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} /><span>{Math.round(opacity * 100)}%</span></label>
      <label className="attention-top-select">Locations<select className="field" value={topCount} onChange={(event) => setTopCount(Number(event.target.value) as 10 | 20)}><option value="10">Top 10</option><option value="20">Top 20</option></select></label>
      <label className="attention-box-toggle"><input type="checkbox" checked={showRankedBoxes} onChange={(event) => setShowRankedBoxes(event.target.checked)} /> Numbered locations</label>
      <details className="attention-filter-options"><summary>Filter overlay</summary><label>Minimum attention percentile: {Math.round(minimum * 100)}<input type="range" min="0" max=".99" step=".01" value={minimum} onChange={(event) => setMinimum(Number(event.target.value))} /></label></details>
    </div>
    <ErrorNotice error={thumbnail.error ?? region.error ?? attention.error ?? top.error ?? downloadError} />
    {top.isError ? <button className="btn btn-secondary" disabled={top.isFetching} onClick={() => void top.refetch()}>Retry top patch ranking</button> : null}
    <div className={`attention-stage ranked-viewer-layout ${railOpen ? '' : 'rail-hidden'}`} ref={viewerRef}>
      <div className="attention-canvas interpretation-viewer">
        <svg ref={svg} viewBox={`${view.x} ${view.y} ${view.width} ${view.height}`} preserveAspectRatio="xMidYMid meet" tabIndex={0} role="group" aria-label={`Slide ${slide.slideId} with ABMIL attention overlay and ranked patch buttons. Drag to pan, plus and minus to zoom, arrows to pan, Home to fit patch coverage.`} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={() => { drag.current = null; }} onKeyDown={keyDown}>
          <title>{`${slide.slideId} — ABMIL pooling attention`}</title><rect x="0" y="0" width={slide.width} height={slide.height} fill="#eee9e2" />
          {thumbnailURL ? <image href={thumbnailURL} x="0" y="0" width={slide.width} height={slide.height} preserveAspectRatio="none" /> : null}
          {zoomed && regionURL ? <image href={regionURL} x={settledView.x} y={settledView.y} width={settledView.width} height={settledView.height} preserveAspectRatio="none" /> : null}
          {heatmap?.signature === signature && !attention.isError ? <image href={heatmap.url} x={settledView.x} y={settledView.y} width={settledView.width} height={settledView.height} preserveAspectRatio="none" opacity={opacity} /> : null}
          {verifiedSelectedPatch ? <g aria-hidden="true" pointerEvents="none"><rect x={verifiedSelectedPatch.x} y={verifiedSelectedPatch.y} width={slide.patchWidthLevel0} height={slide.patchHeightLevel0} fill="none" stroke="white" strokeWidth="6" vectorEffect="non-scaling-stroke" /><rect x={verifiedSelectedPatch.x} y={verifiedSelectedPatch.y} width={slide.patchWidthLevel0} height={slide.patchHeightLevel0} fill="none" stroke="#072b48" strokeWidth="3" vectorEffect="non-scaling-stroke" /></g> : null}
          {showRankedBoxes ? <RankedPatchMarkers patches={ranked.filter((patch) => patchIntersectsRegion(patch, slide.patchWidthLevel0, slide.patchHeightLevel0, view))} patchWidth={slide.patchWidthLevel0} patchHeight={slide.patchHeightLevel0} scale={rankMarkerScale} selectedIndex={verifiedSelectedPatch?.index} onSelect={inspectRankedPatch} /> : null}
        </svg>
        <div className="attention-canvas-tools" aria-label="Slide zoom controls"><button type="button" aria-label="Zoom in" onClick={() => zoom(2)}>+</button><button type="button" aria-label="Zoom out" onClick={() => zoom(.5)}>−</button><span>{zoomLabel}</span><button type="button" disabled={!verifiedSelectedPatch} onClick={() => { if (verifiedSelectedPatch) centerPatch(verifiedSelectedPatch); }}>Center selected</button></div>
        {thumbnail.isPending || region.isFetching || attention.isFetching ? <span className="attention-loading" role="status">{thumbnail.isPending ? 'Loading slide…' : 'Updating view…'}</span> : null}
        <span className="attention-pan-hint">Drag to pan · + / − to zoom</span>
      </div>
      {railOpen ? <aside className="attention-patch-rail" aria-label="Patch inspection and ranked locations">
        <SelectedAttentionPatch key={`${record.id}:${slide.slideId}:${member}`} compact context={cropContext} patch={verifiedSelectedPatch} rank={selectedRank} patchWidth={slide.patchWidthLevel0} patchHeight={slide.patchHeightLevel0} onShowLocation={() => { if (verifiedSelectedPatch) centerPatch(verifiedSelectedPatch); }} />
        {top.isPending ? <p className="attention-ranking-status" role="status">Finding the highest attention weights across the whole slide…</p> : null}
        {top.data && !top.isError ? <RankedAttentionPatches key={`${project}:${record.id}:${slide.slideId}:${member}`} compact context={cropContext} patches={ranked} total={top.data.total} selectedPatch={verifiedSelectedPatch} onSelect={inspectRankedPatch} /> : null}
      </aside> : null}
    </div>
    <div className="attention-view-footer"><span className="interpretation-attention-scale">Lower <i aria-hidden="true" /> Higher attention percentile</span><span>{attention.data ? `${attention.data.patches.length.toLocaleString()} / ${attention.data.total.toLocaleString()} patches in view` : 'Attention pending'}</span></div>
    {attention.data && attention.data.patches.length < attention.data.total ? <p className="callout" role="status">Partial overlay: this region exceeds the {MAX_VISIBLE_PATCHES.toLocaleString()}-patch display limit. Zoom in to see every patch. Top locations still rank the entire slide.</p> : null}
    <details className="attention-view-details"><summary>About this view and slide predictions</summary><p>ABMIL pooling attention is class-independent relative weighting within a slide. It is not tumor probability, a segmentation mask or evidence of causality. Overlapping patches show the highest weight; numbered locations rank the whole slide.</p><p>{slide.width.toLocaleString()} × {slide.height.toLocaleString()} level-0 pixels · {slide.patchCount.toLocaleString()} patches · patch footprint {slide.patchWidthLevel0} × {slide.patchHeightLevel0} level-0 pixels. Original crops keep tissue colors; edge patches are clipped to the slide.</p>{attention.data ? <div className="interpretation-member-probabilities" aria-label="Slide predictions">{attention.data.classOrder.map((name, index) => <span key={name}><strong>{name}</strong>: {formatStatistic(attention.data?.probabilities[index])}</span>)}</div> : null}<button className="btn btn-secondary" onClick={() => void download()}>{record.manifest.memberCount > 1 ? 'Download full ensemble mean attention' : 'Download full attention data'}</button></details>
  </section>;
}
