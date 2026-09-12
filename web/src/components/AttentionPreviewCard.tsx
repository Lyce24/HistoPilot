import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { interpretations, type Interpretation, type VisualizeItem } from '../api/interpretation';
import { computeActive, computeStatusLabel } from '../api/predictors';
import { attentionColor } from '../lib/slideGeometry';
import { useVisibleElement } from './useVisibleElement';
import { useImageBlob } from './SlideGalleryCard';

export default function AttentionPreviewCard({ project, item, initial, disabled, onOpen, onRetry }: { project: string; item: VisualizeItem; initial?: Interpretation; disabled: boolean; onOpen: (record: Interpretation) => void; onRetry?: () => void }) {
  const { ref, visible } = useVisibleElement<HTMLElement>();
  const record = useQuery({ queryKey: ['interpretation', project, item.interpretationId], queryFn: () => interpretations.get(project, item.interpretationId!), initialData: initial ?? item.interpretation, enabled: visible && Boolean(item.interpretationId), staleTime: 15000 });
  const execution = useQuery({ queryKey: ['compute-job', project, 'interpretation', item.interpretationId], queryFn: () => interpretations.execution(project, item.interpretationId!), initialData: record.data?.execution, enabled: visible && Boolean(item.interpretationId), refetchInterval: (query) => computeActive(query.state.data) || !query.state.data ? 2500 : false });
  const slide = record.data?.manifest.slides.find((entry) => entry.slidePath === item.slidePath) ?? record.data?.manifest.slides[0];
  const completed = execution.data?.status === 'completed' && !execution.isError;
  const thumbnail = useQuery({ queryKey: ['interpretation-thumbnail', project, item.interpretationId, slide?.slideId], queryFn: ({ signal }) => interpretations.thumbnail(project, item.interpretationId!, slide!.slideId, signal), enabled: visible && completed && Boolean(slide), staleTime: Infinity });
  const map = useQuery({ queryKey: ['interpretation-preview-attention', project, item.interpretationId, slide?.slideId], queryFn: ({ signal }) => interpretations.attention(project, item.interpretationId!, slide!.slideId, 'mean', { x: 0, y: 0, width: slide!.width, height: slide!.height }, 0, 5000, signal), enabled: visible && completed && Boolean(slide), staleTime: Infinity, gcTime: 60000 });
  const image = useImageBlob(thumbnail.data);
  const [overlay, setOverlay] = useState<{ identity: string; url: string } | null>(null);
  const identity = `${item.interpretationId}:${slide?.slideId}`;
  useEffect(() => {
    if (!map.data || !slide) return;
    const canvas = document.createElement('canvas'); const scale = Math.min(480 / slide.width, 320 / slide.height);
    canvas.width = Math.max(1, Math.ceil(slide.width * scale)); canvas.height = Math.max(1, Math.ceil(slide.height * scale));
    const context = canvas.getContext('2d'); if (!context) return;
    const sx = canvas.width / slide.width, sy = canvas.height / slide.height;
    for (const patch of [...map.data.patches].sort((a, b) => a.weight - b.weight)) { context.fillStyle = attentionColor(patch.percentile); context.fillRect(patch.x * sx, patch.y * sy, map.data.patchWidthLevel0 * sx, map.data.patchHeightLevel0 * sy); }
    let alive = true, url: string | undefined;
    canvas.toBlob((blob) => { if (!alive || !blob) return; url = URL.createObjectURL(blob); setOverlay({ identity, url }); });
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [map.data, slide, identity]);
  const status = execution.isError ? 'Status unavailable' : execution.data ? computeStatusLabel(execution.data) : item.error ? 'Could not start' : item.status.replaceAll('_', ' ');
  const recovered = completed || computeActive(execution.data);
  const failed = !recovered && (Boolean(item.error) || ['failed', 'cancelled', 'interrupted', 'not_started'].includes(execution.data?.status ?? ''));
  const error = (!recovered ? item.error?.message : null) ?? execution.data?.error ?? execution.error?.message ?? record.error?.message ?? map.error?.message ?? thumbnail.error?.message;
  return <article className="slide-gallery-card attention-preview-card" ref={ref}>
    <button className="slide-gallery-image" disabled={!record.data || disabled} onClick={() => { if (record.data) onOpen({ ...record.data, execution: execution.data ?? record.data.execution }); }} aria-label={`Open attention viewer for ${item.slideId}`}>
      {image && slide ? <svg viewBox={`0 0 ${slide.width} ${slide.height}`} role="img" aria-label={`${item.slideId} attention overview`}><image href={image} width={slide.width} height={slide.height} preserveAspectRatio="none" />{overlay?.identity === identity && !map.isError ? <image href={overlay.url} width={slide.width} height={slide.height} preserveAspectRatio="none" opacity=".6" /> : null}</svg> : <span className="slide-gallery-placeholder">{completed ? 'Loading attention preview…' : status}</span>}
      <span className="slide-gallery-status">{status}</span>
    </button>
    <div className="slide-gallery-card-body"><strong>{item.slideId}</strong>{item.reused ? <small>Reused matching attention study</small> : null}
      {execution.data?.progress?.completedPairs !== undefined ? <small>{execution.data.progress.completedPairs} / {execution.data.progress.totalPairs} checkpoint passes</small> : null}
      {error ? <p className="slide-gallery-error" role="status">{error}</p> : null}
      {map.isError || thumbnail.isError ? <small>Attention preview unavailable. Open the viewer to inspect the error or retry loading.</small> : map.data && map.data.patches.length < map.data.total ? <small>Partial preview: {map.data.patches.length.toLocaleString()} / {map.data.total.toLocaleString()} patches. Open the viewer and zoom for full regional detail.</small> : completed && map.data ? <small>Actual ABMIL attention · {record.data?.manifest.method === 'ensemble' ? 'ensemble mean' : 'refit model'}</small> : null}
      <div className="inline-actions">{record.data ? <button className="text-button" disabled={disabled} onClick={() => onOpen({ ...record.data!, execution: execution.data ?? record.data!.execution })}>Open viewer</button> : null}{failed && onRetry ? <button className="text-button" disabled={disabled} onClick={onRetry}>Retry with current resources</button> : null}</div>
    </div>
  </article>;
}
