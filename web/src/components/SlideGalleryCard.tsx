import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { interpretations, type GallerySlide } from '../api/interpretation';
import { computeActive, computeStatusLabel } from '../api/predictors';
import { useVisibleElement } from './useVisibleElement';
export function useImageBlob(blob?: Blob) {
  const [image, setImage] = useState<{ blob: Blob; url: string } | null>(null);
  useEffect(() => { if (!blob) return; const url = URL.createObjectURL(blob); setImage({ blob, url }); return () => URL.revokeObjectURL(url); }, [blob]);
  return image?.blob === blob ? image?.url : undefined;
}
export default function SlideGalleryCard({ project, slide, checked, disabled, selectionDisabled = false, status, interpretationId, onToggle }: {
  project: string; slide: GallerySlide; checked: boolean; disabled: boolean; selectionDisabled?: boolean; status?: string; interpretationId?: string | null; onToggle: (checked: boolean) => void;
}) {
  const { ref, visible } = useVisibleElement<HTMLElement>();
  const thumbnail = useQuery({ queryKey: ['interpretation-gallery-thumbnail', project, slide.slidePath], queryFn: ({ signal }) => interpretations.galleryThumbnail(project, slide.slidePath, signal), enabled: visible, staleTime: 60000, gcTime: 60000 });
  const execution = useQuery({ queryKey: ['compute-job', project, 'interpretation', interpretationId], queryFn: () => interpretations.execution(project, interpretationId!), enabled: visible && Boolean(interpretationId), refetchInterval: (query) => computeActive(query.state.data) ? 2500 : false });
  const currentStatus = execution.isError ? 'Status unavailable' : execution.data ? computeStatusLabel(execution.data) : status?.replaceAll('_', ' ');
  const url = useImageBlob(thumbnail.data);
  return <article ref={ref} className={`slide-gallery-card ${checked ? 'is-selected' : ''}`}>
    <button className="slide-gallery-image" type="button" onClick={() => onToggle(!checked)} disabled={disabled || selectionDisabled || !slide.available} aria-label={`${checked ? 'Deselect' : 'Select'} ${slide.name}`} aria-pressed={checked}>
      {url ? <img src={url} alt={`Overview of ${slide.name}`} loading="lazy" /> : <span className="slide-gallery-placeholder">{thumbnail.isError ? 'Preview unavailable' : visible ? 'Loading preview…' : 'Slide preview'}</span>}
      {currentStatus ? <span className="slide-gallery-status">{currentStatus}</span> : null}
    </button>
    <div className="slide-gallery-card-body"><label className="slide-gallery-selection"><input type="checkbox" checked={checked} disabled={disabled || selectionDisabled || !slide.available} onChange={(event) => onToggle(event.target.checked)} aria-label={`Select ${slide.name} for batch visualization`} /><strong title={slide.relativePath}>{slide.name}</strong></label>
      <small className={slide.available ? 'science-success' : 'muted'}>{slide.available ? `Features matched${slide.patchCount != null ? ` · ${slide.patchCount.toLocaleString()} patches` : ''}` : slide.reason || 'Matching features unavailable'}</small>
      <button type="button" className="text-button" onClick={() => onToggle(!checked)} disabled={disabled || selectionDisabled || !slide.available}>{slide.available ? checked ? 'Selected' : 'Select slide' : 'Unavailable for this source'}</button>
    </div>
  </article>;
}
