import { useQuery } from '@tanstack/react-query';
import { interpretations, type AttentionPatch, type RankedAttentionPatch } from '../api/interpretation';
import { useVisibleElement } from './useVisibleElement';
import { useImageBlob } from './SlideGalleryCard';
import { formatStatistic } from '../lib/evidenceCharts';
import { ErrorNotice } from './ui';

export interface PatchImageContext { project: string; interpretationId: string; slideId: string; member: string }
function usePatchImage(context: PatchImageContext, patchIndex: number | undefined, enabled: boolean) {
  return useQuery({ queryKey: ['interpretation-patch-crop', context.project, context.interpretationId, context.slideId, context.member, patchIndex], queryFn: ({ signal }) => interpretations.patchImage(context.project, context.interpretationId, context.slideId, context.member, patchIndex!, signal), enabled: enabled && patchIndex !== undefined, staleTime: Infinity, gcTime: 60000 });
}
function RankedPatchCard({ context, patch, selected, onSelect }: { context: PatchImageContext; patch: RankedAttentionPatch; selected: boolean; onSelect: () => void }) {
  const { ref, visible } = useVisibleElement<HTMLElement>();
  const crop = usePatchImage(context, patch.index, visible);
  const url = useImageBlob(crop.data);
  return <article ref={ref} className={`ranked-patch-card ${selected ? 'is-selected' : ''}`}>
    <button type="button" aria-label={`View rank ${patch.rank} patch`} aria-pressed={selected} onClick={onSelect}>
      <div className="ranked-patch-image">{url && !crop.isError ? <img src={url} alt={`Original slide crop for attention rank ${patch.rank}, patch ${patch.index}`} loading="lazy" /> : <span>{crop.isError ? 'Crop unavailable' : 'Loading crop…'}</span>}<span className="ranked-patch-badge" aria-hidden="true">{patch.rank}</span></div>
      <div className="ranked-patch-card-label"><strong>Rank {patch.rank}</strong><small>Patch {patch.index} · {patch.weight.toPrecision(3)}</small></div>
    </button>
    {crop.isError ? <p className="ranked-patch-error" role="status">{crop.error.message}</p> : null}
  </article>;
}
export function SelectedAttentionPatch({ context, patch, rank, patchWidth, patchHeight, onShowLocation, compact = false }: {
  context: PatchImageContext; patch: AttentionPatch | null; rank?: number; patchWidth: number; patchHeight: number; onShowLocation: () => void; compact?: boolean;
}) {
  const crop = usePatchImage(context, patch?.index, Boolean(patch));
  const url = useImageBlob(crop.data);
  const coordinates = patch ? <dl><div><dt>Top-left position</dt><dd>({patch.x}, {patch.y}) level-0 pixels</dd></div><div><dt>Patch footprint</dt><dd>{patchWidth} × {patchHeight} level-0 pixels</dd></div></dl> : null;
  return <section className={`ranked-patch-inspector ${compact ? 'is-compact' : ''}`} aria-label="Selected patch crop">
    <h4>{patch ? `${rank ? `Rank ${rank} · ` : ''}Patch ${patch.index}` : 'Original slide patch'}</h4>
    {patch ? <div className="ranked-patch-detail-image">{url && !crop.isError ? <img src={url} alt={`Original uncolored crop of ${context.slideId}, patch ${patch.index}${rank ? `, attention rank ${rank}` : ''}`} /> : <span>{crop.isError ? 'Original crop unavailable' : 'Loading selected patch…'}</span>}</div> : null}
    {patch ? <><p className="ranked-patch-measure"><strong>{patch.weight.toPrecision(4)}</strong> pooling weight <span>Percentile {formatStatistic(patch.percentile * 100, 1)}</span></p>{compact ? <details className="ranked-patch-coordinates"><summary>Coordinates and footprint</summary>{coordinates}</details> : <>{coordinates}<p className="muted">Original slide image without heatmap colors. Edge patches are clipped to the available slide image.</p></>}<button className="btn btn-secondary" onClick={onShowLocation}>Center selected patch</button></> : <p className="muted">Select a numbered location or a patch below to inspect its original tissue.</p>}
    <ErrorNotice error={patch ? crop.error : null} />{patch && crop.isError ? <button className="btn btn-secondary" disabled={crop.isFetching} onClick={() => void crop.refetch()}>Retry selected patch crop</button> : null}
  </section>;
}
export default function RankedAttentionPatches({ context, patches, total, selectedPatch, onSelect, compact = false }: {
  context: PatchImageContext; patches: RankedAttentionPatch[]; total: number; selectedPatch: AttentionPatch | null; onSelect: (patch: RankedAttentionPatch) => void; compact?: boolean;
}) {
  return <section className={`ranked-patches ${compact ? 'is-compact' : ''}`} aria-label="Highest-attention patches">
    <div className="ranked-patches-heading"><h3>Highest-attention patches</h3><p className="muted">{patches.length} highest-weight patches from all {total.toLocaleString()} slide patches.</p>{compact ? <details><summary>How patches are ranked</summary><p>Highest normalized weight across the whole slide. Equal weights are ordered by patch index; equal-weight ranks do not imply different importance. Numbers match slide locations. Crops preserve original tissue colors.</p></details> : <p className="muted">Highest normalized attention across the whole slide. Equal weights are ordered by patch index; equal-weight ranks do not imply different importance. Numbers match the boxes. Crops show original tissue colors.</p>}</div>
    <div className="ranked-patch-grid">{patches.map((patch) => <RankedPatchCard key={`${context.member}:${patch.index}`} context={context} patch={patch} selected={selectedPatch?.index === patch.index} onSelect={() => onSelect(patch)} />)}</div>
  </section>;
}
