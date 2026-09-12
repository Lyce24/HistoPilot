import type { KeyboardEvent, PointerEvent } from 'react';
import type { RankedAttentionPatch } from '../api/interpretation';

export default function RankedPatchMarkers({ patches, patchWidth, patchHeight, scale, selectedIndex, onSelect }: { patches: RankedAttentionPatch[]; patchWidth: number; patchHeight: number; scale: number; selectedIndex?: number; onSelect: (patch: RankedAttentionPatch) => void }) {
  const stop = (event: PointerEvent<SVGGElement>) => event.stopPropagation();
  function keyDown(event: KeyboardEvent<SVGGElement>, patch: RankedAttentionPatch) {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); onSelect(patch); }
  }
  return <g aria-label="Ranked attention patch boxes">{[...patches].reverse().map((patch) => {
    const selected = patch.index === selectedIndex;
    const pinX = patch.x + Math.min(patchWidth / 2, 13 * scale), pinY = patch.y + Math.min(patchHeight / 2, 13 * scale);
    return <g key={patch.index} role="button" tabIndex={0} aria-label={`Inspect rank ${patch.rank}, patch ${patch.index}`} aria-pressed={selected} className={`ranked-patch-marker ${selected ? 'is-selected' : ''}`} onPointerDown={stop} onPointerMove={stop} onPointerUp={stop} onClick={(event) => { event.stopPropagation(); onSelect(patch); }} onKeyDown={(event) => keyDown(event, patch)}>
      <title>{`Rank ${patch.rank} · patch ${patch.index} · attention weight ${patch.weight.toPrecision(5)}`}</title>
      <circle className="ranked-patch-hit-area" cx={pinX} cy={pinY} r={22 * scale} fill="transparent" />
      <rect x={patch.x} y={patch.y} width={patchWidth} height={patchHeight} fill={selected ? '#06b6d422' : 'transparent'} stroke="white" strokeWidth="6" vectorEffect="non-scaling-stroke" />
      <rect x={patch.x} y={patch.y} width={patchWidth} height={patchHeight} fill="none" stroke={selected ? '#072b48' : '#087f8c'} strokeWidth={selected ? '4' : '2.5'} vectorEffect="non-scaling-stroke" />
      <circle cx={pinX} cy={pinY} r={12 * scale} fill={selected ? '#072b48' : '#087f8c'} stroke="white" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      <text x={pinX} y={pinY} dy=".35em" textAnchor="middle" fontSize={12 * scale} fontWeight="700" fill="white" aria-hidden="true">{patch.rank}</text>
    </g>;
  })}</g>;
}
