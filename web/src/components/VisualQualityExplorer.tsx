import { useDeferredValue, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { bundles } from '../api/bundles';
import { scientific, type FeatureSpec } from '../api/scientific';
import { morphology, type MorphologyIndex, type MorphologyPoint, type MorphologyRegion, type QualityEvidence } from '../api/morphology';
import { versionLabelText } from '../lib/versionLabels';
import { reviewStatusLabels } from '../api/slideReviews';
import SlideReviewEditor from './SlideReviewEditor';
import { useImageBlob } from './SlideGalleryCard';
import { ErrorNotice, Panel } from './ui';
import './VisualQualityExplorer.css';

const palette = ['#205e63', '#ac4d22', '#715298', '#387636', '#9c3e67', '#3866a3', '#74672d', '#595959'];

export function projectionGroups(points: MorphologyPoint[], attribute: string) {
  const value = (point: MorphologyPoint) => String(attribute ? point.attributes[attribute] ?? 'Missing' : 'All slides');
  const groups = [...new Set(points.map(value))].sort();
  return { groups, value, color: (point: MorphologyPoint) => palette[groups.indexOf(value(point)) % palette.length] };
}

export function MorphologyPlot({ index, selected, onSelect }: { index: MorphologyIndex; selected: string; onSelect: (slide: string) => void }) {
  const [attribute, setAttribute] = useState('');
  const attributes = [...new Set(index.points.flatMap((point) => Object.keys(point.attributes)))].sort();
  const grouped = projectionGroups(index.points, attribute);
  const xs = index.points.map((point) => point.x), ys = index.points.map((point) => point.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const position = (point: MorphologyPoint) => [40 + (point.x - minX) / (maxX - minX || 1) * 620, 330 - (point.y - minY) / (maxY - minY || 1) * 290];
  return <div className="morphology-projection">
    <label className="label">Color by metadata<select className="field" value={attribute} onChange={(event) => setAttribute(event.target.value)}><option value="">All slides</option>{attributes.map((name) => <option key={name}>{name}</option>)}</select></label>
    <svg viewBox="0 0 700 380" role="group" aria-label="Slide morphology PCA projection">
      <path d="M35 25V340H680" fill="none" stroke="currentColor" opacity=".35" />
      <text x="350" y="374" textAnchor="middle">PC1 · {(index.explainedVariance[0] * 100).toFixed(1)}% variance</text>
      <text x="10" y="190" transform="rotate(-90 10 190)" textAnchor="middle">PC2 · {(index.explainedVariance[1] * 100).toFixed(1)}%</text>
      {index.points.map((point) => { const [x, y] = position(point); return <g key={point.slideId} role="button" tabIndex={0} aria-label={`Inspect ${point.slideId}, ${grouped.value(point)}`} aria-pressed={selected === point.slideId} onClick={() => onSelect(point.slideId)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(point.slideId); } }}>
        <circle cx={x} cy={y} r={selected === point.slideId ? 9 : 6} fill={grouped.color(point)} stroke={selected === point.slideId ? '#111' : 'white'} strokeWidth={selected === point.slideId ? 3 : 1} /><title>{`${point.slideId} · ${grouped.value(point)} · ${point.sampledPatches}/${point.patchCount} sampled patches`}</title>
      </g>; })}
    </svg>
    <div className="morphology-legend">{grouped.groups.slice(0, 24).map((group, index) => <span key={group}><i style={{ background: palette[index % palette.length] }} />{group}</span>)}{grouped.groups.length > 24 ? <span>+ {grouped.groups.length - 24} values</span> : null}</div>
    <p className="muted">{index.method}. Nearby points in two dimensions can differ in the full feature space. Colors show recorded metadata.</p>
  </div>;
}

function Contours({ evidence }: { evidence: QualityEvidence }) {
  return <>{evidence.tissueContours.map((polygon, index) => <path key={index} d={polygon.map((ring) => `M${ring.map((xy) => `${xy[0]},${xy[1]}`).join('L')}Z`).join(' ')} fill="#27ae6040" stroke="#1e7543" strokeWidth={Math.max(evidence.width, evidence.height) / 900} fillRule="evenodd" />)}</>;
}

export function QualitySlide({ project, datasetId, slideId, featureBundleId, patchIndex: controlledPatch, onPatch, showReview = true, onRegion }: { project: string; datasetId: string; slideId: string; featureBundleId?: string; patchIndex?: number; onPatch?: (index: number) => void; showReview?: boolean; onRegion?: (region: MorphologyRegion) => void }) {
  const [coverage, setCoverage] = useState(true);
  const [contours, setContours] = useState(true);
  const [localPatch, setLocalPatch] = useState<number | undefined>();
  const [patchInput, setPatchInput] = useState('');
  const [viewport, setViewport] = useState<MorphologyRegion>();
  const [drawnRegion, setDrawnRegion] = useState<MorphologyRegion>();
  const [drawing, setDrawing] = useState(false);
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const ignorePatchClick = useRef(false);
  const patchIndex = controlledPatch ?? localPatch;
  function selectPatch(index: number) { setLocalPatch(index); setDrawnRegion(undefined); onPatch?.(index); }
  // Geometry must not depend on optional features; thumbnail pixels are not level-0 coordinates.
  const geometry = useQuery({ queryKey: ['morphology-quality', project, datasetId, slideId, undefined], queryFn: ({ signal }) => morphology.quality(project, datasetId, slideId, undefined, signal), staleTime: 30000 });
  const evidence = useQuery({ queryKey: ['morphology-quality', project, datasetId, slideId, featureBundleId], queryFn: ({ signal }) => morphology.quality(project, datasetId, slideId, featureBundleId, signal), enabled: Boolean(featureBundleId), staleTime: 30000 });
  const thumbnail = useQuery({ queryKey: ['morphology-image', project, datasetId, slideId, viewport], queryFn: ({ signal }) => morphology.image(project, datasetId, slideId, signal, viewport), staleTime: 30000, gcTime: 60000 });
  const patchRegion = useQuery({ queryKey: ['morphology-patch-region', project, datasetId, slideId, featureBundleId, patchIndex], queryFn: ({ signal }) => morphology.patchRegion(project, datasetId, slideId, featureBundleId!, patchIndex!, signal), enabled: patchIndex != null && Boolean(featureBundleId), staleTime: 30000 });
  const patch = useQuery({ queryKey: ['morphology-patch', project, datasetId, slideId, featureBundleId, patchIndex], queryFn: ({ signal }) => morphology.patch(project, datasetId, slideId, featureBundleId!, patchIndex!, signal), enabled: patchIndex != null && Boolean(featureBundleId), staleTime: 30000, gcTime: 60000 });
  const url = useImageBlob(thumbnail.data), patchUrl = useImageBlob(patch.data);
  const quality = (!evidence.isError && evidence.data) || (!geometry.isError && geometry.data) || undefined;
  const selectedRegion = drawnRegion ?? (patchRegion.isError ? undefined : patchRegion.data);
  const field = viewport ?? (quality ? { x: 0, y: 0, width: quality.width, height: quality.height } : undefined);
  function point(event: React.PointerEvent<SVGSVGElement>) {
    const svg = event.currentTarget;
    const matrix = svg.getScreenCTM();
    if (!matrix || !field) return null;
    const value = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return { x: Math.max(field.x, Math.min(field.x + field.width, value.x)), y: Math.max(field.y, Math.min(field.y + field.height, value.y)) };
  }
  const thickness = quality ? Math.max(quality.width, quality.height) / 700 : 1;
  return <section className="morphology-slide" aria-label={`Visual quality of ${slideId}`}>
    <h3>{slideId}</h3>
    <ErrorNotice error={geometry.error ?? thumbnail.error} />
    {featureBundleId && evidence.isError ? <div className="callout" role="status">Optional feature coverage is unavailable. Image review and notes remain available.<ErrorNotice error={evidence.error} /></div> : null}
    {geometry.isPending || thumbnail.isPending ? <p role="status">Reading the exact linked slide geometry…</p> : null}
    {quality ? <>
      <div className="inline-actions">{quality.patchCount != null ? <label className="checkbox-label"><input type="checkbox" checked={coverage} onChange={(event) => setCoverage(event.target.checked)} />Patch coverage</label> : null}{quality.tissueContours.length ? <label className="checkbox-label"><input type="checkbox" checked={contours} onChange={(event) => setContours(event.target.checked)} />Recorded tissue contours</label> : null}<button className="btn btn-secondary" aria-pressed={drawing} onClick={() => setDrawing(!drawing)}>{drawing ? 'Cancel region selection' : 'Draw review region'}</button><button className="btn btn-secondary" disabled={!selectedRegion} onClick={() => setViewport(selectedRegion)}>Zoom to selection</button><button className="btn btn-secondary" disabled={!viewport} onClick={() => setViewport(undefined)}>Fit slide</button></div>
      {drawing ? <p role="status">Drag a rectangle on the image. Its level-0 coordinates can be saved with the review.</p> : null}
      {url && !thumbnail.isError && field ? <div className="morphology-image-scroll"><svg style={{ width: '100%', touchAction: drawing ? 'none' : undefined }} viewBox={`${field.x} ${field.y} ${field.width} ${field.height}`} role="group" aria-label={`Exact slide ${slideId}${quality.patchCount != null ? ' with patch coverage' : ''}`}
        onPointerDown={(event) => { ignorePatchClick.current = drawing; if (!drawing) return; dragStart.current = point(event); event.currentTarget.setPointerCapture(event.pointerId); }}
        onPointerUp={(event) => { if (!drawing || !dragStart.current) return; const end = point(event), start = dragStart.current; dragStart.current = null; if (!end) return; const region = { x: Math.floor(Math.min(start.x, end.x)), y: Math.floor(Math.min(start.y, end.y)), width: Math.floor(Math.abs(end.x - start.x)), height: Math.floor(Math.abs(end.y - start.y)) }; if (region.width >= 1 && region.height >= 1) { setDrawnRegion(region); setDrawing(false); onRegion?.(region); } }}
        onPointerCancel={() => { dragStart.current = null; ignorePatchClick.current = false; }}>
        <image href={url} x={field.x} y={field.y} width={field.width} height={field.height} />
        {contours ? <Contours evidence={quality} /> : null}
        {coverage ? quality.patches.map((row) => quality.patchWidth && quality.patchHeight ? <rect key={row.patchIndex} x={row.x} y={row.y} width={Math.min(quality.patchWidth, quality.width - row.x)} height={Math.min(quality.patchHeight, quality.height - row.y)} fill={row.patchIndex === patchIndex ? '#f2a90080' : '#285c7624'} stroke={row.patchIndex === patchIndex ? '#a44900' : '#285c76'} strokeWidth={thickness} onClick={() => { if (!drawing && !ignorePatchClick.current) selectPatch(row.patchIndex); }}><title>{`Patch ${row.patchIndex} at ${row.x}, ${row.y}`}</title></rect> : <circle key={row.patchIndex} cx={row.x} cy={row.y} r={thickness * 1.5} fill="#285c76"><title>{`Patch origin ${row.patchIndex}; footprint unknown`}</title></circle>) : null}
        {selectedRegion ? <rect {...selectedRegion} fill="none" stroke="#a44900" strokeWidth={thickness * 3} pointerEvents="none" /> : null}
      </svg></div> : null}
      <p className="muted">{quality.width.toLocaleString()} × {quality.height.toLocaleString()} level-0 pixels · {quality.patchCount == null ? quality.featureKind === 'slide' ? 'One slide embedding; patch coverage not applicable' : 'No patch coverage available' : `${quality.patchCount.toLocaleString()} extracted patches`}{quality.coverageSampled ? ` · Overlay shows ${quality.patches.length.toLocaleString()} evenly spaced patches; coverage bounds use every patch` : ''}. {quality.artifactRemoval === true ? 'Recorded extraction enabled GrandQC artifact removal.' : quality.artifactRemoval === false ? 'Recorded extraction did not enable artifact removal.' : 'Artifact-removal evidence unavailable.'}</p>
      {quality.patchCount && quality.patchWidth ? <form className="inline-actions" onSubmit={(event) => { event.preventDefault(); const value = Number(patchInput); if (/^\d+$/.test(patchInput) && Number.isSafeInteger(value) && value < quality.patchCount!) selectPatch(value); }}><label className="label">Inspect exact patch index<input className="field" type="number" min={0} max={quality.patchCount - 1} step={1} value={patchInput} onChange={(event) => setPatchInput(event.target.value)} placeholder={`0–${quality.patchCount - 1}`} /></label><button className="btn btn-secondary" disabled={!/^\d+$/.test(patchInput) || Number(patchInput) >= quality.patchCount}>Open patch</button></form> : null}
      {quality.coordinateBounds ? <p className="muted">Patch coverage bounds: ({quality.coordinateBounds.x}, {quality.coordinateBounds.y}), {Math.round(quality.coordinateBounds.width)} × {Math.round(quality.coordinateBounds.height)} pixels.</p> : null}
      {quality.warnings.map((warning) => <p className="muted" key={warning}>{warning}</p>)}
    </> : null}
    {patchIndex != null ? <div className="morphology-patch"><h4>Original patch {patchIndex}</h4><ErrorNotice error={patch.error} />{patch.isPending ? <p role="status">Reading patch crop…</p> : null}{patchUrl && !patch.isError ? <img src={patchUrl} alt={`Exact original-color patch ${patchIndex} in ${slideId}`} /> : null}</div> : null}
    {selectedRegion ? <p>Selected region: ({selectedRegion.x}, {selectedRegion.y}), {Math.round(selectedRegion.width)} × {Math.round(selectedRegion.height)} level-0 pixels.</p> : null}
    {showReview ? <SlideReviewEditor key={`${datasetId}:${slideId}`} project={project} datasetId={datasetId} slideId={slideId} selectedRegion={selectedRegion} /> : selectedRegion && onRegion ? <button className="btn btn-secondary" onClick={() => onRegion(selectedRegion)}>Use selected region for review</button> : null}
  </section>;
}

function FeatureNeighbors({ project, index, slideId, onSelect }: { project: string; index: MorphologyIndex; slideId: string; onSelect: (slide: string, patch?: number) => void }) {
  const [mode, setMode] = useState<'slide' | 'patch'>('slide');
  const [selectedPatch, setSelectedPatch] = useState<number | undefined>();
  const [otherSlides, setOtherSlides] = useState(true);
  const patchRows = index.patches.filter((patch) => patch.slideId === slideId);
  const patchIndex = patchRows.some((row) => row.patchIndex === selectedPatch) ? selectedPatch : patchRows[0]?.patchIndex;
  const query = useQuery({ queryKey: ['morphology-neighbors', project, index.indexId, slideId, mode, patchIndex, otherSlides], queryFn: ({ signal }) => morphology.neighbors(project, { indexId: index.indexId, slideId, mode, ...(mode === 'patch' ? { patchIndex } : {}), otherSlidesOnly: otherSlides }, signal), enabled: index.points.some((point) => point.slideId === slideId) });
  return <section className="morphology-neighbors" aria-label="Similar morphology"><h3>Nearest neighbors</h3>
    <div className="inline-actions"><label className="label">Compare<select className="field" value={mode} onChange={(event) => setMode(event.target.value as 'slide' | 'patch')}><option value="slide">Pooled slides</option><option value="patch">Sampled patches</option></select></label>{mode === 'patch' ? <><label className="label">Query patch<select className="field" value={patchIndex ?? ''} onChange={(event) => { const value = Number(event.target.value); setSelectedPatch(value); onSelect(slideId, value); }}>{patchRows.map((row) => <option key={row.patchIndex} value={row.patchIndex}>Patch {row.patchIndex} · ({row.x}, {row.y})</option>)}</select></label><button className="btn btn-secondary" onClick={() => onSelect(slideId, patchIndex)}>View query patch</button><label className="checkbox-label"><input type="checkbox" checked={otherSlides} onChange={(event) => setOtherSlides(event.target.checked)} />Other slides only</label></> : null}</div>
    <p className="muted">Cosine similarity in the original encoder space. {mode === 'patch' ? 'Search covers only sampled patches; it does not claim nearest neighbors across every patch.' : 'Slide vectors average the indexed patch sample.'}</p>
    <ErrorNotice error={query.error} />{query.isPending ? <p role="status">Finding neighbors…</p> : null}
    {query.data && !query.isError ? <><p className="muted">{query.data.candidateCount.toLocaleString()} comparable {mode === 'patch' ? 'sampled patches' : 'slides'}.</p>{query.data.items.length ? <div className="table-wrap"><table><thead><tr><th>Slide</th>{mode === 'patch' ? <th>Patch</th> : null}<th>Similarity</th><th>Action</th></tr></thead><tbody>{query.data.items.map((item) => <tr key={`${item.slideId}:${item.patchIndex ?? ''}`}><th scope="row">{item.slideId}</th>{mode === 'patch' ? <td>{item.patchIndex}</td> : null}<td>{item.similarity.toFixed(4)}</td><td><button className="text-button" onClick={() => onSelect(item.slideId, item.patchIndex)}>Inspect {item.patchIndex == null ? 'slide' : 'patch'}</button></td></tr>)}</tbody></table></div> : <p>No comparable neighbors in this index.</p>}</> : null}
  </section>;
}

export default function VisualQualityExplorer({ project, datasetId, initialBundleId = '' }: { project: string; datasetId: string; initialBundleId?: string }) {
  const [opened, setOpened] = useState(false);
  return <Panel title="Visual QC & morphology" subtitle="Review exact slides, inspect extracted coverage, and explore similarities in verified features.">
    {!opened ? <button className="btn btn-secondary" onClick={() => setOpened(true)}>Open visual review &amp; feature explorer</button> : <ExplorerContents key={`${project}:${datasetId}:${initialBundleId}`} project={project} datasetId={datasetId} initialBundleId={initialBundleId} />}
  </Panel>;
}

function ExplorerContents({ project, datasetId, initialBundleId }: { project: string; datasetId: string; initialBundleId: string }) {
  const [bundleId, setBundleId] = useState(initialBundleId);
  const [search, setSearch] = useState('');
  const deferredSearch = useDeferredValue(search);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState('');
  const [patchIndex, setPatchIndex] = useState<number | undefined>();
  const [maxSlides, setMaxSlides] = useState(128);
  const [patchesPerSlide, setPatchesPerSlide] = useState(32);
  const [index, setIndex] = useState<MorphologyIndex | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const sources = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const featureId = sources.data?.items.find((item) => item.id === bundleId)?.manifest.feature.id;
  const feature = useQuery({ queryKey: ['scientific', project, 'configuration', featureId], queryFn: () => scientific.configuration(project, featureId!), enabled: Boolean(featureId), staleTime: 30000 });
  const slideFeatures = (feature.data?.manifest.spec as FeatureSpec | undefined)?.featureKind === 'slide';
  const canExplorePatches = Boolean(bundleId && feature.data && !feature.isError && !slideFeatures);
  const rows = useQuery({ queryKey: ['morphology-slides', project, datasetId, deferredSearch, offset], queryFn: ({ signal }) => morphology.slides(project, datasetId, deferredSearch, offset, signal) });
  const selectedSlide = selected || rows.data?.items[0]?.slideId || '';
  function select(slide: string, patch?: number) { setSelected(slide); setPatchIndex(patch); }
  async function build() {
    if (!canExplorePatches || busy) return;
    setBusy(true); setError(null); setIndex(null);
    try { const value = await morphology.build(project, { datasetId, featureBundleId: bundleId, maxSlides, patchesPerSlide }); setIndex(value); if (!value.points.some((point) => point.slideId === selectedSlide)) select(value.points[0].slideId); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Feature exploration failed.')); }
    finally { setBusy(false); }
  }
  return <div className="visual-quality-explorer">
    <p>Quality decisions and reviewer notes are saved separately from frozen datasets. Excluding a slide here records a review decision; it does not change training or evaluation membership.</p>
    <div className="morphology-settings"><label className="label">Feature bundle (optional for slide review)<select className="field" value={bundleId} disabled={busy} onChange={(event) => { setBundleId(event.target.value); setIndex(null); setPatchIndex(undefined); }}><option value="">Slide review only</option>{sources.data?.items.filter((bundle) => bundle.manifest.datasetId == null || bundle.manifest.datasetId === datasetId).map((bundle) => <option key={bundle.id} value={bundle.id} disabled={!bundle.current}>{versionLabelText(bundle, 'Feature bundle')}{!bundle.current ? ' · needs verification' : ''}</option>)}</select></label>
      {!slideFeatures ? <><label className="label">Slides in projection<select className="field" value={maxSlides} disabled={busy} onChange={(event) => { setMaxSlides(Number(event.target.value)); setIndex(null); }}>{[16, 32, 64, 128, 256].map((value) => <option key={value} value={value}>Up to {value}</option>)}</select></label>
      <label className="label">Patches sampled per slide<select className="field" value={patchesPerSlide} disabled={busy} onChange={(event) => { setPatchesPerSlide(Number(event.target.value)); setIndex(null); }}>{[8, 16, 32, 64, 128].map((value) => <option key={value} value={value}>Up to {value}</option>)}</select></label>
      <button className="btn btn-primary" disabled={!canExplorePatches || busy} onClick={() => void build()}>{busy ? 'Reading feature sample…' : 'Build morphology projection'}</button></> : <p>One embedding per slide. Image review is available; patch coverage and patch-based morphology exploration are unavailable for this source.</p>}
    </div>
    <ErrorNotice error={error ?? sources.error ?? feature.error ?? rows.error} />
    {index ? <><p>{index.indexedSlides} / {index.candidateSlides} matching slides · {index.indexedPatches.toLocaleString()} sampled patches · {index.encoderId || 'Encoder label unavailable'}</p><MorphologyPlot key={index.indexId} index={index} selected={selectedSlide} onSelect={(slide) => select(slide)} /><details><summary>Sampling and interpretation limits</summary><p>{index.sampling}</p>{index.warnings.map((warning) => <p key={warning}>{warning}</p>)}</details></> : null}
    <div className="morphology-workspace"><aside><label className="label">Find slide or patient<input className="field" value={search} onChange={(event) => { setSearch(event.target.value); setOffset(0); }} placeholder="Slide or patient ID" /></label>{rows.isPending ? <p role="status">Loading frozen slide records…</p> : null}
      <div className="morphology-slide-list" role="list" aria-label="Frozen dataset slides">{rows.data?.items.map((slide) => <div role="listitem" key={slide.slideId}><button className={`morphology-slide-button ${selectedSlide === slide.slideId ? 'is-selected' : ''}`} aria-pressed={selectedSlide === slide.slideId} onClick={() => select(slide.slideId)}><strong>{slide.slideId}</strong><small>{slide.patientId || 'Patient not recorded'}{!slide.hasImage ? ' · image not linked' : ''}</small><small>{slide.reviewStatus ? reviewStatusLabels[slide.reviewStatus] : 'Review status unavailable'}</small></button></div>)}</div>
      {rows.data ? <div className="inline-actions"><button className="btn btn-secondary btn-small" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 100))}>Previous</button><span>{rows.data.matching ? offset + 1 : 0}–{Math.min(offset + 100, rows.data.matching)} / {rows.data.matching}</span><button className="btn btn-secondary btn-small" disabled={offset + 100 >= rows.data.matching} onClick={() => setOffset(offset + 100)}>Next</button></div> : null}
    </aside><div>{selectedSlide ? <QualitySlide key={`${datasetId}:${selectedSlide}:${bundleId}`} project={project} datasetId={datasetId} slideId={selectedSlide} featureBundleId={bundleId || undefined} patchIndex={patchIndex} onPatch={setPatchIndex} /> : <p>No matching slides.</p>}</div></div>
    {index && selectedSlide && index.points.some((point) => point.slideId === selectedSlide) ? <FeatureNeighbors key={`${index.indexId}:${selectedSlide}`} project={project} index={index} slideId={selectedSlide} onSelect={select} /> : null}
  </div>;
}
