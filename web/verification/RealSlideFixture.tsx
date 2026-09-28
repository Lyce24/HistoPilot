/** Real local slide pixels in the production viewer, without an HTTP server. */
import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import QualitySlideCanvas from '../src/components/QualitySlideCanvas';
import type { MorphologyRegion, QualityEvidence } from '../src/api/morphology';
import { useImageBlob } from '../src/components/SlideGalleryCard';
import { morphology } from '../src/api/morphology';
import { planSlideTiles, SLIDE_VISIBLE_TILE_LIMIT } from '../src/lib/slideTiles';
import '../src/styles.css';
import '../src/scientific.css';
import '../src/local-workspace.css';
import '../src/components/VisualQualityExplorer.css';

interface RealSlide { slideIndex: number; label: string; width: number; height: number; backend: string; sourceFingerprint: string; tissueCenters: { x: number; y: number }[] }
interface BridgeResult { base64?: string; elapsedMs?: number; error?: { message: string; code: string; status: number } }
interface Call { slideIndex: number; region?: MorphologyRegion; maxSize: number; startedAt: number; finishedAt?: number; nativeMs?: number; bytes?: number; aborted?: boolean; error?: string }
declare global { interface Window {
  readRealSlide: (request: { slideIndex: number; region?: MorphologyRegion; maxSize: number }) => Promise<BridgeResult>;
  realSlideTest: { ready: boolean; load: (meta: RealSlide, options?: { dense?: boolean }) => void; selectRegion: (region: MorphologyRegion) => void; calls: Call[]; errors: string[]; plan: typeof planSlideTiles; maxVisibleTiles: number; loadedAt: number; unmount: () => void };
} }
const calls: Call[] = [], errors: string[] = [];
const nativeFetch = window.fetch.bind(window);
let current: RealSlide | undefined;
const root = createRoot(document.getElementById('app')!);
const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
window.fetch = async (input, init) => {
  const url = new URL(String(input), 'http://real-slide-test.invalid');
  if (url.pathname === '/api/v1/session') return reply({ token: 'offline-real-slide-test' });
  if (!url.pathname.endsWith('/morphology/image') || !current) throw new Error('Unexpected real-slide fixture route');
  const meta = current;
  const region = url.searchParams.has('x') ? Object.fromEntries(['x', 'y', 'width', 'height'].map((key) => [key, Number(url.searchParams.get(key))])) as unknown as MorphologyRegion : undefined;
  const request = { slideIndex: meta.slideIndex, region, maxSize: Number(url.searchParams.get('max_size') ?? 1536) };
  if (url.searchParams.get('sourceFingerprint') !== meta.sourceFingerprint) return reply({ detail: 'Source mismatch in test fixture', code: 'MORPHOLOGY_SLIDE_CHANGED' }, 409);
  const call: Call = { ...request, startedAt: performance.now() }; calls.push(call);
  let abort: () => void = () => {};
  try {
    const result = await Promise.race([window.readRealSlide(request), new Promise<never>((_, reject) => {
      abort = () => { call.aborted = true; reject(new DOMException('Aborted', 'AbortError')); };
      if (init?.signal?.aborted) abort(); else init?.signal?.addEventListener('abort', abort, { once: true });
    })]);
    call.nativeMs = result.elapsedMs;
    if (result.error) { call.error = result.error.message; errors.push(result.error.message); return reply({ detail: result.error.message, code: result.error.code }, result.error.status); }
    // Decode the test transport in native browser code; a byte-by-byte JS copy
    // would add main-thread work absent from real HTTP image responses.
    call.bytes = Math.floor(result.base64!.length * .75);
    return nativeFetch(`data:image/png;base64,${result.base64}`, { signal: init?.signal });
  } finally { call.finishedAt = performance.now(); init?.signal?.removeEventListener('abort', abort); }
};

function Workspace({ meta, dense }: { meta: RealSlide; dense: boolean }) {
  const [overview, setOverview] = useState<Blob>();
  const [region, setRegion] = useState<MorphologyRegion>();
  const [patch, setPatch] = useState<number>();
  useEffect(() => { window.realSlideTest.selectRegion = setRegion; }, []);
  useEffect(() => {
    const controller = new AbortController();
    void morphology.image('real-slide-test', 'real-slide-test', String(meta.slideIndex), controller.signal, undefined, 1536, meta.sourceFingerprint).then(setOverview, (error) => { if (!controller.signal.aborted) errors.push(String(error)); });
    return () => controller.abort();
  }, [meta]);
  const url = useImageBlob(overview);
  const [quality] = useState<QualityEvidence>(() => ({ slideId: String(meta.slideIndex), datasetId: 'real-slide-test', sourceFingerprint: meta.sourceFingerprint, width: meta.width, height: meta.height,
    patchWidth: dense ? meta.width / 80 : null, patchHeight: dense ? meta.height / 80 : null, patchCount: dense ? 4096 : null,
    patches: dense ? Array.from({ length: 4096 }, (_, patchIndex) => ({ patchIndex, x: patchIndex % 64 * meta.width / 64, y: Math.floor(patchIndex / 64) * meta.height / 64 })) : [],
    tissueContours: dense ? Array.from({ length: 50 }, (_, index) => [Array.from({ length: 1000 }, (_, vertex) => { const angle = vertex / 999 * Math.PI * 2; return [(index % 10 + .5) * meta.width / 10 + Math.cos(angle) * meta.width / 30, (Math.floor(index / 10) + .5) * meta.height / 5 + Math.sin(angle) * meta.height / 20]; })]) : [],
    coordinateBounds: null, artifactRemoval: null, warnings: [],
  }));
  return <main style={{ padding: 16 }}><p>Real-slide viewer verification · {meta.label} · {meta.backend} · {meta.width} × {meta.height}</p>
    {dense ? <p>Synthetic coverage geometry for rendering stress; the underlying slide pixels are real.</p> : null}
    <QualitySlideCanvas project="real-slide-test" datasetId="real-slide-test" slideId={String(meta.slideIndex)} quality={quality} overviewURL={url} patchIndex={patch} selectedRegion={region} onPatch={setPatch} onRegion={setRegion} onSourceError={(error) => errors.push(error.message)} />
  </main>;
}
window.realSlideTest = { ready: true, calls, errors, plan: planSlideTiles, maxVisibleTiles: SLIDE_VISIBLE_TILE_LIMIT, loadedAt: 0, selectRegion: () => {},
  load(meta, options = {}) { current = meta; calls.length = 0; errors.length = 0; this.loadedAt = performance.now(); root.render(<Workspace key={`${meta.slideIndex}:${this.loadedAt}`} meta={meta} dense={Boolean(options.dense)} />); },
  unmount() { root.render(null); current = undefined; },
};
