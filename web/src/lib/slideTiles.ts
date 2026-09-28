import type { SlideRegion } from '../api/interpretation';

export const SLIDE_TILE_SIZE = 512;
export const SLIDE_VISIBLE_TILE_LIMIT = 64;
export const SLIDE_PREPARATION_LIMIT = 128;
export const SLIDE_PREFETCH_LIMIT = 24;
export interface SlideTile { key: string; level: number; region: SlideRegion; maxSize: number }
export interface LoadedSlideTile extends SlideTile { url: string; bytes: number; touched: number }
export interface TileSnapshot { tiles: LoadedSlideTile[]; cachedTiles: LoadedSlideTile[]; loading: boolean; error: Error | null; preparing: boolean; prepared: number; preparationTotal: number }
export interface DecodedTile { url: string; width: number; height: number }
export interface TileStoreOptions {
  fetch: (tile: SlideTile, signal: AbortSignal) => Promise<Blob>;
  decode: (blob: Blob, signal: AbortSignal) => Promise<DecodedTile>;
  release: (url: string) => void;
  changed: () => void;
  now?: () => number;
  maxBytes?: number;
  maxEntries?: number;
  lifetime?: number;
  timeoutMs?: number;
}

export function regionsIntersect(a: SlideRegion, b: SlideRegion): boolean {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
}

/** Camera painting is independent of the throttled request queue. */
export function selectSlideTiles(cached: LoadedSlideTile[], view: SlideRegion, wanted: SlideTile[]): LoadedSlideTile[] {
  const level = wanted[0]?.level;
  const keys = new Set(cached.map((tile) => tile.key));
  const complete = wanted.length > 0 && wanted.every((tile) => keys.has(tile.key));
  return cached.filter((tile) => regionsIntersect(tile.region, view)
    && (level === undefined || (complete ? tile.level === level : tile.level >= level))).sort((a, b) => b.level - a.level);
}

export function isSlideSourceError(error: unknown): boolean {
  const response = error as { status?: number; code?: string } | null;
  return [401, 403, 404, 409, 410].includes(response?.status ?? 0)
    || ['SLIDE_FORMAT_UNSUPPORTED', 'MORPHOLOGY_SLIDE_CHANGED', 'SLIDE_SOURCE_CHANGED', 'INTERPRETATION_INPUT_INVALID'].includes(response?.code ?? '');
}

/** A fixed grid means nearby views share requests and retain exact level-0 placement. */
export function planSlideTiles(view: SlideRegion, width: number, height: number, viewportWidth: number, viewportHeight: number, pixelRatio = 1): SlideTile[] {
  if (![view.x, view.y, view.width, view.height, width, height, viewportWidth, viewportHeight].every(Number.isFinite)
    || Math.min(view.width, view.height, width, height, viewportWidth, viewportHeight) <= 0) return [];
  const ratio = Number.isFinite(pixelRatio) ? Math.max(1, Math.min(2, pixelRatio)) : 1;
  const scale = Math.min(viewportWidth / view.width, viewportHeight / view.height) * ratio;
  let level = Math.max(0, Math.floor(Math.log2(1 / scale)));
  const left = Math.max(0, view.x), top = Math.max(0, view.y);
  const right = Math.min(width, view.x + view.width), bottom = Math.min(height, view.y + view.height);
  if (right <= left || bottom <= top) return [];
  let span: number, firstX: number, firstY: number, lastX: number, lastY: number;
  do {
    span = SLIDE_TILE_SIZE * 2 ** level;
    firstX = Math.floor(left / span); firstY = Math.floor(top / span);
    lastX = Math.ceil(right / span) - 1; lastY = Math.ceil(bottom / span) - 1;
    if ((lastX - firstX + 1) * (lastY - firstY + 1) <= SLIDE_VISIBLE_TILE_LIMIT) break;
    level += 1;
  } while (level < 40);
  return tilesAtLevel(view, width, height, level);
}

/** Enumerate an exact pyramid level; preparation must not inherit the visible-grid cap. */
function tilesAtLevel(view: SlideRegion, width: number, height: number, level: number): SlideTile[] {
  const span = SLIDE_TILE_SIZE * 2 ** level;
  const left = Math.max(0, view.x), top = Math.max(0, view.y);
  const right = Math.min(width, view.x + view.width), bottom = Math.min(height, view.y + view.height);
  if (right <= left || bottom <= top) return [];
  const tiles: SlideTile[] = [], centerX = (left + right) / 2, centerY = (top + bottom) / 2;
  for (let row = Math.floor(top / span); row < Math.ceil(bottom / span); row += 1) for (let column = Math.floor(left / span); column < Math.ceil(right / span); column += 1) {
    const region = { x: column * span, y: row * span, width: Math.min(span, width - column * span), height: Math.min(span, height - row * span) };
    tiles.push({ key: `${level}:${column}:${row}`, level, region, maxSize: Math.max(64, Math.ceil(Math.max(region.width, region.height) / 2 ** level)) });
  }
  const distance = (tile: SlideTile) => (tile.region.x + tile.region.width / 2 - centerX) ** 2 + (tile.region.y + tile.region.height / 2 - centerY) ** 2;
  return tiles.sort((a, b) => distance(a) - distance(b));
}

/** Prepare three complete levels, coarse first, within a fixed memory/work budget. */
export function planSlidePreparation(width: number, height: number): SlideTile[] {
  if (![width, height].every(Number.isFinite) || Math.min(width, height) <= 0) return [];
  let level = 0;
  const count = (value: number) => Math.ceil(width / (SLIDE_TILE_SIZE * 2 ** value)) * Math.ceil(height / (SLIDE_TILE_SIZE * 2 ** value));
  while (count(level) + count(level + 1) + count(level + 2) > SLIDE_PREPARATION_LIMIT) level += 1;
  const view = { x: 0, y: 0, width, height };
  return [level + 2, level + 1, level].flatMap((value) => tilesAtLevel(view, width, height, value));
}

/** Predict a modest pan and the next centered zoom, never a whole finer pyramid. */
export function planSlidePrefetch(view: SlideRegion, width: number, height: number, viewportWidth: number, viewportHeight: number, pixelRatio = 1): SlideTile[] {
  const visible = planSlideTiles(view, width, height, viewportWidth, viewportHeight, pixelRatio);
  if (!visible.length) return [];
  const level = visible[0].level, span = SLIDE_TILE_SIZE * 2 ** level;
  const known = new Set(visible.map((tile) => tile.key));
  const neighbors = tilesAtLevel({ x: view.x - span, y: view.y - span, width: view.width + span * 2, height: view.height + span * 2 }, width, height, level).filter((tile) => !known.has(tile.key));
  const next = level > 0 ? tilesAtLevel({ x: view.x + view.width / 4, y: view.y + view.height / 4, width: view.width / 2, height: view.height / 2 }, width, height, level - 1) : [];
  const result: SlideTile[] = [];
  // Alternate zoom and pan anticipation so neither consumes the entire budget.
  for (let index = 0; result.length < SLIDE_PREFETCH_LIMIT && index < Math.max(next.length, neighbors.length); index += 1) {
    for (const tile of [next[index], neighbors[index]]) if (tile && !known.has(tile.key) && result.length < SLIDE_PREFETCH_LIMIT) { result.push(tile); known.add(tile.key); }
  }
  return result;
}

/** One viewer owns its cache and at most two reads, including image decoding.
 * Camera movement only replaces queued work; active reads finish rather than
 * flooding a native decoder with abandoned work. Source changes dispose it. */
export class SlideTileStore {
  private entries = new Map<string, LoadedSlideTile>();
  private active = new Map<string, AbortController>();
  private failures = new Map<string, Error>();
  private wanted: SlideTile[] = [];
  private preparation: SlideTile[] = [];
  private prediction: SlideTile[] = [];
  private attempted = new Set<string>();
  private speculative = new Set<string>();
  private prepared = new Set<string>();
  private view: SlideRegion = { x: 0, y: 0, width: 0, height: 0 };
  private disposed = false;
  private generation = 0;
  private sourceError: Error | null = null;
  private now: () => number;
  constructor(private options: TileStoreOptions) { this.now = options.now ?? Date.now; }

  prepare(tiles: SlideTile[]) {
    if (this.disposed) return;
    this.preparation = tiles;
    this.pump(); this.options.changed();
  }

  update(tiles: SlideTile[], view: SlideRegion) {
    if (this.disposed) return;
    this.wanted = tiles; this.view = view; this.prediction = [];
    const wanted = new Set([...tiles, ...this.preparation].map((tile) => tile.key));
    for (const key of this.failures.keys()) if (!wanted.has(key)) this.failures.delete(key);
    for (const tile of tiles) { const entry = this.entries.get(tile.key); if (entry) entry.touched = this.now(); }
    this.trim(); this.pump(); this.options.changed();
  }

  prefetch(tiles: SlideTile[]) {
    if (this.disposed) return;
    this.prediction = tiles.slice(0, SLIDE_PREFETCH_LIMIT);
    const keys = new Set(this.prediction.map((tile) => tile.key));
    for (const key of this.attempted) if (!keys.has(key)) this.attempted.delete(key);
    this.pump();
  }

  retry() { this.sourceError = null; this.failures.clear(); this.pump(); this.options.changed(); }

  snapshot(): TileSnapshot {
    const cachedTiles = [...this.entries.values()];
    return {
      tiles: selectSlideTiles(cachedTiles, this.view, this.wanted), cachedTiles,
      loading: !this.sourceError && this.wanted.some((tile) => !this.entries.has(tile.key) && !this.failures.has(tile.key)),
      error: this.sourceError ?? [...this.wanted, ...this.preparation].map((tile) => this.failures.get(tile.key)).find(Boolean) ?? null,
      preparing: !this.sourceError && this.preparation.some((tile) => !this.prepared.has(tile.key) && !this.failures.has(tile.key)),
      prepared: this.prepared.size,
      preparationTotal: this.preparation.length,
    };
  }

  private trim() {
    const wanted = new Set(this.wanted.map((tile) => tile.key)), now = this.now();
    for (const [key, entry] of this.entries) if (!wanted.has(key) && !this.prepared.has(key) && now - entry.touched > (this.options.lifetime ?? Infinity)) this.remove(key);
    let bytes = [...this.entries.values()].reduce((sum, tile) => sum + tile.bytes, 0);
    const coarseLevel = this.preparation.length ? Math.min(...this.preparation.map((tile) => tile.level)) + 1 : Infinity;
    const importance = (tile: LoadedSlideTile) => wanted.has(tile.key) ? 4
      : this.prepared.has(tile.key) && tile.level >= coarseLevel ? 3
      : regionsIntersect(tile.region, this.view) && tile.level >= (this.wanted[0]?.level ?? Infinity) ? 2
      : this.attempted.has(tile.key) ? 0 : 1;
    const candidates = [...this.entries.values()].sort((a, b) => importance(a) - importance(b) || a.touched - b.touched);
    for (const entry of candidates) {
      if (this.entries.size <= (this.options.maxEntries ?? 192) && bytes <= (this.options.maxBytes ?? 256 * 1024 * 1024)) break;
      bytes -= entry.bytes; this.remove(entry.key);
      // A pathological response must not cause an evict/refetch loop.
      if (wanted.has(entry.key)) this.failures.set(entry.key, new Error('This view exceeds the slide tile cache budget. Zoom into a smaller region.'));
    }
  }
  private remove(key: string) {
    const entry = this.entries.get(key);
    if (entry) { this.options.release(entry.url); this.entries.delete(key); }
  }
  private pump() {
    if (this.disposed || this.sourceError) return;
    while (this.active.size < 2) {
      const pending = (value: SlideTile) => !this.entries.has(value.key) && !this.active.has(value.key) && !this.failures.has(value.key);
      let tile = this.wanted.find(pending) ?? this.preparation.find((value) => !this.prepared.has(value.key) && pending(value));
      if (!tile && this.speculative.size === 0) {
        tile = this.prediction.find((value) => !this.attempted.has(value.key) && pending(value));
        if (tile) { this.attempted.add(tile.key); this.speculative.add(tile.key); }
      }
      if (!tile) return;
      const controller = new AbortController(); this.active.set(tile.key, controller);
      void this.load(tile, controller, this.generation);
    }
  }
  private async load(tile: SlideTile, controller: AbortController, generation: number) {
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let timedOut = false;
    let abort: () => void = () => {};
    const operation = async () => {
      const blob = await this.options.fetch(tile, controller.signal);
      if (this.disposed || controller.signal.aborted || generation !== this.generation) return;
      const decoded = await this.options.decode(blob, controller.signal);
      if (this.disposed || controller.signal.aborted || generation !== this.generation) { this.options.release(decoded.url); return; }
      return { blob, decoded };
    };
    try {
      const result = await Promise.race([operation(), new Promise<never>((_, reject) => {
        abort = () => reject(new DOMException('Slide loading cancelled', 'AbortError'));
        controller.signal.addEventListener('abort', abort, { once: true });
      }), new Promise<never>((_, reject) => {
        timeout = setTimeout(() => {
          timedOut = true;
          reject(new Error('Slide detail took too long to load. Retry this view.'));
          controller.abort();
        }, this.options.timeoutMs ?? 30000);
      })]);
      if (!result) return;
      const { blob, decoded } = result;
      if (this.disposed || controller.signal.aborted || generation !== this.generation) { this.options.release(decoded.url); return; }
      this.entries.set(tile.key, { ...tile, url: decoded.url, bytes: blob.size + decoded.width * decoded.height * 4, touched: this.now() });
      if (this.preparation.some((value) => value.key === tile.key)) this.prepared.add(tile.key);
      this.trim();
    } catch (error) {
      if (!this.disposed && (!controller.signal.aborted || timedOut) && generation === this.generation) {
        const failure = error instanceof Error ? error : new Error('Slide detail could not be loaded.');
        if (isSlideSourceError(error)) {
          this.sourceError = failure; this.generation += 1;
          // Do not display a mixture of generations or fan out a failed source
          // into dozens of doomed reads. The separately loaded overview stays.
          for (const key of this.entries.keys()) this.remove(key);
          this.prepared.clear(); this.failures.clear();
        } else this.failures.set(tile.key, failure);
      }
    } finally {
      if (timeout !== undefined) clearTimeout(timeout);
      controller.signal.removeEventListener('abort', abort);
      if (this.active.get(tile.key) === controller) { this.active.delete(tile.key); this.speculative.delete(tile.key); }
      if (!this.disposed) { this.options.changed(); this.pump(); }
    }
  }
  dispose() {
    this.disposed = true; this.generation += 1;
    for (const controller of this.active.values()) controller.abort();
    for (const key of this.entries.keys()) this.remove(key);
    this.active.clear(); this.failures.clear(); this.wanted = []; this.preparation = []; this.prepared.clear(); this.prediction = []; this.attempted.clear(); this.speculative.clear();
  }
}
