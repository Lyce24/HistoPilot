import { describe, expect, it, vi } from 'vitest';
import { planSlidePreparation, planSlidePrefetch, planSlideTiles, selectSlideTiles, SLIDE_VISIBLE_TILE_LIMIT, SlideTileStore, type DecodedTile, type SlideTile } from './slideTiles';

const full = { x: 0, y: 0, width: 6000, height: 4000 };
const tile = (key: string, x = 0): SlideTile => ({ key, level: 0, region: { x, y: 0, width: 512, height: 512 }, maxSize: 512 });
const flush = async () => { for (let i = 0; i < 8; i += 1) await Promise.resolve(); };

function fixture(options: { maxEntries?: number; maxBytes?: number; lifetime?: number; timeoutMs?: number } = {}) {
  const pending: { tile: SlideTile; signal: AbortSignal; resolve: (blob: Blob) => void; reject: (error: Error) => void }[] = [];
  const release = vi.fn(), changed = vi.fn();
  let now = 0, count = 0;
  const decode = vi.fn(async (): Promise<DecodedTile> => ({ url: `blob:${count++}`, width: 512, height: 512 }));
  const fetch = vi.fn((tile: SlideTile, signal: AbortSignal) => new Promise<Blob>((resolve, reject) => { pending.push({ tile, signal, resolve, reject }); }));
  const store = new SlideTileStore({ fetch, decode, release, changed, now: () => now, ...options });
  async function resolve(key: string) { const value = pending.find((item) => item.tile.key === key); expect(value).toBeDefined(); value!.resolve(new Blob(['pixels'])); await flush(); }
  return { store, pending, fetch, decode, release, changed, resolve, advance: (ms: number) => { now += ms; } };
}

describe('fixed slide tile planning', () => {
  it('reuses identical fixed bounds across nearby views, prioritizing the view center', () => {
    const first = planSlideTiles({ x: 1020, y: 1020, width: 800, height: 800 }, 6000, 4000, 800, 800);
    const nearby = planSlideTiles({ x: 1025, y: 1025, width: 800, height: 800 }, 6000, 4000, 800, 800);
    expect(first[0].key).toBe('0:2:2');
    const shared = first.filter((value) => nearby.some((other) => other.key === value.key));
    expect(shared.length).toBeGreaterThan(0);
    for (const value of shared) expect(nearby.find((other) => other.key === value.key)).toEqual(value);
  });
  it('clips border tiles exactly to the slide without shifting their origins', () => {
    const planned = planSlideTiles({ x: 5500, y: 3500, width: 500, height: 500 }, 6000, 4000, 500, 500);
    const edge = planned.find((value) => value.key === '0:11:7')!;
    expect(edge.region).toEqual({ x: 5632, y: 3584, width: 368, height: 416 });
    expect(edge.maxSize).toBe(416);
    for (const value of planned) {
      expect(value.region.x + value.region.width).toBeLessThanOrEqual(6000);
      expect(value.region.y + value.region.height).toBeLessThanOrEqual(4000);
      expect(value.maxSize).toBeGreaterThanOrEqual(64); expect(value.maxSize).toBeLessThanOrEqual(512);
    }
  });
  it('caps display density and the visible tile budget even on an enormous viewport', () => {
    expect(planSlideTiles(full, 6000, 4000, 800, 600, 8)).toEqual(planSlideTiles(full, 6000, 4000, 800, 600, 2));
    expect(planSlideTiles(full, 6000, 4000, 100000, 100000, 2).length).toBeLessThanOrEqual(SLIDE_VISIBLE_TILE_LIMIT);
    expect(planSlideTiles(full, 6000, 4000, 0, 100)).toEqual([]);
    expect(planSlideTiles({ ...full, x: NaN }, 6000, 4000, 100, 100)).toEqual([]);
  });
  it('retains sharp high-DPI detail across grid alignments in the real wide TIFF geometry', () => {
    const width = 121856, height = 34816;
    const view = { x: 57120, y: 16320, width: 7616, height: 2176 };
    for (const viewportWidth of [1888, 2528]) {
      const centered = planSlideTiles(view, width, height, viewportWidth, 544, 2);
      // This field needs 36 tiles at the correct level. A 32-tile cap silently
      // selected level 2, spreading each image pixel over roughly 2 display pixels.
      expect(centered).toHaveLength(36);
      expect(centered.every((value) => value.level === 1)).toBe(true);
      const shifted = planSlideTiles({ ...view, x: (width - view.width) * .37, y: (height - view.height) * .37 }, width, height, viewportWidth, 544, 2);
      expect(shifted.every((value) => value.level === 1)).toBe(true);
      const displayScale = Math.min(viewportWidth / view.width, 544 / view.height) * 2;
      expect(2 ** centered[0].level * displayScale).toBeLessThanOrEqual(1);
      const standardDensity = planSlideTiles(view, width, height, viewportWidth, 544, 1);
      expect(standardDensity).toHaveLength(15); expect(standardDensity.every((value) => value.level === 2)).toBe(true);
    }
  });
  it('allows a complete 64-tile field and still coarsens a field that exceeds the bound', () => {
    const exact = planSlideTiles({ x: 0, y: 0, width: 4096, height: 4096 }, 100000, 100000, 100000, 100000, 2);
    expect(exact).toHaveLength(64); expect(exact.every((value) => value.level === 0)).toBe(true);
    const excessive = planSlideTiles({ x: 0, y: 0, width: 5120, height: 5120 }, 100000, 100000, 100000, 100000, 2);
    expect(excessive).toHaveLength(25); expect(excessive.every((value) => value.level === 1)).toBe(true);
    expect(excessive.length).toBeLessThanOrEqual(SLIDE_VISIBLE_TILE_LIMIT);
  });
  it('prepares three exact bounded levels, including more than 32 tiles at one level', () => {
    const plan = planSlidePreparation(6000, 4000);
    expect(plan).toHaveLength(126); expect(plan.slice(0, 6).every((value) => value.level === 2)).toBe(true);
    expect(plan.slice(6, 30).every((value) => value.level === 1)).toBe(true);
    expect(plan.slice(30).every((value) => value.level === 0)).toBe(true);
    expect(new Set(plan.map((value) => value.key)).size).toBe(plan.length);
    const huge = planSlidePreparation(300000, 200000);
    expect(huge.length).toBeLessThanOrEqual(128);
    const levels = [...new Set(huge.map((value) => value.level))]; expect(levels).toHaveLength(3);
    expect(Math.min(...levels)).toBeGreaterThan(0);
    for (const level of levels) expect(huge.filter((value) => value.level === level).reduce((area, value) => area + value.region.width * value.region.height, 0)).toBe(300000 * 200000);
  });
  it('predicts both the next zoom and neighboring pan tiles with no visible duplicates', () => {
    const view = { x: 24000, y: 16000, width: 8000, height: 6000 };
    const visible = planSlideTiles(view, 60000, 40000, 1000, 750);
    const predicted = planSlidePrefetch(view, 60000, 40000, 1000, 750);
    expect(predicted.length).toBeGreaterThan(0); expect(predicted.length).toBeLessThanOrEqual(24);
    expect(new Set(predicted.map((value) => value.key)).size).toBe(predicted.length);
    expect(predicted.every((value) => !visible.some((shown) => shown.key === value.key))).toBe(true);
    expect(predicted.some((value) => value.level === visible[0].level - 1)).toBe(true);
    expect(predicted.some((value) => value.level === visible[0].level)).toBe(true);
    const edge = planSlidePrefetch({ x: 0, y: 0, width: 800, height: 600 }, 6000, 4000, 800, 600);
    expect(edge.every((value) => value.level === 0 && value.region.x >= 0 && value.region.y >= 0)).toBe(true);
    expect(planSlidePrefetch(full, 6000, 4000, 0, 0)).toEqual([]);
  });

});

describe('bounded progressive tile loading', () => {
  it('keeps two reads in flight, drops obsolete queued work, and publishes each decoded tile', async () => {
    const f = fixture(); f.store.update([tile('a'), tile('b', 512), tile('obsolete', 1024)], full);
    expect(f.fetch).toHaveBeenCalledTimes(2);
    f.store.update([tile('latest', 1536)], full);
    expect(f.pending.every((value) => !value.signal.aborted)).toBe(true);
    await f.resolve('a');
    expect(f.fetch).toHaveBeenCalledTimes(3); expect(f.pending[2].tile.key).toBe('latest');
    expect(f.store.snapshot().tiles.map((value) => value.key)).toEqual(['a']);
    await f.resolve('latest'); await f.resolve('b');
    expect(f.pending.some((value) => value.tile.key === 'obsolete')).toBe(false);
    expect(f.store.snapshot().loading).toBe(false); f.store.dispose();
  });
  it('reuses decoded tiles and exact URLs after panning away and back', async () => {
    const f = fixture(); f.store.update([tile('a')], full); await f.resolve('a');
    const loaded = f.store.snapshot().tiles[0];
    f.store.update([tile('b', 512)], full); await f.resolve('b');
    f.store.update([tile('a')], full);
    expect(f.fetch).toHaveBeenCalledTimes(2); expect(f.store.snapshot().tiles.find((value) => value.key === 'a')).toEqual(loaded);
    f.store.dispose();
  });
  it('prepares the slide with progress, prioritizes requested detail, and does not restart evicted preparation', async () => {
    const f = fixture({ maxEntries: 2 }); f.store.prepare([tile('coarse'), tile('fine1'), tile('fine2')]);
    expect(f.store.snapshot()).toMatchObject({ preparing: true, prepared: 0, preparationTotal: 3 });
    f.store.update([tile('visible')], full);
    await f.resolve('coarse'); expect(f.pending[2].tile.key).toBe('visible');
    await f.resolve('fine1'); expect(f.pending[3].tile.key).toBe('fine2');
    await f.resolve('visible'); await f.resolve('fine2');
    expect(f.store.snapshot()).toMatchObject({ preparing: false, prepared: 3, preparationTotal: 3 });
    expect(f.store.snapshot().tiles.length).toBeLessThanOrEqual(2);
    f.store.update([], full); expect(f.fetch).toHaveBeenCalledTimes(4); f.store.dispose();
  });
  it('finishes partial preparation with an actionable error and retries only failed work', async () => {
    const f = fixture(); f.store.prepare([tile('a'), tile('b')]);
    await f.resolve('a'); f.pending[1].reject(new Error('Reader unavailable')); await flush();
    expect(f.store.snapshot()).toMatchObject({ preparing: false, prepared: 1, preparationTotal: 2 });
    expect(f.store.snapshot().error?.message).toBe('Reader unavailable');
    f.store.retry(); expect(f.fetch).toHaveBeenCalledTimes(3); expect(f.pending[2].tile.key).toBe('b');
    f.pending[2].resolve(new Blob(['pixels'])); await flush();
    expect(f.store.snapshot()).toMatchObject({ preparing: false, prepared: 2, error: null }); f.store.dispose();
  });
  it('preserves prepared tiles during temporary reader overload and retries only the failed tile', async () => {
    const f = fixture(); f.store.prepare([tile('a'), tile('b'), tile('c')]);
    f.store.update([], full); await f.resolve('a');
    const warm = f.store.snapshot().tiles[0];
    const busy = Object.assign(new Error('Slide readers are busy. Try again shortly.'), { status: 503, code: 'SLIDE_READER_BUSY' });
    f.pending[1].reject(busy); await flush();
    expect(f.store.snapshot()).toMatchObject({ prepared: 1, preparationTotal: 3, error: busy });
    expect(f.store.snapshot().tiles).toContainEqual(warm); expect(f.release).not.toHaveBeenCalled();
    await f.resolve('c');
    expect(f.fetch).toHaveBeenCalledTimes(3);
    expect(f.store.snapshot()).toMatchObject({ prepared: 2, preparing: false, error: busy });
    f.store.retry();
    expect(f.fetch).toHaveBeenCalledTimes(4); expect(f.pending[3].tile.key).toBe('b');
    expect(f.store.snapshot().tiles).toContainEqual(warm);
    f.pending[3].resolve(new Blob(['pixels'])); await flush();
    expect(f.store.snapshot()).toMatchObject({ prepared: 3, preparing: false, error: null });
    expect(f.release).not.toHaveBeenCalled(); f.store.dispose();
  });
  it('stops preparation after a source identity failure and ignores old responses across explicit retry', async () => {
    const f = fixture(); f.store.prepare([tile('a'), tile('b'), tile('c'), tile('d')]);
    const changed = Object.assign(new Error('The slide source changed.'), { status: 409, code: 'MORPHOLOGY_SLIDE_CHANGED' });
    f.pending[0].reject(changed); await flush();
    expect(f.fetch).toHaveBeenCalledTimes(2);
    expect(f.store.snapshot()).toMatchObject({ preparing: false, loading: false, prepared: 0, error: changed });
    f.store.retry();
    expect(f.fetch).toHaveBeenCalledTimes(3); expect(f.pending[2].tile.key).toBe('a');
    f.pending[1].resolve(new Blob(['old-source'])); await flush();
    expect(f.decode).not.toHaveBeenCalled(); expect(f.pending[3].tile.key).toBe('b');
    f.pending[2].resolve(new Blob(['fresh-source'])); await flush();
    expect(f.store.snapshot().prepared).toBe(1); expect(f.store.snapshot().error).toBeNull();
    f.store.dispose();
  });
  it('releases an image decoded after its source has been disposed', async () => {
    let finishDecode: (image: DecodedTile) => void = () => {};
    const release = vi.fn();
    const store = new SlideTileStore({ fetch: async () => new Blob(['pixels']), decode: () => new Promise((resolve) => { finishDecode = resolve; }), release, changed: vi.fn() });
    store.update([tile('a')], full); await flush(); store.dispose();
    finishDecode({ url: 'blob:late', width: 512, height: 512 }); await flush();
    expect(release).toHaveBeenCalledWith('blob:late'); expect(store.snapshot().tiles).toEqual([]);
  });
  it('reports errors only for the latest view or preparation', async () => {
    const f = fixture(); f.store.update([tile('old')], full); f.store.update([tile('new')], full);
    f.pending[0].reject(new Error('Old viewport failed')); await flush();
    expect(f.store.snapshot().error).toBeNull(); await f.resolve('new'); f.store.dispose();
  });
  it('counts decoded pixels toward memory, releases evicted URLs, and avoids an oversized-tile retry loop', async () => {
    const f = fixture({ maxBytes: 1024 * 1024 + 10 });
    f.store.update([tile('a')], full); await f.resolve('a');
    f.store.update([tile('b', 512)], full); await f.resolve('b');
    expect(f.release).toHaveBeenCalledWith('blob:0'); expect(f.store.snapshot().tiles.map((value) => value.key)).toEqual(['b']); f.store.dispose();
    const tiny = fixture({ maxBytes: 100 }); tiny.store.update([tile('large')], full); await tiny.resolve('large');
    expect(tiny.fetch).toHaveBeenCalledTimes(1); expect(tiny.store.snapshot().error?.message).toContain('cache budget'); tiny.store.dispose();
  });
  it('keeps prepared tiles past idle expiry while reclaiming unrelated old views', async () => {
    const f = fixture({ lifetime: 50 }); f.store.prepare([tile('prepared')]); await f.resolve('prepared');
    f.store.update([tile('transient', 512)], full); await f.resolve('transient');
    f.advance(100); f.store.update([], full);
    expect(f.store.snapshot().tiles.map((value) => value.key)).toEqual(['prepared']); expect(f.release).toHaveBeenCalledWith('blob:1'); f.store.dispose();
  });
  it('aborts source work, releases loaded URLs, and ignores late responses after disposal', async () => {
    const f = fixture(); f.store.update([tile('a'), tile('b')], full); await f.resolve('a');
    f.store.dispose(); expect(f.pending[1].signal.aborted).toBe(true); expect(f.release).toHaveBeenCalledWith('blob:0');
    await f.resolve('b'); expect(f.decode).toHaveBeenCalledTimes(1); expect(f.store.snapshot().tiles).toEqual([]);
  });
  it('reserves one decoder slot for navigation and replaces obsolete prediction queues', async () => {
    const f = fixture(); f.store.update([tile('shown')], full); await f.resolve('shown');
    f.store.prefetch([tile('next'), tile('obsolete')]);
    expect(f.fetch).toHaveBeenCalledTimes(2);
    f.store.update([tile('urgent')], full);
    expect(f.pending[2].tile.key).toBe('urgent');
    f.store.prefetch([tile('new-prediction')]);
    await f.resolve('next'); expect(f.pending[3].tile.key).toBe('new-prediction');
    await f.resolve('urgent'); await f.resolve('new-prediction');
    expect(f.pending.some((value) => value.tile.key === 'obsolete')).toBe(false);
    f.store.update([tile('new-prediction')], full);
    expect(f.fetch).toHaveBeenCalledTimes(4); f.store.dispose();
  });
  it('does not prefetch ahead of preparation, report speculative errors, or loop on eviction', async () => {
    const f = fixture({ maxEntries: 1 }); f.store.prepare([tile('prep1'), tile('prep2'), tile('prep3')]);
    f.store.prefetch([tile('pred1'), tile('pred2')]);
    await f.resolve('prep1'); expect(f.pending[2].tile.key).toBe('prep3');
    await f.resolve('prep2'); expect(f.pending[3].tile.key).toBe('pred1');
    await f.resolve('prep3');
    f.pending[3].reject(new Error('Speculative region unavailable')); await flush();
    expect(f.store.snapshot().error).toBeNull(); expect(f.pending[4].tile.key).toBe('pred2');
    await f.resolve('pred2');
    f.store.prefetch([tile('pred1'), tile('pred2')]);
    expect(f.fetch).toHaveBeenCalledTimes(5);
    f.store.update([tile('pred1')], full);
    expect(f.store.snapshot().error?.message).toContain('Speculative'); f.store.dispose();
    const tiny = fixture({ maxBytes: 10 }); tiny.store.prefetch([tile('oversized')]);
    await tiny.resolve('oversized'); tiny.store.prefetch([tile('oversized')]);
    expect(tiny.fetch).toHaveBeenCalledTimes(1); expect(tiny.store.snapshot().error).toBeNull(); tiny.store.dispose();
  });
  it('retains a coarse zoom-out fallback over unused fine tiles under memory pressure', async () => {
    const f = fixture({ maxEntries: 2 });
    const coarse = { ...tile('coarse'), level: 2 };
    f.store.prepare([coarse, tile('fine')]); await f.resolve('coarse'); await f.resolve('fine');
    f.advance(1); f.store.update([tile('visible')], full); await f.resolve('visible');
    f.store.update([], full);
    expect(f.store.snapshot().tiles.map((value) => value.key)).toEqual(['coarse', 'visible']); f.store.dispose();
  });
  it('does not paint prefetched finer levels and drops covered fallback layers', async () => {
    const f = fixture(); const coarse = { ...tile('coarse'), level: 2 }, visible = { ...tile('visible'), level: 1 };
    f.store.prepare([coarse]); f.store.update([visible], full); await f.resolve('coarse');
    expect(f.store.snapshot().tiles.map((value) => value.key)).toEqual(['coarse']);
    await f.resolve('visible'); f.store.prefetch([tile('future')]); await f.resolve('future');
    expect(f.store.snapshot().tiles.map((value) => value.key)).toEqual(['visible']);
    f.store.update([coarse], full);
    expect(f.store.snapshot().tiles.map((value) => value.key)).toEqual(['coarse']); f.store.dispose();
  });

  it('paints already cached neighbors immediately without waiting for a request update', async () => {
    const f = fixture();
    const left = tile('left'), right = tile('right', 512);
    f.store.prepare([left, right]);
    f.store.update([left], left.region);
    await f.resolve('left'); await f.resolve('right');
    const snapshot = f.store.snapshot();
    expect(snapshot.tiles.map((value) => value.key)).toEqual(['left']);
    expect(selectSlideTiles(snapshot.cachedTiles, right.region, [right]).map((value) => value.key)).toEqual(['right']);
    expect(f.fetch).toHaveBeenCalledTimes(2);
    f.store.dispose();
  });
  it('bounds a hung read, frees its slot, and ignores late pixels after retry', async () => {
    vi.useFakeTimers();
    const f = fixture({ timeoutMs: 50 });
    try {
      f.store.update([tile('hung')], full);
      await vi.advanceTimersByTimeAsync(50);
      expect(f.pending[0].signal.aborted).toBe(true);
      expect(f.store.snapshot()).toMatchObject({ loading: false });
      expect(f.store.snapshot().error?.message).toContain('too long');
      f.store.retry();
      expect(f.fetch).toHaveBeenCalledTimes(2);
      f.pending[0].resolve(new Blob(['late'])); await flush();
      expect(f.decode).not.toHaveBeenCalled();
      f.pending[1].resolve(new Blob(['current'])); await flush();
      expect(f.store.snapshot().tiles).toHaveLength(1);
      expect(f.store.snapshot().error).toBeNull();
      expect(vi.getTimerCount()).toBe(0);
    } finally { f.store.dispose(); vi.useRealTimers(); }
  });
  it('releases a decode that completes after its deadline and clears deadlines on disposal', async () => {
    vi.useFakeTimers();
    let finish: (value: DecodedTile) => void = () => {};
    const release = vi.fn();
    const store = new SlideTileStore({ fetch: async () => new Blob(['image']), decode: () => new Promise((resolve) => { finish = resolve; }), release, changed: vi.fn(), timeoutMs: 50 });
    try {
      store.update([tile('slow-decode')], full); await flush();
      await vi.advanceTimersByTimeAsync(50);
      finish({ url: 'blob:late-decode', width: 512, height: 512 }); await flush();
      expect(release).toHaveBeenCalledExactlyOnceWith('blob:late-decode');
      expect(store.snapshot().cachedTiles).toEqual([]);
      store.retry(); await flush();
      expect(vi.getTimerCount()).toBe(1);
      store.dispose(); await flush();
      expect(vi.getTimerCount()).toBe(0);
    } finally { store.dispose(); vi.useRealTimers(); }
  });

  it('does not publish a decoded result disposed between resolution and publication', async () => {
    const release = vi.fn();
    const store = new SlideTileStore({ fetch: async () => new Blob(['image']), decode: async () => {
      queueMicrotask(() => queueMicrotask(() => store.dispose()));
      return { url: 'blob:disposed', width: 512, height: 512 };
    }, release, changed: vi.fn() });
    store.update([tile('race')], full); await flush();
    expect(store.snapshot().cachedTiles).toEqual([]);
    expect(release).toHaveBeenCalledExactlyOnceWith('blob:disposed');
  });

});
