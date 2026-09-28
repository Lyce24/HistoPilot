/** Production QC canvas + genuine slide reads over stdio. Starts no HTTP server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const output = resolve(process.argv[2] ?? '/tmp/histopilot-real-slide-review/browser');
const catalogPath = resolve(process.argv[3] ?? join(output, '../catalog.json'));
const catalog = JSON.parse(await readFile(catalogPath, 'utf8'));
const indices = (process.env.HISTOPILOT_REAL_SLIDES ?? '1,3,4,0,2').split(',').map(Number);
const dprs = (process.env.HISTOPILOT_VERIFY_DPRS ?? '1,2').split(',').map(Number);
const viewportWidth = Number(process.env.HISTOPILOT_VERIFY_WIDTH ?? 1920);
const viewportHeight = Number(process.env.HISTOPILOT_VERIFY_HEIGHT ?? 1080);
const slides = indices.map(index => catalog.slides.find(slide => slide.slideIndex === index));
assert.ok(slides.every(Boolean), 'Every requested slide must exist in the inspected catalog');
await mkdir(output, { recursive: true });
const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter(name => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const python = spawn(process.env.HISTOPILOT_BRIDGE_PYTHON ?? '/home/yc_liu/projects/HistoPilot-dev/.venv/bin/python', ['-u', join(repo, 'scripts/real_slide_bridge.py'), '--catalog', catalogPath, ...(process.env.HISTOPILOT_REAL_API === '1' ? ['--api'] : [])], {
  cwd: repo, env: { ...process.env, HISTOPILOT_TRIDENT_PYTHON: process.env.HISTOPILOT_TRIDENT_PYTHON ?? '/home/yc_liu/miniconda3/envs/trident/bin/python' }, stdio: ['pipe', 'pipe', 'pipe'],
});
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'real-browser-profile')], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
const requests = new Map(), reads = new Map(), exceptions = [], transport = [];
const results = { startedAt: new Date().toISOString(), transport: process.env.HISTOPILOT_REAL_API === '1' ? 'file:// Chromium CDP + local Python stdio + actual FastAPI TestClient; no network listener' : 'file:// Chromium CDP + local Python stdio + production renderer; no HTTP server', limitations: ['Headless Chromium software rendering; not a physical trackpad or GPU benchmark.', 'Reader and encoded-image caches reset before each run; operating-system file cache is not cleared.', 'Dense QC geometry is synthetic; all underlying slide image pixels are genuine.', 'Memory is JavaScript heap and object-URL accounting, not browser/native process RSS.'], runs: [], checks: [], exceptions };
let id = 0, readId = 0, sessionId, browserBuffer = '', pythonBuffer = '', pythonError = '', browserError = '', ending = false;
const delay = ms => new Promise(done => setTimeout(done, ms));
const deadline = setTimeout(() => { void fail(new Error('Ten-minute verification deadline exceeded')); }, 600_000);
function rejectAll(map, error) { for (const pending of map.values()) { clearTimeout(pending.timer); pending.reject(error); } map.clear(); }
python.stderr.on('data', chunk => { pythonError = (pythonError + chunk).slice(-20_000); });
browser.stderr.on('data', chunk => { browserError = (browserError + chunk).slice(-20_000); });
python.on('error', error => rejectAll(reads, error));
python.on('exit', code => { if (!ending) rejectAll(reads, new Error(`Python exited ${code}: ${pythonError}`)); });
browser.on('error', error => rejectAll(requests, error));
browser.on('exit', code => { if (!ending) rejectAll(requests, new Error(`Chromium exited ${code}: ${browserError}`)); });
python.stdin.on('error', error => { if (!ending) rejectAll(reads, error); });
browser.stdio[3].on('error', error => { if (!ending) rejectAll(requests, error); });
python.stdout.on('data', chunk => {
  pythonBuffer += chunk;
  let end;
  while ((end = pythonBuffer.indexOf('\n')) >= 0) {
    const line = pythonBuffer.slice(0, end); pythonBuffer = pythonBuffer.slice(end + 1);
    let response;
    try { response = JSON.parse(line); } catch { pythonError = (pythonError + '\n' + line).slice(-20_000); continue; }
    const pending = reads.get(response.id);
    if (!pending) continue;
    reads.delete(response.id); clearTimeout(pending.timer);
    transport.push({ ...pending.request, elapsedMs: response.elapsedMs, roundtripMs: performance.now() - pending.startedAt, bytes: response.bytes, error: response.error });
    pending.resolve(response);
  }
});
browser.stdio[4].on('data', chunk => {
  browserBuffer += chunk;
  let end;
  while ((end = browserBuffer.indexOf('\0')) >= 0) {
    const message = JSON.parse(browserBuffer.slice(0, end)); browserBuffer = browserBuffer.slice(end + 1);
    if (message.id) {
      const pending = requests.get(message.id); if (!pending) continue;
      requests.delete(message.id); clearTimeout(pending.timer);
      message.error ? pending.reject(new Error(JSON.stringify(message.error))) : pending.resolve(message.result);
    } else if (message.method === 'Runtime.bindingCalled') {
      const request = JSON.parse(message.params.payload);
      void nativeRead(request).then(response => evaluate(`window.__resolveRealRead(${JSON.stringify(request.id)},${JSON.stringify(response)})`), error => evaluate(`window.__resolveRealRead(${JSON.stringify(request.id)},${JSON.stringify({ error: { message: String(error), code: 'BRIDGE_ERROR', status: 500 } })})`)).catch(error => { if (!ending) exceptions.push(String(error)); });
    } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') exceptions.push(message.params.args);
  }
});
function cdp(method, params = {}, session = sessionId) {
  const requestId = ++id;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { requests.delete(requestId); reject(new Error(`CDP timed out: ${method}`)); }, 45_000);
    requests.set(requestId, { resolve, reject, timer });
    browser.stdio[3].write(JSON.stringify({ id: requestId, method, params, ...(session ? { sessionId: session } : {}) }) + '\0');
  });
}
function nativeRead(input) {
  const request = { ...input, id: ++readId };
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { reads.delete(request.id); reject(new Error(`Native request timed out: ${request.id}`)); }, 45_000);
    reads.set(request.id, { resolve, reject, timer, request, startedAt: performance.now() });
    python.stdin.write(JSON.stringify(request) + '\n');
  });
}
async function evaluate(expression) {
  const response = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function until(name, expression, timeout = 40_000) {
  const start = performance.now();
  while (performance.now() - start < timeout) {
    const value = await evaluate(expression);
    if (value) return value;
    await delay(35);
  }
  throw new Error(`${name} timed out: ${await evaluate('document.body.innerText')}`);
}
function pass(name) { results.checks.push(name); console.log('PASS', name); }
async function persist() { await writeFile(join(output, 'results.json'), JSON.stringify({ ...results, transportRequests: transport }, null, 2)); }
async function screenshot(name) {
  const recording = await evaluate('window.__recordFrames');
  await evaluate('window.__recordFrames=false');
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
  await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
  if (recording) await evaluate('window.__recordFrames=true');
}
async function stop() {
  if (ending) return; ending = true; clearTimeout(deadline);
  python.stdin.end();
  try { await cdp('Browser.close', {}, null); } catch { browser.kill(); }
  const killTimer = setTimeout(() => python.kill('SIGTERM'), 4000); killTimer.unref();
}
async function fail(error) {
  if (ending) return;
  results.failure = String(error.stack ?? error); results.pythonStderr = pythonError; results.browserStderr = browserError;
  try { await screenshot('failure'); } catch {}
  await persist(); console.error(results.failure); process.exitCode = 1; await stop();
}
const svgSelector = 'svg[aria-label^="Exact slide "]';
const bootstrap = `(() => {
  const pending = new Map(); let id = 0;
  window.readRealSlide = request => new Promise(resolve => { const key=++id; pending.set(key,resolve); window.__readRealSlideNative(JSON.stringify({...request,id:key})); });
  window.__resolveRealRead = (id,response) => { pending.get(id)?.(response); pending.delete(id); };
  const active = new Map(), originalCreate = URL.createObjectURL.bind(URL), originalRevoke = URL.revokeObjectURL.bind(URL), decode = HTMLImageElement.prototype.decode;
  URL.createObjectURL = value => {const url=originalCreate(value);const entry={bytes:value.size??0,width:0,height:0};active.set(url,entry);if(value instanceof Blob)void value.slice(0,24).arrayBuffer().then(buffer=>{const data=new DataView(buffer);if(data.byteLength>=24&&data.getUint32(0)===0x89504e47){entry.width=data.getUint32(16);entry.height=data.getUint32(20);}});return url;};
  URL.revokeObjectURL = url => { active.delete(url); originalRevoke(url); };
  HTMLImageElement.prototype.decode = async function(...args) { const result=await decode.apply(this,args);const entry=active.get(this.src);if(entry){entry.width=this.naturalWidth;entry.height=this.naturalHeight;}return result; };
  window.__objects = () => {const values=[...active.values()];return{count:values.length,encodedBytes:values.reduce((s,v)=>s+v.bytes,0),decodedEstimateBytes:values.reduce((s,v)=>s+v.width*v.height*4,0),maxSide:Math.max(0,...values.map(v=>Math.max(v.width,v.height)))}};
  window.__frameGaps=[];window.__inputLatency=[];window.__recordFrames=false;let previous;
  const frame=time=>{if(window.__recordFrames&&previous!=null)window.__frameGaps.push(time-previous);previous=time;requestAnimationFrame(frame);};requestAnimationFrame(frame);
  window.addEventListener('wheel',event=>{const svg=document.querySelector(${JSON.stringify(svgSelector)});if(!svg||!svg.contains(event.target))return;const started=performance.now(),before=svg.getAttribute('viewBox');let frames=0;const poll=()=>{if(svg.getAttribute('viewBox')!==before){window.__inputLatency.push(performance.now()-started);return;}if(++frames<15)requestAnimationFrame(poll);};requestAnimationFrame(poll);},{capture:true});
  window.__readyDetail = () => {
    const svg=document.querySelector(${JSON.stringify(svgSelector)});if(!svg)return false;
    const v=svg.viewBox.baseVal,b=svg.getBoundingClientRect(),meta=window.__meta;
    const wanted=window.realSlideTest.plan({x:v.x,y:v.y,width:v.width,height:v.height},meta.width,meta.height,b.width,b.height,devicePixelRatio);
    const present=new Set([...svg.querySelectorAll('image[data-slide-tile]')].map(image=>image.getAttribute('data-slide-tile')));
    return wanted.length>0&&wanted.every(tile=>present.has(tile.key));
  };
})();`;
function summary(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = p => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : null;
  return { count: values.length, p50Ms: quantile(.5), p95Ms: quantile(.95), maxMs: sorted.at(-1) ?? null, over25Ms: values.filter(v => v > 25).length, over50Ms: values.filter(v => v > 50).length };
}
async function camera() { return evaluate(`(() => {const svg=document.querySelector(${JSON.stringify(svgSelector)}),v=svg.viewBox.baseVal,b=svg.getBoundingClientRect();return{view:{x:v.x,y:v.y,width:v.width,height:v.height},box:{x:b.x,y:b.y,width:b.width,height:b.height}};})()`); }
async function click(label) { await evaluate(`(() => {const button=[...document.querySelectorAll('button')].find(el=>el.textContent.trim()===${JSON.stringify(label)});if(!button||button.disabled)throw Error('Button unavailable: '+${JSON.stringify(label)});button.click();})()`); }
let lastDetailTiming;
async function targetRegion(region) {
  await evaluate(`window.realSlideTest.selectRegion(${JSON.stringify(region)})`);
  await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
  const start = await evaluate(`(() => {
    const svg=document.querySelector(${JSON.stringify(svgSelector)}),box=svg.getBoundingClientRect(),region=${JSON.stringify(region)};
    const wanted=new Set(window.realSlideTest.plan(region,window.__meta.width,window.__meta.height,box.width,box.height,devicePixelRatio).map(tile=>tile.key));
    const before=new Set([...svg.querySelectorAll('image[data-slide-tile]')].map(image=>image.getAttribute('data-slide-tile')));
    const probe={start:performance.now(),firstAnyTileMs:null,firstNewTileMs:null,wantedCount:wanted.size};
    const observe=()=>{const v=svg.viewBox.baseVal;if(Math.abs(v.x-region.x)>1||Math.abs(v.y-region.y)>1||Math.abs(v.width-region.width)>1||Math.abs(v.height-region.height)>1)return;const present=[...svg.querySelectorAll('image[data-slide-tile]')].map(image=>image.getAttribute('data-slide-tile')).filter(key=>wanted.has(key));if(present.length&&probe.firstAnyTileMs===null)probe.firstAnyTileMs=performance.now()-probe.start;if(present.some(key=>!before.has(key))&&probe.firstNewTileMs===null)probe.firstNewTileMs=performance.now()-probe.start;};
    window.__detailObserver?.disconnect();window.__detailObserver=new MutationObserver(observe);window.__detailObserver.observe(svg,{childList:true,subtree:true,attributes:true});window.__detailProbe=probe;return probe.start;
  })()`);
  await click('Zoom to selection');
  await until('Region committed', `(() => {const v=document.querySelector(${JSON.stringify(svgSelector)}).viewBox.baseVal;return Math.abs(v.width-${region.width})<1&&Math.abs(v.height-${region.height})<1&&Math.abs(v.x-${region.x})<1&&Math.abs(v.y-${region.y})<1;})()`);
  await until('Sharp visible detail', 'window.__readyDetail()');
  lastDetailTiming = await evaluate('(window.__detailObserver.disconnect(),{...window.__detailProbe,completeMs:performance.now()-window.__detailProbe.start})');
  return (await evaluate('performance.now()')) - start;
}
async function nativeIdle() {
  const started = performance.now();
  while (reads.size && performance.now() - started < 45_000) await delay(50);
  assert.equal(reads.size, 0, 'Native reads must finish before resetting the reader cache');
}
async function run(meta, dpr) {
  await evaluate('window.realSlideTest.unmount()'); await delay(100); await nativeIdle(); await nativeRead({ op: 'reset' });
  await cdp('Emulation.setDeviceMetricsOverride', { width: viewportWidth, height: viewportHeight, deviceScaleFactor: dpr, mobile: false });
  const start = await evaluate(`(window.__meta=${JSON.stringify(meta)},window.realSlideTest.load(window.__meta,{dense:true}),performance.now())`);
  const label = `${meta.slideIndex}-${meta.backend}-dpr${dpr}`;
  const run = { slideIndex: meta.slideIndex, label: meta.label, backend: meta.backend, width: meta.width, height: meta.height, fileBytes: meta.fileBytes, dpr, denseOverlay: true, screenshots: [] };
  results.runs.push(run);
  await until('Overview appears', `document.querySelector(${JSON.stringify(svgSelector + ' image:not([data-slide-tile]):not([data-quality-overlay])')})?.getAttribute('href')`);
  run.overviewMs = (await evaluate('performance.now()')) - start;
  await until('Slide prepared', `(() => {const layer=document.querySelector('[data-slide-tile-layer]');return layer&&layer.getAttribute('data-preparing')==='false'&&Number(layer.getAttribute('data-prepared-tiles'))===Number(layer.getAttribute('data-preparation-total'))&&!document.querySelector('[data-quality-overlay-layer][data-overlay-loading="true"]');})()`, 90_000);
  run.preparationMs = (await evaluate('performance.now()')) - start;
  run.preparationTiles = await evaluate(`Number(document.querySelector('[data-slide-tile-layer]').getAttribute('data-prepared-tiles'))`);
  assert.ok(run.preparationTiles > 0 && run.preparationTiles <= 128);
  pass(`${label}: overview + all ${run.preparationTiles} preparation tiles loaded`);
  await screenshot(label + '-fitted'); run.screenshots.push(label + '-fitted.png');
  const fitted = await camera(); run.viewportCSS = fitted.box;
  const center = meta.tissueCenters[0];
  const width = 2048, height = Math.round(width * fitted.box.height / fitted.box.width);
  const region = { x: Math.max(0, Math.min(meta.width - width, center.x - width / 2)), y: Math.max(0, Math.min(meta.height - height, center.y - height / 2)), width, height };
  run.tissueRegion = region;
  await evaluate('window.__frameGaps=[];window.__inputLatency=[];window.__recordFrames=true');
  run.firstTissueDetailMs = await targetRegion(region);
  run.firstTissueTiming = lastDetailTiming;
  await until('QC overlay finished', `!document.querySelector('[data-quality-overlay-layer][data-overlay-loading="true"]')`);
  const atDetail = await camera();
  run.detailTileLevel = await evaluate(`Number(document.querySelector('image[data-slide-tile]')?.getAttribute('data-tile-level'))`);
  run.detailTileCount = await evaluate(`document.querySelectorAll('image[data-slide-tile]').length`);
  run.detailPhysicalPixelsPerImagePixel = Math.min(atDetail.box.width / atDetail.view.width, atDetail.box.height / atDetail.view.height) * dpr * 2 ** run.detailTileLevel;
  run.visibleTileLimit = await evaluate('window.realSlideTest.maxVisibleTiles??32');
  assert.ok(run.detailTileCount <= run.visibleTileLimit);
  pass(`${label}: first tissue detail is complete at actual viewport DPR`);
  await screenshot(label + '-tissue'); run.screenshots.push(label + '-tissue.png');
  if (meta.slideIndex === 2 && dpr === 2) {
    const probe = { x: 57120, y: 16320, width: 7616, height: 2176 };
    run.wideTiffProbe = { region: probe, sharpDetailMs: await targetRegion(probe) };
    run.wideTiffProbe.timing = lastDetailTiming;
    const view = await camera();
    run.wideTiffProbe.viewport = view;
    run.wideTiffProbe.plan = await evaluate(`window.realSlideTest.plan(${JSON.stringify(probe)},${meta.width},${meta.height},${view.box.width},${view.box.height},2).map(tile=>({key:tile.key,level:tile.level}))`);
    run.wideTiffProbe.physicalPixelsPerImagePixel = Math.min(view.box.width / probe.width, view.box.height / probe.height) * 2 * 2 ** run.wideTiffProbe.plan[0].level;
    if(run.visibleTileLimit===64) {assert.equal(run.wideTiffProbe.plan[0].level,1);assert.equal(run.wideTiffProbe.plan.length,36);assert.ok(run.wideTiffProbe.physicalPixelsPerImagePixel<=1.05);pass(`${label}: wide TIFF retains exact screen-resolution level across 36 visible tiles`);}
    await until('Wide QC overlay finished', `!document.querySelector('[data-quality-overlay-layer][data-overlay-loading="true"]')`);
    await screenshot(label + '-wide-clarity'); run.screenshots.push(label + '-wide-clarity.png');
    if (run.visibleTileLimit===64) {
      const tissueProbe={x:111552,y:25536,width:7616,height:2176};
      run.tissueClarityProbe={region:tissueProbe,sharpDetailMs:await targetRegion(tissueProbe),timing:lastDetailTiming};
      run.tissueClarityProbe.plan=await evaluate(`window.realSlideTest.plan(${JSON.stringify(tissueProbe)},${meta.width},${meta.height},${view.box.width},${view.box.height},2).map(tile=>({key:tile.key,level:tile.level}))`);
      assert.equal(run.tissueClarityProbe.plan[0].level,1);assert.equal(run.tissueClarityProbe.plan.length,36);
      await until('Tissue clarity overlay finished', `!document.querySelector('[data-quality-overlay-layer][data-overlay-loading="true"]')`);
      await screenshot(label+'-wide-tissue-clarity');run.screenshots.push(label+'-wide-tissue-clarity.png');
    }
    await targetRegion(region);
  }
  const point = { x: Math.round(atDetail.box.x + atDetail.box.width * .45), y: Math.round(atDetail.box.y + atDetail.box.height * .5) };
  const wheelStart = await evaluate('performance.now()');
  await cdp('Input.dispatchMouseEvent', { type: 'mouseWheel', ...point, deltaX: 0, deltaY: -120, modifiers: 0 });
  await delay(170); await until('Wheel sharp detail', 'window.__readyDetail()');
  run.wheelSharpDetailMs = (await evaluate('performance.now()')) - wheelStart;
  assert.ok((await camera()).view.width < region.width);
  pass(`${label}: trusted wheel zoom updates camera and reaches sharp tiles`);
  // Interrupt a real coarse-wheel animation with drag. The camera scale must not jump.
  await cdp('Input.dispatchMouseEvent', { type: 'mouseWheel', ...point, deltaX: 0, deltaY: -120, modifiers: 0 });
  await evaluate('new Promise(requestAnimationFrame)');
  await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', ...point, button: 'left', clickCount: 1 });
  const pressed = (await camera()).view;
  await delay(80);
  assert.ok(Math.abs((await camera()).view.width - pressed.width) < .01);
  await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', x: point.x + 35, y: point.y + 18, button: 'left', buttons: 1 });
  await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', x: point.x + 35, y: point.y + 18, button: 'left', clickCount: 1 });
  await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
  const dragged = (await camera()).view;
  run.wheelDragScaleDelta = Math.abs(dragged.width - pressed.width);
  assert.ok(run.wheelDragScaleDelta < .01 && Math.abs(dragged.x - pressed.x) > 1);
  pass(`${label}: wheel-to-drag handoff preserves scale`);
  await targetRegion(region);
  // Alternate one screen-width pan and a return to the exact previously loaded tissue view.
  const other = { ...region, x: Math.max(0, Math.min(meta.width - region.width, region.x + region.width * .8)) };
  run.neighborDetailMs = await targetRegion(other);
  const callsBeforeRevisit = await evaluate('window.realSlideTest.calls.length');
  run.cachedRevisitMs = await targetRegion(region);
  run.requestsDuringCachedRevisit = (await evaluate('window.realSlideTest.calls.length')) - callsBeforeRevisit;
  assert.ok(run.cachedRevisitMs < 250, 'A recent fully cached view should paint promptly');
  pass(`${label}: exact cached revisit paints without waiting for a native read`);
  // Bounded high-frequency pan and alternating wheel inputs; the overview must remain present.
  await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', ...point, button: 'left', clickCount: 1 });
  for (let step = 0; step < 28; step++) {
    await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', x: point.x + Math.sin(step / 4) * 160, y: point.y + Math.cos(step / 5) * 80, button: 'left', buttons: 1 });
    await delay(12);
  }
  await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', ...point, button: 'left', clickCount: 1 });
  for (let step = 0; step < 8; step++) {
    await cdp('Input.dispatchMouseEvent', { type: 'mouseWheel', ...point, deltaX: 0, deltaY: step < 4 ? -30 : 30, modifiers: 0 });
    await delay(18);
  }
  await delay(180); await until('Settled gesture detail', 'window.__readyDetail()');
  const final = (await camera()).view;
  assert.ok(Object.values(final).every(Number.isFinite) && final.width > 0 && final.x >= -1 && final.y >= -1 && final.x + final.width <= meta.width + 1 && final.y + final.height <= meta.height + 1);
  assert.ok(await evaluate(`Boolean(document.querySelector(${JSON.stringify(svgSelector + ' image:not([data-slide-tile]):not([data-quality-overlay])')}))`));
  const metrics = await evaluate(`(window.__recordFrames=false,{frames:window.__frameGaps,latency:window.__inputLatency,objects:window.__objects(),errors:window.realSlideTest.errors,calls:window.realSlideTest.calls,overlayImages:document.querySelectorAll('image[data-quality-overlay]').length,selectedROIs:document.querySelectorAll('rect[stroke-dasharray]').length})`);
  run.frameGaps = summary(metrics.frames); run.inputToCamera = summary(metrics.latency); run.objects = metrics.objects;
  run.calls = metrics.calls; run.overlayImages = metrics.overlayImages; run.heap = await cdp('Runtime.getHeapUsage');
  assert.equal(metrics.errors.length, 0, 'Real-slide read errors'); assert.ok(metrics.overlayImages <= 2);
  assert.ok(metrics.objects.count <= 200, 'Tile cache + bounded overview/overlay URLs');
  assert.ok(metrics.objects.encodedBytes + metrics.objects.decodedEstimateBytes < 300 * 1024 * 1024, 'Viewer URL memory remains bounded including raster overlay/overview');
  pass(`${label}: rapid input settles, overlay stays bounded, no reader errors`);
  console.log('METRICS', JSON.stringify({ label, preparationMs: run.preparationMs, firstDetailMs: run.firstTissueDetailMs, cachedRevisitMs: run.cachedRevisitMs, frameGaps: run.frameGaps, inputToCamera: run.inputToCamera }));
  if (meta.slideIndex === 3 && dpr === 2 && process.env.HISTOPILOT_SKIP_SCAN !== '1') {
    const scanStart = performance.now(), samples = [], scanFramesStart = await evaluate('window.__frameGaps.length');
    await evaluate('window.__recordFrames=true');
    let step = 0;
    while (performance.now() - scanStart < 60_000) {
      const tissue = meta.tissueCenters[step % meta.tissueCenters.length];
      const offset = Math.floor(step / meta.tissueCenters.length) % 7;
      const scanRegion = { ...region, x: Math.max(0, Math.min(meta.width-region.width, tissue.x-region.width/2+offset*region.width*.75)), y: Math.max(0, Math.min(meta.height-region.height, tissue.y-region.height/2+Math.sin(step)*region.height)) };
      const sharpMs = await targetRegion(scanRegion);
      const position = await camera(), anchor = {x:Math.round(position.box.x+position.box.width*.5),y:Math.round(position.box.y+position.box.height*.5)};
      for(let wheel=0;wheel<4;wheel++) { await cdp('Input.dispatchMouseEvent',{type:'mouseWheel',...anchor,deltaX:0,deltaY:wheel<2?-40:40,modifiers:0}); await delay(30); }
      await delay(160);
      samples.push({step,sharpMs,objects:await evaluate('window.__objects()')});
      step++;
    }
    await until('Sustained scan settles', 'window.__readyDetail()');
    const scan = await evaluate(`(window.__recordFrames=false,{frames:window.__frameGaps.slice(${scanFramesStart}),objects:window.__objects(),errors:window.realSlideTest.errors})`);
    run.sustainedScan = { durationMs: performance.now()-scanStart, regions: samples.length, detail: summary(samples.map(sample=>sample.sharpMs)), frames: summary(scan.frames), maxObjectURLs: Math.max(...samples.map(sample=>sample.objects.count)), maxImageBytes: Math.max(...samples.map(sample=>sample.objects.encodedBytes+sample.objects.decodedEstimateBytes)), finalObjects: scan.objects, samples };
    assert.equal(scan.errors.length,0);assert.ok(run.sustainedScan.maxObjectURLs<=200);assert.ok(run.sustainedScan.maxImageBytes<300*1024*1024);
    pass(`${label}: 60-second mixed scan stays bounded without reader errors`);
    console.log('SCAN',JSON.stringify({...run.sustainedScan,samples:undefined}));
  }
  await evaluate('window.realSlideTest.unmount()'); await delay(150); await nativeIdle();
  run.objectsAfterUnmount = await evaluate('window.__objects()');
  assert.equal(run.objectsAfterUnmount.count, 0, 'Unmount releases all overview, tile, and QC image URLs');
  pass(`${label}: unmount releases all decoded image URLs`);
  await persist();
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Runtime.addBinding', { name: '__readRealSlideNative' });
  await cdp('Page.addScriptToEvaluateOnNewDocument', { source: bootstrap });
  await cdp('Page.navigate', { url: pathToFileURL(join(output, 'index.html')).href });
  await until('Real-slide fixture ready', 'window.realSlideTest?.ready');
  for (const meta of slides) for (const dpr of dprs) await run(meta, dpr);
  assert.equal(exceptions.length, 0, 'No browser exceptions or console errors');
  pass('All real-slide runs: no browser exceptions or console errors');
  results.completedAt = new Date().toISOString(); await persist(); await stop();
} catch (error) { await fail(error); }
