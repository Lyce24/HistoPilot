/** Exercise the morphology explorer in local Chromium, offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const output = resolve(process.argv[2] ?? '/tmp/histopilot-morphology-browser');
const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter(name => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
const requests = new Map();
const exceptions = [];
const checks = [];
const roiEvidence = [];
let nextId = 0, buffer = '', stderr = '', sessionId;
browser.stderr.on('data', data => { stderr += data.toString(); });
const rejectPending = error => { for (const request of requests.values()) request.reject(error); requests.clear(); };
browser.on('error', rejectPending);
browser.on('exit', code => rejectPending(new Error(`Chromium exited (${code}): ${stderr}`)));
browser.stdio[3].on('error', rejectPending);
browser.stdio[4].on('error', rejectPending);
browser.stdio[4].on('data', data => {
  buffer += data.toString();
  let end;
  while ((end = buffer.indexOf('\0')) >= 0) {
    const message = JSON.parse(buffer.slice(0, end));
    buffer = buffer.slice(end + 1);
    if (message.id) {
      const request = requests.get(message.id);
      requests.delete(message.id);
      if (message.error) request?.reject(new Error(JSON.stringify(message.error)));
      else request?.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') exceptions.push(message.params.args);
  }
});
function cdp(method, params = {}, session = sessionId) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    requests.set(id, { resolve, reject });
    browser.stdio[3].write(JSON.stringify({ id, method, params, ...(session ? { sessionId: session } : {}) }) + '\0');
  });
}
async function evaluate(expression) {
  const response = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function check(name, expression) {
  for (let attempt = 0; attempt < 80; attempt++) {
    if (await evaluate('Boolean(' + expression + ')')) {
      checks.push(name);
      console.log('PASS', name);
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(name + ': ' + await evaluate('document.body.innerText'));
}
async function click(name) {
  await evaluate(`(() => { const button = [...document.querySelectorAll('button')].find(el => el.textContent.trim() === ${JSON.stringify(name)}); if (!button || button.disabled) throw Error('Button unavailable'); button.click(); })()`);
}
async function screenshot(name) {
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable');
  await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(output, 'index.html')).href });
  await check('Explorer entry loads', "document.body?.innerText.includes('Open visual review & feature explorer')");
  await click('Open visual review & feature explorer');
  await check('Frozen slide and independent review load', "document.querySelector('[aria-label=\"Visual quality of a\"]') && document.body.innerText.includes('Slide review')");
  await check('Patch capabilities resolve before projection is enabled', "window.__morphologyCalls.some(x => x.path.endsWith('/configurations/patch-features')) && [...document.querySelectorAll('button')].some(x => x.textContent === 'Build morphology projection' && !x.disabled)");
  await click('Build morphology projection');
  await check('Projection and measured neighbors appear', "document.querySelector('[aria-label=\"Slide morphology PCA projection\"]') && document.body.innerText.includes('0.9200')");
  await evaluate(`(() => { const field = [...document.querySelectorAll('label')].find(x => x.textContent.startsWith('Color by metadata')).querySelector('select'); field.value = 'site'; field.dispatchEvent(new Event('change', {bubbles:true})); })()`);
  await check('Metadata groups linked to projection points', "document.querySelector('[aria-label=\"Inspect a, Site 1\"]') && document.body.innerText.includes('Site 2')");
  await click('Inspect slide');
  await check('Neighbor opens its exact slide', "document.querySelector('[aria-label=\"Visual quality of b\"]') && window.__morphologyCalls.some(x=>x.path.includes('/image?') && x.path.includes('slideId=b'))");
  await evaluate(`(() => { const field = [...document.querySelectorAll('label')].find(x => x.textContent.startsWith('Compare')).querySelector('select'); field.value = 'patch'; field.dispatchEvent(new Event('change', {bubbles:true})); })()`);
  await check('Sampled patch retrieval is labeled', "document.body.innerText.includes('Search covers only sampled patches')");
  await click('View query patch');
  await check('Selected patch obtains exact geometry and crop', "document.body.innerText.includes('Original patch 0') && document.body.innerText.includes('Selected region: (200, 150)')");
  await click('Zoom to selection');
  await check('Zoom requests bounded original image region', "window.__morphologyCalls.some(x=>x.path.includes('/image?') && x.path.includes('x=200') && x.path.includes('width=150'))");
  await click('Save selected region with review');
  await evaluate(`(() => { const field = [...document.querySelectorAll('label')].find(x => x.textContent.startsWith('Decision')).querySelector('select'); field.value = 'review'; field.dispatchEvent(new Event('change', {bubbles:true})); })()`);
  await click('Save review');
  await check('Review saves revision and exact selected coordinates', "window.__morphologyCalls.some(x=>x.body?.status === 'review' && x.body?.expectedRevision === 0 && x.body?.regions[0]?.x === 200 && x.body?.regions[0]?.width === 150) && document.body.innerText.includes('Review saved · revision 1')");
  await click('Fit slide');
  await check('Full slide image restored', "!document.querySelector('button[disabled]') || window.__morphologyCalls.filter(x=>x.path.includes('/image?')).at(-1).path.includes('max_size=1536')");
  await click('Draw review region');
  await evaluate("document.querySelector('svg[aria-label=\"Exact slide b with patch coverage\"]').scrollIntoView({block:'center'})");
  const box = await evaluate(`(() => {const rect=document.querySelector('svg[aria-label="Exact slide b with patch coverage"]').getBoundingClientRect(); return {x:rect.x,y:rect.y,width:rect.width,height:rect.height};})()`);
  await cdp('Input.dispatchMouseEvent', { type:'mousePressed', x:box.x+box.width*.1, y:box.y+box.height*.1, button:'left', clickCount:1 });
  await cdp('Input.dispatchMouseEvent', { type:'mouseMoved', x:box.x+box.width*.3, y:box.y+box.height*.3, button:'left', buttons:1 });
  await cdp('Input.dispatchMouseEvent', { type:'mouseReleased', x:box.x+box.width*.3, y:box.y+box.height*.3, button:'left', clickCount:1 });
  await check('Drawn region survives the concluding patch click', "document.body.innerText.includes('Selected region: (') && !document.body.innerText.includes('Selected region: (200, 150)') && !document.body.innerText.includes('Drag a rectangle')");
  await screenshot('morphology-desktop');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await check('Explorer fits mobile viewport', 'document.documentElement.scrollWidth <= innerWidth');
  await screenshot('morphology-mobile');
  for (const mode of ['slide', 'broken']) {
    await cdp('Emulation.setDeviceMetricsOverride', { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
    await cdp('Page.navigate', { url: pathToFileURL(join(output, 'index.html')).href + '?mode=' + mode });
    await check(`${mode}: explorer entry loads`, "document.body?.innerText.includes('Open visual review & feature explorer')");
    await click('Open visual review & feature explorer');
    await check(`${mode}: original image uses exact level-0 geometry`, `document.querySelector('svg[aria-label="Exact slide a"]')?.getAttribute('viewBox') === '0 0 6000 4000' && document.querySelector('svg[aria-label="Exact slide a"] image')?.getAttribute('href')?.startsWith('blob:')`);
    if (mode === 'slide') {
      await check('Slide vectors expose image review without patch tools', "document.body.innerText.includes('One embedding per slide') && !document.body.innerText.includes('Build morphology projection') && !document.body.innerText.includes('Patch coverage') && !document.body.innerText.includes('Inspect exact patch index')");
      await check('Slide kind resolves from frozen feature configuration', "window.__morphologyCalls.some(x => x.path.endsWith('/configurations/slide-features')) && !window.__morphologyCalls.some(x => x.path.includes('/morphology/index'))");
    } else {
      await check('Optional coverage failure leaves original image and notes available', "document.body.innerText.includes('Optional feature coverage is unavailable') && document.querySelector('textarea') && document.querySelector('svg[aria-label=\"Exact slide a\"] image') && window.__morphologyCalls.some(x=>x.path.includes('/quality?') && !x.path.includes('featureBundleId'))");
    }
    await click('Draw review region');
    const draw = await evaluate(`(() => {
      const svg = document.querySelector('svg[aria-label="Exact slide a"]');
      svg.scrollIntoView({block:'center'});
      const matrix = svg.getScreenCTM();
      const start = new DOMPoint(1000, 500).matrixTransform(matrix);
      const end = new DOMPoint(3000, 1500).matrixTransform(matrix);
      const first = start.matrixTransform(matrix.inverse()), last = end.matrixTransform(matrix.inverse());
      return {start:{x:start.x,y:start.y},end:{x:end.x,y:end.y},region:{x:Math.floor(first.x),y:Math.floor(first.y),width:Math.floor(last.x-first.x),height:Math.floor(last.y-first.y)}};
    })()`);
    await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', ...draw.start, button: 'left', clickCount: 1 });
    await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', ...draw.end, button: 'left', buttons: 1 });
    await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', ...draw.end, button: 'left', clickCount: 1 });
    await check(`${mode}: drawn ROI uses full-resolution coordinates`, `document.body.innerText.includes('Selected region: (') && [...document.querySelectorAll('button')].some(x => x.textContent === 'Save selected region with review')`);
    await click('Save selected region with review');
    await click('Save review');
    await check(`${mode}: ROI saves successfully without patch coordinates`, "document.body.innerText.includes('Review saved · revision 1') && window.__morphologyCalls.some(x=>x.body?.regions?.length === 1)");
    const saved = await evaluate("window.__morphologyCalls.find(x=>x.body?.regions?.length === 1).body.regions[0]");
    for (const key of ['x', 'y', 'width', 'height']) assert.ok(Math.abs(saved[key] - draw.region[key]) <= 1, `${mode} ${key} must preserve level-0 coordinates`);
    assert.ok(saved.x > 600 && saved.width > 600 && saved.height > 400, 'ROI must not use thumbnail pixels');
    const drawn = await evaluate("(() => {const rect=document.querySelector('svg[aria-label=\"Exact slide a\"] rect[pointer-events=\"none\"]');return Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(rect.getAttribute(key))]));})()");
    for (const key of ['x', 'y', 'width', 'height']) assert.equal(saved[key], drawn[key], `${mode}: saved ROI equals displayed geometry`);
    roiEvidence.push({ mode, thumbnail: [600, 400], level0: [6000, 4000], saved });
    checks.push(`${mode}: saved ROI exactly matches displayed level-0 rectangle`);
    await screenshot(`morphology-${mode}-desktop`);
    await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await check(`${mode}: review fits mobile viewport`, 'document.documentElement.scrollWidth <= innerWidth');
    await screenshot(`morphology-${mode}-mobile`);
  }
  assert.deepEqual(exceptions, [], 'No unexpected browser errors');
  await writeFile(join(output, 'checks.json'), JSON.stringify({ passed: true, checks, roiEvidence, exceptions }, null, 2) + '\n');
  console.log('Artifacts: ' + output);
} finally {
  browser.kill();
}
