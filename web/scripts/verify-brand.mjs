/** Verify shared branding in the real App and Start offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-brand-'));
const fixture = join(output, 'fixture.tsx');
const artifacts = resolve(web, '../docs/dev-review/2026-09-13-interface-consistency-assets');
await mkdir(artifacts, { recursive: true });
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import App from ${source('App.tsx')};
import { Brand, HistoPilotMark } from ${source('components/Brand.tsx')};
import { api } from ${source('api/client.ts')};
import { scientific } from ${source('api/scientific.ts')};
import { bundles } from ${source('api/bundles.ts')};
import { development } from ${source('api/development.ts')};
import { evaluation } from ${source('api/evaluation.ts')};
import { modelEvaluations, predictors } from ${source('api/predictors.ts')};
import { clinicalAnalyses } from ${source('api/clinicalUtility.ts')};
import { interpretations } from ${source('api/interpretation.ts')};
import { experiments } from ${source('api/experiments.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('roadmap.css')};
import ${source('components/StageWorkflow.css')};
const state = window.workflow = { errors: [], calls: [], loadAttempts: 0 };
window.fetch = async (...args) => { state.errors.push('Unexpected API request: ' + args[0]); throw new Error(state.errors.at(-1)); };
const project = { id: 'project', name: 'Lung cancer study', mode: 'local', lifecycleState: 'active', storagePath: '/fixture/project', description: '', available: true, createdAt: '2026-09-12T12:00:00Z', updatedAt: '2026-09-12T12:00:00Z', config: { seed: 42, folds: 2 }, sources: [] };
const workspace = { mode: 'local', project, dataset: { id: '', slideCount: 0 }, sources: [], drafts: [], featureSets: [], cohortSnapshots: [], encoders: [], milModels: [] };
api.projects = async () => ({ projects: [project], defaultStoragePath: '/fixture' });
api.projectWorkspace = async () => structuredClone(workspace);
api.openProject = async () => project;
scientific.datasets = async () => ({ datasets: [] });
scientific.drafts = async () => ({ drafts: [] });
scientific.configurations = async () => ({ configurations: [] });
for (const api of [bundles, evaluation, predictors, modelEvaluations, clinicalAnalyses, interpretations]) api.list = async () => ({ items: [] });
development.list = async () => ({ items: [], executions: [] });
predictors.refits = async () => ({ items: [] });
experiments.summaries = async () => ({ items: [] });

const dataset = { id: 'dataset', projectId: 'project', contentHash: 'fixture', createdAt: '', manifest: {}, artifacts: {} };
scientific.datasets = async () => ({ datasets: [dataset] });
scientific.configurations = async (_, kind) => ({ configurations: kind === 'protocol' ? [{ id: 'protocol', manifest: { kind: 'protocol', datasetId: 'dataset', spec: { datasetId: 'dataset' } } }] : [] });
bundles.list = async () => ({ items: [{ id: 'bundle', current: true, findings: [], manifest: { datasetId: 'dataset', spec: { featureSetId: 'features', packArtifactIds: [] }, feature: { validation: { tensorValidationComplete: true } }, packs: [] } }] });
predictors.list = async () => ({ items: [{ id: 'predictor', lifecycleState: 'active', manifest: { recipe: { model: 'ABMIL' } } }] });
scientific.drafts = async () => ({ drafts: [{ id: 'draft', payload: { type: 'evaluation-cohort' } }] });
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
window.showEmptyRoadmap = () => {
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
  client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
  client.setQueryData(['feature-bundles', 'project'], { items: [] });
  client.setQueryData(['predictors', 'project'], { items: [] });
};
const root = createRoot(document.getElementById('app'));
root.render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
window.showBrandProof = () => root.render(<div style={{ background: '#f6f7f8', padding: '24px 0' }}>
  {(['default', 'inverse'] as const).map(tone => <section key={tone} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 40, padding: '48px 40px', background: tone === 'inverse' ? '#292833' : '#ffffff', color: tone === 'inverse' ? '#f7f6f4' : '#292833' }}>
    <div style={{ display: 'flex', gap: 32, alignItems: 'center' }}>{[16, 24, 32, 40, 48, 96].map(size => <div key={size} style={{ display: 'grid', justifyItems: 'center', gap: 16 }}><HistoPilotMark size={size} /><span style={{fontSize: 11}}>{size} px</span></div>)}</div>
    <Brand size={40} subtitle="Pathology workspace" tone={tone} />
  </section>)}
</div>);

`);
await build({ configFile: false, root: web, base: './', logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } },
  build: { outDir: join(output, 'dist'), emptyOutDir: true, minify: true,
    lib: { entry: fixture, formats: ['es'], fileName: () => 'fixture.js' },
  },
});
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
  + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('')
  + '</head><body><div id="app"></div><script type="module" src="fixture.js"></script></body></html>');

const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter((name) => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const browser = spawn(executable, ['--no-sandbox', '--allow-file-access-from-files', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')],
  { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
let stderr = '';
browser.stderr.on('data', (data) => { stderr += data.toString(); });
const requests = new Map();
let nextId = 0, buffer = '', sessionId;
const exceptions = [];
const downloads = [];
const rejectPending = (error) => { for (const { reject } of requests.values()) reject(error); requests.clear(); };
browser.on('error', rejectPending);
browser.on('exit', (code) => rejectPending(new Error('Chromium exited (' + code + '): ' + stderr)));
browser.stdio[3].on('error', rejectPending);
browser.stdio[4].on('error', rejectPending);
browser.stdio[4].on('data', (data) => {
  buffer += data.toString();
  let end;
  while ((end = buffer.indexOf('\0')) >= 0) {
    const message = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1);
    if (message.id) {
      const pending = requests.get(message.id); requests.delete(message.id);
      if (message.error) pending?.reject(new Error(JSON.stringify(message.error))); else pending?.resolve(message.result);
    } else if (message.method === 'Network.requestWillBeSent') downloads.push(message.params.request.url);
    else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
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
async function waitFor(expression, description = expression) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await evaluate('Boolean(' + expression + ')')) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('Timed out: ' + description + '\n' + await evaluate('document.body.innerText'));
}
async function screenshot(name) {
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  await evaluate('Promise.allSettled(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished))');
  const { cssContentSize } = await cdp('Page.getLayoutMetrics');
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width: cssContentSize.width, height: cssContentSize.height, scale: 1 } });
  await writeFile(join(artifacts, name + '.png'), Buffer.from(data, 'base64'));
}
const layouts = [];
async function verifyLayout(kind, width) {
  const selector = kind === 'start' ? '.start-brand' : '.sidebar .brand';
  const value = await evaluate(`(() => {
    const brand = document.querySelector(${JSON.stringify(selector)});
    const svg = brand.querySelector('svg');
    const word = brand.querySelector('.hp-brand__wordmark');
    const subtitle = brand.querySelector('.hp-brand__subtitle');
    const container = brand.closest('.sidebar') ?? brand.parentElement;
    const rect = svg.getBoundingClientRect(), text = word.getBoundingClientRect(), edge = container.getBoundingClientRect();
    return {
      kind: ${JSON.stringify(kind)}, width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
      svgWidth: rect.width, svgHeight: rect.height, paths: svg.querySelectorAll('path').length,
      images: brand.querySelectorAll('img').length, color: getComputedStyle(word).color,
      wordVisible: text.width > 0 && text.height > 0 && getComputedStyle(word).visibility === 'visible',
      fits: Math.max(text.right, subtitle.getBoundingClientRect().right, rect.right) <= edge.right + 1 && rect.left >= edge.left,
      label: brand.textContent, svgHidden: svg.getAttribute('aria-hidden'),
      svgPosition: {left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom},
      sidebarInert: document.querySelector('.sidebar')?.inert ?? false
    };
  })()`);
  assert.equal(value.width, width);
  assert.ok(value.scrollWidth <= width + 1, `${kind} overflows at ${width}px`);
  // The ring and the two needle halves: a small vector, never a raster asset.
  assert.equal(value.paths, 3);
  assert.equal(value.images, 0);
  assert.equal(value.svgWidth, kind === 'start' ? 40 : 36);
  assert.equal(value.svgHeight, value.svgWidth);
  assert.ok(value.wordVisible && value.fits, `${kind} branding is clipped at ${width}px`);
  // --brand-deep on the light start page, --theme-on-dark inside the navigation.
  assert.equal(value.color, kind === 'start' ? 'rgb(41, 40, 51)' : 'rgb(247, 246, 244)');
  assert.equal(value.svgHidden, 'true', 'Adjacent wordmark must provide the accessible name');
  assert.ok(value.label.includes('HistoPilot'));
  if (kind === 'workspace') assert.equal(value.sidebarInert, false);
  layouts.push(value);
}
async function viewport(width) {
  await cdp('Emulation.setDeviceMetricsOverride', {width, height: width === 390 ? 844 : 1000, deviceScaleFactor: 1, mobile: width === 390});
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
}
async function click(selector) {
  await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block: 'center', inline: 'nearest'})`);
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  const point = await evaluate(`(() => {const rect = document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect(); return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};})()`);
  await cdp('Input.dispatchMouseEvent', {type: 'mousePressed', button: 'left', clickCount: 1, ...point});
  await cdp('Input.dispatchMouseEvent', {type: 'mouseReleased', button: 'left', clickCount: 1, ...point});
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Network.enable');
  await viewport(1440);
  await cdp('Page.navigate', {url: pathToFileURL(join(dist, 'index.html')).href});
  await waitFor('document.querySelector(".start-recent-list button")');
  for (const width of [1440, 800, 390]) {
    await viewport(width);
    await verifyLayout('start', width);
    await screenshot(width === 1440 ? 'brand-start' : `brand-start-${width}`);
  }
  await click('.start-action-new');
  await waitFor('document.querySelector(".start-view-new")');
  await click('.start-brand');
  await waitFor('document.querySelector(".start-view-welcome")');
  await click('.start-recent-list button');
  await waitFor('document.querySelector(".project-roadmap")');
  for (const width of [1440, 800, 390]) {
    await viewport(width);
    if (width === 390) {
      await click('[aria-label="Toggle navigation"]');
      await waitFor('document.querySelector(".nav-open .sidebar") && !document.querySelector(".sidebar").inert');
      await evaluate('new Promise(resolve => setTimeout(resolve, 300))');
    }
    await verifyLayout('workspace', width);
    await screenshot(width === 800 ? 'brand-tablet' : `brand-workspace-${width}`);
  }
  await viewport(1440);
  await click('.sidebar a[href="#experiments"]');
  await waitFor('document.body.innerText.includes("Create your first experiment")');
  await click('.sidebar .brand');
  await waitFor('location.hash === "#overview" && document.querySelector(".project-roadmap")');
  await click('[aria-label="Back to start page"]');
  await waitFor('document.querySelector(".start-view-welcome")');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, [], 'Unexpected browser errors');
  await evaluate('window.showBrandProof()');
  await viewport(1120);
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1120, height: 525, deviceScaleFactor: 1, mobile: false});
  await screenshot('brand-proof');
  const report = {passed: true, scope: 'Real App, Start and shared Brand components with in-memory APIs; local Chromium file:// only, no server.', checks: ['native SVG symbols', 'matching light/inverse palette', 'visible wordmarks and subtitles', 'desktop/tablet/phone overflow and clipping', 'mobile drawer branding', 'start brand returns to welcome', 'workspace brand returns to roadmap', 'back to start remains functional'], layouts, exceptions};
  await writeFile(join(artifacts, 'brand-verification.json'), JSON.stringify(report, null, 2) + '\n');
  console.log('PASS: brand appearance, responsive layout and navigation at 1440, 800 and 390px.');
  console.log('Artifacts: ' + artifacts);
} catch (error) {
  try {await writeFile(join(artifacts, 'brand-failure.txt'), await evaluate('document.body.innerText')); await screenshot('brand-failure');} catch {}
  console.error('Temporary fixture: ' + output);
  throw error;
} finally {
  browser.kill();
}
