/** Verify the compact workflow launcher offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-roadmap-'));
const fixture = join(output, 'fixture.tsx');
const artifacts = resolve(web, '../docs/dev-review/2026-09-13-interface-consistency-assets');
await mkdir(artifacts, { recursive: true });
const source = (path) => JSON.stringify(join(web, 'src', path));
const icon = 'data:image/svg+xml;base64,' + Buffer.from(await readFile(join(web, 'public/favicon.svg'))).toString('base64');
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import App from ${source('App.tsx')};
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
new MutationObserver(() => { for (const image of document.querySelectorAll('img[src^="/favicon.svg"]')) image.src = ${JSON.stringify(icon)}; }).observe(document, { childList: true, subtree: true });
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
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Network.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href + '?project=project#overview' });
  await waitFor('document.querySelectorAll(".project-roadmap [data-module]").length === 8');
  await screenshot('roadmap-desktop');
  const layout = () => evaluate(`({
    width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
    phases: [...document.querySelectorAll('.roadmap-launcher [data-phase]')].map(section => ({name: section.querySelector('h2').textContent, modules: [...section.querySelectorAll('[data-module]')].map(item => item.dataset.module)})),
    optionalModules: [...document.querySelectorAll('.roadmap-analysis [data-module]')].map(item => item.dataset.module),
    actions: [...document.querySelectorAll('.project-roadmap a[data-module]')].map(item => ({text: item.querySelector('.roadmap-item-action').textContent, height: item.getBoundingClientRect().height, href: item.getAttribute('href')})),
    explanationParagraphs: document.querySelectorAll('.roadmap-launcher p, .roadmap-analysis p').length,
    // Charts and legends stay off this page. A module row's own icon and the
    // next-step card's icons are not graphs, so they are excluded here.
    graphCount: document.querySelectorAll('.project-roadmap svg:not(.roadmap-item-icon > svg):not(.roadmap-state-next-icon > svg):not(.roadmap-state-next-action > svg), .roadmap-legend').length,
    moduleIcons: document.querySelectorAll('.project-roadmap [data-module] .roadmap-item-icon > svg').length,
    // Progress and one suggested next step: both state, neither a chart.
    progress: document.querySelector('.roadmap-state-count')?.textContent ?? '',
    nextCard: document.querySelector('.roadmap-state-next')?.dataset.next ?? null,
    markedNext: [...document.querySelectorAll('.roadmap-item.is-next')].map(item => item.dataset.module),
    navInert: document.querySelector('.sidebar').inert
  })`);
  const desktop = await layout();
  assert.deepEqual(desktop.phases, [{name:'Prepare', modules:['dataset','cohort','features']}, {name:'Develop', modules:['experiments']}, {name:'Evaluate', modules:['test-data','evaluation']}]);
  assert.deepEqual(desktop.optionalModules, ['clinical-utility','interpretation']);
  assert.equal(desktop.graphCount, 0);
  assert.equal(desktop.moduleIcons, 8, 'Every module row carries exactly one icon');
  assert.match(desktop.progress, /^[0-6] of 6 required steps complete$/, 'Progress counts required modules only');
  // At most one suggested step, and the card and the marked row always agree.
  assert.deepEqual(desktop.markedNext, desktop.nextCard ? [desktop.nextCard] : [], 'The suggested step is marked where it sits in the sequence');
  assert.equal(desktop.explanationParagraphs, 0);
  assert.ok(desktop.scrollWidth <= desktop.width + 1, 'Launcher overflows desktop viewport');
  assert.ok(desktop.actions.every(item => item.height >= 44 && item.text === 'Open'));
  await evaluate('document.querySelector(".project-roadmap a[data-module]").focus()');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Tab', code:'Tab', windowsVirtualKeyCode:9});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Tab', code:'Tab', windowsVirtualKeyCode:9});
  assert.equal(await evaluate('document.activeElement?.getAttribute("href")'), '#cohort', 'Module links must follow phase order by keyboard');
  await evaluate('document.activeElement.blur()');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('roadmap-mobile');
  const mobile = await layout();
  assert.ok(mobile.scrollWidth <= mobile.width + 1, 'Launcher overflows mobile viewport');
  assert.ok(mobile.actions.every(item => item.height >= 44), 'Module touch target smaller than 44px');
  assert.ok(mobile.navInert);
  await evaluate('window.showEmptyRoadmap()');
  await waitFor('document.querySelector(".project-roadmap [data-module=cohort]")?.getAttribute("aria-disabled") === "true"');
  await screenshot('roadmap-empty-mobile');
  const gates = await evaluate(`({blocked: [...document.querySelectorAll('.project-roadmap [aria-disabled=true][data-module]')].map(item=>item.dataset.module), open: [...document.querySelectorAll('.project-roadmap a[data-module]')].map(item=>item.dataset.module), scrollWidth:document.documentElement.scrollWidth})`);
  assert.deepEqual(gates.blocked,['cohort','features']);
  for(const id of ['dataset','experiments','test-data','evaluation','clinical-utility','interpretation']) assert.ok(gates.open.includes(id));
  assert.ok(gates.scrollWidth <=390);
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, [], 'Unexpected browser errors');
  await writeFile(join(artifacts, 'roadmap-verification.json'), JSON.stringify({passed: true, scope: 'Actual App with compact workflow launcher; in-memory records, Chromium file://, no server.', desktop, mobile, gates, exceptions}, null, 2));
  console.log('PASS: compact phases, consistent Open actions, keyboard navigation, touch targets, input gates, desktop/mobile overflow.');
  console.log('Artifacts: ' + artifacts);
} catch(error) {
  try { await writeFile(join(artifacts, 'roadmap-failure.txt'), await evaluate('document.body.innerText')); await screenshot('roadmap-failure'); } catch {}
  throw error;
} finally {
  browser.kill();
}
