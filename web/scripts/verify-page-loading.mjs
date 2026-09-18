/** Verify on-demand pages in local Chromium using file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-page-loading-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
const icon = 'data:image/svg+xml;base64,' + Buffer.from(await readFile(join(web, 'public/favicon.svg'))).toString('base64');
await writeFile(fixture, `
import React, { Suspense } from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import App from ${source('App.tsx')};
import { lazyPage, PageLoading } from ${source('components/LazyPage.tsx')};
import WorkspaceErrorBoundary from ${source('components/WorkspaceErrorBoundary.tsx')};
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
const project = { id: 'project', name: 'Loading study', mode: 'local', lifecycleState: 'active', storagePath: '/fixture/project', description: '', available: true, createdAt: '2026-09-12T12:00:00Z', updatedAt: '2026-09-12T12:00:00Z', config: { seed: 42, folds: 2 }, sources: [] };
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
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
const root = createRoot(document.getElementById('app'));
root.render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
const probe = createRoot(document.getElementById('probe'));
window.mountLoadingProbe = () => {
  const Page = lazyPage(() => { state.loadAttempts += 1; return new Promise((resolve, reject) => {
    window.resolvePage = () => resolve({ default: () => <h1>Recovered page</h1> });
    window.rejectPage = () => reject(new Error('Fixture page download interrupted'));
  }); });
  probe.render(<WorkspaceErrorBoundary><Suspense fallback={<PageLoading name="Fixture page" />}><Page /></Suspense></WorkspaceErrorBoundary>);
};
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
  + '</head><body><div id="app"></div><div id="probe"></div><script type="module" src="fixture.js"></script></body></html>');

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
const requestedPage = name => downloads.some(url => new RegExp('/' + name + '-[^/]+\\.js$').test(url));
async function screenshot(name) {
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  await evaluate('Promise.allSettled(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished))');
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Network.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href });
  await waitFor('document.querySelector(".start-recent-list button")');
  assert.ok(requestedPage('Start'), 'Start must load on demand');
  for (const name of ['LocalDataset', 'LocalProtocol', 'LocalExperiments', 'LocalFeatures', 'LocalInterpretation']) assert.equal(requestedPage(name), false, name + ' downloaded before opening a project');
  await evaluate('document.querySelector(".start-recent-list button").click()');
  await waitFor('document.querySelector(".sidebar")');
  await evaluate(`document.querySelector('.sidebar a[href="#dataset"]').click()`);
  await waitFor(`document.querySelector('[aria-label="Saved datasets and imports"]') || document.body.innerText.includes("No datasets or import drafts yet")`);
  assert.ok(requestedPage('LocalDataset'), 'Dataset page did not load');
  assert.equal(requestedPage('LocalExperiments'), false, 'Dataset downloaded Experiments');
  await evaluate('window.location.hash = "#cohort"');
  await waitFor('document.body.innerText.includes("Prepare the required inputs")');
  assert.equal(requestedPage('LocalProtocol'), false, 'Locked protocol route downloaded its editor');
  await evaluate(`document.querySelector('.sidebar a[href="#experiments"]').click()`);
  await waitFor('document.body.innerText.includes("Create your first experiment")');
  assert.ok(requestedPage('LocalExperiments'), 'Experiment page did not load');
  assert.equal(requestedPage('LocalInterpretation'), false, 'Experiments downloaded interpretation');
  await evaluate(`document.querySelector('.sidebar a[href="#dataset"]').click()`);
  await waitFor('document.body.innerText.includes("No datasets or import drafts yet")');
  assert.equal(downloads.filter(url => /\/LocalDataset-[^/]+\.js$/.test(url)).length, 1, 'Returning to a loaded page downloaded it again');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('dataset-mobile');
  assert.ok(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), 'Dataset overflows mobile viewport');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, [], 'Unexpected app errors before the intentional failure');
  await evaluate('window.mountLoadingProbe()');
  await waitFor('document.querySelector("#probe [role=status][aria-busy=true]")');
  await screenshot('loading-state');
  await evaluate('window.rejectPage()');
  await waitFor('document.querySelector("#probe [role=alert]")?.innerText.includes("This page could not be loaded")');
  assert.ok(await evaluate('document.querySelector("#probe").innerText.includes("Reload application")'));
  await screenshot('page-load-recovery');
  const expectedErrors = exceptions.splice(0);
  assert.ok(expectedErrors.length > 0, 'The error boundary did not report the intentional import error');
  await evaluate('[...document.querySelectorAll("#probe button")].find(button => button.textContent === "Try this view again").click()');
  await waitFor('window.workflow.loadAttempts === 2 && document.querySelector("#probe [role=status]")');
  await evaluate('window.resolvePage()');
  await waitFor('document.querySelector("#probe h1")?.textContent === "Recovered page"');
  assert.equal(await evaluate('window.workflow.loadAttempts'), 2);
  assert.deepEqual(exceptions, [], 'Unexpected errors after retry');
  await writeFile(join(output, 'verification.json'), JSON.stringify({ passed: true, scope: 'Real App and native ES chunks, local Chromium file://, in-memory APIs; no HistoPilot server.', checks: ['Start defers scientific pages', 'first dataset visit loads dataset chunk', 'prerequisite gate defers locked protocol', 'Experiments loads independently', 'loaded route reuse', 'mobile overflow', 'accessible pending page', 'page-download error boundary', 'explicit retry re-runs importer and recovers'], downloads, expectedErrorCount: expectedErrors.length }, null, 2));
  console.log('PASS: on-demand pages, direct-link gates, cached revisits, loading, and retry recovery.');
  console.log('Artifacts: ' + output);
} catch (error) {
  try { await writeFile(join(output, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch {}
  console.error('Artifacts: ' + output);
  console.error('Downloads: ' + JSON.stringify(downloads));
  console.error('Browser errors: ' + JSON.stringify(exceptions));
  throw error;
} finally {
  browser.kill();
}
