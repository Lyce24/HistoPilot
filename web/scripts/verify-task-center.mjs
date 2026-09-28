/** Run with node web/scripts/verify-task-center.mjs [output]. Uses file:// and local Chromium; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = process.argv[2] ? resolve(process.argv[2]) : await mkdtemp(join(tmpdir(), 'histopilot-task-center-'));
await mkdir(output, { recursive: true });
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import TaskCenter from ${source('pages/TaskCenter.tsx')};
import JobTray from ${source('components/JobTray.tsx')};
import { taskCenter } from ${source('api/taskCenter.ts')};
import { experiments } from ${source('api/experiments.ts')};
import { trident } from ${source('api/trident.ts')};
import { batchTemplate } from ${source('components/DevelopmentBatches.tsx')};
import { fixtureCapacity, fixtureDetail, fixtureHistoryGroup, fixtureOwner, fixtureRollup, fixtureSummary, fixtureTask } from ${source('testFixtures/taskCenter.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
const copy = value => structuredClone(value);
const state = window.workflow = { calls: [], errors: [], loseNext: false, slowNext: null, capacity: fixtureCapacity(), owners: [
  fixtureOwner({ waitingReason: 'Waiting for a GPU slot (4/4)' }),
  fixtureOwner({ key: 'owner-2', id: 'refit-1', kind: 'predictor-refit', title: 'KRAS refit · seed 42', position: 2, counts: { queued: 1 }, etaSeconds: 900, link: '?project=project#post-development?tab=refits&refit=refit-1', actions: { hold: true, release: false, stop: false, cancel: true, retry: false, moveUp: true, moveDown: false } }),
  fixtureOwner({ key: 'owner-3', id: 'other', title: 'Inference cohort (other workspace)', projectId: 'elsewhere', sameWorkspace: false, position: 3, counts: { queued: 6 }, link: null }),
], applied: new Set() };
window.fetch = async (...args) => { const message = 'Unexpected request: ' + args[0]; state.errors.push(message); throw new Error(message); };
window.confirm = () => { state.errors.push('window.confirm used'); return true; };
const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native', packArtifactId: null };
const running = [
  fixtureTask({ progress: { epoch: 23, maxEpochs: 100, trainingLoss: 0.41, cudaPeakReservedBytes: 1.4 * 1024 ** 3 } }),
  fixtureTask({ id: 'task-2', title: 'Baseline · Config 1 · Fold 2 · Train seed 42 · Split seed 7', gpu: 0, progress: { epoch: 61, maxEpochs: 100 }, startedAt: new Date(Date.now() - 45 * 60000).toISOString() }),
  fixtureTask({ id: 'task-3', title: 'Evaluation · TCGA external cohort', kind: 'compute-job', labels: { computeKind: 'evaluation', recordId: 'e1' }, lane: 'cpu', gpu: null, progress: { completedModels: 3, totalModels: 5 }, owner: { ...fixtureTask().owner, key: 'owner-9', title: 'External validation', projectId: 'second-project', projectName: 'Colon' }, link: '?project=second-project#evaluation?evaluation=e1', startedAt: new Date(Date.now() - 5 * 60000).toISOString() }),
].map((task, index) => ({ ...task, startedAt: task.startedAt ?? new Date(Date.now() - (index + 1) * 20 * 60000).toISOString() }));
const pending = [fixtureTask({ id: 'task-q', state: 'queued', startedAt: null, progress: null, resources: null, waitingReason: 'Waiting for a GPU slot (4/4)' })];
const failed = fixtureTask({ id: 'done-1', title: 'Predictors · KRAS study', kind: 'predictor-coordinator', state: 'failed', startedAt: '2026-09-27T08:00:00Z', finishedAt: '2026-09-27T08:12:00Z', progress: null, exit: { reason: 'error', returncode: 0, error: 'Traceback (most recent call last):\\nBlockingIOError: [Errno 11] Resource temporarily unavailable' }, failure: { title: 'The project was busy', cause: 'Another HistoPilot operation was changing this project when the task started, so it stopped without changing anything.', advice: 'Retry once the other operation has finished.', detail: 'BlockingIOError: [Errno 11] Resource temporarily unavailable', retry: 'safe' }, actions: { cancel: false, retry: true } });
const all = [...running, ...pending, failed];
function track(method, ...args) { state.calls.push({ method, args: copy(args) }); }
const summary = () => fixtureSummary({ running: 4, queued: 41, succeeded: 20 }, { paused: state.capacity.settings.paused, recentFailures: 1 });
taskCenter.snapshot = async () => { track('snapshot'); return { summary: summary(), running: copy(running), owners: copy(state.owners), pendingCount: 41, updatedAt: '2026-09-27T10:00:01Z' }; };
taskCenter.rollup = async (scope) => { track('rollup', scope); return fixtureRollup({ scope: {}, counts: summary().counts, recentFailures: 1, progress: null, ownerKey: null, href: '#task-center', lastFailure: { taskId: 'done-1', title: failed.title, state: 'failed', reason: 'error', message: 'The project was busy', cause: '', retry: 'safe', at: failed.finishedAt } }); };
taskCenter.capacity = async () => { track('capacity'); return copy(state.capacity); };
taskCenter.history = async (params) => { track('history', params); return { total: 1, offset: 0, limit: params.limit, groups: [fixtureHistoryGroup({ lastFailure: { taskId: 'done-1', title: failed.title, state: 'failed', reason: 'error', message: 'The project was busy', cause: failed.failure.cause, retry: 'safe', at: failed.finishedAt } })] }; };
taskCenter.tasks = async (params) => { track('tasks', params); const list = params.owner ? [...running.slice(0, 2), ...pending, failed] : params.state === 'live' ? [...running, ...pending].slice(0, params.limit) : running; return { tasks: copy(list), hasMore: false, offset: params.offset ?? 0 }; };
taskCenter.task = async (id) => { track('task', id); const task = all.find(item => item.id === id); return fixtureDetail({ ...copy(task), logTail: 'Epoch 22: train_loss=0.412 val_auroc=0.781\\nEpoch 23: train_loss=0.398 val_auroc=0.784\\n', logTruncated: true, logSize: 120000 }); };
taskCenter.owner = async (key) => { track('owner', key); return copy(state.owners.find(item => item.key === key) ?? state.owners[0]); };
taskCenter.log = async (id) => { track('log', id); return 'FULL LOG line 1\\nEpoch 23: train_loss=0.398 val_auroc=0.784\\n'; };
taskCenter.ownerAction = async (key, action, operationId, position) => {
  track('ownerAction', key, action, operationId, position);
  const owner = state.owners.find(item => item.key === key);
  if (state.slowNext) { const gate = state.slowNext; state.slowNext = null; await gate.promise; }
  // The first lost request never reaches the service, so the same action is still offered.
  if (state.loseNext) { state.loseNext = false; throw new TypeError('Fixture lost the request'); }
  if (!state.applied.has(operationId)) {
    state.applied.add(operationId);
    if (action === 'hold') Object.assign(owner, { held: true, actions: { ...owner.actions, hold: false, release: true } });
    if (action === 'release') Object.assign(owner, { held: false, actions: { ...owner.actions, hold: true, release: false } });
  }
  return copy(owner);
};
window.slowOwnerAction = () => { let release; const promise = new Promise((resolve) => { release = resolve; }); state.slowNext = { promise }; window.releaseOwnerAction = release; };
taskCenter.taskAction = async (id, action, operationId) => { track('taskAction', id, action, operationId); return copy(running[0]); };
taskCenter.updateCapacity = async (patch) => {
  track('updateCapacity', patch);
  if (patch.parallelGpuTasks) { state.capacity.settings.defaultGpuSlots = patch.parallelGpuTasks; state.capacity.settings.gpuSlots = { '0': patch.parallelGpuTasks }; state.capacity.effective.gpuSlots = { '0': patch.parallelGpuTasks }; }
  if (patch.paused !== undefined) state.capacity.settings.paused = patch.paused;
  return copy(state.capacity);
};
experiments.summaries = async () => ({ items: [{ id: 'ready-exp', key: 'draft:ready-exp', name: 'Frozen nnMIL comparison', notes: '', tags: [], revision: 3, state: 'active', status: 'ready', legacy: false, createdAt: '', updatedAt: '', inputs, batches: [], drafts: [], predictorId: null, stage: 'planning', frozenSetupId: 'setup', batchPlans: [{ id: 'plan', spec: batchTemplate('learning-rate', inputs, 'Frozen nnMIL comparison') }] }] });
trident.jobs = async () => ({ jobs: [] });
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 1000 } } });
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><main className="content module-content"><TaskCenter workspace={{ mode: 'local', project: { id: 'project', name: 'KRAS', lifecycleState: 'active' } }} /></main><JobTray projectId="project" /></QueryClientProvider>);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
    outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
    lib: { entry: fixture, name: 'TaskCenterFixture', formats: ['iife'], fileName: () => 'fixture.js' },
  } });
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
  + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('')
  + '<style>body{margin:0}#app{max-width:1320px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');

const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter((name) => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')],
  { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
let stderr = '';
browser.stderr.on('data', (data) => { stderr += data.toString(); });
const requests = new Map();
let nextId = 0, buffer = '', sessionId;
const exceptions = [];
const checks = [];
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
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await evaluate('Boolean(' + expression + ')')) { checks.push(name); console.log('PASS', name); return; }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('Timed out: ' + name + '\n' + await evaluate('document.body.innerText'));
}
const button = (text, scope = 'document') => `[...${scope}.querySelectorAll('button')].find(el => el.checkVisibility() && el.textContent.trim() === ${JSON.stringify(text)})`;
async function click(expression, label) {
  await check('Available: ' + label, expression + ' && !' + expression + '.disabled');
  await evaluate(expression + '.click()');
}
async function screenshot(name) {
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
async function key(name, code, keyCode, modifiers = 0) {
  for (const type of ['keyDown', 'keyUp']) await cdp('Input.dispatchKeyEvent', { type, key: name, code, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode, modifiers });
}
const detailsLink = `[...document.querySelectorAll('.tc-running a')].find(el => el.textContent === 'Details')`;
const ownerCalls = (action) => `window.workflow.calls.filter(call => call.method === 'ownerAction' && call.args[1] === ${JSON.stringify(action)})`;
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  const page = pathToFileURL(join(dist, 'index.html')).href;
  await cdp('Page.navigate', { url: page + '#task-center' });
  await check('Runner and queue summary', 'document.querySelector(".tc-strip")?.innerText.includes("4 running · 41 queued · 1 failed in the last 24 h")');
  await check('Tray reads the machine rollup', 'document.querySelector(".job-tray")?.innerText.includes("4 running · 41 queued · 1 failed")');
  await check('One snapshot read feeds the page', "window.workflow.calls.some(c => c.method === 'snapshot') && !window.workflow.calls.some(c => c.method === 'summary' || c.method === 'owners')");
  await check('Capacity meters', 'document.body.innerText.includes("4 of 5 in use") && document.body.innerText.includes("16 of 34 committed")');
  await check('Owner waiting reason from the server', 'document.querySelector(".tc-queue")?.innerText.includes("Waiting for a GPU slot (4/4)")');
  await check('Evaluation reads as its kind and project', 'document.querySelector(".tc-running")?.innerText.includes("Evaluation") && document.querySelector(".tc-running").innerText.includes("Project Colon")');
  await check('Measured GPU memory', 'document.querySelector(".tc-running")?.innerText.includes("1.4 GiB GPU memory (peak)")');
  await check('History grouped by owner with the failure cause', 'document.querySelector(".tc-history")?.innerText.includes("60 completed · 1 failed") && document.querySelector(".tc-history").innerText.includes("The project was busy")');
  await screenshot('01-task-center');
  await evaluate('document.querySelector(".tc-breakdown").open = true');
  await check('Suggestion breakdown', 'document.querySelector(".tc-breakdown").textContent.includes("Tasks per GPU")');
  await click(button('Use suggestion'), 'Use suggestion');
  await check('Suggestion applied with an operation ID', "(() => { const call = window.workflow.calls.find(c => c.method === 'updateCapacity'); return call && call.args[0].parallelGpuTasks === 5 && typeof call.args[0].operationId === 'string'; })()");
  await check('Setting now matches suggestion', 'document.body.innerText.includes("Matches the current setting.")');
  const scroll = await evaluate('window.scrollY = 400, window.scrollY');
  await click(`[...document.querySelectorAll('.tc-running a')].find(el => el.textContent === 'Details')`, 'first Details');
  await check('Details open in a drawer addressed by the URL', 'location.hash === "#task-center?task=task-1" && document.querySelector(".tc-drawer")?.innerText.includes("val_auroc=0.784")');
  await check('Opening details keeps the scroll position', `window.scrollY === ${scroll}`);
  await check('Drawer shows command, measurements and dependents', 'document.querySelector(".tc-drawer").innerText.includes("Shared the GPU with") && document.querySelector(".tc-drawer").textContent.includes("histopilot.workers.managed_fold") && document.querySelector(".tc-drawer").textContent.includes("Baseline · Final results")');
  await check('Log has an accessible name', 'document.querySelector(".tc-drawer pre.tc-log")?.getAttribute("aria-label")?.startsWith("Log of")');
  await click(button('Show full log', 'document.querySelector(".tc-drawer")'), 'Show full log');
  await check('Full log replaces the tail', 'document.querySelector(".tc-drawer pre.tc-log")?.innerText.includes("FULL LOG line 1")');
  await screenshot('02-drawer');
  await cdp('Page.reload');
  await check('A refresh keeps the open task', 'location.hash === "#task-center?task=task-1" && document.querySelector(".tc-drawer")?.innerText.includes("val_auroc=0.784")');
  await click('document.querySelector(".tc-drawer-head button")', 'Close drawer');
  await check('Closing drops the task from the URL', 'location.hash === "#task-center" && !document.querySelector(".tc-drawer")');
  // From the keyboard, as a user would: focus the row link and press Enter.
  await check('Available: Details again', detailsLink);
  await evaluate(detailsLink + '.focus()');
  await key('Enter', 'Enter', 13);
  await check('The drawer is a modal dialog named by its title', `(() => { const dialog = document.querySelector('.tc-drawer[role="dialog"][aria-modal="true"]'); return dialog && document.getElementById(dialog.getAttribute('aria-labelledby'))?.textContent.startsWith('Baseline'); })()`);
  await check('Focus starts on the close button', 'document.activeElement === document.querySelector(".tc-drawer-head button")');
  await key('Tab', 'Tab', 9, 8);
  await check('Shift+Tab from the first control wraps to the last one in the drawer', 'document.querySelector(".tc-drawer").contains(document.activeElement) && document.activeElement !== document.querySelector(".tc-drawer-head button")');
  await key('Tab', 'Tab', 9);
  await check('Tab from the last control wraps to the first', 'document.activeElement === document.querySelector(".tc-drawer-head button")');
  await evaluate('document.activeElement.blur()');
  await key('Escape', 'Escape', 27);
  await check('Escape closes the drawer wherever focus is', 'location.hash === "#task-center" && !document.querySelector(".tc-drawer")');
  await check('Focus returns to the link that opened it', `document.activeElement === ${detailsLink}`);
  await click(detailsLink, 'Details for Back');
  await check('Drawer open again', 'location.hash === "#task-center?task=task-1" && document.querySelector(".tc-drawer")');
  await evaluate('history.back()');
  await check('Back closes the drawer and stays in the Task Center', 'location.hash === "#task-center" && !document.querySelector(".tc-drawer") && document.querySelector(".tc-running")');
  await evaluate('history.forward()');
  await check('Forward reopens it', 'location.hash === "#task-center?task=task-1" && document.querySelector(".tc-drawer")');
  await click('document.querySelector(".tc-drawer-head button")', 'Close drawer again');
  await check('Closed again', 'location.hash === "#task-center" && !document.querySelector(".tc-drawer")');
  await cdp('Page.navigate', { url: page + '#task-center?task=done-1' });
  await check('A failure is explained above its traceback', `(() => { const text = document.querySelector(".tc-drawer")?.innerText ?? ''; return text.includes("The project was busy") && text.includes("Safe to retry") && document.querySelector(".tc-failure") && !document.querySelector(".tc-failure details[open]"); })()`);
  await screenshot('03-failure');
  await cdp('Page.navigate', { url: page + '#task-center?owner=owner-2&project=project' });
  await check('An owner deep link opens that owner with its tasks', 'document.querySelector(".tc-owner-focus")?.innerText.includes("KRAS refit · seed 42") && document.querySelector(".tc-filter-bar")?.innerText.includes("This project")');
  await check('Owner filter narrows running work', '!document.querySelector(".tc-running") || !document.querySelector(".tc-running").innerText.includes("TCGA external cohort")');
  await screenshot('04-owner');
  await cdp('Page.navigate', { url: page + '#task-center' });
  await check('Queue is back', 'document.querySelector(".tc-queue")');
  await click(`[...document.querySelectorAll('.tc-queue button')].find(el => el.textContent === 'Show tasks')`, 'Show tasks');
  await check('Owner tasks expand', 'document.querySelector(".tc-owner-tasks")?.innerText.includes("Waiting for a GPU slot (4/4)")');
  await evaluate('window.slowOwnerAction()');
  await click(`document.querySelector('[aria-label="Hold KRAS study"]')`, 'Hold (held open)');
  await check('Only that owner waits while its request is in flight', `(() => { const other = document.querySelector('[aria-label="Hold KRAS refit · seed 42"]'); const mine = document.querySelector('[aria-label="Hold KRAS study"]'); return other && !other.disabled && mine && mine.disabled; })()`);
  await evaluate('window.releaseOwnerAction()');
  await check('Held owner offers release', `${ownerCalls('hold')}.length === 1 && document.querySelector('[aria-label="Release KRAS study"]')`);
  await evaluate('window.workflow.loseNext = true');
  await click(`document.querySelector('[aria-label="Release KRAS study"]')`, 'Release');
  await check('Lost acknowledgement explained', 'document.body.innerText.includes("Repeating it is safe")');
  await click(`document.querySelector('[aria-label="Release KRAS study"]')`, 'Release again');
  await check('Retry reuses the operation ID', `(() => { const calls = ${ownerCalls('release')}; return calls.length === 2 && calls[0].args[2] === calls[1].args[2]; })()`);
  await click(`document.querySelector('[aria-label="Cancel KRAS refit · seed 42"]')`, 'Cancel owner');
  await check('Cancel asks inside the page', 'document.querySelector(".confirm-action")?.innerText.includes("Cancel every unfinished task")');
  await click(button('Cancel tasks'), 'Confirm cancel');
  await check('Confirmed cancel is sent once', `${ownerCalls('cancel')}.length === 1`);
  await click(button('Pause queue'), 'Pause queue');
  await check('Queue paused', 'document.body.innerText.includes("Resume queue")');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  // Compare with the viewport width set above: an overflowing page also widens innerWidth.
  await check('Fits a phone viewport', 'document.documentElement.scrollWidth <= 391 && document.body.scrollWidth <= 391');
  await screenshot('05-mobile');
  await cdp('Page.navigate', { url: page + '#task-center?task=task-1' });
  await check('Drawer fills a phone screen', 'document.querySelector(".tc-drawer")?.getBoundingClientRect().width <= 391 && document.documentElement.scrollWidth <= 391');
  await screenshot('06-mobile-drawer');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  await writeFile(join(output, 'verification.json'), JSON.stringify({ passed: true, scope: 'Bundled Task Center page and job tray in Chromium with mocked Task Center APIs; no HistoPilot server or runner started.', checks }, null, 2));
  console.log('Artifacts: ' + output);
} catch (error) {
  try { await writeFile(join(output, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Chromium may not have started. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + output);
  throw error;
} finally {
  browser.kill();
}
