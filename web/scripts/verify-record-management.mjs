/** Run with node scripts/verify-record-management.mjs. Uses local Chromium and file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-record-management-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React, { useEffect, useState } from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import { StageLibrary, StageLibraryToolbar } from ${source('components/StageWorkflow.tsx')};
import { RecordManageButton } from ${source('components/RecordManagement.tsx')};
import { PageHeader } from ${source('components/ui.tsx')};
import { lifecycle } from ${source('api/lifecycle.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('components/StageWorkflow.css')};
const copy = value => structuredClone(value);
const state = window.workflow = {
  calls: [], errors: [], revision: 1, mutations: [], listMounts: 0,
  holdPreview: false, holdApply: false, loseResponse: false,
  records: [
    { key: 'dataset:shared-first', id: 'shared-first', type: 'dataset', kind: 'dataset', name: 'Shared dataset name', state: 'active', dependsOn: [], usedBy: [], createdAt: '2026-09-12T00:00:00Z' },
    { key: 'dataset:shared-second', id: 'shared-second', type: 'dataset', kind: 'dataset', name: 'Shared dataset name', state: 'active', dependsOn: [], usedBy: ['configuration:dependent-targets'], createdAt: '2026-09-12T00:00:00Z' },
    { key: 'configuration:dependent-targets', id: 'dependent-targets', type: 'configuration', kind: 'protocol', name: 'Dependent targets', state: 'active', dependsOn: ['dataset:shared-second'], usedBy: [], createdAt: '2026-09-12T00:00:00Z' },
  ],
};
const completed = new Map();
const previews = new Map();
window.fetch = async (...args) => { state.errors.push('Unexpected request: ' + args[0]); throw new Error(state.errors.at(-1)); };
function checkProject(project) { if (project !== 'project') throw new Error('Wrong project: ' + project); }
lifecycle.inventory = async project => {
  checkProject(project); state.calls.push({ method: 'inventory', project });
  return { projectId: project, revision: state.revision, projectState: 'active', items: copy(state.records), audit: [], note: '' };
};
lifecycle.preview = async (project, action, keys) => {
  checkProject(project); state.calls.push({ method: 'preview', project, action, keys: copy(keys) });
  if (state.holdPreview) await new Promise(resolve => { window.releasePreview = () => { state.holdPreview = false; resolve(); }; });
  if (keys.some(key => !state.records.some(record => record.key === key))) throw new Error('Unknown selected record');
  const requiredKeys = action === 'trash' && keys.includes('dataset:shared-second') && !keys.includes('configuration:dependent-targets')
    && state.records.find(record => record.key === 'configuration:dependent-targets').state !== 'trashed' ? ['configuration:dependent-targets'] : [];
  const previewHash = 'preview-' + state.revision + '-' + action + '-' + keys.join(',');
  const result = { action, keys: copy(keys), revision: state.revision, previewHash, canApply: !requiredKeys.length,
    blockers: [], requiredKeys, recordCount: keys.length, note: '' };
  previews.set(previewHash, copy(result));
  return result;
};
lifecycle.apply = async (project, input) => {
  checkProject(project); state.calls.push({ method: 'apply', project, input: copy(input) });
  if (state.holdApply) await new Promise(resolve => { window.releaseApply = () => { state.holdApply = false; resolve(); }; });
  const previous = completed.get(input.operationId);
  if (previous) {
    if (JSON.stringify(previous.input) !== JSON.stringify(input)) throw new Error('Retry changed the reviewed request');
    return copy(previous.result);
  }
  const review = previews.get(input.previewHash);
  if (!review || review.revision !== state.revision || !review.canApply || review.action !== input.action || JSON.stringify(review.keys) !== JSON.stringify(input.keys)) throw new Error('Applied an invalid or stale review');
  input.keys.forEach(key => { state.records.find(record => record.key === key).state = input.action === 'archive' ? 'archived' : input.action === 'trash' ? 'trashed' : 'active'; });
  state.revision += 1;
  const result = { action: input.action, revision: state.revision, changed: copy(input.keys) };
  completed.set(input.operationId, { input: copy(input), result: copy(result) });
  state.mutations.push(copy(input));
  if (state.loseResponse) { state.loseResponse = false; throw new Error('Fixture lost the confirmation response'); }
  return result;
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
function Records() {
  const [search, setSearch] = useState('');
  useEffect(() => { state.listMounts += 1; }, []);
  const visible = state.records.filter(record => record.type === 'dataset' && (record.name + ' ' + record.id).toLowerCase().includes(search.toLowerCase()));
  return <>
    <StageLibraryToolbar search={search} onSearch={setSearch} searchLabel="Search datasets" count={visible.length} total={2} placeholder="Name or ID" />
    <div className="table-wrap"><table><thead><tr><th>Dataset</th><th>Actions</th></tr></thead><tbody>{visible.map(record => <tr key={record.key} data-record-key={record.key}>
      <th scope="row"><strong>{record.name}</strong><small>{record.id}</small></th>
      <td><RecordManageButton recordKey={record.key} name={record.name} /></td>
    </tr>)}</tbody></table></div>
  </>;
}
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><main className="stage-workspace"><PageHeader title="Datasets" description="Open a dataset or create a new import." /><StageLibrary project="project" title="Datasets"><Records /></StageLibrary></main></QueryClientProvider>);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
    outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
    lib: { entry: fixture, name: 'RecordManagementFixture', formats: ['iife'], fileName: () => 'fixture.js' },
  } });
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
  + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('')
  + '<style>body{padding:24px}#app{max-width:1240px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');

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
const dialogs = [];
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
    } else if (message.method === 'Page.javascriptDialogOpening') { dialogs.push(message.params.message); void cdp('Page.handleJavaScriptDialog', { accept: false }); }
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
const button = (text) => '[...document.querySelectorAll("button")].find(el => el.checkVisibility() && el.textContent.trim() === ' + JSON.stringify(text) + ')';
async function click(text) {
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
}
async function search(value) {
  await evaluate('(() => { const el = document.querySelector("input[type=search]"); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, ' + JSON.stringify(value) + '); el.dispatchEvent(new Event("input", { bubbles: true })); })()');
}
async function openRecord(key) {
  const selector = '[...document.querySelectorAll("tr[data-record-key]")].find(el => el.dataset.recordKey === ' + JSON.stringify(key) + ')?.querySelector("button")';
  await waitFor(selector + '?.checkVisibility()', 'Manage button for ' + key);
  await evaluate(selector + '.click()');
  await waitFor(button('Back to datasets'), 'in-stage management page');
  await waitFor('[...document.querySelectorAll("h1,h2,h3")].some(el => el.checkVisibility() && el.textContent.trim() === "Manage Shared dataset name")');
  assert.equal(await evaluate('document.querySelector("input[type=search]")?.checkVisibility()'), false, 'Manage must replace the visible record library');
  assert.equal(await evaluate('location.hash'), '#dataset', 'Manage must stay in the current stage');
}
async function acknowledge() {
  const checkbox = '[...document.querySelectorAll(".cleanup-ack input[type=checkbox]")].find(el => el.checkVisibility())';
  await waitFor(checkbox + ' && !' + checkbox + '.disabled', 'review acknowledgment');
  assert.equal(await evaluate(checkbox + '.checked'), false, 'Every new review must require a fresh acknowledgment');
  await evaluate(checkbox + '.click()');
}
async function backGuard(description) {
  await waitFor(button('Back to datasets') + '?.disabled', description + ': disabled Back');
  await evaluate("window.dispatchEvent(new Event('histopilot:stage-library'))");
  assert.equal(await evaluate('document.querySelector("input[type=search]")?.checkVisibility()'), false, description + ': sidebar click discarded the review');
  assert.equal(await evaluate('location.hash'), '#dataset');
}
async function screenshot(name) {
  await new Promise(resolve => setTimeout(resolve, 220));
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
const previewsExpression = 'window.workflow.calls.filter(call => call.method === "preview")';
const appliesExpression = 'window.workflow.calls.filter(call => call.method === "apply")';
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href + '#dataset' });
  await waitFor('document.querySelectorAll("tr[data-record-key]").length === 2');
  assert.equal(await evaluate('document.querySelector("a[href*=cleanup]") === null'), true, 'A record should not send users to Workspace cleanup');
  await search('shared');
  await waitFor('document.querySelector("input[type=search]").value === "shared"');
  await screenshot('00-library-desktop');

  await openRecord('dataset:shared-first');
  await click('Archive');
  await waitFor(button('Archive (1)'));
  assert.deepEqual(await evaluate(previewsExpression + '.at(-1).keys'), ['dataset:shared-first'], 'Same-name records must keep their exact identity');
  assert.equal(await evaluate(button('Archive (1)') + '.disabled'), true);
  assert.equal(await evaluate(appliesExpression + '.length'), 0, 'Opening Manage and previewing must not mutate a record');
  await screenshot('01-archive-review-desktop');
  await acknowledge();
  await evaluate('window.workflow.holdApply = true');
  await click('Archive (1)');
  await waitFor('typeof window.releaseApply === "function"');
  await backGuard('Applying an archive');
  assert.equal(await evaluate('window.workflow.records[0].state'), 'active');
  await evaluate('window.releaseApply()');
  await waitFor(button('Restore to Active') + ' && !' + button('Back to datasets') + '.disabled');
  assert.equal(await evaluate('window.workflow.records[0].state'), 'archived');
  assert.equal(await evaluate('window.workflow.records[1].state'), 'active');
  await click('Back to datasets');
  await waitFor('document.querySelector("input[type=search]")?.checkVisibility()');
  assert.equal(await evaluate('document.querySelector("input[type=search]").value'), 'shared', 'Return from Manage must retain search');
  assert.equal(await evaluate('window.workflow.listMounts'), 1, 'Return must preserve the mounted library');

  await openRecord('dataset:shared-second');
  await evaluate('window.workflow.holdPreview = true');
  await click('Delete…');
  await waitFor('typeof window.releasePreview === "function"');
  await backGuard('Loading a deletion review');
  await evaluate('window.releasePreview()');
  await waitFor(button('Include required records and review again'));
  assert.deepEqual(await evaluate(previewsExpression + '.at(-1).keys'), ['dataset:shared-second']);
  assert.equal(await evaluate(appliesExpression + '.length'), 1, 'Dependency discovery must not apply deletion');
  assert.equal(await evaluate(button('Move to Trash (1)') + ' === undefined'), true);
  await screenshot('02-dependent-record-review-desktop');
  await click('Include required records and review again');
  await waitFor(button('Move to Trash (2)'));
  assert.deepEqual(await evaluate(previewsExpression + '.at(-1).keys'), ['dataset:shared-second', 'configuration:dependent-targets']);
  assert.equal(await evaluate(button('Move to Trash (2)') + '.disabled'), true);
  assert.equal(await evaluate(appliesExpression + '.length'), 1, 'Including dependencies still requires explicit confirmation');
  await acknowledge();
  await click('Move to Trash (2)');
  await waitFor(button('Restore to Active') + ' && !' + button('Back to datasets') + '.disabled');
  assert.equal(await evaluate('window.workflow.records[1].state'), 'trashed');
  assert.equal(await evaluate('window.workflow.records[2].state'), 'trashed');
  assert.equal(await evaluate(button('Delete…') + ' === undefined'), true, 'Trashed records should offer Restore, not Delete');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('03-trashed-record-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Manage page overflows mobile viewport');

  await click('Restore to Active');
  await waitFor(button('Restore to Active (1)'));
  await acknowledge();
  await evaluate('window.workflow.loseResponse = true');
  await click('Restore to Active (1)');
  await waitFor(button('Retry this confirmation'));
  assert.equal(await evaluate('window.workflow.records[1].state'), 'active', 'Fixture must simulate a saved change with a lost response');
  await backGuard('An uncertain confirmation');
  await screenshot('04-exact-retry-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Uncertain review overflows mobile viewport');
  const uncertainRequest = await evaluate(appliesExpression + '.at(-1).input');
  const mutationCount = await evaluate('window.workflow.mutations.length');
  await click('Retry this confirmation');
  await waitFor(button('Archive') + ' && !' + button('Back to datasets') + '.disabled');
  assert.deepEqual(await evaluate(appliesExpression + '.at(-1).input'), uncertainRequest, 'Retry must reuse the same key, operation ID, and preview hash');
  assert.equal(await evaluate('window.workflow.mutations.length'), mutationCount, 'An exact retry must not apply the change twice');
  assert.equal(await evaluate('window.workflow.records[2].state'), 'trashed', 'Restoring one record must not silently restore another');
  await click('Back to datasets');
  await waitFor('document.querySelector("input[type=search]")?.checkVisibility()');
  assert.equal(await evaluate('document.querySelector("input[type=search]").value'), 'shared');
  assert.equal(await evaluate('window.workflow.listMounts'), 1);
  await screenshot('05-returned-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Library overflows mobile viewport');

  await openRecord('dataset:shared-first');
  await waitFor(button('Restore to Active'));
  assert.equal(await evaluate(button('Archive') + ' === undefined'), true, 'Reopened management must use the current archived state');
  await click('Restore to Active');
  await waitFor(button('Restore to Active (1)'));
  const beforeBack = await evaluate(appliesExpression + '.length');
  await click('Back to datasets');
  await waitFor('document.querySelector("input[type=search]")?.checkVisibility()');
  assert.equal(await evaluate(appliesExpression + '.length'), beforeBack, 'Leaving an unconfirmed review must not apply it');
  assert.equal(await evaluate('location.hash'), '#dataset');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  assert.deepEqual(dialogs, []);
  await writeFile(join(output, 'verification.json'), JSON.stringify({ passed: true, scope: 'Real shared React library/management components and Chromium with in-memory lifecycle APIs. No HistoPilot server or persisted data writes.', checks: ['in-stage Manage', 'same-name exact record keys', 'archive preview and acknowledgment', 'delete dependencies and explicit inclusion', 'confirmed archive/delete/restore', 'authoritative lifecycle state', 'busy and uncertain return guards', 'exact idempotent retry', 'retained search and mounted library', 'desktop/mobile screenshots and containment'], calls: await evaluate('window.workflow.calls'), mutations: await evaluate('window.workflow.mutations') }, null, 2));
  console.log('PASS: in-stage Manage, exact record keys, reviewed archive/delete/restore, dependency acknowledgment, busy/uncertain guards, idempotent retry, retained library search, and mobile containment.');
  console.log('Artifacts: ' + output);
} catch (error) {
  try { await writeFile(join(output, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Browser may not have launched. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + output);
  throw error;
} finally { browser.kill(); }
