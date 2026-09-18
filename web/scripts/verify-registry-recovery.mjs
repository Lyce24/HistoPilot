/** Verify create-operation and metadata recovery with local Chromium/file://; no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';
const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-registry-recovery-'));
const fixture = join(output, 'fixture.tsx');
const source = name => JSON.stringify(join(web, 'src', name));
await writeFile(fixture, `
import React, { useState } from ${JSON.stringify(join(web,'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web,'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider, useQuery } from ${JSON.stringify(join(web,'node_modules/@tanstack/react-query/build/modern/index.js'))};
import { CreateExperiment, ExperimentMetadata } from ${source('components/ExperimentRegistry.tsx')};
import { experiments } from ${source('api/experiments.ts')};
import { ApiError } from ${source('api/client.ts')};
import { confirmWorkspaceNavigation } from ${source('lib/workspaceNavigation.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('clinical-workspace.css')};
const clone = value => structuredClone(value);
window.registryDocument = crypto.randomUUID();
const initial = { id: 'record', name: 'Saved title', notes: 'Saved note', tags: [], revision: 1, state: 'active', stage: 'planning', status: 'created', legacy: false, inputs: null, batches: [], drafts: [], createdAt: '', updatedAt: '', key: 'draft:record' };
const state = window.registryFixture = JSON.parse(sessionStorage.getItem('fixture:server') ?? 'null') ?? { metadata: initial, calls: [], created: {}, loseNext: true, error: null, successfulNavigations: 0 };
const persist = () => sessionStorage.setItem('fixture:server', JSON.stringify(state));
window.fetch = async input => { throw new Error('Unexpected API request: ' + input); };
experiments.create = async (_, input) => {
  state.calls.push({ method: 'create', input: clone(input) });
  if (!state.created[input.operationId]) state.created[input.operationId] = { ...clone(initial), ...clone(input), id: 'created-' + Object.keys(state.created).length };
  persist();
  if (state.loseNext) {
    state.loseNext = false; persist();
    if (state.error === 'server') throw new ApiError('The upstream creation response is unavailable.', 503);
    throw new Error('The creation response was lost');
  }
  return clone(state.created[input.operationId]);
};
experiments.get = async () => clone(state.metadata);
experiments.update = async (_, __, input) => {
  state.calls.push({ method: 'metadata', input: clone(input) }); persist();
  if (input.expectedRevision !== state.metadata.revision) throw new ApiError('Another editor saved newer experiment details.', 409, 'DRAFT_REVISION_CONFLICT');
  if (window.delayMetadata) await new Promise(resolve => { window.releaseMetadata = resolve; });
  state.metadata = { ...state.metadata, ...clone(input), revision: state.metadata.revision + 1 }; persist();
  return clone(state.metadata);
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
function Fixture() {
  const [view, setView] = useState('create');
  const record = useQuery({ queryKey: ['model-experiment', 'project', 'record'], queryFn: () => experiments.get('project', 'record'), enabled: view === 'metadata' });
  const go = next => { if (confirmWorkspaceNavigation()) setView(next); };
  window.openMetadata = () => go('metadata');
  window.changeMetadata = (notify = false, sameDetails = false) => {
    state.metadata = { ...state.metadata, ...(sameDetails ? {} : { name: 'Other tab title', notes: 'Other tab note' }), revision: state.metadata.revision + 1 }; persist();
    if (notify) client.setQueryData(['model-experiment', 'project', 'record'], clone(state.metadata));
  };
  return <main style={{ maxWidth: 950, margin: '20px auto', padding: 20 }}>
    <button type="button" onClick={() => go(view === 'metadata' ? 'library' : 'metadata')}>${'Switch view'}</button>
    {view === 'create' ? <CreateExperiment project="project" onClose={() => setView('library')} onCreated={item => { if (confirmWorkspaceNavigation()) { state.successfulNavigations++; persist(); setView('created'); } }} />
      : view === 'metadata' ? record.data ? <ExperimentMetadata project="project" record={record.data} /> : <p>Loading metadata</p>
      : view === 'created' ? <p id="created">Created once and opened inputs</p>
      : <button type="button" onClick={() => go('create')}>Resume creation</button>}
  </main>;
}
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><Fixture /></QueryClientProvider>);
`);
await build({configFile:false,root:web,logLevel:'error',plugins:[react()],define:{'process.env.NODE_ENV':JSON.stringify('production')},resolve:{alias:{'react/jsx-runtime':join(web,'node_modules/react/jsx-runtime.js')}},build:{outDir:join(output,'dist'),emptyOutDir:true,minify:false,lib:{entry:fixture,name:'RegistryRecoveryFixture',formats:['iife'],fileName:()=> 'fixture.js'}}});
const dist=join(output,'dist');
const css=(await readdir(dist)).filter(name=>name.endsWith('.css'));
await writeFile(join(dist,'index.html'),'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+css.map(name=>'<link rel="stylesheet" href="'+name+'">').join('')+'</head><body><div id="app"></div><script src="fixture.js"></script></body></html>');
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
    } else if (message.method === 'Page.javascriptDialogOpening') { dialogs.push(message.params.message); void cdp('Page.handleJavaScriptDialog', { accept: true }); }
    else if (message.method === 'Network.requestWillBeSent') downloads.push(message.params.request.url);
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

const field = label => `[...document.querySelectorAll('label')].find(el => el.textContent.startsWith(${JSON.stringify(label)}))?.querySelector('input,textarea')`;
const button = label => `[...document.querySelectorAll('button')].find(el => el.textContent.trim() === ${JSON.stringify(label)})`;
async function click(label) { await waitFor(button(label) + ' && !' + button(label) + '.matches(":disabled")'); await evaluate(button(label) + '.click()'); }
async function fill(label, value) { await waitFor(field(label)); await evaluate(`(() => {const el=${field(label)}; Object.getOwnPropertyDescriptor(el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,'value').set.call(el,${JSON.stringify(value)}); el.dispatchEvent(new Event('input',{bubbles:true}));})()`); }
async function reloadPage() {
  const previous = await evaluate('window.registryDocument');
  await cdp('Page.reload');
  await waitFor('window.registryDocument && window.registryDocument !== ' + JSON.stringify(previous));
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1100, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist,'index.html')).href });
  await fill('Experiment name','A recoverable question'); await fill('Notes','Unfinished creation note');
  assert.equal(await evaluate('document.querySelectorAll(".page-header").length'), 1);
  assert.equal(await evaluate('document.querySelector(".page-header [data-stage-action=back]").textContent'), 'Back to experiments');
  await click('Create & open inputs');
  await waitFor('document.body.innerText.includes("Retry creation")');
  const original = await evaluate('window.registryFixture.calls[0].input');
  assert.equal(await evaluate('Object.keys(window.registryFixture.created).length'),1);
  await click('Back to experiments'); await click('Resume creation');
  await waitFor('document.body.innerText.includes("Retry creation")');
  assert.equal(await evaluate(field('Notes')+'.value'),'Unfinished creation note');
  await reloadPage();
  await waitFor('document.body.innerText.includes("Retry creation")');
  assert.equal(await evaluate(field('Experiment name')+'.value'),'A recoverable question');
  await evaluate('window.registryFixture.loseNext = true; window.registryFixture.error = "server"');
  await click('Retry creation');
  await waitFor('document.body.innerText.includes("The upstream creation response is unavailable.")');
  assert.equal(await evaluate(field('Experiment name')+'.matches(":disabled")'),true);
  assert.deepEqual(await evaluate('window.registryFixture.calls[1].input'),original);
  await click('Retry creation');
  await waitFor('document.querySelector("#created")');
  assert.deepEqual(await evaluate('window.registryFixture.calls[2].input'),original);
  assert.equal(await evaluate('Object.keys(window.registryFixture.created).length'),1);
  assert.equal(await evaluate('window.registryFixture.successfulNavigations'),1);
  assert.deepEqual(dialogs,[], 'Successful acknowledgement must not prompt about an in-flight create');
  assert.equal(await evaluate("sessionStorage.getItem('histopilot:experiment-draft:v1:project:new:create')"),null);

  await evaluate('window.openMetadata()'); await fill('Notes','Local metadata edits');
  await click('Switch view'); await click('Switch view');
  await waitFor(field('Notes')+'.value === "Local metadata edits"');
  await evaluate('window.changeMetadata(true, true)');
  await waitFor(button('Save experiment details')+' && !'+button('Save experiment details')+'.disabled');
  assert.equal(await evaluate(field('Notes')+'.value'),'Local metadata edits','An unrelated batch revision discarded metadata');
  await evaluate('window.changeMetadata()');
  await click('Save experiment details');
  await waitFor('document.body.innerText.includes("Reload saved details")');
  assert.equal(await evaluate(field('Notes')+'.value'),'Local metadata edits');
  await reloadPage(); await waitFor(button('Create & open inputs'));
  await evaluate('window.openMetadata()'); await waitFor('document.body.innerText.includes("Reload saved details")');
  assert.equal(await evaluate(field('Notes')+'.value'),'Local metadata edits');
  await click('Reload saved details');
  await waitFor(field('Notes')+'.value === "Other tab note"');
  assert.equal(dialogs.length,1);
  assert.ok(dialogs[0].includes('Replace your unsaved experiment details'));
  await fill('Notes','Saved after recovery');
  await evaluate('window.delayMetadata = true');
  await click('Save experiment details');
  await waitFor('window.releaseMetadata');
  assert.equal(await evaluate(field('Notes')+'.matches(":disabled")'),true);
  await evaluate('window.releaseMetadata()');
  await waitFor('document.body.innerText.includes("Experiment details saved")');
  assert.equal(await evaluate('window.registryFixture.metadata.notes'),'Saved after recovery');
  assert.equal(await evaluate("sessionStorage.getItem('histopilot:experiment-draft:v1:project:record:metadata')"),null);
  assert.deepEqual(exceptions,[]);
  const report={passed:true,scope:'Real create and metadata forms; in-memory APIs; Chromium file fixture; no HistoPilot server.',checks:['creation response lost after acceptance','module unmount retains create details and pending operation','reload retains exact creation operation','HTTP 503 after acceptance retains exact retry operation','retry returns one original experiment','successful creation handoff has no false guard','metadata remount/reload retains raw fields','unrelated revisions preserve metadata edits','stale save rejects without losing text','explicit reload uses latest details','pending metadata controls disabled','successful save clears recovery'],calls:await evaluate('window.registryFixture.calls'),dialogs};
  await writeFile(join(output,'verification.json'),JSON.stringify(report,null,2));
  console.log('PASS: exact create retry, acknowledged navigation, metadata recovery and stale-save protection.');
  console.log('Artifacts: '+output);
} catch(error) {try{await writeFile(join(output,'failure.txt'),await evaluate('document.body.innerText'));}catch{} console.error('Artifacts: '+output); console.error('Dialogs: '+JSON.stringify(dialogs));console.error('Browser errors: '+JSON.stringify(exceptions));throw error;}
finally { browser.kill(); }
