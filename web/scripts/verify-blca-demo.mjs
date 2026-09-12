/** Verify the full BLCA demo in Chromium with the packaged synthetic workspace; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';
const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-blca-demo-'));
const workspace = JSON.parse(await readFile(join(web, '../histopilot/resources/blca_demo_workspace.json'), 'utf8'));
const fixture = join(output, 'fixture.tsx');
const icon='data:image/svg+xml;base64,'+Buffer.from(await readFile(join(web,'public/favicon.svg'))).toString('base64');
const source = path => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web,'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web,'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web,'node_modules/@tanstack/react-query/build/modern/index.js'))};
import App from ${source('App.tsx')};
import { api } from ${source('api/client.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('roadmap.css')};
import ${source('components/StageWorkflow.css')};
const workspace=${JSON.stringify(workspace)};
new MutationObserver(()=>{for(const image of document.querySelectorAll('img[src^="/favicon.svg"]'))image.src=${JSON.stringify(icon)};}).observe(document,{childList:true,subtree:true});
window.demoVerification={calls:[],errors:[],exports:[]};
window.fetch=async (...args)=>{window.demoVerification.errors.push(String(args[0]));throw new Error('Unexpected network request: '+args[0]);};
api.projects=async()=>{window.demoVerification.calls.push('projects');return {projects:[workspace.project],defaultStoragePath:'demo://blca'};};
api.projectWorkspace=async id=>{if(id!=='blca-demo-v1')throw new Error('Unexpected project');window.demoVerification.calls.push('workspace');return structuredClone(workspace);};
const createURL=URL.createObjectURL.bind(URL);
URL.createObjectURL=blob=>{blob.text().then(text=>window.demoVerification.exports.push(text));return createURL(blob);};
const client=new QueryClient({defaultOptions:{queries:{retry:false,staleTime:Infinity}}});
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><App/></QueryClientProvider>);
`);
await build({configFile:false,root:web,logLevel:'error',plugins:[react()],define:{'process.env.NODE_ENV':JSON.stringify('production')},resolve:{alias:{'react/jsx-runtime':join(web,'node_modules/react/jsx-runtime.js')}},build:{outDir:join(output,'dist'),emptyOutDir:true,minify:false,lib:{entry:fixture,name:'BlcaDemoFixture',formats:['iife'],fileName:()=> 'fixture.js'}}});
const dist=join(output,'dist');
const css=(await readdir(dist)).filter(name=>name.endsWith('.css'));
await writeFile(join(dist,'index.html'),'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+css.map(name=>'<link rel="stylesheet" href="'+name+'">').join('')+'</head><body><div id="app"></div><script src="fixture.js"></script></body></html>');
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
async function waitFor(expression, description = expression) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await evaluate('Boolean(' + expression + ')')) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('Timed out: ' + description + '\n' + await evaluate('document.body.innerText'));
}
const button = (text) => '[...document.querySelectorAll("button")].find(el => !el.closest("[hidden]") && el.textContent.trim() === ' + JSON.stringify(text) + ')';
async function click(text) {
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
}
async function screenshot(name) {
  await new Promise(resolve => setTimeout(resolve, 220));
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Verification steps follow the generated demo records.

async function navigate(module) {
  await evaluate(`document.querySelector('.sidebar a[href="#${module}"]').click()`);
  await waitFor('document.querySelector(".stage-library") && document.querySelector(".blca-demo-record-link")');
}
async function fill(selector, value) {
  await evaluate(`(() => {const input=document.querySelector(${JSON.stringify(selector)});const proto=input instanceof HTMLSelectElement?HTMLSelectElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(input,${JSON.stringify(value)});input.dispatchEvent(new Event(input instanceof HTMLSelectElement?'change':'input',{bubbles:true}));})()`);
}
async function step(title) {
  await evaluate(`([...document.querySelectorAll('.stage-steps button')].find(button=>button.querySelector('strong').textContent===${JSON.stringify(title)})).click()`);
  await waitFor(`document.querySelector('.blca-demo-step>h2')?.textContent===${JSON.stringify(title)}`);
  assert.equal(await evaluate('document.querySelectorAll(".blca-demo-step").length'),1,'Each substage must replace the previous page');
}
try {
  const target=await cdp('Target.createTarget',{url:'about:blank'},null);
  ({sessionId}=await cdp('Target.attachToTarget',{targetId:target.targetId,flatten:true},null));
  await cdp('Page.enable');await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(join(dist,'index.html')).href});
  await click('Open BLCA demo');
  await waitFor('document.querySelector(".blca-demo-hero h1")?.textContent==="BLCA demo"');
  assert.equal(await evaluate('document.querySelectorAll(".blca-demo-stage").length'),8);
  assert.equal(await evaluate('document.body.innerText.includes("138")'),true);
  assert.equal(await evaluate('document.querySelector(".job-tray")===null'),true,'Demo must not expose active compute controls');
  await screenshot('01-blca-overview');
  const visited=[];
  for(const record of workspace.demoPipeline.records) {
    await navigate(record.module);
    await click(record.name);
    await waitFor(`document.querySelector('.blca-demo-record-heading h2')?.textContent===${JSON.stringify(record.name)}`);
    for(const current of record.steps) {
      await step(current.title);
      assert.equal(await evaluate('document.querySelector(".blca-demo-notice").innerText.includes("Synthetic")'),true);
      if(current.runs) {
        await waitFor('document.querySelector(".experiment-history-charts svg") && document.querySelectorAll(".run-resource-chart").length===4');
        const ordered=await evaluate(`Boolean(document.querySelector('.experiment-run-tracker').compareDocumentPosition(document.querySelector('.run-resource-usage'))&4)&&Boolean(document.querySelector('.run-resource-usage').compareDocumentPosition(document.querySelector('.blca-demo-table-section'))&4)`);
        assert.equal(ordered,true,'Run details precede resources, which precede predictor creation');
        await screenshot(record.id+'-runs');
        await click('Diagnostics');
        await waitFor('document.querySelector(".experiment-run-panel:not([hidden])").innerText.includes("Learning rate")');
        await click('Checkpoints');
        await waitFor('document.querySelector(".experiment-run-panel:not([hidden])").innerText.includes("Held-out assessment")');
        await click('Overview');
        const previousExports=await evaluate('window.demoVerification.exports.length');
        await click('Export synthetic epoch history');
        await waitFor('window.demoVerification.exports.length>'+previousExports);
        const historyExport=JSON.parse(await evaluate('window.demoVerification.exports.at(-1)'));
        assert.equal(historyExport.synthetic,true);
        assert.equal(historyExport.executable,false);
        assert.equal(historyExport.runId.startsWith('blca-demo-'),true);
        await fill('.experiment-run-search input','Fold 2');
        await waitFor('document.querySelectorAll(".experiment-runs-table tbody tr").length===1');
        assert.equal(await evaluate('document.querySelector(".experiment-run-detail h3").innerText.includes("Fold 2")'),true);
        await fill('.experiment-run-search input','');
      }
      if(record.module==='evaluation' && current.chart)await screenshot('evaluation-'+current.id);
      if(record.module==='clinical-utility' && current.chart)await screenshot('clinical-'+current.id);
      visited.push(record.id+':'+current.id);
    }
    await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),true,record.id+' mobile overflow');
    await screenshot(record.id+'-mobile');
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
  }
  await navigate('experiments');
  await fill('.stage-library-search input','no-such-demo-record');
  await waitFor('document.body.innerText.includes("No examples match these filters")');
  await click('Clear filters');
  const experiments=workspace.demoPipeline.records.filter(record=>record.module==='experiments');
  await fill('.stage-library-search input',experiments[1].name);
  await waitFor('document.querySelectorAll(".blca-demo-record-link").length===1');
  await click(experiments[1].name);
  await click('← Back to records');
  assert.equal(await evaluate('document.querySelector(".stage-library-search input").value'),experiments[1].name,'Return to library must preserve search');
  await click('Clear filters');
  await fill('.stage-library-toolbar select',experiments[1].tags[0]);
  await waitFor('document.querySelectorAll(".blca-demo-record-link").length>0');
  await click('Clear filters');
  await click(experiments[1].name);
  await step('Runs');
  assert.equal(await evaluate('location.hash.includes("record=blca-baseline-v3")&&location.hash.includes("step=runs")'),true);
  assert.deepEqual(await evaluate('window.demoVerification.errors'),[]);
  const initialRequests=await evaluate('window.demoVerification.calls');
  await cdp('Page.reload');
  await waitFor('document.querySelector("[data-demo-step=runs]") && document.querySelector(".experiment-history-charts svg")');
  await step('Results');
  await evaluate('history.back()');
  await waitFor('document.querySelector("[data-demo-step=runs]")');
  await evaluate('history.forward()');
  await waitFor('document.querySelector("[data-demo-step=results]")');
  // Reload gives a fresh fixture, so earlier exports are intentionally gone.
  await evaluate(`document.querySelector('button[aria-label="Export workspace"]').click()`);
  await waitFor('window.demoVerification.exports.length===1');
  const exported=JSON.parse(await evaluate('window.demoVerification.exports[0]'));
  assert.equal(exported.demoPipeline.synthetic,true);
  assert.equal(exported.executionEnabled,false);
  assert.equal(exported.project.id,'blca-demo-v1');
  assert.equal(JSON.stringify(exported).includes('/home/'),false);
  assert.deepEqual(await evaluate('window.demoVerification.errors'),[]);
  assert.deepEqual(exceptions,[]);
  const result={passed:true,scope:'Full application UI + packaged deterministic synthetic workspace; no live server or project data',visited,requests:[...initialRequests,...await evaluate('window.demoVerification.calls')],output};
  await writeFile(join(output,'verification.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
} catch(error) {
  try{await writeFile(join(output,'failure.txt'),await evaluate('document.body.innerText'));await screenshot('failure');}catch{}
  console.error('Artifacts:',output);if(exceptions.length)console.error(JSON.stringify(exceptions));throw error;
} finally {browser.kill();}
