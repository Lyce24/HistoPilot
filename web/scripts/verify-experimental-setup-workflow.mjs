/** Run with node scripts/verify-experimental-setup-workflow.mjs. Uses local Chromium and file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-experimental-setup-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React, {useState, useEffect} from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import {createRoot} from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import {QueryClient, QueryClientProvider} from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalExperiments from ${source('pages/LocalExperiments.tsx')};
import {experiments} from ${source('api/experiments.ts')};
import {scientific} from ${source('api/scientific.ts')};
import {targetSplits} from ${source('api/targetSplits.ts')};
import {bundles} from ${source('api/bundles.ts')};
import {development} from ${source('api/development.ts')};
import {predictors} from ${source('api/predictors.ts')};
import {experimentHeadlines} from ${source('api/experimentResults.ts')};
import {taskCenter} from ${source('api/taskCenter.ts')};
import {mil} from ${source('api/mil.ts')};
import {lifecycle} from ${source('api/lifecycle.ts')};
import {newDevelopmentSplit} from ${source('lib/protocol.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('components/StageWorkflow.css')};
const copy = value => structuredClone(value);
const target = {field:'grade',task:'binary_classification',unit:'patient',classes:['low','high'],labels:{low:'low',high:'high'},positiveClass:'high',missing:'block',unmapped:'block'};
const dataset = {id:'dataset-main',contentHash:'dataset-hash',versionLabel:{tag:'Bladder dataset'},manifest:{kind:'dataset',summary:{slideCount:80},dictionary:[{key:'grade',sourceColumn:'Grade',type:'categorical',owner:'patient'}]}};
const partition = {id:'target-main',contentHash:'target-hash',versionLabel:{tag:'Grade train/test'},manifest:{kind:'target-split',datasetId:dataset.id,spec:{datasetId:dataset.id,target,predictors:[],eligibility:[],split:{method:'random',testFraction:.2,seed:42,stratify:true,trainRules:[],testRules:[],trainValues:[],testValues:[]}},summary:{totalSlides:80,includedSlides:80,excludedSlides:0,trainingSlides:64,testingSlides:16,trainingPatients:32,testingPatients:8},memberships:[]}};
const bundle = {id:'bundle-main',contentHash:'bundle-hash',versionLabel:{tag:'Verified slide features'},current:true,findings:[],manifest:{kind:'feature-bundle',datasetId:'another-dataset',spec:{featureSetId:'features-main',packArtifactIds:[]},feature:{id:'features-main',featureSetId:'features-main',contentHash:'features-hash',sourceContentHash:'source-hash',validation:{current:true,tensorValidationComplete:true,provenanceComplete:true}},packs:[],summary:{slideCount:64,patchCount:640,dimensions:8,dtype:'float32'}}};
const protocol = {id:'protocol-derived',contentHash:'protocol-hash',versionLabel:{tag:'Derived training plans'},manifest:{kind:'protocol',datasetId:dataset.id,spec:{datasetId:dataset.id,target,predictors:[],eligibility:[],split:newDevelopmentSplit()},memberships:[],summary:{includedSlides:64,includedPatients:32}}};
const inputs = {protocolId:protocol.id,featureBundleId:bundle.id,loadingPolicy:'native',packArtifactId:null};
const resolved = {canPlan:true,findings:[],resolvedLoadingPolicy:'native',packArtifactId:null,featureSetId:'features-main',featureKind:'patch',bundleId:bundle.id,executionImplemented:true};
const record = (id,name,status='draft',stage='planning',frozen=false) => ({id,key:'draft:'+id,name,notes:'Browser fixture for separate setup and execution',tags:[],revision:1,state:'active',status,stage,legacy:false,setupVersion:1,setupStatus:frozen?'frozen':'draft',frozenSetupId:frozen?'frozen-'+id:null,configurationLocked:frozen,createdAt:'2026-09-25T00:00:00Z',updatedAt:'2026-09-25T00:00:00Z',inputs:null,setupDesign:null,batches:[],drafts:[],batchPlans:[],predictorId:null,executionImplemented:true});
const state = window.workflow = {calls:[],errors:[],records:[record('setup-main','Bladder training study')],execution:null};
experimentHeadlines.list = async () => ({items:[]});
window.fetch = async (...args) => { if (String(args[0]).endsWith('/api/v1/session')) return new Response(JSON.stringify({token:'fixture-token'}),{status:200,headers:{'Content-Type':'application/json'}}); const message='Unexpected request '+args[0];state.errors.push(message);throw new Error(message); };
const replace = value => { state.records=state.records.map(item=>item.id===value.id?value:item);return copy(value); };
experiments.summaries = async () => ({items:copy(state.records)});
experiments.get = async (_,id) => copy(state.records.find(item=>item.id===id));
experiments.setupInputs = async (_,id,input) => {state.calls.push({method:'setupInputs',input:copy(input)});const item=state.records.find(item=>item.id===id);if(item.revision!==input.expectedRevision)throw new Error('Stale setup');protocol.manifest.spec.split=copy(input.trainingSplit);return replace({...item,revision:item.revision+1,inputs:{...inputs,featureBundleId:input.featureBundleId,loadingPolicy:input.loadingPolicy,packArtifactId:input.packArtifactId},setupDesign:{datasetId:input.datasetId,targetSplitId:input.targetSplitId,trainingSplit:copy(input.trainingSplit)},inputSnapshot:{dataset:copy(dataset),protocol:copy(protocol),featureBundle:copy(bundle)}});};
experiments.update = async (_,id,input) => {state.calls.push({method:'update',input:copy(input)});const item=state.records.find(item=>item.id===id);if(item.configurationLocked)throw new Error('Frozen setup edited');if(item.revision!==input.expectedRevision)throw new Error('Stale update');return replace({...item,...copy(input),revision:item.revision+1});};
experiments.freezeSetup = async (_,id,input) => {state.calls.push({method:'freezeSetup',input:copy(input)});const item=state.records.find(item=>item.id===id);if(item.revision!==input.expectedRevision||!item.setupDesign||!item.batchPlans.length)throw new Error('Invalid freeze');return replace({...item,revision:item.revision+1,status:'ready',configurationLocked:true,setupStatus:'frozen',frozenSetupId:'frozen-setup-main',frozenSetup:{id:'frozen-setup-main',manifest:{kind:'experiment-setup',datasetId:dataset.id,setupDesign:copy(item.setupDesign),batchPlans:copy(item.batchPlans)}}});};
function batchManifest(spec){const splitPlans=protocol.manifest.spec.split.seeds.flatMap(seed=>[0,1,2].map(fold=>({id:'split-'+seed+'-'+fold,planId:'split-'+seed+'-'+fold,seed,fold,slideCount:64,partitions:{training:34,validation:8,assessment:22}})));return {kind:'mil-batch',version:1,datasetId:dataset.id,spec:copy(spec),configurations:[{id:'candidate-main',number:1,recipe:copy(spec.recipe)}],splitPlans,runs:splitPlans.map(split=>({id:'run-'+split.id,candidateId:'candidate-main',trainingSeed:42,splitPlanId:split.id,status:'planned'})),summary:{configurationCount:1,trainingSeedCount:1,splitPlanCount:splitPlans.length,runCount:splitPlans.length},executionImplemented:true,previewHash:'batch-preview',resolvedInputs:resolved};}
experiments.submit = async (_,id,input) => {state.calls.push({method:'submit',input:copy(input)});const item=state.records.find(item=>item.id===id);if(!item.frozenSetupId||item.status!=='ready'||item.revision!==input.expectedRevision)throw new Error('Only a frozen ready setup can start');const manifest=batchManifest(item.batchPlans[0].spec);state.execution={batchId:'batch-main',status:'queued',sessionName:null,logPath:null,outputPath:null,findings:[],runCounts:{total:manifest.runs.length,queued:manifest.runs.length,running:0,completed:0,failed:0,cancelled:0},runs:manifest.runs.map(run=>({...run,status:'queued'})),createdAt:'2026-09-25T00:00:00Z',updatedAt:'2026-09-25T00:00:01Z'};return replace({...item,revision:item.revision+1,status:'queued',stage:'running',batches:[{id:'batch-main',key:'configuration:batch-main',name:'Setup baseline',state:'active',status:'queued',manifest,execution:copy(state.execution)}],submission:{...input,status:'submitted',batchIds:['batch-main'],submittedAt:'2026-09-25T00:00:01Z',error:null,retryable:false}});};
scientific.datasets = async () => ({datasets:[copy(dataset)]});
scientific.configurations = async (_,kind) => ({configurations:kind==='protocol'?[copy(protocol)]:kind==='target-split'?[copy(partition)]:[]});
scientific.configuration = async (_,id) => id==='features-main'?{id,manifest:{kind:'feature',spec:{featureKind:'patch'}}}:copy(protocol);
targetSplits.list = async () => ({configurations:[copy(partition)]});
bundles.list = async () => ({items:[copy(bundle)]});
mil.preview = async (_,spec) => {state.calls.push({method:'legacyInputPreview',spec:copy(spec)});return copy(resolved);};
development.runtime = async () => ({available:true,python:'/fixture/python',versions:{},cudaAvailable:false,gpuCount:0,host:{cpuCount:8,totalRamGb:32,availableRamGb:24},gpus:[],findings:[]});
development.preview = async (_,spec) => {state.calls.push({method:'batchPreview',spec:copy(spec)});return {...batchManifest(spec),canFreeze:true,findings:[]};};
development.execution = async () => copy(state.execution);
development.history = async (_,__,runId) => ({runId,rows:[],totalRows:0,truncated:false});
development.resourceHistory = async () => ({batchId:'batch-main',rows:[],totalRows:0,truncated:false});
development.results = async () => ({batchId:'batch-main',status:'queued',findings:[],oof:[],candidates:[]});
predictors.list = async () => ({items:[]});
taskCenter.rollup = async (scope) => ({ scope: scope ?? {}, state: 'not-started', counts: {}, byKind: {}, progress: null, live: 0, active: 0, pending: 0, held: false, position: null, queuePosition: null, waitingReason: null, eta: null, runnerAlive: true, paused: false, stopRequest: null, lastFailure: null, recentFailures: 0, current: null, startedAt: null, finishedAt: null, ownerKey: null, ownerKind: null, ownerId: null, title: null, projectId: null, projectName: null, href: '#task-center', updatedAt: '2026-09-25T10:05:00Z' });
taskCenter.owners = async () => ({owners:[]});
lifecycle.inventory = async () => ({projectId:'project',revision:1,projectState:'active',items:state.records.map(item=>({key:item.key,id:item.id,type:'draft',kind:'model-experiment',name:item.name,state:'active',createdAt:item.createdAt,dependsOn:[],usedBy:[]})),audit:[],note:''});
const client = new QueryClient({defaultOptions:{queries:{retry:false,staleTime:Infinity}}});
window.refreshWorkflow = () => client.invalidateQueries();
window.addStatusFixtures = async () => {state.records.push(...[['ready','planning'],['queued','running'],['scheduled','running'],['running','running'],['completed','finished'],['failed','running'],['interrupted','running'],['cancelled','finished']].map(([status,stage])=>record('case-'+status,'Case '+status,status,stage,true)),record('hidden-draft','Unfrozen draft'));await client.invalidateQueries();};
function Fixture(){const [hash,setHash]=useState(location.hash);useEffect(()=>{const update=()=>setHash(location.hash);addEventListener('hashchange',update);addEventListener('popstate',update);return()=>{removeEventListener('hashchange',update);removeEventListener('popstate',update);};},[]);const mode=hash.startsWith('#experimental-setup')?'setup':'execution';return <QueryClientProvider client={client}><main className="stage-workspace"><LocalExperiments key={mode} mode={mode} workspace={{project:{id:'project',name:'Bladder study',config:{}}}}/></main></QueryClientProvider>;}
createRoot(document.getElementById('app')).render(<Fixture/>);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
  outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
  lib: { entry: fixture, name: 'ExperimentalSetupFixture', formats: ['iife'], fileName: () => 'fixture.js' },
} });
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
  + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('')
  + '<style>body{padding:24px}#app{max-width:1320px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');

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
let acceptDialogs = true;
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
    } else if (message.method === 'Page.javascriptDialogOpening') { dialogs.push(message.params.message); void cdp('Page.handleJavaScriptDialog', { accept: acceptDialogs }); }
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
  const response = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true, userGesture: true });
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
const field = (text, selector = 'input,select,textarea') => '[...document.querySelectorAll("label")].find(el => !el.closest("[hidden]") && el.textContent.trim().startsWith(' + JSON.stringify(text) + '))?.querySelector(' + JSON.stringify(selector) + ')';
async function click(text) {
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
}
async function fill(text, value, selector = 'input,select,textarea') {
  const element = field(text, selector);
  await waitFor(element, 'field ' + text);
  await evaluate('(() => { const el = ' + element + '; const prototype = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(prototype, "value").set.call(el, ' + JSON.stringify(value) + '); el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true })); })()');
}
async function waitForDialog(count) {
  for (let attempt = 0; attempt < 100 && dialogs.length < count; attempt++) await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(dialogs.length, count, JSON.stringify(dialogs));
}
async function screenshot(name) {
  await new Promise(resolve => setTimeout(resolve, 220));
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
async function step(label, title) {
  const element = '[...document.querySelectorAll("nav")].find(el => !el.closest("[hidden]") && el.getAttribute("aria-label") === ' + JSON.stringify(label) + ')';
  const match = '[...' + element + '.querySelectorAll("button")].find(el => (el.querySelector("strong")?.textContent ?? el.textContent) === ' + JSON.stringify(title) + ')';
  await waitFor(match + ' && !' + match + '.matches(":disabled")');
  await evaluate(match + '.click()');
}
try {
  const target = await cdp('Target.createTarget', {url:'about:blank'}, null);
  ({sessionId} = await cdp('Target.attachToTarget', {targetId:target.targetId,flatten:true}, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1080,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate', {url:pathToFileURL(join(dist,'index.html')).href+'#experimental-setup?experiment=setup-main'});
  await waitFor('document.body?.innerText.includes("Dataset, features, targets and splits")');
  await fill('Dataset version','dataset-main','select');
  await fill('Targets & splits','target-main','select');
  await fill('Feature bundle','bundle-main','select');
  await fill('Feature loading','native','select');
  await fill('Early-stop validation','20');
  await fill('Number of folds','3');
  await fill('Split seeds','11, 12');
  await waitFor('document.body.innerText.includes("64 training slides")');
  assert.equal(await evaluate('window.workflow.calls.length'),0,'Selection alone must not save or start jobs');
  await screenshot('01-setup-inputs-desktop');
  await cdp('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await screenshot('01-setup-inputs-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth+1'),true,'Setup inputs overflow mobile viewport');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1080,deviceScaleFactor:1,mobile:false});
  await click('Check & continue to hyperparameters');
  await waitFor('window.workflow.calls.some(call=>call.method==="setupInputs")');
  await waitFor('document.body.innerText.includes("Start from a template") || document.body.innerText.includes("Add training batch")');
  const setup = await evaluate('window.workflow.calls.find(call=>call.method==="setupInputs").input');
  assert.equal(setup.datasetId,'dataset-main');
  assert.equal(setup.targetSplitId,'target-main');
  assert.equal(setup.featureBundleId,'bundle-main');
  assert.equal(setup.loadingPolicy,'native');
  assert.equal(setup.trainingSplit.version,4);
  assert.equal(setup.trainingSplit.folds,3);
  assert.equal(setup.trainingSplit.validationFraction,.2);
  assert.deepEqual(setup.trainingSplit.seeds,[11,12]);
  assert.equal(await evaluate('window.workflow.calls.some(call=>call.method==="submit"||call.method==="freezeSetup"||call.method==="legacyInputPreview")'),false);
  if (await evaluate('Boolean('+button('Add training batch')+')')) await click('Add training batch');
  await fill('Start from a template','baseline','select');
  await fill('Batch name','Setup baseline');
  await click('Continue to training settings');
  await fill('Maximum epochs','7');
  await click('Continue to predictors');
  assert.equal(await evaluate('document.body.innerText.includes("Run on")'),false,'Compute settings left the batch editor');
  await click('Continue to batch review');
  await click('Check batch');
  await waitFor('document.body.innerText.includes("Resolved batch")');
  await click('Add batch to plan');
  await waitFor('document.body.innerText.includes("Batch plans (1)")');
  await step('Experimental setup','Review & freeze');
  await waitFor('document.body.innerText.includes("Freeze experimental setup")');
  assert.equal(await evaluate('window.workflow.calls.filter(call=>call.method==="submit").length'),0);
  await screenshot('02-review-setup-desktop');
  await click('Freeze experimental setup');
  await waitFor('document.body.innerText.includes("Frozen experimental setup")');
  assert.equal(await evaluate('window.workflow.calls.filter(call=>call.method==="freezeSetup").length'),1);
  assert.equal(await evaluate('window.workflow.calls.filter(call=>call.method==="submit").length'),0,'Freeze setup must not submit compute');
  assert.equal(await evaluate('window.workflow.records[0].status'),'ready');
  assert.equal(await evaluate('window.workflow.records[0].configurationLocked'),true);
  await step('Experimental setup','Inputs & training design');
  await waitFor('document.querySelector(".science-fieldset")?.disabled');
  assert.equal(await evaluate('[...document.querySelectorAll(".science-fieldset input,.science-fieldset select")].every(el=>el.matches(":disabled"))'),true,'Frozen setup inputs must be locked');
  assert.equal(await evaluate('Boolean('+button('Check & continue to hyperparameters')+')'),false);
  await screenshot('03-frozen-inputs-desktop');
  await step('Experimental setup','Hyperparameters');
  assert.equal(await evaluate('Boolean('+button('Add training batch')+')'),false,'Frozen recipes cannot be added');
  assert.equal(await evaluate('Boolean('+button('Edit batch')+')'),false,'Frozen recipes cannot be edited');
  await evaluate('location.hash="#experiments"');
  await waitFor('document.querySelector(".experiment-record-table")?.innerText.includes("Ready to run")');
  assert.equal(await evaluate('document.body.innerText.includes("Create setup")'),false,'Execution must not create setups');
  await screenshot('04-ready-experiments-desktop');
  await click('Bladder training study');
  await waitFor('document.body.innerText.includes("Start experiment")');
  assert.equal(await evaluate('document.body.innerText.includes("6 fold runs")'),true,'Frozen setup review retains planned run counts');
  assert.equal(await evaluate('document.querySelector(".experiment-context")?.innerText.includes("Grade train/test")'),true,'Execution identifies the frozen targets separately from derived training plans');
  assert.equal(await evaluate('[...document.querySelectorAll("a")].some(el=>el.textContent.trim()==="View frozen setup" && el.getAttribute("href").startsWith("#experimental-setup?"))'),true,'Execution links to frozen setup');
  assert.equal(await evaluate('[...document.querySelectorAll("button")].some(el=>!el.closest("[hidden]") && el.textContent.trim()==="Back to batches")'),false,'Execution has no inert preparation button');
  assert.equal(await evaluate('window.workflow.calls.filter(call=>call.method==="submit").length'),0);
  assert.equal(await evaluate('Boolean('+button('Freeze experimental setup')+')'),false);
  await screenshot('05-explicit-start-desktop');
  await click('Start experiment');
  await waitFor('window.workflow.calls.filter(call=>call.method==="submit").length===1');
  await waitFor('window.workflow.records[0].status==="queued"');
  assert.equal(await evaluate('window.workflow.calls.at(-1).method'),'submit');
  await evaluate('location.hash="#experiments"; window.addStatusFixtures()');
  await waitFor('document.querySelector(".experiment-record-table")?.innerText.includes("Case completed")');
  const names = () => evaluate('[...document.querySelectorAll(".experiment-name")].map(el=>el.textContent.trim()).sort()');
  assert.equal((await names()).includes('Unfrozen draft'),false,'Unfrozen drafts must be absent from execution registry');
  const expected = {
    ready:['Case ready'], queued:['Bladder training study','Case queued','Case scheduled'],
    running:['Case running'], completed:['Case completed'],
    'needs-attention':['Case failed', 'Case interrupted'], cancelled:['Case cancelled'],
  };
  for (const [status,records] of Object.entries(expected)) {
    await fill('Status',status,'select');
    await waitFor('document.querySelectorAll(".experiment-name").length==='+records.length,'filtered '+status);
    assert.deepEqual(await names(),records.sort(),'Exact execution status filter '+status);
  }
  await fill('Status','running','select');
  await screenshot('06-running-filter-desktop');
  await fill('Status','','select');
  await cdp('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await screenshot('06-execution-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth+1'),true,'Execution registry overflows mobile viewport');
  assert.deepEqual(await evaluate('window.workflow.errors'),[]);
  assert.deepEqual(exceptions,[]);
  assert.deepEqual(dialogs,[]);
  await writeFile(join(output,'verification.json'),JSON.stringify({passed:true,scope:'Real React setup, frozen design, execution and registry components; APIs mocked; file:// Chromium; no server or training jobs.',calls:await evaluate('window.workflow.calls'),statusFilters:expected},null,2));
  console.log('PASS: dataset/targets/features/training design saved through setupInputs, recipe saved, setup frozen without compute submission, frozen inputs and batches locked, ready execution entry, explicit Start experiment submits once, exact execution status filters, and desktop/mobile layouts.');
  console.log('Artifacts: '+output);
} catch(error) {
  try {await writeFile(join(output,'failure.txt'),await evaluate('document.body.innerText'));await screenshot('failure');} catch {}
  if(exceptions.length)console.error(JSON.stringify(exceptions,null,2));
  console.error('Artifacts: '+output);throw error;
} finally {browser.kill();}
