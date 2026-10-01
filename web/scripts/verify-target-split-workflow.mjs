/** Isolated React/Chromium regression using file:// and mocked APIs. Starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';
const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-target-split-'));
const artifacts = join(output, 'artifacts');
await mkdir(artifacts);
const source = (path) => JSON.stringify(join(web, 'src', path));
const fixture = join(output, 'fixture.tsx');
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalTargetSplit from ${source('pages/LocalTargetSplit.tsx')};
import { scientific } from ${source('api/scientific.ts')};
import { targetSplits } from ${source('api/targetSplits.ts')};
import { ApiError } from ${source('api/client.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('components/StageWorkflow.css')};
const copy = value => structuredClone(value);
const state = window.workflow = { calls: [], errors: [], drafts: [], records: [], failFreeze: true, nextPartitionDelay: 0 };
window.fetch = async (...args) => { state.errors.push('Unexpected request: ' + args[0]); throw new Error(state.errors.at(-1)); };
const rows=Array.from({length:20},(_,i)=>{const patient=Math.floor(i/2);return {slideId:'slide-'+i,patientId:'patient-'+patient,patientIdSource:'provided',attributes:{grade:patient%2?'high':'low',test_grade:patient%2?'H':'L',site:patient<6?'A':'B',partition:patient<8?'development':'heldout',Requested_Split:patient<6?'train':'test'}};});
const dictionary = ['grade','test_grade','partition','site','Requested_Split'].map(key => ({key,sourceColumn:key,owner:'patient',type:'categorical'}));
const dataset = {id:'dataset-a',projectId:'project',contentHash:'data-hash',createdAt:'2026-09-25T12:00:00Z',artifacts:{},versionLabel:{tag:'Study slides'},manifest:{dictionary,summary:{slideCount:20,unlinkedSlideCount:0,fallbackSlideCount:0}}};
scientific.datasets = async () => ({datasets:[dataset]});
scientific.drafts = async () => ({drafts:copy(state.drafts)});
scientific.configurations = async () => ({configurations:[]});
scientific.draft = async (_, id) => copy(state.drafts.find(draft=>draft.id===id));
scientific.saveDraft = async (_, input, current) => { state.calls.push({method:'saveDraft',input:copy(input)}); const saved={id:current?.id??'draft-'+(state.drafts.length+1),projectId:'project',...copy(input),revision:(current?.revision??0)+1,status:'editable',createdAt:'2026-09-25T12:00:00Z',updatedAt:'2026-09-25T12:00:00Z'}; state.drafts=[saved,...state.drafts.filter(item=>item.id!==saved.id)]; return copy(saved); };
const counts=values=>Object.fromEntries([...new Set(values)].map(value=>[value,values.filter(item=>item===value).length]));
scientific.queryDataset = async (_,id,query) => { state.calls.push({method:'queryDataset',id,field:query.field}); const valueCounts=Object.entries(counts(rows.map(row=>row.attributes[query.field]??row[query.field==='Patient_ID'?'patientId':'slideId']))).map(([value,count])=>({value,count}));return {records:rows,total:20,totalSlides:20,offset:0,limit:1,summary:{slideCount:20,mappedPatientCount:10,unlinkedSlideCount:0},distribution:{kind:'categorical',unit:'slide',counts:valueCounts,missingCount:0,total:20},valueCounts,valuesTruncated:false,valueCountsUnit:'slide'}; };
function match(row,condition){if(condition.conditions)return condition.op==='any'?condition.conditions.some(c=>match(row,c)):condition.conditions.every(c=>match(row,c));const value=row.attributes[condition.field];if(condition.op==='eq')return value===condition.value;if(condition.op==='in')return condition.value.includes(value);return true;}
function cohortStats(items){return {totalSlides:items.length,patientCount:new Set(items.map(row=>row.patientId)).size,fallbackSlideCount:0,unlinkedSlideCount:0,groupCount:new Set(items.map(row=>row.patientId)).size,sample:items.slice(0,5)};}
scientific.exploreProtocol=async(_,request)=>{state.calls.push({method:'exploreProtocol',request:copy(request)});const eligible=rows.filter(row=>(request.eligibility??[]).every(c=>match(row,c)));return {datasetId:request.datasetId,dataset:cohortStats(rows),cohort:cohortStats(eligible),target:null,partitions:null,unassigned:null,findings:[],valid:true,splitMode:'kfold'};};
function selection(spec){const eligible=rows.filter(row=>(spec.eligibility??[]).every(c=>match(row,c)));const split=spec.split;let train=[],test=[];
 if(split.method==='rules'){test=eligible.filter(row=>split.testRules.length&&split.testRules.every(c=>match(row,c)));train=eligible.filter(row=>split.trainRules.length?split.trainRules.every(c=>match(row,c)):!test.includes(row));}
 else if(split.method==='imported'){train=eligible.filter(row=>split.trainValues.includes(row.attributes[split.partitionField]));test=eligible.filter(row=>split.testValues.includes(row.attributes[split.partitionField]));}
 else{const patients=[...new Set(eligible.map(row=>row.patientId))];const testPatients=patients.slice(patients.length-Math.round(patients.length*split.testFraction));test=eligible.filter(row=>testPatients.includes(row.patientId));train=eligible.filter(row=>!testPatients.includes(row.patientId));}
 return {eligible,train,test};}
function partition(items,field,target){const patients=[...new Set(items.map(row=>row.patientId))];const result={slides:items.length,patients:patients.length,groups:patients.length,fallbackSlides:0,unlinkedSlides:0};if(!field)return result;const values=[...new Set(items.map(row=>row.attributes[field]??null))];result.target={field,values:values.map(value=>({value,slides:items.filter(row=>(row.attributes[field]??null)===value).length,patients:new Set(items.filter(row=>(row.attributes[field]??null)===value).map(row=>row.patientId)).size})),distinctCount:values.length,classCounts:target?counts(items.map(row=>target.labels[row.attributes[field]]).filter(Boolean)):{},patientClassCounts:target?counts(patients.map(id=>target.labels[items.find(row=>row.patientId===id).attributes[field]]).filter(Boolean)):{},missingSlides:0,unmappedSlides:0,mixedValuePatients:0,mixedLabelPatients:0,missingLabelPatients:0,unmappedLabelPatients:0,labeledPatients:target?patients.length:0,unlabeledPatients:target?0:patients.length};return result;}
function exploration(spec){const {eligible,train,test}=selection(spec);const testing=spec.testTarget===undefined?spec.target:spec.testTarget;const partitions={train:partition(train,spec.targetFields?.train??spec.target?.field,spec.target),test:partition(test,spec.testTarget===null?null:spec.targetFields?.test??testing?.field,testing)};for(const [role,items] of [['train',train],['test',test]]){const rules=role==='train'?spec.split.trainRules:spec.split.testRules;const mode=spec.split.method==='rules'?(rules.length?'rules':role==='train'?'remaining':'none'):spec.split.method;partitions[role].selection={mode,directMatches:cohortStats(items),expanded:cohortStats(items),assigned:cohortStats(items)};}return {dataset:cohortStats(rows),cohort:cohortStats(eligible),summary:{totalSlides:20,eligibleSlides:eligible.length,selectedSlides:train.length+test.length,excludedSlides:20-train.length-test.length,eligibilityExcludedSlides:20-eligible.length,partitionExcludedSlides:eligible.length-train.length-test.length,trainingSlides:train.length,testingSlides:test.length,trainingPatients:partitions.train.patients,testingPatients:partitions.test.patients},partitions,findings:[],valid:train.length>0,algorithm:'fixture-patient-partition'};}
targetSplits.partitionPreview=async(_,request)=>{state.calls.push({method:'partitionPreview',request:copy(request)});const result=exploration(request);const delay=state.nextPartitionDelay;state.nextPartitionDelay=0;if(delay)await new Promise(resolve=>setTimeout(resolve,delay));return result;};
function preview(spec){const {train,test}=selection(spec);const value=exploration(spec);const target=spec.testTarget===undefined?spec.target:spec.testTarget;const memberships=[...train.map(row=>({...row,label:spec.target.labels[row.attributes[spec.target.field]],partition:'train'})),...test.map(row=>({...row,label:target?target.labels[row.attributes[target.field]]:null,partition:'test'}))];return {spec:copy(spec),summary:{...value.summary,includedSlides:memberships.length,includedPatients:value.summary.trainingPatients+value.summary.testingPatients,trainingGroups:value.summary.trainingPatients,testingGroups:value.summary.testingPatients,classCounts:counts(memberships.map(row=>row.label).filter(Boolean)),trainingClassCounts:value.partitions.train.target.classCounts,testingClassCounts:value.partitions.test.target?.classCounts??{},trainingPatientClassCounts:value.partitions.train.target.patientClassCounts,testingPatientClassCounts:value.partitions.test.target?.patientClassCounts??{}},partitions:value.partitions,memberships,findings:[],canFreeze:true,previewHash:'preview-'+JSON.stringify(spec)};}
targetSplits.list=async()=>({configurations:copy(state.records)});
targetSplits.get=async(_,id)=>copy(state.records.find(item=>item.id===id));
targetSplits.preview=async(_,id,revision)=>{state.calls.push({method:'preview',id,revision});return preview(state.drafts.find(draft=>draft.id===id).payload.spec);};
targetSplits.freeze=async(_,id,revision,hash,label,operationId)=>{state.calls.push({method:'freeze',id,revision,hash,label:copy(label),operationId});if(state.failFreeze){state.failFreeze=false;throw new ApiError('Try another version tag.',409,'VERSION_LABEL_CONFLICT');}const draft=state.drafts.find(item=>item.id===id);const value=preview(draft.payload.spec);const record={id:'target-split-fixed',projectId:'project',createdAt:'2026-09-25T12:00:00Z',contentHash:'fixed',versionLabel:{...copy(label),revision:1,createdAt:'',updatedAt:''},manifest:{kind:'target-split',datasetId:draft.payload.spec.datasetId,...value}};state.records.push(record);draft.status='frozen';return copy(record);};
const client=new QueryClient({defaultOptions:{queries:{retry:false,staleTime:Infinity}}});
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><main className="stage-workspace"><LocalTargetSplit workspace={{project:{id:'project',name:'Study',config:{seed:42}},dataset:{id:'dataset-a'}}}/></main></QueryClientProvider>);
`);
await build({configFile:false,root:web,logLevel:'error',plugins:[react()],define:{'process.env.NODE_ENV':JSON.stringify('production')},resolve:{alias:{'react/jsx-runtime':join(web,'node_modules/react/jsx-runtime.js')}},build:{outDir:join(output,'dist'),emptyOutDir:true,minify:false,lib:{entry:fixture,name:'TargetSplitFixture',formats:['iife'],fileName:()=> 'fixture.js'}}});
const dist=join(output,'dist');
const css=(await readdir(dist)).filter(name=>name.endsWith('.css'));
await writeFile(join(dist,'index.html'),'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'+css.map(name=>'<link rel="stylesheet" href="'+name+'">').join('')+'<style>body{padding:24px}#app{max-width:1320px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');
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
    try { if (await evaluate('Boolean(' + expression + ')')) return; }
    catch (error) { if (!/navigated|execution context|Cannot find context/i.test(String(error))) throw error; }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('Timed out: ' + description + '\n' + await evaluate('document.body.innerText'));
}

const button = text => '[...document.querySelectorAll("button")].find(el=>el.checkVisibility()&&el.textContent.trim()==='+JSON.stringify(text)+')';
const field = text => '[...document.querySelectorAll("label")].find(el=>el.checkVisibility()&&el.textContent.trim().startsWith('+JSON.stringify(text)+'))?.querySelector("input,select,textarea")';
async function click(text){await waitFor(button(text)+'&&!'+button(text)+'.matches(":disabled")');await evaluate(button(text)+'.click()');}
async function setInput(selector,value){await waitFor(selector);await evaluate('(()=>{const el='+selector+';const type=el instanceof HTMLSelectElement?HTMLSelectElement.prototype:el instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(type,"value").set.call(el,'+JSON.stringify(value)+');el.dispatchEvent(new Event(el instanceof HTMLSelectElement?"change":"input",{bubbles:true}));})()');}
async function fill(text,value){await setInput(field(text),value);}
const condition = title => '[...document.querySelectorAll(".condition-editor")].find(el=>el.querySelector("h3").textContent==='+JSON.stringify(title)+')';
const scopedField=(scope,text)=>'[...('+scope+').querySelectorAll("label")].find(el=>el.textContent.trim().startsWith('+JSON.stringify(text)+'))?.querySelector("input,select,textarea")';
async function addRule(title,fieldName,value){const scope=condition(title);await evaluate('[...('+scope+').querySelectorAll("button")].find(el=>el.textContent.trim()==="Add condition").click()');await setInput(scopedField(scope,'Field'),fieldName);await setInput(scopedField(scope,'Condition'),'eq');await setInput(scopedField(scope,'Value'),value);}
const targetPanel=title=>'[...document.querySelectorAll("section.card")].find(el=>el.querySelector("h2")?.textContent==='+JSON.stringify(title)+')';
const eligibilityProgress='document.querySelector('+JSON.stringify('progress[aria-label="Slides included by eligibility rules"]')+')';
const liveSelection=role=>'document.querySelector('+JSON.stringify('[aria-label="'+role+' live selection"]')+')';
const distribution=role=>'document.querySelector('+JSON.stringify('[aria-label="'+role+' target distributions"]')+')';
async function screenshot(name){await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');const {data}=await cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});await writeFile(join(artifacts,name+'.png'),Buffer.from(data,'base64'));}
try {
 const target=await cdp('Target.createTarget',{url:'about:blank'},null);
 ({sessionId}=await cdp('Target.attachToTarget',{targetId:target.targetId,flatten:true},null));
 await cdp('Page.enable');await cdp('Runtime.enable');
 await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
 await cdp('Page.navigate',{url:pathToFileURL(join(dist,'index.html')).href});
 await waitFor('document.body?.innerText.includes("No targets and splits yet")');
 await click('Create targets & splits');
 await fill('Draft name','Fixed cohort');
 await fill('Dataset','dataset-a');
 await cdp('Page.reload');
 await waitFor('document.body?.innerText.includes("Return to current draft")');
 await click('Return to current draft');
 assert.equal(await evaluate(field('Draft name')+'.value'),'Fixed cohort');
 assert.deepEqual(await evaluate('[...document.querySelectorAll(".stage-step-copy strong")].map(el=>el.textContent)'),['Dataset & cohort','Training & Testing split','Prediction Targets','Review & Freeze']);
 await waitFor(eligibilityProgress+'?.value===20');
 assert.equal(await evaluate('Boolean('+liveSelection('Training')+'||'+liveSelection('Testing')+')'),false,'Dataset & cohort must not expose train/test details');
 assert.equal(await evaluate('window.workflow.calls.some(call=>call.method==="partitionPreview")'),false,'Dataset & cohort must not depend on split settings');
 assert.equal(await evaluate('window.workflow.calls.filter(call=>call.method==="exploreProtocol").every(call=>call.request.cohortOnly===true)'),true,'Dataset & cohort must defer patient assignment validation');
 await addRule('Cohort conditions','site','no-matches');
 await waitFor(eligibilityProgress+'?.value===0');
 await setInput(scopedField(condition('Cohort conditions'),'Value'),'A');
 await waitFor(eligibilityProgress+'?.value===12');
 await screenshot('cohort-filtered');
 await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),true,'Cohort filtering overflows mobile viewport');
 await screenshot('cohort-filtered-mobile');
 await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
 await evaluate('document.querySelector('+JSON.stringify('button[aria-label="Remove Cohort conditions condition 1"]')+').click()');
 await waitFor(eligibilityProgress+'?.value===20');
 await click('Continue to training & testing');
 assert.equal(await evaluate(field('Split method')+'.value'),'random');
 assert.equal(await evaluate('document.querySelectorAll("[name=folds]").length'),0);
 assert.equal(await evaluate('Boolean('+field('Target attribute')+')'),false);
 await fill('Testing percentage','20');await fill('Split seed','123');
 await waitFor(liveSelection('Training')+'?.textContent.includes("Training slides16")');
 await evaluate('window.workflow.nextPartitionDelay=1200');
 await fill('Testing percentage','40');
 await waitFor('window.workflow.calls.some(call=>call.method==="partitionPreview"&&call.request.split.testFraction===0.4)');
 await fill('Testing percentage','20');
 await waitFor(liveSelection('Training')+'?.textContent.includes("16")&&'+liveSelection('Testing')+'?.textContent.includes("4")');
 await evaluate('new Promise(resolve=>setTimeout(resolve,1400))');
 assert.equal(await evaluate(liveSelection('Training')+'.querySelector("progress").value'),16,'A stale preview must not overwrite the latest filter counts');
 assert.equal(await evaluate(liveSelection('Testing')+'.querySelector("progress").value'),4);
 await screenshot('train-test-random');
 await fill('Split method','rules');
 assert.deepEqual(await evaluate('[...document.querySelectorAll(".condition-editor h3")].map(el=>el.textContent)'),['Training conditions','Testing conditions']);
 await addRule('Testing conditions','Requested_Split','test');
 await waitFor(liveSelection('Testing')+'?.textContent.includes("Testing slides8")');
 assert.equal(await evaluate(liveSelection('Training')+'.textContent.includes("Training slides12")'),true);
 await addRule('Training conditions','Requested_Split','train');
 await waitFor(liveSelection('Training')+'?.textContent.includes("12")&&'+liveSelection('Testing')+'?.textContent.includes("8")');
 assert.equal(await evaluate(liveSelection('Training')+'.querySelector("progress").value'),12);
 assert.equal(await evaluate(liveSelection('Testing')+'.querySelector("progress").value'),8);
 assert.equal(await evaluate(condition('Training conditions')+'.parentElement.contains('+liveSelection('Training')+')'),true,'Training counts must be next to training filters');
 assert.equal(await evaluate(condition('Testing conditions')+'.parentElement.contains('+liveSelection('Testing')+')'),true,'Testing counts must be next to testing filters');
 const unitOption=unit=>'document.querySelector('+JSON.stringify('.target-split-unit-control input[value="'+unit+'"]')+')';
 assert.equal(await evaluate(unitOption('slide')+'.checked'),true,'New targets & splits default to slide-level splitting');
 assert.equal(await evaluate(liveSelection('Training')+'.textContent.toLowerCase().includes("patient")'),false,'Slide-level selections must not report patients');
 await evaluate(unitOption('patient')+'.click()');
 await waitFor(liveSelection('Training')+'?.textContent.includes("Training verified patients")');
 await evaluate(unitOption('slide')+'.click()');
 await waitFor(liveSelection('Training')+'&&!'+liveSelection('Training')+'.textContent.toLowerCase().includes("patient")');
 await screenshot('train-test-filtered');
 await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),true,'Training/testing filters overflow mobile viewport');
 await screenshot('train-test-filtered-mobile');
 await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
 const selectionRequest=await evaluate('window.workflow.calls.filter(call=>call.method==="partitionPreview").at(-1).request');
 assert.equal(selectionRequest.target,undefined,'Partitions must be explored before defining targets');
 await click('Continue to prediction targets');
 await fill('Target attribute','grade');await waitFor(field('Task')+'.value==="binary_classification"');await fill('Positive class','high');
 await waitFor(distribution('Training')+'?.textContent.includes("training slides by mapped label")');
 assert.equal(await evaluate(distribution('Training')+'.textContent.includes("12 selected slides")'),true);
 assert.equal(await evaluate(distribution('Testing')+'.textContent.includes("8 selected slides")'),true);
 for(const role of ['Training','Testing']){assert.equal(await evaluate(distribution(role)+'.querySelectorAll("figure").length'),1);assert.equal(await evaluate(distribution(role)+'.textContent.toLowerCase().includes("patient")'),false,'Slide-level distributions must not count patients');}
 await screenshot('prediction-distributions');
 await fill('Testing target','separate');
 const testingPanel=targetPanel('Testing prediction target');
 await setInput(scopedField(testingPanel,'Target attribute'),'test_grade');
 await waitFor(testingPanel+'?.querySelectorAll(".science-label-row").length===2');
 for(const [raw,value] of [['L','low'],['H','high']]){const selector='[...('+testingPanel+').querySelectorAll(".science-label-row")].find(el=>el.querySelector("input").value==='+JSON.stringify(raw)+').querySelector("select")';await setInput(selector,value);}
 await waitFor(distribution('Testing')+'?.textContent.includes("test_grade · testing slides by mapped label")');
 assert.equal(await evaluate(scopedField(testingPanel,'Class names')+'.disabled'),true);
 await screenshot('testing-separate-target');
 await fill('Testing target','none');
 await waitFor('!'+distribution('Testing')+'&&'+distribution('Training'));
 await screenshot('testing-inference-only');
 await click('Review assignments');await waitFor('document.body.innerText.includes("Review targets and splits")');
 const saved=await evaluate('window.workflow.calls.filter(call=>call.method==="saveDraft").at(-1).input');
 assert.equal(saved.payload.type,'target-split');assert.equal(saved.payload.spec.split.seed,123);assert.equal(saved.payload.spec.testTarget,null);
 for(const key of ['featureBundleId','featureSetId','featurePackId'])assert.equal(key in saved.payload.spec,false);
 for(const key of ['folds','validationFraction','seeds'])assert.equal(key in saved.payload.spec.split,false);
 await screenshot('review');
 await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),true,'Review overflows mobile viewport');
 await screenshot('review-mobile');
 await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
 await click('Name & freeze version');await fill('Version tag','Fixed test cohort');await fill('Commit note','Keep testing separate');
 await click('Freeze target and split version');await waitFor('document.body.innerText.includes("Try another version tag.")');
 assert.equal(await evaluate(field('Version tag')+'.value'),'Fixed test cohort');
 await click('Freeze target and split version');await waitFor('document.body.innerText.includes("Continue to Experiments")');
 const freezes=await evaluate('window.workflow.calls.filter(call=>call.method==="freeze")');
 assert.equal(freezes.length,2);assert.equal(freezes[0].operationId,freezes[1].operationId);
 assert.equal(await evaluate('document.querySelector("a[href^=\\"#experiments?\\"]").getAttribute("href")'),'#experiments?dataset=dataset-a&targetSplit=target-split-fixed');
 const frozen=await evaluate('window.workflow.records[0]');assert.equal(frozen.manifest.memberships.filter(row=>row.partition==="test").length,8);assert.equal(frozen.manifest.memberships.filter(row=>row.partition==="test").every(row=>row.label===null),true);
 await screenshot('frozen');
 await click('Back to targets & splits');await waitFor('[...document.querySelectorAll("table")].some(el=>el.getAttribute("aria-label")==="Targets and splits library")');
 assert.equal(await evaluate('[...document.querySelectorAll("thead th")].map(el=>el.textContent).join(",")'),'Name,Status,Dataset,Training,Testing,Target,Labels,Updated,Actions');
 await click('Open');await click('Copy into a new draft');
 await click('Continue to training & testing');
 await fill('Split method','imported');await fill('Partition column','partition');
 await waitFor('[...document.querySelectorAll("select")].some(el=>el.getAttribute("aria-label")==="Assign development partition")');
 await evaluate('(()=>{for(const [label,value] of [["development","train"],["heldout","test"]]){const el=[...document.querySelectorAll("select")].find(el=>el.getAttribute("aria-label") === "Assign "+label+" partition");Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,"value").set.call(el,value);el.dispatchEvent(new Event("change",{bubbles:true}));}})()');
 await click('Continue to prediction targets');await click('Review assignments');
 const imported=await evaluate('window.workflow.calls.filter(call=>call.method==="saveDraft").at(-1).input.payload.spec.split');
 assert.equal(imported.method,'imported');assert.deepEqual(imported.trainValues,['development']);assert.deepEqual(imported.testValues,['heldout']);
 await click('Back');await click('Back');await fill('Split method','rules');await click('Continue to prediction targets');await click('Review assignments');
 const rules=await evaluate('window.workflow.calls.filter(call=>call.method==="saveDraft").at(-1).input.payload.spec.split');assert.equal(rules.method,'rules');assert.equal(rules.partitionField,undefined);assert.deepEqual(rules.trainValues,[]);assert.deepEqual(rules.testValues,[]);
 assert.deepEqual(await evaluate('window.workflow.errors'),[]);assert.deepEqual(exceptions,[]);
 await writeFile(join(artifacts,'verification.json'),JSON.stringify({passed:true,scope:'file:// React/Chromium fixture; no HistoPilot server',checks:['draft recovery','partition before prediction targets','cohort-only filtering visualization','training-first filter layout','exact train/test role labels','live adjacent filter counts','slide split unit by default with a patient switch','slide-only distributions','separate testing mapping','unlabeled inference testing','random split controls','metadata rules','predefined partition mapping','no features or training configuration','mobile review','freeze retries preserve intent','fixed membership summary','exact Experimental Setup handoff'],calls:await evaluate('window.workflow.calls')},null,2));
 console.log('PASS: partition-first target construction, live filter counts, slide/patient split unit, optional testing targets, draft recovery, immutable freeze and experiment handoff.');console.log('Artifacts: '+artifacts);
} catch(error){try{await writeFile(join(artifacts,'failure.txt'),await evaluate('document.body.innerText'));await screenshot('failure');}catch{}if(exceptions.length)console.error(JSON.stringify(exceptions));console.error('Artifacts: '+artifacts);throw error;}finally{browser.kill();}
