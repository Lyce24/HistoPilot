/** Real production App → read-only ASGI, over pipes. No HTTP listener or server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const { values } = parseArgs({ options: {
  state: { type: 'string', default: join(repository, '.local/blca-e2e-20260917/state.json') },
  project: { type: 'string' }, output: { type: 'string' }, workspace: { type: 'string', default: join(homedir(), '.histopilot/workspace') },
  python: { type: 'string', default: join(repository, '.venv/bin/python') },
  'training-python': { type: 'string', default: process.env.HISTOPILOT_TRAINING_PYTHON ?? join(repository, '.venv-training/bin/python') },
  'data-root': { type: 'string', multiple: true }, 'static-dir': { type: 'string', default: join(repository, 'histopilot/static') },
  smoke: { type: 'boolean', default: false },
  phase: { type: 'string', default: 'all' },
} });
assert.ok(['all', 'prepare', 'live'].includes(values.phase), 'Phase must be all, prepare, or live');
let state = {};
try { state = JSON.parse(await readFile(values.state, 'utf8')); } catch (error) { if (!values.project) throw error; }
const project = values.project ?? state.projectId ?? state.project?.id;
assert.match(project ?? '', /^project-[a-f0-9]{32}$/, 'Provide --project or state.projectId');
const output = resolve(values.output ?? join(dirname(values.state), 'browser'));
await mkdir(output, { recursive: true });
const base = 'http://127.0.0.1:8787';
const projectBase = '/api/v1/projects/' + encodeURIComponent(project);
const checks = [], traffic = [], exceptions = [], snapshots = [], skipped = [];
const logs = createWriteStream(join(output, 'bridge.log'), { flags: 'w' });
const bridge = spawn(values.python, [join(repository, 'scripts/blca_browser_bridge.py'), '--project', project, '--workspace', values.workspace, '--static-dir', values['static-dir'], ...([...new Set(values['data-root'] ?? ['/mnt/d/YC.Liu', '/mnt/wsl/oceanpath-hot', values.workspace])].flatMap(path => ['--data-root', path]))], { cwd: repository, env: { ...process.env, HISTOPILOT_TRAINING_PYTHON: values['training-python'] }, stdio: ['pipe', 'pipe', 'pipe'] });
bridge.stderr.pipe(logs);
let bridgeBuffer = '', bridgeNumber = 0, bridgeReady;
const bridgeRequests = new Map();
let resolveReady, rejectReady;
const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
function rejectBridge(error) { rejectReady(error); for (const { reject, timer } of bridgeRequests.values()) { clearTimeout(timer); reject(error); } bridgeRequests.clear(); }
bridge.on('error', rejectBridge);
bridge.on('exit', (code) => { if (code !== 0) rejectBridge(new Error(`ASGI bridge exited (${code}); see bridge.log`)); });
bridge.stdout.on('data', chunk => {
  bridgeBuffer += chunk;
  let end;
  while ((end = bridgeBuffer.indexOf('\n')) >= 0) {
    const line = bridgeBuffer.slice(0, end); bridgeBuffer = bridgeBuffer.slice(end + 1);
    let message;
    try { message = JSON.parse(line); } catch { rejectBridge(new Error('Invalid ASGI bridge response; see bridge.log')); continue; }
    if (message.ready) { bridgeReady = message; resolveReady(message); continue; }
    const request = bridgeRequests.get(message.id); if (!request) continue;
    clearTimeout(request.timer); bridgeRequests.delete(message.id);
    if (message.error) request.reject(new Error(message.error)); else request.resolve(message);
  }
});
function requestASGI(method, url, headers = {}, body) {
  const id = ++bridgeNumber;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { bridgeRequests.delete(id); reject(new Error(`ASGI request timeout: ${method} ${new URL(url).pathname}`)); }, 120000);
    bridgeRequests.set(id, { resolve, reject, timer });
    bridge.stdin.write(JSON.stringify({ id, method, url, headers, body }) + '\n');
  });
}
let browser, sessionId, token, failWorkspace = false, navigationNumber = 0;
let pendingNetwork = 0, lastNetworkChange = 0;
const requests = new Map();
let nextId = 0, buffer = '', browserStderr = '';
function cdp(method, params = {}, session = sessionId) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { requests.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 120000);
    requests.set(id, { resolve, reject, timer });
    browser.stdio[3].write(JSON.stringify({ id, method, params, ...(session ? { sessionId: session } : {}) }) + '\0');
  });
}
async function intercept(event) {
  const { requestId, request } = event;
  const pathname = new URL(request.url).pathname;
  pendingNetwork += 1;
  lastNetworkChange = Date.now();
  try {
    const injected = failWorkspace && pathname === projectBase + '/workspace';
    const result = injected ? { status: 503, headers: [{ name: 'Content-Type', value: 'application/json' }], body: Buffer.from(JSON.stringify({ detail: 'Verification-injected temporary workspace outage', code: 'VERIFICATION_OUTAGE' })).toString('base64') }
      : await requestASGI(request.method, request.url, request.headers, request.postData);
    traffic.push({ method: request.method, path: new URL(request.url).pathname + new URL(request.url).search, status: result.status, blocked: result.blocked === true, injected });
    await cdp('Fetch.fulfillRequest', { requestId, responseCode: result.status, responseHeaders: result.headers, body: result.body });
  } catch (error) {
    // Navigation can cancel an expensive tile/API read before ASGI finishes it.
    // That is normal browser cancellation, not an application failure.
    if (/Invalid [Ii]nterceptionId|Invalid [Ii]nterception|No resource with given identifier|Target closed|Session with given id not found/.test(String(error))) return;
    exceptions.push({ bridge: String(error), path: pathname });
    await cdp('Fetch.failRequest', { requestId, errorReason: 'Failed' }).catch(() => {});
  } finally {
    pendingNetwork -= 1;
    lastNetworkChange = Date.now();
  }
}
async function evaluate(expression) {
  const response = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function check(name, expression, timeout = 60000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await evaluate(`Boolean(${expression})`)) { checks.push(name); console.log('PASS', name); return; }
    await new Promise(resolve => setTimeout(resolve, 125));
  }
  throw new Error(name + ': ' + (await evaluate('document.body?.innerText')).slice(0, 16000));
}
async function clickText(text) {
  await evaluate(`(() => { const element = [...document.querySelectorAll('button,a')].find(x => x.getClientRects().length && x.textContent.trim() === ${JSON.stringify(text)}); if (!element || element.disabled) throw Error('Unavailable control: ' + ${JSON.stringify(text)}); element.click(); })()`);
}
async function clickSelector(selector) {
  await evaluate(`(() => { const element = document.querySelector(${JSON.stringify(selector)}); if (!element || element.disabled) throw Error('Unavailable control'); element.click(); })()`);
}
async function selectCaseFilter(label, value) {
  await evaluate(`(() => { const label = [...document.querySelectorAll('.case-review-controls label')].find(element => element.firstChild?.textContent.trim() === ${JSON.stringify(label)}); const select = label?.querySelector('select'); if (!select) throw Error('Missing case filter'); select.value = ${JSON.stringify(value)}; select.dispatchEvent(new Event('change', {bubbles:true})); })()`);
}
async function captureElement(name, selector) {
  const clip = await evaluate(`(() => { const box = document.querySelector(${JSON.stringify(selector)})?.getBoundingClientRect(); if (!box?.width || !box.height) throw Error('Missing screenshot region'); return {x:box.x+scrollX,y:box.y+scrollY,width:box.width,height:box.height,scale:1}; })()`);
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
async function snapshot(name) {
  const deadline = Date.now() + 30000;
  while ((pendingNetwork || Date.now() - lastNetworkChange < 250) && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
  await evaluate(`Promise.all(document.getAnimations().filter(a => a.effect?.getComputedTiming().iterations !== Infinity).map(a => a.finished.catch(() => {})))`);
  // Responsive reflow can preserve a nonzero scroll offset. Normalize capture
  // position so fixed navigation is not drawn halfway down a full-page image.
  await evaluate(`new Promise(resolve => { window.scrollTo({top:0,left:0,behavior:'instant'}); requestAnimationFrame(() => requestAnimationFrame(resolve)); })`);
  const dom = await evaluate(`({url:location.href,title:document.title,layout:{viewportWidth:innerWidth,documentWidth:document.documentElement.scrollWidth},headings:[...document.querySelectorAll('h1,h2,h3')].filter(e=>e.getClientRects().length).map(e=>e.textContent),controls:[...document.querySelectorAll('button,a,input,select,summary')].filter(e=>e.getClientRects().length).map(e=>({tag:e.tagName,label:e.getAttribute('aria-label')||e.textContent?.trim()||e.getAttribute('placeholder'),disabled:e.disabled||false})),text:document.querySelector('#main-content')?.innerText??document.body.innerText})`);
  snapshots.push({ name, ...dom });
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
  await writeFile(join(output, name + '.json'), JSON.stringify(dom, null, 2) + '\n');
}
async function navigate(hash) {
  const marker = `before-navigation-${++navigationNumber}`;
  await evaluate(`window.__verificationDocument = ${JSON.stringify(marker)}`);
  // A unique query parameter forces a new document; hash-only Page.navigate can
  // otherwise observe the preceding page while React is still switching chunks.
  await cdp('Page.navigate', { url: base + '/?project=' + encodeURIComponent(project) + '&verification=' + navigationNumber + '#' + hash });
  await check(`${hash.split('?')[0]}: real project shell loads`, `window.__verificationDocument !== ${JSON.stringify(marker)} && document.readyState === 'complete' && document.querySelector('#main-content h1') && !document.querySelector('#main-content [aria-busy=true]') && document.querySelector('.breadcrumb-project')?.textContent === ${JSON.stringify(state.projectName)}`);
}
async function reloadPage() {
  const marker = `before-reload-${++navigationNumber}`;
  await evaluate(`window.__verificationDocument = ${JSON.stringify(marker)}`);
  await cdp('Page.reload', { ignoreCache: true });
  await check('Reload reaches a new document', `window.__verificationDocument !== ${JSON.stringify(marker)} && document.readyState === 'complete' && document.body.innerText.length > 0`);
}
async function readAPI(path) {
  const response = await requestASGI('GET', base + path, token ? { 'X-HistoPilot-Token': token } : {});
  assert.equal(response.status, 200, `Read API ${path}`);
  return JSON.parse(Buffer.from(response.body, 'base64').toString());
}
function choose(items, id, name, required = true) {
  const selected = items.find(item => item.id === id) ?? (!id ? items[0] : undefined);
  if (!selected && !values.smoke && required) throw new Error(`Missing ${name}; complete the project before full browser verification`);
  if (!selected && required) skipped.push(name);
  return selected;
}
const versionName = (record, fallback) => record.versionLabel?.tag?.trim() || `${fallback} · ${record.id.slice(-8)}`;
try {
  await Promise.race([ready, new Promise((_, reject) => setTimeout(() => reject(new Error('ASGI bridge initialization timed out')), 30000).unref())]);
  token = (await readAPI('/api/v1/session')).token;
  const workspace = await readAPI(projectBase + '/workspace');
  state.projectName = workspace.project.name;
  assert.equal(workspace.project.id, project);
  const [datasets, protocols, bundles, experiments, cohorts, evaluations, interpretations] = await Promise.all([
    '/datasets', '/configurations?kind=protocol', '/feature-bundles', '/model-experiments?summary=true', '/evaluation-cohorts', '/evaluation-runs', '/interpretations',
  ].map(path => readAPI(projectBase + path)));
  const dataset = choose(datasets.datasets ?? [], state.datasetId, 'frozen dataset');
  const protocol = choose(protocols.configurations ?? [], state.protocolId, 'frozen protocol');
  const bundle = choose(bundles.items ?? [], state.featureBundleId ?? state.bundleId, 'feature bundle');
  const experiment = choose(experiments.items ?? [], state.experimentId, 'model experiment', values.phase !== 'prepare');
  const cohort = choose(cohorts.items ?? [], state.evaluationCohortId ?? state.cohortId, 'frozen test cohort');
  const completedEvaluations = (evaluations.items ?? []).filter(item => item.execution?.status === 'completed');
  const evaluation = choose(completedEvaluations, state.evaluationId ?? state.evaluationIds?.ensemble, 'completed primary evaluation', values.phase === 'all');
  const refitEvaluation = state.evaluationIds?.refit ? choose(completedEvaluations, state.evaluationIds.refit, 'completed refit evaluation', values.phase === 'all') : null;
  const interpretation = choose((interpretations.items ?? []).filter(item => item.execution?.status === 'completed'), state.interpretationId ?? state.attentionBatchId ?? Object.values(state.interpretationIds ?? {})[0], 'completed interpretation', values.phase === 'all');
  await writeFile(join(output, 'inputs.json'), JSON.stringify({ project, projectName: state.projectName, datasetId: dataset?.id, protocolId: protocol?.id, bundleId: bundle?.id, experimentId: experiment?.id, evaluationId: evaluation?.id, interpretationId: interpretation?.id, cohortIds: cohorts.items?.map(item => item.id), bridge: bridgeReady }, null, 2) + '\n');
  // Test the barrier directly; blocked requests never reach application handlers.
  for (const [method, path] of [['POST', projectBase + '/model-experiments/blocked/submit'], ['PUT', projectBase + '/datasets/blocked/slide-reviews/blocked'], ['GET', '/api/v1/projects/project-' + '0'.repeat(32) + '/workspace']]) {
    assert.equal((await requestASGI(method, base + path, { 'X-HistoPilot-Token': token }, '{}')).blocked, true);
  }
  checks.push('Bridge rejects mutations and access to other projects before ASGI dispatch');
  const cache = join(homedir(), '.cache/ms-playwright');
  const candidate = (await readdir(cache)).filter(name => name.startsWith('chromium_headless_shell-')).sort().at(-1);
  const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
  browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
  browser.stderr.on('data', chunk => { browserStderr += chunk; });
  const rejectCDP = error => { for (const request of requests.values()) { clearTimeout(request.timer); request.reject(error); } requests.clear(); };
  browser.on('error', rejectCDP); browser.on('exit', code => rejectCDP(new Error(`Chromium exited ${code}: ${browserStderr}`)));
  browser.stdio[3].on('error', rejectCDP); browser.stdio[4].on('error', rejectCDP);
  browser.stdio[4].on('data', chunk => {
    buffer += chunk;
    let end;
    while ((end = buffer.indexOf('\0')) >= 0) {
      const message = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1);
      if (message.id) {
        const request = requests.get(message.id); requests.delete(message.id); if (!request) continue;
        clearTimeout(request.timer); if (message.error) request.reject(new Error(JSON.stringify(message.error))); else request.resolve(message.result);
      } else if (message.method === 'Fetch.requestPaused') void intercept(message.params);
      else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
      else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') exceptions.push({ console: message.params.args });
    }
  });
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Network.enable'); await cdp('Network.setCacheDisabled', { cacheDisabled: true });
  await cdp('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
  await cdp('Fetch.enable', { patterns: [{ urlPattern: '*', requestStage: 'Request' }] });
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1050, deviceScaleFactor: 1, mobile: false });
  if (values.phase === 'all') {
  await navigate('overview');
  await check('Roadmap resolves real stored records', "document.querySelectorAll('[data-module]').length >= 7 && !document.body.innerText.includes('Loading project progress')");
  await snapshot('01-roadmap');
  }
  if (values.phase !== 'live') {
  await navigate('dataset');
  if (dataset) {
    await check('Dataset library contains frozen version', "document.querySelector('button[aria-label^=\"Open dataset \"]')");
    await clickText(versionName(dataset, dataset.manifest.name?.trim() || 'Dataset'));
    await check('Dataset detail exposes frozen metadata', "document.body.innerText.includes('Frozen dataset') && document.body.innerText.includes('Explore your dataset')");
    await check('Dataset detail retains the requested frozen identity', `document.querySelector('[title=${JSON.stringify(dataset.id)}]')`);
  }
  await snapshot('02-dataset');
  await navigate('cohort');
  if (protocol) {
    await check('Protocol library contains saved protocol', "document.querySelector('button[aria-label^=\"Open protocol \"]')");
    await clickText(versionName(protocol, protocol.manifest.spec.target.field ? `${protocol.manifest.spec.target.field} protocol` : 'Development protocol'));
    await check('Frozen target and split inputs render', "document.querySelector('[aria-label=\"Frozen protocol inputs\"]')");
    const target = protocol.manifest.spec.target;
    await check('Frozen protocol makes reference label, prediction unit and class mapping readable', `(() => { const section = document.querySelector('[aria-label="Frozen prediction target and identity groups"]'); return section && ${JSON.stringify([target.field, `One prediction per ${target.unit}`, ...Object.entries(target.labels).map(([raw, mapped]) => `${raw} → ${mapped}`), ...(target.positiveClass ? [target.positiveClass] : [])])}.every(text => section.textContent.includes(text)); })()`);
    if (protocol.manifest.summary.fallbackSlideCount) await check('Protocol separates supplied patient IDs from acknowledged slide groups', "document.querySelector('[aria-label=\"Frozen prediction target and identity groups\"]').textContent.includes('Patient independence cannot be verified') && document.body.innerText.includes('Supplied patient IDs') && !document.body.innerText.includes('patient-grouped development splits')");
  }
  await snapshot('03-target-splits');
  if (values.phase === 'prepare') {
    await check('Frozen protocol fits desktop viewport', 'document.documentElement.scrollWidth <= innerWidth + 1');
    await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await check('Frozen protocol fits phone viewport', 'document.documentElement.scrollWidth <= innerWidth + 1');
    await snapshot('03-target-splits-mobile');
    await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1050, deviceScaleFactor: 1, mobile: false });
  }
  await navigate('features');
  if (bundle) {
    await check('Bundle library renders verified input', "document.querySelector('table[aria-label=\"Saved feature bundles\"] button.stage-record-name')");
    await clickText(versionName(bundle, 'Feature bundle'));
    await check('Feature bundle detail opens', "document.body.innerText.includes('Verified inputs') && !document.querySelector('table[aria-label=\"Saved feature bundles\"]')");
  }
  await snapshot('04-features');
  }
  if (values.phase !== 'prepare') {
  await navigate(experiment ? 'experiments?' + new URLSearchParams({ experiment: experiment.id, tab: values.phase === 'live' ? 'runs' : 'results' }) : 'experiments');
  if (experiment) await check('Saved experiment results render', `document.querySelector('.experiment-detail-id')?.textContent.includes(${JSON.stringify(experiment.id)})`);
  if (experiment && values.phase === 'all') await check('Experiment results retain frozen scoring-unit meaning', "document.body.innerText.includes('frozen scoring unit') && !document.body.innerText.includes('slide-level metrics are secondary')");
  await snapshot('05-experiment-results');
  if (values.phase === 'live') {
    await reloadPage();
    await check('Reload reattaches to the same submitted experiment', `document.querySelector('.experiment-detail-id')?.textContent.includes(${JSON.stringify(experiment.id)}) && !document.querySelector('#main-content [aria-busy=true]')`);
    await snapshot('05-experiment-reattached');
  }
  }
  if (values.phase !== 'live') {
  await navigate('test-data');
  await check('Test-cohort library renders', "document.querySelector('#main-content h1')?.textContent.includes('Test cohorts')");
  if (cohort) await check('Requested frozen test cohort is listed', `document.querySelector('#main-content').textContent.includes(${JSON.stringify(versionName(cohort, 'Test cohort'))})`);
  if (cohort) {
    await clickText(versionName(cohort, 'Test cohort'));
    await check('Frozen test cohort details open', "document.querySelector('#main-content').textContent.includes('Frozen test cohort')");
    if (cohort.manifest.memberships?.some(row => row.patientIdSource === 'slide_fallback')) {
      const supplied = new Set(cohort.manifest.memberships.filter(row => row.patientId && ['source', 'crosswalk'].includes(row.patientIdSource)).map(row => row.patientId)).size;
      const fallback = new Set(cohort.manifest.memberships.filter(row => row.patientId && row.patientIdSource === 'slide_fallback').map(row => row.patientId)).size;
      await check('Frozen test cohort distinguishes actual supplied IDs and slide/case groups', `document.querySelector('#main-content').textContent.includes(${JSON.stringify(`${supplied.toLocaleString()} supplied patient IDs · ${fallback.toLocaleString()} acknowledged slide / case groups`)})`);
    }
  }
  await snapshot('06-test-cohort');
  if (values.phase === 'prepare') {
    await check('Frozen test cohort fits desktop viewport', 'document.documentElement.scrollWidth <= innerWidth + 1');
    await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await check('Frozen test cohort fits phone viewport', 'document.documentElement.scrollWidth <= innerWidth + 1');
    await snapshot('06-test-cohort-mobile');
  }
  }
  if (values.phase === 'all') {
  if (refitEvaluation && refitEvaluation.id !== evaluation?.id) {
    await navigate('evaluation?' + new URLSearchParams({ evaluation: refitEvaluation.id }));
    await check('Secondary refit metrics render from the exact completed evaluation', "document.body.innerText.includes('AUROC') && document.body.innerText.includes('Download metrics')");
    await snapshot('07-refit-metrics');
  }
  await navigate(evaluation ? 'evaluation?' + new URLSearchParams({ evaluation: evaluation.id }) : 'evaluation');
  if (evaluation) {
    await check('Real completed metrics render', "document.body.innerText.includes('AUROC') && document.body.innerText.includes('Download metrics')");
    await snapshot('07-ensemble-metrics');
    await clickText('Review cases and model errors');
    await check('Case evidence loads from saved predictions', "document.querySelector('[aria-label=\"Case and error review\"]') && document.querySelector('.case-review-list tbody tr')");
    await selectCaseFilter('Cases', 'false_negative');
    await check('False-negative filter resolves real cases or an explicit empty result', `(() => { const section = document.querySelector('[aria-label="Case and error review"]'); const rows = [...section.querySelectorAll('.case-review-list tbody tr')]; return !section.textContent.includes('Verifying saved predictions') && (rows.length ? rows.every(row => row.textContent.includes('False negatives')) : section.textContent.includes('No matching cases')); })()`);
    await selectCaseFilter('Cases', 'all');
    await check('All real cases restore after filtering', "document.querySelector('.case-review-list tbody tr')");
    if (refitEvaluation && refitEvaluation.id !== evaluation.id) {
      await selectCaseFilter('Compare with', refitEvaluation.id);
      await check('The same case displays saved ensemble and refit probabilities', "document.querySelectorAll('.case-predictions > div').length === 2");
    }
    await check('Case review loads the exact original slide image', "document.querySelector('.case-review-detail .morphology-slide svg image')?.getAttribute('href')?.startsWith('blob:')", 120000);
    await check('Slide-level case review uses a neutral grouping label', "document.querySelector('.case-review-detail').textContent.includes('Case/group ID') && !/Actual label:[^\\n]* · Patient /.test(document.querySelector('.case-review-detail').innerText)");
  }
  await snapshot('07-evaluation-cases');
  if (state.clinicalAnalysisIds?.ensemble) {
    await navigate('clinical-utility?' + new URLSearchParams({ clinical: state.clinicalAnalysisIds.ensemble }));
    await check('Saved exploratory calibration report renders with slide unit and validity limits', "document.body.innerText.includes('Download report JSON') && document.body.innerText.includes('labeled slide records') && document.body.innerText.includes('do not establish clinical validity') && document.body.innerText.includes('Calibration')");
    await snapshot('07-exploratory-calibration');
  }
  let attentionHash = 'interpretation';
  if (interpretation) attentionHash += '?' + new URLSearchParams({ stage: 'viewer', interpretation: interpretation.id, slide: state.slideId ?? Object.keys(state.interpretationIds ?? {})[0] ?? interpretation.manifest.slides[0].slideId });
  await navigate(attentionHash);
  if (interpretation) {
    await check('Actual attention and original slide render', "document.querySelector('.attention-workspace svg image') && document.querySelector('.attention-workspace svg image').getAttribute('href')?.startsWith('blob:')");
    await check('Ranked attention locations are available', "document.querySelector('[aria-label=\"Patch inspection and ranked locations\"]') && document.querySelector('button[aria-label=\"View rank 1 patch\"]')");
    await clickSelector('button[aria-label="View rank 1 patch"]');
    await check('Ranked patch opens an original-color crop', "document.querySelector('[aria-label=\"Selected patch crop\"] img')?.complete && document.querySelector('[aria-label=\"Selected patch crop\"] img')?.naturalWidth > 0");
    const beforeZoom = await evaluate("document.querySelector('.attention-workspace svg[role=group]').getAttribute('viewBox')");
    await clickSelector('button[aria-label="Zoom in"]');
    await check('Attention viewer zoom responds', `document.querySelector('.attention-workspace svg[role=group]').getAttribute('viewBox') !== ${JSON.stringify(beforeZoom)}`);
  }
  await snapshot('08-attention');
  if (interpretation) await captureElement('08-attention-original-patch', '[aria-label="Selected patch crop"]');
  const canonicalAttentionHash = await evaluate('location.hash');
  await reloadPage();
  await check('Reload restores project and exact stage', `document.querySelector('#main-content h1') && !document.querySelector('#main-content [aria-busy=true]') && location.hash === ${JSON.stringify(canonicalAttentionHash)}`);
  if (interpretation) await check('Reload restores exact completed attention', "document.querySelector('.attention-workspace svg image')?.getAttribute('href')?.startsWith('blob:')");
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await check('Attention stage fits phone viewport', 'document.documentElement.scrollWidth <= innerWidth + 1');
  await snapshot('09-attention-mobile');
  await navigate('overview');
  await check('Mobile roadmap resolves saved records', "document.querySelectorAll('[data-module]').length >= 7 && !document.body.innerText.includes('Loading workflow')");
  await check('Roadmap fits phone viewport', 'document.documentElement.scrollWidth <= innerWidth + 1');
  await clickSelector('button[aria-label="Toggle navigation"]');
  await check('Mobile navigation opens', "document.querySelector('button[aria-label=\"Toggle navigation\"]').getAttribute('aria-expanded') === 'true'");
  await cdp('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await check('Escape closes mobile navigation and restores focus', "document.querySelector('button[aria-label=\"Toggle navigation\"]').getAttribute('aria-expanded') === 'false' && document.activeElement?.getAttribute('aria-label') === 'Toggle navigation'");
  await snapshot('10-roadmap-mobile');
  failWorkspace = true;
  await reloadPage();
  await check('Unavailable workspace offers reconnect', "document.body.innerText.includes('Could not open the project') && [...document.querySelectorAll('button')].some(x=>x.textContent==='Reconnect')");
  await snapshot('11-reconnect');
  failWorkspace = false;
  await clickText('Reconnect');
  await check('Reconnect restores the same real project', `document.querySelector('.breadcrumb-project')?.textContent === ${JSON.stringify(state.projectName)}`);
  await check('Recovered roadmap has no horizontal overflow', 'document.documentElement.scrollWidth <= innerWidth + 1');
  }
  assert.equal(traffic.filter(item => item.blocked).length, 0, 'UI must not issue forbidden requests during read-only review');
  assert.deepEqual(exceptions, [], 'No unexpected runtime or console errors');
  assert.equal(traffic.filter(item => item.status >= 400 && !item.injected).length, 0, 'All real browser requests must succeed');
  checks.push('No unexpected browser errors, forbidden UI mutations, or failed real API responses');
  await writeFile(join(output, 'checks.json'), JSON.stringify({ passed: true, project, checks, skipped, traffic, snapshots: snapshots.map(item => ({ name: item.name, url: item.url, headings: item.headings })), bridge: bridgeReady }, null, 2) + '\n');
  console.log(`${checks.length} checks passed; private artifacts: ${output}`);
} catch (error) {
  if (browser && sessionId) await snapshot('failure').catch(() => {});
  await writeFile(join(output, 'checks.json'), JSON.stringify({ passed: false, project, error: String(error), checks, skipped, traffic, exceptions, bridge: bridgeReady }, null, 2) + '\n');
  throw error;
} finally {
  browser?.kill();
  bridge.stdin.end();
  bridge.kill();
  logs.end();
}
