/** Exercise the case-review workspace in local Chromium, offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const output = resolve(process.argv[2] ?? '/tmp/histopilot-case-review-browser');
const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter(name => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
const requests = new Map();
const exceptions = [];
const checks = [];
let nextId = 0, buffer = '', stderr = '', sessionId;
browser.stderr.on('data', data => { stderr += data.toString(); });
const rejectPending = error => { for (const request of requests.values()) request.reject(error); requests.clear(); };
browser.on('error', rejectPending);
browser.on('exit', code => rejectPending(new Error(`Chromium exited (${code}): ${stderr}`)));
browser.stdio[3].on('error', rejectPending);
browser.stdio[4].on('error', rejectPending);
browser.stdio[4].on('data', data => {
  buffer += data.toString();
  let end;
  while ((end = buffer.indexOf('\0')) >= 0) {
    const message = JSON.parse(buffer.slice(0, end));
    buffer = buffer.slice(end + 1);
    if (message.id) {
      const request = requests.get(message.id);
      requests.delete(message.id);
      if (message.error) request?.reject(new Error(JSON.stringify(message.error)));
      else request?.resolve(message.result);
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
  for (let attempt = 0; attempt < 80; attempt++) {
    if (await evaluate('Boolean(' + expression + ')')) {
      checks.push(name);
      console.log('PASS', name);
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(name + ': ' + await evaluate('document.body.innerText'));
}
async function click(name) {
  await evaluate(`(() => { const button = [...document.querySelectorAll('button')].find(el => el.textContent.trim() === ${JSON.stringify(name)}); if (!button || button.disabled) throw Error('Button unavailable'); button.click(); })()`);
}
const seed = `document.querySelector('input[placeholder="Choose later"]')`;
async function fillSeed(value) {
  await evaluate(`(() => {const field = ${seed}; Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(field, ${JSON.stringify(value)}); field.dispatchEvent(new Event('input', {bubbles:true}));})()`);
}
async function screenshot(name) {
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable');
  await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(output, 'index.html')).href });
  await check('Case workspace loads verified predictions', "document.body?.innerText.includes('2 matching patient cases')");
  await check('Attention-capable patch model retains interpretation action', "[...document.querySelectorAll('button')].some(x => x.textContent === 'Inspect model attention') && document.querySelector('a[href^=\"#interpretation?\"]')");
  const selectLabel = async (label, value) => evaluate(`(() => {const field = [...document.querySelectorAll('label')].find(x=>x.textContent.startsWith(${JSON.stringify(label)}))?.querySelector('select'); if(!field) throw Error('Select not found'); field.value=${JSON.stringify(value)}; field.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const notes = `document.querySelector('textarea')`;
  const fillNotes = async value => evaluate(`(() => {const field=${notes}; Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(field,${JSON.stringify(value)}); field.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  await selectLabel('Cases', 'false_negative');
  await check('False-negative filter requests exact outcome', "document.body.innerText.includes('1 matching patient cases') && window.__morphologyCalls.some(x=>x.body?.outcome==='false_negative')");
  await selectLabel('Compare with', 'comparison');
  await check('Comparison is available', "document.body.innerText.toLowerCase().includes('models disagree')");
  await selectLabel('Cases', 'disagreement');
  await check('Disagreement query and paired probabilities', "window.__morphologyCalls.some(x=>x.body?.outcome==='disagreement' && x.body?.comparisonId==='comparison') && document.querySelectorAll('.case-predictions>div').length===2");
  await selectLabel('Patient slide', 'b');
  await check('Patient slide selection loads exact image and own review', "document.querySelector('[aria-label=\"Visual quality of b\"]') && document.querySelector('[aria-label=\"Review b\"]')");
  await fillNotes('My first saved morphology review');
  await selectLabel('Decision', 'review');
  await click('Save review');
  await check('Review is saved at expected revision', "document.body.innerText.includes('Review saved · revision 1') && window.__morphologyCalls.some(x=>x.body?.expectedRevision===0 && x.body?.notes==='My first saved morphology review')");
  await fillNotes('My unsaved correction survives conflict');
  await evaluate('window.__caseFixture.conflictNext=true');
  await click('Save review');
  await check('409 preserves unsaved reviewer notes', "document.body.innerText.includes('Review changed') && document.querySelector('textarea').value==='My unsaved correction survives conflict'");
  await click('Inspect latest saved review');
  await check('Latest review can be inspected without overwriting edits', "document.body.innerText.includes('Other reviewer note') && document.body.innerText.includes('Saved revision 2') && document.querySelector('textarea').value==='My unsaved correction survives conflict'");
  await click('Keep my edits against this revision');
  await click('Save review');
  await check('Explicit conflict recovery advances correct revision', "document.body.innerText.includes('Review saved · revision 3') && window.__morphologyCalls.some(x=>x.body?.expectedRevision===2 && x.body?.notes==='My unsaved correction survives conflict')");
  await selectLabel('Cases', 'all');
  await check('All cases restored', "document.body.innerText.includes('2 matching patient cases')");
  await click('patient-two');
  await check('Missing images preserve predictions and notes', "document.body.innerText.includes('no linked image for this slide') && document.querySelector('[aria-label=\"Review missing\"]') && document.querySelector('.case-predictions')");
  await check('Case table fits its pane and status labels have separate lines', `(() => {const pane=document.querySelector('.case-review-list'); const table=pane.querySelector('table'); return table.getBoundingClientRect().width <= pane.getBoundingClientRect().width + 1 && [...pane.querySelectorAll('tbody th')].every(cell => {const parts=[cell.querySelector('button'),...cell.querySelectorAll('small')].filter(Boolean); return parts.every((part,index)=>index===0 || part.getBoundingClientRect().top >= parts[index-1].getBoundingClientRect().bottom - 1);});})()`);
  await screenshot('case-review-desktop');
  await cdp('Emulation.setDeviceMetricsOverride', { width:390, height:844, deviceScaleFactor:1, mobile:true });
  await check('Case review fits phone viewport', 'document.documentElement.scrollWidth <= innerWidth');
  await screenshot('case-review-mobile');
  await cdp('Emulation.setDeviceMetricsOverride', { width:1200, height:1000, deviceScaleFactor:1, mobile:false });
  await cdp('Page.navigate', { url: pathToFileURL(join(output, 'index.html')).href + '?mode=slide' });
  await check('Slide-probe cases retain original images and predictions', "document.body?.innerText.includes('2 matching patient cases') && document.querySelector('.case-predictions') && document.querySelector('svg image')");
  await check('Slide-probe cases hide unsupported attention actions', "document.body.innerText.includes('Patch attention is unavailable') && ![...document.querySelectorAll('button')].some(x => x.textContent === 'Inspect model attention') && !document.querySelector('a[href^=\"#interpretation?\"]') && !window.__morphologyCalls.some(x => x.path.endsWith('/interpretations'))");
  await selectLabel('Compare with', 'comparison');
  await check('Comparing slide probes does not introduce attention controls', "document.querySelectorAll('.case-predictions>div').length === 2 && !document.querySelector('a[href^=\"#interpretation?\"]') && !document.body.innerText.includes('Inspect model attention')");
  await screenshot('case-review-slide-model');
  assert.deepEqual(exceptions, [], 'No unexpected browser errors');
  await writeFile(join(output, 'checks.json'), JSON.stringify({ passed: true, checks, exceptions }, null, 2) + '\n');
  console.log('Artifacts: ' + output);
} finally {
  browser.kill();
}
