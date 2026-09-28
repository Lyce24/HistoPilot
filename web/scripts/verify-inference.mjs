/** Exercise Run inference in local Chromium, offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const output = resolve(process.argv[2] ?? '/tmp/histopilot-current-review/inference-browser');
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
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false });
  const page = pathToFileURL(join(output, 'index.html')).href;
  await cdp('Page.navigate', { url: page + '?view=library' });
  await check('Library lists inference runs with predicted mix and agreement', "document.body?.innerText.includes('BD ensemble · fold 5×') && document.body.innerText.toLowerCase().includes('predicted classes') && document.body.innerText.toLowerCase().includes('unanimous')");
  await check('Library carries no evaluation metrics', "!/AUROC|AUPRC|Accuracy/.test(document.body.innerText)");
  await screenshot('inference-library');
  await cdp('Page.navigate', { url: page });
  await check('Run detail explains predictions-only scope', "document.body?.innerText.includes('Predictions only: no labels are read') && document.body.innerText.includes('146 patients in this cohort also contributed development slides')");
  await check('Summary tiles and charts render', "document.body.innerText.includes('Slides predicted') && document.body.innerText.includes('Fold members disagree') && document.querySelectorAll('.inference-histogram svg path').length > 10");
  await check('Detail carries no evaluation metrics', "!/AUROC|AUPRC|Balanced accuracy|Confusion matrix/.test(document.body.innerText)");
  await evaluate(`(() => { const mark = document.querySelector('.inference-histogram svg path'); mark.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, relatedTarget: document.body })); })()`);
  await check('Histogram segments carry a hover readout', "document.querySelector('.inference-tooltip strong')");
  await evaluate(`document.querySelectorAll('.inference-histogram svg path')[3].focus()`);
  await check('Keyboard focus shows the same readout', "document.activeElement?.classList.contains('is-active') && document.querySelector('.inference-tooltip strong')");
  await screenshot('inference-detail');
  const selectLabel = async (label, value) => evaluate(`(() => {const field = [...document.querySelectorAll('label')].find(x=>x.textContent.startsWith(${JSON.stringify(label)}))?.querySelector('select'); if(!field) throw Error('Select not found: ' + ${JSON.stringify(label)}); const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; setter.call(field, ${JSON.stringify(value)}); field.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await selectLabel('Break down by', 'consensus');
  await check('Attribute breakdown requests and renders', "window.__inferenceCalls.some(x => x.body?.attribute === 'consensus') && document.body.innerText.includes('Predicted class by Reader consensus')");
  await evaluate('window.__summaryDelay = 250; window.__summaryFailure = true');
  await selectLabel('Break down by', 'part');
  await check('Changing summary scope never displays stale charts', "!document.querySelector('.inference-tiles') && [...document.querySelectorAll('button')].find(x => x.textContent.includes('Download predictions with metadata'))?.disabled");
  await check('Failed verification hides summary and offers retry', "document.body.innerText.includes('Saved predictions failed verification') && !document.querySelector('.inference-tiles') && document.body.innerText.includes('Retry prediction summary')");
  await evaluate('window.__summaryDelay = 0; window.__summaryFailure = false');
  await click('Retry prediction summary');
  await check('Retry recovers the chosen analysis scope', "document.querySelector('.inference-tiles') && document.body.innerText.includes('Predicted class by Part')");
  await selectLabel('Compare with', 'refit');
  await check('Label-free run comparison renders agreement and kappa', "document.body.innerText.includes('versus BD refit') && document.body.innerText.includes('Agreement is not accuracy')");
  await check('Review queue defaults to the decision boundary', "window.__inferenceCalls.some(x => x.path.endsWith('/cases/query') && x.body?.sort === 'margin_asc') && document.body.innerText.includes('Closest to a decision boundary')");
  await evaluate(`[...document.querySelectorAll('button')].find(x => x.textContent.trim() === 'review' && x.closest('.inference-tiles div')?.textContent.includes('Margin below')).click()`);
  await check('Margin tile reviews exactly the selected borderline predictions', "window.__inferenceCalls.some(x => x.path.endsWith('/cases/query') && x.body?.maxMargin === 0.2 && x.body?.sort === 'margin_asc')");
  await selectLabel('Minimum predicted probability', '0.9');
  await selectLabel('Maximum predicted probability', '0.6');
  await check('Crossing confidence bounds remain valid', "window.__inferenceCalls.some(x => x.path.endsWith('/cases/query') && x.body?.minConfidence === 0.6 && x.body?.maxConfidence === 0.6) && !window.__inferenceCalls.some(x => x.path.endsWith('/cases/query') && x.body?.minConfidence > (x.body?.maxConfidence ?? 1))");
  await click('Reset review filters');
  await evaluate(`[...document.querySelectorAll('button')].find(x => x.textContent.trim() === 'review' && x.closest('.inference-tiles div')?.textContent.includes('Fold members disagree')).click()`);
  await check('Disagreement tile opens a filtered review queue', "window.__inferenceCalls.some(x => x.path.endsWith('/cases/query') && x.body?.memberDisagreement === true && x.body?.sort === 'agreement_asc')");
  await click('Inspect model attention');
  await check('Missing attention offers one-step computation', "[...document.querySelectorAll('button')].some(x => x.textContent === 'Compute attention for this slide')");
  await click('Compute attention for this slide');
  await check('Attention request goes to the run and shows queued state', "window.__inferenceCalls.some(x => x.path.endsWith('/attention') && x.body?.slideIds?.length === 1) && document.body.innerText.includes('queued')");
  await screenshot('inference-review');
  await check('No unexpected API requests', "window.__unexpected.length === 0");
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await check('Run detail fits a phone viewport', 'document.documentElement.scrollWidth <= innerWidth');
  await screenshot('inference-mobile');
  assert.deepEqual(exceptions, [], 'No unexpected browser errors');
  await writeFile(join(output, 'checks.json'), JSON.stringify({ passed: true, checks, exceptions }, null, 2) + '\n');
  console.log('Artifacts: ' + output);
} finally {
  browser.kill();
}
