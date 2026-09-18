/** Exercise the operations workspace in local Chromium, offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const output = resolve(process.argv[2] ?? '/tmp/histopilot-operations-browser');
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
  await check('Operations page mounts with inventory pending', "document.body?.innerText.includes('Loading jobs and reservations')");
  await check('Operations form fields have visible solid borders', "(() => {const fields=[...document.querySelectorAll('.operations-field input, .operations-field select')]; return fields.length > 0 && fields.every(field => {const css=getComputedStyle(field); return parseFloat(css.borderTopWidth)>0 && css.borderTopStyle==='solid' && css.borderTopColor!=='rgba(0, 0, 0, 0)';});})()");
  const fillArchive = async () => evaluate(`(() => {const input=document.querySelector('input[placeholder="/storage/backups/study.zip"]'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,'/offline/backups/study.zip'); input.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  const exportButton = `[...document.querySelectorAll('button')].find(x=>x.textContent==='Export & verify project')`;
  await fillArchive();
  await check('Export disabled until job inventory verified', `${exportButton}.disabled`);
  await evaluate('window.__operationsState.releaseInventory()');
  await check('Running job displayed with cancel control', "document.body.innerText.includes('Synthetic training batch') && [...document.querySelectorAll('button')].some(x=>x.textContent==='Cancel job')");
  await check('Export disabled while pipeline job active', `${exportButton}.disabled`);
  await click('Cancel job');
  await check('First cancellation acknowledged', "document.body.innerText.includes('0 active jobs') && window.__operationsCalls.filter(x=>x.path.endsWith('/cleanup/cancel')).length===1");
  await evaluate('window.__operationsState.pipelineRunning=true');
  await click('Refresh');
  await check('Resumed job can be cancelled', "document.body.innerText.includes('1 active jobs')");
  await click('Cancel job');
  await check('Resumed job cancellation gets new receipt', "(()=>{const values=window.__operationsCalls.filter(x=>x.path.endsWith('/cleanup/cancel'));return values.length===2 && values[0].body.operationId!==values[1].body.operationId && document.body.innerText.includes('0 active jobs');})()");
  await evaluate('window.__operationsState.pipelineRunning=true;window.__operationsState.loseCancel=true');
  await click('Refresh');
  await check('Running job restored before uncertain response', "document.body.innerText.includes('1 active jobs')");
  await click('Cancel job');
  await check('Lost cancellation acknowledgement shown', "document.body.innerText.includes('Could not reach HistoPilot')");
  await click('Cancel job');
  await check('Uncertain cancellation retry reuses receipt', "(()=>{const values=window.__operationsCalls.filter(x=>x.path.endsWith('/cleanup/cancel'));return values.length===4 && values[2].body.operationId===values[3].body.operationId && document.body.innerText.includes('0 active jobs');})()");
  await evaluate('window.__operationsState.loseArchive=true');
  await click('Export & verify project');
  await check('Lost archive response still discovers durable job', "document.body.innerText.includes('Could not reach HistoPilot') && document.querySelectorAll('.operations-receipt').length===1");
  await click('Export & verify project');
  await check('Same-page archive retry preserves request ID', "(()=>{const values=window.__operationsCalls.filter(x=>x.path.endsWith('/operations/archives') && x.body);return values.length===2 && values[0].body.operationId===values[1].body.operationId && window.__operationsState.acceptedArchives===1;})()");
  await click('Reconnect view');
  await check('Reconnect rediscovers active archive', "document.querySelectorAll('.operations-receipt').length===1 && document.body.innerText.includes('Cancel archive operation')");
  await fillArchive();
  await click('Export & verify project');
  await check('Reconnected duplicate request retains one archive', "window.__operationsState.acceptedArchives===1 && document.querySelectorAll('.operations-receipt').length===1");
  await click('Cancel archive operation');
  await check('Cancelled archive exposes retry', "document.body.innerText.includes('Retry saved operation')");
  await click('Retry saved operation');
  await check('Retry restarts saved job without duplicate submission', "document.body.innerText.includes('Cancel archive operation') && window.__operationsCalls.some(x=>x.path.endsWith('/archives/archive-1/retry')) && window.__operationsState.acceptedArchives===1");
  await screenshot('operations-desktop');
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await check('Operations page fits phone viewport','document.documentElement.scrollWidth<=innerWidth');
  await screenshot('operations-mobile');
  assert.deepEqual(exceptions, [], 'No unexpected browser errors');
  await writeFile(join(output, 'checks.json'), JSON.stringify({ passed: true, checks }, null, 2) + '\n');
  console.log('Artifacts: ' + output);
} finally {
  browser.kill();
}
