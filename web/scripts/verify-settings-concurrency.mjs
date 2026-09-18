/** Exercise the settings editor in local Chromium, offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const output = resolve(process.argv[2] ?? '/tmp/histopilot-current-review/settings-browser');
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
  await check('Saved seed loads', `${seed}?.value === '7'`);
  await fillSeed('13');
  await click('Simulate change in another tab');
  await check('Background refresh preserves unsaved entry', `${seed}.value === '13'`);
  await click('Save initial settings');
  await check('Stale baseline is rejected without losing edits', `${seed}.value === '13' && window.__settingsRequests.length === 1 && window.__settingsRequests[0].expectedConfig.seed === 7 && window.__settingsRequests[0].status === 409 && document.body.innerText.includes('Your entries are still in this form')`);
  await screenshot('settings-conflict');
  await evaluate('window.__settingsTransport.delayed = true');
  await click('Reload saved settings');
  await check('Reload locks editor while response is pending', `${seed}.matches(':disabled') && typeof window.__settingsTransport.release === 'function'`);
  await evaluate('window.__settingsTransport.delayed = false; window.__settingsTransport.release()');
  await check('Explicit reload restores current server values', `${seed}.value === '11' && document.body.innerText.includes('Saved settings loaded')`);
  await fillSeed('17');
  await evaluate('window.__settingsTransport.delayed = true; delete window.__settingsTransport.release');
  await click('Save initial settings');
  await check('Save locks editor while response is pending', `${seed}.matches(':disabled') && typeof window.__settingsTransport.release === 'function'`);
  await evaluate('window.__settingsTransport.delayed = false; window.__settingsTransport.release()');
  await check('Recovered editor saves against reloaded baseline', "window.__settingsRequests.length === 2 && window.__settingsRequests[1].expectedConfig.seed === 11 && window.__settingsRequests[1].status === 200 && document.body.innerText.includes('Initial settings saved')");
  await fillSeed('23');
  await click('Save initial settings');
  await check('Successful save advances baseline for subsequent edits', 'window.__settingsRequests.length === 3 && window.__settingsRequests[2].expectedConfig.seed === 17 && window.__settingsRequests[2].status === 200');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await check('Settings fit a mobile viewport', 'document.documentElement.scrollWidth <= innerWidth');
  await screenshot('settings-mobile');
  assert.deepEqual(exceptions, [], 'No unexpected browser errors');
  await writeFile(join(output, 'checks.json'), JSON.stringify({ passed: true, checks }, null, 2) + '\n');
  console.log('Artifacts: ' + output);
} finally {
  browser.kill();
}
