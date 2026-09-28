/** Exercise the morphology explorer in local Chromium, offline; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const output = resolve(process.argv[2] ?? process.env.HISTOPILOT_VERIFICATION_ARTIFACTS ?? '/tmp/histopilot-morphology-browser');
const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter(name => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
const requests = new Map();
const exceptions = [];
const checks = [];
const roiEvidence = [];
const gestureEvidence = [];
const tileEvidence = [];
const overlayEvidence = [];
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
async function screenshot(name) {
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
const slideSvg = 'svg[aria-label^="Exact slide "]';
const frame = 'new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))';
async function viewport() {
  return evaluate(`document.querySelector(${JSON.stringify(slideSvg)}).getAttribute('viewBox').split(' ').map(Number)`);
}
async function imageLayers() {
  return evaluate(`Array.from(document.querySelectorAll(${JSON.stringify(slideSvg + ' image')}), image => ({href:image.getAttribute('href'),detail:image.hasAttribute('data-slide-detail'),tile:image.getAttribute('data-slide-tile'),level:Number(image.getAttribute('data-tile-level')),source:image.getAttribute('data-tile-source'),...Object.fromEntries(['x','y','width','height'].map(key => [key, Number(image.getAttribute(key))]))}))`);
}
async function wheelPoint() {
  return evaluate(`(() => {
    const svg = document.querySelector(${JSON.stringify(slideSvg)}); svg.scrollIntoView({block:'center'});
    const field = svg.viewBox.baseVal, matrix = svg.getScreenCTM();
    const level0 = {x:field.x + field.width * .37,y:field.y + field.height * .43};
    const screen = new DOMPoint(level0.x, level0.y).matrixTransform(matrix);
    const x=Math.round(screen.x), y=Math.round(screen.y);
    const rounded = new DOMPoint(x,y).matrixTransform(matrix.inverse());
    return {x,y,level0:{x:rounded.x,y:rounded.y}};
  })()`);
}
async function assertSmoothWheel() {
  const anchor=await wheelPoint();
  const animation=await evaluate(`(async()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)}),samples=[],start=performance.now();svg.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:-120,clientX:${anchor.x},clientY:${anchor.y}}));while(performance.now()-start<240){await new Promise(requestAnimationFrame);samples.push({time:performance.now()-start,width:svg.viewBox.baseVal.width});}const point=new DOMPoint(${anchor.x},${anchor.y}).matrixTransform(svg.getScreenCTM().inverse());return{samples,point:{x:point.x,y:point.y}};})()`);
  assert.ok(new Set(animation.samples.map(value=>Math.round(value.width*1000))).size>=3,'A coarse wheel notch should animate through multiple visible intermediate camera positions');
  assert.ok(Math.abs(animation.samples.at(-1).width-6000*Math.exp(-120*.004))<.01,'Easing must finish at the exact requested zoom');
  assert.ok(animation.samples.every((value,index)=>!index||value.width<=animation.samples[index-1].width+.01),'Zoom animation must not reverse or overshoot');
  assert.ok(Math.abs(animation.point.x-anchor.level0.x)<1&&Math.abs(animation.point.y-anchor.level0.y)<1,'Eased zoom must keep the exact slide point under the cursor');
  checks.push('motion: coarse wheel animates through multiple frames and preserves the cursor anchor');
  await evaluate(`(async()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)});svg.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:-120,clientX:${anchor.x},clientY:${anchor.y}}));await new Promise(requestAnimationFrame);Array.from(document.querySelectorAll('button')).find(button=>button.textContent.trim()==='Fit slide').click();await new Promise(resolve=>setTimeout(resolve,220));})()`);
  assert.deepEqual(await viewport(),[0,0,6000,4000]);checks.push('motion: Fit slide cancels an unfinished zoom animation');
  await cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  const reduced=await evaluate(`(async()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)});svg.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:-120,clientX:${anchor.x},clientY:${anchor.y}}));await ${frame};const first=svg.viewBox.baseVal.width;await new Promise(resolve=>setTimeout(resolve,160));return{first,last:svg.viewBox.baseVal.width};})()`);
  assert.ok(Math.abs(reduced.first-6000*Math.exp(-120*.004))<.01);assert.equal(reduced.first,reduced.last);checks.push('motion: reduced-motion preference reaches the target without easing');
  await cdp('Emulation.setEmulatedMedia',{features:[]});await click('Fit slide');
  gestureEvidence.push({mode:'coarse-wheel-easing',samples:animation.samples,anchorPreserved:true,reducedMotion:reduced});
}
async function assertWheelPanHandoff() {
  if ((await viewport())[2] !== 6000) await click('Fit slide');await evaluate('new Promise(resolve=>setTimeout(resolve,200))');
  const anchor=await wheelPoint();
  await cdp('Input.dispatchMouseEvent',{type:'mouseWheel',x:anchor.x,y:anchor.y,deltaX:0,deltaY:-120,modifiers:0});
  await evaluate(frame);
  const whileAnimating=await viewport();
  assert.ok(whileAnimating[2]<6000 && whileAnimating[2]>6000*Math.exp(-.48),'Begin the drag while coarse wheel easing is unfinished');
  await cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:anchor.x,y:anchor.y,button:'left',clickCount:1});
  const pressed=await viewport();await evaluate('new Promise(resolve=>setTimeout(resolve,70))');
  assert.deepEqual(await viewport(),pressed,'Pointer-down must stop remaining wheel frames before panning captures its view');
  await cdp('Input.dispatchMouseEvent',{type:'mouseMoved',x:anchor.x+24,y:anchor.y+12,button:'left',buttons:1});
  await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:anchor.x+24,y:anchor.y+12,button:'left',clickCount:1});
  await evaluate(frame);const panned=await viewport();
  assert.equal(panned[2],pressed[2]);assert.equal(panned[3],pressed[3]);assert.ok(panned[0]<pressed[0]);
  await evaluate('new Promise(resolve=>setTimeout(resolve,180))');assert.deepEqual(await viewport(),panned);
  checks.push('motion: trusted wheel-to-drag handoff preserves zoom scale and cancels the old animation');
  gestureEvidence.push({mode:'wheel-to-drag',whileAnimating,pressed,panned,scaleJumpPercent:0});
  await click('Fit slide');
}
async function assertOrdinaryWheelFlow(label) {
  const anchor = await wheelPoint();
  const before = await viewport(), scrollBefore = await evaluate('scrollY');
  await cdp('Input.dispatchMouseEvent', {type:'mouseWheel',x:anchor.x,y:anchor.y,deltaX:0,deltaY:-120,modifiers:0});
  await check(label('ordinary wheel zooms into the slide without Ctrl'), `document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width < ${before[2]} && window.__wheelEvidence.at(-1)?.trusted && !window.__wheelEvidence.at(-1)?.ctrl && window.__wheelEvidence.at(-1)?.prevented`);
  await evaluate('new Promise(resolve=>setTimeout(resolve,180))');
  const zoomed = await viewport();
  const afterAnchor = await evaluate(`(() => {const value=new DOMPoint(${anchor.x},${anchor.y}).matrixTransform(document.querySelector(${JSON.stringify(slideSvg)}).getScreenCTM().inverse());return {x:value.x,y:value.y};})()`);
  assert.ok(Math.abs(anchor.level0.x-afterAnchor.x)<1 && Math.abs(anchor.level0.y-afterAnchor.y)<1, label('ordinary wheel keeps the cursor over the same slide location'));
  assert.equal(await evaluate('scrollY'),scrollBefore,label('inside wheel must not scroll the page'));
  checks.push(label('ordinary wheel anchors at cursor and keeps page still'));
  await cdp('Input.dispatchMouseEvent', {type:'mouseWheel',x:anchor.x,y:anchor.y,deltaX:0,deltaY:60,modifiers:0});
  await check(label('ordinary wheel zooms back out without Ctrl'),`document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width > ${zoomed[2]}`);
  await evaluate('new Promise(resolve=>setTimeout(resolve,180))');
  assert.equal(await evaluate('scrollY'),scrollBefore,label('inside zoom-out must not scroll the page'));
  const control = await evaluate(`(() => {const box=document.querySelector('.morphology-zoom-controls span').getBoundingClientRect();return {x:Math.round(box.x+box.width/2),y:Math.round(box.y+box.height/2)};})()`);
  const controlBefore = await viewport();
  await cdp('Input.dispatchMouseEvent', {type:'mouseWheel',...control,deltaX:0,deltaY:-40,modifiers:0});
  await check(label('ordinary wheel over floating zoom controls still zooms'),`document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width < ${controlBefore[2]}`);
  assert.equal(await evaluate('scrollY'),scrollBefore,label('floating controls must not leak wheel scrolling to the page'));
  await click('Fit slide');
  await check(label('fit resets after ordinary-wheel navigation'),`document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width === 6000`);
  const fitAnchor=await wheelPoint(), fitScroll=await evaluate('scrollY'), wheelCount=await evaluate('window.__wheelEvidence.length');
  await cdp('Input.dispatchMouseEvent', {type:'mouseWheel',x:fitAnchor.x,y:fitAnchor.y,deltaX:0,deltaY:120,modifiers:0});
  await check(label('full-fit zoom limit still consumes inside wheel'),`window.__wheelEvidence.length > ${wheelCount} && window.__wheelEvidence.at(-1)?.prevented`);
  await evaluate(frame);
  assert.deepEqual(await viewport(),[0,0,6000,4000]);
  assert.equal(await evaluate('scrollY'),fitScroll,label('full-fit limit must not cause page-scroll bleed'));
  const letterbox = await evaluate(`(() => {const svg=document.querySelector(${JSON.stringify(slideSvg)}), box=svg.getBoundingClientRect(), matrix=svg.getScreenCTM(); const origin=new DOMPoint(0,0).matrixTransform(matrix); if(origin.x-box.x>3)return{x:Math.round((box.x+origin.x)/2),y:Math.round(box.y+box.height/2)}; if(origin.y-box.y>3)return{x:Math.round(box.x+box.width/2),y:Math.round((box.y+origin.y)/2)};return null;})()`);
  if(letterbox) {
    await cdp('Input.dispatchMouseEvent',{type:'mouseWheel',...letterbox,deltaX:0,deltaY:-60,modifiers:0});
    await check(label('letterbox area inside slide window also zooms'),`document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width < 6000`);
    assert.equal(await evaluate('scrollY'),fitScroll,label('letterbox wheel must remain inside the slide window'));
    await click('Fit slide');
  }
  await evaluate(`document.querySelector(${JSON.stringify(slideSvg)}).focus({preventScroll:true})`);
  const outsideBefore=await viewport(), outsideScroll=await evaluate('scrollY');
  assert.equal(await evaluate("Boolean(document.elementFromPoint(8,500)?.closest('.morphology-image-scroll'))"),false,'outside wheel target must be outside the slide window');
  await cdp('Input.dispatchMouseEvent',{type:'mouseMoved',x:8,y:500,buttons:0});
  await cdp('Input.dispatchMouseEvent',{type:'mouseWheel',x:8,y:500,deltaX:0,deltaY:120,modifiers:0});
  await check(label('ordinary wheel outside scrolls the page down with canvas focused'),`scrollY > ${outsideScroll} && !window.__wheelEvidence.at(-1)?.prevented && document.activeElement === document.querySelector(${JSON.stringify(slideSvg)})`);
  await evaluate('new Promise(resolve=>setTimeout(resolve,150))');
  const scrolledDown=await evaluate('scrollY');
  assert.deepEqual(await viewport(),outsideBefore,label('outside page scrolling must not zoom the focused slide'));
  await cdp('Input.dispatchMouseEvent',{type:'mouseWheel',x:8,y:500,deltaX:0,deltaY:-120,modifiers:0});
  await check(label('ordinary wheel outside scrolls the page back up'),`scrollY < ${scrolledDown} && !window.__wheelEvidence.at(-1)?.prevented`);
  await evaluate('new Promise(resolve=>setTimeout(resolve,150))');
  assert.deepEqual(await viewport(),outsideBefore,label('outside scroll-up must leave slide coordinates unchanged'));
  checks.push(label('page scrolling outside leaves focused viewer unchanged in both directions'));
}
async function tileRequests(start = 0) {
  return evaluate(`window.__morphologyCalls.slice(${start}).filter(call=>call.path.includes('/image?') && new URL(call.path,'http://offline.invalid').searchParams.has('x')).map(call=>{const p=new URL(call.path,'http://offline.invalid').searchParams;return {...call,slide:p.get('slideId'),maxSize:Number(p.get('max_size')),region:Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(p.get(key))]))};})`);
}
const rectangleKey = value => ['x','y','width','height'].map(key=>value[key]).join(':');
async function expectedTileGrid() {
  return evaluate(`(() => {
    const svg=document.querySelector(${JSON.stringify(slideSvg)}), view=svg.viewBox.baseVal, box=svg.getBoundingClientRect();
    const {width,height}=window.__morphologyGeometry;let level=Math.max(0,Math.floor(Math.log2(Math.max(view.width/box.width,view.height/box.height)/Math.min(2,devicePixelRatio))));let span=512*2**level;while((Math.ceil((view.x+view.width)/span)-Math.floor(view.x/span))*(Math.ceil((view.y+view.height)/span)-Math.floor(view.y/span))>32){level++;span*=2;}const tiles=[];
    for(let row=Math.floor(view.y/span);row<Math.ceil((view.y+view.height)/span);row++) for(let col=Math.floor(view.x/span);col<Math.ceil((view.x+view.width)/span);col++) tiles.push({key:level+':'+col+':'+row,level,region:{x:col*span,y:row*span,width:Math.min(span,width-col*span),height:Math.min(span,height-row*span)}});
    return tiles;
  })()`);
}
async function waitForTileGrid(label, expected) {
  await check(label, `(() => {const keys=new Set(Array.from(document.querySelectorAll(${JSON.stringify(slideSvg+' image[data-slide-tile]')}),image=>image.getAttribute('data-slide-tile')));return ${JSON.stringify(expected.map(tile=>tile.key))}.every(key=>keys.has(key)) && !document.body.innerText.includes('Loading detail…');})()`);
}
async function panBy(dx, dy=0) {
  const anchor=await wheelPoint();
  await cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:anchor.x,y:anchor.y,button:'left',clickCount:1});
  await cdp('Input.dispatchMouseEvent',{type:'mouseMoved',x:anchor.x+dx,y:anchor.y+dy,button:'left',buttons:1});
  await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:anchor.x+dx,y:anchor.y+dy,button:'left',clickCount:1});
}
async function assertImageFingerprints(label) {
  const values=await evaluate("window.__morphologyCalls.filter(call=>call.path.includes('/morphology/image?')).map(call=>({path:call.path,expected:call.expectedFingerprint,rejected:call.rejectedFingerprint,actual:new URL(call.path,'http://offline.invalid').searchParams.get('sourceFingerprint')}))");
  assert.ok(values.some(call=>!call.path.includes('&x=')) && values.some(call=>call.path.includes('&x=')),'Both overview and regional reads must be exercised');
  for(const call of values.filter(call=>!call.rejected))assert.equal(call.actual,call.expected,'Every morphology image must carry the identity returned by its slide geometry');
  checks.push(label+': overview and tiles use the matching geometry fingerprint');
}
async function settleImages(label) {
  await evaluate('new Promise(resolve=>setTimeout(resolve,450))');
  await check(label, 'window.__morphologyImages.active===0');
}
function preparationCount(width,height) {
  const count=level=>Math.ceil(width/(512*2**level))*Math.ceil(height/(512*2**level));
  let level=0;while(count(level)+count(level+1)+count(level+2)>128)level++;
  return count(level)+count(level+1)+count(level+2);
}
async function assertTiledReading() {
  const label=name=>'tiles: '+name;
  await cdp('Emulation.setDeviceMetricsOverride',{width:1200,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(join(output,'index.html')).href+'?mode=slide&tiles=1&hold=1&fail=1'});
  await check(label('fresh explorer loads'),"document.body?.innerText.includes('Open visual review & feature explorer')");
  await click('Open visual review & feature explorer');
  const dimensions=await evaluate('window.__morphologyGeometry'), preparedCount=preparationCount(dimensions.width,dimensions.height), fullView=`0 0 ${dimensions.width} ${dimensions.height}`;
  await check(label('original overview is ready'),`document.querySelector(${JSON.stringify(slideSvg+' image:not([data-slide-detail])')})?.getAttribute('href')?.startsWith('blob:')`);
  await check(label('three-level preparation reports bounded progress'),`document.querySelector('[data-slide-tile-layer]')?.getAttribute('data-prepared-tiles')==='0' && document.querySelector('[data-slide-tile-layer]')?.getAttribute('data-preparation-total')==='${preparedCount}' && document.querySelector('[data-slide-tile-layer]')?.getAttribute('data-preparing')==='true' && window.__morphologyImages.active===2 && window.__morphologyImages.pending.length===2`);
  assert.equal((await tileRequests()).length,2,'Startup must initially occupy only two decoder slots');
  await evaluate('window.__morphologyImages.pending[0]()');
  await check(label('one temporary preparation failure preserves remaining work'),'document.body.innerText.includes("Sharper detail could not be loaded") && window.__morphologyImages.pending.length===2');
  await evaluate('window.__morphologyImages.pending[0]()');
  await check(label('successful preparation advances visible progress'),`document.querySelector('[data-slide-tile-layer]')?.getAttribute('data-prepared-tiles')==='1' && window.__morphologyImages.pending.length===2`);
  await evaluate('window.__releaseMorphologyImages()');
  await check(label('partial preparation ends with an actionable failure'),'!document.body.innerText.includes("Preparing slide ·") && document.body.innerText.includes("Some slide detail could not be prepared") && !document.body.innerText.includes("Slide ready") && window.__morphologyImages.active===0');
  const preparation=await tileRequests(), failed=preparation.find(call=>call.failed);
  assert.equal(preparation.length,preparedCount,'Prepare exactly the bounded three-level full-slide grid');assert.ok(failed);
  const retryStart=await evaluate('window.__morphologyCalls.length');await click('Retry slide detail');
  await check(label('retry completes preparation and reports ready'),'document.body.innerText.includes("Slide ready") && !document.body.innerText.includes("Sharper detail could not be loaded") && window.__morphologyImages.active===0');
  const retried=await tileRequests(retryStart);assert.equal(retried.length,1);assert.equal(retried[0].path,failed.path);
  const warmStart=await evaluate('window.__morphologyCalls.length');
  await evaluate("document.querySelector('button[aria-label=\"Zoom in\"]').click()");await evaluate(frame);
  const warmGrid=await expectedTileGrid();await waitForTileGrid(label('first zoom displays prepared detail'),warmGrid);
  assert.equal((await tileRequests(warmStart)).filter(call=>warmGrid.some(tile=>rectangleKey(tile.region)===rectangleKey(call.region))).length,0,'Prepared visible tissue must need no new reads');
  await click('Fit slide');await evaluate('new Promise(resolve=>setTimeout(resolve,180))');
  const anchor=await wheelPoint(), start=await evaluate('window.__morphologyCalls.length');
  await evaluate(`(async()=>{window.__morphologyImages.holdRegions=true;window.__tileGestureStart=performance.now();const svg=document.querySelector(${JSON.stringify(slideSvg)});for(let n=0;n<20;n++){svg.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:n<9?-70:-2,clientX:${anchor.x},clientY:${anchor.y}}));await new Promise(resolve=>setTimeout(resolve,50));}window.__tileGestureEnd=performance.now();await new Promise(resolve=>setTimeout(resolve,180));})()`);
  await check(label('continuous gesture starts two bounded held detail reads'),'window.__morphologyImages.pending.length===2 && window.__morphologyImages.active===2');
  const initial=await tileRequests(start), gestureEnd=await evaluate('window.__tileGestureEnd');
  assert.equal(initial.length,2);assert.ok(initial[0].startedAt<gestureEnd-500,'Start detail in the first half of the continuing gesture');assert.equal(await evaluate('window.__morphologyImages.maxActive'),2);
  checks.push(label('detail loading progresses during continuous zoom'));
  await evaluate(`(()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)});for(let n=0;n<12;n++){svg.dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,key:'ArrowRight'}));svg.dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,key:'ArrowDown'}));}})()`);
  await evaluate('new Promise(resolve=>setTimeout(resolve,120))');const latest=await expectedTileGrid();
  await evaluate('window.__morphologyImages.pending[0]()');
  await check(label('released slot serves the latest visible field'),`window.__morphologyCalls.slice(${start}).filter(call=>call.path.includes('/image?') && call.path.includes('&x=')).length===3 && window.__morphologyImages.pending.length===2`);
  const next=(await tileRequests(start))[2];assert.ok(latest.some(tile=>rectangleKey(tile.region)===rectangleKey(next.region)),'Discard obsolete queued viewport work');
  const keys=latest.map(tile=>tile.key);
  for(let n=0;n<40;n++) {
    if(await evaluate(`(()=>{const keys=new Set(Array.from(document.querySelectorAll(${JSON.stringify(slideSvg+' image[data-slide-tile]')}),image=>image.getAttribute('data-slide-tile')));return ${JSON.stringify(keys)}.every(key=>keys.has(key));})()`))break;
    await evaluate('window.__morphologyImages.pending[0]?.()');await evaluate('new Promise(resolve=>setTimeout(resolve,25))');
  }
  await waitForTileGrid(label('latest visible field fills progressively'),latest);
  await check(label('idle anticipation occupies only one decoder slot'),'window.__morphologyImages.active===1 && window.__morphologyImages.pending.length===1');
  const speculative=(await tileRequests()).filter(call=>!call.completed&&!call.aborted);assert.equal(speculative.length,1);
  const beforeForeground=await evaluate('window.__morphologyCalls.length');
  await evaluate(`document.querySelector(${JSON.stringify(slideSvg)}).dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,key:'ArrowLeft'}))`);
  await check(label('foreground navigation uses the reserved second decoder slot'),'window.__morphologyImages.active===2 && window.__morphologyImages.pending.length===2');
  assert.equal((await tileRequests(beforeForeground)).length,1,'One held prediction must leave room for one immediate foreground read');
  await evaluate('window.__releaseMorphologyImages()');await waitForTileGrid(label('new foreground detail resolves before idle work'),await expectedTileGrid());await settleImages(label('idle prediction finishes within a bounded work queue'));
  const beforePredictionZoom=await evaluate('window.__morphologyCalls.length');
  await evaluate("document.querySelector('button[aria-label=\"Zoom in\"]').click()");await evaluate(frame);
  const predictedZoom=await expectedTileGrid();await waitForTileGrid(label('anticipated next zoom displays cached tissue'),predictedZoom);
  assert.equal((await tileRequests(beforePredictionZoom)).filter(call=>predictedZoom.some(tile=>rectangleKey(tile.region)===rectangleKey(call.region))).length,0,'The centered next zoom must reuse its idle-prefetched tiles');
  await settleImages(label('next zoom prefetch remains finite'));
  const stableLayers=(await imageLayers()).filter(layer=>layer.detail), stableView=await viewport(), beforePan=await evaluate('window.__morphologyCalls.length');
  await panBy(100);await evaluate(frame);const shifted=await expectedTileGrid();await waitForTileGrid(label('anticipated nearby pan displays cached tissue'),shifted);
  const panReads=(await tileRequests(beforePan)).filter(call=>shifted.some(tile=>rectangleKey(tile.region)===rectangleKey(call.region)));
  assert.equal(panReads.length,0,'A nearby pan should use the prefetched surrounding grid');
  const reused=(await imageLayers()).filter(layer=>stableLayers.some(old=>old.tile===layer.tile));assert.ok(reused.length>0);
  for(const layer of reused)assert.deepEqual(layer,stableLayers.find(old=>old.tile===layer.tile),'Cached URLs and level-0 bounds remain unchanged');
  const revisitStart=await evaluate('window.__morphologyCalls.length');await panBy(-100);await evaluate(frame);const revisited=await viewport();
  for(let n=0;n<4;n++)assert.ok(Math.abs(revisited[n]-stableView[n])<.01);await waitForTileGrid(label('revisited tissue uses the same cached image tiles'),predictedZoom);
  assert.equal((await tileRequests(revisitStart)).filter(call=>predictedZoom.some(tile=>rectangleKey(tile.region)===rectangleKey(call.region))).length,0);
  await click('Fit slide');await evaluate('new Promise(resolve=>setTimeout(resolve,150))');await waitForTileGrid(label('zoom out returns to its cached coarse resolution'),await expectedTileGrid());
  assert.ok((await imageLayers()).filter(layer=>layer.detail).length<=32,'Zooming out must not mount every cached fine tile');checks.push(label('fitted rendering remains bounded after deeper detail and prediction'));
  await evaluate(`(()=>{window.__morphologyImages.holdRegions=true;for(let n=0;n<5;n++)document.querySelector('button[aria-label="Zoom in"]').click();})()`);
  await check(label('source change begins with two active detail reads'),'window.__morphologyImages.pending.length===2');
  const held=(await tileRequests()).filter(call=>!call.completed&&!call.aborted),oldLayers=(await imageLayers()).filter(layer=>layer.detail);
  await evaluate("Array.from(document.querySelectorAll('.morphology-slide-button')).find(button=>button.querySelector('strong')?.textContent==='b').click()");
  await check(label('changing slides aborts old reads and clears old tile images'),`document.querySelector('svg[aria-label="Exact slide b"]')?.getAttribute('viewBox')===${JSON.stringify(fullView)} && document.querySelectorAll(${JSON.stringify(slideSvg+' image[data-slide-tile]')}).length===0 && window.__morphologyImages.pending.length===2`);
  const afterSwitch=await tileRequests();for(const call of held)assert.equal(afterSwitch.find(item=>item.path===call.path).aborted,true);
  await evaluate('window.__releaseMorphologyImages()');await check(label('new source finishes its own preparation'),'document.body.innerText.includes("Slide ready") && window.__morphologyImages.active===0');
  const oldURLs=new Set(oldLayers.map(layer=>layer.href));assert.ok((await imageLayers()).filter(layer=>layer.detail).every(layer=>!oldURLs.has(layer.href)));
  const replacementStart=await evaluate('window.__morphologyCalls.length'),replacementOldLayers=(await imageLayers()).filter(layer=>layer.detail);
  await evaluate("(async()=>{window.__morphologyImages.holdRegions=true;window.__morphologyFingerprints.b='c'.repeat(64);await window.__refreshMorphologyQuality();})()");
  await check(label('changed source identity clears cached detail before reloading'),`document.querySelector('svg[aria-label="Exact slide b"]')?.getAttribute('viewBox')===${JSON.stringify(fullView)} && document.querySelectorAll(${JSON.stringify(slideSvg+' image[data-slide-tile]')}).length===0 && window.__morphologyImages.pending.length===2`);
  await evaluate('window.__releaseMorphologyImages()');await check(label('replacement source finishes its fresh preparation'),'document.body.innerText.includes("Slide ready") && window.__morphologyImages.active===0');
  const replacementReads=await evaluate(`window.__morphologyCalls.slice(${replacementStart}).filter(call=>call.path.includes('/morphology/image?')).map(call=>new URL(call.path,'http://offline.invalid').searchParams.get('sourceFingerprint'))`);
  assert.equal(replacementReads.length,preparedCount+1);assert.ok(replacementReads.every(value=>value==='c'.repeat(64)));
  const replacementOldURLs=new Set(replacementOldLayers.map(layer=>layer.href));assert.ok((await imageLayers()).filter(layer=>layer.detail).every(layer=>!replacementOldURLs.has(layer.href)));
  assert.equal(await evaluate('window.__morphologyImages.maxActive'),2);
  tileEvidence.push({dimensions,preparationTiles:preparedCount,retryReads:retried.length,startedDuringGestureMs:gestureEnd-initial[0].startedAt,idleActive:1,foregroundWithPredictionActive:2,predictedZoomReads:0,predictedPanReads:0,fittedImageCount:(await imageLayers()).filter(layer=>layer.detail).length,maxActive:2});
  await screenshot('morphology-progressive-tiles');await assertImageFingerprints('tiles');
}
async function assertAttentionTiles() {
  const label=name=>'attention: '+name, svgSelector='.attention-canvas svg';
  await cdp('Emulation.setDeviceMetricsOverride',{width:1200,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(join(output,'index.html')).href+'?mode=attention&hold=1'});
  await check(label('real attention viewer prepares a bounded startup cache'),"document.querySelector('.attention-canvas [data-slide-tile-layer]')?.getAttribute('data-preparation-total')==='56' && document.querySelector('.attention-canvas [data-slide-tile-layer]')?.getAttribute('data-preparing')==='true' && window.__morphologyImages.pending.length===2");
  await evaluate('window.__releaseMorphologyImages()');
  await check(label('prepared attention viewer reports ready with ranked markers'),"document.body.innerText.includes('Slide ready') && document.querySelector('.ranked-patch-marker[aria-label=\"Inspect rank 1, patch 1700\"]') && window.__morphologyImages.active===0");
  await click('Fit slide');
  await check(label('whole-slide fit is established before continuous zoom'),"document.querySelector('.attention-canvas svg').getAttribute('viewBox')==='0 0 60000 40000'");
  // Let deferred attention data settle at full extent; image eligibility must follow the next immediate camera state.
  await evaluate('new Promise(resolve=>setTimeout(resolve,220))');
  await check(label('full-slide attention overlay is ready before navigation'),"Number(document.querySelector('.attention-canvas image[opacity]')?.getAttribute('width'))===60000");
  const previousOverlay=await evaluate("Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(document.querySelector('.attention-canvas image[opacity]').getAttribute(key))]))");
  await evaluate('window.__attentionControl.hold=true');
  const geometry=await evaluate(`(()=>{const svg=document.querySelector(${JSON.stringify(svgSelector)});svg.scrollIntoView({block:'center'});const box=svg.getBoundingClientRect(),matrix=svg.getScreenCTM(),screen=new DOMPoint(20750,16600).matrixTransform(matrix);return{x:Math.round(screen.x),y:Math.round(screen.y),initialScale:Math.max(60000/box.width,40000/box.height)};})()`);
  const initialMarker=await evaluate("(()=>{const marker=document.querySelector('.ranked-patch-marker[aria-label=\"Inspect rank 1, patch 1700\"]');const rect=marker.querySelector('rect'),circle=marker.querySelectorAll('circle')[1];return {region:Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(rect.getAttribute(key))])),screenRadius:Number(circle.getAttribute('r'))*circle.getScreenCTM().a};})()");
  assert.deepEqual(initialMarker.region,{x:20000,y:16000,width:1500,height:1200});
  const firstDelta=Math.log(geometry.initialScale/1.5)/.004/8, requestStart=await evaluate('window.__morphologyCalls.length');
  await evaluate(`(async()=>{window.__morphologyImages.holdRegions=true;const svg=document.querySelector(${JSON.stringify(svgSelector)});for(let n=0;n<20;n++){svg.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:n<8?-${firstDelta}:-2,clientX:${geometry.x},clientY:${geometry.y}}));await new Promise(resolve=>setTimeout(resolve,50));}window.__attentionGestureEnd=performance.now();await ${frame};})()`);
  await check(label('uncached detail starts while continuous attention zoom is still running'),'window.__morphologyImages.pending.length===2 && window.__morphologyImages.active===2');
  const calls=await evaluate(`window.__morphologyCalls.slice(${requestStart}).filter(call=>call.path.includes('/region?'))`), ended=await evaluate('window.__attentionGestureEnd');
  assert.equal(calls.length,2,'Attention must hold at most two native region reads');
  assert.ok(calls[0].startedAt<ended-250,'Image tile eligibility must follow the immediate view; deferred attention overlays must not starve image loading until gesture end');
  assert.ok(calls.every(call=>new URL(call.path,'http://offline.invalid').searchParams.get('max_size')==='512'));
  const duringMarker=await evaluate("(()=>{const marker=document.querySelector('.ranked-patch-marker[aria-label=\"Inspect rank 1, patch 1700\"]');if(!marker)return null;const rect=marker.querySelector('rect'),circle=marker.querySelectorAll('circle')[1];return {region:Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(rect.getAttribute(key))])),screenRadius:Number(circle.getAttribute('r'))*circle.getScreenCTM().a};})()");
  assert.ok(duringMarker,'Visible ranked location must remain during partial tile loading');
  assert.deepEqual(duringMarker.region,initialMarker.region,'Ranked patch boxes retain exact level-0 placement through zoom');
  assert.ok(Math.abs(duringMarker.screenRadius-initialMarker.screenRadius)<.1,'Ranked marker circles retain readable screen size while zooming');
  checks.push(label('ranked patch coordinates and circle marker size survive progressive zoom'));
  const retainedOverlay=await evaluate("Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(document.querySelector('.attention-canvas image[opacity]').getAttribute(key))]))");
  assert.deepEqual(retainedOverlay,previousOverlay,'Pending attention must retain its original region without stretching onto the new camera');
  checks.push(label('previous attention overlay keeps its own coordinates while new data loads'));
  await evaluate('window.__releaseAttention();window.__releaseMorphologyImages()');
  await check(label('new detail resolves while attention overlay remains spatially aligned'),"(()=>{const svg=document.querySelector('.attention-canvas svg'),view=svg.viewBox.baseVal,map=svg.querySelector('image[opacity]');return window.__morphologyImages.active===0 && svg.querySelector('image[data-tile-level=\"0\"]') && map && Number(map.getAttribute('x'))===Math.floor(view.x) && Number(map.getAttribute('y'))===Math.floor(view.y) && Number(map.getAttribute('width'))===Math.ceil(view.x+view.width)-Math.floor(view.x) && Number(map.getAttribute('height'))===Math.ceil(view.y+view.height)-Math.floor(view.y);})()");
  assert.equal(await evaluate('window.__morphologyImages.maxActive'),2);
  tileEvidence.push({mode:'attention',preparationTiles:56,startedDuringGestureMs:ended-calls[0].startedAt,maxActive:2,rankedPatchRegion:duringMarker.region,markerScreenRadius:duringMarker.screenRadius});
  await screenshot('attention-progressive-tiles');
}
async function clickSlideLocation(x,y) {
  const point=await evaluate(`(()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)});svg.scrollIntoView({block:'center'});const screen=new DOMPoint(${x},${y}).matrixTransform(svg.getScreenCTM());return{x:screen.x,y:screen.y};})()`);
  await cdp('Input.dispatchMouseEvent',{type:'mousePressed',...point,button:'left',clickCount:1});
  await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',...point,button:'left',clickCount:1});
}
async function assertDenseOverlayAndReload() {
  const label=name=>'dense QC: '+name;
  await cdp('Emulation.setDeviceMetricsOverride',{width:1200,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(join(output,'index.html')).href+'?mode=patch&dense=1'});
  await check(label('explorer opens with dense recorded geometry'),"document.body?.innerText.includes('Open visual review & feature explorer')");await click('Open visual review & feature explorer');
  await check(label('4096 patches and 50000 vertices prepare as one fitted image'),"document.body.innerText.includes('Slide ready') && document.body.innerText.includes('4,096 extracted patches') && document.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='false' && document.querySelectorAll('image[data-quality-overlay]').length===1");
  const initialDOM=await evaluate(`(()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)});return {rects:svg.querySelectorAll('rect').length,paths:svg.querySelectorAll('path').length,images:svg.querySelectorAll('image[data-quality-overlay]').length,all:svg.querySelectorAll('*').length};})()`);
  assert.ok(initialDOM.rects<10 && initialDOM.paths<10,'Dense geometry must not recreate thousands of SVG hit-test or paint nodes');
  await screenshot('morphology-dense-fitted');
  await clickSlideLocation(2970,1980);
  await check(label('raster click selects the exact level-0 patch'),"document.body.innerText.includes('Original patch 2080') && document.body.innerText.includes('Selected region: (2940, 1960), 60') && document.querySelector('rect[data-selected-patch]')");
  const selected=await evaluate("Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(document.querySelector('rect[data-selected-patch]').getAttribute(key))]))");
  assert.deepEqual(selected,{x:2940,y:1960,width:60,height:40});
  const patchCalls=await evaluate("window.__morphologyCalls.filter(call=>call.path.includes('/morphology/patch?') || call.path.includes('/morphology/patch-region?'))");
  assert.ok(patchCalls.some(call=>call.path.includes('/patch-region?'))&&patchCalls.some(call=>call.path.includes('/patch?')));
  for(const call of patchCalls)assert.equal(new URL(call.path,'http://offline.invalid').searchParams.get('sourceFingerprint'),'a'.repeat(64));
  checks.push(label('patch crop and geometry carry the current source fingerprint'));
  await click('Zoom to selection');
  await check(label('zoom sharpens coverage with at most two cached overlay images'),"document.querySelector('image[data-quality-overlay=detail]') && document.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='false'");
  assert.equal(await evaluate("document.querySelectorAll('image[data-quality-overlay]').length"),2);
  await screenshot('morphology-dense-selected-patch');
  const previous=await evaluate("(()=>{const image=document.querySelector('image[data-quality-overlay=detail]');return {href:image.getAttribute('href'),...Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(image.getAttribute(key))]))};})()");
  await evaluate(`window.__rasterControl.hold=true;document.querySelector(${JSON.stringify(slideSvg)}).dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,key:'ArrowRight'}))`);
  await check(label('previous overlay stays visible while new detail renders'),"window.__rasterControl.pending.length>0 && document.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='true'");
  const retained=await evaluate("(()=>{const image=document.querySelector('image[data-quality-overlay=detail]');return {href:image.getAttribute('href'),...Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(image.getAttribute(key))]))};})()");
  assert.deepEqual(retained,previous,'Pending raster must retain its original placement, not stretch to the moving camera');
  assert.ok(await evaluate("document.querySelector('image[data-quality-overlay=overview]').hasAttribute('clip-path')"),'Overview excludes the detailed region to avoid drawing alpha twice');
  await evaluate('window.__releaseRasters()');
  await check(label('replacement overlay uses its own current camera coordinates'),`(()=>{const image=document.querySelector('image[data-quality-overlay=detail]'),view=document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal;return image&&['x','y','width','height'].every(key=>Math.abs(Number(image.getAttribute(key))-view[key])<.001)&&document.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='false';})()`);
  await evaluate("Array.from(document.querySelectorAll('label')).find(label=>label.textContent==='Patch coverage').querySelector('input').click();Array.from(document.querySelectorAll('label')).find(label=>label.textContent==='Recorded tissue contours').querySelector('input').click()");
  await evaluate(frame);assert.equal(await evaluate("document.querySelectorAll('image[data-quality-overlay]').length"),0);checks.push(label('turning both overlays off removes their images immediately'));
  await evaluate("Array.from(document.querySelectorAll('label')).find(label=>label.textContent==='Patch coverage').querySelector('input').click()");
  await check(label('coverage can be restored independently of contours'),"document.querySelector('image[data-quality-overlay=overview]') && document.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='false'");
  await click('Draw review region');
  const drawing=await evaluate(`(()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)}),view=svg.viewBox.baseVal,matrix=svg.getScreenCTM();const start=new DOMPoint(view.x+view.width*.2,view.y+view.height*.2).matrixTransform(matrix),end=new DOMPoint(view.x+view.width*.8,view.y+view.height*.8).matrixTransform(matrix);return {start:{x:start.x,y:start.y},end:{x:end.x,y:end.y},region:{x:Math.floor(view.x+view.width*.2),y:Math.floor(view.y+view.height*.2),width:Math.floor(view.width*.6),height:Math.floor(view.height*.6)}};})()`);
  await cdp('Input.dispatchMouseEvent',{type:'mousePressed',...drawing.start,button:'left',clickCount:1});await cdp('Input.dispatchMouseEvent',{type:'mouseMoved',...drawing.end,button:'left',buttons:1});await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',...drawing.end,button:'left',clickCount:1});
  await check(label('ROI drawing remains independent from raster patch picking'),"!document.body.innerText.includes('Drag a rectangle') && document.body.innerText.includes('Selected region: (')");
  await click('Save selected region with review');await click('Save review');
  await check(label('ROI retains exact slide coordinates after raster navigation'),"document.body.innerText.includes('Review saved · revision 1')");
  const saved=await evaluate("window.__morphologyCalls.find(call=>call.body?.regions?.length===1).body.regions[0]");
  for(const key of ['x','y','width','height'])assert.ok(Math.abs(saved[key]-drawing.region[key])<=1,'Dense raster navigation must preserve ROI '+key);
  await evaluate("document.querySelector('button[aria-label=\"Zoom out\"]').click()");
  await check(label('selected patch and saved ROI remain visible together after zooming out'),`(()=>{const svg=document.querySelector(${JSON.stringify(slideSvg)}),image=svg.querySelector('image[data-quality-overlay=detail]');return svg.querySelector('rect[data-selected-patch]')&&image&&Math.abs(Number(image.getAttribute('width'))-svg.viewBox.baseVal.width)<.001&&svg.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='false';})()`);
  await screenshot('morphology-dense-selected-roi');
  await click('Fit slide');await check(label('fitted dense view returns to one overlay image'),"document.querySelectorAll('image[data-quality-overlay]').length===1 && document.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='false'");
  const oldURLs=await evaluate("Array.from(document.querySelectorAll('.morphology-slide svg image'),image=>image.getAttribute('href'))");
  await evaluate("window.__morphologyFingerprints.a='d'.repeat(64);window.__morphologySource.geometry.a={width:7200,height:4800}");
  await clickSlideLocation(3060,1980);
  await check(label('changed patch source hides cached images overlays crops and review'),"Array.from(document.querySelectorAll('button')).some(button=>button.textContent==='Reload slide') && !document.querySelector('.morphology-slide svg') && !document.querySelector('.morphology-patch') && !document.querySelector('.slide-review-editor')");
  const reloadStart=await evaluate('window.__morphologyCalls.length');await click('Reload slide');
  await check(label('Reload slide rereads replacement geometry and prepares its new source'),`document.querySelector(${JSON.stringify(slideSvg)})?.getAttribute('viewBox')==='0 0 7200 4800' && document.body.innerText.includes('Slide ready') && !document.querySelector('rect[data-selected-patch]') && !document.querySelector('.morphology-patch')`);
  const refreshed=await evaluate(`window.__morphologyCalls.slice(${reloadStart})`);
  assert.ok(refreshed.some(call=>call.path.includes('/quality?')&&!call.path.includes('featureBundleId'))&&refreshed.some(call=>call.path.includes('/quality?')&&call.path.includes('featureBundleId')),'Reload must obtain fresh geometry and optional evidence');
  assert.ok(refreshed.filter(call=>call.path.includes('/morphology/image?')).every(call=>new URL(call.path,'http://offline.invalid').searchParams.get('sourceFingerprint')==='d'.repeat(64)));
  assert.ok(await evaluate(`Array.from(document.querySelectorAll('.morphology-slide svg image'),image=>image.getAttribute('href')).every(url=>!${JSON.stringify(oldURLs)}.includes(url))`),'Replacement must not mount a former overview, tile or overlay image');
  await evaluate('window.__morphologySource.blocked=true');await clickSlideLocation(3150,1980);
  await check(label('changed frozen source exposes explicit reload'),"Array.from(document.querySelectorAll('button')).some(button=>button.textContent==='Reload slide') && !document.querySelector('.morphology-slide svg')");
  await click('Reload slide');
  await check(label('a still-invalid frozen source stays blocked after reload'),"document.body.innerText.includes('This frozen slide has changed on disk') && !document.querySelector('.morphology-slide svg') && !document.querySelector('.slide-review-editor')");
  await evaluate('window.__morphologySource.blocked=false');await click('Reload slide');
  await check(label('restored source recovers through another explicit reload'),"document.querySelector('.morphology-slide svg') && document.body.innerText.includes('Slide ready')");
  overlayEvidence.push({patches:4096,vertices:50000,initialDOM,selected,saved,overlayImagesMax:2,replacementView:[0,0,7200,4800],sourceFailureHidesReview:true});
  await check(label('restored fitted overlay finishes preparing before visual capture'),"document.querySelector('[data-quality-overlay-layer]')?.getAttribute('data-overlay-loading')==='false' && document.body.innerText.includes('Slide ready')");
  await screenshot('morphology-dense-overlays');
}
async function assertGestureFlow(mode) {
  const label = name => `${mode}: ${name}`;
  await check(label('startup cache is ready before gesture checks'), 'document.body.innerText.includes("Slide ready") && window.__morphologyImages.active===0');
  const before = await viewport(), anchor = await wheelPoint();
  const browserScale = await evaluate('visualViewport.scale');
  await evaluate("window.__wheelEvidence=[]; document.addEventListener('wheel', event => window.__wheelEvidence.push({trusted:event.isTrusted,ctrl:event.ctrlKey,prevented:event.defaultPrevented}), {passive:true})");
  await cdp('Input.dispatchMouseEvent', { type: 'mouseWheel', x: anchor.x, y: anchor.y, deltaX: 0, deltaY: -120, modifiers: 2 });
  await check(label('trusted Ctrl+wheel zooms into the slide'), `document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width < ${before[2]} && window.__wheelEvidence.some(event => event.trusted && event.ctrl && event.prevented)`);
  const zoomed = await viewport();
  const afterAnchor = await evaluate(`(() => { const value = new DOMPoint(${anchor.x},${anchor.y}).matrixTransform(document.querySelector(${JSON.stringify(slideSvg)}).getScreenCTM().inverse()); return {x:value.x,y:value.y}; })()`);
  assert.ok(Math.abs(anchor.level0.x - afterAnchor.x) < 1 && Math.abs(anchor.level0.y - afterAnchor.y) < 1, label('zoom anchor must remain under the cursor: ' + JSON.stringify({anchor,afterAnchor,before,zoomed})));
  assert.equal(await evaluate('visualViewport.scale'), browserScale, label('viewer consumes Ctrl+wheel without browser zoom'));
  checks.push(label('Ctrl+wheel preserves cursor position and browser scale'));
  await cdp('Input.dispatchMouseEvent', { type: 'mouseWheel', x: anchor.x, y: anchor.y, deltaX: 0, deltaY: 60, modifiers: 2 });
  await check(label('Ctrl+wheel zooms back out'), `document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width > ${zoomed[2]}`);
  await click('Fit slide');
  await check(label('Fit slide restores full level-0 bounds'), `document.querySelector(${JSON.stringify(slideSvg)}).getAttribute('viewBox') === '0 0 6000 4000'`);
  if(mode==='slide'){await assertSmoothWheel();await assertWheelPanHandoff();}
  await assertOrdinaryWheelFlow(label);
  const burstAnchor = await wheelPoint();
  await evaluate(`(() => {const svg=document.querySelector(${JSON.stringify(slideSvg)}); svg.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,ctrlKey:true,deltaY:-12,clientX:${burstAnchor.x},clientY:${burstAnchor.y}}));})()`);
  await check(label('small trackpad delta updates the view'), `document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width < 6000`);
  const stepRatio = (await viewport())[2] / 6000;
  await click('Fit slide');
  await check(label('fit resets between gesture samples'), `document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.width === 6000`);
  const requestStart = await evaluate('window.__morphologyCalls.length');
  await evaluate(`(async () => {
    window.__burstPrevented=[];
    const svg=document.querySelector(${JSON.stringify(slideSvg)});
    for(let index=0;index<12;index++) { const event=new WheelEvent('wheel',{bubbles:true,cancelable:true,ctrlKey:true,deltaY:-12,clientX:${burstAnchor.x},clientY:${burstAnchor.y}}); svg.dispatchEvent(event); window.__burstPrevented.push(event.defaultPrevented); await new Promise(resolve=>setTimeout(resolve,10)); }
    await ${frame};
  })()`);
  const burst = await viewport();
  assert.ok(Math.abs(burst[2] - 6000 * stepRatio ** 12) < 1, label('every small pinch delta must accumulate, including events between React renders'));
  assert.equal(await evaluate('window.__burstPrevented.every(Boolean)'), true);
  const base = (await imageLayers()).find(layer => !layer.detail);
  assert.deepEqual([base.x,base.y,base.width,base.height], [0,0,6000,4000], label('base image remains at its full original coordinates'));
  await check(label('progressive detail tiles resolve over persistent overview'), `document.querySelector(${JSON.stringify(slideSvg + ' image[data-slide-tile]')})?.getAttribute('href')?.startsWith('blob:') && window.__morphologyImages.active === 0 && !document.body.innerText.includes('Loading detail…')`);
  const cropCount = await evaluate(`window.__morphologyCalls.slice(${requestStart}).filter(call => call.path.includes('/image?') && call.path.includes('&x=')).length`);
  assert.ok(await evaluate('window.__morphologyImages.maxActive <= 2'),label('progressive image reads must stay within two concurrent requests'));
  checks.push(label('trackpad-style burst accumulates smoothly with bounded concurrent tile reads'));
  const panAnchor = await wheelPoint(), beforePan = await viewport();
  await cdp('Input.dispatchMouseEvent', { type:'mousePressed', x:panAnchor.x, y:panAnchor.y, button:'left', clickCount:1 });
  await cdp('Input.dispatchMouseEvent', { type:'mouseMoved', x:panAnchor.x+30, y:panAnchor.y+20, button:'left', buttons:1 });
  await cdp('Input.dispatchMouseEvent', { type:'mouseReleased', x:panAnchor.x+30, y:panAnchor.y+20, button:'left', clickCount:1 });
  await check(label('drag pans the zoomed slide'), `document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.x < ${beforePan[0]} && document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.y < ${beforePan[1]}`);
  const releasedView = await viewport(), releaseAnchor = await wheelPoint();
  await cdp('Input.dispatchMouseEvent', {type:'mousePressed',x:releaseAnchor.x,y:releaseAnchor.y,button:'left',clickCount:1});
  await cdp('Input.dispatchMouseEvent', {type:'mouseReleased',x:1,y:1,button:'left',clickCount:1});
  await cdp('Input.dispatchMouseEvent', {type:'mouseMoved',x:releaseAnchor.x+45,y:releaseAnchor.y+20,buttons:0});
  await evaluate(frame);
  assert.deepEqual(await viewport(), releasedView, label('release outside the slide must not leave a stale drag on hover'));
  checks.push(label('release outside followed by hover cannot pan the slide'));
  gestureEvidence.push({mode,before,zoomed,burst,cropCount,afterPan:await viewport()});
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable');
  await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
  if (process.env.HISTOPILOT_VERIFY_SCOPE !== 'dense') {
    await cdp('Page.navigate', { url: pathToFileURL(join(output, 'index.html')).href });
    await check('Explorer entry loads', "document.body?.innerText.includes('Open visual review & feature explorer')");
    await click('Open visual review & feature explorer');
    await check('Frozen slide and independent review load', "document.querySelector('[aria-label=\"Visual quality of a\"]') && document.body.innerText.includes('Slide review')");
    await check('Patch capabilities resolve before projection is enabled', "window.__morphologyCalls.some(x => x.path.endsWith('/configurations/patch-features')) && [...document.querySelectorAll('button')].some(x => x.textContent === 'Build morphology projection' && !x.disabled)");
    await click('Build morphology projection');
    await check('Projection and measured neighbors appear', "document.querySelector('[aria-label=\"Slide morphology PCA projection\"]') && document.body.innerText.includes('0.9200')");
    await evaluate(`(() => { const field = [...document.querySelectorAll('label')].find(x => x.textContent.startsWith('Color by metadata')).querySelector('select'); field.value = 'site'; field.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    await check('Metadata groups linked to projection points', "document.querySelector('[aria-label=\"Inspect a, Site 1\"]') && document.body.innerText.includes('Site 2')");
    await click('Inspect slide');
    await check('Neighbor opens its exact slide', "document.querySelector('[aria-label=\"Visual quality of b\"]') && window.__morphologyCalls.some(x=>x.path.includes('/image?') && x.path.includes('slideId=b'))");
    await evaluate(`(() => { const field = [...document.querySelectorAll('label')].find(x => x.textContent.startsWith('Compare')).querySelector('select'); field.value = 'patch'; field.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    await check('Sampled patch retrieval is labeled', "document.body.innerText.includes('Search covers only sampled patches')");
    await click('View query patch');
    await check('Selected patch obtains exact geometry and crop', "document.body.innerText.includes('Original patch 0') && document.body.innerText.includes('Selected region: (200, 150)')");
    await click('Zoom to selection');
    await check('Zoom requests fixed-size detail tiles over selected original region', "window.__morphologyCalls.some(call=>{if(!call.path.includes('/image?'))return false;const p=new URL(call.path,'http://offline.invalid').searchParams;return Number(p.get('max_size'))<=512 && p.has('x') && Number(p.get('x'))<=200 && Number(p.get('x'))+Number(p.get('width'))>=350;})");
    await evaluate(`document.querySelector(${JSON.stringify(slideSvg)}).scrollIntoView({block:'center'})`);
    const patchPan = await evaluate(`(() => {const matrix=document.querySelector(${JSON.stringify(slideSvg)}).getScreenCTM(); const start=new DOMPoint(250,190).matrixTransform(matrix); return {x:start.x,y:start.y};})()`);
    const patchPanBefore = await viewport();
    await cdp('Input.dispatchMouseEvent', {type:'mousePressed',...patchPan,button:'left',clickCount:1});
    await cdp('Input.dispatchMouseEvent', {type:'mouseMoved',x:patchPan.x+30,y:patchPan.y+20,button:'left',buttons:1});
    await cdp('Input.dispatchMouseEvent', {type:'mouseReleased',x:patchPan.x+30,y:patchPan.y+20,button:'left',clickCount:1});
    await check('Dragging patch coverage pans without selecting a different patch', `document.querySelector(${JSON.stringify(slideSvg)}).viewBox.baseVal.x < ${patchPanBefore[0]} && document.body.innerText.includes('Original patch 0') && !document.body.innerText.includes('Original patch 3')`);
    await click('Save selected region with review');
    await evaluate(`(() => { const field = [...document.querySelectorAll('label')].find(x => x.textContent.startsWith('Decision')).querySelector('select'); field.value = 'review'; field.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    await click('Save review');
    await check('Review saves revision and exact selected coordinates', "window.__morphologyCalls.some(x=>x.body?.status === 'review' && x.body?.expectedRevision === 0 && x.body?.regions[0]?.x === 200 && x.body?.regions[0]?.width === 150) && document.body.innerText.includes('Review saved · revision 1')");
    await click('Fit slide');
    await check('Full slide image restored', `document.querySelector(${JSON.stringify(slideSvg)}).getAttribute('viewBox')==='0 0 6000 4000' && document.querySelector(${JSON.stringify(slideSvg+' image:not([data-slide-detail])')})?.getAttribute('href')?.startsWith('blob:')`);
    await click('Draw review region');
    await evaluate("document.querySelector('svg[aria-label=\"Exact slide b with patch coverage\"]').scrollIntoView({block:'center'})");
    const box = await evaluate(`(() => {const rect=document.querySelector('svg[aria-label="Exact slide b with patch coverage"]').getBoundingClientRect(); return {x:rect.x,y:rect.y,width:rect.width,height:rect.height};})()`);
    await cdp('Input.dispatchMouseEvent', { type:'mousePressed', x:box.x+box.width*.1, y:box.y+box.height*.1, button:'left', clickCount:1 });
    await cdp('Input.dispatchMouseEvent', { type:'mouseMoved', x:box.x+box.width*.3, y:box.y+box.height*.3, button:'left', buttons:1 });
    await cdp('Input.dispatchMouseEvent', { type:'mouseReleased', x:box.x+box.width*.3, y:box.y+box.height*.3, button:'left', clickCount:1 });
    await check('Drawn region survives the concluding patch click', "document.body.innerText.includes('Selected region: (') && !document.body.innerText.includes('Selected region: (200, 150)') && !document.body.innerText.includes('Drag a rectangle')");
    await screenshot('morphology-desktop');
    await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await check('Explorer fits mobile viewport', 'document.documentElement.scrollWidth <= innerWidth');
    await screenshot('morphology-mobile');
    await assertImageFingerprints('patch');
    for (const mode of ['slide', 'broken']) {
      await cdp('Emulation.setDeviceMetricsOverride', { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
      await cdp('Page.navigate', { url: pathToFileURL(join(output, 'index.html')).href + '?mode=' + mode });
      await check(`${mode}: explorer entry loads`, "document.body?.innerText.includes('Open visual review & feature explorer')");
      await click('Open visual review & feature explorer');
      await check(`${mode}: original image uses exact level-0 geometry`, `document.querySelector('svg[aria-label="Exact slide a"]')?.getAttribute('viewBox') === '0 0 6000 4000' && document.querySelector('svg[aria-label="Exact slide a"] image')?.getAttribute('href')?.startsWith('blob:')`);
      if (mode === 'slide') {
        await check('Slide vectors expose image review without patch tools', "document.body.innerText.includes('One embedding per slide') && !document.body.innerText.includes('Build morphology projection') && !document.body.innerText.includes('Patch coverage') && !document.body.innerText.includes('Inspect exact patch index')");
        await check('Slide kind resolves from frozen feature configuration', "window.__morphologyCalls.some(x => x.path.endsWith('/configurations/slide-features')) && !window.__morphologyCalls.some(x => x.path.includes('/morphology/index'))");
      } else {
        await check('Optional coverage failure leaves original image and notes available', "document.body.innerText.includes('Optional feature coverage is unavailable') && document.querySelector('textarea') && document.querySelector('svg[aria-label=\"Exact slide a\"] image') && window.__morphologyCalls.some(x=>x.path.includes('/quality?') && !x.path.includes('featureBundleId'))");
      }
      await assertGestureFlow(mode);
      await click('Draw review region');
      const draw = await evaluate(`(() => {
        const svg = document.querySelector('svg[aria-label="Exact slide a"]');
        svg.scrollIntoView({block:'center'});
        const matrix = svg.getScreenCTM();
        const field = svg.viewBox.baseVal;
        const start = new DOMPoint(field.x + field.width * .15, field.y + field.height * .15).matrixTransform(matrix);
        const end = new DOMPoint(field.x + field.width * .7, field.y + field.height * .7).matrixTransform(matrix);
        const first = start.matrixTransform(matrix.inverse()), last = end.matrixTransform(matrix.inverse());
        return {start:{x:start.x,y:start.y},end:{x:end.x,y:end.y},region:{x:Math.floor(first.x),y:Math.floor(first.y),width:Math.floor(last.x-first.x),height:Math.floor(last.y-first.y)}};
      })()`);
      await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', ...draw.start, button: 'left', clickCount: 1 });
      await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', ...draw.end, button: 'left', buttons: 1 });
      await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', ...draw.end, button: 'left', clickCount: 1 });
      await check(`${mode}: drawn ROI uses full-resolution coordinates`, `document.body.innerText.includes('Selected region: (') && [...document.querySelectorAll('button')].some(x => x.textContent === 'Save selected region with review')`);
      await click('Save selected region with review');
      await click('Save review');
      await check(`${mode}: ROI saves successfully without patch coordinates`, "document.body.innerText.includes('Review saved · revision 1') && window.__morphologyCalls.some(x=>x.body?.regions?.length === 1)");
      const saved = await evaluate("window.__morphologyCalls.find(x=>x.body?.regions?.length === 1).body.regions[0]");
      for (const key of ['x', 'y', 'width', 'height']) assert.ok(Math.abs(saved[key] - draw.region[key]) <= 1, `${mode} ${key} must preserve level-0 coordinates`);
      assert.ok(saved.x > 600 && saved.width > 600 && saved.height > 400, 'ROI must not use thumbnail pixels');
      const drawn = await evaluate("(() => {const rect=document.querySelector('svg[aria-label=\"Exact slide a\"] rect[pointer-events=\"none\"]');return Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(rect.getAttribute(key))]));})()");
      for (const key of ['x', 'y', 'width', 'height']) assert.equal(saved[key], drawn[key], `${mode}: saved ROI equals displayed geometry`);
      roiEvidence.push({ mode, thumbnail: [600, 400], level0: [6000, 4000], saved });
      checks.push(`${mode}: saved ROI exactly matches displayed level-0 rectangle`);
      await screenshot(`morphology-${mode}-desktop`);
      await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
      await check(`${mode}: review fits mobile viewport`, 'document.documentElement.scrollWidth <= innerWidth');
      await screenshot(`morphology-${mode}-mobile`);
      await assertImageFingerprints(mode);
    }
    await assertTiledReading();
    await assertAttentionTiles();
  }
  await assertDenseOverlayAndReload();
  assert.deepEqual(exceptions, [], 'No unexpected browser errors');
  await writeFile(join(output, process.env.HISTOPILOT_VERIFY_SCOPE === 'dense' ? 'dense-checks.json' : 'checks.json'), JSON.stringify({ passed: true, checks, roiEvidence, gestureEvidence, tileEvidence, overlayEvidence, exceptions }, null, 2) + '\n');
  console.log('Artifacts: ' + output);
} finally {
  browser.kill();
}
