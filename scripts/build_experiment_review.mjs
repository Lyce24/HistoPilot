// Build a standalone file:// fixture; never opens a port or contacts the control service.
import { build } from '../web/node_modules/vite/dist/node/index.js';
import react from '../web/node_modules/@vitejs/plugin-react/dist/index.js';
import tailwindcss from '../web/node_modules/@tailwindcss/vite/dist/index.mjs';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const repository = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(process.argv[2] ?? '/tmp/histopilot-experiment-review/browser');
await build({
  configFile: false, root: resolve(repository, 'web'), plugins: [react(), tailwindcss()],
  build: {
    outDir: resolve(output, 'assets'), emptyOutDir: true,
    lib: { entry: resolve(repository, 'web/verification/ExperimentFixture.tsx'), formats: ['iife'], name: 'ExperimentReviewHarness', fileName: () => 'review.js' },
    cssCodeSplit: false, minify: false,
  },
  resolve: { dedupe: ['react', 'react-dom'] }, define: { 'process.env.NODE_ENV': JSON.stringify('production') },
});
const assets = await readdir(resolve(output, 'assets'));
const css = (await Promise.all(assets.filter((name) => name.endsWith('.css')).map((name) => readFile(resolve(output, 'assets', name), 'utf8')))).join('\n');
const js = await readFile(resolve(output, 'assets/review.js'), 'utf8');
const favicon = await readFile(resolve(repository, 'web/public/favicon.svg'));
const inlineJs = js.replaceAll('"/favicon.svg"', JSON.stringify(`data:image/svg+xml;base64,${favicon.toString('base64')}`));
await writeFile(resolve(output, 'index.html'), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HistoPilot experiments offline verification</title><style>${css}</style></head><body><div id="app"></div><script>${inlineJs.replaceAll('</script', '<\\/script')}</script></body></html>`);
console.log(`Offline fixture: ${resolve(output, 'index.html')}`);
