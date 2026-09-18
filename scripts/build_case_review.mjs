// Build an offline file:// fixture containing the real case-review editor.
import { build } from '../web/node_modules/vite/dist/node/index.js';
import react from '../web/node_modules/@vitejs/plugin-react/dist/index.js';
import tailwindcss from '../web/node_modules/@tailwindcss/vite/dist/index.mjs';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(process.argv[2] ?? '/tmp/histopilot-current-review/case-review-browser');
await build({
  configFile: false, root: resolve(root, 'web'), plugins: [react(), tailwindcss()],
  build: {
    outDir: resolve(output, 'assets'), emptyOutDir: true,
    lib: { entry: resolve(root, 'web/verification/CaseReviewFixture.tsx'), formats: ['iife'], name: 'CaseReview', fileName: () => 'case-review.js' },
    cssCodeSplit: false,
  },
  resolve: { dedupe: ['react', 'react-dom'] }, define: { 'process.env.NODE_ENV': JSON.stringify('production') },
});
const files = await readdir(resolve(output, 'assets'));
const css = (await Promise.all(files.filter((name) => name.endsWith('.css')).map((name) => readFile(resolve(output, 'assets', name), 'utf8')))).join('\n');
const js = await readFile(resolve(output, 'assets/case-review.js'), 'utf8');
await writeFile(resolve(output, 'index.html'), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline case-review verification</title><style>${css}</style></head><body><div id="app"></div><script>${js.replaceAll('</script', '<\\/script')}</script></body></html>`);
console.log(`Offline fixture: ${output}/index.html`);
