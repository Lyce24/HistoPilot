/** Compare current route loading against an original App.tsx without changing sources or web/dist.
 * Usage: node scripts/measure-page-loading.mjs /path/to/original-App.tsx
 */
import { build } from 'vite';
import { gzipSync } from 'node:zlib';
import { readFile, mkdtemp, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

if (!process.argv[2]) throw new Error('Pass the original eager-import App.tsx as the first argument.');
const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const original = await readFile(resolve(process.argv[2]), 'utf8');
const output = await mkdtemp(join(tmpdir(), 'histopilot-route-size-'));
const results = {};

for (const baseline of [true, false]) {
  const directory = join(output, baseline ? 'before' : 'after');
  await build({
    root: web,
    configFile: join(web, 'vite.config.ts'),
    logLevel: 'silent',
    plugins: baseline ? [{
      name: 'measure-original-app',
      enforce: 'pre',
      transform(_, id) {
        if (id === join(web, 'src/App.tsx')) return { code: original, map: null };
      },
    }] : [],
    build: { outDir: directory, manifest: true },
  });
  const manifest = JSON.parse(await readFile(join(directory, '.vite/manifest.json'), 'utf8'));
  const files = new Set();
  const visited = new Set();
  function collect(key) {
    if (visited.has(key)) return;
    visited.add(key);
    const chunk = manifest[key];
    files.add(chunk.file);
    for (const file of chunk.css ?? []) files.add(file);
    for (const child of chunk.imports ?? []) collect(child);
  }
  collect('index.html');
  // No-project navigation immediately opens Start; include all its shared dependencies.
  if (manifest['src/pages/Start.tsx']) collect('src/pages/Start.tsx');
  let jsBytes = 0, jsGzipBytes = 0, cssBytes = 0, cssGzipBytes = 0;
  for (const file of files) {
    const bytes = await readFile(join(directory, file));
    if (file.endsWith('.js')) { jsBytes += bytes.length; jsGzipBytes += gzipSync(bytes).length; }
    if (file.endsWith('.css')) { cssBytes += bytes.length; cssGzipBytes += gzipSync(bytes).length; }
  }
  results[baseline ? 'before' : 'after'] = {
    entryBytes: (await readFile(join(directory, manifest['index.html'].file))).length,
    jsBytes, jsGzipBytes, cssBytes, cssGzipBytes, files: [...files].sort(),
  };
}
results.reductionPercent = 100 * (1 - results.after.jsBytes / results.before.jsBytes);
const artifact = join(output, 'measurement.json');
await writeFile(artifact, JSON.stringify(results, null, 2) + '\n');
console.log(JSON.stringify({ ...results, artifact }, null, 2));
