import { configDefaults, defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true } },
  },
  build: { outDir: 'dist', emptyOutDir: true },
  // The browser tests in e2e/ run in Playwright against a served demo (npm run e2e).
  test: { exclude: [...configDefaults.exclude, 'e2e/**'] },
});
