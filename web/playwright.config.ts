import { defineConfig, devices } from '@playwright/test';

// The browser test of the public demo (npm run e2e). Playwright starts the demo service itself:
// HISTOPILOT_E2E_SERVER is the command (CI runs the wheel it just built; the default is this
// checkout's venv, which needs the UI bundled into histopilot/static by scripts/bundle_web.py),
// on HISTOPILOT_E2E_PORT. HISTOPILOT_E2E_URL tests a demo that is already running instead,
// such as a Hugging Face Space.
const port = Number(process.env.HISTOPILOT_E2E_PORT ?? 8826);
const external = process.env.HISTOPILOT_E2E_URL;
const baseURL = external ?? `http://127.0.0.1:${port}`;
const command =
  process.env.HISTOPILOT_E2E_SERVER ?? `../.venv/bin/histopilot serve --demo --port ${port} --no-browser`;

export default defineConfig({
  testDir: 'e2e',
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  workers: 1,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: { baseURL, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: external
    ? undefined
    : {
        command,
        url: `${baseURL}/api/v1/health`,
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
        stdout: 'pipe',
      },
});
