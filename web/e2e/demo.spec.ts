import { expect, test, type Page } from '@playwright/test';

// A visitor's walk through the public demo, in a real browser against a served demo: the start
// page, the BLCA walkthrough's overview, then every stage, record and step of the packaged
// fixture. Each step must show what the fixture holds (facts, every table row, a drawn chart,
// training runs). Throughout, the page must not send a change to the service, offer a control
// that would make one, fail an API read or log an error.

interface DemoStep {
  id: string;
  title: string;
  facts?: { label: string; value: string }[];
  table?: { rows: unknown[][] };
  chart?: { series: { points: unknown[] }[] };
  runs?: unknown;
}
interface DemoRecord {
  id: string;
  module: string;
  name: string;
  steps: DemoStep[];
}

const DEMO = 'blca-demo-v1';
const READS = new Set(['GET', 'HEAD', 'OPTIONS']);
// Buttons whose name starts like this would change a study. The walkthrough's step tabs, such
// as "Freeze design" and "Start experiment", describe those stages from inside navigation,
// which this check leaves out; so do links within the page and links to other sites.
const OFFERS_CHANGE =
  /^(save|freeze|start|launch|train|apply|run|delete|remove|trash|archive|create|new|import|upload|submit|confirm|approve|retry|cancel|rename)\b/i;

async function fixture(page: Page): Promise<{ records: DemoRecord[]; slideCount: number }> {
  const session = await (await page.request.get('/api/v1/session')).json();
  const response = await page.request.get(`/api/v1/projects/${DEMO}/workspace`, {
    headers: { 'X-HistoPilot-Token': session.token },
  });
  expect(response.ok(), 'the demo workspace loads').toBeTruthy();
  const workspace = await response.json();
  return { records: workspace.demoPipeline.records, slideCount: workspace.dataset.slideCount };
}

async function expectNothingOffersAChange(page: Page, where: string) {
  const offers = await page.evaluate((pattern) => {
    const changes = new RegExp(pattern, 'i');
    const controls = [...document.querySelectorAll<HTMLElement>('button, [role="button"], input[type="submit"]')];
    return controls
      .filter((control) => !control.closest('nav') && control.offsetParent !== null)
      .map((control) => (control.getAttribute('aria-label') ?? control.textContent ?? '').trim())
      .filter((name) => changes.test(name));
  }, OFFERS_CHANGE.source);
  expect(offers, `controls offering a change on ${where}`).toEqual([]);
  await expect(page.locator('textarea, input[type="file"]'), `editable fields on ${where}`).toHaveCount(0);
}

test('a visitor walks every stage, record and step of the demo without being offered a change', async ({ page }) => {
  const changes: string[] = [];
  const failedReads: string[] = [];
  const errors: string[] = [];
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.startsWith('/api/') && !READS.has(request.method())) {
      changes.push(`${request.method()} ${request.url()}`);
    }
  });
  page.on('response', (response) => {
    if (new URL(response.url()).pathname.startsWith('/api/') && response.status() >= 400) {
      failedReads.push(`${response.status()} ${response.request().method()} ${response.url()}`);
    }
  });
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });

  await page.goto('/');
  await expect(page.getByText('Public demo · read-only')).toBeVisible();
  const { records, slideCount } = await fixture(page);
  expect(records.length, 'the fixture has walkthrough records').toBeGreaterThan(0);
  await expectNothingOffersAChange(page, 'the start page');

  await page.getByRole('button', { name: /Open the BLCA demo/ }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'BLCA demo' })).toBeVisible();
  await expect(page.getByRole('note').filter({ hasText: 'Synthetic · Read-only' })).toBeVisible();
  await expect(page.locator('.blca-demo-summary dd').first()).toHaveText(String(slideCount));
  await expectNothingOffersAChange(page, 'the overview');

  const modules = [...new Set(records.map((record) => record.module))];
  let visited = 0;
  for (const module of modules) {
    await page.evaluate((hash) => {
      window.location.hash = hash;
    }, module);
    const moduleRecords = records.filter((record) => record.module === module);
    await expect(page.locator('[data-demo-record]'), `records listed for ${module}`).toHaveCount(moduleRecords.length);
    await expectNothingOffersAChange(page, `the ${module} records`);

    for (const record of moduleRecords) {
      await page.locator(`[data-demo-record="${record.id}"]`).click();
      await expect(page.getByRole('heading', { level: 2, name: record.name, exact: true })).toBeVisible();
      const tabs = page.getByRole('navigation', { name: `${record.name} walkthrough steps` });
      await expect(tabs.getByRole('button')).toHaveCount(record.steps.length);

      for (const [index, step] of record.steps.entries()) {
        if (index > 0) await tabs.getByRole('button', { name: step.title, exact: true }).click();
        const where = `${record.name} › ${step.title}`;
        const section = page.locator(`section[data-demo-step="${step.id}"]`);
        await expect(section.getByRole('heading', { level: 2, name: step.title, exact: true }), where).toBeVisible();
        if (step.facts?.length) {
          await expect(section.locator(':scope > dl.blca-demo-facts > div'), `facts on ${where}`).toHaveCount(step.facts.length);
        }
        if (step.table) {
          await expect(section.locator('.blca-demo-table-section tbody tr'), `table rows on ${where}`).toHaveCount(step.table.rows.length);
        }
        if (step.chart) {
          const chart = section.locator('.blca-demo-chart svg[role="img"]');
          await expect(chart, `chart on ${where}`).toBeVisible();
          expect(await chart.locator('g path, g circle').count(), `curves drawn on ${where}`).toBeGreaterThan(0);
        }
        if (step.runs) {
          const runs = section.locator('.experiment-runs-table tbody tr');
          await expect(runs.first(), `training runs on ${where}`).toBeVisible();
          await expect(section.locator('.blca-demo-run-status')).toContainText('completed');
        }
        await expectNothingOffersAChange(page, where);
        visited += 1;
      }
      await page.getByRole('button', { name: '← Back to records' }).click();
      await expect(page.locator('[data-demo-record]')).toHaveCount(moduleRecords.length);
    }
  }

  expect(visited, 'steps visited').toBe(records.reduce((total, record) => total + record.steps.length, 0));
  expect(changes, 'changes the page sent').toEqual([]);
  expect(failedReads, 'API reads that failed').toEqual([]);
  expect(errors, 'errors in the page').toEqual([]);
});
