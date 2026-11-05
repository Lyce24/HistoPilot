import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactElement } from 'react';
import type { ModelEvaluation } from '../api/predictors';
import type { ShiftReport, ShiftStatus } from '../api/shift';
import { ReadOnlyContext } from '../lib/readOnly';
import RunDatasetShift, { ShiftPlane } from './RunDatasetShift';

const record = { id: 'run', lifecycleState: 'active', createdAt: '', contentHash: 'run', manifest: { kind: 'model-evaluation', name: 'CPTAC run', cohortId: 'cohort', predictorId: 'p', experimentId: 'e', status: 'planned', target: { classes: ['LUAD', 'LUSC'] } }, execution: { status: 'completed' } } as unknown as ModelEvaluation;

const report: ShiftReport = {
  version: 1, method: {}, inputHash: 'x',
  summary: {
    developmentSlides: 946, cohortSlides: 451, dimensions: 1536,
    domainAuroc: { value: 0.97, lower: 0.96, upper: 0.98 },
    beyond: { count: 135, fraction: 0.3, lower: 0.26, upper: 0.34, threshold: 0.1 },
    distance: { development: { p05: 0, p25: 0, median: 0, p75: 0, p95: 0 }, cohort: { p05: 0, p25: 0, median: 0, p75: 0, p95: 0 } },
    verdict: 'strong', verdictText: 'The cohort differs clearly from the development data in feature space.',
  },
  slides: [
    { slideId: 'd1', set: 'development', distance: 0.05, percentile: 0.5, beyond: false, x: 0, y: 0 },
    { slideId: 'd2', set: 'development', distance: 0.06, percentile: 0.6, beyond: false, x: 1, y: 1 },
    { slideId: 'c1', set: 'cohort', distance: 0.2, percentile: 1, beyond: true, domainScore: 0.9, x: 3, y: 2 },
  ],
};

const status = (value: Partial<ShiftStatus>): ShiftStatus => ({ evaluationId: 'run', analysisId: 'a', exists: false, execution: null, findings: [], canRun: true, report: null, ...value });

function render(data: ShiftStatus, readOnly = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(['run-shift', 'p', 'run'], data);
  const element: ReactElement = <RunDatasetShift project="p" record={record} unit="selected" referenceId={null} />;
  return renderToStaticMarkup(<QueryClientProvider client={client}><ReadOnlyContext.Provider value={readOnly}>{element}</ReadOnlyContext.Provider></QueryClientProvider>);
}

describe('dataset shift of a run', () => {
  it('reports separability, the share beyond the development range and the plane', () => {
    const html = render(status({ exists: true, execution: { status: 'completed' } as ShiftStatus['execution'], report }));
    expect(html).toContain('The cohort differs clearly');
    expect(html).toContain('0.970 (0.960–0.980)');
    expect(html).toContain('30%');
    expect(html).toContain('135 of 451');
    expect(html).toContain('Shift report (JSON)');
    expect(html).not.toContain('Analyze dataset shift');
  });

  it('offers the analysis on a local project, and nothing to start in a public demo', () => {
    expect(render(status({}))).toContain('Analyze dataset shift');
    const demo = render(status({}), true);
    expect(demo).not.toContain('Analyze dataset shift');
    expect(demo).toContain('No dataset shift analysis for this run.');
  });

  it('draws development slides first and rings the cohort slides beyond the range', () => {
    const html = renderToStaticMarkup(<ShiftPlane slides={report.slides} label="plane" />);
    expect(html.indexOf('shift-point-development')).toBeLessThan(html.indexOf('shift-point-beyond'));
    expect(html.match(/<circle/g)).toHaveLength(3);
    expect(html).toContain('aria-label="plane"');
  });
});
