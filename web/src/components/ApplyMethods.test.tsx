import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ModelEvaluation } from '../api/predictors';
import { fixtureEvaluation } from '../testFixtures/evaluations';
import { fixturePredictor } from '../testFixtures/predictors';
import ApplyMethods from './ApplyMethods';

function render(records: ModelEvaluation[], predictors = [fixturePredictor(1, 11, 'ensemble'), fixturePredictor(1, 11, 'refit')]) {
  const client = new QueryClient();
  try { return renderToStaticMarkup(<QueryClientProvider client={client}><ApplyMethods project="p" runs={records} predictors={predictors} experiments={[]} cohorts={[]} loading={false} onOpen={() => {}} /></QueryClientProvider>); }
  finally { client.clear(); }
}

describe('method comparison across scored runs', () => {
  it('presents matched method means and separates patient and slide scoring without an automatic winner', () => {
    const ensemble = fixturePredictor(1, 11, 'ensemble'), refit = fixturePredictor(1, 11, 'refit');
    const html = render([fixtureEvaluation(ensemble, 0.7, 0.6), fixtureEvaluation(refit, 0.8, 0.7), fixtureEvaluation(ensemble, 0.9, 0.9, 'cohort', 'slide')], [ensemble, refit]);
    expect(html).toContain('Ensemble vs refit'); expect(html).toContain('Mean AUROC (1 matched pair)');
    expect(html).toContain('patient scoring'); expect(html).toContain('slide scoring');
    expect(html).toContain('0.700'); expect(html).toContain('0.800');
    expect(html).toContain('Unmatched results are excluded from method means');
    expect(html).toContain('Freeze the strategy using development evidence before external evaluation');
    expect(html).not.toContain('Highest AUROC'); expect(html).not.toContain('winner');
    expect(html).toContain('Labeled cohort filter');
    expect(html).toContain('Runs on unlabeled cohorts are not compared here, including runs scored against a reference standard.');
  });

  it('leaves runs on unlabeled cohorts out: they have nothing to compare', () => {
    const ensemble = fixturePredictor(1, 11, 'ensemble'), refit = fixturePredictor(1, 11, 'refit');
    const unlabeled = (record: ModelEvaluation): ModelEvaluation => ({ ...record, manifest: { ...record.manifest, purpose: 'inference' } });
    const html = render([unlabeled(fixtureEvaluation(ensemble, 0.7, 0.6)), unlabeled(fixtureEvaluation(refit, 0.8, 0.7))], [ensemble, refit]);
    expect(html).toContain('Completed scored runs will appear here.');
    expect(html).not.toContain('Mean AUROC');
  });
});
