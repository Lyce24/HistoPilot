import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { fixturePredictor } from '../testFixtures/predictors';
import { fixtureEvaluation } from '../testFixtures/evaluations';
import EvaluationResultsTable from './EvaluationResultsTable';

describe('evaluation result comparison', () => {
  it('presents matched method means and separates patient and slide scoring without an automatic winner', () => {
    const ensemble = fixturePredictor(1, 11, 'ensemble'), refit = fixturePredictor(1, 11, 'refit');
    const records = [fixtureEvaluation(ensemble, 0.7, 0.6), fixtureEvaluation(refit, 0.8, 0.7), fixtureEvaluation(ensemble, 0.9, 0.9, 'cohort', 'slide')];
    const html = renderToStaticMarkup(<EvaluationResultsTable records={records} predictors={[ensemble, refit]} cohorts={[]} loading={false} onOpen={() => {}} />);
    expect(html).toContain('Ensemble vs refit'); expect(html).toContain('Mean AUROC (1 matched pair)');
    expect(html).toContain('patient scoring'); expect(html).toContain('slide scoring');
    expect(html).toContain('0.700'); expect(html).toContain('0.800');
    expect(html).toContain('Unmatched results are excluded from method means');
    expect(html).toContain('confirm it on independent data');
    expect(html).not.toContain('Highest AUROC'); expect(html).not.toContain('winner');
  });
});
