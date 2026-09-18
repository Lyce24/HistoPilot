import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fixturePredictor } from '../testFixtures/predictors';
import { fixtureExperiment } from '../testFixtures/evaluations';
import BulkEvaluationRunner from './BulkEvaluationRunner';
import type { EvaluationCohort } from '../api/evaluation';
import { initialEvaluationInputs } from './EvaluationInputSettings';

function render(ids?: string[], linkedExperiment = '', cohorts: EvaluationCohort[] = []) {
  const client = new QueryClient();
  client.setQueryData(['evaluation-batches', 'p'], { items: [] });
  client.setQueryData(['feature-bundles', 'p'], { items: [] });
  const items = [fixturePredictor(1, 11, 'ensemble', 'one'), fixturePredictor(1, 11, 'refit', 'one'), fixturePredictor(2, 22, 'ensemble', 'two')];
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}><BulkEvaluationRunner project="p" predictors={items} experiments={[fixtureExperiment('one'), fixtureExperiment('two'), fixtureExperiment('pending', 'running'), fixtureExperiment('skip-policy')]} cohorts={cohorts} linkedCohort={cohorts[0]?.id} linkedExperiment={linkedExperiment} experimentIds={ids} onOpenEvaluation={() => {}} /></QueryClientProvider>);
  } finally { client.clear(); }
}

describe('evaluation source selection', () => {
  it('starts by selecting experiments and defaults to no project-wide predictor selection', () => {
    const html = render();
    expect(html).toContain('1. Select experiments'); expect(html).toContain('0 predictors to evaluate from 0 selected experiments');
    expect(html).not.toContain('Both available methods');
    expect(html).not.toContain('Test cohort for selected experiments');
    expect(html).not.toContain('Review experiment evaluation');
    expect(html).not.toContain('Configuration candidate-1');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*><span>Continue to evaluation inputs/);
  });
  it('scopes a linked experiment and supports several selected experiments', () => {
    const linked = render(undefined, 'one');
    expect(linked).toContain('2 predictors to evaluate from 1 selected experiment');
    expect(linked).toMatch(/aria-label="Evaluate experiment one \(one\)"[^>]*checked=""/);
    expect(linked).not.toContain('Configuration candidate-1');
    expect(linked).not.toContain('Configuration candidate-2');
    expect(linked).toContain('1 ensemble / 1 refit ready');
    expect(linked).not.toMatch(/<button[^>]*disabled=""[^>]*><span>Continue to evaluation inputs/);
    const multiple = render(['one', 'two']);
    expect(multiple).toContain('3 predictors to evaluate from 2 selected experiments');
    expect(multiple).toMatch(/aria-label="Evaluate experiment two \(two\)"[^>]*checked=""/);
  });
  it('shows pending and skipped sources without substituting another experiment’s weights', () => {
    const html = render(undefined, 'skip-policy');
    expect(html).toContain('Running · No ready predictors');
    expect(html).toContain('Finished · No ready predictors');
    expect(html).toContain('0 predictors to evaluate from 1 selected experiment');
    expect(html).toMatch(/aria-label="Evaluate experiment skip-policy \(skip-policy\)"[^>]*checked=""/);
    expect(html).not.toContain('Configuration candidate-1');
    expect(html).not.toContain('Review experiment evaluation');
  });

  it('keeps linked cohort inputs on the next page while preserving experiment selection', () => {
    const cohort = { id: 'standalone', current: false, findings: [{ severity: 'error', code: 'OLD_FEATURE_CHECK', message: 'Old feature check failed.' }], manifest: { kind: 'evaluation-cohort', datasetId: 'test', spec: { datasetId: 'test', target: null, eligibility: [], patientIdentifiers: 'shared', inference: initialEvaluationInputs().inference }, target: null, summary: { includedSlides: 6 } } } as unknown as EvaluationCohort;
    const html = render(['one'], '', [cohort]);
    expect(html).toContain('2 predictors to evaluate from 1 selected experiment');
    expect(html).not.toMatch(/<option[^>]*value="standalone"[^>]*disabled/);
    expect(html).not.toContain('Test features and inference');
    expect(html).not.toContain('Test cohort for selected experiments');
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*><span>Continue to evaluation inputs/);
  });
});
