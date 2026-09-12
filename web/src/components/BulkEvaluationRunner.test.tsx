import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fixturePredictor } from '../testFixtures/predictors';
import { fixtureExperiment } from '../testFixtures/evaluations';
import BulkEvaluationRunner from './BulkEvaluationRunner';

function render(ids?: string[], linkedExperiment = '') {
  const client = new QueryClient();
  client.setQueryData(['evaluation-batches', 'p'], { items: [] });
  const items = [fixturePredictor(1, 11, 'ensemble', 'one'), fixturePredictor(1, 11, 'refit', 'one'), fixturePredictor(2, 22, 'ensemble', 'two')];
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}><BulkEvaluationRunner project="p" predictors={items} experiments={[fixtureExperiment('one'), fixtureExperiment('two'), fixtureExperiment('pending', 'running'), fixtureExperiment('skip-policy')]} cohorts={[]} linkedExperiment={linkedExperiment} experimentIds={ids} onOpenEvaluation={() => {}} /></QueryClientProvider>);
  } finally { client.clear(); }
}

describe('evaluation source selection', () => {
  it('starts by selecting experiments and defaults to no project-wide predictor selection', () => {
    const html = render();
    expect(html).toContain('1. Select experiments'); expect(html).toContain('0 predictors to evaluate from 0 selected experiments');
    expect(html).toContain('Both available methods');
    expect(html).not.toContain('Configuration candidate-1');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review experiment evaluation/);
  });
  it('scopes a linked experiment and supports several selected experiments', () => {
    const linked = render(undefined, 'one');
    expect(linked).toContain('2 predictors to evaluate from 1 selected experiment');
    expect(linked).toContain('Configuration candidate-1');
    expect(linked).not.toContain('Configuration candidate-2');
    expect(linked).toContain('5 fold models'); expect(linked).toContain('1 refit model');
    const multiple = render(['one', 'two']);
    expect(multiple).toContain('3 predictors to evaluate from 2 selected experiments');
    expect(multiple).toContain('Configuration candidate-2');
  });
  it('shows pending and skipped sources without substituting another experiment’s weights', () => {
    const html = render(undefined, 'skip-policy');
    expect(html).toContain('Running · No ready predictors');
    expect(html).toContain('Finished · No ready predictors');
    expect(html).toContain('0 predictors to evaluate from 1 selected experiment');
    expect(html).toContain('Batches using Skip produce no predictors');
    expect(html).toContain('copy an experiment as a template');
    expect(html).not.toContain('Configuration candidate-1');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review experiment evaluation/);
  });
});
