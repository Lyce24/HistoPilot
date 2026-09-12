import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fixturePredictor } from '../testFixtures/predictors';
import BulkEvaluationRunner from './BulkEvaluationRunner';

describe('evaluation source selection', () => {
  it('scopes the ready predictor table to a linked experiment and preserves configuration and seed identity', () => {
    const client = new QueryClient();
    client.setQueryData(['evaluation-batches', 'p'], { items: [] });
    const items = [fixturePredictor(1, 11, 'ensemble', 'one'), fixturePredictor(1, 11, 'refit', 'one'), fixturePredictor(2, 22, 'ensemble', 'two')];
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><BulkEvaluationRunner project="p" predictors={items} cohorts={[]} linkedExperiment="one" onOpenEvaluation={() => {}} /></QueryClientProvider>);
      expect(html).toContain('All shown predictors (2)');
      expect(html).toContain('Configuration candidate-1');
      expect(html).toContain('Fold ensemble'); expect(html).toContain('Refit');
      expect(html).toContain('5 fold models'); expect(html).toContain('1 refit model');
      expect(html).not.toContain('Configuration candidate-2');
      expect(html).toContain('#experiments?experiment=one&amp;tab=predictors');
      expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review all shown predictors/);
    } finally { client.clear(); }
  });

  it('never replaces a source with no ready predictors by another experiment’s weights', () => {
    const client = new QueryClient();
    client.setQueryData(['evaluation-batches', 'p'], { items: [] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><BulkEvaluationRunner project="p" predictors={[fixturePredictor(1, 11, 'ensemble', 'other')]} cohorts={[]} linkedExperiment="skip-policy" onOpenEvaluation={() => {}} /></QueryClientProvider>);
      expect(html).toContain('All shown predictors (0)');
      expect(html).toContain('Experiments submitted with Skip produce no predictors');
      expect(html).toContain('use one as a template');
      expect(html).not.toContain('Configuration candidate-1');
      expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review all shown predictors/);
    } finally { client.clear(); }
  });
});
