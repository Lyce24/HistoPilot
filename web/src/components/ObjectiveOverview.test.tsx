import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Interpretation } from '../api/interpretation';
import { fixtureEvaluation } from '../testFixtures/evaluations';
import { fixturePredictor } from '../testFixtures/predictors';
import ObjectiveOverview from './ObjectiveOverview';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
const study = (id: string, experimentId: string, minute: number) => ({ id, createdAt: new Date(Date.UTC(2026, 8, 30, 10, minute)).toISOString(), contentHash: id, lifecycleState: 'active',
  manifest: { kind: 'model-interpretation', name: id, predictorId: 'model', experimentId, encoderId: 'uni', method: 'refit', memberCount: 1, featureBundleId: 'bundle', slideFolder: '/slides', slides: [{}] } }) as unknown as Interpretation;

function render(filings: Record<string, unknown>, objectives = [{ id: 'aim-1', name: 'Aim 1 · Grade', rank: 'primary', description: 'Four-grade dysplasia.', revision: 1, createdAt: '', updatedAt: '' }]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }); clients.push(client);
  const predictors = [fixturePredictor(1, 11, 'refit', 'main'), fixturePredictor(1, 11, 'ensemble', 'main'), fixturePredictor(1, 11, 'refit', 'pilot'), { ...fixturePredictor(2, 11, 'refit', 'main'), lifecycleState: 'archived' as const }];
  client.setQueryData(['objectives', 'p'], { revision: 1, items: objectives, filings });
  client.setQueryData(['predictors', 'p'], { items: predictors });
  client.setQueryData(['model-evaluations', 'p'], { items: [fixtureEvaluation(predictors[0], 0.8, 0.7), fixtureEvaluation(predictors[2], 0.7, 0.6)] });
  client.setQueryData(['interpretations', 'p'], { items: [study('s1', 'main', 0), study('s2', 'main', 5), study('s3', 'main', 90)] });
  return renderToStaticMarkup(<QueryClientProvider client={client}><ObjectiveOverview project="p" /></QueryClientProvider>);
}

describe('objectives on the roadmap', () => {
  it('counts each objective\'s experiments, ready predictors, runs and studies', () => {
    const html = render({ main: { objectiveId: 'aim-1', role: 'primary', series: null, seriesMember: null }, clinical: { objectiveId: 'aim-1', role: 'comparison', series: null, seriesMember: null } });
    expect(html).toContain('Aim 1 · Grade'); expect(html).toContain('Primary objective'); expect(html).toContain('Four-grade dysplasia.');
    expect(html).toContain('2 experiments<small>1 primary · 1 comparison</small>');
    expect(html).toContain('2 ready predictors'); expect(html).toContain('<dd>1 run</dd>'); expect(html).toContain('2 studies');
    // The pilot's predictor and run come from an experiment under no objective.
    expect(html).toContain('1 ready predictor · 1 run come from experiments under no objective.');
    expect(html).toContain('File them in Experiments');
  });

  it('stays off the roadmap until the project has objectives', () => {
    expect(render({}, [])).toBe('');
  });
});
