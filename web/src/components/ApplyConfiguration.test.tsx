import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { FrozenPredictor, SeedEnsembleChoice } from '../api/predictors';
import { fixturePredictor, fixtureSeedEnsemble } from '../testFixtures/predictors';
import ApplyConfiguration from './ApplyConfiguration';

const choice = (changes: Partial<SeedEnsembleChoice> = {}): SeedEnsembleChoice => ({ experimentId: 'study', experimentName: 'Study', batchId: 'batch', batchName: 'Baseline', candidateId: 'candidate-1', candidateNumber: 1,
  trainingSeeds: [11, 22, 33], splitSeeds: [42], seedGroups: 3, members: 15, completedRuns: 15, eligible: true, reason: null, existingPredictorId: null, ...changes });
const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function render(choices: SeedEnsembleChoice[], predictors: FrozenPredictor[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['seed-ensembles', 'p', 'study'], { items: choices });
  client.setQueryData(['predictors', 'p'], { items: predictors });
  return renderToStaticMarkup(<QueryClientProvider client={client}><ApplyConfiguration project="p" record={{ id: 'study', name: 'Study' }} batchId="batch" candidateId="candidate-1" number={1} /></QueryClientProvider>);
}

describe('applying the configuration shown in Results', () => {
  it('applies a built seed ensemble, the form of the seed-ensemble row', () => {
    const built = fixtureSeedEnsemble(1);
    const html = render([choice({ eligible: false, existingPredictorId: built.id })], [built]);
    expect(html).toContain('Its seed ensemble is built: 3 training × 1 split seed · 15 fold models.');
    expect(html).toContain(`href="#apply?view=new&amp;experiment=study&amp;predictor=${built.id}">Apply this configuration`);
  });

  it('builds a missing seed ensemble from verified checkpoints before applying it', () => {
    const html = render([choice()], [fixturePredictor(1, 11, 'ensemble'), fixturePredictor(1, 22, 'ensemble')]);
    expect(html).toContain('nothing trains');
    expect(html).toMatch(/<button[^>]*>Build seed ensemble and apply<\/button>/);
  });

  it('applies the fold ensemble of a single-seed configuration', () => {
    const single = fixturePredictor(1, 11, 'ensemble');
    const html = render([], [single, fixturePredictor(2, 11, 'ensemble')]);
    expect(html).toContain('Its fold ensemble is ready.');
    expect(html).toContain(`predictor=${single.id}">Apply this configuration`);
  });

  it('says why nothing can be applied yet, without a dead action', () => {
    const waiting = render([choice({ eligible: false, reason: 'Seed 33 has 2 of 5 folds finished.' })], []);
    expect(waiting).toContain('Seed 33 has 2 of 5 folds finished.');
    expect(waiting).not.toContain('href="#apply');
    expect(waiting).not.toContain('<button');
    expect(render([], [])).toContain('No predictor is ready for this configuration yet.');
  });
});
