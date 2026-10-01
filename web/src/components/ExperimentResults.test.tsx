import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ExperimentResults from './ExperimentResults';
import type { ExperimentResults as Results } from '../api/experimentResults';
import type { ModelExperiment } from '../api/experiments';
import { batch, comparisonResults, configuration, experimentResults } from '../testFixtures/experimentResults';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
const record = { id: 'exp', key: 'draft:exp', name: 'study4', notes: '', tags: [], revision: 1, state: 'active', status: 'completed', stage: 'finished', legacy: false,
  createdAt: '', updatedAt: '', inputs: null, batches: [], drafts: [], predictorId: null, submission: { status: 'submitted' } } as unknown as ModelExperiment;

function render(results?: Results, changes: Partial<ModelExperiment> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  if (results) client.setQueryData(['experiment-results', 'project', 'exp'], results);
  return renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentResults project="project" record={{ ...record, ...changes }} /></QueryClientProvider>);
}
const text = (html: string) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

describe('experiment results', () => {
  it('leads with the design, the notes and a plain-language takeaway', () => {
    const html = text(render(experimentResults()));
    expect(html).toContain('2-fold cross-validation');
    expect(html).toContain('2 training seeds');
    expect(html).toContain('3 classes (HG, LG, ND)');
    expect(html).toContain('ABMIL baseline: LG recall is 0.250');
    expect(html).toContain('nnMIL has the highest macro AUROC (0.950 ± 0.005)');
    expect(html).toContain('Against ABMIL baseline the difference is +0.040 (95% interval +0.020 to +0.060): a clear difference , better in 2 of 2 shared folds.');
    expect(html).not.toContain('Partial results');
  });

  it('compares batches on seed means with intervals, and pairs them leader first', () => {
    const html = text(render(experimentResults()));
    expect(html).toContain('Model comparison');
    expect(html).toContain('0.910 ± 0.014 95% CI 0.880–0.940');
    expect(html).toMatch(/nnMIL vs ABMIL baseline/);
    expect(html).toContain('+0.040 +0.020 to +0.060 Higher (better) better in 2 of 2 folds');
    expect(html).toContain('No clear difference');
    // Only the strictly best batch in a column is marked.
    expect(render(experimentResults()).match(/highest in this column/g)).toHaveLength(1);
  });

  it('shows every seed, the seed mean, its interval and the seed ensemble', () => {
    const html = text(render(experimentResults()));
    expect(html).toContain('Training seeds (OOF)');
    expect(html).toContain('Seed 42 0.900');
    expect(html).toContain('Seed 43 0.920');
    expect(html).toContain('Mean ± SD 2 seeds 0.910 ± 0.014');
    expect(html).toContain('95% interval 2,000 slide resamples 0.880–0.940');
    expect(html).toContain('Seed ensemble seeds’ mean prediction 0.925');
  });

  it('lays test folds out by fold and seed with checkpoint epochs and footers', () => {
    const html = text(render(experimentResults()));
    expect(html).toContain('Test folds · Macro AUROC');
    expect(html).toContain('Fold 1 20 slides 0.930 epoch 12 of 17 0.950 epoch 4 of 9 0.940 ± 0.014');
    expect(html).toContain('Fold mean ± SD');
    expect(html).toContain('Pooled OOF 0.900 0.920 0.910 ± 0.014');
    expect(html).toContain('Checkpoint epoch median (range) 7 (2–12) 5 (4–6) 5 (2–12)');
    expect(render(experimentResults())).toMatch(/class="exp-heat" style="background:#[0-9a-f]{6}"/);
  });

  it('describes a held-out assessment as one run per seed and shows no fold statistics over one row', () => {
    const base = configuration();
    const single = { ...base, splitSeeds: base.splitSeeds.map((split) => ({ ...split, folds: split.folds.slice(0, 1), seeds: split.seeds.map((seed) => ({ ...seed, folds: seed.folds.slice(0, 1) })) })) };
    const results = experimentResults({ design: { strategy: 'held_out', splitUnit: 'slide', groupByPatient: false, folds: 1, splitSeeds: [7], slideCount: 40, resamplingUnit: 'slide' }, batches: [batch('abmil', 'ABMIL baseline', {}, single)], comparisons: [], findings: [] });
    const html = render(results);
    const plain = text(html);
    expect(html).toContain('<th scope="col">Held-out assessment</th>');
    expect(plain).toContain('Held-out assessment · Macro AUROC');
    expect(plain).toContain('only the held-out set is assessed');
    expect(plain).toContain('Every seed assesses the same slides');
    expect(plain).not.toContain('Test fold');
    expect(plain).not.toContain('Fold mean ± SD');
    // A fold design keeps its fold wording and the spread over folds.
    expect(text(render(experimentResults()))).toContain('One run’s held-out fold (one of 2 folds)');
  });

  it('flags weak classes and prints the confusion shares', () => {
    const html = render(experimentResults());
    expect(text(html)).toMatch(/LG 8 slides 0\.250 ± 0\.050\s+Low/);
    expect(text(html)).toContain('90% 9.0');
    expect(html).toContain('is-diagonal');
  });

  it('shows a single batch without comparison or pairing', () => {
    const single = experimentResults({ batches: [batch('batch-abmil', 'ABMIL baseline')], comparisons: [] });
    const html = text(render(single));
    expect(html).toContain('Headline results');
    expect(html).toContain('ABMIL baseline : macro AUROC 0.910 ± 0.014 across 2 training seeds (95% interval 0.880–0.940); its test folds ranged 0.890–0.950.');
    expect(html).not.toContain('Paired differences');
    expect(html).not.toContain('Batch shown in detail');
  });

  it('marks incomplete seeds as partial and never averages them in', () => {
    const partial = configuration({ complete: false, seedCount: 1, plannedSeedCount: 2 });
    partial.splitSeeds[0].seeds[1] = { ...partial.splitSeeds[0].seeds[1], oof: null, complete: false, completedRuns: 1 };
    const html = text(render(experimentResults({ batches: [batch('batch-abmil', 'ABMIL baseline', {}, partial)], comparisons: [] }), { stage: 'running', status: 'running' }));
    expect(html).toContain('Partial results.');
    expect(html).toContain('1 of 2 test folds finished; OOF waits for all of them');
  });

  it('ranks configurations by validation and never by OOF', () => {
    const grid = batch('grid', 'LR grid', {
      selection: { source: 'validation', metric: 'validation_auroc', ready: true, scores: {} }, selectedCandidateId: 'second',
      configurations: [configuration({ candidateId: 'first', number: 1, selected: false, validationScore: 0.8 }), configuration({ candidateId: 'second', number: 2, selected: true, validationScore: 0.9 })],
    });
    const html = text(render(experimentResults({ batches: [grid], comparisons: [] })));
    expect(html).toContain('Configurations, ranked by validation');
    expect(html.indexOf('2 Selected')).toBeLessThan(html.indexOf(' 1 abmil'));
    expect(html).toContain('configuration 2 of 2, chosen on validation');
  });

  it('reports a controlled comparison: every arm, and reference minus each arm with paired intervals and Holm p', () => {
    const html = text(render(comparisonResults()));
    expect(html).toContain('Controlled comparison');
    expect(html).toContain('Reference: configuration 1 (ABMIL · Image only). Primary metric: AUROC.');
    // Arms, reference marked, with seed mean ± SD and the interval of that mean.
    expect(html).toContain('Configuration 1 Reference ABMIL Image only 0.910 ± 0.010 0.880–0.940 2 of 2 seeds');
    expect(html).toContain('Configuration 2 nnMIL Image only 0.870 ± 0.010 0.840–0.900');
    expect(html).toContain('Configuration 3 — Clinical only 0.900 ± 0.010');
    expect(html).toContain('Configuration 4 Mean pooling MIL Image only — — 0 of 2 seeds 1 of 4 test folds');
    // Contrasts: reference − arm, the paired interval and a verdict, p, Holm p, and fold wins.
    expect(html).toContain('Configuration 2 nnMIL · Image only +0.040 0.910 vs 0.870 +0.010 to +0.070 Reference better &lt; 0.001 &lt; 0.001 better in 2 of 2 folds');
    expect(html).toContain('Configuration 3 Clinical only +0.010 0.910 vs 0.900 −0.020 to +0.040 No clear difference 0.250 0.500 better in 1 of 2 folds');
    // An arm without complete results says why instead of showing numbers.
    expect(html).toContain('Configuration 4 Mean pooling MIL · Image only The arm and the reference both need complete out-of-fold results.');
    expect(html).toContain('Some arms have unfinished training seeds.');
    expect(html).toContain('Differences use the same resampled slides for every arm; Holm adjusts p for the number of planned contrasts.');
  });

  it('presents a comparison batch as its reference arm, not a missing validation choice', () => {
    const html = text(render(comparisonResults()));
    expect(html).toContain('configuration 1 of 4, the comparison’s reference');
    expect(html).toContain('or the reference of a controlled comparison');
    expect(html).toContain('Comparison arms');
    expect(html).not.toContain('Configurations, ranked by validation');
    expect(html).not.toContain('no validation-based configuration choice');
    // Batches without a declared comparison keep the batch-vs-batch view only.
    expect(text(render(experimentResults()))).not.toContain('Controlled comparison');
  });

  it('explains an empty experiment and a loading one', () => {
    const empty = experimentResults({ batches: [batch('b', 'Batch', {}, { seedCount: 0, foldCount: 0 })], comparisons: [], findings: [] });
    expect(text(render(empty, { stage: 'running' }))).toContain('Test fold results appear as each fold finishes');
    expect(text(render(undefined))).toContain('Loading results…');
  });
});
