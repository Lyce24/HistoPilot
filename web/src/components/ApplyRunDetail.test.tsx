import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { EvaluationCohort } from '../api/evaluation';
import type { EvaluationMetrics, ModelEvaluation } from '../api/predictors';
import type { ReferenceStandard } from '../api/references';
import type { ProtocolSpec } from '../api/scientific';
import type { RunTab } from '../lib/applyRoutes';
import { fixtureEvaluation } from '../testFixtures/evaluations';
import { fixturePredictor, fixtureSeedEnsemble } from '../testFixtures/predictors';
import ApplyRunDetail, { labelSources, runTabs } from './ApplyRunDetail';

const source = fixtureSeedEnsemble(1);
const cohort = (id: string, purpose?: 'inference') => ({ id, current: true, versionLabel: { tag: `Cohort ${id}` }, manifest: { spec: { purpose }, summary: { includedSlides: 76, labeledSlides: purpose ? 0 : 76 } } }) as unknown as EvaluationCohort;
const target: ProtocolSpec['target'] = { field: 'grade', task: 'binary_classification', unit: 'slide', classes: ['a', 'b'], labels: {}, positiveClass: 'b', missing: 'exclude', unmapped: 'exclude' };
const labeled = fixtureEvaluation(source, 0.81, 0.74, 'grade-2', 'slide', 'scored');
labeled.manifest.name = 'Evaluation · Seed ensemble';
labeled.manifest.target = target;
labeled.execution!.result!.metrics = { ...labeled.execution!.result!.metrics!, selected: { ...labeled.execution!.result!.metrics!.selected, confusionMatrix: [[40, 14], [5, 17]] } };
const unlabeled: ModelEvaluation = { ...fixtureEvaluation(source, null, null, 'external', 'slide', 'predicted'), manifest: { ...labeled.manifest, cohortId: 'external', name: 'Inference · Seed ensemble', purpose: 'inference' } };
unlabeled.execution = { status: 'completed', result: { summary: { unit: 'slide' } } } as ModelEvaluation['execution'];
const standard = (id: string, name: string, cohortId: string, classes = ['b', 'a'], lifecycleState: ReferenceStandard['lifecycleState'] = 'active') => ({
  id, createdAt: '', contentHash: id, lifecycleState,
  manifest: { kind: 'reference-standard', name, cohortId, cohort: { id: cohortId, contentHash: cohortId }, datasetId: 'd2', datasets: [{ id: 'd2', contentHash: 'd2' }], field: 'reader', matchedBy: 'slideId', classes, labels: {}, summary: { slides: 76, labeledSlides: 70, missingSlides: 6, unmappedSlides: 0, unmatchedSlides: 0, classCounts: {}, conflictingPatients: 0, values: [], valuesTruncated: false }, findings: [] },
}) as ReferenceStandard;

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function render(record: ModelEvaluation, tab?: RunTab, options: { standards?: ReferenceStandard[]; reference?: string; scores?: EvaluationMetrics } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['clinical-analyses', 'p', 'all'], { items: [] });
  client.setQueryData(['reference-standards', 'p', record.manifest.cohortId], { items: options.standards ?? [] });
  if (options.scores) client.setQueryData(['reference-scores', 'p', record.id, options.reference ?? options.standards?.[0]?.id], options.scores);
  // Performance reads only the frozen data dictionary from the prediction summary it shares.
  if (record === labeled || options.standards?.length) client.setQueryData(['inference-summary', 'p', record.id, 'selected', ''], { attributes: [{ key: 'site', label: 'Site' }] });
  return renderToStaticMarkup(<QueryClientProvider client={client}><ApplyRunDetail project="p" record={record} runs={[labeled, unlabeled]}
    predictors={[source, fixturePredictor(1, 11, 'ensemble')]} experiments={[]} cohorts={[cohort('grade-2'), cohort('external', 'inference')]} tab={tab} onTabChange={() => {}}
    reference={options.reference} onReferenceChange={() => {}} /></QueryClientProvider>);
}

describe('one run in Apply models', () => {
  it('opens on what the run is for: performance and agreement when it has labels, predictions otherwise', () => {
    expect(runTabs(true)).toEqual(['performance', 'agreement', 'predictions', 'cases', 'compare']);
    expect(runTabs(false)).toEqual(['predictions', 'cases', 'compare']);
  });

  it('counts the cohort’s fitting reference standards in the run context', () => {
    const html = render(unlabeled, 'performance', { standards: [standard('r1', 'Consensus', 'external'), standard('r2', 'Reader B', 'external'), standard('other', 'Other classes', 'external', ['a', 'c'])] });
    expect(html).toContain('Unlabeled · 2 reference standards · 76 slides');
    expect(html).toContain('Scoring the saved predictions against Consensus…');
  });

  it('scores against the cohort’s labels first, then each fitting reference standard by name', () => {
    const standards = [standard('r2', 'Reader B', 'grade-2'), standard('r1', 'Consensus', 'grade-2'), standard('other', 'Other cohort', 'external'),
      standard('classes', 'Other classes', 'grade-2', ['a', 'c']), standard('trash', 'Trashed', 'grade-2', ['a', 'b'], 'trashed'), standard('old', 'Archived', 'grade-2', ['a', 'b'], 'archived')];
    expect(labelSources(labeled, standards).map((item) => item.name)).toEqual(['Cohort labels', 'Consensus', 'Reader B']);
    expect(labelSources(labeled, standards, 'old').map((item) => item.id)).toEqual([null, 'old', 'r1', 'r2']);
    // An unlabeled cohort has no labels of its own.
    expect(labelSources(unlabeled, standards).map((item) => item.name)).toEqual(['Other cohort']);
  });

  it('scores a labeled run, with subgroups and clinical utility on its performance', () => {
    const html = render(labeled);
    expect(html).toContain('Cohort grade-2<small>Labeled · scored · 76 slides</small>');
    expect(html).toContain('Seed ensemble · 3 training × 1 split seed · 15 models');
    expect(html).toMatch(/id="run-tab-performance" aria-current="page"/);
    for (const tab of ['predictions', 'cases', 'compare']) expect(html).toContain(`id="run-tab-${tab}"`);
    expect(html).toContain('AUROC');
    expect(html).toContain('aria-label="Review 14 cases: actual a, predicted b"');
    expect(html).toContain('Review errors');
    expect(html).toContain('Performance by subgroup');
    expect(html).toContain('<option value="site">Site</option>');
    expect(html).toContain('Clinical utility');
    expect(html).toContain('Scored slide table (CSV)');
  });

  it('keeps an unlabeled run to predictions until a reference standard labels its cohort', () => {
    const html = render(unlabeled, 'performance');
    expect(html).toContain('Cohort external<small>Unlabeled · predictions only · 76 slides</small>');
    expect(html).toContain('No labels · predictions only');
    expect(html).toContain('Add reference standard');
    expect(html).not.toContain('id="run-tab-performance"');
    expect(html).not.toContain('id="run-tab-agreement"');
    expect(html).toMatch(/id="run-tab-predictions" aria-current="page"/);
    expect(html).toContain('Predictions only: no labels are read and no performance metrics are computed.');
    expect(html).toContain('Verifying saved predictions and summarizing…');
    expect(html).not.toContain('Performance by subgroup');
    expect(html).not.toContain('Clinical utility');
  });

  it('scores an unlabeled run against a reference standard added to its cohort later', () => {
    const inference = { ...unlabeled, manifest: { ...unlabeled.manifest, target } };
    const standards = [standard('r2', 'Reader B', 'external'), standard('r1', 'Consensus', 'external')];
    const pending = render(inference, undefined, { standards });
    expect(pending).toMatch(/<option value="r1" selected="">Consensus<\/option>/);
    expect(pending).toContain('Reference standard · 70 of 76 slides labeled');
    expect(pending).toMatch(/id="run-tab-performance" aria-current="page"/);
    expect(pending).toContain('id="run-tab-agreement"');
    expect(pending).toContain('Scoring the saved predictions against Consensus…');
    const scores = { ...labeled.execution!.result!.metrics!, reference: { id: 'r2', name: 'Reader B' }, conflictingPatients: 2 } as EvaluationMetrics;
    const scored = render(inference, 'performance', { standards, reference: 'r2', scores });
    expect(scored).toMatch(/<option value="r2" selected="">Reader B<\/option>/);
    expect(scored).toContain('Scored against the reference standard Reader B');
    expect(scored).toContain('2 patients have slides this reference labels differently');
    expect(scored).toContain('Performance by subgroup');
    expect(scored).toContain('with outcomes from Reader B');
    expect(scored).toContain('Scored slide table (CSV)');
    expect(render(inference, 'performance', { standards, reference: 'missing' })).toContain('The linked reference standard does not label this run');
  });

  it('opens the tab a link names, and keeps a trashed run’s results out of reach', () => {
    expect(render(labeled, 'compare')).toMatch(/id="run-tab-compare" aria-current="page"/);
    const trashed = render({ ...labeled, lifecycleState: 'trashed' });
    expect(trashed).toContain('This run is in Trash');
    expect(trashed).not.toContain('id="run-tab-performance"');
  });
});
