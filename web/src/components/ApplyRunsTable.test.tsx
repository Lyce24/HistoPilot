import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { EvaluationCohort } from '../api/evaluation';
import type { ModelEvaluation } from '../api/predictors';
import type { ReferenceStandard } from '../api/references';
import { fixtureEvaluation } from '../testFixtures/evaluations';
import { fixturePredictor, fixtureSeedEnsemble } from '../testFixtures/predictors';
import ApplyRunsTable, { newRunFilters } from './ApplyRunsTable';

const seeds = fixtureSeedEnsemble(1), fold = fixturePredictor(1, 11, 'ensemble');
const cohort = (id: string, purpose?: 'inference') => ({ id, current: true, versionLabel: { tag: `Cohort ${id}` }, manifest: { spec: { purpose }, summary: { includedSlides: 80, labeledSlides: purpose ? 0 : 76 } } }) as unknown as EvaluationCohort;
const scored = { ...fixtureEvaluation(seeds, 0.812, 0.74, 'grade-2', 'slide', 'scored'), createdAt: '2026-09-28T00:00:00Z' };
scored.manifest.name = 'Evaluation · Seed ensemble';
const predicted: ModelEvaluation = { id: 'predicted', createdAt: '2026-09-29T00:00:00Z', contentHash: 'p', lifecycleState: 'active',
  manifest: { kind: 'model-evaluation', predictorId: fold.id, experimentId: 'study', cohortId: 'external', name: 'Inference · Fold ensemble', status: 'planned', purpose: 'inference' },
  execution: { status: 'completed', result: { summary: { unit: 'slide', selected: { predicted: [{ label: 'ND', count: 60 }, { label: 'LG', count: 20 }], ensemble: { records: 80, unanimous: 60, disagreements: 20, memberCount: 5 } } } } } as unknown as ModelEvaluation['execution'] };
const render = (filters = newRunFilters()) => renderToStaticMarkup(<ApplyRunsTable runs={[scored, predicted, { ...scored, id: 'archived', lifecycleState: 'archived', manifest: { ...scored.manifest, name: 'Archived run' } }]}
  predictors={[seeds, fold]} experiments={[]} cohorts={[cohort('grade-2'), cohort('external', 'inference')]} loading={false} onOpen={() => {}} onCreate={() => {}} filters={filters} onFiltersChange={() => {}} />);

describe('runs in Apply models', () => {
  it('lists labeled and unlabeled runs together, grouped by cohort, each with what it found', () => {
    const html = render();
    expect(html).toContain('Cohort grade-2 · Labeled · 76 of 80 slides scored');
    expect(html).toContain('Cohort external · Unlabeled · predictions only');
    // A scored run shows its metrics; a run on unlabeled slides shows what it predicted.
    expect(html).toMatch(/AUROC 0\.812/);
    expect(html).toContain('ND 60 · LG 20');
    expect(html).toContain('75% of slides unanimous');
    expect(html).toContain('Seed ensemble · 3 training × 1 split seed · 15 models');
    expect(html).not.toContain('Archived run');
    expect(html.indexOf('Inference · Fold ensemble')).toBeLessThan(html.indexOf('Evaluation · Seed ensemble'));
  });

  it('says when an unlabeled cohort has reference standards that score its runs', () => {
    const standard = (id: string, cohortId: string, classes: string[], lifecycleState = 'active') => ({ id, lifecycleState, manifest: { kind: 'reference-standard', name: id, cohortId, classes } }) as unknown as ReferenceStandard;
    const withClasses = { ...predicted, manifest: { ...predicted.manifest, target: { classes: ['ND', 'LG'] } } } as ModelEvaluation;
    const references = [standard('r1', 'external', ['LG', 'ND']), standard('r2', 'external', ['ND', 'LG', 'HG']), standard('r3', 'external', ['ND', 'LG'], 'trashed'), standard('r4', 'grade-2', ['ND', 'LG'])];
    const html = renderToStaticMarkup(<ApplyRunsTable runs={[withClasses, scored]} predictors={[seeds, fold]} experiments={[]} cohorts={[cohort('grade-2'), cohort('external', 'inference')]} references={references} loading={false} onOpen={() => {}} onCreate={() => {}} />);
    // The run counts only the active references that label its classes; the cohort group counts every active one.
    expect(html).toContain('Unlabeled · 1 reference standard</small>');
    expect(html).toContain('Cohort external · Unlabeled · 2 reference standards');
    expect(html).not.toContain('predictions only');
    expect(html).toContain('<option value="unlabeled">Unlabeled cohort</option>');
  });

  it('filters by labels and scopes to one predictor, saying so with a way back', () => {
    const labeled = render({ ...newRunFilters(), kind: 'labeled' });
    expect(labeled).toContain('Evaluation · Seed ensemble');
    expect(labeled).not.toContain('Inference · Fold ensemble');
    const scoped = render(newRunFilters({ predictor: fold.id }));
    expect(scoped).toContain('Showing the runs of Same study name · Batch batch · Configuration candidate-1 · Fold ensemble');
    expect(scoped).toContain('Show all runs');
    expect(scoped).toContain('Inference · Fold ensemble');
    expect(scoped).not.toContain('Evaluation · Seed ensemble');
  });

  it('offers to apply predictors when nothing has run yet', () => {
    const html = renderToStaticMarkup(<ApplyRunsTable runs={[]} predictors={[]} experiments={[]} cohorts={[]} loading={false} onOpen={() => {}} onCreate={() => {}} />);
    expect(html).toContain('No models applied yet');
    expect(html).toContain('Apply predictors');
    expect(html).not.toContain('Search runs');
  });
});
