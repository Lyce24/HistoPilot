import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fixturePredictor } from '../testFixtures/predictors';
import { fixtureExperiment } from '../testFixtures/evaluations';
import BulkEvaluationRunner, { EvaluationBatchStatus, batchMemberStatus } from './BulkEvaluationRunner';
import { legacyRecordNote } from './LegacyRecordNote';
import { taskCenterKeys } from '../api/taskCenter';
import { fixtureRollup } from '../testFixtures/taskCenter';
import type { EvaluationCohort } from '../api/evaluation';
import { initialEvaluationInputs } from './EvaluationInputSettings';

function render(ids?: string[], linkedExperiment = '', cohorts: EvaluationCohort[] = [], linkedPredictor = '') {
  const client = new QueryClient();
  client.setQueryData(['evaluation-batches', 'p'], { items: [] });
  client.setQueryData(['feature-bundles', 'p'], { items: [] });
  const items = [fixturePredictor(1, 11, 'ensemble', 'one'), fixturePredictor(1, 11, 'refit', 'one'), fixturePredictor(2, 22, 'ensemble', 'two')];
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}><BulkEvaluationRunner project="p" predictors={items} experiments={[fixtureExperiment('one'), fixtureExperiment('two'), fixtureExperiment('pending', 'running'), fixtureExperiment('skip-policy')]} cohorts={cohorts} linkedCohort={cohorts[0]?.id} linkedExperiment={linkedExperiment} linkedPredictor={linkedPredictor} experimentIds={ids} onOpenEvaluation={() => {}} /></QueryClientProvider>);
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
  it('honors a linked predictor without adding other ready predictors from its experiment', () => {
    const linked = fixturePredictor(1, 11, 'ensemble', 'one');
    const html = render(['one'], '', [], linked.id);
    expect(html).toContain('1 predictor to evaluate from 1 selected experiment');
    expect(html).not.toContain('2 predictors to evaluate');
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

describe('evaluation batch status', () => {
  it('shows the batch owner status line and member results, with no cancel or raw statuses', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['evaluation-batch', 'p', 'batch'], { id: 'batch', status: 'running', cohortId: 'c', items: [{ predictorId: 'p1', predictorName: 'Ensemble · seed 42', method: 'ensemble', status: 'not_started' }, { predictorId: 'p2', predictorName: 'Refit · seed 42', method: 'refit', status: 'running', evaluationId: 'e2' }] });
    client.setQueryData(taskCenterKeys.rollup({ ownerKind: 'evaluation-batch', ownerId: 'batch', project: 'p' }), fixtureRollup({ counts: { running: 1, queued: 1 }, active: 1, pending: 1, live: 2, href: '#task-center?owner=o&project=p' }));
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><EvaluationBatchStatus project="p" id="batch" onOpen={() => {}} kind="inference" /></QueryClientProvider>);
      expect(html).toContain('href="#task-center?owner=o&amp;project=p"');
      expect(html).toContain('>Not started<');
      expect(html).toContain('>Running<');
      expect(html).toContain('Open predictions');
      expect(html).not.toContain('Cancel inference batch');
      expect(batchMemberStatus('not_started')).toBe('Not started');
      expect(batchMemberStatus('something_new')).toBe('Something new');
    } finally { client.clear(); }
  });

  it('shows a batch whose members ran before the Task Center read-only, with no Cancel', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const batch = { id: 'batch', status: 'interrupted', cohortId: 'c', items: [{ predictorId: 'p1', predictorName: 'Ensemble · seed 42', method: 'ensemble' as const, status: 'interrupted', evaluationId: 'e1', execution: { status: 'interrupted' as const, executor: 'tmux' as const, sessionName: 'hp-eval' } }] };
    client.setQueryData(['evaluation-batch', 'p', 'batch'], batch);
    client.setQueryData(taskCenterKeys.rollup({ ownerKind: 'evaluation-batch', ownerId: 'batch', project: 'p' }), fixtureRollup({ state: 'not-started', counts: {}, byKind: {}, live: 0, active: 0, pending: 0 }));
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><EvaluationBatchStatus project="p" id="batch" onOpen={() => {}} /></QueryClientProvider>);
      expect(html).toContain(legacyRecordNote);
      expect(html).toContain('>Interrupted<');
      for (const text of ['Cancel evaluation batch', 'hp-eval', 'tmux attach']) expect(html).not.toContain(text);
      client.setQueryData(['evaluation-batch', 'p', 'batch'], { ...batch, items: [{ ...batch.items[0], execution: { status: 'interrupted', executor: 'task-center' } }] });
      expect(renderToStaticMarkup(<QueryClientProvider client={client}><EvaluationBatchStatus project="p" id="batch" onOpen={() => {}} /></QueryClientProvider>)).not.toContain(legacyRecordNote);
    } finally { client.clear(); }
  });
});
