import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { EvaluationCohort } from '../api/evaluation';
import type { Workspace } from '../api/types';
import { fixtureEvaluation } from '../testFixtures/evaluations';
import { fixturePredictor } from '../testFixtures/predictors';
import LocalApplyModels from './LocalApplyModels';

const workspace = { mode: 'local', project: { id: 'project', name: 'Study', lifecycleState: 'active' } } as Workspace;
const source = fixturePredictor(1, 11, 'ensemble');
const cohort = (id: string, purpose?: 'inference') => ({ id, current: true, createdAt: '2026-09-28', versionLabel: { tag: `Cohort ${id}` }, manifest: { spec: { datasetId: 'dataset-1', purpose, target: purpose ? null : { field: 'grade' }, eligibility: [] }, summary: { includedSlides: 120, labeledSlides: purpose ? 0 : 120 } } }) as unknown as EvaluationCohort;
const run = { ...fixtureEvaluation(source, 0.8, 0.7, 'testing'), id: 'run-1' };
run.manifest.name = 'Evaluation · Baseline · Fold ensemble';
const batch = { id: 'batch-1', name: 'External validation', status: 'completed', cohortId: 'unlabeled', createdAt: '2026-09-29', items: [{ predictorId: source.id, predictorName: source.manifest.name, method: 'ensemble' as const, status: 'completed', evaluationId: 'run-1' }] };
const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.unstubAllGlobals(); });
function render(hash: string) {
  vi.stubGlobal('window', { location: { hash } });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['predictors', 'project'], { items: [source] });
  client.setQueryData(['model-evaluations', 'project'], { items: [run] });
  client.setQueryData(['evaluation-cohorts', 'project'], { items: [cohort('testing'), cohort('unlabeled', 'inference')] });
  client.setQueryData(['evaluation-batches', 'project'], { items: [batch] });
  client.setQueryData(['evaluation-batch', 'project', 'batch-1'], batch);
  client.setQueryData(['model-experiment-summaries', 'project'], { items: [] });
  client.setQueryData(['clinical-analyses', 'project', 'all'], { items: [] });
  client.setQueryData(['clinical-analysis', 'project', 'analysis-1'], { id: 'analysis-1', manifest: { evaluationId: 'run-1', selection: { evaluationId: 'run-1' } } });
  client.setQueryData(['evaluation-drafts', 'project'], { drafts: [] });
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
  return renderToStaticMarkup(<QueryClientProvider client={client}><LocalApplyModels workspace={workspace} /></QueryClientProvider>);
}

describe('Apply models views', () => {
  it('lists batches of both kinds of cohort and opens one on its runs', () => {
    const library = render('#apply?view=batches');
    expect(library).toMatch(/aria-pressed="true">Batches<span>1<\/span>/);
    expect(library).toContain('External validation');
    expect(library).toContain('Cohort unlabeled<small>Unlabeled · predictions only</small>');
    const detail = render('#apply?batch=batch-1');
    expect(detail).toContain('<h1>External validation</h1>');
    expect(detail).toContain('1 predictor applied to Cohort unlabeled · Unlabeled · predictions only.');
    expect(detail).toContain('Back to batches');
    expect(detail).toContain('Runs in this batch');
    expect(detail).toContain('Open run');
  });

  it('keeps cohorts, method comparison and runs as views of one library', () => {
    const cohorts = render('#apply?view=cohorts');
    expect(cohorts).toMatch(/aria-pressed="true">Cohorts<span>2<\/span>/);
    expect(cohorts).toContain('Create labeled cohort');
    expect(cohorts).toContain('Create unlabeled cohort');
    expect(cohorts).toContain('<div class="eyebrow">04 Apply models</div>');
    const methods = render('#apply?view=methods');
    expect(methods).toMatch(/aria-pressed="true">Compare methods</);
    expect(methods).toContain('Ensemble vs refit');
  });

  it('opens a saved clinical analysis inside its run, on Performance', () => {
    const html = render('#apply?clinical=analysis-1');
    expect(html).toContain('<h1>Evaluation · Baseline · Fold ensemble</h1>');
    expect(html).toMatch(/id="run-tab-performance" aria-current="page"/);
    expect(html).toContain('Back to runs');
  });

  it('says when a linked run is unavailable instead of opening another', () => {
    const html = render('#apply?run=missing');
    expect(html).toContain('This run is unavailable.');
    expect(html).not.toContain('id="run-tab-');
  });
});
