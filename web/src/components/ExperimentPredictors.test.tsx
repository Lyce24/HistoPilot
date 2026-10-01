import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ExperimentPredictors from './ExperimentPredictors';
import type { ExperimentPredictorExecution, ModelExperiment } from '../api/experiments';
import { legacyRecordNote } from './LegacyRecordNote';
import type { SeedEnsembleChoice } from '../api/predictors';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
const execution = (changes: Partial<ExperimentPredictorExecution> = {}): ExperimentPredictorExecution => ({
  status: 'waiting', counts: { total: 4, ensemble: 2, refit: 2, completed: 1, waiting: 2, active: 1, failed: 0, cancelled: 0 },
  items: [{ key: 'item', source: { experimentId: 'exp', batchId: 'batch', candidateId: 'c', trainingSeed: 42, splitSeed: 7 }, method: 'refit', configurationNumber: 1, foldCount: 5, runIds: [], status: 'queued', recordId: 'refit', predictorId: null, epochBudget: null, execution: { status: 'queued', executor: 'task-center', waitingReason: 'Waiting for a GPU slot (5/5)' }, error: null }],
  error: null, updatedAt: '2026-09-27T10:00:00Z', sessionName: 'hp-predictors', logPath: '/coordinator/worker.log', retryable: false, cancellable: true, ...changes,
});
function render(changes: Partial<ExperimentPredictorExecution> = {}, view: 'all' | 'progress' | 'library' = 'all', seedEnsembles: SeedEnsembleChoice[] = []) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['predictors', 'project'], { items: [], executionEnabled: true });
  client.setQueryData(['seed-ensembles', 'project', 'exp'], { items: seedEnsembles });
  const record = { id: 'exp', key: 'draft:exp', name: 'Study', notes: '', tags: [], revision: 1, state: 'active', status: 'running', stage: 'running', legacy: false, createdAt: '', updatedAt: '', inputs: null, batches: [], drafts: [], predictorId: null, predictorPolicies: { batch: { method: 'both', refitPercentile: 75 } }, predictorExecution: execution(changes) } as ModelExperiment;
  return renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentPredictors project="project" record={record} view={view} /></QueryClientProvider>);
}

describe('experiment predictor creation', () => {
  it('explains waiting jobs without a per-experiment refit slot and links the Task Center', () => {
    const html = render({ executor: 'task-center' });
    expect(html).toContain('Waiting jobs need their source batch to complete. Predictor jobs run through the <a href="#task-center?project=project">Task Center</a>.');
    expect(html).not.toContain('one at a time');
    expect(html).toContain('Waiting for a GPU slot (5/5)');
    expect(html).not.toContain('tmux attach');
  });

  it('shows a coordinator created before the Task Center read-only, without its session or controls', () => {
    const html = render({ status: 'interrupted', retryable: true, cancellable: true, error: { code: 'CREATED_BEFORE_TASK_CENTER', message: 'Created before the Task Center; it did not finish.' } }, 'progress');
    expect(html).toContain('>Interrupted<');
    expect(html).toContain('Created before the Task Center; it did not finish.');
    expect(html).toContain(legacyRecordNote);
    expect(html).toContain('/coordinator/worker.log');
    for (const text of ['hp-predictors', 'tmux attach', 'Resume predictor creation', 'Cancel remaining predictors']) expect(html).not.toContain(text);
    expect(render({ executor: 'task-center' })).not.toContain(legacyRecordNote);
  });

  it('does not call a coordinator cancelled before it ever started a pre-Task Center record', () => {
    const html = render({ status: 'cancelled', sessionName: null, logPath: null, cancellable: false }, 'progress');
    expect(html).toContain('>Cancelled<');
    expect(html).not.toContain(legacyRecordNote);
  });

  it('says so when the Task Center runner is not running for queued predictor work', () => {
    const down = render({ executor: 'task-center', status: 'queued', runnerAlive: false });
    expect(down).toContain('The Task Center runner is not running');
    expect(down).toContain('href="#task-center?project=project">Start it in the Task Center →</a>');
    expect(render({ executor: 'task-center', status: 'queued', runnerAlive: null })).not.toContain('runner is not running');
    expect(render({ executor: 'task-center', status: 'completed', runnerAlive: false })).not.toContain('runner is not running');
  });

  it('leaves jobs, logs and controls of Task Center experiments to the Task Center on the Runs tab', () => {
    const html = render({ executor: 'task-center', status: 'attention', retryable: true, error: { code: 'PROJECT_BUSY', message: 'Another operation is changing this workspace.' } }, 'progress');
    expect(html).toContain('Predictors: 1 of 4 created');
    expect(html).toContain('Ready predictors are listed under Predictors');
    for (const text of ['Predictor jobs ·', 'Resume predictor creation', 'Cancel remaining predictors', '/coordinator/worker.log', 'Another operation is changing', 'ready to apply']) expect(html).not.toContain(text);
  });

  it('shows only the predictor library on the Predictors tab', () => {
    const html = render({ executor: 'task-center' }, 'library');
    expect(html).toContain('ready to apply');
    expect(html).not.toContain('Predictor creation');
    expect(html).not.toContain('created');
  });
  it('offers one seed ensemble per multi-seed configuration and links a built one to evaluation', () => {
    const choice = (candidate: number, changes: Partial<SeedEnsembleChoice> = {}): SeedEnsembleChoice => ({
      experimentId: 'exp', experimentName: 'Study', batchId: 'batch', batchName: 'ABMIL baseline', candidateId: `candidate-${candidate}`, candidateNumber: candidate,
      trainingSeeds: [42, 43, 44], splitSeeds: [42], seedGroups: 3, members: 15, completedRuns: 15, eligible: true, reason: null, existingPredictorId: null, ...changes,
    });
    const html = render({ executor: 'task-center', status: 'completed' }, 'library', [
      choice(1), choice(2, { eligible: false, existingPredictorId: 'pooled' }), choice(3, { seedGroups: 1, trainingSeeds: [42], members: 5 }),
    ]);
    expect(html).toContain('Seed ensembles');
    expect(html).toContain('3 training × 1 split seed');
    expect(html).toContain('15 fold models');
    expect(html.match(/Build seed ensemble/g)).toHaveLength(1);
    expect(html).toContain('href="#apply?view=new&amp;experiment=exp&amp;predictor=pooled">Built · apply</a>');
    // A configuration with one seed group has nothing to pool.
    expect(html).not.toContain('Configuration 3');
    expect(render({ executor: 'task-center', status: 'completed' }, 'library')).not.toContain('Seed ensembles');
  });
});
