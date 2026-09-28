import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ExperimentPredictors from './ExperimentPredictors';
import type { ExperimentPredictorExecution, ModelExperiment } from '../api/experiments';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
const execution = (changes: Partial<ExperimentPredictorExecution> = {}): ExperimentPredictorExecution => ({
  status: 'waiting', counts: { total: 4, ensemble: 2, refit: 2, completed: 1, waiting: 2, active: 1, failed: 0, cancelled: 0 },
  items: [{ key: 'item', source: { experimentId: 'exp', batchId: 'batch', candidateId: 'c', trainingSeed: 42, splitSeed: 7 }, method: 'refit', configurationNumber: 1, foldCount: 5, runIds: [], status: 'queued', recordId: 'refit', predictorId: null, epochBudget: null, execution: { status: 'queued', executor: 'task-center', waitingReason: 'Waiting for a GPU slot (5/5)' }, error: null }],
  error: null, updatedAt: '2026-09-27T10:00:00Z', sessionName: 'hp-predictors', logPath: '/coordinator/worker.log', retryable: false, cancellable: true, ...changes,
});
function render(changes: Partial<ExperimentPredictorExecution> = {}, view: 'all' | 'progress' | 'library' = 'all') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['predictors', 'project'], { items: [], executionEnabled: true });
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

  it('keeps the tmux reconnect hint for coordinators started before the Task Center', () => {
    expect(render()).toContain('tmux attach -t hp-predictors');
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
    for (const text of ['Predictor jobs ·', 'Resume predictor creation', 'Cancel remaining predictors', '/coordinator/worker.log', 'Another operation is changing', 'ready to evaluate']) expect(html).not.toContain(text);
  });

  it('shows only the predictor library on the Predictors tab', () => {
    const html = render({ executor: 'task-center' }, 'library');
    expect(html).toContain('ready to evaluate');
    expect(html).not.toContain('Predictor creation');
    expect(html).not.toContain('created');
  });
});
