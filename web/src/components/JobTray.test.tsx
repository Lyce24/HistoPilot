import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import JobTray, { JobTrayLinks, extractionJobLink, trainingJobLinks, computeJobLink } from './JobTray';
import type { ExtractionJob } from '../api/trident';
import type { DevelopmentBatchList } from '../api/development';

const clients: QueryClient[] = [];
function client() {
  const value = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(value);
  value.setQueryData(['extractions', 'project', 'jobs'], { jobs: [] });
  return value;
}
const render = (value: QueryClient) => renderToStaticMarkup(<QueryClientProvider client={value}><JobTray projectId="project" /></QueryClientProvider>);
const running = { id: 'refit', manifest: { name: 'Refit model' }, execution: { status: 'running' } };
afterEach(() => clients.splice(0).forEach((value) => value.clear()));

const extraction: ExtractionJob = { id: 'extract/one', state: 'running', createdAt: '', updatedAt: '', outputPath: '/features', logPath: '/logs/extraction', sessionName: 'extraction', spec: { datasetId: null, outputPath: '/features', options: { patch_encoder: 'uni_v2' } }, progress: { stage: 'patch_features', stages: [], label: 'Patch features', detail: '', completed: 12, total: 80, unit: 'slides', percent: 15, currentSlide: null, elapsedSeconds: null, etaSeconds: null, ratePerSecond: null, scope: 'stage', warnings: [] } };

describe('compute job summary', () => {
  it('counts extraction before any feature bundle or training job exists', () => {
    const value = client();
    value.setQueryData(['extractions', 'project', 'jobs'], { jobs: [extraction] });
    for (const key of ['refit-builds', 'model-evaluations', 'interpretations']) value.setQueryData([key, 'project'], { items: [] });
    value.setQueryData(['development-batches', 'project'], { items: [], executionImplemented: true, executions: [] });
    expect(render(value)).toContain('1 active job');
    expect(render(value)).not.toContain('No jobs yet');
    value.setQueryData(['extractions', 'project', 'jobs'], { jobs: [{ ...extraction, state: 'queued' }, { ...extraction, id: 'second', state: 'cancelling' }] });
    expect(render(value)).toContain('2 active jobs');
  });

  it('does not declare a project empty while extraction status is still loading', () => {
    const value = client();
    value.removeQueries({ queryKey: ['extractions', 'project', 'jobs'] });
    for (const key of ['refit-builds', 'model-evaluations', 'interpretations']) value.setQueryData([key, 'project'], { items: [] });
    value.setQueryData(['development-batches', 'project'], { items: [], executions: [] });
    expect(render(value)).toContain('Checking compute jobs…');
    expect(render(value)).not.toContain('No jobs yet');
  });

  it('links extraction to its exact progress page and displays stage counts', () => {
    const html = renderToStaticMarkup(<JobTrayLinks jobs={[extractionJobLink(extraction)]} />);
    expect(html).toContain('href="#features?extraction=extract%2Fone"');
    expect(html).toContain('View progress: uni_v2 extraction');
    expect(html).toContain('Patch features · 12/80 slides');
    expect(html).toContain('running');
  });

  it('opens the selected training batch, including historical batches', () => {
    const data = { items: [{ id: 'batch/one', manifest: { spec: { experimentId: 'experiment/one', batchName: 'First batch' } } }], executions: [{ batchId: 'batch/one', status: 'running', runCounts: { completed: 2, total: 5 } }] } as DevelopmentBatchList;
    const linked = trainingJobLinks(data)[0];
    expect(linked.link).toBe('#experiments?experiment=experiment%2Fone&tab=runs&batch=batch%2Fone');
    const html = renderToStaticMarkup(<JobTrayLinks jobs={[linked]} />);
    expect(html).toContain('First batch');
    expect(html).toContain('2/5 completed');
    expect(trainingJobLinks({ ...data, items: [] })[0].link).toBe('#experiments?experiment=legacy-batch%2Fone&tab=runs&batch=batch%2Fone');
  });

  it('links compute records to the selected job rather than a parent library', () => {
    expect(computeJobLink('refit', 'job/one')).toBe('#post-development?tab=refits&refit=job%2Fone');
    expect(computeJobLink('evaluation', 'job/one')).toBe('#evaluation?evaluation=job%2Fone');
    expect(computeJobLink('interpretation', 'job/one')).toBe('#interpretation?interpretation=job%2Fone');
  });

  it('waits for every compute family before reporting completed counts', () => {
    const value = client();
    value.setQueryData(['development-batches', 'project'], { items: [], executionImplemented: true, executions: [] });
    expect(render(value)).toContain('Checking compute jobs…');
    expect(render(value)).not.toContain('0 fold runs completed');
    value.setQueryData(['refit-builds', 'project'], { items: [running] });
    expect(render(value)).toContain('At least 1 active job · checking remaining jobs…');
  });

  it('shows refit activity even when this service cannot launch training', () => {
    const value = client();
    value.setQueryData(['development-batches', 'project'], { items: [], executionImplemented: false, executions: [] });
    value.setQueryData(['refit-builds', 'project'], { items: [running] });
    value.setQueryData(['model-evaluations', 'project'], { items: [] });
    value.setQueryData(['interpretations', 'project'], { items: [] });
    expect(render(value)).toContain('1 active job · 0 fold runs completed');
    expect(render(value)).not.toContain('Execution unavailable');
  });

  it('names an empty project instead of reporting zero completed fold runs', () => {
    const value = client();
    for (const key of ['refit-builds', 'model-evaluations', 'interpretations'] as const) value.setQueryData([key, 'project'], { items: [] });
    value.setQueryData(['development-batches', 'project'], { items: [], executionImplemented: true, executions: [] });
    expect(render(value)).toContain('No jobs yet');
    expect(render(value)).not.toContain('0 fold runs completed');
    value.setQueryData(['development-batches', 'project'], { items: [], executionImplemented: true, executions: [{ batchId: 'batch', status: 'completed', runCounts: { total: 2, completed: 2 }, runs: [] }] });
    expect(render(value)).toContain('No active jobs · 2 fold runs completed');
  });

  it('retains known activity with an explicit stale status when a query fails', () => {
    const value = client();
    value.setQueryData(['refit-builds', 'project'], { items: [running] });
    value.getQueryCache().find({ queryKey: ['refit-builds', 'project'] })!.setState({ status: 'error', error: new Error('Connection lost') });
    expect(render(value)).toContain('1 active job · 0 fold runs completed · status may be outdated');
  });
});
