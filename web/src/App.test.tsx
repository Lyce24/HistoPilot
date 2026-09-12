import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Content, moduleForPage, pageFromHash, projectFromUrl } from './App';
import { buildRoadmap } from './lib/roadmap';
import type { Workspace } from './api/types';
import type { Roadmap } from './pages/ProjectRoadmap';
import { useRoadmap } from './components/useRoadmap';

const workspace = { mode: 'local', project: { id: 'project' }, dataset: { slideCount: 0 }, drafts: [], featureSets: [], cohortSnapshots: [] } as unknown as Workspace;
function roadmap(unlocked = false, hasData = true): Roadmap {
  const modules = buildRoadmap(workspace).map((module) => module.id === 'experiments' ? { ...module, unlocked, blockers: unlocked ? [] : module.blockers } : module);
  return {
    modules, byId: Object.fromEntries(modules.map((module) => [module.id, module])),
    checksById: Object.fromEntries(modules.map((module) => [module.id, { hasData, isLoading: !hasData, error: null }])),
    hasData, isLoading: !hasData, loading: !hasData, error: null, refetch: async () => {},
  } as Roadmap;
}

describe('experiment predictor navigation and direct URL gates', () => {
  it.each(['overview', 'dataset', 'experiments', 'source-cv', 'test-data', 'clinical-utility', 'interpretation'] as const)('keeps a trashed project on its recovery path for direct %s navigation', (page) => {
    const html = renderToStaticMarkup(<Content page={page} workspace={{ ...workspace, project: { ...workspace.project, lifecycleState: 'trashed' } }} roadmap={roadmap(false, false)} />);
    expect(html).toContain('This project is in Trash');
    expect(html).toContain('href="#cleanup"');
    expect(html).not.toContain('Checking module prerequisites');
    expect(html).not.toContain('<form');
  });

  it.each(['experiments', 'source-cv'] as const)('opens retained model work through direct %s navigation with archived inputs hidden from selectors', (page) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const source = { ...workspace, project: { ...workspace.project, name: 'Retained study', config: {} }, encoders: [], milModels: [] } as Workspace;
    client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
    for (const kind of ['protocol', 'feature']) client.setQueryData(['scientific', 'project', 'configurations', kind], { configurations: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    client.setQueryData(['scientific', 'project', 'drafts'], { drafts: [{ id: 'mil-draft', name: 'Retained experiment inputs', revision: 1, status: 'editable', payload: { type: 'mil-experiment', spec: {} } }] });
    client.setQueryData(['development-batches', 'project'], { items: [], executions: [] });
    client.setQueryData(['evaluation-cohorts', 'project'], { items: [] });
    client.setQueryData(['predictors', 'project'], { items: [] });
    client.setQueryData(['model-evaluations', 'project'], { items: [] });
    client.setQueryData(['model-experiments', 'project', 'summary'], { items: [{ id: 'legacy-mil-draft', key: 'draft:mil-draft', name: 'Retained experiment inputs', revision: 1, state: 'active', status: 'planned', legacy: true, notes: '', tags: [], inputs: null, batches: [], drafts: [], createdAt: '', updatedAt: '', predictorId: null }] });
    function DirectRoute() { return <Content page={page} workspace={source} roadmap={useRoadmap(source)} />; }
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><DirectRoute /></QueryClientProvider>);
      expect(html).toContain('Retained experiment inputs');
      expect(html).toContain('Create experiment');
      expect(html).not.toContain('Complete the prerequisites to unlock this module');
      expect(html).not.toContain('Launch batch');
    } finally { client.clear(); }
  });

  it('routes cleanup directly for recovery even when project prerequisites cannot load', () => {
    expect(pageFromHash('#cleanup')).toBe('cleanup');
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['cleanup', 'project'], { projectId: 'project', revision: 1, projectState: 'trashed', items: [], audit: [], note: '' });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><Content page="cleanup" workspace={{ ...workspace, project: { ...workspace.project, lifecycleState: 'trashed' } }} roadmap={roadmap(false, false)} /></QueryClientProvider>);
    expect(html).toContain('Workspace cleanup');
    expect(html).toContain('Restore to Active');
    expect(html).not.toContain('Checking module prerequisites');
  });
  it.each(['#selection', '#predictor', '#post-development'])('keeps %s as a legacy route owned by Experiments', (hash) => {
    expect(pageFromHash(hash)).toBe('post-development');
    expect(moduleForPage(pageFromHash(hash))).toBe('experiments');
  });

  it.each([
    ['#clinical', 'clinical-utility'], ['#clinical-utility?evaluation=one', 'clinical-utility'],
    ['#interpret', 'interpretation'], ['#interpretation?predictor=one', 'interpretation'],
  ] as const)('routes %s to its Clinical insights module', (hash, page) => {
    expect(pageFromHash(hash)).toBe(page);
    expect(moduleForPage(page)).toBe(page);
  });

  it('preserves the development-results alias without using it to bypass a ready predictor', () => {
    expect(pageFromHash('#source-cv')).toBe('source-cv');
    expect(moduleForPage('source-cv')).toBe('experiments');
    expect(moduleForPage('selection')).toBe('experiments');
    expect(moduleForPage('predictor')).toBe('experiments');
    expect(pageFromHash('#unknown')).toBe('overview');
    expect(pageFromHash('#experiments?experiment=draft-1&tab=runs')).toBe('experiments');
    expect(pageFromHash('#post-development?experiment=draft-1')).toBe('post-development');
    expect(pageFromHash('#evaluation?predictor=configuration-1')).toBe('evaluation');
    expect(pageFromHash('#cleanup?key=configuration%3Aone')).toBe('cleanup');
  });

  it.each(['post-development', 'selection', 'predictor'] as const)('applies the experiment gate to legacy %s navigation', (page) => {
    const html = renderToStaticMarkup(<Content page={page} workspace={workspace} roadmap={roadmap()} />);
    expect(html).toContain('Complete the prerequisites to unlock this module');
    expect(html).toContain('Experiments');
    expect(html).not.toContain('Predictor selection and freeze');
  });

  it('waits for prerequisite evidence before rendering a saved post-development route', () => {
    const html = renderToStaticMarkup(<Content page="post-development" workspace={workspace} roadmap={roadmap(true, false)} />);
    expect(html).toContain('Checking module prerequisites');
    expect(html).not.toContain('Predictor selection and freeze');
  });

  it('retains historical predictors and refit recovery without a new build step', () => {
    const client = new QueryClient();
    client.setQueryData(['predictors', 'project'], { items: [] });
    client.setQueryData(['predictor-choices', 'project'], { items: [] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><Content page="post-development" workspace={workspace} roadmap={roadmap(true)} /></QueryClientProvider>);
      expect(html).toContain('Historical predictors');
      expect(html).toContain('automatic builds are managed inside each experiment');
      expect(html).toContain('Predictor library');
      expect(html).toContain('Refit jobs');
      expect(html).toContain('href="#experiments"');
      expect(html).not.toContain('Launch batch');
      expect(html).not.toContain('Review predictor');
    } finally { client.clear(); }
  });

  it('explains test cohort and experiment predictor readiness at evaluation entry', () => {
    const client = new QueryClient();
    for (const key of ['predictors', 'model-evaluations', 'evaluation-cohorts']) client.setQueryData([key, 'project'], { items: [] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><Content page="evaluation" workspace={workspace} roadmap={roadmap(true)} /></QueryClientProvider>);
      expect(html).toContain('Test cohorts');
      expect(html).toContain('href="#experiments"');
      expect(html).toContain('metrics use labeled records only');
      expect(html).toContain('Each ready ensemble or refit predictor keeps its own results');
      expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review experiment evaluation/);
    } finally { client.clear(); }
  });
});

describe('project URL parameter', () => {
  it('reads ?project= and keeps ?experiment= links from earlier versions working', () => {
    expect(projectFromUrl('http://127.0.0.1/?project=abc#overview')).toBe('abc');
    expect(projectFromUrl('http://127.0.0.1/?experiment=old#overview')).toBe('old');
    expect(projectFromUrl('http://127.0.0.1/?project=new&experiment=old')).toBe('new');
    expect(projectFromUrl('http://127.0.0.1/')).toBeNull();
  });
});
