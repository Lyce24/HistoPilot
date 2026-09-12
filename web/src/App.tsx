import { useEffect, useRef, useState } from 'react';
import type { Page, Workspace } from './api/types';
import { useWorkspace } from './api/queries';
import { Badge, Icon } from './components/ui';
import JobTray from './components/JobTray';
import Dataset from './pages/Dataset';
import Cohort from './pages/Cohort';
import Features from './pages/Features';
import Experiments from './pages/Experiments';
import System from './pages/System';
import { EvaluationPage, ExplorerPage, ProvenancePage } from './pages/Results';
import { downloadJSON } from './lib/download';
import { useUIStore } from './store/ui';
import Start from './pages/Start';
import LocalWorkspace from './pages/LocalWorkspace';
import LocalDataset from './pages/LocalDataset';
import LocalProtocol from './pages/LocalProtocol';
import LocalFeatures from './pages/LocalFeatures';
import LocalExperiments from './pages/LocalExperiments';
import LocalEvaluationSetup from './pages/LocalEvaluationSetup';
import LegacyPredictorRoute from './components/LegacyPredictorRoute';
import LocalModelEvaluation from './pages/LocalModelEvaluation';
import WorkspaceCleanup from './pages/WorkspaceCleanup';
import LocalClinicalUtility from './pages/LocalClinicalUtility';
import LocalInterpretation from './pages/LocalInterpretation';
import ProjectRoadmap, { ModuleStatus, moduleIcons, completedModuleLabel, type Roadmap } from './pages/ProjectRoadmap';
import RoadmapModule from './pages/RoadmapModule';
import { useRoadmap } from './components/useRoadmap';
import WorkspaceErrorBoundary from './components/WorkspaceErrorBoundary';

const pages: Record<Page, string> = {
  overview: 'Project roadmap', dataset: 'Datasets',
  cohort: 'Targets & splits', features: 'Slide features',
  experiments: 'Experiments', 'source-cv': 'Development results',
  'post-development': 'Historical predictors',
  selection: 'Experiments', predictor: 'Experiments',
  'test-data': 'Test cohorts', evaluation: 'Evaluate models',
  'clinical-utility': 'Clinical utility', interpretation: 'Model interpretation',
  reports: 'Metrics, clinical analyses and reports', 'example-results': 'Illustrative results', explorer: 'Slide explorer',
  provenance: 'Provenance', cleanup: 'Workspace cleanup', system: 'System & storage',
};
export function pageFromHash(value: string): Page {
  const hash = value.replace(/^#/, '').split('?')[0];
  if (['selection', 'predictor', 'predictors', 'build-predictors'].includes(hash)) return 'post-development';
  if (hash === 'test-cohorts') return 'test-data';
  if (hash === 'evaluate-models') return 'evaluation';
  if (hash === 'clinical') return 'clinical-utility';
  if (hash === 'interpret') return 'interpretation';
  return Object.hasOwn(pages, hash) ? (hash as Page) : 'overview';
}
function currentPage(): Page { return pageFromHash(window.location.hash); }
/** The workspace is a project; `?experiment=` remains an alias for links saved by earlier versions. */
export function projectFromUrl(href = window.location.href): string | null {
  const params = new URL(href).searchParams;
  return params.get('project') ?? params.get('experiment');
}
export function moduleForPage(page: Page) {
  return ['selection', 'predictor', 'post-development', 'source-cv'].includes(page) ? 'experiments' : page === 'reports' ? 'evaluation' : page;
}
export function Content({ page, workspace, roadmap }: { page: Page; workspace: Workspace; roadmap: Roadmap }) {
  if (page === 'cleanup' && workspace.mode === 'local') return <WorkspaceCleanup workspace={workspace} />;
  if (workspace.mode === 'local' && workspace.project.lifecycleState === 'trashed') return <div className="roadmap-loading" role="status">
    <h1>This project is in Trash</h1>
    <p>Its records and files have been retained. Restore the project in Workspace cleanup before continuing your work.</p>
    <a className="btn btn-primary" href="#cleanup">Open Workspace cleanup</a>
  </div>;
  if (page === 'overview') return <ProjectRoadmap workspace={workspace} roadmap={roadmap} />;
  if (page === 'example-results' && workspace.mode === 'synthetic-demo') return <EvaluationPage workspace={workspace} />;
  const moduleId = moduleForPage(page);
  const module = roadmap.modules.find((item) => item.id === moduleId);
  const check = module ? roadmap.checksById[module.id] : null;
  if (check && !check.hasData) return <div className="roadmap-loading" role={check.error ? 'alert' : 'status'}>
    <h1>{check.error ? 'Could not check module prerequisites' : 'Checking module prerequisites'}</h1>
    <p>{check.error?.message ?? 'Loading saved project records…'}</p>
    {check.error ? <button className="btn btn-primary" onClick={() => void roadmap.refetch()}>Try again</button> : null}
    <a className="btn btn-secondary" href="#overview">Back to roadmap</a>
  </div>;
  if (module && !module.unlocked) return <RoadmapModule module={module} roadmap={roadmap} workspace={workspace} />;
  if (workspace.mode === 'local') {
    if (page === 'test-data') return <LocalEvaluationSetup workspace={workspace} />;
    if (page === 'clinical-utility') return <LocalClinicalUtility workspace={workspace} />;
    if (page === 'interpretation') return <LocalInterpretation workspace={workspace} />;
    if (page === 'source-cv') return <LocalExperiments workspace={workspace} initialTab="results" />;
    if (['post-development', 'selection', 'predictor'].includes(page)) return <LegacyPredictorRoute workspace={workspace} />;
    if (page === 'dataset') return <LocalDataset workspace={workspace} />;
    if (page === 'cohort') return <LocalProtocol workspace={workspace} />;
    if (page === 'features') return <LocalFeatures workspace={workspace} />;
    if (page === 'experiments') return <LocalExperiments workspace={workspace} />;
    if (moduleId === 'evaluation') return <LocalModelEvaluation workspace={workspace} />;
    if (page !== 'system') return <LocalWorkspace page={page} workspace={workspace} />;
  }
  switch (page) {
    case 'dataset': return <Dataset workspace={workspace} />;
    case 'cohort': return <Cohort workspace={workspace} />;
    case 'features': return <Features workspace={workspace} />;
    case 'experiments': return <Experiments workspace={workspace} />;
    case 'evaluation': return <EvaluationPage workspace={workspace} />;
    case 'explorer': return <ExplorerPage workspace={workspace} />;
    case 'provenance': return <ProvenancePage workspace={workspace} />;
    case 'system': return <System />;
    default: return <ProjectRoadmap workspace={workspace} roadmap={roadmap} />;
  }
}
export default function App() {
  const [projectId, setProjectId] = useState(() => projectFromUrl());
  useEffect(() => {
    const restore = () => {
      useUIStore.getState().resetSelection();
      setProjectId(projectFromUrl());
    };
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
  }, []);
  useEffect(() => { if (!projectId) document.title = 'HistoPilot · Start'; }, [projectId]);
  function navigate(id: string | null, page: Page = 'overview') {
    const url = new URL(window.location.href);
    url.searchParams.delete('experiment');
    if (id) url.searchParams.set('project', id);
    else url.searchParams.delete('project');
    url.hash = id ? page : '';
    window.history.pushState({}, '', url);
    useUIStore.getState().resetSelection();
    setProjectId(id);
    window.scrollTo({ top: 0 });
  }
  return <WorkspaceErrorBoundary key={projectId ?? 'start'} onExit={() => navigate(null)}>
    {projectId ? <ExperimentWorkspace projectId={projectId} onExit={() => navigate(null)} />
      : <Start onOpen={(id, page) => navigate(id, page)} />}
  </WorkspaceErrorBoundary>;
}
function ExperimentWorkspace({ projectId, onExit }: { projectId: string; onExit: () => void }) {
  const workspace = useWorkspace(projectId);
  if (!workspace.data) return <main className="connection-state">
    <Icon name="system" size={36} />
    <h1>{workspace.isPending ? 'Connecting to your workspace' : 'Could not open the project'}</h1>
    <p>{workspace.isPending ? 'Loading the project and its saved records.' : 'Check that the Python service is running and the project folder is available.'}</p>
    {workspace.isError ? <><p className="muted">{workspace.error.message}</p><button className="btn btn-primary" disabled={workspace.isFetching} onClick={() => void workspace.refetch()}>Reconnect</button></> : null}
    <button className="btn btn-secondary" onClick={onExit}>Back to start</button>
  </main>;
  return <WorkspaceShell workspace={workspace.data} onExit={onExit} />;
}
function WorkspaceShell({ workspace: w, onExit }: { workspace: Workspace; onExit: () => void }) {
  const roadmap = useRoadmap(w);
  const [page, setPage] = useState(currentPage);
  const [menu, setMenu] = useState(false);
  const [mobile, setMobile] = useState(() => matchMedia('(max-width: 720px)').matches);
  const main = useRef<HTMLElement>(null);
  const menuButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    function route() {
      setPage(currentPage()); setMenu(false); window.scrollTo({ top: 0 });
      main.current?.focus({ preventScroll: true });
    }
    function escape(event: KeyboardEvent) {
      if (event.key === 'Escape' && menu) { setMenu(false); menuButton.current?.focus(); }
    }
    const query = matchMedia('(max-width: 720px)');
    const resize = () => setMobile(query.matches);
    window.addEventListener('hashchange', route);
    window.addEventListener('popstate', route);
    window.addEventListener('keydown', escape);
    query.addEventListener('change', resize);
    return () => {
      window.removeEventListener('hashchange', route); window.removeEventListener('popstate', route);
      window.removeEventListener('keydown', escape); query.removeEventListener('change', resize);
    };
  }, [menu]);
  useEffect(() => { document.title = `HistoPilot · ${pages[page]}`; }, [page]);
  useEffect(() => { if (menu) closeButton.current?.focus(); }, [menu]);
  const module = roadmap.modules.find((item) => item.id === moduleForPage(page));
  const showState = roadmap.hasData;
  const completed = roadmap.modules.filter((item) => item.status === 'complete').length;
  return <>
    <a className="skip-link" href="#main-content" onClick={(event) => { event.preventDefault(); main.current?.focus(); }}>Skip to content</a>
    <div className={`app-shell research-shell ${menu ? 'nav-open' : ''}`}>
      <aside className="sidebar" id="sidebar" aria-label="Workspace navigation" inert={mobile && !menu}>
        <a className="brand" href="#overview"><img src="/favicon.svg" width="35" height="35" alt="" /><span>HistoPilot<small>RESEARCH WORKSPACE</small></span></a>
        <button ref={closeButton} className="icon-button sidebar-close" aria-label="Close navigation" onClick={() => { setMenu(false); menuButton.current?.focus(); }}><Icon name="close" /></button>
        <button type="button" className="project-switch project-switch-button" onClick={onExit} aria-label="Back to start page"><span className="project-icon">{w.project.name.slice(0, 1).toUpperCase()}</span><div><small>{(w.project.lifecycleState ?? 'active').toUpperCase()} PROJECT</small><strong>{w.project.name}</strong></div><Icon name="chevron" size={14} /></button>
        <nav aria-label="Project roadmap and modules">
          <a className={`nav-link roadmap-home-link ${page === 'overview' ? 'active' : ''}`} href="#overview" aria-current={page === 'overview' ? 'page' : undefined}><Icon name="branch" /><span>Project roadmap</span>{showState ? <small>{completed}/{roadmap.modules.length}</small> : null}</a>
          <div className="nav-label">PROJECT MODULES</div>
          {roadmap.modules.map((item) => {
            const locked = !roadmap.checksById[item.id].hasData || !item.unlocked;
            const contents = <><Icon name={moduleIcons[item.id]} size={16} /><span>{item.shortTitle}</span>{showState ? locked ? <Icon name="lock" size={12} /> : <span className={`nav-status status-${item.status}`} aria-label={item.status === 'complete' ? completedModuleLabel(item.id) : item.status === 'draft' ? 'Saved work' : 'Not started'} /> : null}</>;
            return locked ? <div key={item.id} className="nav-link nav-locked" aria-disabled="true" title={showState ? `Requires ${item.blockers.map((id) => roadmap.modules.find((m) => m.id === id)?.shortTitle).join(' + ') || 'compatible frozen inputs'}` : 'Checking prerequisites'}>{contents}</div>
              : <a key={item.id} className={`nav-link ${module?.id === item.id ? 'active' : ''}`} href={`#${item.id}`} aria-current={module?.id === item.id ? 'page' : undefined}>{contents}</a>;
          })}
          <div className="nav-label nav-system">PROJECT TOOLS</div>
          {w.mode === 'local' ? <a className={`nav-link ${page === 'cleanup' ? 'active' : ''}`} href="#cleanup" aria-current={page === 'cleanup' ? 'page' : undefined}><Icon name="folder" size={16} /><span>Workspace cleanup</span></a> : null}
          {w.mode === 'synthetic-demo' ? <a className={`nav-link ${page === 'example-results' ? 'active' : ''}`} href="#example-results" aria-current={page === 'example-results' ? 'page' : undefined}><Icon name="evaluation" size={16} /><span>Illustrative results</span></a> : null}
          {(w.mode === 'local' ? ['system'] as const : ['explorer', 'provenance', 'system'] as const).map((id) => <a key={id} className={`nav-link ${page === id ? 'active' : ''}`} href={`#${id}`} aria-current={page === id ? 'page' : undefined}><Icon name={id} size={16} /><span>{pages[id]}</span></a>)}
        </nav>
        <div className="sidebar-bottom"><div className="prototype-label"><span />Local workspace <b>v0.1</b></div><p>Your data. Your infrastructure.<br />Every step, traceable.</p></div>
      </aside>
      {mobile && menu ? <button className="nav-backdrop" aria-label="Close navigation overlay" onClick={() => setMenu(false)} /> : null}
      <div className="main-shell">
        <header className="topbar"><div className="breadcrumbs"><button ref={menuButton} className="icon-button mobile-menu" aria-label="Toggle navigation" aria-controls="sidebar" aria-expanded={menu} onClick={() => setMenu(!menu)}><Icon name="menu" /></button><button type="button" className="breadcrumb-home" onClick={onExit}>Projects</button><Icon name="chevron" size={12} /><a href="#overview" className="breadcrumb-project">{w.project.name}</a><Icon name="chevron" size={12} /><strong>{page === 'overview' ? 'Roadmap' : module?.shortTitle ?? pages[page]}</strong></div><div className="topbar-actions"><span className="demo-indicator"><span />{w.mode === 'synthetic-demo' ? 'Synthetic demo' : 'Saved locally'}</span><button className="btn btn-secondary btn-small" aria-label="Export workspace" onClick={() => downloadJSON('histopilot-workspace.json', { schema_version: '0.1.0', executable: false, ...w })}><Icon name="download" size={15} /><span>Export</span></button></div></header>
        <main className={`content ${page === 'overview' ? 'roadmap-content' : 'module-content'}`} id="main-content" tabIndex={-1} ref={main}>
          {page !== 'overview' ? <div className="module-context"><a href="#overview"><span className="back-arrow"><Icon name="arrow" size={16} /></span>Back to roadmap</a>{module && showState ? <div><ModuleStatus status={module.status} completedLabel={completedModuleLabel(module.id)} />{!module.unlocked ? <Badge>Locked</Badge> : null}</div> : null}</div> : null}
          {page !== 'overview' && roadmap.error ? <div className="callout callout-warning roadmap-refresh-warning" role="status"><span>Some project progress could not refresh. Your open module remains available.</span><button className="text-button" onClick={() => void roadmap.refetch()}>Retry</button></div> : null}
          <WorkspaceErrorBoundary key={page} onExit={onExit}>
            <Content page={page} workspace={w} roadmap={roadmap} />
          </WorkspaceErrorBoundary>
          {page === 'dataset' || page === 'cohort' ? <JobTray inline projectId={w.mode === 'local' ? w.project.id : undefined} /> : null}
          <footer className="content-footer"><span>HistoPilot <span className="footer-dot">·</span> Interactive PFM–MIL workflows</span><span>{w.mode === 'synthetic-demo' ? 'Synthetic data · Demonstration workspace' : 'Saved in your project folder'}</span></footer>
        </main>
      </div>
    </div>
    {page !== 'dataset' && page !== 'cohort' ? <JobTray projectId={w.mode === 'local' ? w.project.id : undefined} /> : null}
  </>;
}
