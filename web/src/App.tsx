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
import { OverviewPage, EvaluationPage, ExplorerPage, ProvenancePage } from './pages/Results';
import { downloadJSON } from './lib/download';
import { useUIStore } from './store/ui';
import Start from './pages/Start';
import LocalWorkspace from './pages/LocalWorkspace';
import LocalDataset from './pages/LocalDataset';
import LocalProtocol from './pages/LocalProtocol';
import LocalFeatures from './pages/LocalFeatures';
import LocalExperiments from './pages/LocalExperiments';

const pages: Record<Page, string> = {
  overview: 'Overview',
  dataset: 'Dataset workspace',
  cohort: 'Target & split',
  features: 'PFM & features',
  experiments: 'MIL experiments',
  evaluation: 'Evaluation',
  explorer: 'Slide explorer',
  provenance: 'Provenance',
  system: 'System & storage',
};
function currentPage(): Page {
  const hash = window.location.hash.slice(1);
  return Object.hasOwn(pages, hash) ? (hash as Page) : 'overview';
}
function Content({ page, workspace }: { page: Page; workspace: Workspace }) {
  if (workspace.mode === 'local') {
    if (page === 'dataset') return <LocalDataset workspace={workspace} />;
    if (page === 'cohort') return <LocalProtocol workspace={workspace} />;
    if (page === 'features') return <LocalFeatures workspace={workspace} />;
    if (page === 'experiments') return <LocalExperiments workspace={workspace} />;
    if (page !== 'system') return <LocalWorkspace page={page} workspace={workspace} />;
  }
  switch (page) {
    case 'dataset':
      return <Dataset workspace={workspace} />;
    case 'cohort':
      return <Cohort workspace={workspace} />;
    case 'features':
      return <Features workspace={workspace} />;
    case 'experiments':
      return <Experiments workspace={workspace} />;
    case 'evaluation':
      return <EvaluationPage workspace={workspace} />;
    case 'explorer':
      return <ExplorerPage workspace={workspace} />;
    case 'provenance':
      return <ProvenancePage workspace={workspace} />;
    case 'system':
      return <System />;
    default:
      return <OverviewPage workspace={workspace} />;
  }
}
export default function App() {
  const [projectId, setProjectId] = useState(() =>
    new URL(window.location.href).searchParams.get('experiment'),
  );
  useEffect(() => {
    const restore = () => {
      useUIStore.getState().resetSelection();
      setProjectId(new URL(window.location.href).searchParams.get('experiment'));
    };
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
  }, []);
  useEffect(() => {
    if (!projectId) document.title = 'HistoPilot · Start';
  }, [projectId]);
  function navigate(id: string | null) {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set('experiment', id);
    else url.searchParams.delete('experiment');
    url.hash = id ? 'overview' : '';
    window.history.pushState({}, '', url);
    useUIStore.getState().resetSelection();
    setProjectId(id);
    window.scrollTo({ top: 0 });
  }
  return projectId ? (
    <ExperimentWorkspace key={projectId} projectId={projectId} onExit={() => navigate(null)} />
  ) : (
    <Start onOpen={(id) => navigate(id)} />
  );
}

function ExperimentWorkspace({ projectId, onExit }: { projectId: string; onExit: () => void }) {
  const workspace = useWorkspace(projectId);
  const [page, setPage] = useState(currentPage);
  const [menu, setMenu] = useState(false);
  const [mobile, setMobile] = useState(() => matchMedia('(max-width: 720px)').matches);
  const main = useRef<HTMLElement>(null);
  const menuButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    function route() {
      setPage(currentPage());
      setMenu(false);
      window.scrollTo({ top: 0 });
      main.current?.focus({ preventScroll: true });
    }
    function escape(event: KeyboardEvent) {
      if (event.key === 'Escape' && menu) {
        setMenu(false);
        menuButton.current?.focus();
      }
    }
    const query = matchMedia('(max-width: 720px)');
    const resize = () => setMobile(query.matches);
    window.addEventListener('hashchange', route);
    window.addEventListener('keydown', escape);
    query.addEventListener('change', resize);
    return () => {
      window.removeEventListener('hashchange', route);
      window.removeEventListener('keydown', escape);
      query.removeEventListener('change', resize);
    };
  }, [menu]);
  useEffect(() => {
    document.title = `HistoPilot · ${pages[page]}`;
  }, [page]);
  useEffect(() => {
    if (menu) closeButton.current?.focus();
  }, [menu]);
  const w = workspace.data;
  return (
    <>
      <a
        className="skip-link"
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        Skip to content
      </a>
      <div className={`app-shell ${menu ? 'nav-open' : ''}`}>
        <aside
          className="sidebar"
          id="sidebar"
          aria-label="Workspace navigation"
          inert={mobile && !menu}
        >
          <a className="brand" href="#overview">
            <img src="/favicon.svg" width="37" height="37" alt="" />
            <span>
              HistoPilot<small>PATHOLOGY WORKSPACE</small>
            </span>
          </a>
          <button
            ref={closeButton}
            className="icon-button sidebar-close"
            aria-label="Close navigation"
            onClick={() => {
              setMenu(false);
              menuButton.current?.focus();
            }}
          >
            <Icon name="close" />
          </button>
          <button
            type="button"
            className="project-switch project-switch-button"
            onClick={onExit}
            aria-label="Back to start page"
          >
            <span className="project-icon">
              {w?.project.name.slice(0, 1).toUpperCase() ?? 'H'}
            </span>
            <div>
              <small>ACTIVE EXPERIMENT</small>
              <strong>{w?.project.name ?? 'Local workspace'}</strong>
            </div>
            <Icon name="chevron" size={14} />
          </button>
          <div className="nav-label">WORKSPACE</div>
          <nav>
            {Object.entries(pages).map(([key, title], index) => (
              <div key={key}>
                {index === 7 ? (
                  <div className="nav-label nav-system">TRACEABILITY & SYSTEM</div>
                ) : null}
                <a
                  className={`nav-link ${page === key ? 'active' : ''}`}
                  href={`#${key}`}
                  aria-current={page === key ? 'page' : undefined}
                  onClick={() => setMenu(false)}
                >
                  <Icon name={key} />
                  <span>{title}</span>
                  {key === 'experiments' && w?.drafts.length ? (
                    <span className="nav-count">{w.drafts.length}</span>
                  ) : null}
                </a>
              </div>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <div className="prototype-label">
              <span />
              Local-first <b>v0.1</b>
            </div>
            <p>
              Interactive PFM–MIL workflows
              <br />
              for computational pathology.
            </p>
            <div className="sidebar-footer">
              <span className="avatar">HP</span>
              <div>
                <strong>Research workspace</strong>
                <small>Python service · local storage</small>
              </div>
            </div>
          </div>
        </aside>
        <div className="main-shell">
          <header className="topbar">
            <div className="breadcrumbs">
              <button
                ref={menuButton}
                className="icon-button mobile-menu"
                aria-label="Toggle navigation"
                aria-controls="sidebar"
                aria-expanded={menu}
                onClick={() => setMenu(!menu)}
              >
                <Icon name="menu" />
              </button>
              <button type="button" className="breadcrumb-home" onClick={onExit}>
                Experiments
              </button>
              <Icon name="chevron" size={13} />
              <span className="breadcrumb-project">{w?.project.name ?? 'HistoPilot'}</span>
              <Icon name="chevron" size={13} />
              <strong>{pages[page]}</strong>
            </div>
            <div className="topbar-actions">
              <span className="demo-indicator">
                <span />
                {w?.mode === 'synthetic-demo' ? 'Synthetic demo' : 'Local experiment'}
              </span>
              <button
                className="btn btn-secondary btn-small"
                aria-label="Export workspace"
                disabled={!w || workspace.isError}
                onClick={() =>
                  w &&
                  downloadJSON('histopilot-workspace.json', {
                    schema_version: '0.1.0',
                    executable: false,
                    ...w,
                  })
                }
              >
                <Icon name="download" size={15} />
                <span>Export workspace</span>
              </button>
            </div>
          </header>
          <main className="content" id="main-content" tabIndex={-1} ref={main}>
            {workspace.isPending ? (
              <div className="connection-state">
                <Icon name="system" size={36} />
                <h1>Connecting to your workspace</h1>
                <p>The Python service is loading the project and its saved records.</p>
              </div>
            ) : workspace.isError || !w ? (
              <div className="connection-state" role="alert">
                <Icon name="system" size={36} />
                <Badge tone="amber">Disconnected</Badge>
                <h1>Could not open the experiment</h1>
                <p>
                  Check that the Python service is running and the saved experiment folder is
                  available.
                </p>
                <pre className="code-block">histopilot serve</pre>
                <p className="muted">{workspace.error?.message}</p>
                <button
                  className="btn btn-primary"
                  disabled={workspace.isFetching}
                  onClick={() => void workspace.refetch()}
                >
                  <Icon name="reset" />
                  Reconnect
                </button>
                <button type="button" className="btn btn-secondary" onClick={onExit}>
                  Back to start
                </button>
              </div>
            ) : (
              <Content page={page} workspace={w} />
            )}
            {w && !workspace.isError && (page === 'dataset' || page === 'cohort') ? (
              <JobTray inline />
            ) : null}
            <footer className="content-footer">
              <span>
                HistoPilot <span className="footer-dot">·</span> Interactive PFM–MIL workflows
              </span>
              <span>
                {w?.mode === 'synthetic-demo'
                  ? 'Synthetic data · Demonstration workspace'
                  : 'Saved in your experiment folder · Source files stay in place'}
              </span>
            </footer>
          </main>
        </div>
      </div>
      {w && !workspace.isError && page !== 'dataset' && page !== 'cohort' ? <JobTray /> : null}
    </>
  );
}
