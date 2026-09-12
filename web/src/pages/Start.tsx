import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import type { Page, ProjectInput, ProjectSummary } from '../api/types';
import { lifecycleLabel } from '../api/lifecycle';
import type { LifecycleState } from '../api/lifecycle';
import ServerFolderPicker from '../components/ServerFolderPicker';
import { Badge, ErrorNotice, Icon } from '../components/ui';
import '../start.css';

type StartView = 'welcome' | 'new' | 'load';

function folderName(name: string) {
  return (
    name
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N}_-]+/gu, '-')
      .replace(/^-+|-+$/g, '') || 'experiment'
  );
}

function childPath(parent: string, name: string) {
  return parent ? `${parent.replace(/\/+$/, '')}/${folderName(name)}` : '';
}

function savedDate(project: ProjectSummary) {
  const date = new Date(project.updatedAt ?? project.createdAt);
  return Number.isNaN(date.getTime())
    ? 'Saved project'
    : `Saved ${date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}`;
}

function SourcePathField({
  id,
  label,
  value,
  onChange,
  placeholder,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  return (
    <div className="start-field-group">
      <label className="label" htmlFor={id}>
        {label}
      </label>
      <div className="start-path-field">
        <input
          id={id}
          className="field mono"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          maxLength={4096}
          spellCheck={false}
        />
        <ServerFolderPicker
          onSelect={onChange}
          title={`Choose ${label.toLowerCase()}`}
          purpose="data"
          label={`Browse ${label.toLowerCase()}`}
        />
      </div>
    </div>
  );
}

export function savedProjectsByState(projects: ProjectSummary[], state: LifecycleState) {
  return projects.filter((project) => project.mode !== 'synthetic-demo' && (project.lifecycleState ?? 'active') === state);
}

export default function Start({ onOpen }: { onOpen: (id: string, page?: Page) => void }) {
  const [view, setView] = useState<StartView>('welcome');
  const [projectState, setProjectState] = useState<LifecycleState>('active');
  const [name, setName] = useState('');
  const [storageOverride, setStorageOverride] = useState<string | null>(null);
  const [storageParent, setStorageParent] = useState<string | null>(null);
  const [dataPath, setDataPath] = useState('');
  const [slidePath, setSlidePath] = useState('');
  const [featurePath, setFeaturePath] = useState('');
  const [loadPath, setLoadPath] = useState('');
  const [validationError, setValidationError] = useState<Error | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const previousView = useRef(view);
  const client = useQueryClient();
  const projects = useQuery({ queryKey: ['projects'], queryFn: api.projects });
  const create = useMutation({
    mutationFn: api.createProject,
    onSuccess: (project) => {
      void client.invalidateQueries({ queryKey: ['projects'] });
      onOpen(project.id);
    },
  });
  const open = useMutation({
    mutationFn: api.openProject,
    onSuccess: (project) => {
      void client.invalidateQueries({ queryKey: ['projects'] });
      onOpen(project.id);
    },
  });
  const storagePath =
    storageOverride ?? childPath(storageParent ?? projects.data?.defaultStoragePath ?? '', name);
  const savedProjects = (projects.data?.projects ?? []).filter(
    (project) => project.mode !== 'synthetic-demo',
  );
  const filteredProjects = savedProjectsByState(savedProjects, projectState);
  const busy = create.isPending || open.isPending;

  useEffect(() => {
    if (previousView.current !== view) heading.current?.focus();
    previousView.current = view;
  }, [view]);

  function chooseView(next: StartView) {
    setValidationError(null);
    create.reset();
    open.reset();
    setView(next);
  }

  function startExperiment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setValidationError(null);
    create.reset();
    if (!name.trim() || !storagePath.trim()) {
      setValidationError(new Error('Enter a project name and its storage folder.'));
      return;
    }
    const input: ProjectInput = {
      name: name.trim(),
      storagePath: storagePath.trim(),
      ...(dataPath.trim() ? { dataPath: dataPath.trim() } : {}),
      ...(slidePath.trim() ? { slidePath: slidePath.trim() } : {}),
      ...(featurePath.trim() ? { featurePath: featurePath.trim() } : {}),
    };
    create.mutate(input);
  }

  function loadExperiment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setValidationError(null);
    if (!loadPath.trim()) {
      setValidationError(new Error('Choose the folder containing your saved project.'));
      return;
    }
    open.mutate(loadPath.trim());
  }

  function recentList(items: ProjectSummary[]) {
    return (
      <ul className="start-recent-list">
        {items.map((project) => (
          <li key={project.id}>
            <button
              type="button"
              className="start-recent-item"
              onClick={() => {
                setValidationError(null);
                if (project.lifecycleState && project.lifecycleState !== 'active') onOpen(project.id, 'cleanup');
                else open.mutate(project.storagePath);
              }}
              disabled={busy || project.available === false}
              aria-label={`${project.lifecycleState && project.lifecycleState !== 'active' ? 'Manage' : 'Open'} ${project.name}`}
            >
              <span className="start-recent-icon">
                <Icon name="experiments" size={20} />
              </span>
              <span className="start-recent-copy">
                <strong>{project.name}</strong>
                {project.lifecycleState && project.lifecycleState !== 'active' ? <span><Badge>{lifecycleLabel[project.lifecycleState]}</Badge></span> : null}
                <span className="mono">{project.storagePath}</span>
                {project.available === false ? (
                  <span className="start-unavailable">
                    {project.unavailableReason ?? 'Storage folder unavailable'}
                  </span>
                ) : null}
              </span>
              <span className="start-recent-date">
                {open.isPending && open.variables === project.storagePath
                  ? 'Opening…'
                  : savedDate(project)}
              </span>
              <Icon name="arrow" size={17} />
            </button>
            <div className="start-project-manage"><button type="button" className="text-link" disabled={busy || project.available === false} onClick={() => onOpen(project.id, 'cleanup')} aria-label={`Manage cleanup for ${project.name}`}>Manage cleanup</button></div>
          </li>
        ))}
      </ul>
    );
  }

  const title =
    view === 'new'
      ? 'Start a new project'
      : view === 'load'
        ? 'Load an existing project'
        : 'Your next discovery starts here.';
  const projectFilters = <div className="start-project-filters" role="group" aria-label="Project state">{(['active', 'archived', 'trashed'] as const).map((state) => <button key={state} type="button" className={state === projectState ? 'selected' : ''} aria-pressed={state === projectState} disabled={busy} onClick={() => setProjectState(state)}>{lifecycleLabel[state]} <span>{savedProjectsByState(savedProjects, state).length}</span></button>)}</div>;

  return (
    <div className="start-shell">
      <a className="skip-link" href="#start-content">
        Skip to content
      </a>
      <header className="start-topbar">
        <button
          type="button"
          className="start-brand"
          onClick={() => chooseView('welcome')}
          disabled={busy}
          aria-label="HistoPilot start page"
        >
          <img src="/favicon.svg?v=brown-palette-v2" width="38" height="38" alt="" />
          <span>
            HistoPilot<small>PATHOLOGY WORKSPACE</small>
          </span>
        </button>
        <span className="start-server-label">
          <Icon name="system" size={15} /> Local workspace
        </span>
      </header>

      <main id="start-content" className={`start-main start-view-${view}`}>
        {view !== 'welcome' ? (
          <button
            type="button"
            className="start-back"
            onClick={() => chooseView('welcome')}
            disabled={busy}
          >
            <Icon name="arrow" size={16} /> Back to start
          </button>
        ) : null}
        <div className="start-intro">
          <span className="eyebrow">
            {view === 'welcome' ? 'WELCOME TO HISTOPILOT' : 'YOUR PROJECT WORKSPACE'}
          </span>
          <h1 ref={heading} tabIndex={-1}>
            {title}
          </h1>
          <p>
            {view === 'welcome'
              ? 'Bring your data, shape your experiment, and keep every step in one place.'
              : view === 'new'
                ? 'Give your project a home. You can add data and choose the details as you go.'
                : 'Pick up where you left off, with your data references and settings together.'}
          </p>
        </div>

        <ErrorNotice error={validationError ?? create.error ?? open.error} />
        {projects.isError ? (
          <div className="start-service-error">
            <ErrorNotice error={projects.error} />
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => void projects.refetch()}
              disabled={projects.isFetching}
            >
              <Icon name="reset" size={15} />{' '}
              {projects.isFetching ? 'Retrying…' : 'Retry connection'}
            </button>
          </div>
        ) : null}

        {view === 'welcome' ? (
          <>
            <div className="start-actions">
              <button
                type="button"
                className="start-action-card start-action-new"
                onClick={() => chooseView('new')}
                disabled={busy}
              >
                <span className="start-action-icon">
                  <Icon name="plus" size={26} />
                </span>
                <span className="start-action-copy">
                  <strong>Start a new project</strong>
                  <span>Choose a location, connect your data, and make it your own.</span>
                </span>
                <span className="start-action-link">
                  Create project <Icon name="arrow" size={18} />
                </span>
              </button>
              <button
                type="button"
                className="start-action-card start-action-load"
                onClick={() => chooseView('load')}
                disabled={busy}
              >
                <span className="start-action-icon">
                  <Icon name="folder" size={26} />
                </span>
                <span className="start-action-copy">
                  <strong>Load an existing project</strong>
                  <span>Open a saved project and continue from its roadmap.</span>
                </span>
                <span className="start-action-link">
                  Find project <Icon name="arrow" size={18} />
                </span>
              </button>
            </div>
            <section className="start-recent" aria-labelledby="recent-heading">
              <div className="start-section-heading">
                <h2 id="recent-heading">Recent projects</h2>
                {filteredProjects.length > 3 ? (
                  <button
                    type="button"
                    className="text-link"
                    onClick={() => chooseView('load')}
                    disabled={busy}
                  >
                    View all <Icon name="arrow" size={14} />
                  </button>
                ) : null}
              </div>
              {projectFilters}
              {projects.isPending ? (
                <p className="start-list-note" role="status">
                  Loading saved projects…
                </p>
              ) : null}
              {filteredProjects.length ? (
                recentList(filteredProjects.slice(0, 3))
              ) : !projects.isPending && !projects.isError ? (
                <p className="start-list-note">No {lifecycleLabel[projectState].toLowerCase()} projects. Archived projects and Trash can be restored through Manage cleanup.</p>
              ) : null}
            </section>
          </>
        ) : null}

        {view === 'new' ? (
          <form className="start-form card" onSubmit={startExperiment} aria-busy={create.isPending}>
            <fieldset disabled={busy} className="start-form-section start-basics">
              <legend className="sr-only">Project name and storage</legend>
              <div className="start-field-group">
                <label className="label" htmlFor="experiment-name">
                  Project name <span className="start-required">Required</span>
                </label>
                <input
                  id="experiment-name"
                  className="field"
                      placeholder="e.g. Slide classification"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  required
                  maxLength={120}
                  autoComplete="off"
                />
              </div>
              <div className="start-field-group">
                <label className="label" htmlFor="experiment-storage">
                  Project storage folder <span className="start-required">Required</span>
                </label>
                <div className="start-path-field">
                  <input
                    id="experiment-storage"
                    className="field mono"
                      placeholder="/path/to/projects/my-project"
                    value={storagePath}
                    onChange={(event) => setStorageOverride(event.target.value)}
                    required
                    maxLength={4096}
                    spellCheck={false}
                    aria-describedby="storage-help"
                  />
                  <ServerFolderPicker
                    title="Choose where to store projects"
                    purpose="storage"
                    label="Browse storage location"
                    onSelect={(path) => {
                      setStorageParent(path);
                      setStorageOverride(null);
                    }}
                  />
                </div>
                <p id="storage-help" className="start-field-help">
                  Your project will be saved in this exact folder on the server. Browse chooses a
                  parent folder and adds your project name.
                </p>
              </div>
            </fieldset>

            <details className="start-optional" open>
              <summary>
                <span>
                  <Icon name="dataset" size={19} /> Data locations <Badge>Optional</Badge>
                </span>
                <Icon name="down" size={17} />
              </summary>
              <fieldset disabled={busy} className="start-form-section start-optional-body">
                <legend className="sr-only">Optional data locations</legend>
                <p className="start-section-description">
                  Connect folders on the server now, or add them later in Dataset. Your files stay
                  in their original locations.
                </p>
                <SourcePathField
                  id="experiment-data"
                  label="Data folder"
                  value={dataPath}
                  onChange={setDataPath}
                  placeholder="/path/to/metadata-or-dataset"
                />
                <SourcePathField
                  id="experiment-slides"
                  label="Slide folder"
                  value={slidePath}
                  onChange={setSlidePath}
                  placeholder="/path/to/slides"
                />
                <SourcePathField
                  id="experiment-features"
                  label="Feature folder"
                  value={featurePath}
                  onChange={setFeaturePath}
                  placeholder="/path/to/features"
                />
              </fieldset>
            </details>

            <p className="start-section-description">Define targets and development splits in Stage 2. Configure training recipes and seeds in Experiments; prepare test data separately when ready.</p>

            <div className="start-form-footer">
              <p>
                <Icon name="overview" size={17} /> Continue to your project roadmap
              </p>
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {create.isPending ? 'Creating project…' : 'Create project'}{' '}
                <Icon name="arrow" size={17} />
              </button>
            </div>
          </form>
        ) : null}

        {view === 'load' ? (
          <div className="start-load-content">
            <section className="start-saved card" aria-labelledby="saved-heading">
              <div className="start-section-heading">
                <h2 id="saved-heading">Saved projects</h2>
                <Badge>{filteredProjects.length} {lifecycleLabel[projectState].toLowerCase()}</Badge>
              </div>
              {projectFilters}
              {projects.isPending ? (
                <p className="start-list-note" role="status">
                  Loading saved projects…
                </p>
              ) : null}
              {filteredProjects.length ? (
                recentList(filteredProjects)
              ) : !projects.isPending && !projects.isError ? (
                <p className="start-list-note">
                  No {lifecycleLabel[projectState].toLowerCase()} projects in this view. You can open a saved project folder below.
                </p>
              ) : null}
            </section>
            <form
              className="start-load-form card"
              onSubmit={loadExperiment}
              aria-busy={open.isPending}
            >
              <h2>Open from a folder</h2>
              <p className="start-section-description">
                Choose the project folder you saved previously on the server.
              </p>
              <fieldset disabled={busy} className="start-load-fields">
                <legend className="sr-only">Saved project folder</legend>
                <label className="label" htmlFor="load-experiment-path">
                  Project folder
                </label>
                <div className="start-path-field">
                  <input
                    id="load-experiment-path"
                    className="field mono"
                    value={loadPath}
                    onChange={(event) => setLoadPath(event.target.value)}
                    placeholder="/path/to/saved-project"
                    required
                    maxLength={4096}
                    spellCheck={false}
                  />
                  <ServerFolderPicker
                    onSelect={setLoadPath}
                    purpose="storage"
                    title="Choose a saved project folder"
                    label="Browse project folders"
                  />
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={busy || !loadPath.trim()}
                >
                  {open.isPending ? 'Opening project…' : 'Load project'}{' '}
                  <Icon name="arrow" size={17} />
                </button>
              </fieldset>
            </form>
          </div>
        ) : null}

        {view !== 'new' ? (
          <div className="start-demo">
            <span>Explore a sample workflow</span>
            <button
              type="button"
              className="text-link"
              onClick={() => onOpen('blca-demo-v1')}
              disabled={busy}
            >
              Open BLCA demo <Icon name="arrow" size={15} />
            </button>
            <Badge>Synthetic data</Badge>
          </div>
        ) : null}
      </main>
      <footer className="start-footer">
        <Icon name="provenance" size={15} /> From tissue to evidence. One experiment at a time.
      </footer>
    </div>
  );
}
