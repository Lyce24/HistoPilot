import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import type { InitialConfig, ProjectInput, ProjectSummary } from '../api/types';
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
    ? 'Saved experiment'
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

export default function Start({ onOpen }: { onOpen: (id: string) => void }) {
  const [view, setView] = useState<StartView>('welcome');
  const [name, setName] = useState('');
  const [storageOverride, setStorageOverride] = useState<string | null>(null);
  const [storageParent, setStorageParent] = useState<string | null>(null);
  const [dataPath, setDataPath] = useState('');
  const [slidePath, setSlidePath] = useState('');
  const [featurePath, setFeaturePath] = useState('');
  const [task, setTask] = useState<NonNullable<InitialConfig['task']> | ''>('');
  const [targetColumn, setTargetColumn] = useState('');
  const [positiveLabel, setPositiveLabel] = useState('');
  const [seed, setSeed] = useState('');
  const [folds, setFolds] = useState('');
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
      setValidationError(new Error('Enter an experiment name and its storage folder.'));
      return;
    }
    if (
      seed.trim() &&
      (!/^\d+$/.test(seed.trim()) ||
        !Number.isSafeInteger(Number(seed)) ||
        Number(seed) > 4294967295)
    ) {
      setValidationError(new Error('Choose a whole-number seed between 0 and 4294967295.'));
      return;
    }
    if (folds.trim() && (!/^\d+$/.test(folds.trim()) || Number(folds) < 2 || Number(folds) > 10)) {
      setValidationError(new Error('Choose a whole-number fold count between 2 and 10.'));
      return;
    }
    const config: InitialConfig = {
      ...(task ? { task } : {}),
      ...(targetColumn.trim() ? { targetColumn: targetColumn.trim() } : {}),
      ...(task === 'binary_classification' && positiveLabel.trim()
        ? { positiveLabel: positiveLabel.trim() }
        : {}),
      ...(seed.trim() ? { seed: Number(seed) } : {}),
      ...(folds.trim() ? { folds: Number(folds) } : {}),
    };
    const input: ProjectInput = {
      name: name.trim(),
      storagePath: storagePath.trim(),
      ...(dataPath.trim() ? { dataPath: dataPath.trim() } : {}),
      ...(slidePath.trim() ? { slidePath: slidePath.trim() } : {}),
      ...(featurePath.trim() ? { featurePath: featurePath.trim() } : {}),
      ...(Object.keys(config).length ? { config } : {}),
    };
    create.mutate(input);
  }

  function loadExperiment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setValidationError(null);
    if (!loadPath.trim()) {
      setValidationError(new Error('Choose the folder containing your saved experiment.'));
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
                open.mutate(project.storagePath);
              }}
              disabled={busy || project.available === false}
              aria-label={`Open ${project.name}`}
            >
              <span className="start-recent-icon">
                <Icon name="experiments" size={20} />
              </span>
              <span className="start-recent-copy">
                <strong>{project.name}</strong>
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
          </li>
        ))}
      </ul>
    );
  }

  const title =
    view === 'new'
      ? 'Start a new experiment'
      : view === 'load'
        ? 'Load an existing experiment'
        : 'Your next discovery starts here.';

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
          <img src="/favicon.svg" width="38" height="38" alt="" />
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
            {view === 'welcome' ? 'WELCOME TO HISTOPILOT' : 'YOUR EXPERIMENT WORKSPACE'}
          </span>
          <h1 ref={heading} tabIndex={-1}>
            {title}
          </h1>
          <p>
            {view === 'welcome'
              ? 'Bring your data, shape your experiment, and keep every step in one place.'
              : view === 'new'
                ? 'Give your experiment a home. You can add data and choose the details as you go.'
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
                  <strong>Start a new experiment</strong>
                  <span>Choose a location, connect your data, and make it your own.</span>
                </span>
                <span className="start-action-link">
                  Create experiment <Icon name="arrow" size={18} />
                </span>
              </button>
              <button
                type="button"
                className="start-action-card"
                onClick={() => chooseView('load')}
                disabled={busy}
              >
                <span className="start-action-icon">
                  <Icon name="folder" size={26} />
                </span>
                <span className="start-action-copy">
                  <strong>Load an existing experiment</strong>
                  <span>Open a saved experiment and continue from its overview.</span>
                </span>
                <span className="start-action-link">
                  Find experiment <Icon name="arrow" size={18} />
                </span>
              </button>
            </div>
            <section className="start-recent" aria-labelledby="recent-heading">
              <div className="start-section-heading">
                <h2 id="recent-heading">Recent experiments</h2>
                {savedProjects.length > 3 ? (
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
              {projects.isPending ? (
                <p className="start-list-note" role="status">
                  Loading saved experiments…
                </p>
              ) : null}
              {savedProjects.length ? (
                recentList(savedProjects.slice(0, 3))
              ) : !projects.isPending && !projects.isError ? (
                <p className="start-list-note">Your saved experiments will appear here.</p>
              ) : null}
            </section>
          </>
        ) : null}

        {view === 'new' ? (
          <form className="start-form card" onSubmit={startExperiment} aria-busy={create.isPending}>
            <fieldset disabled={busy} className="start-form-section start-basics">
              <legend className="sr-only">Experiment name and storage</legend>
              <div className="start-field-group">
                <label className="label" htmlFor="experiment-name">
                  Experiment name <span className="start-required">Required</span>
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
                  Experiment storage folder <span className="start-required">Required</span>
                </label>
                <div className="start-path-field">
                  <input
                    id="experiment-storage"
                    className="field mono"
                      placeholder="/path/to/experiments/my-experiment"
                    value={storagePath}
                    onChange={(event) => setStorageOverride(event.target.value)}
                    required
                    maxLength={4096}
                    spellCheck={false}
                    aria-describedby="storage-help"
                  />
                  <ServerFolderPicker
                    title="Choose where to store experiments"
                    purpose="storage"
                    label="Browse storage location"
                    onSelect={(path) => {
                      setStorageParent(path);
                      setStorageOverride(null);
                    }}
                  />
                </div>
                <p id="storage-help" className="start-field-help">
                  Your experiment will be saved in this exact folder on the server. Browse chooses a
                  parent folder and adds your experiment name.
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

            <details className="start-optional">
              <summary>
                <span>
                  <Icon name="experiments" size={19} /> Primary configuration{' '}
                  <Badge>Optional</Badge>
                </span>
                <Icon name="down" size={17} />
              </summary>
              <fieldset disabled={busy} className="start-form-section start-optional-body">
                <legend className="sr-only">Optional primary configuration</legend>
                <p className="start-section-description">
                  Save any choices you already know. Every field can be left blank and configured in
                  the workspace.
                </p>
                <div className="start-config-grid">
                  <label className="label" htmlFor="experiment-task">
                    Prediction task
                    <select
                      id="experiment-task"
                      className="field"
                      value={task}
                      onChange={(event) => setTask(event.target.value as typeof task)}
                    >
                      <option value="">Choose later</option>
                      <option value="binary_classification">Binary classification</option>
                      <option value="multiclass_classification">Multiclass classification</option>
                    </select>
                  </label>
                  <label className="label" htmlFor="experiment-target">
                    Target column
                    <input
                      id="experiment-target"
                      className="field"
                      value={targetColumn}
                      onChange={(event) => setTargetColumn(event.target.value)}
                          placeholder="e.g. grade or outcome"
                      maxLength={128}
                    />
                  </label>
                  {task === 'binary_classification' ? (
                    <label className="label" htmlFor="experiment-positive">
                      Positive label
                      <input
                        id="experiment-positive"
                        className="field"
                        value={positiveLabel}
                        onChange={(event) => setPositiveLabel(event.target.value)}
                        placeholder="e.g. high or mutated"
                        maxLength={128}
                        aria-describedby="positive-help"
                      />
                      <small id="positive-help">For a binary prediction task</small>
                    </label>
                  ) : null}
                  <label className="label" htmlFor="experiment-seed">
                    Random seed
                    <input
                      id="experiment-seed"
                      className="field"
                      type="number"
                      value={seed}
                      onChange={(event) => setSeed(event.target.value)}
                      placeholder="Choose later, e.g. 42"
                      min={0}
                      max={4294967295}
                      step={1}
                    />
                  </label>
                  <label className="label" htmlFor="experiment-folds">
                    Cross-validation folds
                    <input
                      id="experiment-folds"
                      className="field"
                      type="number"
                      value={folds}
                      onChange={(event) => setFolds(event.target.value)}
                      placeholder="Choose later, e.g. 5"
                      min={2}
                      max={10}
                      step={1}
                    />
                  </label>
                </div>
              </fieldset>
            </details>

            <div className="start-form-footer">
              <p>
                <Icon name="overview" size={17} /> Continue to your experiment overview
              </p>
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {create.isPending ? 'Creating experiment…' : 'Start experiment'}{' '}
                <Icon name="arrow" size={17} />
              </button>
            </div>
          </form>
        ) : null}

        {view === 'load' ? (
          <div className="start-load-content">
            <section className="start-saved card" aria-labelledby="saved-heading">
              <div className="start-section-heading">
                <h2 id="saved-heading">Saved experiments</h2>
                <Badge>{savedProjects.length} saved</Badge>
              </div>
              {projects.isPending ? (
                <p className="start-list-note" role="status">
                  Loading saved experiments…
                </p>
              ) : null}
              {savedProjects.length ? (
                recentList(savedProjects)
              ) : !projects.isPending && !projects.isError ? (
                <p className="start-list-note">
                  No saved experiments yet. Open an experiment folder below.
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
                Choose the experiment folder you saved previously on the server.
              </p>
              <fieldset disabled={busy} className="start-load-fields">
                <legend className="sr-only">Saved experiment folder</legend>
                <label className="label" htmlFor="load-experiment-path">
                  Experiment folder
                </label>
                <div className="start-path-field">
                  <input
                    id="load-experiment-path"
                    className="field mono"
                    value={loadPath}
                    onChange={(event) => setLoadPath(event.target.value)}
                    placeholder="/path/to/saved-experiment"
                    required
                    maxLength={4096}
                    spellCheck={false}
                  />
                  <ServerFolderPicker
                    onSelect={setLoadPath}
                    purpose="storage"
                    title="Choose a saved experiment folder"
                    label="Browse experiment folders"
                  />
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={busy || !loadPath.trim()}
                >
                  {open.isPending ? 'Opening experiment…' : 'Load experiment'}{' '}
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
              onClick={() => onOpen('synthetic-v1')}
              disabled={busy}
            >
              Open CRC KRAS demo <Icon name="arrow" size={15} />
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
