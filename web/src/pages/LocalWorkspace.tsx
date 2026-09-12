import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { InitialConfig, Page, Source, Workspace } from '../api/types';
import type { ProtocolSpec } from '../api/scientific';
import { api } from '../api/client';
import { workspaceKey } from '../api/queries';
import ServerFolderPicker from '../components/ServerFolderPicker';
import { useConfigurations, useDatasets } from '../components/ScientificUI';
import {
  configurationVersionLabel,
  datasetVersionLabel,
  versionLabelText,
} from '../lib/versionLabels';
import {
  Badge,
  EmptyState,
  ErrorNotice,
  Icon,
  Metric,
  PageHeader,
  Panel,
} from '../components/ui';

const stages: { page: Page; name: string; detail: string }[] = [
  { page: 'dataset', name: 'Dataset', detail: 'Slides + tables' },
  { page: 'cohort', name: 'Target & split', detail: 'Labels + patient groups' },
  { page: 'features', name: 'Features', detail: 'PFM embeddings' },
  { page: 'experiments', name: 'MIL experiments', detail: 'Models + validation' },
  { page: 'evaluation', name: 'Evaluation', detail: 'Compare results' },
  { page: 'explorer', name: 'Slide explorer', detail: 'Inspect regions' },
  { page: 'provenance', name: 'Provenance', detail: 'Trace every result' },
];
const taskName = (task: InitialConfig['task']) =>
  task === 'binary_classification'
    ? 'Binary classification'
    : task === 'multiclass_classification'
      ? 'Multiclass classification'
      : 'Choose later';

function Sources({ workspace: w }: { workspace: Workspace }) {
  const [role, setRole] = useState<NonNullable<Source['role']>>('data');
  const [path, setPath] = useState('');
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: () => api.addSource(path.trim(), w.project.id, role),
    onSuccess: async () => {
      setPath('');
      await client.invalidateQueries({ queryKey: [...workspaceKey, w.project.id] });
    },
  });
  return (
    <Panel
      title="Source folders"
      subtitle="Optional locations on the computer running HistoPilot"
    >
      {w.sources.length ? (
        <ul className="detail-list">
          {w.sources.map((source) => (
            <li key={source.id}>
              <div>
                <strong>
                  {source.role === 'slides'
                    ? 'Slides'
                    : source.role === 'features'
                      ? 'Features'
                      : 'Data'}
                </strong>
                <p className="mono local-path">{source.path}</p>
              </div>
              <Badge>Path saved</Badge>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No source paths yet. Add them whenever you are ready.</p>
      )}
      <form
        className="local-source-form"
        onSubmit={(event) => {
          event.preventDefault();
          save.mutate();
        }}
      >
        <div className="grid-2">
          <label className="label">
            Source type
            <select
              className="field"
              value={role}
              onChange={(event) => setRole(event.target.value as typeof role)}
            >
              <option value="data">Data / metadata</option>
              <option value="slides">Slides</option>
              <option value="features">Features</option>
            </select>
          </label>
          <label className="label">
            Server folder path
            <input
              className="field"
              required
              value={path}
              placeholder="/path/to/source-folder"
              onChange={(event) => {
                setPath(event.target.value);
                save.reset();
              }}
            />
          </label>
        </div>
        <ErrorNotice error={save.error} />
        <div className="inline-actions">
          <ServerFolderPicker onSelect={setPath} />
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!path.trim() || save.isPending}
          >
            <Icon name="plus" />
            {save.isPending ? 'Saving…' : 'Save source path'}
          </button>
        </div>
      </form>
      <p className="muted local-note">
        Source folders remain read-only. These paths are saved references; dataset import and
        mapping are available in Dataset. A saved source path is not an imported dataset.
      </p>
    </Panel>
  );
}

export function Settings({ workspace: w }: { workspace: Workspace }) {
  const initial = w.project.config;
  const [task, setTask] = useState(initial.task ?? '');
  const [target, setTarget] = useState(initial.targetColumn ?? '');
  const [positive, setPositive] = useState(initial.positiveLabel ?? '');
  const [seed, setSeed] = useState(initial.seed?.toString() ?? '');
  const [folds, setFolds] = useState(initial.folds?.toString() ?? '');
  const [encoder, setEncoder] = useState(initial.encoderId ?? '');
  const [mil, setMil] = useState(initial.milId ?? '');
  const [message, setMessage] = useState('');
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: (config: InitialConfig) => api.updateProject(w.project.id, { config }),
    onSuccess: async () => {
      setMessage('Initial settings saved to your experiment folder.');
      await Promise.all([
        client.invalidateQueries({ queryKey: [...workspaceKey, w.project.id] }),
        client.invalidateQueries({ queryKey: ['projects'] }),
      ]);
    },
  });
  return (
    <Panel
      title="Initial experiment settings"
      subtitle="Everything here is optional. You can refine these choices as the dataset takes shape."
    >
      <form
        className="stack"
        onChange={() => setMessage('')}
        onSubmit={(event) => {
          event.preventDefault();
          const config: InitialConfig = {};
          if (task) config.task = task as InitialConfig['task'];
          if (target.trim()) config.targetColumn = target.trim();
          if (task === 'binary_classification' && positive.trim())
            config.positiveLabel = positive.trim();
          if (seed !== '') config.seed = Number(seed);
          if (folds !== '') config.folds = Number(folds);
          if (encoder) config.encoderId = encoder;
          if (mil) config.milId = mil;
          save.mutate(config);
        }}
      >
        <div className="grid-2">
          <label className="label">
            Task
            <select
              className="field"
              value={task}
              onChange={(event) => setTask(event.target.value as typeof task)}
            >
              <option value="">Choose later</option>
              <option value="binary_classification">Binary classification</option>
              <option value="multiclass_classification">Multiclass classification</option>
            </select>
          </label>
          <label className="label">
            Target column
            <input
              className="field"
              value={target}
              maxLength={128}
                placeholder="e.g. grade or outcome"
              onChange={(event) => setTarget(event.target.value)}
            />
          </label>
          {task === 'binary_classification' ? (
            <label className="label">
              Positive label
              <input
                className="field"
                value={positive}
                maxLength={128}
                placeholder="e.g. high_grade or Mutant"
                onChange={(event) => setPositive(event.target.value)}
              />
            </label>
          ) : null}
          <label className="label">
            Seed
            <input
              className="field"
              type="number"
              min={0}
              max={4294967295}
              step={1}
              value={seed}
              placeholder="Choose later"
              onChange={(event) => setSeed(event.target.value)}
            />
          </label>
          <label className="label">
            Cross-validation folds
            <input
              className="field"
              type="number"
              min={2}
              max={10}
              step={1}
              value={folds}
              placeholder="Choose later"
              onChange={(event) => setFolds(event.target.value)}
            />
          </label>
          <label className="label">
            Preferred encoder
            <select
              className="field"
              value={encoder}
              onChange={(event) => setEncoder(event.target.value)}
            >
              <option value="">Choose later</option>
              {w.encoders.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label className="label">
            Preferred MIL model
            <select
              className="field"
              value={mil}
              onChange={(event) => setMil(event.target.value)}
            >
              <option value="">Choose later</option>
              {w.milModels.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <ErrorNotice error={save.error} />
        {message ? (
          <p role="status" className="callout">
            {message}
          </p>
        ) : null}
        <div className="inline-actions">
          <button className="btn btn-primary" type="submit" disabled={save.isPending}>
            <Icon name="check" />
            {save.isPending ? 'Saving…' : 'Save initial settings'}
          </button>
          <span className="muted">No model run starts when you save.</span>
        </div>
      </form>
    </Panel>
  );
}

function SavedExperimentInputs({ project }: { project: string }) {
  const datasets = useDatasets(project);
  const protocols = useConfigurations(project, 'protocol');
  const features = useConfigurations(project, 'feature');
  const datasetById = new Map(datasets.data?.datasets.map((item) => [item.id, item]) ?? []);
  const featureById = new Map(features.data?.configurations.map((item) => [item.id, item]) ?? []);
  const datasetLabel = (identity: string) => {
    const dataset = datasetById.get(identity);
    return dataset ? datasetVersionLabel(dataset) : versionLabelText({ id: identity }, 'Dataset');
  };
  const featureLabel = (identity: string | null | undefined) => {
    if (!identity) return 'Not selected';
    const feature = featureById.get(identity);
    return feature
      ? configurationVersionLabel(feature)
      : versionLabelText({ id: identity }, 'Features');
  };
  const loading = datasets.isPending || protocols.isPending || features.isPending;

  return (
    <Panel
      title="Saved inputs for MIL experiments"
      subtitle="Use version tags and notes to identify the datasets, cohorts and features behind your analysis."
      actions={<Badge>Planning only</Badge>}
    >
      <ErrorNotice error={datasets.error ?? protocols.error ?? features.error} />
      {loading ? <p className="muted" role="status">Loading saved versions…</p> : (
        <div className="stack">
          <section aria-label="Dataset versions">
            <h3>Datasets</h3>
            {datasets.data?.datasets.length ? (
              <ul className="detail-list">
                {datasets.data.datasets.map((dataset) => (
                  <li key={dataset.id}>
                    <div>
                      <a className="text-link" href="#dataset" title={dataset.id}>
                        {datasetVersionLabel(dataset)}
                      </a>
                      {dataset.versionLabel?.note ? <p className="muted">{dataset.versionLabel.note}</p> : null}
                    </div>
                    <span>{dataset.manifest.summary?.slideCount ?? '?'} slides</span>
                  </li>
                ))}
              </ul>
            ) : <p className="muted">Freeze a dataset in <a href="#dataset">Dataset</a> to add an input version.</p>}
          </section>
          <section aria-label="Cohort and protocol versions">
            <h3>Cohorts & protocols</h3>
            <p className="muted">Each protocol preserves its study cohort, target and split together.</p>
            {protocols.data?.configurations.length ? (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Development protocol</th><th>Dataset</th><th>Bound features</th><th>Version note</th></tr></thead>
                  <tbody>
                    {protocols.data.configurations.map((protocol) => {
                      const featureId = (protocol.manifest.spec as ProtocolSpec).featureSetId;
                      return (
                        <tr key={protocol.id}>
                          <td><a className="text-link" href="#cohort" title={protocol.id}>{configurationVersionLabel(protocol)}</a></td>
                          <td title={protocol.manifest.datasetId}>{datasetLabel(protocol.manifest.datasetId)}</td>
                          <td title={featureId ?? undefined}>{featureLabel(featureId)}</td>
                          <td>{protocol.versionLabel?.note || 'No note yet'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : <p className="muted">Save a cohort and evaluation design in <a href="#cohort">Target & split</a>.</p>}
          </section>
          <section aria-label="Feature versions">
            <h3>Available features</h3>
            {features.data?.configurations.length ? (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Feature version</th><th>Dataset</th><th>Version note</th></tr></thead>
                  <tbody>
                    {features.data.configurations.map((feature) => (
                      <tr key={feature.id}>
                        <td><a className="text-link" href="#features" title={feature.id}>{configurationVersionLabel(feature)}</a></td>
                        <td title={feature.manifest.datasetId}>{datasetLabel(feature.manifest.datasetId)}</td>
                        <td>{feature.versionLabel?.note || 'No note yet'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p className="muted">Inspect and attach embeddings in <a href="#features">PFM & features</a>.</p>}
          </section>
          <p className="muted">Use these frozen inputs to configure development batches. Supported ABMIL batches can be launched explicitly from Experiments.</p>
        </div>
      )}
    </Panel>
  );
}

export default function LocalWorkspace({
  workspace: w,
  page,
}: {
  workspace: Workspace;
  page: Page;
}) {
  if (page === 'overview')
    return (
      <>
        <PageHeader
          eyebrow="YOUR EXPERIMENT WORKSPACE"
          title={w.project.name}
          description={
            w.project.description ||
            'Your workspace is ready. Add data and refine the experiment at your own pace.'
          }
          actions={
            <a className="btn btn-primary" href="#dataset">
              <Icon name="dataset" />
              Set up dataset
            </a>
          }
        />
        <section className="workspace-banner">
          <div className="workspace-mark">
            <Icon name="folder" size={24} />
          </div>
          <div>
            <div className="inline-actions">
              <h2>Experiment saved</h2>
              <Badge tone="green">Local workspace</Badge>
            </div>
            <p className="mono local-path">{w.project.storagePath}</p>
          </div>
          <a href="#experiments" className="version-link">
            Initial settings <Icon name="chevron" size={14} />
          </a>
        </section>
        <div className="grid-4 metrics">
          <Metric
            label="Source folders"
            value={w.sources.length}
            note="Paths saved with this experiment"
          />
          <Metric
            label="Known patients"
            value={w.dataset.patientCount}
            note={
              w.dataset.fallbackSlideCount
                ? `${w.dataset.fallbackSlideCount} additional Slide ID fallback groups`
                : w.dataset.id
                  ? 'Mapped patient identifiers'
                  : 'Waiting for dataset import'
            }
          />
          <Metric
            label="Slides"
            value={w.dataset.slideCount}
            note={w.dataset.id ? 'Latest frozen dataset' : 'Waiting for slide mapping'}
          />
          <Metric label="Model runs" value={0} note="No models have been trained" />
        </div>
        <section className="workflow-section">
          <div className="section-heading">
            <div>
              <h2>Your experiment workflow</h2>
              <p>One connected workspace, from source data to provenance.</p>
            </div>
          </div>
          <div className="pipeline">
            {stages.map((stage, index) => (
              <a key={stage.page} className="stage stage-neutral" href={`#${stage.page}`}>
                <div className="stage-top">
                  <span>{String(index + 1).padStart(2, '0')}</span>
                  <Icon name={stage.page} size={19} />
                </div>
                <strong>{stage.name}</strong>
                <small>{stage.detail}</small>
                <span className="stage-arrow">
                  <Icon name="arrow" size={15} />
                </span>
              </a>
            ))}
          </div>
        </section>
        <div className="grid-2">
          <Panel title="Experiment setup" subtitle="Optional choices saved at the start">
            <ul className="detail-list">
              <li>
                <span>Task</span>
                <strong>{taskName(w.project.config.task)}</strong>
              </li>
              <li>
                <span>Target column</span>
                <strong>{w.project.config.targetColumn || 'Choose later'}</strong>
              </li>
              <li>
                <span>Seed / folds</span>
                <strong>
                  {w.project.config.seed ?? '—'} / {w.project.config.folds ?? '—'}
                </strong>
              </li>
              <li>
                <span>Prediction unit</span>
                <strong>Patient</strong>
              </li>
            </ul>
            <a className="text-link" href="#experiments">
              Edit initial settings →
            </a>
          </Panel>
          <Panel
            title="Scientific workspace"
            subtitle="Saved records in this experiment folder"
          >
            <ul className="detail-list">
              <li>
                <span>Frozen dataset versions</span>
                <strong>{w.scientificSummary?.datasetCount ?? 0}</strong>
              </li>
              <li>
                <span>Frozen analysis protocols</span>
                <strong>{w.scientificSummary?.protocolCount ?? 0}</strong>
              </li>
              <li>
                <span>Existing feature configurations</span>
                <strong>{w.scientificSummary?.featureCount ?? 0}</strong>
              </li>
            </ul>
            <p className="muted">
              {w.dataset.id
                ? 'Review patient mapping and labels, then preview reproducible assignments.'
                : 'Connect a metadata table and optional slide folder to create your first dataset version.'}
            </p>
            <a className="btn btn-secondary" href={w.dataset.id ? '#cohort' : '#dataset'}>
              {w.dataset.id ? 'Configure target & split' : 'Import Dataset'}{' '}
              <Icon name="arrow" />
            </a>
          </Panel>
        </div>
      </>
    );
  if (page === 'dataset')
    return (
      <>
        <PageHeader
          eyebrow="01 / WORKSPACE"
          title="Dataset workspace"
          description="Keep the source folders for this experiment together. Slides and features remain in their original locations."
        />
        <div className="grid-3 metrics">
          <Metric label="Patients" value={0} note="No dataset imported" />
          <Metric label="Slides" value={0} note="No slides mapped" />
          <Metric
            label="Source folders"
            value={w.sources.length}
            note="Saved to this experiment"
          />
        </div>
        <Sources workspace={w} />
      </>
    );
  if (page === 'experiments')
    return (
      <>
        <PageHeader
          eyebrow="04 / EXPERIMENT DESIGN"
          title="MIL experiments"
          description="Set your initial preferences now, then configure model runs once a dataset and validated cohort are available."
        />
        <Panel
          title="Analysis protocols"
          subtitle="Target labels and patient assignments are configured separately from these initial model preferences."
        >
          <div className="inline-actions">
            <Badge tone="purple">
              {w.scientificSummary?.protocolCount ?? 0} frozen protocols
            </Badge>
            <a className="btn btn-primary" href="#cohort">
              Open Target & split <Icon name="arrow" />
            </a>
          </div>
        </Panel>
        <SavedExperimentInputs project={w.project.id} />
        <Settings workspace={w} />
      </>
    );
  if (page === 'provenance')
    return (
      <>
        <PageHeader
          eyebrow="TRACEABILITY"
          title="Provenance"
          description="Trace saved dataset versions, analysis protocols and feature configurations to their experiment folder."
        />
        <Panel title="Experiment record" subtitle="Saved on the server in your chosen folder">
          <ul className="detail-list">
            <li>
              <span>Experiment</span>
              <strong>{w.project.name}</strong>
            </li>
            <li>
              <span>Storage</span>
              <strong className="mono local-path">{w.project.storagePath}</strong>
            </li>
            <li>
              <span>Created</span>
              <strong>{new Date(w.project.createdAt).toLocaleString()}</strong>
            </li>
          </ul>
          <ul className="detail-list">
            <li>
              <a className="text-link" href="#dataset">
                Dataset versions and source fingerprints
              </a>
              <strong>{w.scientificSummary?.datasetCount ?? 0}</strong>
            </li>
            <li>
              <a className="text-link" href="#cohort">
                Frozen targets and patient assignments
              </a>
              <strong>{w.scientificSummary?.protocolCount ?? 0}</strong>
            </li>
            <li>
              <a className="text-link" href="#features">
                Existing feature configurations
              </a>
              <strong>{w.scientificSummary?.featureCount ?? 0}</strong>
            </li>
          </ul>
          <p className="muted">
            Each scientific page exposes its saved immutable records and provenance. Model
            development also records batch execution, fold checkpoints and assessment predictions.
          </p>
        </Panel>
      </>
    );
  const copy: Partial<Record<Page, [string, string, string]>> = {
    cohort: [
      'Cohort builder',
      'Define the population, target, and patient split for this experiment.',
      'Import a dataset before defining a cohort.',
    ],
    evaluation: [
      'Evaluation',
      'Compare results from your model runs.',
      'No evaluation results yet. Results will appear after a model run.',
    ],
    explorer: [
      'Slide explorer',
      'Inspect slide images and their linked annotations.',
      'Slide files can be linked in Dataset. Pixel viewing and region inspection are not connected yet.',
    ],
  };
  const [title, description, empty] = copy[page] ?? [
    'Workspace',
    'Continue setting up your experiment.',
    'No records yet.',
  ];
  return (
    <>
      <PageHeader eyebrow="EXPERIMENT WORKSPACE" title={title} description={description} />
      <Panel title="Getting started">
        <EmptyState title="Ready for the next stage" description={empty} />
        <a className="btn btn-secondary" href="#dataset">
          Open Dataset <Icon name="arrow" />
        </a>
      </Panel>
    </>
  );
}
