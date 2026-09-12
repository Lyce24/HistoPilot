import { useId, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { DatasetVersion } from '../api/scientific';
import { datasetVersionLabel, versionLabelText } from '../lib/versionLabels';
import {
  extractionActive,
  normalizeTridentOptions,
  trident,
} from '../api/trident';
import type {
  ExtractionJob,
  ExtractionPreview,
  TridentOption,
  TridentOutputLayout,
} from '../api/trident';
import { DatasetSelect, Findings } from './ScientificUI';
import { Badge, EmptyState, ErrorNotice, Icon, Panel } from './ui';
import ServerFolderPicker from './ServerFolderPicker';
import ExtractionProgress, { extractionModelLabel, extractionStateLabel, extractionTaskLabel } from './ExtractionProgress';
import { StagePage, StageSteps } from './StageWorkflow';
import './TridentExtraction.css';

const stages = [
  { value: 'all', label: 'Full pipeline', description: 'Segment → patch → extract', icon: 'features' },
  { value: 'seg', label: 'Segment tissue', description: 'Tissue masks & contours', icon: 'cohort' },
  { value: 'coords', label: 'Patch coordinates', description: 'Sample tissue regions', icon: 'overview' },
  { value: 'feat', label: 'Extract features', description: 'Encode existing patches', icon: 'experiments' },
];
const basicNames = new Set(['task', 'segmenter', 'patch_encoder', 'mag', 'patch_size']);
const groupLabels = {
  execution: 'Compute & execution',
  slides: 'Slide reading & selection',
  segmentation: 'Tissue segmentation',
  patching: 'Patching & coordinates',
  features: 'Feature extraction',
};
const groupOrder: TridentOption['group'][] = [
  'segmentation', 'patching', 'features', 'slides', 'execution',
];
const optionPickers = {
  wsi_cache: { selection: 'folder', purpose: 'storage', label: 'Browse cache folders' },
  coords_dir: { selection: 'folder', purpose: 'storage', label: 'Browse coordinate folders' },
  custom_list_of_wsis: { selection: 'table', purpose: 'data', label: 'Browse slide lists' },
  patch_encoder_ckpt_path: { selection: 'file', purpose: 'data', label: 'Browse checkpoints' },
} as const;
const jobTone = (job: ExtractionJob) =>
  job.state === 'succeeded'
    ? 'green'
    : job.state === 'failed' || job.state === 'interrupted'
      ? 'red'
      : extractionActive(job)
        ? 'purple'
        : 'neutral';

export default function TridentExtraction({
  workspace: w,
  datasets,
  onAttach,
}: {
  workspace: Workspace;
  datasets: DatasetVersion[];
  onAttach: (input: { datasetId: string; path: string; encoderId?: string; sourceExtractionJobId?: string }) => void;
}) {
  const project = w.project.id;
  const client = useQueryClient();
  const queryKey = ['extractions', project];
  const catalog = useQuery({
    queryKey: [...queryKey, 'catalog'],
    queryFn: () => trident.catalog(project),
    staleTime: 60_000,
  });
  const jobs = useQuery({
    queryKey: [...queryKey, 'jobs'],
    queryFn: () => trident.jobs(project),
    refetchInterval: (query) => query.state.data?.jobs.some(extractionActive) ? 3000 : false,
  });
  const [datasetId, setDatasetId] = useState(w.dataset.id);
  const [outputPath, setOutputPath] = useState(`${w.project.storagePath.replace(/\/$/, '')}/trident`);
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [preview, setPreview] = useState<ExtractionPreview | null>(null);
  const [selectedJob, setSelectedJob] = useState('');
  const [busy, setBusy] = useState<'preview' | 'start' | 'cancel' | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const operationId = useRef<string | null>(null);
  const [selectedPage, setPage] = useState<'settings' | 'review' | 'activity' | null>(null);
  const activeJobs = jobs.data?.jobs.filter(extractionActive) ?? [];
  const page = selectedPage ?? (activeJobs.length ? 'activity' : 'settings');
  const finishedJobs = jobs.data?.jobs.filter((item) => !extractionActive(item)) ?? [];
  const recentJobs = [...activeJobs, ...finishedJobs.slice(0, 1)];
  const historyJobs = finishedJobs.slice(1);
  const selectedId = selectedJob || recentJobs[0]?.id || '';
  const jobDetail = useQuery({
    queryKey: [...queryKey, 'job', selectedId],
    queryFn: () => trident.job(project, selectedId),
    enabled: Boolean(selectedId),
    refetchInterval: (query) => extractionActive(query.state.data) ? 3000 : false,
  });
  const values = { ...catalog.data?.defaults, ...overrides };
  const task = String(values.task ?? 'all');
  const job = jobDetail.data;
  const advanced = catalog.data?.options.filter((option) => !basicNames.has(option.name)) ?? [];
  const datasetById = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  const datasetLabel = (identity: string) => {
    const dataset = datasetById.get(identity);
    return dataset ? datasetVersionLabel(dataset) : versionLabelText({ id: identity }, 'Dataset');
  };

  function invalidatePreview() {
    setPreview(null); setPage('settings');
    setError(null);
    operationId.current = null;
  }
  function updateOption(name: string, value: unknown) {
    setOverrides((current) => ({ ...current, [name]: value }));
    invalidatePreview();
  }
  async function run(action: NonNullable<typeof busy>, work: () => Promise<void>) {
    setBusy(action);
    setError(null);
    try {
      await work();
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The extraction request failed.'));
    } finally {
      setBusy(null);
    }
  }
  function inspect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!catalog.data) return;
    const options = normalizeTridentOptions(catalog.data.options, values);
    void run('preview', async () => {
      const next = await trident.preview(project, { datasetId, outputPath: outputPath.trim(), options });
      setPreview(next);
      operationId.current = null;
      setPage('review');
    });
  }
  async function updateJob(next: ExtractionJob) {
    client.setQueryData([...queryKey, 'job', next.id], next);
    setSelectedJob(next.id);
    await client.invalidateQueries({ queryKey: [...queryKey, 'jobs'] });
  }

  return (
    <div className="trident-workflow">
      <ErrorNotice error={error ?? catalog.error ?? jobs.error ?? jobDetail.error} />
      <StageSteps label="Extraction steps" current={page} disabled={busy !== null} steps={[
        { id: 'settings', title: 'Extraction settings' },
        { id: 'review', title: 'Review extraction', disabled: !preview },
        { id: 'activity', title: 'Runs & outputs', description: `${jobs.data?.jobs.length ?? 0} runs` },
      ]} onChange={(step) => { if (step === 'settings' || step === 'activity' || step === 'review' && preview) setPage(step); }} />
      <StagePage pageKey={page}>
      {page === 'settings' ? <Panel
        title="Extract with TRIDENT"
        subtitle="Extract with a pathology foundation model, review the outputs, then freeze the features alone or together with verified packs."
        actions={<Badge tone="purple">TRIDENT</Badge>}
      >
        {catalog.isPending ? (
          <p className="muted" role="status">Loading TRIDENT models and options…</p>
        ) : catalog.data ? (
          <form onSubmit={inspect}>
            <fieldset className="science-fieldset trident-form" disabled={busy !== null}>
              <div className="trident-inputs">
                <DatasetSelect
                  versions={datasets}
                  value={datasetId}
                  onChange={(value) => { setDatasetId(value); invalidatePreview(); }}
                />
                <div className="trident-output-input">
                  <label className="label">
                    TRIDENT output directory
                    <input
                      className="field mono"
                      value={outputPath}
                      placeholder="/path/to/trident-output"
                      required
                      onChange={(event) => { setOutputPath(event.target.value); invalidatePreview(); }}
                    />
                  </label>
                  <ServerFolderPicker
                    purpose="storage"
                    label="Browse output"
                    onSelect={(path) => { setOutputPath(path); invalidatePreview(); }}
                  />
                </div>
              </div>
              <fieldset className="trident-stage-fieldset">
                <legend>Pipeline stage</legend>
                <div className="trident-stages">
                  {stages.map((stage) => (
                    <label key={stage.value} className={`trident-stage ${task === stage.value ? 'is-selected' : ''}`}>
                      <input
                        type="radio"
                        name="trident-task"
                        value={stage.value}
                        checked={task === stage.value}
                        onChange={() => updateOption('task', stage.value)}
                      />
                      <Icon name={stage.icon} size={20} />
                      <span><strong>{stage.label}</strong><small>{stage.description}</small></span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <div className="trident-basic-options">
                {['patch_encoder', 'segmenter', 'mag', 'patch_size'].map((name) => {
                  const option = catalog.data.options.find((item) => item.name === name);
                  return option ? <OptionField
                    key={name}
                    option={option}
                    value={values[name]}
                    onChange={(value) => updateOption(name, value)}
                    inactive={name === 'patch_encoder' ? task === 'seg' || task === 'coords' : name === 'segmenter' ? task === 'feat' || task === 'coords' : task === 'seg'}
                  /> : null;
                })}
              </div>
              <details className="trident-advanced">
                <summary>
                  <span><Icon name="system" size={17} /> Advanced Options</span>
                  <small>{advanced.length} TRIDENT settings</small>
                </summary>
                <p className="trident-help">Defaults come from the TRIDENT option catalog. Settings apply to the selected stage; segmentation and patch coordinates can be reused between encoders.</p>
                {groupOrder.map((group) => {
                  const options = advanced.filter((option) => option.group === group);
                  return options.length ? (
                    <fieldset key={group} className="trident-option-group">
                      <legend>{groupLabels[group]}</legend>
                      <div className="trident-advanced-grid">
                        {options.map((option) => (
                          <OptionField
                            key={option.name}
                            option={option}
                            value={values[option.name]}
                            onChange={(value) => updateOption(option.name, value)}
                            showFlag
                            outputPath={outputPath}
                          />
                        ))}
                      </div>
                    </fieldset>
                  ) : null;
                })}
                <p className="trident-help">
                  The selected dataset supplies <code>--wsi_dir</code> and the output directory supplies <code>--job_dir</code>.
                  {' '}<a className="text-button" href={catalog.data.source} target="_blank" rel="noreferrer">TRIDENT CLI reference</a>
                </p>
                <button type="button" className="btn btn-secondary btn-small" onClick={() => { setOverrides({}); invalidatePreview(); }}>
                  <Icon name="reset" size={15} /> Reset TRIDENT defaults
                </button>
              </details>
              <div className="trident-review-bar">
                <div>
                  <strong>Review before extraction</strong>
                  <p>Check slide paths, runtime availability, and the exact output layout before launching a job.</p>
                </div>
                <button type="submit" className="btn btn-primary" disabled={!datasetId || !outputPath.trim()}>
                  {busy === 'preview' ? 'Checking extraction…' : 'Preview extraction'} <Icon name="arrow" size={17} />
                </button>
              </div>
            </fieldset>
          </form>
        ) : (
          <button type="button" className="btn btn-secondary" onClick={() => void catalog.refetch()}>Retry loading TRIDENT options</button>
        )}
      </Panel> : null}

      {page === 'review' && preview ? (
        <div className="trident-preview">
          <button type="button" className="btn btn-secondary pfm-back" disabled={busy !== null} onClick={() => setPage('settings')}>← Back to extraction settings</button>
          <Panel
            title="Extraction preflight"
            subtitle={`${preview.slideCount.toLocaleString()} slides · ${stages.find((stage) => stage.value === preview.spec.options.task)?.label ?? 'TRIDENT pipeline'}`}
            actions={<Badge tone={preview.canRun ? 'green' : 'orange'}>{preview.canRun ? 'Ready to launch' : 'Review findings'}</Badge>}
          >
            <dl className="trident-run-settings">
              <div><dt>Dataset version</dt><dd title={preview.spec.datasetId}>{datasetLabel(preview.spec.datasetId)}</dd></div>
            </dl>
            <Findings findings={preview.findings} />
            <OutputLayout layout={preview.outputLayout} />
            <details className="trident-command">
              <summary>Runtime & command</summary>
              <pre className="code-block">{Array.isArray(preview.command) ? preview.command.map(shellDisplay).join(' ') : preview.command}</pre>
              <pre className="code-block">{JSON.stringify(preview.runtime, null, 2)}</pre>
            </details>
            <div className="trident-review-bar">
              <div>
                <strong>Launch on this workstation</strong>
                <p>Jobs run in tmux with persistent logs. Existing TRIDENT outputs are available for resuming work.</p>
              </div>
              <button
                type="button"
                className="btn btn-primary"
                disabled={!preview.canRun || busy !== null}
                onClick={() => void run('start', async () => {
                  operationId.current ??= `extraction:${Array.from(crypto.getRandomValues(new Uint32Array(4)), (value) => value.toString(16)).join('')}`;
                  await updateJob(await trident.start(project, preview.spec, preview.previewHash, operationId.current));
                  setPreview(null);
                  setPage('activity');
                })}
              >
                <Icon name="experiments" size={17} /> {busy === 'start' ? 'Launching…' : 'Start extraction'}
              </button>
            </div>
          </Panel>
        </div>
      ) : null}

      {page === 'activity' ? <div className="trident-activity">
      <Panel
        title="Extraction activity"
        subtitle="Follow active runs. When features are ready, inspect and save them before choosing a pack."
        actions={activeJobs.length ? <Badge tone="purple">{activeJobs.length} active</Badge> : undefined}
      >
        {jobs.isPending ? <p className="muted" role="status">Loading extraction history…</p> : jobs.data?.jobs.length ? (
          <div className="trident-jobs">
            <nav className="trident-job-navigation" aria-label="Extraction jobs">
              <p className="trident-job-group-label">{activeJobs.length ? 'Active & latest runs' : 'Latest run'}</p>
              <div className="trident-job-list">
                {recentJobs.map((item) => (
                  <ExtractionJobButton key={item.id} job={item} selected={selectedId === item.id} datasetLabel={datasetLabel(item.spec.datasetId)} onSelect={() => setSelectedJob(item.id)} />
                ))}
              </div>
              {historyJobs.length ? (
                <details className="trident-job-history">
                  <summary>Earlier runs <span>{historyJobs.length}</span></summary>
                  <div className="trident-job-list">
                    {historyJobs.map((item) => (
                      <ExtractionJobButton key={item.id} job={item} selected={selectedId === item.id} datasetLabel={datasetLabel(item.spec.datasetId)} onSelect={() => setSelectedJob(item.id)} />
                    ))}
                  </div>
                </details>
              ) : null}
            </nav>
            {job ? (
              <JobDetail
                job={job}
                dataset={datasetById.get(job.spec.datasetId)}
                busy={busy !== null}
                onCancel={() => void run('cancel', async () => updateJob(await trident.cancel(project, job.id)))}
                onUseSettings={() => {
                  setDatasetId(job.spec.datasetId);
                  setOutputPath(job.spec.outputPath);
                  setOverrides(job.spec.options);
                  invalidatePreview();
                }}
                onAttach={onAttach}
              />
            ) : <p className="muted" role="status">Loading job details…</p>}
          </div>
        ) : (
          <EmptyState title="No extraction jobs yet" description="Choose extraction settings, then preview your TRIDENT pipeline." />
        )}
      </Panel>
      </div> : null}
      </StagePage>
    </div>
  );
}

function ExtractionJobButton({ job, selected, datasetLabel, onSelect }: {
  job: ExtractionJob;
  selected: boolean;
  datasetLabel: string;
  onSelect: () => void;
}) {
  return (
    <button type="button" className={`trident-job-row ${selected ? 'is-selected' : ''}`} aria-pressed={selected} onClick={onSelect}>
      <span>
        <strong>{extractionTaskLabel(job)}</strong>
        <span className="trident-job-model">{extractionModelLabel(job)}</span>
        <small title={job.spec.datasetId}>Dataset: {datasetLabel}</small>
        {extractionActive(job) && job.progress?.label ? <span className="trident-job-stage">{job.progress.label}</span> : null}
        <small>{new Date(job.createdAt).toLocaleString()}</small>
      </span>
      <Badge tone={jobTone(job)}>{extractionStateLabel[job.state]}</Badge>
    </button>
  );
}

function OptionField({
  option, value, onChange, inactive = false, showFlag = false, outputPath,
}: {
  option: TridentOption;
  value: unknown;
  onChange: (value: unknown) => void;
  inactive?: boolean;
  showFlag?: boolean;
  outputPath?: string;
}) {
  const id = useId();
  const helpId = `${id}-help`;
  const displayed = value ?? option.default ?? '';
  const picker = optionPickers[option.name as keyof typeof optionPickers];
  const outputRoot = outputPath?.trim().replace(/\/+$/, '');
  const help = <small id={helpId}>{option.description}{inactive ? ' Not used by this stage.' : ''}</small>;
  if (option.type === 'boolean') return (
    <label className="science-check trident-toggle">
      <input type="checkbox" checked={Boolean(displayed)} onChange={(event) => onChange(event.target.checked)} aria-describedby={helpId} />
      <span>{option.label}{showFlag ? <code>{option.flag}</code> : null}{help}</span>
    </label>
  );
  return (
    <div className={`trident-option ${inactive ? 'is-inactive' : ''}`}>
      <label className="label" htmlFor={id}>
        <span>{option.label}{showFlag ? <code>{option.flag}</code> : null}</span>
      </label>
      {option.choices ? (
        <select id={id} className="field" value={String(displayed)} onChange={(event) => onChange(event.target.value)} aria-describedby={helpId}>
          {option.default === null ? <option value="">None / automatic</option> : null}
          {option.choices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
        </select>
      ) : (
        <input
          id={id}
          className={`field ${option.type === 'string' || option.type.endsWith('[]') ? 'mono' : ''}`}
          type={option.type === 'integer' || option.type === 'number' ? 'number' : 'text'}
          step={option.type === 'integer' ? 1 : option.type === 'number' ? 'any' : undefined}
          min={option.minimum ?? (option.exclusiveMinimum !== undefined && option.type === 'integer' ? option.exclusiveMinimum + 1 : undefined)}
          max={option.maximum}
          required={!option.nullable && option.default !== null}
          value={Array.isArray(displayed) ? displayed.join(', ') : String(displayed)}
          placeholder={option.type.endsWith('[]') ? 'Comma-separated values' : option.default === null ? 'Automatic / not set' : undefined}
          onChange={(event) => onChange(event.target.value)}
          aria-describedby={helpId}
        />
      )}
      {help}
      {picker ? (
        <div className="trident-path-picker">
          <ServerFolderPicker
            label={picker.label}
            title={picker.selection === 'folder' ? 'Choose a server folder' : 'Choose a server file'}
            selection={picker.selection}
            purpose={picker.purpose}
            initialPath={option.name === 'coords_dir' || option.name === 'wsi_cache' ? outputRoot : undefined}
            onSelect={(path) => {
              if (option.name === 'coords_dir') {
                if (!outputRoot || !path.startsWith(`${outputRoot}/`))
                  throw new Error('Choose a coordinate folder inside the TRIDENT output directory.');
                onChange(path.slice(outputRoot.length + 1));
              } else {
                onChange(path);
              }
            }}
          />
        </div>
      ) : null}
    </div>
  );
}

function shellDisplay(value: string) {
  return /^[a-zA-Z0-9_./:=+-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`;
}

function OutputLayout({ layout }: { layout: TridentOutputLayout }) {
  const paths = [
    ['Output root', layout.jobDir],
    ['Tissue contours', layout.contoursDir],
    ['GeoJSON contours', layout.geojsonDir],
    ['Thumbnails', layout.thumbnailsDir],
    ['Patch coordinates', layout.coordinatePattern],
    [layout.featureKind === 'slide' ? 'Slide features' : 'Patch features', layout.featurePattern],
  ];
  return (
    <div className="trident-layout">
      <div><Icon name="folder" size={18} /><strong>TRIDENT output structure</strong></div>
      <dl>
        {paths.map(([label, path]) => (
          <div key={label}><dt>{label}</dt><dd className="mono">{path}</dd></div>
        ))}
      </dl>
    </div>
  );
}

function JobDetail({
  job, dataset, busy, onCancel, onUseSettings, onAttach,
}: {
  job: ExtractionJob;
  dataset?: DatasetVersion;
  busy: boolean;
  onCancel: () => void;
  onUseSettings: () => void;
  onAttach: (input: { datasetId: string; path: string; encoderId?: string; sourceExtractionJobId?: string }) => void;
}) {
  const layout = job.result?.outputLayout ?? job.outputLayout;
  const featurePath = job.result?.featurePath ?? job.result?.featureDirectory;
  return (
    <div className="trident-job-detail">
      <div className="trident-job-heading">
        <div><h3>{extractionTaskLabel(job)}</h3><p className="muted">Started <time dateTime={job.createdAt}>{new Date(job.createdAt).toLocaleString()}</time></p></div>
        <Badge tone={jobTone(job)}>{extractionStateLabel[job.state]}</Badge>
      </div>
      <dl className="trident-run-settings">
        <div><dt>Dataset version</dt><dd title={job.spec.datasetId}>{dataset ? datasetVersionLabel(dataset) : versionLabelText({ id: job.spec.datasetId }, 'Dataset')}</dd></div>
        <div><dt>{job.spec.options.task === 'seg' ? 'Tissue model' : 'Model'}</dt><dd>{extractionModelLabel(job)}</dd></div>
        {job.spec.options.task !== 'seg' ? <>
          <div><dt>Magnification</dt><dd>{job.spec.options.mag !== undefined ? `${job.spec.options.mag}×` : 'Automatic'}</dd></div>
          <div><dt>Patch size</dt><dd>{job.spec.options.patch_size !== undefined ? `${job.spec.options.patch_size} px` : 'Automatic'}</dd></div>
        </> : null}
      </dl>
      <ExtractionProgress job={job} />
      {job.error ? <div className="callout callout-warning trident-run-error" role="alert"><strong>{extractionActive(job) ? 'Processing notice' : 'Processing stopped'}</strong><p>{job.error}</p><small>Technical details are available in Troubleshooting.</small></div> : null}
      {job.result?.findings?.length ? <Findings findings={job.result.findings} /> : null}
      {job.state !== 'succeeded' && job.result?.missingSlides ? <p className="trident-job-coverage">{job.result.missingSlides.toLocaleString()} slides have missing outputs. Review the run details before attaching features.</p> : null}
      <div className="inline-actions trident-job-actions">
        <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={onUseSettings}><Icon name="reset" size={15} /> {extractionActive(job) || job.state === 'succeeded' ? 'Reuse settings' : 'Review & resume'}</button>
        {extractionActive(job) ? <button type="button" className="btn btn-secondary btn-small" disabled={busy || job.state === 'cancelling'} onClick={onCancel}><Icon name="close" size={15} /> {job.state === 'cancelling' ? 'Cancelling…' : 'Cancel job'}</button> : null}
      </div>
      {job.state === 'succeeded' && featurePath && layout?.featureKind !== 'slide' ? (
        <div className="trident-completion">
          <div><strong>Features are ready to review</strong><p>Inspect coverage, choose whether to include an existing or new pack, then name and freeze the bundle.</p></div>
          <button type="button" className="btn btn-primary btn-small" disabled={busy} onClick={() => onAttach({ datasetId: job.spec.datasetId, path: featurePath, encoderId: String(job.spec.options.patch_encoder || '') || undefined, sourceExtractionJobId: job.id })}>Inspect features for a bundle <Icon name="arrow" size={15} /></button>
        </div>
      ) : null}
      {job.state === 'succeeded' && layout?.featureKind === 'slide' ? <p className="trident-help">Slide embeddings are saved in the output directory. Feature configurations for MIL currently require patch embeddings with coordinates.</p> : null}
      {layout ? <details className="trident-run-details"><summary>Output folders</summary><OutputLayout layout={layout} /></details> : null}
      <details className="trident-run-details">
        <summary>Troubleshooting</summary>
        <dl className="trident-job-paths">
          <div><dt>Job identifier</dt><dd className="mono">{job.id}</dd></div>
          <div><dt>Output directory</dt><dd className="mono">{job.outputPath}</dd></div>
          <div><dt>Persistent log</dt><dd className="mono">{job.logPath || 'Waiting for log file'}</dd></div>
          <div><dt>Reconnect in a terminal</dt><dd className="mono">{job.sessionName ? `tmux attach -t ${job.sessionName}` : 'Waiting for session'}</dd></div>
        </dl>
        <div className="trident-log-heading"><strong>Technical log</strong>{extractionActive(job) ? <span><i /> Updates every 3 seconds</span> : null}</div>
        <pre className="trident-log" aria-label="Extraction log" tabIndex={0}>{job.logs || (extractionActive(job) ? 'Waiting for TRIDENT output…' : 'No log output is available.')}</pre>
      </details>
    </div>
  );
}
