import { useEffect, useId, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Configuration, FeaturePreview, FeatureSpec } from '../api/scientific';
import { ApiError } from '../api/client';
import { featurePackActive, featurePackSpecKey, packing } from '../api/packing';
import type { FeaturePackArtifact, FeaturePackJob, FeaturePackJobs, FeaturePackPreview, FeaturePackSpec, FeaturePackState, FeatureValidationReport } from '../api/packing';
import { configurationVersionLabel } from '../lib/versionLabels';
import { Findings } from './ScientificUI';
import { Badge, ErrorNotice, Icon, Metric } from './ui';
import ServerFolderPicker from './ServerFolderPicker';
import PackFolderExamples from './PackFolderExamples';
import './FeaturePacking.css';

const stateLabel: Record<FeaturePackState, string> = {
  starting: 'Preparing', running: 'Running', cancelling: 'Stopping', succeeded: 'Complete',
  failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted',
};
const stageLabel: Record<string, string> = {
  preparing: 'Preparing files', validating: 'Validating feature contents', validation: 'Validating feature contents',
  hashing: 'Calculating content checksums', checksumming: 'Calculating content checksums', packing: 'Writing training pack',
  verifying: 'Verifying packed contents', 'verifying-output': 'Verifying packed contents', publishing: 'Publishing pack', complete: 'Finalizing result',
  comparing: 'Comparing pack with source features', 'verifying-source': 'Comparing pack with source features',
};
const jobTone = (job: FeaturePackJob) => job.state === 'succeeded' ? 'green'
  : job.state === 'failed' || job.state === 'interrupted' ? 'red' : featurePackActive(job) ? 'purple' : 'neutral';
const dtypeLabel = (dtype: string | string[] | null) => Array.isArray(dtype) ? dtype.join(', ') : dtype ?? 'Unresolved';
const actionLabel = (action: FeaturePackSpec['action']) => action === 'pack' ? 'Training pack' : action === 'attach' ? 'Existing pack verification' : 'Content validation';

export function formatPackBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) return 'Unavailable';
  if (bytes < 1024) return `${bytes.toLocaleString()} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index++; }
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${units[index]}`;
}

export function FeaturePackCoverage({ summary }: { summary: FeaturePreview['summary'] }) {
  return summary.missingSlides > 0 ? (
    <p className="feature-pack-notice" role="note">
      <Icon name="info" size={18} />
      <span>This version contains {summary.matchedSlides.toLocaleString()} of {summary.slideCount.toLocaleString()} dataset slides.
        Validation and packing cover these attached slides only; {summary.missingSlides.toLocaleString()} {summary.missingSlides === 1 ? 'slide still lacks' : 'slides still lack'} features.
        Attach the missing files and freeze a new version to expand coverage.</span>
    </p>
  ) : null;
}

export function FeatureValidationSummary({ report }: { report: FeatureValidationReport | null | undefined }) {
  const valid = report?.valid && report.tensorValidationComplete && report.current !== false;
  return (
    <div className="feature-validation-summary" aria-label="Feature content validation">
      <Badge tone={valid ? 'green' : report ? 'orange' : 'neutral'}>
        {valid ? 'Contents validated' : report?.current === false ? 'Source files changed' : report ? 'Validation needs attention' : 'Headers inspected only'}
      </Badge>
      <p>{valid
        ? `${report.slideCount.toLocaleString()} slides and ${report.totalPatches.toLocaleString()} patches passed full tensor checks. Checksums identify the validated contents.`
        : report?.current === false
          ? 'A source file changed since validation. Inspect the changed files and freeze a new version before reusing them.'
          : 'The feature inventory records file references and headers. Validate every feature and coordinate array before freezing a bundle.'}</p>
      {valid && !report.provenanceComplete ? <p className="muted">Tensor checks passed; encoder or checkpoint provenance remains incomplete. Review the source metadata before comparing models.</p> : null}
      {report?.validatedAt ? <small className="muted">Last check: {new Date(report.validatedAt).toLocaleString()}</small> : null}
      {report?.findings?.length ? <Findings findings={report.findings} /> : null}
    </div>
  );
}

export function FeaturePackProgress({ job }: { job: FeaturePackJob }) {
  const progress = job.progress;
  const working = job.state === 'starting' || job.state === 'running';
  const succeeded = job.state === 'succeeded';
  const percent = progress?.percent !== null && progress?.percent !== undefined && Number.isFinite(progress.percent)
    ? Math.max(0, Math.min(100, progress.percent)) : null;
  const action = job.spec.action === 'pack' ? 'Packing' : job.spec.action === 'attach' ? 'Pack verification' : 'Validation';
  const counts = progress?.completed !== null && progress?.completed !== undefined
    ? `${progress.completed.toLocaleString()}${progress.total !== null && progress.total !== undefined ? ` of ${progress.total.toLocaleString()}` : ''} ${progress.unit}` : null;
  return (
    <section className={`feature-pack-progress feature-pack-progress-${job.state}`} aria-label={`${action} progress`}>
      <div className="feature-pack-progress-heading" aria-live="polite">
        <Icon name={succeeded ? 'check' : working ? 'features' : 'info'} size={22} />
        <div>
          <h4>{succeeded ? job.spec.action === 'pack' ? 'Training pack created' : job.spec.action === 'attach' ? 'Existing pack verified' : 'Content validation complete'
            : job.state === 'cancelling' ? 'Stopping safely'
              : working ? stageLabel[progress?.stage ?? 'preparing'] ?? 'Processing feature files'
                : `${action} ${stateLabel[job.state].toLowerCase()}`}</h4>
          <p>{succeeded ? job.spec.action === 'pack'
            ? 'The pack was built and verified. Add it to this bundle draft, then freeze the bundle to preserve its feature and pack identities.'
            : job.spec.action === 'attach' ? 'Every feature value and coordinate matches this feature version. Add the verified pack to this bundle draft.'
              : 'Content checks completed for this frozen feature inventory.'
            : job.state === 'cancelling' ? 'Waiting for the worker to stop. The job remains visible here.'
              : working ? 'This job runs on the server. You can leave this page and return to its progress.'
                : 'Review the error and run details, then review the settings to start a new job.'}</p>
        </div>
        <Badge tone={jobTone(job)}>{stateLabel[job.state]}</Badge>
      </div>
      {!succeeded ? <>
        <div className="feature-pack-progress-caption"><span>{working ? 'Current stage' : 'Last reported stage progress'}</span><strong>{percent !== null ? `${percent.toLocaleString(undefined, { maximumFractionDigits: 1 })}%` : working ? 'In progress' : 'Stopped'}</strong></div>
        <div className={`feature-pack-track ${working && percent === null ? 'is-indeterminate' : ''}`} role="progressbar" aria-label="Current packing or validation stage" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={counts ?? (working ? 'Waiting for progress' : 'Worker stopped')}>
          <span style={percent !== null ? { width: `${percent}%` } : undefined} />
        </div>
        <div className="feature-pack-progress-caption"><span>{counts ?? 'No counts reported yet'}</span><small>Stage progress; verification and publishing may follow.</small></div>
      </> : null}
      {progress?.currentSlide && !succeeded ? <p className="feature-pack-current">{working ? 'Current slide' : 'Last reported slide'} <strong className="mono">{progress.currentSlide}</strong></p> : null}
      {job.error ? <ErrorNotice error={new Error(job.error)} /> : null}
    </section>
  );
}

export function ExistingPackComparison({ preview }: { preview: FeaturePackPreview }) {
  const pack = preview.packInspection;
  if (!pack) return null;
  const rows = [
    ['Slides', preview.slideCount.toLocaleString(), pack.slideCount.toLocaleString()],
    ['Patch rows', preview.patchCount.toLocaleString(), pack.totalPatches.toLocaleString()],
    ['Feature dimensions', String(preview.dimensions ?? 'Unknown'), String(pack.dimensions)],
    ['Feature dtype', dtypeLabel(preview.sourceDtype), pack.outputDtype],
    ['Feature payload', formatPackBytes(pack.expectedFeatureBytes), formatPackBytes(pack.featureBytes)],
    ['Coordinate payload', formatPackBytes(pack.expectedCoordinateBytes), formatPackBytes(pack.coordinateBytes)],
  ];
  return <div className="stack">
    <Badge tone={preview.matchesFeatures ? 'green' : 'orange'}>{preview.matchesFeatures ? 'Structure matches · full verification pending' : 'Pack differs from source features'}</Badge>
    <div className="table-wrap"><table aria-label="Source features compared with existing pack"><thead><tr><th>Check</th><th>Expected from features</th><th>Existing pack</th></tr></thead><tbody>{rows.map(([name, expected, actual]) => <tr key={name}><th scope="row">{name}</th><td>{expected}</td><td>{actual}</td></tr>)}</tbody></table></div>
    <p className="muted">Pack folder: {formatPackBytes(pack.totalBytes)}. Source HDF5 files: {formatPackBytes(pack.sourceContainerBytes)}. HDF5 metadata and compression affect file size; dense payload sizes and per-slide patch counts are checked separately.</p>
    {pack.missingSlideCount || pack.extraSlideCount || pack.mismatchedSlideCount ? <p className="feature-pack-notice" role="alert">{pack.missingSlideCount.toLocaleString()} missing slides · {pack.extraSlideCount.toLocaleString()} extra slides · {pack.mismatchedSlideCount.toLocaleString()} slides with different patch counts. Choose a matching pack or create a new pack from this feature version.</p> : null}
    {pack.missingSlides.length || pack.extraSlides.length || pack.mismatchedSlides.length ? <details><summary>Slide mismatch examples</summary><p className="muted">Up to 20 examples per mismatch type.</p><ul>{pack.missingSlides.map((id) => <li key={`missing-${id}`}><code>{id}</code>: missing from the pack</li>)}{pack.extraSlides.map((id) => <li key={`extra-${id}`}><code>{id}</code>: absent from the feature version</li>)}{pack.mismatchedSlides.map((slide) => <li key={`count-${slide.slideId}`}><code>{slide.slideId}</code>: {slide.sourcePatches.toLocaleString()} source patches; {slide.packPatches.toLocaleString()} packed patches</li>)}</ul></details> : null}
    <p className="muted">Matching counts and sizes do not prove the values match. Verification reads all source features and packed arrays and compares every feature row and coordinate before this pack can be included in a bundle.</p>
  </div>;
}

/** A structural match or an old receipt alone never authorizes pack inclusion. */
export function canIncludeFeaturePack(artifact: FeaturePackArtifact): boolean {
  return artifact.current === true && artifact.validation.valid
    && artifact.validation.tensorValidationComplete && artifact.validation.current !== false;
}

/** Removing a stale pack remains possible; adding one requires current verification. */
export function nextBundlePackIds(ids: string[], artifact: FeaturePackArtifact): string[] {
  if (ids.includes(artifact.id)) return ids.filter((id) => id !== artifact.id);
  return canIncludeFeaturePack(artifact) ? [...new Set([...ids, artifact.id])] : ids;
}

export function SavedPackChoice({ artifact, included, busy, onToggle }: {
  artifact: FeaturePackArtifact; included: boolean; busy: boolean; onToggle: () => void;
}) {
  const current = canIncludeFeaturePack(artifact);
  const lossy = artifact.preservesSourcePrecision === false
    || (artifact.preservesSourcePrecision === undefined && artifact.dtypePolicy === 'float16' && artifact.sourceDtype !== 'float16');
  return <article className="feature-pack-artifact">
    <div className="feature-pack-artifact-heading">
      <div><Badge tone={current ? 'green' : 'orange'}>{current ? 'Contents verified' : 'Verification required'}</Badge>
        <strong>{artifact.slideCount.toLocaleString()} slides · {artifact.totalPatches.toLocaleString()} patches · {artifact.outputDtype}</strong>
      </div>
      <button type="button" className={`btn ${included ? 'btn-secondary' : 'btn-primary'} science-fit`} aria-pressed={included}
        disabled={busy || (!included && !current)} onClick={onToggle}>
        {included ? 'Remove from bundle' : lossy ? 'Add float16 pack to bundle' : 'Add pack to bundle'}
      </button>
    </div>
    <p className="mono">{artifact.outputPath}</p>
    {lossy ? <p className="feature-pack-notice" role="note">This float16 derivative changes source feature precision. Include it only if you intend to preserve this rounded derivative in the bundle.</p> : null}
    {!current ? <p className="muted">Connect this folder below and verify it against the current features before adding it to a bundle.</p> : null}
    {artifact.findings?.length ? <Findings findings={artifact.findings} /> : null}
    <details className="feature-pack-identity"><summary>Verification record</summary>
      <dl><div><dt>Precision</dt><dd>{lossy ? 'Converted to float16' : 'Source precision preserved'}</dd></div>
        <div><dt>Dimensions</dt><dd>{artifact.dimensions.toLocaleString()}</dd></div>
        <div><dt>Pack ID</dt><dd className="mono">{artifact.id}</dd></div>
        <div><dt>Source checksum</dt><dd className="mono">{artifact.sourceContentHash}</dd></div>
      </dl>
    </details>
  </article>;
}

function FeatureJobDetails({ job }: { job: FeaturePackJob }) {
  return <details className="feature-pack-run-details"><summary>Run details &amp; logs</summary>
    <dl><div><dt>Job ID</dt><dd className="mono">{job.id}</dd></div>
      <div><dt>Session</dt><dd className="mono">{job.sessionName || 'Not assigned'}</dd></div>
      <div><dt>Reconnect</dt><dd className="mono">{job.sessionName ? `tmux attach -t ${job.sessionName}` : 'Not available'}</dd></div>
      <div><dt>Log file</dt><dd className="mono">{job.logPath}</dd></div>
      {job.outputPath ? <div><dt>Pack folder</dt><dd className="mono">{job.outputPath}</dd></div> : null}
    </dl><pre className="code-block" aria-label="Feature job logs">{job.logs || 'No log output yet.'}</pre>
  </details>;
}

function PreviousFeatureJob({ project, summary }: { project: string; summary: FeaturePackJob }) {
  const detail = useQuery({ queryKey: ['feature-packs', project, 'job', summary.id], queryFn: () => packing.job(project, summary.id) });
  return <div className="stack"><ErrorNotice error={detail.error} /><FeaturePackProgress job={detail.data ?? summary} /><FeatureJobDetails job={detail.data ?? summary} /></div>;
}

export default function FeaturePacking({ project, configuration, configurations, onSelectVersion, selectedPackIds, onSelectedPackIdsChange }: {
  project: string;
  configuration: Configuration;
  configurations: Configuration[];
  onSelectVersion: (id: string) => void;
  selectedPackIds: string[];
  onSelectedPackIdsChange: (ids: string[]) => void;
}) {
  const client = useQueryClient();
  const featureSetId = configuration.id;
  const actionId = useId();
  const [chosenAction, setChosenAction] = useState<FeaturePackSpec['action'] | null>(null);
  const [dtype, setDtype] = useState<FeaturePackSpec['dtype']>('preserve');
  const [outputPath, setOutputPath] = useState('');
  const [existingPath, setExistingPath] = useState('');
  const [review, setReview] = useState<{ preview: FeaturePackPreview; inputKey: string } | null>(null);
  const [selectedJob, setSelectedJob] = useState('');
  const [historyJob, setHistoryJob] = useState('');
  const [busy, setBusy] = useState<'preview' | 'start' | 'cancel' | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const operationId = useRef<string | null>(null);
  const jobsRef = useRef<HTMLDivElement>(null);
  const key = ['feature-packs', project];
  const jobs = useQuery({
    queryKey: key,
    queryFn: () => packing.jobs(project),
    refetchInterval: (query) => query.state.data?.jobs.some(featurePackActive) ? 2500 : false,
    refetchIntervalInBackground: true,
  });
  const versionJobs = jobs.data?.jobs.filter((job) => job.featureSetId === featureSetId) ?? [];
  const activeJobs = versionJobs.filter(featurePackActive);
  const completedJobs = versionJobs.filter((job) => !featurePackActive(job));
  const selectedId = activeJobs[0]?.id || selectedJob;
  const selectedSummary = versionJobs.find((job) => job.id === selectedId);
  const detail = useQuery({
    queryKey: [...key, 'job', selectedId], queryFn: () => packing.job(project, selectedId), enabled: Boolean(selectedId),
    refetchInterval: (query) => featurePackActive(query.state.data) || featurePackActive(selectedSummary) ? 2500 : false,
    refetchIntervalInBackground: true,
  });
  const validation = useQuery({
    queryKey: [...key, 'validation', featureSetId], queryFn: () => packing.validation(project, featureSetId),
    refetchInterval: activeJobs.length ? 3000 : false,
  });
  const latestCompletion = completedJobs[0];
  const completionKey = latestCompletion ? `${latestCompletion.id}:${latestCompletion.updatedAt}` : '';
  useEffect(() => {
    if (completionKey) {
      void client.invalidateQueries({ queryKey: ['feature-packs', project, 'validation', featureSetId] });
    }
  }, [client, project, featureSetId, completionKey]);
  const artifacts = jobs.data?.artifacts.filter((artifact) => artifact.featureSetId === featureSetId) ?? [];
  const orderedArtifacts = [...artifacts].sort((left, right) => Number(selectedPackIds.includes(right.id)) - Number(selectedPackIds.includes(left.id)));
  const action = chosenAction ?? (selectedPackIds.length ? 'attach' : 'validate');
  const spec: FeaturePackSpec = {
    featureSetId, action, dtype: action === 'pack' ? dtype : 'preserve',
    outputPath: action === 'pack' ? outputPath.trim() || null : null,
    existingPath: action === 'attach' ? existingPath.trim() || null : null,
  };
  const preview = review?.inputKey === featurePackSpecKey(spec) ? review.preview : null;
  const summary = configuration.manifest.summary as FeaturePreview['summary'];
  const job = detail.data ?? selectedSummary;
  const completedArtifact = job?.state === 'succeeded' && job.result?.artifact
    ? artifacts.find((artifact) => artifact.id === job.result?.artifact?.id) : undefined;
  const otherActive = jobs.data?.jobs.filter((item) => item.featureSetId !== featureSetId && featurePackActive(item)) ?? [];
  const previousJob = completedJobs.find((item) => item.id === historyJob);

  function edit(change: () => void) {
    change(); setReview(null); setError(null); operationId.current = null;
  }
  async function run(next: NonNullable<typeof busy>, work: () => Promise<void>) {
    if (busy) return;
    setBusy(next); setError(null);
    try { await work(); }
    catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The feature job request failed.'));
      if (reason instanceof ApiError && reason.status === 409 && next === 'start') setReview(null);
    } finally { setBusy(null); }
  }
  async function recordJob(result: FeaturePackJob) {
    client.setQueryData<FeaturePackJobs>(key, (current) => current ? { ...current, jobs: [result, ...current.jobs.filter((item) => item.id !== result.id)] } : current);
    client.setQueryData([...key, 'job', result.id], result);
    setSelectedJob(result.id);
    await client.invalidateQueries({ queryKey: key, exact: true });
  }
  function togglePack(artifact: FeaturePackArtifact) {
    if (busy !== null || artifact.featureSetId !== featureSetId) return;
    onSelectedPackIdsChange(nextBundlePackIds(selectedPackIds, artifact));
  }
  function requestPreview() {
    void run('preview', async () => {
      const inputKey = featurePackSpecKey(spec);
      const result = await packing.preview(project, spec);
      operationId.current = `feature-pack:${crypto.randomUUID()}`;
      setReview({ preview: result, inputKey });
    });
  }
  const previewDisabled = busy !== null || !jobs.data || activeJobs.length > 0;
  const attachmentFields = <div className="stack">
    <p className="muted">Choose a HistoPilot or OceanPath v1 pack for this feature version. The source features must remain available for comparison.</p>
    <div className="feature-pack-destination">
      <label className="label">Existing pack folder<input className="field mono" value={existingPath} placeholder="/path/to/mmap/blca" onChange={(event) => edit(() => setExistingPath(event.target.value))} /></label>
      <ServerFolderPicker purpose="data" label="Browse existing pack" title="Choose an existing pack folder" initialPath={existingPath || undefined} onSelect={(path) => edit(() => setExistingPath(path))} />
    </div>
    <PackFolderExamples />
    <ol className="feature-pack-check-steps" aria-label="Existing pack verification steps">
      <li><strong>Check structure</strong><span>Slide IDs, per-slide patch counts, dimensions, precision and payload sizes.</span></li>
      <li><strong>Verify all contents</strong><span>Compare every feature value and coordinate against the original files.</span></li>
      <li><strong>Add to the bundle</strong><span>Include the verified pack before freezing the bundle.</span></li>
    </ol>
    <button type="button" className="btn btn-secondary science-fit" disabled={previewDisabled || !existingPath.trim()} onClick={requestPreview}>{busy === 'preview' ? 'Checking structure…' : 'Check pack compatibility'} <Icon name="arrow" size={16} /></button>
  </div>;

  return <section className="feature-packing stack" aria-label="Feature bundle preparation">
    <div className="feature-pack-section-heading"><div><h3>Prepare the feature bundle</h3><p>Keep the features alone, include verified existing packs, or create a new pack for this bundle.</p></div><Badge>Optional packing</Badge></div>
    <ErrorNotice error={error ?? jobs.error ?? validation.error} />
    <FeaturePackCoverage summary={summary} />
    <section className="feature-pack-bundle-summary" aria-label="Current bundle draft contents">
      <div className="feature-pack-section-heading"><div><h4>Included in this draft</h4><p>{configurationVersionLabel(configuration)} · {summary.matchedSlides.toLocaleString()} slides · {summary.patchCount.toLocaleString()} patches</p></div><Badge>{selectedPackIds.length ? `Features + ${selectedPackIds.length} ${selectedPackIds.length === 1 ? 'pack' : 'packs'}` : 'Features only'}</Badge></div>
      {selectedPackIds.length ? <ul className="feature-pack-included-list">{selectedPackIds.map((id) => {
        const artifact = artifacts.find((item) => item.id === id);
        const current = artifact && canIncludeFeaturePack(artifact);
        return <li key={id}><div><span className="mono">{artifact?.outputPath ?? id}</span><Badge tone={current ? 'green' : jobs.data ? 'orange' : 'neutral'}>{current ? `${artifact.outputDtype} · verified` : jobs.data ? 'Needs verification' : 'Checking pack'}</Badge></div><button type="button" className="btn btn-secondary btn-small" disabled={busy !== null} aria-label={`Remove ${artifact?.outputPath ?? id} from bundle`} onClick={() => onSelectedPackIdsChange(selectedPackIds.filter((item) => item !== id))}>Remove</button></li>;
      })}</ul> : <p className="muted">This draft contains the original feature files without a pack. Full feature validation is required before freezing.</p>}
      <p className="muted">Freezing records these exact feature and pack identities. You can include multiple verified packs.</p>
    </section>

    <fieldset className="science-fieldset stack" disabled={busy !== null}>
      <legend className="feature-pack-choice-legend">Choose what to include</legend>
      <div className="feature-pack-actions" role="group" aria-label="Feature bundle packing options">
        <button type="button" className={`feature-pack-action ${action === 'validate' ? 'is-selected' : ''}`} aria-pressed={action === 'validate'} aria-controls={actionId} onClick={() => edit(() => { setChosenAction('validate'); onSelectedPackIdsChange([]); })}>
          <Icon name="dataset" size={22} /><span><strong>Features only — skip packing</strong><small>Validate features and freeze without a pack</small></span>
        </button>
        <button type="button" className={`feature-pack-action ${action === 'attach' ? 'is-selected' : ''}`} aria-pressed={action === 'attach'} aria-controls={actionId} onClick={() => edit(() => setChosenAction('attach'))}>
          <Icon name="folder" size={22} /><span><strong>Features + existing pack</strong><small>{artifacts.length ? `${artifacts.length} connected ${artifacts.length === 1 ? 'pack' : 'packs'}, or choose another folder` : 'Reuse a verified pack or connect a folder'}</small></span>
        </button>
        <button type="button" className={`feature-pack-action ${action === 'pack' ? 'is-selected' : ''}`} aria-pressed={action === 'pack'} aria-controls={actionId} onClick={() => edit(() => setChosenAction('pack'))}>
          <Icon name="features" size={22} /><span><strong>Features + new pack</strong><small>Choose a destination, then build and verify</small></span>
        </button>
      </div>
      <div id={actionId} className="feature-pack-options stack">
        {action === 'validate' ? <>
          <div className="feature-pack-section-heading"><div><h4>Validate features before freezing</h4><p>Full validation reads every feature and coordinate array and records source checksums. Creating or verifying a pack includes the same source checks.</p></div>
          </div>
          {validation.isPending ? <p className="muted" role="status">Checking saved validation…</p> : <FeatureValidationSummary report={validation.data} />}
          <button type="button" className="btn btn-secondary science-fit" disabled={previewDisabled} onClick={requestPreview}>{busy === 'preview' ? 'Reviewing…' : 'Validate feature contents'} <Icon name="arrow" size={16} /></button>
        </> : action === 'attach' ? <>
          {artifacts.length ? <section className="feature-pack-artifacts stack" aria-label="Packs for this feature version"><div><h4>Packs verified for these features</h4><p className="muted">Add one or more current verified packs to this bundle, or connect another folder below.</p></div>
            {orderedArtifacts.slice(0, 3).map((artifact) => <SavedPackChoice key={artifact.id} artifact={artifact} included={selectedPackIds.includes(artifact.id)} busy={busy !== null} onToggle={() => togglePack(artifact)} />)}
            {orderedArtifacts.length > 3 ? <details className="feature-pack-connect"><summary>More connected packs ({orderedArtifacts.length - 3})</summary><div className="stack">{orderedArtifacts.slice(3).map((artifact) => <SavedPackChoice key={artifact.id} artifact={artifact} included={selectedPackIds.includes(artifact.id)} busy={busy !== null} onToggle={() => togglePack(artifact)} />)}</div></details> : null}
          </section> : null}
          {artifacts.length ? <details className="feature-pack-connect"><summary>Connect another pack folder</summary>{attachmentFields}</details> : <><h4>Connect your existing pack</h4>{attachmentFields}</>}
        </> : <>
          <div><h4>Create a pack from this feature version</h4><p className="muted">HistoPilot validates the source, writes the indexed feature and coordinate arrays, then verifies the finished pack. Labels and splits stay with their saved configurations.</p></div>
          <div className="feature-pack-destination">
            <label className="label">Destination folder<input className="field mono" value={outputPath} placeholder="Choose automatically in project storage" onChange={(event) => edit(() => setOutputPath(event.target.value))} /><small>Choose a new or empty folder, or leave blank to use project storage. The review shows the exact destination and available space.</small></label>
            <ServerFolderPicker purpose="storage" label="Choose pack destination" title="Choose an empty pack output folder" initialPath={outputPath || jobs.data?.defaultOutputRoot} onSelect={(path) => edit(() => setOutputPath(path))} />
          </div>
          <p className="feature-pack-precision"><Badge tone={dtype === 'preserve' ? 'neutral' : 'orange'}>{dtype === 'preserve' ? 'Source precision preserved' : 'Lossy float16 conversion'}</Badge><span>{dtype === 'preserve' ? 'Feature values keep their original dtype.' : 'Float16 rounding changes higher precision feature values.'}</span></p>
          <details className="feature-pack-advanced"><summary>Advanced: feature precision</summary><label className="label">Feature precision<select className="field" value={dtype} onChange={(event) => edit(() => setDtype(event.target.value as FeaturePackSpec['dtype']))}><option value="preserve">Preserve source precision (recommended)</option><option value="float16">Convert to float16 (smaller, lossy)</option></select><small>Overflow fails validation. Original feature files remain in place.</small></label></details>
          <PackFolderExamples />
          <button type="button" className="btn btn-secondary science-fit" disabled={previewDisabled} onClick={requestPreview}>{busy === 'preview' ? 'Reviewing…' : 'Review pack creation'} <Icon name="arrow" size={16} /></button>
        </>}
        {activeJobs.length ? <p className="muted" role="status">A job is already processing this version. Follow its progress below before starting another.</p> : null}
      </div>
    </fieldset>

    {preview ? <section className="feature-pack-review stack" aria-label="Feature job review">
      <div className="feature-pack-section-heading"><div><h4>{action === 'pack' ? 'Review destination and build' : action === 'attach' ? 'Check complete · review the comparison' : 'Review content validation'}</h4><p>{configurationVersionLabel(configuration)} · Encoder: {configuration.manifest.layout?.encoderId || (configuration.manifest.spec as FeatureSpec).encoderId || 'Unspecified'}</p></div><Badge tone={preview.canRun ? 'green' : 'orange'}>{preview.canRun ? action === 'attach' ? 'Ready for full verification' : 'Ready to start' : 'Resolve findings'}</Badge></div>
      {action === 'attach' ? <ExistingPackComparison preview={preview} /> : <div className="science-metrics feature-pack-metrics">
        <Metric label="Attached slides" value={preview.slideCount.toLocaleString()} note={`${summary.slideCount.toLocaleString()} in the dataset`} />
        <Metric label="Patches" value={preview.patchCount.toLocaleString()} note={`${preview.dimensions ?? 'Unknown'} dimensions`} />
        <Metric label="Precision" value={action === 'pack' ? dtypeLabel(preview.outputDtype) : dtypeLabel(preview.sourceDtype)} note={action === 'pack' ? `${dtypeLabel(preview.sourceDtype)} source${preview.spec.dtype === 'float16' && preview.sourceDtype !== preview.outputDtype ? ' · lossy conversion' : ' · preserved'}` : 'No conversion'} />
        {action === 'pack' ? <Metric label="Estimated pack size" value={formatPackBytes(preview.estimatedBytes)} note={`${formatPackBytes(preview.availableBytes)} available`} /> : null}
      </div>}
      {preview.outputPath && action === 'pack' ? <p className="feature-pack-output"><strong>Save pack to</strong><span className="mono">{preview.outputPath}</span></p> : null}
      {preview.existingPath && action === 'attach' ? <p className="feature-pack-output"><strong>Verify existing pack</strong><span className="mono">{preview.existingPath}</span></p> : null}
      <Findings findings={preview.findings} />
      <div className="feature-pack-launch"><p>{action === 'attach' ? 'Structure checks are the first step. The next job reads every feature and coordinate before this pack can be added to the bundle.' : 'Runs on the server. You can leave this page and return to the saved result.'}</p><button type="button" className="btn btn-primary" disabled={busy !== null || !preview.canRun || activeJobs.length > 0} onClick={() => void run('start', async () => {
        if (!operationId.current) return;
        await recordJob(await packing.start(project, preview.spec, preview.previewHash, operationId.current));
        setReview(null); operationId.current = null;
        requestAnimationFrame(() => jobsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
      })}>{busy === 'start' ? 'Starting…' : action === 'pack' ? 'Build and verify pack' : action === 'attach' ? 'Verify all pack contents' : 'Start validation'} <Icon name="arrow" size={16} /></button></div>
    </section> : null}

    <div ref={jobsRef} className="feature-pack-jobs stack">
      {job && (featurePackActive(job) || job.spec.action === action) ? <section className="stack" aria-label="Current feature job"><ErrorNotice error={detail.error} /><FeaturePackProgress job={job} />
        {featurePackActive(job) ? <button type="button" className="btn btn-secondary science-fit" disabled={busy !== null || job.state === 'cancelling'} onClick={() => void run('cancel', async () => { await recordJob(await packing.cancel(project, job.id)); })}>{job.state === 'cancelling' || busy === 'cancel' ? 'Stopping…' : 'Cancel job'}</button> : null}
        {completedArtifact ? <SavedPackChoice artifact={completedArtifact} included={selectedPackIds.includes(completedArtifact.id)} busy={busy !== null} onToggle={() => togglePack(completedArtifact)} /> : null}
        <FeatureJobDetails job={job} />
      </section> : null}
      {completedJobs.length ? <details className="feature-pack-history"><summary>Previous validation &amp; packing jobs ({completedJobs.length})</summary><div className="stack">
        <div className="feature-pack-job-list" aria-label="Saved feature jobs">{completedJobs.map((item) => <button key={item.id} type="button" className={`feature-pack-job ${historyJob === item.id ? 'is-selected' : ''}`} aria-pressed={historyJob === item.id} onClick={() => setHistoryJob(item.id)}><span><strong>{actionLabel(item.spec.action)}</strong><small>{new Date(item.createdAt).toLocaleString()}</small></span><Badge tone={jobTone(item)}>{stateLabel[item.state]}</Badge></button>)}</div>
        {previousJob ? <PreviousFeatureJob key={previousJob.id} project={project} summary={previousJob} /> : <p className="muted">Select a previous job to review its result and logs.</p>}
      </div></details> : null}
      {otherActive.length ? <div className="feature-pack-other-jobs"><span>Active jobs on other versions:</span>{otherActive.map((item) => <button key={item.id} type="button" className="btn btn-secondary btn-small" onClick={() => onSelectVersion(item.featureSetId)}>{configurations.find((version) => version.id === item.featureSetId) ? configurationVersionLabel(configurations.find((version) => version.id === item.featureSetId)!) : item.featureSetId.slice(-8)} · {stateLabel[item.state]}</button>)}</div> : null}
    </div>
  </section>;
}
