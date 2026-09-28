import { useId, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { ApiError } from '../api/client';
import { scientific, type Condition, type VersionLabelInput } from '../api/scientific';
import { newTargetSplitSpec, targetDefinitionReady, targetSplitMethod, targetSplitPartitionRequest, targetSplitTestingRemainder, targetSplitTrainingTarget, targetSplitTestingIssue, targetSplitUnit, targetSplitWithUnit, TARGET_SPLIT_STEPS, targetSplitSetupLink, targetSplits, type TargetSplit, type TargetSplitDraft, type TargetSplitPreview, type TargetSplitSpec, type TargetSplitUnit } from '../api/targetSplits';
import ConditionEditor from '../components/ConditionEditor';
import PredictionTargetEditor from '../components/PredictionTargetEditor';
import { CohortStats, useProtocolExploration, type ProtocolFieldContext } from '../components/ProtocolExploration';
import { TargetSplitDistributions, TargetSplitPartitionCounts, TargetSplitSelection, TargetSplitSelectionSummary, targetSplitExplorationKey, useTargetSplitExploration } from '../components/TargetSplitExploration';
import FreezeVersionDialog from '../components/FreezeVersionDialog';
import VersionLabelEditor from '../components/VersionLabelEditor';
import { DatasetSelect, Findings, SavedNotice, scienceKey, useConfigurations, useDatasets, useDrafts } from '../components/ScientificUI';
import { StageBackButton, StageContinueButton, StageCreateButton } from '../components/StageActions';
import { StageLibrary, StageLibraryToolbar, StagePage, StageRecordManageButton, StageSteps, useStageLibrary } from '../components/StageWorkflow';
import { Badge, EmptyState, ErrorNotice, Metric, PageHeader, Panel } from '../components/ui';
import { datasetVersionLabel, versionLabelText } from '../lib/versionLabels';
import { readHashParameters } from '../lib/hashRoute';
import { sameJSON } from '../lib/json';
import { conditionFields, describeCondition } from '../lib/conditions';
import { historicalProtocols, inferTargetSettings } from '../lib/protocol';
import { readEditorRecovery, recoveredStep, useEditorRecoveryBackup, type EditorRecovery } from '../lib/editorRecovery';
import { useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import { scientificReviewInvalidated } from '../lib/scientificReview';
import './LocalTargetSplit.css';
import { stageEyebrow } from '../lib/roadmap';

export const targetSplitKey = (project: string) => [...scienceKey(project), 'configurations', 'target-split'];
const methodNames = { random: 'Random split', rules: 'Metadata conditions', imported: 'Predefined partition values' };
const targetReady = (spec: TargetSplitSpec) => !targetSplitTestingIssue(spec) && targetDefinitionReady(spec.target) && (spec.testTarget == null || targetDefinitionReady(spec.testTarget));

export function TargetSplitSummary({ value }: { value: Pick<TargetSplitPreview, 'spec' | 'summary' | 'memberships' | 'partitions'> }) {
  const { spec, summary, memberships } = value;
  const unit = targetSplitUnit(spec);
  const testingTarget = spec.testTarget === undefined ? spec.target : spec.testTarget;
  return <div className="stack target-split-summary">
    <div className="science-metrics">
      <Metric label="Training slides" value={summary.trainingSlides.toLocaleString()} note={unit === 'patient' ? `${summary.trainingPatients.toLocaleString()} patients · ${summary.trainingGroups.toLocaleString()} groups` : undefined} />
      <Metric label="Testing slides" value={summary.testingSlides.toLocaleString()} note={unit === 'patient' ? `${summary.testingPatients.toLocaleString()} patients · ${summary.testingGroups.toLocaleString()} groups` : undefined} />
      <Metric label="Included slides" value={summary.includedSlides.toLocaleString()} />
      <Metric label="Excluded slides" value={summary.excludedSlides.toLocaleString()} />
    </div>
    {summary.selectedTrainingSlides !== undefined && (summary.selectedTrainingSlides !== summary.trainingSlides || summary.selectedTestingSlides !== summary.testingSlides) ? <p className="callout">Before label handling: {summary.selectedTrainingSlides.toLocaleString()} selected training slides and {(summary.selectedTestingSlides ?? summary.testingSlides).toLocaleString()} selected testing slides. Counts above show retained memberships after recorded label exclusions. Distributions below describe the original selections.</p> : null}
    <dl className="target-split-facts">
      <div><dt>Split unit</dt><dd>{unit === 'slide' ? 'Slide' : 'Patient'}</dd></div>
      <div><dt>Training target</dt><dd>{spec.target.field} · {spec.target.unit === 'patient' ? 'Patient labels' : 'Slide labels'}</dd></div>
      <div><dt>Testing target</dt><dd>{testingTarget ? `${testingTarget.field} · ${spec.testTarget === undefined ? 'Same mapping as training' : 'Separate source mapping'}` : 'None · Pure inference'}</dd></div>
      <div><dt>Selection</dt><dd>{methodNames[spec.split.method]}{spec.split.method === 'random' ? ` · ${Math.round(spec.split.testFraction * 100)}% testing · seed ${spec.split.seed}` : spec.split.testRemaining ? ` · testing takes every eligible ${unit === 'slide' ? 'slide' : 'patient group'} outside training` : ''}</dd></div>
      {spec.target.positiveClass ? <div><dt>Positive class</dt><dd>{spec.target.positiveClass}</dd></div> : null}
      <div><dt>Cohort conditions</dt><dd>{spec.eligibility.length ? spec.eligibility.map(describeCondition).join(' AND ') : 'All dataset records'}</dd></div>
    </dl>
    <div className="table-wrap"><table aria-label="Target labels by split"><thead><tr><th>Label</th><th>Training slides</th><th>Testing slides</th><th>All labeled slides</th></tr></thead>
      <tbody>{spec.target.classes.map((label) => <tr key={label}><th scope="row">{label}</th><td>{(summary.trainingClassCounts[label] ?? 0).toLocaleString()}</td><td>{testingTarget ? (summary.testingClassCounts[label] ?? 0).toLocaleString() : 'Not labeled'}</td><td>{(summary.classCounts[label] ?? 0).toLocaleString()}</td></tr>)}</tbody>
    </table></div>
    {value.partitions ? <TargetSplitDistributions partitions={value.partitions} splitUnit={unit} mapped unlabeledTest={testingTarget === null} /> : testingTarget === null ? <p className="callout">The testing set is for pure inference. Its selected slides are retained without reading labels.</p> : null}
    <p className="muted">Training and testing memberships are fixed in this version. {unit === 'slide' ? 'Each slide is selected and assigned independently.' : 'Known patients stay in one set. Groups include any explicitly accepted slide-ID fallback groups.'}</p>
    <details><summary>View assigned slides ({memberships.length.toLocaleString()})</summary><div className="table-wrap target-split-memberships"><table aria-label="Assigned slides"><thead><tr><th>Slide</th>{unit === 'patient' ? <th>Patient</th> : null}<th>Set</th><th>Label</th></tr></thead>
      <tbody>{memberships.slice(0, 100).map((row) => <tr key={row.slideId}><td>{row.slideId}</td>{unit === 'patient' ? <td>{row.patientId ?? 'Slide group'}</td> : null}<td>{row.partition === 'train' ? 'Training' : 'Testing'}</td><td>{row.label ?? 'Not labeled'}</td></tr>)}</tbody>
    </table></div>{memberships.length > 100 ? <p className="muted">Showing the first 100 assignments. The saved version retains every assignment.</p> : null}</details>
  </div>;
}

/**
 * How the achieved split can differ from the requested percentage. The service rounds each
 * balanced group (or the whole cohort) to whole slides or patients separately, and keeps at least
 * one in each set when the group has two or more.
 */
export function randomSplitNote(unit: 'slide' | 'patient', stratified: boolean) {
  if (unit === 'patient') return 'Patient groups stay intact, so exact counts can differ.';
  return stratified
    ? 'Each value of the balanced field is split on its own, rounded to whole slides, with at least one slide in each set, so small groups can shift the overall percentage. The counts below show the actual split.'
    : 'The testing set is rounded to whole slides and keeps at least one slide in each set, so small cohorts can differ from the requested percentage. The counts below show the actual split.';
}

/** Freezing turns the testing set into a test cohort. Older versions, or a freeze whose
 * cohort step failed, create it here; reading a version never creates it. */
export function TargetSplitTestCohort({ project, record }: { project: string; record: TargetSplit }) {
  const client = useQueryClient();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const status = record.testCohort;
  const id = status ? status.id : record.evaluationCohortId ?? null;
  if (status ? !status.required : !id) return null;
  const inference = record.manifest.spec.testTarget === null;
  const kind = inference ? 'inference cohort' : 'evaluation cohort';
  async function create() {
    setPending(true); setError(null);
    try {
      await targetSplits.createTestCohort(project, record.id);
      await Promise.all([
        client.invalidateQueries({ queryKey: [...targetSplitKey(project), record.id] }),
        client.invalidateQueries({ queryKey: ['evaluation-cohorts', project] }),
      ]);
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('The test cohort could not be created.')); }
    finally { setPending(false); }
  }
  return <section className="stack" aria-label="Test cohort">
    <h3>Test cohort</h3>
    {id && status?.state !== 'trashed' ? <p>Freezing this version saved its testing set as an {kind}. <a href={`#${inference ? 'inference' : 'evaluation'}?${new URLSearchParams({ cohort: id })}`}>{inference ? 'Run inference on it' : 'Evaluate models on it'}</a> · <a href="#test-data">Open test cohorts</a></p>
      : id ? <p className="callout callout-warning">The {kind} made from this testing set is in Trash. Restore it to evaluate this testing set.</p>
        : <div className="callout callout-warning" role="status">
          <p>{record.testCohortError ? `This version is frozen, but its testing set could not be saved as an ${kind}: ${record.testCohortError.message}` : `This version's testing set has no ${kind} yet.`}</p>
          <button className="btn btn-secondary btn-small" disabled={pending} onClick={() => void create()}>{pending ? 'Creating test cohort…' : record.testCohortError ? 'Retry test cohort' : 'Create test cohort'}</button>
        </div>}
    <ErrorNotice error={error} />
  </section>;
}

export default function LocalTargetSplit({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const client = useQueryClient();
  const datasets = useDatasets(project);
  const drafts = useDrafts(project);
  // Combined protocols from before Targets & Splits; Setup-derived designs do not count.
  const legacyProtocols = useConfigurations(project, 'protocol');
  const legacyHistory = historicalProtocols(legacyProtocols.data?.configurations ?? [], drafts.data?.drafts ?? []);
  const frozen = useQuery({ queryKey: targetSplitKey(project), queryFn: () => targetSplits.list(project) });
  const [parameters] = useState(readHashParameters);
  const defaultName = `${workspace.project.name} targets and splits`;
  const [recovered] = useState(() => readEditorRecovery<TargetSplitSpec>(project, 'target-split'));
  const [spec, setSpec] = useState<TargetSplitSpec>(() => recovered?.spec ?? newTargetSplitSpec(parameters.get('dataset') ?? '', workspace.project.config.seed ?? 42));
  const [name, setName] = useState(recovered?.name ?? defaultName);
  const [draft, setDraft] = useState<TargetSplitDraft | null>(recovered?.draft ?? null);
  const [view, setView] = useState<'library' | 'editor' | 'detail'>(parameters.get('targetSplit') ? 'detail' : 'library');
  // Review requires a fresh server preview; recovery stops at Prediction Targets.
  const [step, setStep] = useState(recoveredStep(recovered?.step, 3) || 1);
  const [resumeAvailable, setResumeAvailable] = useState(Boolean(recovered));
  const [openedId, setOpenedId] = useState(parameters.get('targetSplit') ?? '');
  const [saved, setSaved] = useState<TargetSplit | null>(null);
  // The detail read is side-effect free and reports the version's testing cohort.
  const selectedRecord = useQuery({ queryKey: [...targetSplitKey(project), openedId], queryFn: () => targetSplits.get(project, openedId), enabled: Boolean(openedId) });
  const record = selectedRecord.data ?? frozen.data?.configurations.find((item) => item.id === openedId) ?? saved;
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('all');
  const [preview, setPreview] = useState<TargetSplitPreview | null>(null);
  const [freezeReview, setFreezeReview] = useState<{ draft: TargetSplitDraft; preview: TargetSplitPreview; operationId: string } | null>(null);
  const [freezeLabel, setFreezeLabel] = useState<VersionLabelInput>({ tag: '', note: '' });
  const [busy, setBusy] = useState(false);
  const pending = useRef(false);
  const targetRequest = useRef(0);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState(recovered ? 'Unsaved targets and splits were recovered in this tab. Return to the current draft to continue.' : '');
  const dataset = datasets.data?.datasets.find((item) => item.id === spec.datasetId);
  const dictionary = dataset?.manifest.dictionary ?? [];
  const fields = [...new Set(['Slide_ID', 'Patient_ID', ...dictionary.map((item) => item.key), ...conditionFields(spec.eligibility), ...conditionFields(spec.split.trainRules), ...conditionFields(spec.split.testRules)])];
  const fieldContext = { project, datasetId: spec.datasetId, dictionary };
  const initial = newTargetSplitSpec(parameters.get('dataset') ?? '', workspace.project.config.seed ?? 42);
  const dirty = draft ? name !== draft.name || !sameJSON(spec, draft.payload.spec) : name !== defaultName || !sameJSON(spec, initial);
  const unsaved = draft?.status !== 'frozen' && dirty;
  const recovery: EditorRecovery<TargetSplitSpec> | null = unsaved ? { version: 1, name, spec, draft, step } : null;
  const backup = useEditorRecoveryBackup(project, 'target-split', recovery);
  useWorkspaceNavigationGuard(busy ? 'A target and split request is still pending. Leaving now may hide its outcome.' : recovery && backup.error ? 'Save this draft before leaving; browser recovery is unavailable.' : null);
  useStageLibrary(() => { if (!pending.current && !freezeReview) { setView('library'); setResumeAvailable(view === 'editor' || resumeAvailable); } });

  // Cohort filtering is independent of every saved split and target setting.
  const cohortLive = useProtocolExploration(project, { datasetId: view === 'editor' && step === 1 ? spec.datasetId : '', cohortOnly: true, eligibility: spec.eligibility, rules: { train: [], val: [], test: [] }, splitMode: 'kfold' });
  const liveRequest = step === 3 ? targetSplitPartitionRequest(spec) : { datasetId: spec.datasetId, ...(spec.splitUnit ? { splitUnit: spec.splitUnit } : {}), eligibility: spec.eligibility, split: spec.split };
  const live = useTargetSplitExploration(project, liveRequest, view === 'editor' && step > 1 && step < 4 && Boolean(dataset));
  const cohortReady = Boolean(cohortLive.data?.valid && cohortLive.data.cohort?.totalSlides);
  const testingTarget = spec.testTarget === undefined ? spec.target : spec.testTarget;
  const unit = targetSplitUnit(spec);
  const labelData = (partition: 'train' | 'test') => {
    const target = live.data?.partitions[partition].target;
    return target ? { valueCounts: target.values.map(({ value, slides }) => ({ value, count: slides })), valuesTruncated: target.distinctCount > target.values.length } : undefined;
  };
  const rawValues = (partition: 'train' | 'test') => (labelData(partition)?.valueCounts ?? []).map((item) => item.value).filter((value): value is string => value !== null && value.trim() !== '');

  async function run(action: () => Promise<void>) {
    if (pending.current) return;
    pending.current = true; setBusy(true); setError(null); setMessage('');
    try { await action(); } catch (reason) { setError(reason instanceof Error ? reason : new Error('The target and split request failed.')); }
    finally { pending.current = false; setBusy(false); }
  }
  function edit(update: Partial<TargetSplitSpec>) {
    targetRequest.current += 1;
    setSpec((current) => ({ ...current, ...update })); setPreview(null); setError(null); setMessage('');
  }
  function editUnit(value: TargetSplitUnit) { edit(targetSplitWithUnit(spec, value)); }
  function editSplit(update: Partial<TargetSplitSpec['split']>) { edit({ split: { ...spec.split, ...update } }); }
  function editTrainingTarget(update: Partial<TargetSplitSpec['target']>) { edit(targetSplitTrainingTarget(spec, { ...spec.target, ...update })); }
  async function chooseTarget(field: string, partition: 'train' | 'test' = 'train') {
    const target = partition === 'train' ? { ...spec.target, field, ...inferTargetSettings([]) }
      : { ...spec.target, ...(spec.testTarget ?? {}), field, labels: {}, task: spec.target.task, classes: spec.target.classes, unit: spec.target.unit, positiveClass: spec.target.positiveClass };
    const update = partition === 'train' ? targetSplitTrainingTarget(spec, target) : { testTarget: target };
    edit(update);
    const requestId = targetRequest.current;
    if (!field) return;
    const request = targetSplitPartitionRequest({ ...spec, ...update }, undefined, true);
    try {
      const result = await client.fetchQuery({ queryKey: targetSplitExplorationKey(project, request), queryFn: () => targetSplits.partitionPreview(project, request) });
      if (requestId !== targetRequest.current) return;
      if (!result.valid) throw new Error(result.findings.find((finding) => finding.severity === 'error')?.message ?? 'Review partition conditions before choosing a target.');
      const distribution = result.partitions[partition].target;
      const values = distribution?.values ?? [];
      setSpec((current) => partition === 'train'
        ? { ...current, ...targetSplitTrainingTarget(current, { ...current.target, ...inferTargetSettings(values.map((item) => item.value), (distribution?.distinctCount ?? 0) > values.length) }) }
        : { ...current, testTarget: { ...current.testTarget!, labels: Object.fromEntries(values.filter((item) => item.value !== null && item.value.trim()).map((item) => [item.value!, current.target.classes.includes(item.value!) ? item.value! : ''])) } });
    } catch (reason) { if (requestId === targetRequest.current) setError(reason instanceof Error ? reason : new Error('Target values could not be loaded.')); }
  }

  async function save(): Promise<TargetSplitDraft> {
    if (!name.trim()) throw new Error('Name this target and split draft before saving.');
    if (draft && !dirty) return draft;
    const next = await scientific.saveDraft(project, { kind: 'experiment', name: name.trim(), payload: { type: 'target-split', spec } }, draft?.status === 'editable' ? draft : undefined);
    setDraft(next); setName(next.name);
    client.setQueryData<{ drafts: TargetSplitDraft[] }>([...scienceKey(project), 'drafts'], (current) => ({ drafts: [next, ...(current?.drafts ?? []).filter((item) => item.id !== next.id)] }));
    return next;
  }
  async function keepCurrent() { if (unsaved) await save(); }
  function openLibrary() { setResumeAvailable(view === 'editor' || resumeAvailable); setView('library'); }
  async function create() {
    await keepCurrent();
    targetRequest.current += 1;
    setSpec(newTargetSplitSpec(parameters.get('dataset') ?? (datasets.data?.datasets.length === 1 ? datasets.data.datasets[0].id : ''), workspace.project.config.seed ?? 42));
    setName(defaultName); setDraft(null); setPreview(null); setSaved(null); setOpenedId(''); setStep(1); setView('editor'); setResumeAvailable(true); setFreezeLabel({ tag: '', note: '' });
  }
  async function openDraft(id: string) {
    await keepCurrent();
    const next = await scientific.draft<TargetSplitSpec>(project, id);
    targetRequest.current += 1;
    setDraft(next); setSpec(next.payload.spec); setName(next.name); setPreview(null); setStep(1); setView('editor'); setResumeAvailable(true); setFreezeLabel({ tag: '', note: '' });
  }
  // The library row lacks testing-cohort status: show it at once and let the detail read fill it in.
  async function openRecord(item: TargetSplit) { await keepCurrent(); targetRequest.current += 1; setSaved(item); setOpenedId(item.id); setView('detail'); }
  async function check() {
    const current = await save();
    const checked = await targetSplits.preview(project, current.id, current.revision);
    setPreview(checked); setStep(4);
  }
  async function copyRecord() {
    if (!record) return;
    await keepCurrent();
    targetRequest.current += 1;
    setSpec(structuredClone(record.manifest.spec)); setName(`${versionLabelText(record, 'Targets and splits')} copy`); setDraft(null); setPreview(null); setSaved(null); setOpenedId(''); setStep(1); setView('editor'); setResumeAvailable(true); setFreezeLabel({ tag: '', note: '' });
  }
  const frozenRows = (frozen.data?.configurations ?? []).map((item) => ({ id: item.id, name: versionLabelText(item, 'Targets and splits'), status: 'frozen', spec: item.manifest.spec, summary: item.manifest.summary, updatedAt: item.versionLabel?.updatedAt ?? item.createdAt, item }));
  const draftRows = (drafts.data?.drafts ?? []).filter((item) => item.payload.type === 'target-split' && item.status !== 'frozen').map((item) => ({ id: item.id, name: item.name, status: 'draft', spec: item.payload.spec as unknown as TargetSplitSpec, summary: null, updatedAt: item.updatedAt, item: null }));
  const rows = [...frozenRows, ...draftRows].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  const visible = rows.filter((row) => (status === 'all' || row.status === status) && `${row.name} ${row.spec.target.field} ${row.spec.target.classes.join(' ')}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));
  const datasetName = (id: string) => { const item = datasets.data?.datasets.find((value) => value.id === id); return item ? datasetVersionLabel(item) : versionLabelText({ id }, 'Dataset'); };
  const loading = datasets.isPending || drafts.isPending || frozen.isPending;

  return <div className="clinical-workspace scientific-page target-split-workspace">
    <PageHeader eyebrow={stageEyebrow('cohort')} title="Targets & splits" description="Choose dataset records, split training and testing, then define each prediction target."
      actions={view === 'library' ? <StageCreateButton disabled={busy} onClick={() => void run(create)}>Create targets &amp; splits</StageCreateButton> : <StageBackButton disabled={busy} onClick={openLibrary}>Back to targets &amp; splits</StageBackButton>} />
    <ErrorNotice error={error ?? datasets.error ?? drafts.error ?? frozen.error ?? selectedRecord.error} />
    <SavedNotice>{message}</SavedNotice>
    <StagePage pageKey={`${view}:${step}`}>
      {view === 'library' ? <>
        <StageLibrary project={project} title="Saved targets and splits">
          <StageLibraryToolbar search={search} onSearch={setSearch} searchLabel="Search targets and splits" count={visible.length} total={rows.length}
            onReset={search || status !== 'all' ? () => { setSearch(''); setStatus('all'); } : undefined}
            actions={<>{resumeAvailable ? <button className="btn btn-secondary btn-small" disabled={busy} onClick={() => setView('editor')}>Return to current draft</button> : null}<button className="btn btn-secondary btn-small" disabled={busy || loading} onClick={() => void run(async () => { await Promise.all([datasets.refetch(), drafts.refetch(), frozen.refetch()]); })}>Refresh</button></>}>
            <label className="label">Status<select className="field" value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All statuses</option><option value="frozen">Frozen</option><option value="draft">Draft</option></select></label>
          </StageLibraryToolbar>
          {loading ? <p className="muted" role="status">Loading saved targets and splits…</p> : visible.length ? <div className="table-wrap"><table aria-label="Targets and splits library"><thead><tr><th>Name</th><th>Status</th><th>Dataset</th><th>Training</th><th>Testing</th><th>Target</th><th>Labels</th><th>Updated</th><th>Actions</th></tr></thead><tbody>{visible.map((row) => <tr key={row.id}>
            <td><strong>{row.name}</strong><small className="science-block">{targetSplitUnit(row.spec) === 'slide' ? 'Slide split' : 'Patient split'}</small></td><td><Badge tone={row.status === 'frozen' ? 'green' : 'neutral'}>{row.status === 'frozen' ? 'Frozen' : 'Draft'}</Badge></td><td>{datasetName(row.spec.datasetId)}</td>
            <td>{row.summary ? `${row.summary.trainingSlides.toLocaleString()} slides` : 'Not frozen'}</td><td>{row.summary ? `${row.summary.testingSlides.toLocaleString()} slides` : 'Not frozen'}</td><td>{row.spec.target.field || 'Not selected'}{row.spec.testTarget === null ? <small className="science-block">Testing: None · inference</small> : row.spec.testTarget ? <small className="science-block">Testing: {row.spec.testTarget.field || 'Not selected'}</small> : null}</td><td>{row.spec.target.classes.join(', ') || 'Not defined'}</td><td>{row.updatedAt ? new Date(row.updatedAt).toLocaleDateString() : '—'}</td>
            <td><div className="inline-actions"><button className="btn btn-secondary btn-small" disabled={busy} onClick={() => void run(() => row.item ? openRecord(row.item) : openDraft(row.id))}>Open</button><StageRecordManageButton type={row.item ? 'configuration' : 'draft'} id={row.id} name={row.name} /></div></td>
          </tr>)}</tbody></table></div> : <EmptyState title={rows.length ? 'No matching targets and splits' : 'No targets and splits yet'} description={datasets.data?.datasets.length ? 'Create a target and split version from a saved dataset.' : 'Save a dataset first, then define its target and training/testing membership.'} />}
        </StageLibrary>
        {!datasets.data?.datasets.length && !datasets.isPending ? <a className="btn btn-secondary" href="#dataset">Open Datasets</a> : null}
        {legacyHistory.versions.length || legacyHistory.drafts.length ? <p className="muted"><a href="#legacy-protocol">Open legacy development protocols</a></p> : null}
      </> : view === 'detail' ? record ? <Panel title={versionLabelText(record, 'Targets and splits')} subtitle={`Dataset: ${datasetName(record.manifest.datasetId)}`} actions={<Badge tone="green">Frozen</Badge>}>
        <TargetSplitSummary value={record.manifest} /><Findings findings={record.manifest.findings ?? []} />
        <TargetSplitTestCohort project={project} record={record} />
        {record.versionLabel?.note ? <p>{record.versionLabel.note}</p> : null}
        <div className="inline-actions"><StageContinueButton href={targetSplitSetupLink(record)}>Continue to Experimental Setup</StageContinueButton><button className="btn btn-secondary" disabled={busy} onClick={() => void run(copyRecord)}>Copy into a new draft</button></div>
        <VersionLabelEditor project={project} resourceType="configuration" resource={record} tagLabel="Target and split tag" />
      </Panel> : <p role="status">Loading target and split version…</p> : <>
        <StageSteps label="Target and split steps" current={String(step)} disabled={busy} onChange={(id) => setStep(Number(id))} steps={[
          { id: '1', title: TARGET_SPLIT_STEPS[0], description: dataset ? datasetVersionLabel(dataset) : 'Select records', complete: Boolean(dataset) },
          { id: '2', title: TARGET_SPLIT_STEPS[1], description: methodNames[spec.split.method], disabled: !dataset || (step === 1 && !cohortReady), complete: Boolean(live.data?.valid) },
          { id: '3', title: TARGET_SPLIT_STEPS[2], description: spec.target.field || 'Define partition labels', disabled: !dataset || !live.data?.valid, complete: targetReady(spec) },
          { id: '4', title: TARGET_SPLIT_STEPS[3], description: preview ? 'Review fixed memberships' : 'Check assignments', disabled: !preview },
        ]} />
        <fieldset className="target-split-editor" disabled={busy}><legend className="sr-only">Target and split editor</legend>
          {step === 1 ? <Panel title="Dataset and cohort" subtitle="Choose the dataset and any conditions that define your study population.">
            <label className="label">Draft name<input className="field" value={name} maxLength={200} onChange={(event) => { setName(event.target.value); setPreview(null); }} /></label>
            <DatasetSelect versions={datasets.data?.datasets ?? []} value={spec.datasetId} allowEmpty label="Dataset" onChange={(datasetId) => { edit(targetSplitWithUnit(newTargetSplitSpec(datasetId, spec.split.seed), unit)); }} />
            <TargetSplitUnitControl value={unit} onChange={editUnit} legacy={spec.splitUnit === undefined} />
            {dataset ? <div className="stack target-split-cohort-filter">
              <ConditionEditor title="Cohort conditions" conditions={spec.eligibility} columns={fields} fieldContext={fieldContext} onChange={(eligibility) => edit({ eligibility })} description="Select dataset records using their imported attributes." emptyMessage="Use every dataset record." />
              <section aria-label="Live cohort preview" aria-live="polite">
                <ErrorNotice error={cohortLive.error} />
                {cohortLive.loading ? <p className="protocol-live-status" role="status">Updating eligible cohort…</p> : cohortLive.data ? <>
                  {cohortLive.data.cohort ? <CohortStats splitUnit={unit} stats={cohortLive.data.cohort} total={cohortLive.data.dataset.totalSlides} /> : null}
                  {cohortLive.data.cohort?.totalSlides === 0 ? <p className="callout">No slides match these conditions. Adjust the filters to continue.</p> : null}
                  <Findings findings={cohortLive.data.findings} />
                </> : null}
              </section>
            </div> : <p className="muted">Choose a frozen dataset to continue.</p>}
          </Panel> : step === 2 ? <Panel title="Training and testing sets" subtitle={unit === 'slide' ? 'Select individual slides for training and testing. Conditions match each slide exactly.' : 'Select records before defining targets. Keep each patient in one set; testing can support evaluation or pure inference.'}>
            <TargetSplitUnitControl value={unit} onChange={editUnit} legacy={spec.splitUnit === undefined} />
            <label className="label">Split method<select className="field" value={spec.split.method} onChange={(event) => edit({ split: targetSplitMethod(spec.split, event.target.value as TargetSplitSpec['split']['method']) })}>{Object.entries(methodNames).map(([id, title]) => <option key={id} value={id}>{title}</option>)}</select></label>
            {spec.split.method === 'random' ? <div className="stack"><div className="science-grid-two"><label className="label">Testing percentage<input className="field" type="number" min={0} max={99} step={1} value={Math.round(spec.split.testFraction * 100)} onChange={(event) => editSplit({ testFraction: Number(event.target.value) / 100 })} /><small>The remaining {100 - Math.round(spec.split.testFraction * 100)}% forms the training set. {randomSplitNote(unit, spec.split.stratify)}</small></label><label className="label">Split seed<input className="field" type="number" min={0} max={4294967295} step={1} value={spec.split.seed} onChange={(event) => editSplit({ seed: Number(event.target.value) })} /><small>Reproduces this train/test assignment.</small></label></div><label className="science-check"><input type="checkbox" checked={spec.split.stratify} onChange={(event) => editSplit({ stratify: event.target.checked, stratifyField: event.target.checked ? spec.split.stratifyField : undefined })} /><span>Balance a metadata field between training and testing</span></label>{spec.split.stratify ? <label className="label">Stratification field<select className="field" value={spec.split.stratifyField ?? ''} onChange={(event) => editSplit({ stratifyField: event.target.value || undefined })}><option value="">Choose a metadata field</option>{dictionary.map((item) => <option key={item.key}>{item.key}</option>)}</select><small>Choose this field before defining prediction targets. Target mapping never redraws these assignments.</small></label> : null}<p className="muted">Choose 0% testing if an independent test cohort will be supplied later.</p></div> : spec.split.method === 'rules' ? <div className="stack">
              <div className="target-split-rule-block">
                <ConditionEditor title="Training conditions" conditions={spec.split.trainRules} columns={fields} fieldContext={fieldContext} onChange={(trainRules) => editSplit({ trainRules })} description={unit === 'slide' ? 'Choose the slides used for model development. Leave empty to use the remainder outside testing.' : 'Choose the patients used for model development. Leave empty to use the remainder outside testing.'} emptyMessage={unit === 'slide' ? 'Use all eligible slides outside the testing set.' : 'Use all eligible patient groups outside the testing set.'} />
                {live.loading ? <p className="protocol-live-status" role="status">Updating training selection…</p> : live.data ? <TargetSplitSelection splitUnit={unit} part={live.data.partitions.train} total={live.data.summary.eligibleSlides} label="Training" /> : null}
              </div>
              <div className="target-split-rule-block">
                <TargetSplitTestingRules split={spec.split} unit={unit} columns={fields} fieldContext={fieldContext} onChange={(split) => edit({ split })} />
                {live.loading ? <p className="protocol-live-status" role="status">Updating testing selection…</p> : live.data ? <TargetSplitSelection splitUnit={unit} part={live.data.partitions.test} total={live.data.summary.eligibleSlides} label="Testing" /> : null}
              </div>
              <p className="muted">{spec.split.testRemaining ? unit === 'slide' ? 'Every eligible slide is assigned: training conditions select training and all other slides are tested.' : 'Every eligible patient group is assigned: training conditions select training and all other groups are tested.'
                : unit === 'slide' ? 'With explicit training conditions, unmatched slides are excluded. A slide matching both training and testing blocks freezing.' : 'With explicit training conditions, unmatched patient groups are excluded. A patient matching both training and testing blocks freezing.'}</p>
            </div> : <ImportedPartitions project={project} spec={spec} fields={dictionary.map((item) => item.key)} onChange={editSplit} />}
            <ErrorNotice error={live.error} />
            {live.loading && spec.split.method !== 'rules' ? <p className="protocol-live-status" role="status">Updating training and testing selections…</p> : live.data ? <>
              {spec.split.method !== 'rules' ? <TargetSplitPartitionCounts splitUnit={unit} value={live.data} /> : null}
              <TargetSplitSelectionSummary splitUnit={unit} value={live.data} /><Findings findings={live.data.findings} />
            </> : null}
          </Panel> : step === 3 ? <div className="stack target-split-targets">
            <Panel title="Training prediction target" subtitle="Map labels from the selected training records to the model classes.">
              <PredictionTargetEditor splitUnit={unit} target={spec.target} fieldContext={fieldContext} showFieldProfile={false} unlinkedSlideCount={live.data?.partitions.train.unlinkedSlides} fallbackSlideCount={live.data?.partitions.train.fallbackSlides}
                labelValues={{ data: labelData('train'), isPending: live.loading, error: live.error }} rawValues={rawValues('train')} dataLabel="selected training slides" onChooseTarget={(field) => chooseTarget(field)} onChange={editTrainingTarget} />
              {live.loading ? <p className="protocol-live-status" role="status">Updating training target distribution…</p> : live.data ? <TargetSplitDistributions splitUnit={unit} partitions={live.data.partitions} partition="train" mapped={Boolean(liveRequest.target)} /> : null}
            </Panel>
            <Panel title="Testing prediction target" subtitle="Choose labels for evaluation, or retain every selected testing slide for pure inference.">
              {targetSplitTestingIssue(spec) ? <p className="callout callout-warning" role="alert">{targetSplitTestingIssue(spec)}</p> : null}
              <label className="label">Testing target<select className="field" value={spec.testTarget === null ? 'none' : spec.testTarget === undefined ? 'same' : 'separate'} onChange={(event) => edit({ testTarget: event.target.value === 'none' ? null : event.target.value === 'same' ? undefined : { ...spec.target, field: '', labels: {} } })}>
                <option value="same">Same field and mapping as training</option><option value="separate">Separate testing field and mapping</option><option value="none">None · Pure inference</option>
              </select></label>
              {spec.testTarget === null ? <p className="callout">No testing labels are required or read. Missing labels do not remove testing slides. This set is used for predictions without evaluation metrics.</p>
                : spec.testTarget ? <PredictionTargetEditor splitUnit={unit} target={spec.testTarget} classDefinitionLocked showFieldProfile={false} fieldContext={fieldContext} unlinkedSlideCount={live.data?.partitions.test.unlinkedSlides} fallbackSlideCount={live.data?.partitions.test.fallbackSlides}
                  labelValues={{ data: labelData('test'), isPending: live.loading, error: live.error }} rawValues={rawValues('test')} dataLabel="selected testing slides" onChooseTarget={(field) => chooseTarget(field, 'test')} onChange={(update) => edit({ testTarget: { ...spec.testTarget!, ...update } })} />
                  : <p className="muted">Testing uses {spec.target.field || 'the training target'} and the same label mapping. Its source values are counted only within the testing selection.</p>}
              {testingTarget !== null ? live.loading ? <p className="protocol-live-status" role="status">Updating testing target distribution…</p> : live.data ? <TargetSplitDistributions splitUnit={unit} partitions={live.data.partitions} partition="test" mapped={Boolean(liveRequest.target)} /> : null : null}
            </Panel>
            <ErrorNotice error={live.error} />
            {live.data ? <Findings findings={live.data.findings} /> : null}
          </div> : preview ? <Panel title="Review targets and splits" subtitle={unit === 'slide' ? 'Review slide counts, labels and separation before saving fixed memberships.' : 'Review label counts and patient separation before saving fixed memberships.'}><TargetSplitSummary value={preview} /><Findings findings={preview.findings} /></Panel> : null}
          <div className="setup-step-actions"><div className="inline-actions">{step > 1 ? <StageBackButton onClick={() => setStep(step - 1)}>Back</StageBackButton> : null}<button className="btn btn-secondary" disabled={!name.trim()} onClick={() => void run(async () => { await save(); setMessage('Target and split draft saved.'); })}>Save draft</button></div>
            {step < 3 ? <StageContinueButton disabled={!name.trim() || (step === 1 ? !dataset || cohortLive.loading || !cohortReady : live.loading || !live.data?.valid)} onClick={() => setStep(step + 1)}>Continue to {step === 1 ? 'training & testing' : 'prediction targets'}</StageContinueButton> : step === 3 || !preview ? <StageContinueButton disabled={!dataset || !targetReady(spec) || live.loading || !live.data?.valid} onClick={() => void run(check)}>Review assignments</StageContinueButton> : <button className="btn btn-primary" disabled={!preview.canFreeze || !draft} onClick={() => { if (draft) setFreezeReview({ draft, preview, operationId: crypto.randomUUID() }); }}>Name &amp; freeze version</button>}
          </div>
        </fieldset>
      </>}
    </StagePage>
    {freezeReview ? <FreezeVersionDialog kind="target-split" initialLabel={freezeLabel} onLabelChange={setFreezeLabel} onClose={() => setFreezeReview(null)} onFreeze={async (versionLabel) => {
      pending.current = true; setBusy(true);
      try {
        const result = await targetSplits.freeze(project, freezeReview.draft.id, freezeReview.draft.revision, freezeReview.preview.previewHash, versionLabel, freezeReview.operationId);
        await Promise.all([client.cancelQueries({ queryKey: targetSplitKey(project) }), client.cancelQueries({ queryKey: [...scienceKey(project), 'drafts'] })]);
        client.setQueryData<{ configurations: TargetSplit[] }>(targetSplitKey(project), (current) => ({ configurations: [result, ...(current?.configurations ?? []).filter((item) => item.id !== result.id)] }));
        client.setQueryData<{ drafts: TargetSplitDraft[] }>([...scienceKey(project), 'drafts'], (current) => ({ drafts: (current?.drafts ?? []).map((item) => item.id === freezeReview.draft.id ? { ...item, status: 'frozen' as const } : item) }));
        client.setQueryData([...targetSplitKey(project), result.id], result);
        setDraft({ ...freezeReview.draft, status: 'frozen' }); setSaved(result); setOpenedId(result.id); setView('detail'); setPreview(null); setResumeAvailable(false); setError(null);
        setMessage(`Targets and splits “${result.versionLabel?.tag ?? versionLabel.tag}” frozen. Training and testing memberships are saved.${result.evaluationCohortId ? ' Its testing set is ready as a test cohort.' : ''}`);
        void Promise.all([client.invalidateQueries({ queryKey: ['workspace', project] }), client.invalidateQueries({ queryKey: ['evaluation-cohorts', project] })]);
      } catch (reason) {
        if (scientificReviewInvalidated(reason) || reason instanceof ApiError && reason.code === 'TARGET_SPLIT_BLOCKED') { setFreezeReview(null); setPreview(null); setStep(3); setError(reason as Error); }
        throw reason;
      } finally { pending.current = false; setBusy(false); }
    }}><p>{freezeReview.preview.summary.trainingSlides.toLocaleString()} training slides · {freezeReview.preview.summary.testingSlides.toLocaleString()} testing slides · {freezeReview.preview.spec.target.field}</p></FreezeVersionDialog> : null}
  </div>;
}

/** Testing takes its own conditions, or every eligible slide or patient group that training leaves out. */
export function TargetSplitTestingRules({ split, unit, columns, fieldContext, onChange }: { split: TargetSplitSpec['split']; unit: TargetSplitUnit; columns: string[]; fieldContext: ProtocolFieldContext; onChange: (split: TargetSplitSpec['split']) => void }) {
  // Conditions set aside by the remainder option return if it is turned off again.
  const setAside = useRef<Condition[]>([]);
  const record = unit === 'slide' ? 'slide' : 'patient group';
  const remaining = Boolean(split.testRemaining);
  const unavailable = !remaining && !split.trainRules.length;
  function choose(checked: boolean) {
    if (checked) setAside.current = split.testRules;
    onChange(targetSplitTestingRemainder(split, checked, checked ? [] : setAside.current));
  }
  return <>
    {remaining ? <div className="condition-editor"><h3>Testing conditions</h3><p className="muted">Testing uses every eligible {record} that the training conditions do not select, for evaluation or pure inference.</p></div>
      : <ConditionEditor title="Testing conditions" conditions={split.testRules} columns={columns} fieldContext={fieldContext} onChange={(testRules) => onChange({ ...split, testRules })} description={unit === 'slide' ? 'Choose the slides reserved for evaluation or pure inference.' : 'Choose the patients reserved for evaluation or pure inference.'} emptyMessage={unit === 'slide' ? 'No testing slides selected. Add a condition, or use all slides outside training below.' : 'No testing records selected. Add a condition, or use all patient groups outside training below.'} />}
    <label className="science-check"><input type="checkbox" checked={remaining} disabled={unavailable} onChange={(event) => choose(event.target.checked)} /><span>Use all eligible {record}s outside the training set</span></label>
    {unavailable ? <small className="muted">Add training conditions first. Testing can take the remainder only when training is selected by conditions.</small> : null}
  </>;
}

function ImportedPartitions({ project, spec, fields, onChange }: { project: string; spec: TargetSplitSpec; fields: string[]; onChange: (update: Partial<TargetSplitSpec['split']>) => void }) {
  const [extra, setExtra] = useState('');
  const [added, setAdded] = useState<string[]>([]);
  const values = useQuery({ queryKey: [...scienceKey(project), 'target-split-partitions', spec.datasetId, spec.split.partitionField],
    queryFn: () => scientific.queryDataset(project, spec.datasetId, { field: spec.split.partitionField ?? undefined, search: '', filters: [], offset: 0, limit: 1 }), enabled: Boolean(spec.split.partitionField) });
  const observed = [...new Set([...(values.data?.valueCounts ?? []).map((item) => item.value).filter((item): item is string => item !== null), ...spec.split.trainValues, ...spec.split.testValues, ...added])];
  return <div className="stack"><label className="label">Partition column<select className="field" value={spec.split.partitionField ?? ''} onChange={(event) => { setAdded([]); setExtra(''); onChange({ partitionField: event.target.value || undefined, trainValues: [], testValues: [] }); }}><option value="">Choose a column</option>{fields.map((field) => <option key={field}>{field}</option>)}</select></label>
    <ErrorNotice error={values.error} />
    {values.isFetching ? <p role="status">Reading partition values…</p> : null}
    {spec.split.partitionField ? <><div className="table-wrap"><table aria-label="Partition value mapping"><thead><tr><th>Source value</th><th>Assign to</th></tr></thead><tbody>{observed.map((value) => <tr key={value}><th scope="row">{value || '(empty)'}</th><td><select className="field" aria-label={`Assign ${value || 'empty'} partition`} value={spec.split.trainValues.includes(value) ? 'train' : spec.split.testValues.includes(value) ? 'test' : ''} onChange={(event) => onChange({ trainValues: [...spec.split.trainValues.filter((item) => item !== value), ...(event.target.value === 'train' ? [value] : [])], testValues: [...spec.split.testValues.filter((item) => item !== value), ...(event.target.value === 'test' ? [value] : [])] })}><option value="">Exclude</option><option value="train">Training</option><option value="test">Testing</option></select></td></tr>)}</tbody></table></div>
      {values.data?.valuesTruncated ? <p className="muted">Only the most common values are shown. Add any exact value below.</p> : null}
      <div className="inline-actions"><label className="label">Add exact partition value<input className="field" value={extra} onChange={(event) => setExtra(event.target.value)} /></label><button className="btn btn-secondary btn-small" disabled={!extra || observed.includes(extra)} onClick={() => { setAdded((current) => [...current, extra]); setExtra(''); }}>Add value</button></div>
      <p className="muted">{targetSplitUnit(spec) === 'slide' ? 'Unmapped slides are excluded. Slides matching both training and testing block freezing.' : 'Unmapped patient groups are excluded. Patient groups matching both training and testing block freezing.'}</p></> : null}
  </div>;
}


export function TargetSplitUnitControl({ value, onChange, legacy = false }: { value: TargetSplitUnit; onChange: (value: TargetSplitUnit) => void; legacy?: boolean }) {
  const name = useId();
  return <fieldset className="target-split-unit-control"><legend>Split unit</legend>
    <div className="target-split-unit-options">{(['slide', 'patient'] as const).map((unit) => <label className={`target-split-unit-option${value === unit ? ' is-selected' : ''}`} key={unit}>
      <input type="radio" name={name} value={unit} checked={value === unit} onChange={() => onChange(unit)} onClick={() => { if (legacy && value === unit) onChange(unit); }} />
      <span><strong>{unit === 'slide' ? 'Slide' : 'Patient'}</strong><small>{unit === 'slide' ? 'Select and split each slide independently' : 'Keep every selected slide from a patient together'}</small></span>
    </label>)}</div>
    <p className="muted">{value === 'slide' ? 'Filters match individual slides. Targets, distributions and experimental folds use slides.' : 'Matching a slide selects its eligible patient group. Targets, distributions and experimental folds use patient grouping.'}</p>
    {legacy ? <small className="muted">This saved draft uses the original patient grouping. Choose Slide to change it explicitly.</small> : null}
  </fieldset>;
}
