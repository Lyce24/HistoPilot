import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { AttributeMapping, Condition, ConditionValue, ProtocolExploration, ProtocolSpec, VersionLabelInput } from '../api/scientific';
import { scientific } from '../api/scientific';
import { evaluation, type EvaluationDraft, type EvaluationCohort, type EvaluationPreview, type EvaluationSpec } from '../api/evaluation';
import { datasetVersionLabel, versionLabelText } from '../lib/versionLabels';
import { sameJSON } from '../lib/json';
import { Badge, EmptyState, ErrorNotice, Icon, Metric, PageHeader, Panel } from '../components/ui';
import { Findings, SavedNotice, scienceKey, useDatasets, useRefreshScientific } from '../components/ScientificUI';
import { reportEditorValidity } from '../components/NumericField';
import { CohortSample, CohortStats, DistributionBars } from '../components/ProtocolExploration';
import PredictionTargetEditor from '../components/PredictionTargetEditor';
import FreezeVersionDialog from '../components/FreezeVersionDialog';
import { ConditionEditor } from './LocalProtocol';
import { inferTargetSettings, newDevelopmentSplit } from '../lib/protocol';
import { scientificReviewInvalidated } from '../lib/scientificReview';
import { taskLabel, unitLabel } from '../lib/labels';
import { StageLibrary, StageLibraryToolbar, StageRecordManageButton, StagePage, StageSteps, useStageLibrary } from '../components/StageWorkflow';
import './protocol-workflow.css';
import './LocalEvaluationSetup.css';

const newTestTarget = (): ProtocolSpec['target'] => ({ field: '', task: '', unit: 'patient', classes: [], labels: {}, missing: 'block', unmapped: 'block' });

export const newEvaluationSpec = (): EvaluationSpec => ({
  protocolId: '', developmentFeatureBundleId: '', datasetId: '', featureBundleId: '',
  target: newTestTarget(), eligibility: [], patientIdentifiers: 'shared',
  inference: { loadingPolicy: 'per_slide', packArtifactId: null, batchSize: 1, numWorkers: 0,
    device: 'auto', precision: 'float32', patientAggregation: 'mean', decisionThreshold: 0.5 },
});

export function evaluationConditionValue(text: string, op: Condition['op'], type?: AttributeMapping['type']): ConditionValue {
  const numeric = ['lt', 'lte', 'gt', 'gte'].includes(op) || ['integer', 'decimal'].includes(type ?? '');
  const scalar = (value: string) => numeric && value.trim() !== '' && Number.isFinite(Number(value)) ? Number(value) : value;
  if (op === 'in' || op === 'not_in') return text.split('|').map((value) => scalar(value.trim()));
  if (op === 'regex') return text;
  if (type === 'boolean' && (text === 'true' || text === 'false')) return text === 'true';
  return scalar(text);
}

function SlideIds({ label, ids }: { label: string; ids: string[] }) {
  return ids.length > 0 ? <details className="evaluation-slide-ids"><summary>{label} · {ids.length.toLocaleString()}</summary><pre>{ids.join('\n')}</pre></details> : null;
}

export function evaluationLabelProblem(rows: [string, string][], classes: string[]): string | null {
  if (rows.some(([raw]) => !raw.trim())) return 'Enter a raw value for every label mapping.';
  if (new Set(rows.map(([raw]) => raw)).size !== rows.length) return 'Each raw value must map to exactly one class. Remove or change duplicate values.';
  if (classes.some((label) => !rows.some(([, mapped]) => mapped === label))) return 'Map at least one raw value to every inherited class.';
  return null;
}

export function EvaluationTargetMapping({ rows, classes, onChange }: {
  rows: [string, string][]; classes: string[]; onChange: (rows: [string, string][]) => void;
}) {
  return <div className="evaluation-label-mapping">
    {rows.map(([raw, mapped], index) => <div className="evaluation-mapping-row" key={index}>
      <label className="label">Raw dataset value<input className="field" value={raw} maxLength={4096} onChange={(event) => onChange(rows.map((row, at) => at === index ? [event.target.value, mapped] : row))} /></label>
      <label className="label">Inherited class<select className="field" value={mapped} onChange={(event) => onChange(rows.map((row, at) => at === index ? [raw, event.target.value] : row))}>{classes.map((label) => <option key={label}>{label}</option>)}</select></label>
      <button type="button" className="icon-button" aria-label={`Remove label mapping ${index + 1}`} onClick={() => onChange(rows.filter((_, at) => at !== index))}><Icon name="close" /></button>
    </div>)}
    <button type="button" className="btn btn-secondary btn-small" disabled={rows.length >= 200} onClick={() => onChange([...rows, ['', classes[0]]])}><Icon name="plus" size={14} />Add raw label value</button>
  </div>;
}

export { EvaluationInferenceFields } from '../components/EvaluationInputSettings';

export function EvaluationEvidence({ preview }: { preview: EvaluationPreview }) {
  const { summary, coverage } = preview;
  return <div className="evaluation-evidence">
    <div className="evaluation-counts">
      <div><strong>{summary.includedSlides.toLocaleString()}</strong><span>Selected slides</span></div>
      <div><strong>{summary.includedPatients.toLocaleString()}</strong><span>Patient groups</span></div>
      <div><strong>{summary.labeledSlides.toLocaleString()}</strong><span>Labeled slides</span></div>
      <div><strong>{summary.excludedSlides.toLocaleString()}</strong><span>Excluded slides</span></div>
    </div>
    <Findings findings={preview.findings} />
    <div className="grid-2">
      <div><h3>Exact feature coverage</h3><p>{coverage.missingFeatureSlideIds.length === 0 ? 'Every selected slide has a feature file.' : `${coverage.missingFeatureSlideIds.length.toLocaleString()} selected slides are missing feature files.`}</p>
        {coverage.packChecked ? <p>{coverage.missingPackSlideIds.length === 0 ? 'The selected pack covers every selected slide.' : `${coverage.missingPackSlideIds.length.toLocaleString()} selected slides are missing from the pack.`}</p> : <p className="muted">{preview.spec.inference.loadingPolicy === 'packed' ? 'Packed loading was requested, but pack coverage could not be verified. Review the findings above.' : 'Original feature files selected; no pack coverage is required.'}</p>}
      </div>
      <div><h3>Overlap with development</h3><p>{summary.developmentSlideOverlap.toLocaleString()} overlapping slide IDs</p><p>{preview.overlap.patientsComparable ? `${summary.developmentPatientOverlap.toLocaleString()} overlapping patient IDs` : 'Patient overlap cannot be checked across separate naming systems.'}</p><p className="muted">Overlap is checked against the selected development protocol. Identifier conventions and prior use of these records still need to be correct.</p></div>
    </div>
    <SlideIds label="Missing feature slide IDs" ids={coverage.missingFeatureSlideIds} />
    <SlideIds label="Missing pack slide IDs" ids={coverage.missingPackSlideIds} />
    <SlideIds label="Overlapping development slide IDs" ids={preview.overlap.slideIds} />
    <SlideIds label="Source files already used in development under different slide IDs" ids={preview.overlap.sourceSlideIds ?? []} />
    {preview.overlap.patientsComparable ? <SlideIds label="Overlapping development patient IDs" ids={preview.overlap.patientIds} /> : null}
    <SlideIds label="Selected slide IDs" ids={coverage.selectedSlideIds} />
    <p className="muted">Development representation: {preview.compatibility.development.encoderId ?? 'unspecified encoder'} · {preview.compatibility.development.dimensions ?? '?'} dimensions. Test representation: {preview.compatibility.evaluation.encoderId ?? 'unspecified encoder'} · {preview.compatibility.evaluation.dimensions ?? '?'} dimensions.</p>
    {Object.keys(summary.classCounts).length > 0 ? <p className="muted">Mapped class counts: {Object.entries(summary.classCounts).map(([label, count]) => `${label}: ${count.toLocaleString()}`).join(' · ')}</p> : null}
  </div>;
}

export function cohortDatasetIds(spec: EvaluationSpec): string[] {
  return spec.datasetIds?.length ? spec.datasetIds : spec.datasetId ? [spec.datasetId] : [];
}

export function independentCohortSpec(spec: EvaluationSpec): EvaluationSpec {
  return { ...newEvaluationSpec(), datasetId: spec.datasetId, datasetIds: spec.datasetIds,
    target: spec.target, eligibility: spec.eligibility };
}

function describeTestCondition(condition: Condition) {
  if (condition.op === 'exists') return `${condition.field} ${condition.value ? 'is present' : 'is missing'}`;
  const labels = { eq: 'equals', ne: 'does not equal', in: 'is one of', not_in: 'is not one of', regex: 'matches', lt: 'is less than', lte: 'is at most', gt: 'is greater than', gte: 'is at least' };
  return `${condition.field} ${labels[condition.op]} ${Array.isArray(condition.value) ? condition.value.join(' | ') : String(condition.value)}`;
}

export function mergeTestDistributions(results: ProtocolExploration[]) {
  const counts = new Map<string | null, number>();
  let truncated = false;
  for (const result of results) {
    if (!result.target) continue;
    truncated ||= result.target.distinctCount > result.target.values.length;
    for (const { value, slides } of result.target.values) counts.set(value, (counts.get(value) ?? 0) + slides);
  }
  return {
    valueCounts: [...counts].map(([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count),
    valuesTruncated: truncated,
  };
}

function explorationRequest(datasetId: string, eligibility: Condition[], targetField: string) {
  return { datasetId, eligibility, targetField: targetField || undefined,
    rules: { train: [], val: [], test: [] }, splitMode: 'kfold' as const, split: newDevelopmentSplit() };
}

function useTestExploration(project: string, ids: string[], eligibility: Condition[], field: string, enabled: boolean) {
  const serialized = JSON.stringify({ ids, eligibility, field });
  const [settled, setSettled] = useState(serialized);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(serialized), 300);
    return () => window.clearTimeout(timer);
  }, [serialized]);
  const changing = serialized !== settled;
  const query = useQuery({
    queryKey: [...scienceKey(project), 'test-cohort-exploration', settled],
    queryFn: async () => {
      const input = JSON.parse(settled) as { ids: string[]; eligibility: Condition[]; field: string };
      return Promise.all(input.ids.map((id) => scientific.exploreProtocol(project, explorationRequest(id, input.eligibility, input.field))));
    },
    enabled: enabled && ids.length > 0 && !changing,
    retry: false,
  });
  return { data: changing ? undefined : query.data, error: changing ? null : query.error,
    loading: enabled && ids.length > 0 && (changing || query.isPending) };
}

export function TestCohortSummary({ preview }: { preview: Pick<EvaluationPreview, 'summary' | 'findings'> & { coverage: { selectedSlideIds: string[] } } }) {
  const summary = preview.summary;
  return <div className="stack">
    <div className="science-metrics protocol-metrics">
      <Metric label="Selected test slides" value={summary.includedSlides.toLocaleString()} />
      <Metric label="Patient groups" value={summary.includedPatients.toLocaleString()} />
      <Metric label="Labeled slides" value={summary.labeledSlides.toLocaleString()} />
      <Metric label="Excluded slides" value={summary.excludedSlides.toLocaleString()} />
    </div>
    {Object.keys(summary.classCounts).length ? <DistributionBars values={Object.entries(summary.classCounts).map(([value, count]) => ({ value, count }))} caption="Prediction target · selected test slides by class" /> : null}
    <Findings findings={preview.findings} />
    <SlideIds label="Selected test slide IDs" ids={preview.coverage.selectedSlideIds} />
  </div>;
}

export default function LocalEvaluationSetup({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const datasets = useDatasets(project);
  const drafts = useQuery({ queryKey: ['evaluation-drafts', project], queryFn: () => evaluation.drafts(project) });
  const frozen = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project) });
  const client = useQueryClient();
  const refreshScientific = useRefreshScientific(project);
  const [librarySearch, setLibrarySearch] = useState('');
  const [libraryStatus, setLibraryStatus] = useState('all');
  const [librarySort, setLibrarySort] = useState('recent');
  const [step, setStep] = useState<0 | 1 | 2 | 3>(0);
  const [spec, setSpec] = useState<EvaluationSpec>(newEvaluationSpec);
  const [name, setName] = useState(`${workspace.project.name} test cohort`);
  const [draft, setDraft] = useState<EvaluationDraft | null>(null);
  const [savedCohort, setSavedCohort] = useState<EvaluationCohort | null>(null);
  const [preview, setPreview] = useState<EvaluationPreview | null>(null);
  const [distributionField, setDistributionField] = useState('');
  const [label, setLabel] = useState<VersionLabelInput>({ tag: '', note: '' });
  const [freezeReview, setFreezeReview] = useState<{ draft: EvaluationDraft; preview: EvaluationPreview; operationId: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const pending = useRef(false);
  const targetRequest = useRef(0);
  const editor = useRef<HTMLFieldSetElement>(null);
  const ids = cohortDatasetIds(spec);
  const selectedDatasets = (datasets.data?.datasets ?? []).filter((item) => ids.includes(item.id));
  const dictionary = [...new Map(selectedDatasets.flatMap((item) => item.manifest.dictionary ?? []).map((item) => [item.key, item])).values()];
  const columns = dictionary.map((item) => item.key);
  const fieldContext = { project, datasetId: spec.datasetId, dictionary };
  const activeField = step === 2 ? spec.target?.field ?? '' : distributionField || spec.target?.field || columns[0] || '';
  const live = useTestExploration(project, ids, spec.eligibility, activeField, step === 1 || step === 2);
  const targetValues = live.data ? mergeTestDistributions(live.data) : undefined;
  const rawValues = (targetValues?.valueCounts ?? []).map((item) => item.value).filter((value): value is string => value !== null && value.trim() !== '');
  const dirty = !draft || draft.name !== name.trim() || !sameJSON(draft.payload.spec, spec);
  const editable = !savedCohort && draft?.status !== 'frozen';
  const targetReady = spec.target === null || Boolean(spec.target.field && spec.target.task && spec.target.classes.length >= 2 && (spec.target.task !== 'binary_classification' || spec.target.positiveClass));
  const dataReady = ids.length > 0 && selectedDatasets.length === ids.length;
  const datasetNames = (value: EvaluationSpec) => cohortDatasetIds(value).map((id) => {
    const item = datasets.data?.datasets.find((dataset) => dataset.id === id);
    return item ? datasetVersionLabel(item) : versionLabelText({ id }, 'Dataset');
  }).join(', ') || 'Not selected';

  function showStep(next: typeof step) {
    setStep(next);
  }
  async function refresh() {
    await Promise.all([refreshScientific(), client.invalidateQueries({ queryKey: ['evaluation-drafts', project] }), client.invalidateQueries({ queryKey: ['evaluation-cohorts', project] })]);
  }
  function edit(update: Partial<EvaluationSpec>) {
    targetRequest.current += 1;
    setSpec((current) => ({ ...current, ...update }));
    setPreview(null); setError(null); setMessage(''); setFreezeReview(null);
  }
  function reset(nextSpec = newEvaluationSpec(), nextName = `${workspace.project.name} test cohort`) {
    targetRequest.current += 1;
    setSpec(nextSpec); setName(nextName); setDraft(null); setSavedCohort(null); setPreview(null);
    setDistributionField(''); setLabel({ tag: '', note: '' }); setError(null); setMessage(''); setFreezeReview(null);
    showStep(1);
  }
  async function run(action: () => Promise<void>, validate = false) {
    if (pending.current || (validate && !reportEditorValidity(editor.current))) return;
    pending.current = true; setBusy(true); setError(null); setMessage('');
    try { await action(); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('The test cohort could not be saved.')); }
    finally { pending.current = false; setBusy(false); }
  }
  async function save() {
    const saved = !dirty && draft ? draft : await evaluation.saveDraft(project, name.trim(), spec, editable && draft ? draft : undefined);
    setDraft(saved); setSpec(saved.payload.spec); setName(saved.name);
    await refresh();
    return saved;
  }
  async function review() {
    setPreview(null);
    const saved = await save();
    const checked = await evaluation.preview(project, saved);
    setPreview(checked);
    showStep(3);
  }
  async function chooseTarget(field: string) {
    edit({ target: { ...(spec.target ?? newTestTarget()), field, ...inferTargetSettings([]) } });
    const requestId = targetRequest.current;
    if (!field) return;
    try {
      const results = await Promise.all(ids.map((id) => scientific.exploreProtocol(project, explorationRequest(id, spec.eligibility, field))));
      if (requestId !== targetRequest.current) return;
      const values = mergeTestDistributions(results);
      if (results.some((result) => !result.valid)) throw new Error('Resolve the test data findings before reading target values.');
      setSpec((current) => current.target?.field === field ? { ...current, target: { ...current.target, ...inferTargetSettings(values.valueCounts.map((item) => item.value), values.valuesTruncated) } } : current);
    } catch (reason) {
      if (requestId === targetRequest.current) setError(reason instanceof Error ? reason : new Error('Target values could not be read.'));
    }
  }
  async function openDraft(id: string) {
    const saved = await evaluation.draft(project, id);
    reset(independentCohortSpec(saved.payload.spec), saved.name);
    setDraft(saved);
  }
  async function openCohort(id: string) {
    const saved = await evaluation.get(project, id);
    targetRequest.current += 1;
    setSavedCohort(saved); setSpec(saved.manifest.spec); setName(versionLabelText(saved, 'Test cohort'));
    setDraft(null); setPreview(null); setLabel({ tag: '', note: '' }); showStep(3);
  }
  function copyCohort() {
    // A copied cohort starts an independent definition. Model inputs belong to its evaluations.
    reset(independentCohortSpec(spec), `${name.slice(0, 110)} copy`);
  }
  const draftRows = (drafts.data?.drafts ?? []).filter((item) => item.status !== 'frozen');
  const cohortRows = frozen.data?.items ?? [];
  const libraryRows = [
    ...draftRows.map((record) => ({ kind: 'draft' as const, record, name: record.name, spec: record.payload.spec, status: 'planned', created: record.createdAt ?? '', updated: record.updatedAt ?? record.createdAt ?? '' })),
    ...cohortRows.map((record) => ({ kind: 'configuration' as const, record, name: versionLabelText(record, 'Test cohort'), spec: record.manifest.spec, status: record.current === false ? 'review' : 'frozen', created: record.createdAt, updated: record.versionLabel?.updatedAt ?? record.createdAt })),
  ];
  const visibleRows = libraryRows.filter((item) => (libraryStatus === 'all' || item.status === libraryStatus)
    && `${item.name} ${item.record.id} ${datasetNames(item.spec)} ${item.spec.target?.field ?? ''} ${item.kind === 'configuration' ? item.record.versionLabel?.note ?? '' : ''}`.toLowerCase().includes(librarySearch.trim().toLowerCase()))
    .sort((a, b) => (librarySort === 'name' ? a.name.localeCompare(b.name) : librarySort === 'oldest' ? a.created.localeCompare(b.created) : b.updated.localeCompare(a.updated)) || a.record.id.localeCompare(b.record.id));
  function resetLibraryFilters() { setLibrarySearch(''); setLibraryStatus('all'); setLibrarySort('recent'); }
  function openLibrary() { if (busy || freezeReview) return; void run(async () => { if (step !== 0 && editable && dirty) await save(); showStep(0); }); }
  useStageLibrary(openLibrary);

  return <div className="clinical-workspace protocol-workspace evaluation-setup" id="test-cohort-page" tabIndex={-1}>
    <PageHeader eyebrow="03 EVALUATE · TEST COHORTS" title={step === 0 ? 'Test cohorts' : savedCohort ? name : 'Create test cohort'}
      description={step === 0 ? 'Open a test cohort or create one from your datasets.' : 'Select test data, define prediction targets, then review and freeze your cohort.'}
      actions={step === 0 ? <button type="button" className="btn btn-primary" disabled={busy} onClick={() => reset()}><Icon name="plus" />Create test cohort</button> : <button type="button" className="btn btn-secondary" disabled={busy || Boolean(freezeReview)} onClick={openLibrary}>Back to test cohorts</button>} />
    <ErrorNotice error={error ?? datasets.error ?? drafts.error ?? frozen.error} />
    {error && draft && editable && step !== 0 ? <button type="button" className="btn btn-secondary science-fit" disabled={busy} onClick={() => void run(() => openDraft(draft.id))}>Reload saved draft</button> : null}
    <SavedNotice>{message}</SavedNotice>
    <StagePage pageKey={step === 0 ? 'library' : savedCohort?.id ?? step}>
    {step === 0 ? <StageLibrary project={project} title="Test cohorts">
      <StageLibraryToolbar search={librarySearch} onSearch={setLibrarySearch} searchLabel="Search test cohorts" placeholder="Name, ID, dataset or target" count={drafts.isPending || frozen.isPending ? undefined : visibleRows.length} total={libraryRows.length}
        actions={<button type="button" className="btn btn-secondary btn-small" disabled={drafts.isFetching || frozen.isFetching} onClick={() => void refresh()}>Refresh</button>}
        onReset={librarySearch || libraryStatus !== 'all' || librarySort !== 'recent' ? resetLibraryFilters : undefined}>
        <label className="label">Status<select className="field" aria-label="Test cohort status" value={libraryStatus} onChange={(event) => setLibraryStatus(event.target.value)}><option value="all">All statuses</option><option value="planned">Planned</option><option value="frozen">Frozen</option><option value="review">Needs review</option></select></label>
        <label className="label">Sort<select className="field" aria-label="Sort test cohorts" value={librarySort} onChange={(event) => setLibrarySort(event.target.value)}><option value="recent">Last updated</option><option value="oldest">Oldest first</option><option value="name">Name</option></select></label>
      </StageLibraryToolbar>
      {drafts.isPending || frozen.isPending ? <p className="muted" role="status">Loading test cohorts…</p> : visibleRows.length ? <div className="table-wrap"><table className="test-cohort-registry"><thead><tr><th scope="col">Cohort</th><th scope="col">Status</th><th scope="col">Datasets</th><th scope="col">Slides</th><th scope="col">Prediction target</th><th scope="col">Actions</th></tr></thead><tbody>
        {visibleRows.map((item) => <tr key={`${item.kind}:${item.record.id}`}><th scope="row"><button type="button" className="text-button stage-record-name" disabled={busy} onClick={() => void run(() => item.kind === 'draft' ? openDraft(item.record.id) : openCohort(item.record.id))}>{item.name}</button>{item.kind === 'draft' ? <small>Revision {item.record.revision}</small> : item.record.versionLabel?.note ? <small>{item.record.versionLabel.note}</small> : null}</th><td><Badge tone={item.status === 'frozen' ? 'green' : 'orange'}>{item.status === 'planned' ? 'Planned' : item.status === 'frozen' ? 'Frozen' : 'Needs review'}</Badge></td><td>{datasetNames(item.spec)}</td><td>{item.kind === 'draft' ? 'Pending review' : item.record.manifest.summary.includedSlides.toLocaleString()}</td><td>{item.spec.target?.field || (item.spec.target === null ? 'Unlabeled predictions' : 'Not selected')}</td><td><StageRecordManageButton type={item.kind} id={item.record.id} name={item.name} /></td></tr>)}
      </tbody></table></div> : libraryRows.length ? <EmptyState title="No matching test cohorts" description="Try another search or clear the filters." /> : <EmptyState title="No test cohorts yet" description="Create a cohort to select test records and define its prediction target." />}
    </StageLibrary> : <>
      {!savedCohort ? <StageSteps label="Test cohort stages" current={String(step)} disabled={busy || Boolean(freezeReview)} steps={[
        { id: '1', title: 'Test Data', description: 'Select datasets and test records', complete: dataReady },
        { id: '2', title: 'Prediction Targets', description: 'Choose the label to predict', disabled: !name.trim() || !dataReady, complete: targetReady },
        { id: '3', title: 'Review and Freeze', description: 'Review the selected cohort', disabled: !name.trim() || !dataReady || !targetReady, complete: Boolean(preview?.canFreeze) },
      ]} onChange={(next) => void run(async () => { if (next === '3') await review(); else showStep(Number(next) as 1 | 2); }, true)} /> : null}
      {!savedCohort ? <div className="test-cohort-draft-status"><Badge>{draft ? `${draft.status} · revision ${draft.revision}${dirty ? ' · unsaved changes' : ''}` : 'New cohort'}</Badge><span className="muted">Features and model compatibility are checked in Evaluate models.</span></div> : null}
      <fieldset ref={editor} className="evaluation-fields" disabled={busy || !editable || Boolean(freezeReview)}>
        <legend className="sr-only">Test cohort preparation</legend>
        {step === 1 ? <section className="protocol-section test-cohort-stage"><Panel title="1. Choose the test data" subtitle="Select datasets, then use conditions to choose the slides in this test cohort." actions={<Badge>Test Data</Badge>}>
          <div className="stack">
            <label className="label">Cohort name<input className="field" required maxLength={120} value={name} onChange={(event) => { setName(event.target.value); setPreview(null); setMessage(''); }} /></label>
            <fieldset className="test-cohort-datasets"><legend>Datasets</legend>
              {datasets.isPending ? <p className="muted">Loading datasets…</p> : datasets.data?.datasets.length ? datasets.data.datasets.map((item) => <label className="science-check" key={item.id}>
                <input type="checkbox" checked={ids.includes(item.id)} onChange={(event) => {
                  const next = event.target.checked ? [...ids, item.id] : ids.filter((id) => id !== item.id);
                  edit({ datasetId: next[0] ?? '', datasetIds: next.length > 1 ? next : undefined, eligibility: [], target: spec.target === null ? null : newTestTarget() });
                  setDistributionField('');
                }} /><span>{datasetVersionLabel(item)}<small>{item.manifest.summary?.slideCount?.toLocaleString() ?? 'Frozen'} slides</small></span>
              </label>) : <p className="muted">No datasets yet. <a href="#dataset">Import and freeze a dataset in Data</a> to select test records.</p>}
              {ids.filter((id) => !selectedDatasets.some((item) => item.id === id)).map((id) => <label className="science-check" key={id}><input type="checkbox" checked onChange={() => { const next = ids.filter((value) => value !== id); edit({ datasetId: next[0] ?? '', datasetIds: next.length > 1 ? next : undefined }); }} /><span>Unavailable dataset · {id}<small>Remove this selection or restore the dataset.</small></span></label>)}
            </fieldset>
            {ids.length > 1 ? <p className="muted">Conditions apply to every selected dataset. Slide IDs must be unique across datasets; shared patient IDs are treated as the same patient.</p> : null}
            {dataReady ? <>
              <ConditionEditor title="Which test slides should be included?" description="A slide is included when it matches every condition below." emptyMessage="All selected dataset slides are included. Add a condition to narrow the test cohort." conditions={spec.eligibility} columns={columns} fieldContext={fieldContext} onChange={(eligibility) => edit({ eligibility })} />
              <label className="label test-cohort-distribution-field">Show distribution by<select className="field" value={activeField} onChange={(event) => setDistributionField(event.target.value)}><option value="">Choose an attribute</option>{columns.map((column) => <option key={column}>{column}</option>)}</select></label>
              <ErrorNotice error={live.error} />
              {live.loading ? <p className="protocol-live-status" role="status">Updating selected test slides and distributions…</p> : live.data ? <div className="stack" aria-live="polite">
                {ids.length > 1 ? <div className="science-metrics"><Metric label="Selected test slides across datasets" value={live.data.reduce((sum, item) => sum + (item.cohort?.totalSlides ?? 0), 0).toLocaleString()} note="Duplicate IDs and patient groups are checked in the final review." /></div> : null}
                {targetValues && activeField ? <DistributionBars values={targetValues.valueCounts} caption={`${activeField} · selected test slides${targetValues.valuesTruncated ? ' (most frequent values per dataset)' : ''}`} /> : null}
                {live.data.map((item) => <div className="stack test-cohort-population" key={item.datasetId}>
                  {ids.length > 1 ? <h3>{datasetNames({ ...spec, datasetId: item.datasetId, datasetIds: undefined })}</h3> : null}
                  {item.cohort ? <><CohortStats stats={item.cohort} total={item.dataset.totalSlides} /><CohortSample stats={item.cohort} fields={[...spec.eligibility.map((condition) => condition.field), activeField]} /></> : null}
                  <Findings findings={item.findings} />
                </div>)}
                <p className="muted">These counts apply the conditions above. Label exclusions and patient consistency are checked in Review and Freeze.</p>
              </div> : null}
            </> : null}
          </div>
        </Panel></section> : null}
        {step === 2 ? <section className="protocol-section test-cohort-stage"><Panel title="2. What should the model predict?" subtitle="Choose the target column and review the classes found in your selected test data." actions={<Badge>Prediction Targets</Badge>}>
          <div className="stack">
            <label className="science-check"><input type="checkbox" checked={spec.target === null} onChange={(event) => edit({ target: event.target.checked ? null : newTestTarget() })} /><span>Unlabeled predictions<small>This cohort has no known outcomes. It can receive predictions; scoring needs labeled records.</small></span></label>
            {spec.target ? <PredictionTargetEditor key={draft?.id ?? 'new'} target={spec.target} fieldContext={fieldContext} unlinkedSlideCount={selectedDatasets.reduce((sum, item) => sum + (item.manifest.summary?.unlinkedSlideCount ?? 0), 0)} labelValues={{ data: targetValues, isPending: live.loading, error: live.error }} rawValues={rawValues} dataLabel="selected test records" onChooseTarget={chooseTarget} onChange={(update) => edit({ target: { ...spec.target!, ...update } })} /> : null}
            {live.data?.filter((item) => item.findings.length > 0).map((item) => <Findings key={item.datasetId} findings={item.findings} />)}
          </div>
        </Panel></section> : null}
        {step === 3 ? <section className="protocol-section test-cohort-stage"><Panel title={savedCohort ? 'Frozen test cohort' : '3. Review and Freeze'} subtitle="Review the test records and prediction target saved in this cohort." actions={savedCohort ? <Badge tone="green">Frozen</Badge> : undefined}>
          <dl className="protocol-review-facts">
            <div><dt>Cohort name</dt><dd>{name}</dd></div><div><dt>Test Data</dt><dd>{datasetNames(spec)}</dd></div>
            <div><dt>Prediction target</dt><dd>{spec.target?.field || (spec.target === null ? 'Unlabeled predictions' : 'Not selected')}{spec.target?.task ? ` · ${taskLabel(spec.target.task)} · ${unitLabel(spec.target.unit)}` : ''}</dd></div>
            <div><dt>Included records</dt><dd>{spec.eligibility.length ? spec.eligibility.map(describeTestCondition).join('; ') : 'All slides in the selected datasets'}</dd></div>
            {spec.target ? <><div><dt>Class order</dt><dd>{spec.target.classes.join(' → ') || 'Not selected'}</dd></div><div><dt>Positive class</dt><dd>{spec.target.positiveClass || 'Not selected'}</dd></div><div><dt>Label mapping</dt><dd>{Object.entries(spec.target.labels).map(([raw, mapped]) => `${raw} → ${mapped}`).join('; ') || 'Not configured'}</dd></div><div><dt>Missing / unmapped labels</dt><dd>{spec.target.missing} / {spec.target.unmapped}</dd></div></> : null}
          </dl>
          {preview ? <TestCohortSummary preview={preview} /> : savedCohort ? <TestCohortSummary preview={{ summary: savedCohort.manifest.summary, coverage: savedCohort.manifest.coverage ?? { selectedSlideIds: [] }, findings: savedCohort.findings ?? savedCohort.manifest.findings }} /> : <p className="callout">Review the cohort to check selected records, labels and patient consistency before freezing.</p>}
          {!savedCohort ? <div className="inline-actions evaluation-actions"><button type="button" className="btn btn-secondary" disabled={!name.trim() || !dataReady || !targetReady} onClick={() => void run(review, true)}><Icon name="check" />{preview ? 'Review cohort again' : 'Review cohort'}</button><button type="button" className="btn btn-primary" disabled={!preview?.canFreeze || dirty || !draft} onClick={() => { if (preview && draft) setFreezeReview({ draft, preview, operationId: `evaluation:${crypto.randomUUID()}` }); }}><Icon name="lock" />Freeze test cohort</button></div> : null}
        </Panel></section> : null}
      </fieldset>
      {savedCohort ? <div className="inline-actions"><button type="button" className="btn btn-secondary" disabled={busy} onClick={copyCohort}>Copy into a new draft</button><a className="btn btn-primary" href={`#evaluation?cohort=${encodeURIComponent(savedCohort.id)}`}>Evaluate models <Icon name="arrow" /></a></div> : <div className="protocol-step-actions test-cohort-step-actions">
        <p>{step === 1 ? 'Select test records before defining the prediction target.' : step === 2 ? 'Review the class mapping and positive class before continuing.' : 'The frozen cohort can be selected later in Evaluate models.'}</p>
        <div className="inline-actions">
          {step > 1 ? <button type="button" className="btn btn-secondary" disabled={busy || Boolean(freezeReview)} onClick={() => showStep((step - 1) as 1 | 2)}>Back</button> : null}
          <button type="button" className="btn btn-secondary" disabled={busy || Boolean(freezeReview) || !name.trim()} onClick={() => void run(async () => { await save(); setMessage('Test cohort draft saved.'); }, true)}>Save draft</button>
          {step < 3 ? <button type="button" className="btn btn-primary" disabled={busy || !name.trim() || !dataReady || (step === 2 && !targetReady)} onClick={() => void run(async () => { if (step === 2) await review(); else { await save(); showStep(2); } }, true)}>Continue to {step === 1 ? 'Prediction Targets' : 'Review and Freeze'} <Icon name="arrow" /></button> : null}
        </div>
      </div>}
    </>}
    </StagePage>
    {freezeReview ? <FreezeVersionDialog kind="cohort" initialLabel={label} onLabelChange={setLabel} onClose={() => setFreezeReview(null)} onFreeze={async (versionLabel) => {
      try {
        const saved = await evaluation.freeze(project, freezeReview.draft, freezeReview.preview.previewHash, freezeReview.operationId, versionLabel);
        setSavedCohort(saved); setDraft(null); setPreview(null); setName(versionLabelText(saved, 'Test cohort')); setSpec(saved.manifest.spec);
        setMessage('Test cohort frozen. Select it with a development model in Evaluate models when you are ready.');
        await refresh();
      } catch (reason) {
        if (scientificReviewInvalidated(reason)) { setFreezeReview(null); setPreview(null); setError(new Error(`${reason.message} Review the cohort again before freezing. Your tag and note have been kept.`)); }
        throw reason;
      }
    }}><p><strong>{name}</strong></p><p>{freezeReview.preview.summary.includedSlides.toLocaleString()} selected test slides · {freezeReview.preview.summary.includedPatients.toLocaleString()} patient groups</p></FreezeVersionDialog> : null}
  </div>;
}
