import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { AttributeMapping, Condition, ConditionValue, ProtocolSpec, VersionLabelInput } from '../api/scientific';
import { bundles } from '../api/bundles';
import { evaluation, type EvaluationDraft, type EvaluationInference, type EvaluationPreview, type EvaluationSpec } from '../api/evaluation';
import { configurationVersionLabel, versionLabelText } from '../lib/versionLabels';
import { sameJSON } from '../lib/json';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import { DatasetSelect, Findings, SavedNotice, useConfigurations, useDatasets, useRefreshScientific } from '../components/ScientificUI';
import ModuleSteps from '../components/ModuleSteps';
import NumericField, { reportEditorValidity } from '../components/NumericField';
import './LocalEvaluationSetup.css';

export const newEvaluationSpec = (): EvaluationSpec => ({
  protocolId: '', developmentFeatureBundleId: '', datasetId: '', featureBundleId: '',
  target: null, eligibility: [], patientIdentifiers: 'shared',
  inference: { loadingPolicy: 'per_slide', packArtifactId: null, batchSize: 1, numWorkers: 0,
    device: 'auto', precision: 'float32', patientAggregation: 'mean', decisionThreshold: 0.5 },
});

const operators: [Condition['op'], string][] = [
  ['eq', 'equals'], ['ne', 'does not equal'], ['in', 'is one of'], ['not_in', 'is not one of'],
  ['lt', 'less than'], ['lte', 'at most'], ['gt', 'greater than'], ['gte', 'at least'],
  ['regex', 'matches regular expression'], ['exists', 'is present / missing'],
];

export function evaluationConditionValue(text: string, op: Condition['op'], type?: AttributeMapping['type']): ConditionValue {
  const numeric = ['lt', 'lte', 'gt', 'gte'].includes(op) || ['integer', 'decimal'].includes(type ?? '');
  const scalar = (value: string) => numeric && value.trim() !== '' && Number.isFinite(Number(value)) ? Number(value) : value;
  if (op === 'in' || op === 'not_in') return text.split('|').map((value) => scalar(value.trim()));
  if (op === 'regex') return text;
  if (type === 'boolean' && (text === 'true' || text === 'false')) return text === 'true';
  return scalar(text);
}

function CohortConditions({ value, dictionary, onChange }: {
  value: Condition[]; dictionary: AttributeMapping[]; onChange: (conditions: Condition[]) => void;
}) {
  const columns = [...new Set(['Slide_ID', 'Patient_ID', ...dictionary.map((item) => item.key)])];
  const update = (index: number, change: Partial<Condition>) => onChange(value.map((item, at) => at === index ? { ...item, ...change } : item));
  return <div className="evaluation-conditions">
    <div className="science-subheading"><h3>Include records matching every condition</h3><button type="button" className="btn btn-secondary btn-small" disabled={value.length >= 30} onClick={() => onChange([...value, { field: dictionary[0]?.key ?? 'Slide_ID', op: 'eq', value: '' }])}><Icon name="plus" size={14} />Add condition</button></div>
    {value.length === 0 ? <p className="muted">All records are selected until you add a condition. For a shared metadata file, select the test cohort using its cohort or partition column.</p> : value.map((condition, index) => <div className="evaluation-condition" key={index}>
      <label className="label">Field<select className="field" value={condition.field} onChange={(event) => update(index, { field: event.target.value, op: 'eq', value: '' })}>
        {!columns.includes(condition.field) ? <option value={condition.field}>{condition.field} · unavailable</option> : null}
        {columns.map((column) => <option key={column}>{column}</option>)}
      </select></label>
      <label className="label">Condition<select className="field" value={condition.op} onChange={(event) => { const op = event.target.value as Condition['op']; update(index, { op, value: op === 'exists' ? true : op === 'in' || op === 'not_in' ? [] : '' }); }}>
        {operators.map(([op, label]) => <option key={op} value={op}>{label}</option>)}
      </select></label>
      <label className="label">{condition.op === 'in' || condition.op === 'not_in' ? 'Values, separated by |' : 'Value'}
        {condition.op === 'exists' ? <select className="field" value={String(condition.value)} onChange={(event) => update(index, { value: event.target.value === 'true' })}><option value="true">Is present</option><option value="false">Is missing</option></select>
          : <input className="field" value={Array.isArray(condition.value) ? condition.value.join(' | ') : String(condition.value ?? '')} onChange={(event) => update(index, { value: evaluationConditionValue(event.target.value, condition.op, dictionary.find((item) => item.key === condition.field)?.type) })} />}
      </label>
      <button type="button" className="icon-button" aria-label={`Remove cohort condition ${index + 1}`} onClick={() => onChange(value.filter((_, at) => at !== index))}><Icon name="close" /></button>
    </div>)}
  </div>;
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

export function EvaluationInferenceFields({ value, target, onChange }: {
  value: EvaluationInference; target?: ProtocolSpec['target']; onChange: (update: Partial<EvaluationInference>) => void;
}) {
  return <details><summary>Inference settings</summary><div className="grid-2 evaluation-inference">
    <NumericField label="Batch size" min={1} max={1024} value={value.batchSize} onChange={(batchSize) => onChange({ batchSize })} />
    <NumericField label="Data-loading workers" min={0} max={64} value={value.numWorkers} onChange={(numWorkers) => onChange({ numWorkers })} />
    <label className="label">Device<select className="field" value={value.device} onChange={(event) => onChange({ device: event.target.value as EvaluationInference['device'] })}><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label>
    <label className="label">Inference precision<select className="field" value={value.precision} onChange={(event) => onChange({ precision: event.target.value as EvaluationInference['precision'] })}><option value="float32">Float32</option><option value="float16">Float16</option><option value="bfloat16">BFloat16</option></select></label>
    {target?.unit === 'patient' || value.patientAggregation !== 'mean' ? <label className="label">Combine slides for each patient<select className="field" value={value.patientAggregation} onChange={(event) => onChange({ patientAggregation: event.target.value as EvaluationInference['patientAggregation'] })}>
      {value.patientAggregation !== 'mean' ? <option value={value.patientAggregation} disabled>Maximum probabilities · unsupported; choose mean</option> : null}
      <option value="mean">Mean probabilities</option>
    </select><small>Mean probabilities match the patient aggregation used by frozen predictors.</small></label> : null}
    {target?.task === 'binary_classification' ? <div><NumericField label="Decision threshold" integer={false} min={0} max={1} value={value.decisionThreshold} onChange={(decisionThreshold) => onChange({ decisionThreshold })} /><small>Set from development evidence before reviewing test outcomes.</small></div> : null}
  </div></details>;
}

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

export default function LocalEvaluationSetup({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const protocols = useConfigurations(project, 'protocol');
  const datasets = useDatasets(project);
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const drafts = useQuery({ queryKey: ['evaluation-drafts', project], queryFn: () => evaluation.drafts(project) });
  const frozen = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project) });
  const client = useQueryClient();
  const refreshScientific = useRefreshScientific(project);
  const [spec, setSpec] = useState<EvaluationSpec>(newEvaluationSpec);
  const [mappingRows, setMappingRows] = useState<[string, string][] | null>(null);
  const [name, setName] = useState(`${workspace.project.name} test cohort`);
  const [draft, setDraft] = useState<EvaluationDraft | null>(null);
  const [preview, setPreview] = useState<EvaluationPreview | null>(null);
  const [label, setLabel] = useState<VersionLabelInput>({ tag: '', note: '' });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const operation = useRef<string | null>(null);
  const pending = useRef(false);
  const editor = useRef<HTMLFieldSetElement>(null);
  const protocol = protocols.data?.configurations.find((item) => item.id === spec.protocolId);
  const protocolSpec = protocol?.manifest.spec as ProtocolSpec | undefined;
  const dataset = datasets.data?.datasets.find((item) => item.id === spec.datasetId);
  const dictionary = dataset?.manifest.dictionary ?? [];
  const allBundles = featureBundles.data?.items ?? [];
  const selectedBundle = allBundles.find((item) => item.id === spec.featureBundleId);
  const packIds = selectedBundle?.manifest.spec.packArtifactIds ?? [];
  const labelRows = mappingRows ?? Object.entries(spec.target?.labels ?? {});
  const mappingError = spec.target ? evaluationLabelProblem(labelRows, spec.target.classes) : null;
  const dirty = Boolean(mappingError) || !draft || draft.name !== name.trim() || !sameJSON(draft.payload.spec, spec);
  const ready = Boolean(name.trim() && spec.protocolId && spec.developmentFeatureBundleId && spec.datasetId && spec.featureBundleId && !mappingError);
  const editable = draft?.status !== 'frozen';

  async function refresh() {
    await Promise.all([refreshScientific(), client.invalidateQueries({ queryKey: ['evaluation-drafts', project] }), client.invalidateQueries({ queryKey: ['evaluation-cohorts', project] })]);
  }
  function edit(update: Partial<EvaluationSpec>) {
    if ('protocolId' in update || 'datasetId' in update || update.target === null) setMappingRows(null);
    setSpec((current) => ({ ...current, ...update }));
    setPreview(null); setMessage(''); setError(null); operation.current = null;
  }
  function inference(update: Partial<EvaluationInference>) { edit({ inference: { ...spec.inference, ...update } }); }
  function reset() {
    setMappingRows(null);
    setSpec(newEvaluationSpec()); setName(`${workspace.project.name} test cohort`); setDraft(null);
    setPreview(null); setLabel({ tag: '', note: '' }); setError(null); setMessage(''); operation.current = null;
  }
  async function run(action: () => Promise<void>, validate = false) {
    if (pending.current || (validate && !reportEditorValidity(editor.current))) return;
    pending.current = true; setBusy(true); setError(null); setMessage('');
    try { await action(); } catch (reason) { setError(reason instanceof Error ? reason : new Error('The test cohort could not be saved.')); }
    finally { pending.current = false; setBusy(false); }
  }
  async function save(review: boolean) {
    setPreview(null); operation.current = null;
    const saved = !dirty && editable && draft ? draft : await evaluation.saveDraft(project, name.trim(), spec, editable && draft ? draft : undefined);
    setDraft(saved); setName(saved.name);
    if (review) { setPreview(await evaluation.preview(project, saved)); setMessage('Cohort draft saved and checked. Review the evidence before freezing.'); }
    else setMessage('Test cohort draft saved. You can continue model development and return later.');
    await refresh();
  }

  return <div className="clinical-workspace evaluation-setup">
    <PageHeader eyebrow="03 EVALUATE · TEST COHORT" title="Test cohorts"
      description="Prepare the test cohort and feature inputs while models are being developed, or after training is complete."
      actions={<button type="button" className="btn btn-secondary" disabled={busy} onClick={reset}><Icon name="plus" />New test cohort</button>} />
    <ModuleSteps input="A development protocol, its feature bundle, and a frozen metadata dataset."
      procedure="Select test records, check feature coverage and development overlap, then save inference settings."
      output="A frozen test cohort ready to bind to a compatible predictor." />
    <div className="science-toolbar">
      <label className="label">Saved test cohort drafts<select className="field" value={draft?.id ?? ''} disabled={busy} onChange={(event) => {
        const id = event.target.value; if (!id) { reset(); return; }
        void run(async () => { const saved = await evaluation.draft(project, id); if (saved.payload.type !== 'evaluation-cohort') throw new Error('This is not a test cohort draft.'); setDraft(saved); setName(saved.name); setSpec(saved.payload.spec); setMappingRows(null); setPreview(null); setLabel({ tag: '', note: '' }); operation.current = null; });
      }}><option value="">Start a new draft</option>{(drafts.data?.drafts ?? []).map((item) => <option key={item.id} value={item.id}>{item.name} · revision {item.revision} · {item.status}</option>)}</select></label>
      <Badge>{draft ? `${draft.status} · revision ${draft.revision}${dirty ? ' · unsaved changes' : ''}` : 'New cohort'}</Badge>
    </div>
    <ErrorNotice error={error ?? protocols.error ?? datasets.error ?? featureBundles.error ?? drafts.error ?? frozen.error} />
    <SavedNotice>{message}</SavedNotice>
    <fieldset ref={editor} className="evaluation-fields" disabled={busy}>
      <legend className="sr-only">Test cohort preparation</legend>
      <Panel title="1. Inherit the development target" subtitle="The protocol supplies the prediction task, classes and unit.">
        <div className="stack">
          <label className="label">Cohort name<input className="field" maxLength={120} value={name} onChange={(event) => { setName(event.target.value); setMessage(''); }} /></label>
          <label className="label">Development protocol<select className="field" value={spec.protocolId} onChange={(event) => {
            const selected = protocols.data?.configurations.find((item) => item.id === event.target.value);
            const inherited = (selected?.manifest.spec as ProtocolSpec | undefined)?.target;
            edit({ protocolId: event.target.value, developmentFeatureBundleId: '', datasetId: selected?.manifest.datasetId ?? '', featureBundleId: '', eligibility: [], target: inherited ? { ...inherited, classes: [...inherited.classes], labels: { ...inherited.labels } } : null, patientIdentifiers: 'shared', inference: { ...spec.inference, loadingPolicy: 'per_slide', packArtifactId: null } });
          }}><option value="">{protocols.isPending ? 'Loading protocols…' : 'Choose a frozen development protocol'}</option>{spec.protocolId && !protocol ? <option value={spec.protocolId} disabled>Saved protocol unavailable</option> : null}{(protocols.data?.configurations ?? []).map((item) => { const legacy = (item.manifest.spec as ProtocolSpec).split.version !== 4; return <option key={item.id} value={item.id} disabled={legacy}>{configurationVersionLabel(item)}{legacy ? ' · update in Targets & splits' : ''}</option>; })}</select></label>
          {protocolSpec ? <div className="evaluation-inherited" aria-label="Inherited prediction target"><div><span>Task</span><strong>{protocolSpec.target.task === 'binary_classification' ? 'Binary classification' : 'Multiclass classification'}</strong></div><div><span>Target</span><strong>{protocolSpec.target.field}</strong></div><div><span>Prediction unit</span><strong>{protocolSpec.target.unit}</strong></div><div><span>Class order</span><strong>{protocolSpec.target.classes.join(' → ')}</strong></div>{protocolSpec.target.positiveClass ? <div><span>Positive class</span><strong>{protocolSpec.target.positiveClass}</strong></div> : null}</div> : <p className="muted">Freeze a target in <a href="#cohort">Targets &amp; splits</a> to begin.</p>}
          <label className="label">Development feature bundle<select className="field" disabled={!protocol} value={spec.developmentFeatureBundleId} onChange={(event) => edit({ developmentFeatureBundleId: event.target.value })}>
            <option value="">Choose the bundle used for model development</option>
            {spec.developmentFeatureBundleId && !allBundles.some((item) => item.id === spec.developmentFeatureBundleId) ? <option value={spec.developmentFeatureBundleId} disabled>Saved development bundle unavailable</option> : null}
            {allBundles.filter((item) => item.manifest.datasetId === protocol?.manifest.datasetId).map((item) => <option key={item.id} value={item.id} disabled={!item.current}>{versionLabelText(item, 'Feature bundle')}{!item.current ? ' · needs verification' : ''}</option>)}
          </select><small>Test features will be checked against this representation.</small></label>
        </div>
      </Panel>
      <Panel title="2. Select the test cohort" subtitle="Use the same metadata file with a test-cohort filter, or a separately imported dataset.">
        <div className="stack">
          <DatasetSelect versions={datasets.data?.datasets ?? []} value={spec.datasetId} disabled={!protocol} allowEmpty onChange={(datasetId) => edit({ datasetId, featureBundleId: '', eligibility: [], patientIdentifiers: 'shared', inference: { ...spec.inference, loadingPolicy: 'per_slide', packArtifactId: null } })} />
          <p className="muted">The development dataset is selected first so shared IDs and label definitions stay consistent. To use another CSV or XLSX file, <a href="#dataset">import and freeze it in Data</a>, then select it here.</p>
          {spec.datasetId ? <CohortConditions value={spec.eligibility} dictionary={dictionary} onChange={(eligibility) => edit({ eligibility })} /> : null}
          <p className="callout">This stage selects evaluation records. It does not generate training, validation or cross-validation splits.</p>
          {spec.datasetId && spec.datasetId !== protocol?.manifest.datasetId ? <label className="label">Patient identifiers across datasets<select className="field" value={spec.patientIdentifiers} onChange={(event) => edit({ patientIdentifiers: event.target.value as EvaluationSpec['patientIdentifiers'] })}><option value="shared">Shared IDs identify the same patients</option><option value="independent">IDs belong to separate naming systems</option></select><small>Use separate naming systems only when the same ID can mean different people. This choice cannot establish whether patients overlap.</small></label> : null}
          {protocolSpec ? <label className="evaluation-checkbox"><input type="checkbox" checked={spec.target !== null} onChange={(event) => edit({ target: event.target.checked ? { ...protocolSpec.target, classes: [...protocolSpec.target.classes], labels: { ...protocolSpec.target.labels } } : null })} /><span>Dataset includes target labels for later performance evaluation</span></label> : null}
          {spec.target ? <details className="evaluation-labels"><summary>Review label mapping for this dataset</summary><div className="stack">
            <p className="muted">The task and class order are inherited. Match the label column and raw values in this dataset to those same classes.</p>
            <label className="label">Label column<select className="field" value={spec.target.field} onChange={(event) => edit({ target: { ...spec.target!, field: event.target.value } })}>{!dictionary.some((item) => item.key === spec.target!.field) ? <option value={spec.target.field}>{spec.target.field}{dataset ? ' · column not found' : ''}</option> : null}{dictionary.map((item) => <option key={item.key}>{item.key}</option>)}</select></label>
            <EvaluationTargetMapping rows={labelRows} classes={spec.target.classes} onChange={(rows) => { setMappingRows(rows); edit({ target: { ...spec.target!, labels: Object.fromEntries(rows) } }); }} />
            {mappingError ? <p className="callout callout-warning" role="alert">{mappingError}</p> : null}
            <p className="muted">Map this file’s raw values, such as 0 and 1, to the inherited classes. Task, class order and positive class stay fixed.</p>
            <label className="label">Missing target labels<select className="field" value={spec.target.missing} onChange={(event) => edit({ target: { ...spec.target!, missing: event.target.value as 'block' | 'exclude' } })}><option value="block">Block until labels are complete</option><option value="exclude">Exclude unlabeled records</option></select></label>
            <label className="label">Unmapped target values<select className="field" value={spec.target.unmapped} onChange={(event) => edit({ target: { ...spec.target!, unmapped: event.target.value as 'block' | 'exclude' } })}><option value="block">Block until values are mapped</option><option value="exclude">Exclude records with unmapped values</option></select></label>
          </div></details> : <p className="muted">An unlabeled cohort can be prepared for predictions; outcome-based metrics require labels.</p>}
        </div>
      </Panel>
      <Panel title="3. Match features & inference settings" subtitle="Choose verified features covering every selected test slide.">
        <div className="stack">
          <label className="label">Test feature bundle<select className="field" value={spec.featureBundleId} disabled={!spec.datasetId} onChange={(event) => edit({ featureBundleId: event.target.value, inference: { ...spec.inference, loadingPolicy: 'per_slide', packArtifactId: null } })}>
            <option value="">Choose test features</option>{spec.featureBundleId && !selectedBundle ? <option value={spec.featureBundleId} disabled>Saved test bundle unavailable</option> : null}
            {allBundles.map((item) => <option key={item.id} value={item.id} disabled={!item.current}>{versionLabelText(item, 'Feature bundle')} · {item.manifest.summary.dimensions ?? '?'} dimensions{item.manifest.datasetId === spec.datasetId ? ' · selected dataset' : ' · match by slide ID'}{!item.current ? ' · needs verification' : ''}</option>)}
          </select></label>
          <p className="muted">Prepare test features in <a href="#features">Features</a>. Review checks the actual selected slide IDs, representation and pack coverage.</p>
          <div className="grid-2">
            <label className="label">Feature loading<select className="field" value={spec.inference.loadingPolicy} onChange={(event) => inference({ loadingPolicy: event.target.value as EvaluationInference['loadingPolicy'], packArtifactId: null })}><option value="per_slide">Original feature files</option><option value="packed" disabled={packIds.length === 0}>Packed mmap</option></select></label>
            {spec.inference.loadingPolicy === 'packed' ? <label className="label">Pack in test bundle<select className="field" value={spec.inference.packArtifactId ?? ''} onChange={(event) => inference({ packArtifactId: event.target.value || null })}><option value="">Choose a verified pack</option>{spec.inference.packArtifactId && !packIds.includes(spec.inference.packArtifactId) ? <option value={spec.inference.packArtifactId} disabled>Saved pack unavailable</option> : null}{packIds.map((id) => { const pack = selectedBundle?.manifest.packs.find((item) => item.id === id); return <option key={id} value={id}>{pack ? `${pack.outputDtype} · ${pack.outputPath}` : id}</option>; })}</select></label> : null}
          </div>
          <EvaluationInferenceFields value={spec.inference} target={protocolSpec?.target} onChange={inference} />
        </div>
      </Panel>
      <Panel title="4. Review & freeze the test cohort" subtitle="Persist the exact cohort, label mapping, feature references and inference settings.">
        {preview ? <EvaluationEvidence preview={preview} /> : <p className="muted">Review saves the draft, checks selected slides and reports any development overlap or incompatible features.</p>}
        <div className="inline-actions evaluation-actions"><button type="button" className="btn btn-secondary" disabled={!name.trim() || Boolean(mappingError) || (!dirty && editable)} onClick={() => void run(() => save(false), true)}>Save draft</button><button type="button" className="btn btn-primary" disabled={!ready} onClick={() => void run(() => save(true), true)}><Icon name="check" />{busy ? 'Checking…' : 'Review cohort'}</button></div>
        {preview?.canFreeze && !dirty && editable ? <div className="evaluation-freeze">
          <label className="label">Version tag<input className="field" value={label.tag} maxLength={80} placeholder="e.g. held-out-cohort-v1" onChange={(event) => { setLabel({ ...label, tag: event.target.value }); operation.current = null; }} /></label>
          <label className="label">Version note (optional)<textarea className="field" value={label.note} maxLength={2000} rows={2} onChange={(event) => { setLabel({ ...label, note: event.target.value }); operation.current = null; }} /></label>
          <button type="button" className="btn btn-primary" disabled={!label.tag.trim()} onClick={() => void run(async () => {
            if (!draft) return;
            operation.current ??= crypto.randomUUID();
            const saved = await evaluation.freeze(project, draft, preview.previewHash, operation.current, { tag: label.tag.trim(), note: label.note });
            setDraft({ ...draft, status: 'frozen', revision: draft.revision + 1 });
            setMessage(`Test cohort frozen as ${versionLabelText(saved, 'Test cohort')}. It is ready to evaluate all compatible predictors.`);
            await refresh();
          }, true)}><Icon name="lock" />Save test cohort</button>
        </div> : null}
        <p className="callout evaluation-availability">After saving this test cohort, choose ready predictors from your experiments in Evaluate models to run inference and review results.</p>
      </Panel>
    </fieldset>
    <Panel title="Frozen test cohorts" subtitle="Reuse a saved cohort or create another version when its records or settings change.">
      {frozen.isPending ? <p className="muted">Loading frozen test cohorts…</p> : frozen.data?.items.length ? <div className="table-wrap"><table><thead><tr><th>Version</th><th>Slides</th><th>Patient groups</th><th>Action</th></tr></thead><tbody>{frozen.data.items.map((item) => <tr key={item.id}><td>{versionLabelText(item, 'Test cohort')}{item.current === false ? <Badge tone="orange">Needs review</Badge> : null}{item.versionLabel?.note ? <small className="evaluation-version-note">{item.versionLabel.note}</small> : null}</td><td>{item.manifest.summary.includedSlides.toLocaleString()}</td><td>{item.manifest.summary.includedPatients.toLocaleString()}</td><td><button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => void run(async () => { const saved = await evaluation.get(project, item.id); setSpec(saved.manifest.spec); setMappingRows(null); setName(`${versionLabelText(saved, 'Test cohort')} copy`); setDraft(null); setPreview(null); setLabel({ tag: '', note: '' }); operation.current = null; setMessage('Loaded as a new draft. The frozen cohort remains available.'); })}>Use as new draft</button>{item.current ? <a className="btn btn-primary btn-small" href={`#evaluation?cohort=${encodeURIComponent(item.id)}`}>Evaluate experiment predictors</a> : null}</td></tr>)}</tbody></table></div> : <p className="muted">No frozen test cohorts yet. A trained model is not required to prepare one.</p>}
    </Panel>
  </div>;
}
