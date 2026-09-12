import { useEffect, useId, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { development, defaultRecipe, defaultResources, parseNumberList, withRecipeDefaults } from '../api/development';
import type { BatchPreview, DevelopmentBatchSpec, TrainingRecipe } from '../api/development';
import { experiments, type ExperimentBatch, type ExperimentBatchPlan, type ExperimentStage, type ModelExperiment } from '../api/experiments';
import ExperimentLifecycle from './ExperimentLifecycle';
import type { ScientificDraft } from '../api/scientific';
import type { MILExperimentSpec } from '../api/mil';
import { Badge, ErrorNotice, Panel } from './ui';
import { Findings, SavedNotice } from './ScientificUI';
import { downloadJSON } from '../lib/download';
import { sameJSON } from '../lib/json';
import './DevelopmentBatches.css';
import DevelopmentExecution from './DevelopmentExecution';
import NumericField, { reportEditorValidity } from './NumericField';
import TrainingCapacity from './TrainingCapacity';

export type DevelopmentTab = 'setup' | 'batches' | 'runs' | 'results';
export const developmentTabs: { id: DevelopmentTab; label: string }[] = [
  { id: 'setup', label: 'Inputs' }, { id: 'batches', label: 'Batches' }, { id: 'runs', label: 'Runs' },
  { id: 'results', label: 'Results' },
];

export function batchVersionTag(name: string, experimentId: string) {
  const suffix = ` · ${experimentId}`;
  if (suffix.length >= 80) throw new Error('The experiment ID is too long for a batch version tag.');
  return `${name.trim().slice(0, 80 - suffix.length)}${suffix}`;
}

export function updateBatchPlans(plans: ExperimentBatchPlan[], plan: ExperimentBatchPlan): ExperimentBatchPlan[] {
  return plans.some((item) => item.id === plan.id) ? plans.map((item) => item.id === plan.id ? plan : item) : [...plans, plan];
}

export const batchTemplates = [
  { id: 'baseline', name: 'ABMIL baseline', description: 'One configuration with standard training settings.' },
  { id: 'quick', name: 'Quick check', description: 'Five epochs and smaller sampled bags to check the training setup.' },
  { id: 'learning-rate', name: 'Learning-rate comparison', description: 'Compare three learning rates with the same folds and training seed.' },
] as const;

export function batchTemplate(id: string, inputs: MILExperimentSpec, experimentName: string): DevelopmentBatchSpec {
  const recipe = { ...defaultRecipe(), ...(id === 'quick' ? { maxEpochs: 5, bagSize: 1024, patience: 3 } : {}) };
  return { version: 1, experimentName, batchName: batchTemplates.find((item) => item.id === id)?.name ?? 'Baseline', inputs,
    recipe, mode: id === 'learning-rate' ? 'grid' : 'single', grid: { learningRates: id === 'learning-rate' ? [0.0001, 0.0003, 0.001] : [recipe.learningRate], weightDecays: [recipe.weightDecay], maxEpochs: [recipe.maxEpochs] },
    configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '' };
}

export function RecipeFields({ value, onChange, gridMode = false }: { value: TrainingRecipe; onChange: (value: TrainingRecipe) => void; gridMode?: boolean }) {
  const resolved = withRecipeDefaults(value);
  const bagModeId = useId();
  const [sampledBagSize, setSampledBagSize] = useState(value.bagSize ?? 4096);
  return <><div className="development-fields">
    {!gridMode ? <>
    <NumericField label="Learning rate" value={value.learningRate} integer={false} min={0} minExclusive onChange={(learningRate) => onChange({ ...value, learningRate })} />
    <NumericField label="Weight decay" value={value.weightDecay} integer={false} min={0} onChange={(weightDecay) => onChange({ ...value, weightDecay })} />
    <NumericField label="Maximum epochs" value={value.maxEpochs} min={1} max={100000} onChange={(maxEpochs) => onChange({ ...value, maxEpochs })} />
    </> : null}
    <NumericField label="Batch size" value={value.batchSize} min={1} max={4096} onChange={(batchSize) => onChange({ ...value, batchSize })} />
    <fieldset className="development-bag-mode"><legend>Training bag</legend>
      <label className="development-check"><input type="radio" name={bagModeId} checked={value.bagSize !== null} onChange={() => onChange({ ...value, bagSize: sampledBagSize })} /> Sample patches per bag</label>
      <label className="development-check"><input type="radio" name={bagModeId} checked={value.bagSize === null} onChange={() => { if (value.bagSize !== null) setSampledBagSize(value.bagSize); onChange({ ...value, bagSize: null }); }} /> Use whole bag for training</label>
      {value.bagSize === null ? <p className="development-bag-help">Train with every available patch in each slide. No patch sampling is applied. Larger bags require more memory.</p>
        : <NumericField label="Patches per bag" value={value.bagSize} min={1} max={1000000} onChange={(bagSize) => onChange({ ...value, bagSize })} />}
      <small>Validation and assessment always use whole bags.</small>
    </fieldset>
    <NumericField label="Early-stopping patience" value={value.patience} min={1} max={10000} disabled={!value.earlyStopping} onChange={(patience) => onChange({ ...value, patience })} />
    <label className="label">Checkpoint selection<select className="field" value={value.checkpointMetric} onChange={(e) => onChange({ ...value, checkpointMetric: e.target.value as TrainingRecipe['checkpointMetric'] })}><option value="validation_loss">Lowest validation loss</option><option value="validation_auroc">Highest validation AUROC</option><option value="validation_accuracy">Highest validation accuracy</option></select></label>
    <label className="development-check"><input type="checkbox" checked={value.earlyStopping} onChange={(e) => onChange({ ...value, earlyStopping: e.target.checked })} /> Use early stopping</label>
  </div><details className="development-architecture setup-details"><summary>Advanced model &amp; training settings <span className="muted">· ABMIL · {resolved.precision === '32-true' ? 'FP32' : resolved.precision} · {resolved.lrScheduler === 'cosine' ? 'Cosine learning rate' : 'Constant learning rate'}</span></summary>
  <div className="development-fields">
    <label className="label">Model<select className="field" value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })}><option value="abmil">ABMIL</option>{value.model !== 'abmil' ? <option value={value.model} disabled>{value.model} (unavailable)</option> : null}</select></label>
    <label className="label">Optimizer<select className="field" value={value.optimizer} onChange={(e) => onChange({ ...value, optimizer: e.target.value as TrainingRecipe['optimizer'] })}><option value="adamw">AdamW</option><option value="adam">Adam</option><option value="sgd">SGD</option></select></label>
    <label className="label">Precision<select className="field" value={resolved.precision} onChange={(e) => onChange({ ...value, precision: e.target.value as TrainingRecipe['precision'] })}><option value="32-true">FP32 (standard)</option><option value="16-mixed">FP16 mixed (CUDA)</option><option value="bf16-mixed">BF16 mixed (supported device)</option></select><small>Mixed precision can reduce GPU memory use. Device support is checked before training.</small></label>
    <NumericField label="Accumulate batches" value={resolved.accumulateGradBatches!} min={1} max={4096} onChange={(accumulateGradBatches) => onChange({ ...value, accumulateGradBatches })} />
    <NumericField label="Gradient clipping norm" value={resolved.gradientClipNorm!} min={0} integer={false} onChange={(gradientClipNorm) => onChange({ ...value, gradientClipNorm })} />
    <label className="label">Learning-rate schedule<select className="field" value={resolved.lrScheduler} onChange={(e) => onChange({ ...value, lrScheduler: e.target.value as TrainingRecipe['lrScheduler'], warmupEpochs: e.target.value === 'none' ? 0 : resolved.warmupEpochs })}><option value="none">Constant learning rate</option><option value="cosine">Cosine decay</option></select></label>
    {resolved.lrScheduler === 'cosine' ? <>
      <NumericField label="Warmup epochs" value={resolved.warmupEpochs!} min={0} max={gridMode ? 99999 : value.maxEpochs - 1} onChange={(warmupEpochs) => onChange({ ...value, warmupEpochs })} />
      <NumericField label="Final LR fraction" value={resolved.finalLrFraction!} min={0} minExclusive max={1} integer={false} onChange={(finalLrFraction) => onChange({ ...value, finalLrFraction })} />
    </> : null}
    <NumericField label="Minimum training epochs" value={resolved.minEpochs!} min={1} max={gridMode ? 100000 : value.maxEpochs} onChange={(minEpochs) => onChange({ ...value, minEpochs })} />
    <NumericField label="Early-stopping minimum improvement" value={resolved.earlyStoppingMinDelta!} min={0} integer={false} disabled={!value.earlyStopping} onChange={(earlyStoppingMinDelta) => onChange({ ...value, earlyStoppingMinDelta })} />
  </div>
  <p className="muted">Gradient clipping 0 disables clipping. Accumulation combines several batches per optimizer update. Minimum epochs delay stopping; the best validation checkpoint can still come from an earlier epoch.</p>
  <div className="development-fields">
    {([['embedDim', 'Embedding dimensions', 1], ['attentionDim', 'Attention dimensions', 1], ['numFcLayers', 'Fully connected layers', 1], ['dropout', 'Dropout', 0], ['inputDropout', 'Input dropout', 0]] as const).map(([key, label, min]) => <NumericField key={key} label={label} value={resolved[key]!} min={min} max={key === 'dropout' || key === 'inputDropout' ? 1 : key === 'numFcLayers' ? 8 : 8192} maxExclusive={key === 'dropout' || key === 'inputDropout'} integer={key !== 'dropout' && key !== 'inputDropout'} onChange={(number) => onChange({ ...value, [key]: number })} />)}
    <label className="development-check"><input type="checkbox" checked={resolved.gatedAttention} onChange={(e) => onChange({ ...value, gatedAttention: e.target.checked })} /> Gated attention</label>
    <label className="development-check"><input type="checkbox" checked={resolved.gradientCheckpointing} onChange={(e) => onChange({ ...value, gradientCheckpointing: e.target.checked })} /> Gradient checkpointing</label>
  </div></details></>;
}

export default function DevelopmentBatches({ project, inputs, experimentName, experimentId, experimentRevision, ownedBatches, ownedDrafts, executionImplemented = false, readOnly = false, record, experimentStage, onPlanDirtyChange, tab, onOpenSetup, onRestoreInputs }: {
  project: string; inputs: MILExperimentSpec; experimentName: string; experimentId: string; experimentRevision: number;
  record?: ModelExperiment; experimentStage?: ExperimentStage; onPlanDirtyChange?: (dirty: boolean) => void;
  ownedBatches: ExperimentBatch[]; ownedDrafts: ScientificDraft[]; executionImplemented?: boolean; readOnly?: boolean; tab: Exclude<DevelopmentTab, 'setup'>; onOpenSetup: () => void;
  onRestoreInputs: (inputs: MILExperimentSpec, name: string) => void;
}) {
  const client = useQueryClient();
  const runtime = useQuery({ queryKey: ['training-runtime', project], queryFn: () => development.runtime(project), enabled: tab === 'batches', staleTime: 30000 });
  const [name, setName] = useState('Baseline');
  const [recipe, setRecipe] = useState(defaultRecipe);
  const [resources, setResources] = useState(defaultResources);
  const [mode, setMode] = useState<DevelopmentBatchSpec['mode']>('single');
  const [rows, setRows] = useState(() => [{ id: 0, recipe: defaultRecipe() }]);
  const nextRowId = useRef(1);
  const [seeds, setSeeds] = useState('42');
  const [lrs, setLrs] = useState('0.0001, 0.0003, 0.001');
  const [wds, setWds] = useState('0, 0.0001');
  const [epochs, setEpochs] = useState('50, 100');
  const [gpus, setGpus] = useState('0');
  const [notes, setNotes] = useState('');
  const [preview, setPreview] = useState<BatchPreview | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState('');
  const [workingPlan, setWorkingPlan] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [editorRevision, setEditorRevision] = useState(experimentRevision);
  const busyRef = useRef(false);
  const editor = useRef<HTMLFieldSetElement>(null);
  const [editorVersion, setEditorVersion] = useState(0);
  const items = ownedBatches;
  const selectedBatch = items.find((item) => item.id === selected) ?? (!selected && items.length === 1 ? items[0] : undefined);
  const savedDrafts = ownedDrafts.filter((draft) => draft.payload.type === 'development-batch');
  const plans = record?.batchPlans ?? [];
  const locked = readOnly || (experimentStage !== undefined && experimentStage !== 'planning');
  const stale = Boolean(workingPlan || dirty) && editorRevision !== experimentRevision;
  useEffect(() => { onPlanDirtyChange?.(dirty || stale); }, [dirty, stale, onPlanDirtyChange]);
  useEffect(() => () => onPlanDirtyChange?.(false), [onPlanDirtyChange]);
  let gpuSelection: number[] = [];
  let gpuSelectionError = '';
  try {
    gpuSelection = gpus.trim() ? parseNumberList(gpus, 'GPU IDs', true) : [];
    if (gpuSelection.some((id) => id > 127)) throw new Error('GPU IDs must be between 0 and 127.');
  } catch (reason) { gpuSelectionError = reason instanceof Error ? reason.message : 'Enter valid GPU IDs.'; }

  function edit() { if (!dirty && !workingPlan) setEditorRevision(experimentRevision); setDirty(true); setPreview(null); setMessage(''); setError(null); }
  function specification(): DevelopmentBatchSpec {
    return { version: 1, experimentId, experimentRevision, experimentName, batchName: name.trim(), inputs, recipe, mode,
      grid: mode === 'grid' ? { learningRates: parseNumberList(lrs, 'Learning rates', false, Number.MIN_VALUE), weightDecays: parseNumberList(wds, 'Weight decay'), maxEpochs: parseNumberList(epochs, 'Maximum epochs', true, 1) } : { learningRates: [recipe.learningRate], weightDecays: [recipe.weightDecay], maxEpochs: [recipe.maxEpochs] },
      configurations: mode === 'explicit' ? rows.map((row) => row.recipe) : [], trainingSeeds: parseNumberList(seeds, 'Training seeds', true),
      resources: { ...resources, gpuIds: gpus.trim() ? parseNumberList(gpus, 'GPU IDs', true) : [] }, notes };
  }
  async function savePlans(next: ExperimentBatchPlan[]) {
    if (!record) throw new Error('Reload this experiment before saving its batch plans.');
    const saved = await experiments.update(project, experimentId, { name: record.name, batchPlans: next, expectedRevision: experimentRevision });
    client.setQueryData(['model-experiment', project, experimentId], saved);
    await client.invalidateQueries({ queryKey: ['model-experiments', project] });
    setEditorRevision(saved.revision);
    return saved;
  }
  async function action(kind: 'preview' | 'save') {
    if (locked || stale || busyRef.current || !reportEditorValidity(editor.current)) return;
    busyRef.current = true;
    setBusy(true); setError(null); setMessage('');
    try {
      const spec = specification();
      if (kind === 'save') {
        const id = workingPlan ?? crypto.randomUUID();
        if (!dirty && !workingPlan) setEditorRevision(experimentRevision);
        setWorkingPlan(id);
        await savePlans(updateBatchPlans(plans, { id, spec }));
        setDirty(false); setPreview(null);
        setMessage('Batch plan saved. It remains editable until you submit this experiment.');
      } else {
        const result = await development.preview(project, spec); setPreview(result);
      }
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Batch could not be saved.')); }
    finally { busyRef.current = false; setBusy(false); }
  }
  function load(spec: DevelopmentBatchSpec, planId?: string, copy = false) {
    if (dirty && !window.confirm('Discard the unsaved batch edits and open this configuration?')) return false;
    setEditorVersion((version) => version + 1);
    if (!sameJSON(inputs, spec.inputs)) onRestoreInputs(spec.inputs, spec.experimentName);
    setWorkingPlan(planId ?? null); setEditorRevision(experimentRevision);
    setName(copy ? `${spec.batchName.slice(0, 70)} copy` : spec.batchName); setRecipe(withRecipeDefaults(spec.recipe)); setResources(spec.resources); setMode(spec.mode);
    setRows((spec.configurations.length ? spec.configurations : [spec.recipe]).map((value) => ({ id: nextRowId.current++, recipe: withRecipeDefaults(value) }))); setSeeds(spec.trainingSeeds.join(', '));
    setLrs(spec.grid.learningRates.join(', ')); setWds(spec.grid.weightDecays.join(', ')); setEpochs(spec.grid.maxEpochs.join(', '));
    setGpus(spec.resources.gpuIds.join(', ')); setNotes(spec.notes); setDirty(!planId); setPreview(null); setMessage(''); setError(null);
    return true;
  }
  async function removePlan(id: string) {
    if (locked || busyRef.current || dirty || stale) return;
    busyRef.current = true; setBusy(true); setError(null);
    try { await savePlans(plans.filter((item) => item.id !== id)); if (workingPlan === id) setWorkingPlan(null); setMessage('Batch removed from the plan.'); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Batch could not be removed.')); }
    finally { busyRef.current = false; setBusy(false); }
  }
  if (experimentStage === 'planning' && (tab === 'runs' || tab === 'results')) return <p className="callout">Runs unlock after submission. Results unlock when the experiment finishes.</p>;
  return <div className="development-batches">
    <ErrorNotice error={error} /><SavedNotice>{tab === 'batches' ? message : ''}</SavedNotice>
    {tab !== 'batches' && items.length > 1 ? <ExperimentBatchOverview batches={items} view={tab} onSelect={setSelected} /> : null}
    {tab === 'batches' && plans.length ? <Panel title={`Batch plans (${plans.length})`} subtitle={locked ? 'These settings were locked when the experiment was submitted.' : 'Edit or remove a batch before submission. All batches use the experiment’s saved inputs.'}>
      <div className="batch-plan-list">{plans.map((plan) => <article key={plan.id} className={`batch-plan-card${workingPlan === plan.id ? ' is-editing' : ''}`}>
        <div><h3>{plan.spec.batchName}</h3><p className="muted">{plan.spec.mode === 'grid' ? 'Parameter grid' : plan.spec.mode === 'explicit' ? `${plan.spec.configurations.length} configurations` : 'Single configuration'} · {plan.spec.trainingSeeds.length} training seed{plan.spec.trainingSeeds.length === 1 ? '' : 's'}</p></div>
        {!locked ? <div className="inline-actions"><button className="btn btn-secondary btn-small" disabled={busy} onClick={() => load(plan.spec, plan.id)}>Edit batch</button><button className="text-button" disabled={busy} onClick={() => load(plan.spec, undefined, true)}>Duplicate</button><button className="text-button" disabled={busy || dirty || stale} onClick={() => void removePlan(plan.id)}>Remove</button></div> : <Badge>Locked</Badge>}
        <details className="batch-plan-spec"><summary>View settings</summary><pre className="experiment-snapshot">{JSON.stringify(plan.spec, null, 2)}</pre></details>
      </article>)}</div>
    </Panel> : null}
    {tab === 'batches' && !locked ? <>
      <Panel title={workingPlan ? `Edit batch: ${name || 'Untitled'}` : 'Add a training batch'} subtitle="Choose one setup or compare parameter combinations. Save your batches, then submit the experiment when the plan is ready.">
        <div className="batch-template-picker"><label className="label">Start from a template<select className="field" value="" disabled={busy} onChange={(event) => { if (event.target.value) load(batchTemplate(event.target.value, inputs, experimentName)); }}><option value="">Choose a template…</option>{batchTemplates.map((template) => <option key={template.id} value={template.id}>{template.name}</option>)}</select></label><p className="muted">Templates fill this editor; you can adjust every setting before saving.</p></div>
        {!inputs.protocolId || !inputs.featureBundleId ? <p className="callout">Choose prepared targets and features in <button type="button" className="text-button" onClick={onOpenSetup}>Inputs</button> first.</p> : <p className="development-input-summary">Inputs selected for <strong>{experimentName}</strong>. <button type="button" className="text-button" onClick={onOpenSetup}>Review inputs</button></p>}
        {stale ? <p role="alert" className="callout">The saved experiment changed while you were editing. <button className="text-button" onClick={() => load(plans.find((plan) => plan.id === workingPlan)?.spec ?? batchTemplate('baseline', inputs, experimentName), plans.find((plan) => plan.id === workingPlan)?.id)}>Reload the saved plan</button> before saving.</p> : null}
        <fieldset key={editorVersion} ref={editor} disabled={busy || stale} className="development-editor" onChange={edit}>
          <legend className="sr-only">Batch configuration</legend>
          <div className="development-fields"><label className="label">Batch name<input required className="field" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></label>
          <label className="label">Configuration mode<select className="field" value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}><option value="single">Single configuration</option><option value="grid">Parameter grid</option><option value="explicit">Explicit configuration rows</option></select></label></div>
          {mode !== 'explicit' ? <RecipeFields value={recipe} onChange={setRecipe} gridMode={mode === 'grid'} /> : <div className="stack">{rows.map((row, index) => <div className="development-recipe-row" key={row.id}><strong>Configuration {index + 1}</strong><RecipeFields value={row.recipe} onChange={(value) => setRows((current) => current.map((item) => item.id === row.id ? { ...item, recipe: value } : item))} /><button type="button" className="text-button" disabled={rows.length === 1} onClick={() => { setRows((current) => current.filter((item) => item.id !== row.id)); edit(); }}>Remove configuration</button></div>)}<button type="button" className="btn btn-secondary" onClick={() => { const id = nextRowId.current++; setRows((current) => [...current, { id, recipe: { ...current[current.length - 1].recipe } }]); edit(); }}>Add configuration</button></div>}
          {mode === 'grid' ? <div className="development-fields">{([['Learning rates', lrs, setLrs], ['Weight decays', wds, setWds], ['Maximum epochs', epochs, setEpochs]] as const).map(([label, value, setter]) => <label className="label" key={label}>{label}<input className="field" value={value} onChange={(e) => setter(e.target.value)} /><small>Comma-separated values; these replace the corresponding base recipe values.</small></label>)}</div> : null}
          <label className="label">Training seeds<input className="field" value={seeds} onChange={(e) => setSeeds(e.target.value)} /><small>Comma-separated seeds. Frozen split assignments stay unchanged.</small></label>
          <section className="development-resource-settings" aria-label="Parallel training"><h3>Parallel training</h3><div className="development-fields">
            <label className="label">Run on<select className="field" value={gpus.trim() ? 'gpu' : 'cpu'} onChange={(e) => setGpus(e.target.value === 'cpu' ? '' : '0')}><option value="gpu">GPU</option><option value="cpu">CPU</option></select></label>
            <NumericField label="Concurrent runs" value={resources.maxConcurrentRuns} min={1} max={128} onChange={(maxConcurrentRuns) => setResources((current) => ({ ...current, maxConcurrentRuns }))} />
            {gpus.trim() ? <NumericField label="Runs per GPU" value={resources.runsPerGpu} min={1} max={16} onChange={(runsPerGpu) => setResources((current) => ({ ...current, runsPerGpu }))} /> : null}
          </div>{gpuSelectionError ? <p className="callout" role="status">{gpuSelectionError}</p> : <TrainingCapacity resources={{ ...resources, gpuIds: gpuSelection }} runtime={runtime.data} />}
          <details className="setup-details"><summary>Advanced resource settings</summary><p className="muted">CPU threads run model operations; data workers load features. RAM is a scheduling reservation per run, not an enforced memory cap.</p><div className="development-fields">
            <label className="label">Allowed GPU IDs<input className="field" value={gpus} onChange={(e) => setGpus(e.target.value)} /><small>Comma-separated; leave empty for CPU.</small></label>
            {([['cpuThreadsPerRun', 'CPU threads per run', 1, 256], ['dataLoaderWorkers', 'Data workers per loader', 0, 64], ['ramGbPerRun', 'RAM reservation per run (GiB)', Number.MIN_VALUE, undefined]] as const).map(([key, label, min, max]) => <NumericField key={key} label={label} value={resources[key]} min={min} max={max} integer={key !== 'ramGbPerRun'} onChange={(number) => setResources((current) => ({ ...current, [key]: number }))} />)}
          </div></details></section>
          <details className="setup-details"><summary>Batch notes (optional)</summary><label className="label">Notes<textarea className="field" value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} /></label></details>
        </fieldset>
        <div className="inline-actions"><button type="button" className="btn btn-primary" disabled={busy || stale || !inputs.protocolId || !inputs.featureBundleId || !name.trim() || (!!workingPlan && !dirty)} onClick={() => void action('save')}>{busy ? 'Working…' : workingPlan ? 'Save batch changes' : 'Add batch to plan'}</button><button type="button" className="btn btn-secondary" disabled={busy || stale || !inputs.protocolId || !inputs.featureBundleId || !name.trim()} onClick={() => void action('preview')}>Check batch</button>{workingPlan || dirty ? <button className="text-button" disabled={busy} onClick={() => { if (load(batchTemplate('baseline', inputs, experimentName))) setDirty(false); }}>{dirty ? 'Discard batch edits' : 'New batch'}</button> : null}{dirty ? <span className="muted" role="status">Unsaved batch edits</span> : null}</div>
      </Panel>
      {preview ? <Panel title="Resolved batch"><Findings findings={preview.findings} /><p className="development-count" aria-live="polite"><strong>{preview.summary.configurationCount}</strong> configurations × <strong>{preview.summary.trainingSeedCount}</strong> training seeds × <strong>{preview.summary.splitPlanCount}</strong> frozen split plans = <strong>{preview.summary.runCount}</strong> planned runs</p><ConfigurationTable batch={preview} /></Panel> : null}
      {savedDrafts.length ? <details className="setup-details"><summary>Earlier batch drafts</summary><p className="muted">Load an earlier draft and add it to this experiment’s plan. Drafts are not submitted automatically.</p>{savedDrafts.map((draft) => <div className="development-saved-row" key={draft.id}><span>{draft.name}</span><button type="button" className="btn btn-secondary btn-small" onClick={() => load(draft.payload.spec as unknown as DevelopmentBatchSpec)}>Use draft settings</button></div>)}</details> : null}
    </> : null}
    {items.length || tab !== 'batches' ? <Panel title={tab === 'runs' ? 'Runs' : tab === 'results' ? 'Results' : 'Frozen batches'} subtitle={tab === 'batches' ? 'Saved configurations and split memberships are immutable. Experiment submission includes every active frozen batch.' : undefined}>
      {!items.length ? <p>No submitted batches are available yet.</p> : <>
        <label className="label">Batch<select className="field" value={selectedBatch?.id ?? ''} onChange={(e) => setSelected(e.target.value)}><option value="">Choose a batch in this experiment</option>{items.map((item) => <option key={item.id} value={item.id}>{item.manifest.spec.batchName} · {item.status} · {item.state}</option>)}</select></label>
        {selectedBatch ? <><div className="inline-actions"><Badge>{selectedBatch.manifest.summary.runCount} runs in plan</Badge><button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${selectedBatch.manifest.spec.batchName}.json`, selectedBatch)}>Export batch</button>{tab === 'batches' && !locked ? <button type="button" className="btn btn-secondary btn-small" onClick={() => load(selectedBatch.manifest.spec, undefined, true)}>Use as editable batch</button> : null}</div>
          {tab === 'batches' ? <ConfigurationTable batch={selectedBatch.manifest} /> : null}
          {tab === 'batches' && !locked ? <ExperimentLifecycle key={`lifecycle:${selectedBatch.id}`} project={project} recordKey={selectedBatch.key} state={selectedBatch.state} name={selectedBatch.manifest.spec.batchName} /> : null}
          {tab !== 'batches' || experimentStage === undefined ? <DevelopmentExecution key={selectedBatch.id} project={project} batch={selectedBatch} implemented={selectedBatch.state !== 'trashed' && executionImplemented} stage={experimentStage} readOnly={experimentStage === 'finished' || !!record?.legacy || (!!record && record.state !== 'active')} allowChanges={(experimentStage === 'running' || !readOnly) && selectedBatch.state === 'active'} knownExecution={selectedBatch.execution ?? undefined} view={tab} /> : null}
        </> : null}
      </>}
    </Panel> : null}
  </div>;
}

export function ConfigurationTable({ batch }: { batch: Pick<BatchPreview, 'configurations'> }) {
  return <div className="development-table"><table><thead><tr><th>Configuration</th><th>Model</th><th>LR</th><th>WD</th><th>Max epochs</th><th>Training bag</th></tr></thead><tbody>{batch.configurations.map((item) => <tr key={item.id}><td>{item.number}</td><td>{item.recipe.model}</td><td>{item.recipe.learningRate}</td><td>{item.recipe.weightDecay}</td><td>{item.recipe.maxEpochs}</td><td>{item.recipe.bagSize === null ? 'Whole bag (all patches)' : `${item.recipe.bagSize} patches maximum`}</td></tr>)}</tbody></table></div>;
}

export function ExperimentBatchOverview({ batches, view, onSelect }: { batches: ExperimentBatch[]; view: 'runs' | 'results'; onSelect: (id: string) => void }) {
  const total = batches.reduce((sum, batch) => sum + batch.manifest.summary.runCount, 0);
  const completed = batches.reduce((sum, batch) => sum + (batch.execution?.runCounts.completed ?? 0), 0);
  return <Panel title="Experiment overview" subtitle={`${completed} of ${total} fold runs completed across ${batches.length} batches.`}>
    <div className="development-table"><table><thead><tr><th>Batch</th><th>Status</th><th>Completed runs</th><th>Progress</th><th><span className="sr-only">Open batch</span></th></tr></thead><tbody>{batches.map((batch) => {
      const counts = batch.execution?.runCounts;
      const finished = counts?.completed ?? 0;
      const planned = batch.manifest.summary.runCount;
      return <tr key={batch.id}><th scope="row">{batch.manifest.spec.batchName}</th><td>{batch.status}{counts?.failed ? <small> · {counts.failed} failed</small> : null}</td><td>{batch.status === 'unknown' ? 'Status unavailable' : `${finished} / ${planned}`}</td><td>{batch.status === 'unknown' ? '—' : <progress aria-label={`${batch.manifest.spec.batchName}: completed fold runs`} value={finished} max={Math.max(1, planned)} />}</td><td><button className="text-button" onClick={() => onSelect(batch.id)}>{view === 'results' ? 'View results' : 'View runs'}</button></td></tr>;
    })}</tbody></table></div>
    {batches.some((batch) => batch.status === 'unknown') ? <p className="callout" role="status">Some execution status is unavailable. Completed counts include only the batches whose status could be read.</p> : null}
  </Panel>;
}
