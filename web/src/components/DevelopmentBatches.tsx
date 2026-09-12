import { useId, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { development, defaultRecipe, defaultResources, parseNumberList, withRecipeDefaults } from '../api/development';
import type { BatchPreview, DevelopmentBatchSpec, TrainingRecipe } from '../api/development';
import type { ExperimentBatch } from '../api/experiments';
import ExperimentLifecycle from './ExperimentLifecycle';
import type { ScientificDraft } from '../api/scientific';
import type { MILExperimentSpec } from '../api/mil';
import { scientific } from '../api/scientific';
import { Badge, ErrorNotice, Panel } from './ui';
import { Findings, SavedNotice, useRefreshScientific } from './ScientificUI';
import { downloadJSON } from '../lib/download';
import { sameJSON } from '../lib/json';
import './DevelopmentBatches.css';
import DevelopmentExecution from './DevelopmentExecution';
import NumericField, { reportEditorValidity } from './NumericField';
import TrainingCapacity from './TrainingCapacity';

export type DevelopmentTab = 'setup' | 'batches' | 'runs' | 'results';
export const developmentTabs: { id: DevelopmentTab; label: string }[] = [
  { id: 'setup', label: 'Inputs' }, { id: 'batches', label: 'Batches' }, { id: 'runs', label: 'Runs' },
  { id: 'results', label: 'Development results' },
];

export function batchVersionTag(name: string, experimentId: string) {
  const suffix = ` · ${experimentId}`;
  if (suffix.length >= 80) throw new Error('The experiment ID is too long for a batch version tag.');
  return `${name.trim().slice(0, 80 - suffix.length)}${suffix}`;
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

export default function DevelopmentBatches({ project, inputs, experimentName, experimentId, experimentRevision, ownedBatches, ownedDrafts, executionImplemented = false, readOnly = false, tab, onOpenSetup, onRestoreInputs }: {
  project: string; inputs: MILExperimentSpec; experimentName: string; experimentId: string; experimentRevision: number;
  ownedBatches: ExperimentBatch[]; ownedDrafts: ScientificDraft[]; executionImplemented?: boolean; readOnly?: boolean; tab: Exclude<DevelopmentTab, 'setup'>; onOpenSetup: () => void;
  onRestoreInputs: (inputs: MILExperimentSpec, name: string) => void;
}) {
  const client = useQueryClient();
  const runtime = useQuery({ queryKey: ['training-runtime', project], queryFn: () => development.runtime(project), enabled: tab === 'batches', staleTime: 30000 });
  const refresh = useRefreshScientific(project);
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
  const [operation, setOperation] = useState<string | null>(null);
  const [selected, setSelected] = useState('');
  const [workingDraft, setWorkingDraft] = useState<{ id: string; revision: number } | null>(null);
  const editor = useRef<HTMLFieldSetElement>(null);
  const [editorVersion, setEditorVersion] = useState(0);
  const items = ownedBatches;
  const selectedBatch = items.find((item) => item.id === selected) ?? (!selected && items.length === 1 ? items[0] : undefined);
  const savedDrafts = ownedDrafts.filter((draft) => draft.payload.type === 'development-batch');
  const reviewed = preview?.canFreeze && sameJSON(preview.spec.inputs, inputs) && preview.spec.experimentName === experimentName && preview.spec.experimentId === experimentId && preview.spec.experimentRevision === experimentRevision;
  let gpuSelection: number[] = [];
  let gpuSelectionError = '';
  try {
    gpuSelection = gpus.trim() ? parseNumberList(gpus, 'GPU IDs', true) : [];
    if (gpuSelection.some((id) => id > 127)) throw new Error('GPU IDs must be between 0 and 127.');
  } catch (reason) { gpuSelectionError = reason instanceof Error ? reason.message : 'Enter valid GPU IDs.'; }

  function edit() { setPreview(null); setOperation(null); setMessage(''); setError(null); }
  function specification(): DevelopmentBatchSpec {
    return { version: 1, experimentId, experimentRevision, experimentName, batchName: name.trim(), inputs, recipe, mode,
      grid: mode === 'grid' ? { learningRates: parseNumberList(lrs, 'Learning rates', false, Number.MIN_VALUE), weightDecays: parseNumberList(wds, 'Weight decay'), maxEpochs: parseNumberList(epochs, 'Maximum epochs', true, 1) } : { learningRates: [recipe.learningRate], weightDecays: [recipe.weightDecay], maxEpochs: [recipe.maxEpochs] },
      configurations: mode === 'explicit' ? rows.map((row) => row.recipe) : [], trainingSeeds: parseNumberList(seeds, 'Training seeds', true),
      resources: { ...resources, gpuIds: gpus.trim() ? parseNumberList(gpus, 'GPU IDs', true) : [] }, notes };
  }
  async function action(kind: 'preview' | 'save' | 'freeze') {
    if (readOnly || !reportEditorValidity(editor.current)) return;
    setBusy(true); setError(null); setMessage('');
    try {
      const spec = specification();
      if (kind === 'save') {
        const saved = await scientific.saveDraft(project, { kind: 'experiment', name: `${experimentName} · ${name}`, payload: { type: 'development-batch', experimentId, spec } }, workingDraft ?? undefined);
        setWorkingDraft(saved); setMessage('Editable batch draft saved. Review its inputs before freezing.'); await refresh(); await client.invalidateQueries({ queryKey: ['model-experiment', project] }); await client.invalidateQueries({ queryKey: ['model-experiments', project] });
      } else if (kind === 'freeze') {
        if (!reviewed || !preview) throw new Error('Review this batch before freezing.');
        const id = operation ?? crypto.randomUUID(); setOperation(id);
        const saved = await development.freeze(project, spec, preview.previewHash, id, { tag: batchVersionTag(name, experimentId), note: notes });
        setSelected(saved.id); setMessage('Batch plan saved. Review its saved runs, then use Launch batch when ready to train.');
        await client.invalidateQueries({ predicate: (query) => query.queryKey.includes(project) });
      } else {
        const result = await development.preview(project, spec); setPreview(result); setOperation(null);
      }
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Batch could not be saved.')); }
    finally { setBusy(false); }
  }
  function load(spec: DevelopmentBatchSpec, draft?: { id: string; revision: number }) {
    setEditorVersion((version) => version + 1);
    onRestoreInputs(spec.inputs, spec.experimentName); setWorkingDraft(draft ?? null);
    setName(draft ? spec.batchName : `${spec.batchName.slice(0, 70)} copy`); setRecipe(withRecipeDefaults(spec.recipe)); setResources(spec.resources); setMode(spec.mode);
    setRows((spec.configurations.length ? spec.configurations : [spec.recipe]).map((value) => ({ id: nextRowId.current++, recipe: withRecipeDefaults(value) }))); setSeeds(spec.trainingSeeds.join(', '));
    setLrs(spec.grid.learningRates.join(', ')); setWds(spec.grid.weightDecays.join(', ')); setEpochs(spec.grid.maxEpochs.join(', '));
    setGpus(spec.resources.gpuIds.join(', ')); setNotes(spec.notes); edit();
  }
  return <div className="development-batches">
    <ErrorNotice error={error} /><SavedNotice>{tab === 'batches' ? message : ''}</SavedNotice>
    {tab === 'batches' && !readOnly ? <>
      <Panel title="Configure a training batch" subtitle="Choose one setup or compare parameter combinations. Every run reuses your saved targets, features and folds.">
        {!inputs.protocolId || !inputs.featureBundleId ? <p className="callout">Choose prepared targets and features in <button type="button" className="text-button" onClick={onOpenSetup}>Inputs</button> first.</p> : <p className="development-input-summary">Inputs selected for <strong>{experimentName}</strong>. <button type="button" className="text-button" onClick={onOpenSetup}>Review inputs</button></p>}
        <fieldset key={editorVersion} ref={editor} disabled={busy} className="development-editor" onChange={edit}>
          <legend className="sr-only">Batch configuration</legend>
          <div className="development-fields"><label className="label">Batch name<input className="field" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></label>
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
        <div className="inline-actions"><button type="button" className="btn btn-secondary" disabled={busy || !inputs.protocolId || !inputs.featureBundleId || !name.trim()} onClick={() => void action('preview')}>Review batch</button><button type="button" className="btn btn-secondary" disabled={busy || !name.trim()} onClick={() => void action('save')}>Save draft</button><button type="button" className="btn btn-primary" disabled={busy || !reviewed} onClick={() => void action('freeze')}>Save batch plan</button></div>
      </Panel>
      {preview ? <Panel title="Resolved batch"><Findings findings={preview.findings} /><p className="development-count" aria-live="polite"><strong>{preview.summary.configurationCount}</strong> configurations × <strong>{preview.summary.trainingSeedCount}</strong> training seeds × <strong>{preview.summary.splitPlanCount}</strong> frozen split plans = <strong>{preview.summary.runCount}</strong> planned runs</p><ConfigurationTable batch={preview} /></Panel> : null}
      {savedDrafts.length ? <Panel title="Saved batch drafts">{savedDrafts.map((draft) => <div className="development-saved-row" key={draft.id}><span>{draft.name}</span><button type="button" className="btn btn-secondary btn-small" onClick={() => load(draft.payload.spec as unknown as DevelopmentBatchSpec, draft)}>Open draft</button></div>)}</Panel> : null}
    </> : null}
    <Panel title={tab === 'runs' ? 'Runs' : tab === 'results' ? 'Development results' : 'Frozen batches'} subtitle="Frozen plans retain exact configurations and split memberships. Launching is a separate action; run state and artifacts update as training progresses.">
      {!items.length ? <p>No frozen batches yet. Review and freeze a batch to retain its exact configurations and run plan.</p> : <>
        <label className="label">Batch<select className="field" value={selectedBatch?.id ?? ''} onChange={(e) => setSelected(e.target.value)}><option value="">Choose a batch in this experiment</option>{items.map((item) => <option key={item.id} value={item.id}>{item.manifest.spec.batchName} · {item.status} · {item.state}</option>)}</select></label>
        {selectedBatch ? <><div className="inline-actions"><Badge>{selectedBatch.manifest.summary.runCount} runs in plan</Badge><button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${selectedBatch.manifest.spec.batchName}.json`, selectedBatch)}>Export batch</button>{tab === 'batches' && !readOnly ? <button type="button" className="btn btn-secondary btn-small" onClick={() => load(selectedBatch.manifest.spec)}>Clone batch</button> : null}</div>
          {tab === 'batches' ? <ConfigurationTable batch={selectedBatch.manifest} /> : null}
          <ExperimentLifecycle key={`lifecycle:${selectedBatch.id}`} project={project} recordKey={selectedBatch.key} state={selectedBatch.state} name={selectedBatch.manifest.spec.batchName} />
          <DevelopmentExecution key={selectedBatch.id} project={project} batch={selectedBatch} implemented={selectedBatch.state !== 'trashed' && executionImplemented} allowChanges={!readOnly && selectedBatch.state === 'active'} knownExecution={selectedBatch.execution ?? undefined} view={tab} />
        </> : null}
      </>}
    </Panel>
  </div>;
}

export function ConfigurationTable({ batch }: { batch: Pick<BatchPreview, 'configurations'> }) {
  return <div className="development-table"><table><thead><tr><th>Configuration</th><th>Model</th><th>LR</th><th>WD</th><th>Max epochs</th><th>Training bag</th></tr></thead><tbody>{batch.configurations.map((item) => <tr key={item.id}><td>{item.number}</td><td>{item.recipe.model}</td><td>{item.recipe.learningRate}</td><td>{item.recipe.weightDecay}</td><td>{item.recipe.maxEpochs}</td><td>{item.recipe.bagSize === null ? 'Whole bag (all patches)' : `${item.recipe.bagSize} patches maximum`}</td></tr>)}</tbody></table></div>;
}
