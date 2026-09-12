import { useEffect, useId, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { development, defaultRecipe, defaultResources, parseNumberList, withRecipeDefaults } from '../api/development';
import type { BatchPreview, DevelopmentBatchSpec, TrainingRecipe } from '../api/development';
import { experiments, type ExperimentBatch, type ExperimentBatchPlan, type ExperimentStage, type ExperimentPredictorPolicy, type ModelExperiment } from '../api/experiments';
import ExperimentLifecycle from './ExperimentLifecycle';
import type { ProtocolSpec, ScientificDraft } from '../api/scientific';
import type { MILExperimentSpec } from '../api/mil';
import { Badge, ErrorNotice, Panel } from './ui';
import { Findings, SavedNotice } from './ScientificUI';
import { downloadJSON } from '../lib/download';
import { sameJSON } from '../lib/json';
import { batchPredictorPolicy, defaultPredictorPolicy, plannedBatchPredictorCount, plannedConfigurationCount } from '../lib/experimentPredictors';
import './DevelopmentBatches.css';
import DevelopmentExecution from './DevelopmentExecution';
import NumericField from './NumericField';
import TrainingCapacity from './TrainingCapacity';
import BatchNumberList, { validateBatchNumberList } from './BatchNumberList';
import BatchPredictorFields, { batchPredictorLabel } from './BatchPredictorFields';
import { StagePage, StageSteps } from './StageWorkflow';

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
  { id: 'blank', name: 'Start blank', description: 'Start with default settings and give this batch a name.' },
  { id: 'baseline', name: 'ABMIL baseline', description: 'One configuration with standard training settings.' },
  { id: 'quick', name: 'Quick check', description: 'Five epochs and smaller sampled bags to check the training setup.' },
  { id: 'learning-rate', name: 'Learning-rate comparison', description: 'Compare three learning rates with the same folds and training seed.' },
] as const;

export function batchTemplate(id: string, inputs: MILExperimentSpec, experimentName: string): DevelopmentBatchSpec {
  const recipe = { ...defaultRecipe(), ...(id === 'quick' ? { maxEpochs: 5, bagSize: 1024, patience: 3 } : {}) };
  return { version: 1, experimentName, batchName: id === 'blank' ? '' : batchTemplates.find((item) => item.id === id)?.name ?? 'Baseline', inputs,
    recipe, mode: id === 'learning-rate' ? 'grid' : 'single', grid: { learningRates: id === 'learning-rate' ? [0.0001, 0.0003, 0.001] : [recipe.learningRate], weightDecays: [recipe.weightDecay], maxEpochs: [recipe.maxEpochs] },
    configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '', predictorPolicy: defaultPredictorPolicy() };
}

export function RecipeFields({ value, onChange, gridMode = false }: { value: TrainingRecipe; onChange: (value: TrainingRecipe) => void; gridMode?: boolean }) {
  const resolved = withRecipeDefaults(value);
  const bagModeId = useId();
  const [sampledBagSize, setSampledBagSize] = useState(value.bagSize ?? 4096);
  return <div className="batch-recipe-settings">
    <section className="batch-editor-section" aria-label={gridMode ? 'Shared training settings' : 'Training settings'}>
      <div className="batch-section-heading"><h3>{gridMode ? 'Shared training settings' : 'Training settings'}</h3><p>{gridMode ? 'Applied to every combination in the parameter grid.' : 'The model and settings used for each fold.'}</p></div>
      <div className="development-fields batch-primary-fields">
        <label className="label">Model<select className="field" value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })}><option value="abmil">ABMIL</option>{value.model !== 'abmil' ? <option value={value.model} disabled>{value.model} (unavailable)</option> : null}</select></label>
        {!gridMode ? <>
          <NumericField label="Learning rate" value={value.learningRate} integer={false} min={0} minExclusive onChange={(learningRate) => onChange({ ...value, learningRate })} />
          <NumericField label="Weight decay" value={value.weightDecay} integer={false} min={0} onChange={(weightDecay) => onChange({ ...value, weightDecay })} />
          <NumericField label="Maximum epochs" value={value.maxEpochs} min={1} max={100000} onChange={(maxEpochs) => onChange({ ...value, maxEpochs })} />
        </> : null}
        <NumericField label="Batch size" value={value.batchSize} min={1} max={4096} onChange={(batchSize) => onChange({ ...value, batchSize })} />
      </div>
      <fieldset className="development-bag-mode"><legend>Training bag</legend>
        <div className="batch-bag-options">
          <label className="development-check"><input type="radio" name={bagModeId} checked={value.bagSize !== null} onChange={() => onChange({ ...value, bagSize: sampledBagSize })} /> Sample patches per bag</label>
          <label className="development-check"><input type="radio" name={bagModeId} checked={value.bagSize === null} onChange={() => { if (value.bagSize !== null) setSampledBagSize(value.bagSize); onChange({ ...value, bagSize: null }); }} /> Use whole bag for training</label>
        </div>
        {value.bagSize === null ? <p className="development-bag-help">Train with every available patch in each slide. No patch sampling is applied. Larger bags require more memory.</p>
          : <NumericField label="Patches per bag" value={value.bagSize} min={1} max={1000000} onChange={(bagSize) => onChange({ ...value, bagSize })} />}
        <small>Validation and assessment always use whole bags.</small>
      </fieldset>
    </section>
    <details className="batch-settings-details setup-details"><summary><span>Optimization &amp; stopping</span>{' '}<small>{resolved.optimizer.toUpperCase()} · {resolved.lrScheduler === 'cosine' ? 'Cosine decay' : 'Constant learning rate'} · {resolved.earlyStopping ? `Patience ${resolved.patience}` : 'No early stopping'}</small></summary>
      <div className="development-fields">
        <label className="label">Optimizer<select className="field" value={value.optimizer} onChange={(e) => onChange({ ...value, optimizer: e.target.value as TrainingRecipe['optimizer'] })}><option value="adamw">AdamW</option><option value="adam">Adam</option><option value="sgd">SGD</option></select></label>
        <label className="label">Checkpoint selection<select className="field" value={value.checkpointMetric} onChange={(e) => onChange({ ...value, checkpointMetric: e.target.value as TrainingRecipe['checkpointMetric'] })}><option value="validation_loss">Lowest validation loss</option><option value="validation_auroc">Highest validation AUROC</option><option value="validation_accuracy">Highest validation accuracy</option></select></label>
        <label className="label">Learning-rate schedule<select className="field" value={resolved.lrScheduler} onChange={(e) => onChange({ ...value, lrScheduler: e.target.value as TrainingRecipe['lrScheduler'], warmupEpochs: e.target.value === 'none' ? 0 : resolved.warmupEpochs })}><option value="none">Constant learning rate</option><option value="cosine">Cosine decay</option></select></label>
        <NumericField label="Gradient clipping norm" value={resolved.gradientClipNorm!} min={0} integer={false} onChange={(gradientClipNorm) => onChange({ ...value, gradientClipNorm })} />
        {resolved.lrScheduler === 'cosine' ? <>
          <NumericField label="Warmup epochs" value={resolved.warmupEpochs!} min={0} max={gridMode ? 99999 : value.maxEpochs - 1} onChange={(warmupEpochs) => onChange({ ...value, warmupEpochs })} />
          <NumericField label="Final LR fraction" value={resolved.finalLrFraction!} min={0} minExclusive max={1} integer={false} onChange={(finalLrFraction) => onChange({ ...value, finalLrFraction })} />
        </> : null}
      </div>
      <div className="batch-stopping-settings"><label className="development-check"><input type="checkbox" checked={value.earlyStopping} onChange={(e) => onChange({ ...value, earlyStopping: e.target.checked })} /> Use early stopping</label>
        <div className="development-fields">
          <NumericField label="Early-stopping patience" value={value.patience} min={1} max={10000} disabled={!value.earlyStopping} onChange={(patience) => onChange({ ...value, patience })} />
          <NumericField label="Early-stopping minimum improvement" value={resolved.earlyStoppingMinDelta!} min={0} integer={false} disabled={!value.earlyStopping} onChange={(earlyStoppingMinDelta) => onChange({ ...value, earlyStoppingMinDelta })} />
          <NumericField label="Minimum training epochs" value={resolved.minEpochs!} min={1} max={gridMode ? 100000 : value.maxEpochs} onChange={(minEpochs) => onChange({ ...value, minEpochs })} />
        </div>
      </div>
      <p className="muted">Gradient clipping 0 disables clipping. Minimum epochs delay stopping; the best validation checkpoint can still come from an earlier epoch.</p>
    </details>
    <details className="batch-settings-details setup-details"><summary><span>Model architecture</span>{' '}<small>{resolved.embedDim} embedding · {resolved.attentionDim} attention · Dropout {resolved.dropout}</small></summary>
      <div className="development-fields">
        {([['embedDim', 'Embedding dimensions', 1], ['attentionDim', 'Attention dimensions', 1], ['numFcLayers', 'Fully connected layers', 1], ['dropout', 'Dropout', 0], ['inputDropout', 'Input dropout', 0]] as const).map(([key, label, min]) => <NumericField key={key} label={label} value={resolved[key]!} min={min} max={key === 'dropout' || key === 'inputDropout' ? 1 : key === 'numFcLayers' ? 8 : 8192} maxExclusive={key === 'dropout' || key === 'inputDropout'} integer={key !== 'dropout' && key !== 'inputDropout'} onChange={(number) => onChange({ ...value, [key]: number })} />)}
        <label className="development-check"><input type="checkbox" checked={resolved.gatedAttention} onChange={(e) => onChange({ ...value, gatedAttention: e.target.checked })} /> Gated attention</label>
      </div>
    </details>
    <details className="batch-settings-details setup-details"><summary><span>Precision &amp; memory</span>{' '}<small>{resolved.precision === '32-true' ? 'FP32' : resolved.precision} · {resolved.accumulateGradBatches} batch{resolved.accumulateGradBatches === 1 ? '' : 'es'} per update</small></summary>
      <div className="development-fields">
        <label className="label">Precision<select className="field" value={resolved.precision} onChange={(e) => onChange({ ...value, precision: e.target.value as TrainingRecipe['precision'] })}><option value="32-true">FP32 (standard)</option><option value="16-mixed">FP16 mixed (CUDA)</option><option value="bf16-mixed">BF16 mixed (supported device)</option></select><small>Mixed precision can reduce GPU memory use. Device support is checked before training.</small></label>
        <NumericField label="Accumulate batches" value={resolved.accumulateGradBatches!} min={1} max={4096} onChange={(accumulateGradBatches) => onChange({ ...value, accumulateGradBatches })} />
        <label className="development-check"><input type="checkbox" checked={resolved.gradientCheckpointing} onChange={(e) => onChange({ ...value, gradientCheckpointing: e.target.checked })} /> Gradient checkpointing</label>
      </div>
      <p className="muted">Accumulation combines several batches per optimizer update. Gradient checkpointing trades extra computation for lower memory use.</p>
    </details>
  </div>;
}

const configurationModes = [
  { id: 'single', name: 'Single configuration', description: 'One model setup, repeated for each seed.' },
  { id: 'grid', name: 'Parameter grid', description: 'Run every combination of the parameter values.' },
  { id: 'explicit', name: 'Custom configurations', description: 'Set up and compare individual configurations.' },
] as const;

export const batchConfigurationCount = plannedConfigurationCount;

export function RecipeSummary({ recipe }: { recipe: TrainingRecipe }) {
  return <span className="batch-recipe-summary"><strong>{recipe.model.toUpperCase()}</strong><span>LR {recipe.learningRate}</span><span>WD {recipe.weightDecay}</span><span>{recipe.maxEpochs} epochs max</span><span>{recipe.bagSize === null ? 'Whole bags' : `${recipe.bagSize.toLocaleString()} patches / bag`}</span></span>;
}

export function BatchPredictorSummary({ spec, protocol, fallbackPredictorPolicy, count: frozenCount }: {
  spec: DevelopmentBatchSpec; protocol?: ProtocolSpec; fallbackPredictorPolicy?: ExperimentPredictorPolicy; count?: number;
}) {
  const policy = spec.predictorPolicy ?? fallbackPredictorPolicy;
  const count = frozenCount ?? (policy ? plannedBatchPredictorCount(spec, protocol, policy)?.total : undefined);
  return <div className="batch-predictor-summary">
    <Badge>{policy ? `Predictors: ${batchPredictorLabel(policy)}` : 'Predictors not configured'}</Badge>
    {count !== undefined ? <span>{count.toLocaleString()} predictor{count === 1 ? '' : 's'} planned</span> : null}
  </div>;
}

export function BatchPlanSettings({ spec, fallbackPredictorPolicy }: { spec: DevelopmentBatchSpec; fallbackPredictorPolicy?: ExperimentPredictorPolicy }) {
  const policy = spec.predictorPolicy ?? fallbackPredictorPolicy;
  const count = batchConfigurationCount(spec);
  const recipes = spec.mode === 'explicit' ? spec.configurations : [spec.recipe];
  return <div className="batch-plan-settings">
    <dl className="batch-settings-summary">
      <div><dt>Parameter search</dt><dd>{configurationModes.find((mode) => mode.id === spec.mode)?.name} · {count} configuration{count === 1 ? '' : 's'}</dd></div>
      <div><dt>Predictors</dt><dd>{policy ? batchPredictorLabel(policy) : 'Not configured (historical batch)'}</dd></div>
      <div><dt>Training seeds</dt><dd>{spec.trainingSeeds.join(', ')}</dd></div>
      {spec.mode === 'grid' ? <><div><dt>Learning rates</dt><dd>{spec.grid.learningRates.join(', ')}</dd></div><div><dt>Weight decays</dt><dd>{spec.grid.weightDecays.join(', ')}</dd></div><div><dt>Maximum epochs</dt><dd>{spec.grid.maxEpochs.join(', ')}</dd></div></> : null}
      <div><dt>Compute</dt><dd>{spec.resources.gpuIds.length ? `GPU ${spec.resources.gpuIds.join(', ')}` : 'CPU'} · {spec.resources.maxConcurrentRuns} concurrent run{spec.resources.maxConcurrentRuns === 1 ? '' : 's'}</dd></div>
      <div><dt>Reservation per run</dt><dd>{spec.resources.cpuThreadsPerRun} CPU threads · {spec.resources.ramGbPerRun} GiB RAM</dd></div>
    </dl>
    {spec.mode === 'grid' ? <p className="muted">The parameter grid supplies learning rate, weight decay and maximum epochs. Other training settings are shared.</p> : null}
    {recipes.map((recipe, index) => {
      const resolved = withRecipeDefaults(recipe);
      return <details key={index} className="batch-saved-recipe"><summary>{spec.mode === 'grid' ? <strong>{recipe.model.toUpperCase()} · Shared training settings</strong> : <><strong>Configuration {index + 1}</strong><RecipeSummary recipe={recipe} /></>}</summary>
        <dl className="batch-settings-summary">
          <div><dt>Training bag</dt><dd>{recipe.bagSize === null ? 'Whole bag (all patches)' : `${recipe.bagSize} patches maximum`} · Batch size {recipe.batchSize}</dd></div>
          <div><dt>Checkpoint selection</dt><dd>{recipe.checkpointMetric === 'validation_loss' ? 'Lowest validation loss' : recipe.checkpointMetric === 'validation_auroc' ? 'Highest validation AUROC' : 'Highest validation accuracy'}</dd></div>
          <div><dt>Optimization</dt><dd>{recipe.optimizer.toUpperCase()} · {resolved.lrScheduler === 'cosine' ? `Cosine decay · ${resolved.warmupEpochs} warmup epochs · Final LR fraction ${resolved.finalLrFraction}` : 'Constant learning rate'}</dd></div>
          <div><dt>Early stopping</dt><dd>{recipe.earlyStopping ? `Patience ${recipe.patience} · Minimum improvement ${resolved.earlyStoppingMinDelta}` : 'Disabled'} · Minimum epochs {resolved.minEpochs}</dd></div>
          <div><dt>Model architecture</dt><dd>{resolved.embedDim} embedding · {resolved.attentionDim} attention · {resolved.numFcLayers} fully connected layer{resolved.numFcLayers === 1 ? '' : 's'} · {resolved.gatedAttention ? 'Gated attention' : 'Ungated attention'}</dd></div>
          <div><dt>Regularization</dt><dd>Dropout {resolved.dropout} · Input dropout {resolved.inputDropout} · Gradient clipping {resolved.gradientClipNorm}</dd></div>
          <div><dt>Precision &amp; memory</dt><dd>{resolved.precision === '32-true' ? 'FP32' : resolved.precision === '16-mixed' ? 'FP16 mixed' : 'BF16 mixed'} · Accumulate {resolved.accumulateGradBatches} batch{resolved.accumulateGradBatches === 1 ? '' : 'es'} · Gradient checkpointing {resolved.gradientCheckpointing ? 'on' : 'off'}</dd></div>
        </dl>
      </details>;
    })}
    {spec.notes ? <p className="batch-plan-notes">{spec.notes}</p> : null}
    <details className="batch-exact-settings"><summary>All saved settings</summary><pre className="experiment-snapshot">{JSON.stringify(spec, null, 2)}</pre></details>
  </div>;
}

export default function DevelopmentBatches({ project, inputs, experimentName, experimentId, experimentRevision, ownedBatches, ownedDrafts, executionImplemented = false, readOnly = false, record, experimentStage, protocol, onPlanDirtyChange, tab, onOpenSetup }: {
  project: string; inputs: MILExperimentSpec; experimentName: string; experimentId: string; experimentRevision: number;
  record?: ModelExperiment; experimentStage?: ExperimentStage; protocol?: ProtocolSpec; onPlanDirtyChange?: (dirty: boolean) => void;
  ownedBatches: ExperimentBatch[]; ownedDrafts: ScientificDraft[]; executionImplemented?: boolean; readOnly?: boolean; tab: Exclude<DevelopmentTab, 'setup'>; onOpenSetup: () => void;
}) {
  const client = useQueryClient();
  const runtime = useQuery({ queryKey: ['training-runtime', project], queryFn: () => development.runtime(project), enabled: tab === 'batches', staleTime: 30000 });
  const [name, setName] = useState('');
  const [editorOpen, setEditorOpen] = useState(!(record?.batchPlans?.length || ownedBatches.length));
  const [batchPage, setBatchPage] = useState(1);
  const [templateId, setTemplateId] = useState('blank');
  const [predictorPolicy, setPredictorPolicy] = useState(defaultPredictorPolicy);
  const [recipe, setRecipe] = useState(defaultRecipe);
  const [resources, setResources] = useState(defaultResources);
  const [mode, setMode] = useState<DevelopmentBatchSpec['mode']>('single');
  const [rows, setRows] = useState(() => [{ id: 0, recipe: defaultRecipe() }]);
  const nextRowId = useRef(1);
  const explicitInitialized = useRef(false);
  const configurationModeId = useId();
  const [seeds, setSeeds] = useState('42');
  const [lrs, setLrs] = useState('0.0001, 0.0003, 0.001');
  const [wds, setWds] = useState('0, 0.0001');
  const [epochs, setEpochs] = useState('40');
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
  const stale = !locked && Boolean(workingPlan || dirty) && editorRevision !== experimentRevision;
  useEffect(() => { onPlanDirtyChange?.(!locked && (dirty || stale)); }, [dirty, stale, locked, onPlanDirtyChange]);
  useEffect(() => () => onPlanDirtyChange?.(false), [onPlanDirtyChange]);
  let gpuSelection: number[] = [];
  let gpuSelectionError = '';
  try {
    gpuSelection = gpus.trim() ? parseNumberList(gpus, 'GPU IDs', true) : [];
    if (gpuSelection.some((id) => id > 127)) throw new Error('GPU IDs must be between 0 and 127.');
  } catch (reason) { gpuSelectionError = reason instanceof Error ? reason.message : 'Enter valid GPU IDs.'; }

  let plannedConfigurations: number | null = null;
  let plannedSeeds: number | null = null;
  try {
    if (validateBatchNumberList(seeds, { label: 'Training seeds', min: 0, max: 2 ** 32 - 1 })) throw new Error('Invalid seeds');
    if (mode === 'grid' && (validateBatchNumberList(lrs, { label: 'Learning rates', integer: false, min: 0, minExclusive: true }) || validateBatchNumberList(wds, { label: 'Weight decays', integer: false, min: 0 }) || validateBatchNumberList(epochs, { label: 'Maximum epochs', min: 1, max: 100000 }))) throw new Error('Invalid grid');
    plannedConfigurations = batchConfigurationCount({ mode, configurations: rows.map((row) => row.recipe), grid: {
      learningRates: mode === 'grid' ? parseNumberList(lrs, 'Learning rates', false, Number.MIN_VALUE) : [],
      weightDecays: mode === 'grid' ? parseNumberList(wds, 'Weight decays') : [],
      maxEpochs: mode === 'grid' ? parseNumberList(epochs, 'Maximum epochs', true, 1) : [],
    } });
    plannedSeeds = parseNumberList(seeds, 'Training seeds', true).length;
  } catch { /* Counts remain unavailable while a list is incomplete or invalid. */ }
  function validateEditor(throughPage = 4) {
    const invalid = [...(editor.current?.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('input, select, textarea') ?? [])]
      .find((field) => Number(field.closest<HTMLElement>('[data-batch-step]')?.dataset.batchStep ?? 1) <= throughPage && !field.checkValidity());
    if (!invalid) return true;
    setBatchPage(Number(invalid.closest<HTMLElement>('[data-batch-step]')?.dataset.batchStep ?? 1));
    setEditorOpen(true);
    window.requestAnimationFrame(() => {
      let details = invalid.closest('details');
      while (details) { details.open = true; details = details.parentElement?.closest('details') ?? null; }
      invalid.reportValidity();
    });
    return false;
  }
  function showBatchPage(next: number) {
    if (busy || stale || (next > batchPage && !validateEditor(batchPage))) return;
    setBatchPage(next);
  }
  function changeMode(next: DevelopmentBatchSpec['mode']) {
    if (next === 'explicit' && !explicitInitialized.current) {
      let initial = recipe;
      if (mode === 'grid') {
        if (!validateEditor()) return;
        initial = { ...recipe, learningRate: parseNumberList(lrs, 'Learning rates', false, Number.MIN_VALUE)[0], weightDecay: parseNumberList(wds, 'Weight decays')[0], maxEpochs: parseNumberList(epochs, 'Maximum epochs', true, 1)[0] };
      }
      explicitInitialized.current = true;
      setRows([{ id: nextRowId.current++, recipe: { ...initial } }]);
    }
    setMode(next);
  }
  function edit() { if (!dirty && !workingPlan) setEditorRevision(experimentRevision); setDirty(true); setPreview(null); setMessage(''); setError(null); }
  function specification(): DevelopmentBatchSpec {
    return { version: 1, experimentId, experimentRevision, experimentName, batchName: name.trim(), inputs, recipe, mode,
      grid: mode === 'grid' ? { learningRates: parseNumberList(lrs, 'Learning rates', false, Number.MIN_VALUE), weightDecays: parseNumberList(wds, 'Weight decay'), maxEpochs: parseNumberList(epochs, 'Maximum epochs', true, 1) } : { learningRates: [recipe.learningRate], weightDecays: [recipe.weightDecay], maxEpochs: [recipe.maxEpochs] },
      configurations: mode === 'explicit' ? rows.map((row) => row.recipe) : [], trainingSeeds: parseNumberList(seeds, 'Training seeds', true),
      resources: { ...resources, gpuIds: gpus.trim() ? parseNumberList(gpus, 'GPU IDs', true) : [] }, notes, predictorPolicy };
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
    if (locked || stale || busyRef.current || !validateEditor()) return;
    busyRef.current = true;
    setBusy(true); setError(null); setMessage('');
    try {
      const spec = specification();
      if (kind === 'save') {
        const id = workingPlan ?? crypto.randomUUID();
        if (!dirty && !workingPlan) setEditorRevision(experimentRevision);
        setWorkingPlan(id);
        await savePlans(updateBatchPlans(plans, { id, spec }));
        setDirty(false); setPreview(null); setEditorOpen(false);
        setMessage('Batch plan saved. It remains editable until you submit this experiment.');
      } else {
        const result = await development.preview(project, spec); setPreview(result); setBatchPage(4);
      }
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Batch could not be saved.')); }
    finally { busyRef.current = false; setBusy(false); }
  }
  function load(spec: DevelopmentBatchSpec, planId?: string, copy = false, template = 'saved') {
    if (dirty && !window.confirm('Discard the unsaved batch edits and open this configuration?')) return false;
    setEditorOpen(true); setBatchPage(1);
    setEditorVersion((version) => version + 1);
    setTemplateId(template);
    setPredictorPolicy(batchPredictorPolicy(spec, record?.predictorPolicy));
    explicitInitialized.current = spec.mode === 'explicit';
    const inputsChanged = !sameJSON(inputs, spec.inputs);
    setWorkingPlan(planId ?? null); setEditorRevision(experimentRevision);
    setName(copy ? `${spec.batchName.slice(0, 70)} copy` : spec.batchName); setRecipe(withRecipeDefaults(spec.recipe)); setResources(spec.resources); setMode(spec.mode);
    setRows((spec.configurations.length ? spec.configurations : [spec.recipe]).map((value) => ({ id: nextRowId.current++, recipe: withRecipeDefaults(value) }))); setSeeds(spec.trainingSeeds.join(', '));
    setLrs(spec.grid.learningRates.join(', ')); setWds(spec.grid.weightDecays.join(', ')); setEpochs(spec.grid.maxEpochs.join(', '));
    setGpus(spec.resources.gpuIds.join(', ')); setNotes(spec.notes); setDirty(!planId); setPreview(null); setMessage(inputsChanged ? 'Batch settings copied. This batch will use this experiment’s verified inputs.' : ''); setError(null);
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
  return <StagePage pageKey={`${tab}:${editorOpen}:${batchPage}`} className="development-batches">
    <ErrorNotice error={error} /><SavedNotice>{tab === 'batches' ? message : ''}</SavedNotice>
    {tab !== 'batches' && items.length > 1 ? <ExperimentBatchOverview batches={items} view={tab} onSelect={setSelected} /> : null}
    {tab === 'batches' && !editorOpen && !locked ? <div className="stage-actions"><p>{dirty ? 'Your unsaved batch edits are retained while you review the plan.' : 'Add a batch or open a saved plan to adjust its settings.'}</p>{dirty ? <button className="btn btn-secondary" onClick={() => setEditorOpen(true)}>Resume batch edits</button> : null}<button className="btn btn-primary" disabled={busy} onClick={() => load(batchTemplate('blank', inputs, experimentName), undefined, false, 'blank')}>Add training batch</button></div> : null}
    {tab === 'batches' && (locked || !editorOpen) && plans.length ? <Panel title={`Batch plans (${plans.length})`} subtitle={locked ? 'These settings were locked when the experiment was submitted.' : 'Edit or remove a batch before submission. All batches use the experiment’s saved inputs.'}>
      <div className="batch-plan-list">{plans.map((plan) => <article key={plan.id} className={`batch-plan-card${workingPlan === plan.id ? ' is-editing' : ''}`}>
        <div className="batch-plan-heading"><h3>{plan.spec.batchName}</h3><p className="muted">{batchConfigurationCount(plan.spec)} configuration{batchConfigurationCount(plan.spec) === 1 ? '' : 's'} × {plan.spec.trainingSeeds.length} training seed{plan.spec.trainingSeeds.length === 1 ? '' : 's'}</p><BatchPredictorSummary spec={plan.spec} protocol={protocol} fallbackPredictorPolicy={record?.predictorPolicy ?? (!locked ? defaultPredictorPolicy() : undefined)} /></div>
        {!locked ? <div className="inline-actions"><button className="btn btn-secondary btn-small" disabled={busy} onClick={() => load(plan.spec, plan.id)}>Edit batch</button><button className="text-button" disabled={busy} onClick={() => load(plan.spec, undefined, true)}>Duplicate</button><button className="text-button" disabled={busy || dirty || stale} onClick={() => void removePlan(plan.id)}>Remove</button></div> : <Badge>Locked</Badge>}
        <details className="batch-plan-spec"><summary>View settings</summary><BatchPlanSettings spec={plan.spec} fallbackPredictorPolicy={record?.predictorPolicy ?? (!locked ? defaultPredictorPolicy() : undefined)} /></details>
      </article>)}</div>
    </Panel> : null}
    {tab === 'batches' && !locked ? <div hidden={!editorOpen}>
      <StageSteps label="Batch configuration steps" current={String(batchPage)} disabled={busy || stale} onChange={(id) => showBatchPage(Number(id))} steps={[{ id: '1', title: 'Configuration', description: 'Name, parameter search and seeds' }, { id: '2', title: 'Training settings', description: 'Model and optimization' }, { id: '3', title: 'Compute & predictors', description: 'Resources and prediction methods' }, { id: '4', title: 'Review batch', description: 'Check and save the plan' }].map((step) => ({ ...step, disabled: Number(step.id) > batchPage + 1 }))} />
      <Panel title={workingPlan ? `Edit batch: ${name || 'Untitled'}` : 'Add a training batch'} subtitle="Choose one setup or compare parameter combinations. Save your batches, then submit the experiment when the plan is ready.">
        <div hidden={batchPage !== 1}><div className="batch-template-picker"><label className="label">Start from a template<select className="field" value={templateId} disabled={busy} onChange={(event) => load(batchTemplate(event.target.value, inputs, experimentName), undefined, false, event.target.value)}>{templateId === 'saved' ? <option value="saved" disabled>Saved batch settings</option> : null}{batchTemplates.map((template) => <option key={template.id} value={template.id}>{template.name}</option>)}</select></label><p className="muted">{batchTemplates.find((template) => template.id === templateId)?.description ?? 'Adjust the saved settings, or choose a template to start another batch.'}</p></div></div>
        {!inputs.protocolId || !inputs.featureBundleId ? <p className="callout">Choose prepared targets and features in <button type="button" className="text-button" onClick={onOpenSetup}>Inputs</button> first.</p> : <p className="development-input-summary">Inputs selected for <strong>{experimentName}</strong>. <button type="button" className="text-button" onClick={onOpenSetup}>Review inputs</button></p>}
        {stale ? <p role="alert" className="callout">The saved experiment changed while you were editing. <button className="text-button" onClick={() => load(plans.find((plan) => plan.id === workingPlan)?.spec ?? batchTemplate('blank', inputs, experimentName), plans.find((plan) => plan.id === workingPlan)?.id)}>Reload the saved plan</button> before saving.</p> : null}
        <fieldset key={editorVersion} ref={editor} disabled={busy || stale} className="development-editor" onChange={edit}>
          <legend className="sr-only">Batch configuration</legend>
          <div data-batch-step="1" hidden={batchPage !== 1}><label className="label batch-name-field">Batch name<input required className="field" value={name} placeholder="Name this batch" maxLength={80} onChange={(e) => setName(e.target.value)} /></label>
          <section className="batch-editor-section" aria-label="Parameter search and repeats">
            <div className="batch-section-heading"><h3>Parameter search &amp; repeats</h3><p>Each distinct configuration runs across the experiment’s frozen folds for every training seed.</p></div>
            <fieldset className="batch-configuration-modes"><legend>Configuration mode</legend><div className="batch-mode-options">{configurationModes.map((option) => <label key={option.id} className={`batch-mode-option${mode === option.id ? ' is-selected' : ''}`}>
              <input type="radio" name={configurationModeId} value={option.id} checked={mode === option.id} onChange={() => changeMode(option.id)} /><span><strong>{option.name}</strong><small>{option.description}</small></span>
            </label>)}</div></fieldset>
            {mode === 'grid' ? <div className="batch-grid-values"><div className="development-fields"><BatchNumberList label="Learning rates" value={lrs} onChange={setLrs} integer={false} min={0} minExclusive /><BatchNumberList label="Weight decays" value={wds} onChange={setWds} integer={false} min={0} /><BatchNumberList label="Maximum epochs" value={epochs} onChange={setEpochs} min={1} max={100000} /></div><p className="muted">Enter comma-separated values. Every learning rate × weight decay × epoch limit becomes a configuration.</p></div> : null}
            <div className="batch-seeds-field"><BatchNumberList label="Training seeds" value={seeds} onChange={setSeeds} min={0} max={2 ** 32 - 1} hint="Comma-separated, for example 42, 43, 44. These repeat training; they do not change the frozen folds." /></div>
            <div className="batch-size-summary" role="status" aria-live="polite">{plannedConfigurations !== null && plannedSeeds !== null ? <><strong>{plannedConfigurations} configuration{plannedConfigurations === 1 ? '' : 's'} × {plannedSeeds} training seed{plannedSeeds === 1 ? '' : 's'} = {plannedConfigurations * plannedSeeds} training group{plannedConfigurations * plannedSeeds === 1 ? '' : 's'}</strong><span>Each group runs all frozen folds. Check batch to confirm the total fold runs.</span></> : <span>Enter valid parameter values and training seeds to see the planned size.</span>}</div>
          </section>
          </div><div data-batch-step="2" hidden={batchPage !== 2}><section className="batch-editor-section batch-all-settings" aria-label="Settings"><div className="batch-section-heading"><h3>Settings</h3></div>
          {mode !== 'explicit' ? <RecipeFields value={recipe} onChange={setRecipe} gridMode={mode === 'grid'} /> : <section className="batch-custom-configurations" aria-label="Custom configurations"><div className="batch-section-heading"><h3>Configurations</h3><p>Open a configuration to adjust its settings. Added configurations copy the last one; identical rows train only once.</p></div>{rows.map((row, index) => <details className="batch-configuration-card" key={row.id} open={index === 0 ? true : undefined}>
            <summary><strong>Configuration {index + 1}</strong><RecipeSummary recipe={row.recipe} /></summary>
            <div className="batch-configuration-body"><RecipeFields value={row.recipe} onChange={(value) => setRows((current) => current.map((item) => item.id === row.id ? { ...item, recipe: value } : item))} /><button type="button" className="text-button" disabled={rows.length === 1} onClick={() => { setRows((current) => current.filter((item) => item.id !== row.id)); edit(); }}>Remove configuration {index + 1}</button></div>
          </details>)}<button type="button" className="btn btn-secondary" disabled={rows.length >= 512} onClick={() => { const id = nextRowId.current++; setRows((current) => [...current, { id, recipe: { ...current[current.length - 1].recipe } }]); edit(); }}>Add configuration</button></section>}
          <details className="setup-details batch-settings-details"><summary>Batch notes (optional)</summary><label className="label">Notes<textarea className="field" value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} /></label></details></section>
          </div><div data-batch-step="3" hidden={batchPage !== 3}><section className="development-resource-settings batch-editor-section" aria-label="Parallel training"><div className="batch-section-heading"><h3>Compute &amp; parallelism</h3><p>Choose the device and how many fold runs may train at once.</p></div><div className="development-fields">
            <label className="label">Run on<select className="field" value={gpus.trim() ? 'gpu' : 'cpu'} onChange={(e) => setGpus(e.target.value === 'cpu' ? '' : '0')}><option value="gpu">GPU</option><option value="cpu">CPU</option></select></label>
            <NumericField label="Concurrent runs" value={resources.maxConcurrentRuns} min={1} max={128} onChange={(maxConcurrentRuns) => setResources((current) => ({ ...current, maxConcurrentRuns }))} />
            {gpus.trim() ? <NumericField label="Runs per GPU" value={resources.runsPerGpu} min={1} max={16} onChange={(runsPerGpu) => setResources((current) => ({ ...current, runsPerGpu }))} /> : null}
          </div>{gpuSelectionError ? <p className="callout" role="status">{gpuSelectionError}</p> : <TrainingCapacity resources={{ ...resources, gpuIds: gpuSelection }} runtime={runtime.data} />}
          <details className="setup-details batch-settings-details"><summary><span>Advanced resource settings</span>{' '}<small>{resources.cpuThreadsPerRun} CPU threads · {resources.ramGbPerRun} GiB per run</small></summary><p className="muted">CPU threads run model operations; data workers load features. RAM is a scheduling reservation per run, not an enforced memory cap.</p><div className="development-fields">
            <BatchNumberList label="Allowed GPU IDs" value={gpus} onChange={setGpus} min={0} max={127} maxItems={128} allowEmpty hint="Comma-separated; leave empty for CPU." />
            {([['cpuThreadsPerRun', 'CPU threads per run', 1, 256], ['dataLoaderWorkers', 'Data workers per loader', 0, 64], ['ramGbPerRun', 'RAM reservation per run (GiB)', Number.MIN_VALUE, undefined]] as const).map(([key, label, min, max]) => <NumericField key={key} label={label} value={resources[key]} min={min} max={max} integer={key !== 'ramGbPerRun'} onChange={(number) => setResources((current) => ({ ...current, [key]: number }))} />)}
          </div></details></section>
          <BatchPredictorFields value={predictorPolicy} onChange={setPredictorPolicy} configurationCount={plannedConfigurations} trainingSeedCount={plannedSeeds} splitSeedCount={protocol?.split.mode === 'kfold' ? new Set(protocol.split.seeds).size : undefined} foldCount={protocol?.split.mode === 'kfold' ? protocol.split.folds : undefined} />
          </div>
        </fieldset>
        {batchPage === 4 ? <div className="batch-page-review"><h3>Review {name || 'this batch'}</h3><p>{plannedConfigurations ?? '—'} configurations × {plannedSeeds ?? '—'} training seeds. All batches use the experiment’s verified inputs and frozen splits.</p><BatchPredictorSummary spec={specification()} protocol={protocol} /><details className="setup-details"><summary>Review all batch settings</summary><BatchPlanSettings spec={specification()} /></details></div> : null}
        <div hidden={batchPage !== 4}><div className="inline-actions batch-editor-actions"><button type="button" className="btn btn-primary" disabled={busy || stale || !inputs.protocolId || !inputs.featureBundleId || !name.trim() || (!!workingPlan && !dirty)} onClick={() => void action('save')}>{busy ? 'Working…' : workingPlan ? 'Save batch changes' : 'Add batch to plan'}</button><button type="button" className="btn btn-secondary" disabled={busy || stale || !inputs.protocolId || !inputs.featureBundleId || !name.trim()} onClick={() => void action('preview')}>Check batch</button>{workingPlan || dirty ? <button className="text-button" disabled={busy} onClick={() => { if (load(batchTemplate('blank', inputs, experimentName), undefined, false, 'blank')) setDirty(false); }}>{dirty ? 'Discard batch edits' : 'New batch'}</button> : null}{dirty ? <span className="muted" role="status">Unsaved batch edits</span> : null}</div></div>
        <div className="stage-actions"><button type="button" className="btn btn-secondary" disabled={busy} onClick={() => batchPage > 1 ? setBatchPage((page) => page - 1) : setEditorOpen(false)}>{batchPage > 1 ? 'Back' : 'Back to batch plans'}</button><p>{batchPage === 4 ? 'Saving retains an editable batch. Training starts when you submit the experiment.' : 'Your settings are retained when you go back.'}</p>{batchPage < 4 ? <button type="button" className="btn btn-primary" disabled={busy || stale} onClick={() => showBatchPage(batchPage + 1)}>Continue to {batchPage === 1 ? 'training settings' : batchPage === 2 ? 'compute & predictors' : 'batch review'}</button> : null}</div>
      </Panel>
      {preview && batchPage === 4 ? <Panel title="Resolved batch"><Findings findings={preview.findings} /><p className="development-count" aria-live="polite"><strong>{preview.summary.configurationCount}</strong> configurations × <strong>{preview.summary.trainingSeedCount}</strong> training seeds × <strong>{preview.summary.splitPlanCount}</strong> frozen split plans = <strong>{preview.summary.runCount}</strong> planned runs</p><ConfigurationTable batch={preview} /></Panel> : null}
      {savedDrafts.length ? <details className="setup-details"><summary>Earlier batch drafts</summary><p className="muted">Load an earlier draft and add it to this experiment’s plan. Drafts are not submitted automatically.</p>{savedDrafts.map((draft) => <div className="development-saved-row" key={draft.id}><span>{draft.name}</span><button type="button" className="btn btn-secondary btn-small" onClick={() => load(draft.payload.spec as unknown as DevelopmentBatchSpec)}>Use draft settings</button></div>)}</details> : null}
    </div> : null}
    {(tab !== 'batches' || locked || !editorOpen) && (items.length || tab !== 'batches') ? <Panel title={tab === 'runs' ? 'Runs' : tab === 'results' ? 'Results' : 'Frozen batches'} subtitle={tab === 'batches' ? 'Saved configurations and split memberships are immutable. Experiment submission includes every active frozen batch.' : undefined}>
      {!items.length ? <p>No submitted batches are available yet.</p> : <>
        <div className="development-batch-selector"><label className="label">Batch<select className="field" value={selectedBatch?.id ?? ''} onChange={(e) => setSelected(e.target.value)}><option value="">Choose a batch in this experiment</option>{items.map((item) => <option key={item.id} value={item.id}>{item.manifest.spec.batchName} · {item.status} · {item.state}</option>)}</select></label>
        {selectedBatch ? <div className="inline-actions"><Badge>{selectedBatch.manifest.summary.runCount} runs in plan</Badge><button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${selectedBatch.manifest.spec.batchName}.json`, selectedBatch)}>Export batch</button>{tab === 'batches' && !locked ? <button type="button" className="btn btn-secondary btn-small" onClick={() => load(selectedBatch.manifest.spec, undefined, true)}>Use as editable batch</button> : null}</div> : null}</div>
        {selectedBatch ? <>
          {tab === 'batches' ? <><BatchPredictorSummary spec={selectedBatch.manifest.spec} protocol={protocol} fallbackPredictorPolicy={record?.predictorPolicies?.[selectedBatch.id] ?? record?.predictorPolicy ?? undefined} /><ConfigurationTable batch={selectedBatch.manifest} /></> : null}
          {tab === 'batches' && !locked ? <ExperimentLifecycle key={`lifecycle:${selectedBatch.id}`} project={project} recordKey={selectedBatch.key} state={selectedBatch.state} name={selectedBatch.manifest.spec.batchName} /> : null}
          {tab !== 'batches' || experimentStage === undefined ? <DevelopmentExecution key={selectedBatch.id} project={project} batch={selectedBatch} implemented={selectedBatch.state !== 'trashed' && executionImplemented} stage={experimentStage} readOnly={experimentStage === 'finished' || !!record?.legacy || (!!record && record.state !== 'active')} allowChanges={(experimentStage === 'running' || !readOnly) && selectedBatch.state === 'active'} knownExecution={selectedBatch.execution ?? undefined} view={tab} /> : null}
        </> : null}
      </>}
    </Panel> : null}
  </StagePage>;
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
