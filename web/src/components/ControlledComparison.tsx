import { useEffect, useRef, useState } from 'react';
import { defaultRecipe } from '../api/development';
import type { ComparisonMetric, DevelopmentBatchSpec, TrainingRecipe } from '../api/development';
import type { Finding } from '../api/scientific';
import { featureKindOf, modelLabel, modelsForFeatureKind, type FeatureKind } from '../lib/modelCapabilities';
import { clinicalOnlyRecipe, matchedInputRecipes } from './ClinicalInputFields';
import { StageCreateButton } from './StageActions';
import { Findings } from './ScientificUI';

/** The service accepts two to eight arms in one controlled comparison. */
export const MAX_COMPARISON_ARMS = 8;
export const comparisonMetrics: { id: ComparisonMetric; label: string }[] = [
  { id: 'auroc', label: 'AUROC' }, { id: 'auprc', label: 'AUPRC' }, { id: 'balancedAccuracy', label: 'Balanced accuracy' },
  { id: 'macroF1', label: 'Macro F1' }, { id: 'accuracy', label: 'Accuracy' },
];
export const comparisonMetricLabel = (metric: ComparisonMetric) => comparisonMetrics.find((item) => item.id === metric)?.label ?? metric;

/** The editor's comparison choice. The reference is a configuration row, so it survives removing other rows. */
export interface ComparisonDraft { referenceRow: number; primaryMetric: ComparisonMetric }

type InputMode = NonNullable<TrainingRecipe['inputMode']>;
const inputChoices: { id: InputMode; label: string }[] = [
  { id: 'image', label: 'Image only' }, { id: 'clinical', label: 'Clinical only' }, { id: 'multimodal', label: 'Clinical + image' },
];

/** A short arm name: "ABMIL · image", "nnMIL · clinical + image" or "clinical only". */
export function armLabel(recipe: TrainingRecipe) {
  if (recipe.inputMode === 'clinical') return 'clinical only';
  return `${modelLabel(recipe.model)} · ${recipe.inputMode === 'multimodal' ? 'clinical + image' : 'image'}`;
}

/** "Controlled comparison · reference: configuration 1 (ABMIL · image) · primary metric AUROC" */
export function comparisonSummary(spec: Pick<DevelopmentBatchSpec, 'configurations' | 'comparison'>) {
  if (!spec.comparison) return null;
  const reference = spec.configurations[spec.comparison.reference - 1];
  return `Controlled comparison · reference: configuration ${spec.comparison.reference}${reference ? ` (${armLabel(reference)})` : ''} · primary metric ${comparisonMetricLabel(spec.comparison.primaryMetric)}`;
}

const nnmilOff = { nnmilBatchSampler: 'patient_weighted', nnmilCheckpointSelection: 'best_validation', nnmilWindowSeedFromTraining: false } as const;

/**
 * The base recipe with another model. Only settings that model owns change: nnMIL gets its
 * attention width and a new feature-window order per training seed, ABMIL its standard
 * attention, and non-nnMIL models drop nnMIL-only choices. Optimisation, bags, epochs and
 * batch size stay the base recipe's, so the comparison measures the model alone.
 */
export function withArmModel(base: TrainingRecipe, model: string): TrainingRecipe {
  if (model === base.model) return base;
  const reference = defaultRecipe();
  const own = model === 'nnmil' ? { attentionDim: 256, gatedAttention: true, nnmilWindowSeedFromTraining: true }
    : { ...nnmilOff, ...(model === 'abmil' ? { attentionDim: reference.attentionDim, gatedAttention: reference.gatedAttention } : {}) };
  // A slide embedding has no patch bag; this only applies when the base recipe is clinical only.
  const slide = featureKindOf(model) === 'slide' && featureKindOf(base.model) !== 'slide'
    ? { bagSize: 1, bagCurriculum: false, instanceDropout: 0, evalBagSize: null, gradientCheckpointing: false } : {};
  return { ...base, model, ...own, ...slide };
}

/**
 * Every ticked model × input as explicit configurations, the base recipe first as the
 * reference. Clinical-only arms read no image, so there is one whatever models are ticked.
 */
export function ablationArms(base: TrainingRecipe, models: string[], inputs: InputMode[]): TrainingRecipe[] {
  const baseInput = base.inputMode ?? 'image';
  const clinical = Boolean(base.clinicalFields?.length);
  const arms: TrainingRecipe[] = [base];
  for (const input of ['image', 'multimodal'] as const) {
    if (!inputs.includes(input) || (input === 'multimodal' && !clinical)) continue;
    for (const model of models) {
      if (model === base.model && input === baseInput) continue;
      const armBase = withArmModel(base, model);
      const variants = clinical ? matchedInputRecipes(armBase) : [{ ...armBase, inputMode: 'image' as const, clinicalFields: [] }];
      arms.push(variants.find((arm) => (arm.inputMode ?? 'image') === input)!);
    }
  }
  if (inputs.includes('clinical') && clinical && baseInput !== 'clinical') arms.push(clinicalOnlyRecipe(base));
  return arms;
}

/** The editor state that turns arms into a declared comparison with the first arm as reference. */
export function comparisonFromArms(arms: TrainingRecipe[], nextId: () => number) {
  const rows = arms.map((recipe) => ({ id: nextId(), recipe }));
  return { rows, mode: 'explicit' as const, candidateSelection: 'all' as const,
    comparison: { referenceRow: rows[0].id, primaryMetric: 'auroc' as const } satisfies ComparisonDraft };
}

/** Builds comparison arms from the single configuration being edited. */
export function AblationArms({ recipe, featureKind, onCreate }: {
  recipe: TrainingRecipe; featureKind?: FeatureKind; onCreate: (arms: TrainingRecipe[]) => void;
}) {
  const baseInput = recipe.inputMode ?? 'image';
  const clinical = Boolean(recipe.clinicalFields?.length);
  const available = modelsForFeatureKind(featureKind ?? (baseInput === 'clinical' ? undefined : featureKindOf(recipe.model)));
  // The base recipe's model and inputs are always the reference arm; extra ticks are kept apart.
  const baseModel = baseInput === 'clinical' ? null : recipe.model;
  const [extraModels, setExtraModels] = useState<string[]>([]);
  const [extraInputs, setExtraInputs] = useState<InputMode[]>([]);
  const models = [...new Set([...(baseModel ? [baseModel] : []), ...extraModels.filter((model) => available.some((item) => item.name === model))])];
  const inputs = [...new Set([baseInput, ...extraInputs.filter((input) => input === 'image' || clinical)])];
  const arms = ablationArms(recipe, models, inputs);
  const tooMany = arms.length > MAX_COMPARISON_ARMS;
  const toggle = <T,>(list: T[], item: T, on: boolean) => on ? [...list, item] : list.filter((entry) => entry !== item);
  // Ticks here only plan arms; stopping the change event keeps them from marking the batch as edited.
  return <section className="batch-editor-section batch-ablation" aria-label="Ablation arms">
    <div className="batch-section-heading"><h3>Ablation arms (optional)</h3>
      <p>Compare models or inputs with everything else held equal. Each ticked model is combined with each ticked input; every other setting is copied from this configuration. This configuration becomes the reference, configuration 1.</p></div>
    <div className="batch-ablation-choices">
      <fieldset><legend>Models</legend>
        {available.map((item) => {
          const locked = item.name === baseModel;
          return <label key={item.name} className="development-check"><input type="checkbox" checked={locked || extraModels.includes(item.name)} disabled={locked}
            onChange={(event) => { event.stopPropagation(); setExtraModels((current) => toggle(current, item.name, event.target.checked)); }} />{item.label}{locked ? <small> (this configuration)</small> : null}</label>;
        })}
      </fieldset>
      <fieldset><legend>Inputs</legend>
        {inputChoices.map((item) => {
          const locked = item.id === baseInput;
          const blocked = item.id !== 'image' && !clinical;
          return <label key={item.id} className="development-check"><input type="checkbox" checked={locked || (!blocked && extraInputs.includes(item.id))} disabled={locked || blocked}
            onChange={(event) => { event.stopPropagation(); setExtraInputs((current) => toggle(current, item.id, event.target.checked)); }} />{item.label}{locked ? <small> (this configuration)</small> : null}</label>;
        })}
        {!clinical ? <small>Select clinical fields under Model inputs to add clinical arms.</small> : null}
      </fieldset>
    </div>
    <div className="batch-size-summary" role="status" aria-live="polite">
      <strong>{arms.length} configuration{arms.length === 1 ? '' : 's'}</strong>
      <span>{arms.map((arm, index) => `${index + 1}. ${armLabel(arm)}${index === 0 ? ' (reference)' : ''}`).join(' · ')}</span>
      {tooMany ? <span role="alert">A controlled comparison holds at most {MAX_COMPARISON_ARMS} configurations. Untick some models or inputs.</span>
        : arms.length < 2 ? <span>Tick at least one more model or input.</span> : null}
    </div>
    <div className="inline-actions"><StageCreateButton type="button" tone="secondary" disabled={arms.length < 2 || tooMany} onClick={() => onCreate(arms)}>Create {arms.length} comparison configurations</StageCreateButton>
      <small className="muted">This switches the batch to custom configurations and turns on the controlled comparison.</small></div>
  </section>;
}

/** The comparison switch, reference and primary metric shown above custom configurations. */
export function ComparisonSettings({ rows, value, onChange }: {
  rows: { id: number; recipe: TrainingRecipe }[]; value: ComparisonDraft | null; onChange: (value: ComparisonDraft | null) => void;
}) {
  const reference = useRef<HTMLSelectElement>(null);
  const count = rows.length;
  const invalid = count < 2 || count > MAX_COMPARISON_ARMS
    ? `A controlled comparison needs 2 to ${MAX_COMPARISON_ARMS} configurations; this batch has ${count}.` : '';
  // The editor's page check reads native validity, so an out-of-range arm count blocks Continue here.
  useEffect(() => { reference.current?.setCustomValidity(invalid); }, [invalid, value]);
  const referenceRow = rows.find((row) => row.id === value?.referenceRow)?.id ?? rows[0]?.id;
  return <section className="batch-comparison-settings" aria-label="Controlled comparison">
    <label className="development-check"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked ? { referenceRow: rows[0].id, primaryMetric: 'auroc' } : null)} /><strong>Run as a controlled comparison</strong></label>
    <p>Every configuration is compared with one reference configuration on the same folds, training seeds and slides, and all of them are built. Configurations may differ only in the model and its inputs; if any other setting differs, the batch review blocks the batch.</p>
    {value ? <>
      <div className="development-fields">
        <label className="label">Reference configuration<select ref={reference} className="field" value={referenceRow} onChange={(event) => onChange({ ...value, referenceRow: Number(event.target.value) })}>
          {rows.map((row, index) => <option key={row.id} value={row.id}>Configuration {index + 1} · {armLabel(row.recipe)}</option>)}
        </select></label>
        <label className="label">Primary metric<select className="field" value={value.primaryMetric} onChange={(event) => onChange({ ...value, primaryMetric: event.target.value as ComparisonMetric })}>
          {comparisonMetrics.map((metric) => <option key={metric.id} value={metric.id}>{metric.label}{metric.id === 'auroc' ? ' (default)' : ''}</option>)}
        </select><small>Results give p-values, adjusted for the number of arms, on this metric.</small></label>
      </div>
      {invalid ? <p className="callout callout-warning" role="alert">{invalid} {count < 2 ? 'Add a configuration below.' : 'Remove configurations below.'}</p> : null}
    </> : null}
  </section>;
}

const settingLabels: Record<string, string> = {
  learningRate: 'learning rate', weightDecay: 'weight decay', maxEpochs: 'maximum epochs', batchSize: 'batch size',
  bagSize: 'patches per bag', bagSizeMode: 'training bag', bagSizeFraction: 'fraction of median patch count',
  lrScheduler: 'learning-rate schedule', lrScheduleInterval: 'cosine schedule interval', warmupEpochs: 'warmup epochs',
  finalLrFraction: 'final LR fraction', minEpochs: 'minimum training epochs', patience: 'early-stopping patience',
  earlyStopping: 'early stopping', checkpointMetric: 'checkpoint selection', evalBatchSize: 'evaluation batch size',
  evalBagSize: 'evaluation patch limit', weightDecayPolicy: 'weight decay target', gradientClipNorm: 'gradient clipping',
  accumulateGradBatches: 'accumulated batches', embedDim: 'embedding dimensions', numFcLayers: 'fully connected layers',
  dropout: 'dropout', inputDropout: 'input dropout', lossType: 'training loss', classWeighting: 'class loss weights',
  samplingStrategy: 'training record sampling', patientAggregation: 'patient prediction aggregation',
};
const settingLabel = (key: string) => settingLabels[key] ?? key.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();
/** The review names recipe keys; say them as the editor labels them. */
export const readableComparisonFinding = (finding: Finding): Finding => ({ ...finding,
  message: finding.message.replace(/(differs from the reference arm in )([^.]+)\./, (_, lead: string, keys: string) => `${lead}${keys.split(', ').map(settingLabel).join(', ')}.`) });

/** Comparison findings from the batch review, shown above the other findings. */
export function ComparisonReview({ findings }: { findings: Finding[] }) {
  const blocking = findings.filter((item) => item.code.startsWith('COMPARISON_')).map(readableComparisonFinding);
  if (!blocking.length) return <p className="callout batch-comparison-ready" role="status">The configurations differ only in the model and its inputs, so the comparison is controlled.</p>;
  return <div className="callout callout-warning batch-comparison-findings" role="alert">
    <p><strong>The controlled comparison cannot run yet.</strong> Configurations may differ only in the model and its inputs. Make the settings named below the same in every configuration, then check the batch again.</p>
    <Findings findings={blocking} />
  </div>;
}

