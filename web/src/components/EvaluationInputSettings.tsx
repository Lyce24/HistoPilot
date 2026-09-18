import { useQuery } from '@tanstack/react-query';
import { bundles } from '../api/bundles';
import type { EvaluationCohort, EvaluationInference } from '../api/evaluation';
import type { ProtocolSpec } from '../api/scientific';
import type { ModelEvaluation } from '../api/predictors';
import { versionLabelText } from '../lib/versionLabels';
import NumericField from './NumericField';
import { ErrorNotice } from './ui';

export interface EvaluationExecutionInputs {
  featureBundleId: string;
  inference: EvaluationInference;
  patientIdentifiers: 'shared' | 'independent';
}

export function initialEvaluationInputs(cohort?: EvaluationCohort): EvaluationExecutionInputs {
  return {
    featureBundleId: cohort?.manifest.spec.featureBundleId ?? '',
    patientIdentifiers: cohort?.manifest.spec.patientIdentifiers ?? 'shared',
    inference: { loadingPolicy: 'per_slide', packArtifactId: null, batchSize: 1, numWorkers: 0,
      device: 'auto', precision: 'float32',
      ...cohort?.manifest.spec.inference, patientAggregation: 'predictor', decisionThreshold: 'predictor' },
  };
}

export function evaluationExecutionSelection(inputs: EvaluationExecutionInputs) {
  return { ...(inputs.featureBundleId ? { featureBundleId: inputs.featureBundleId } : {}), inference: inputs.inference, patientIdentifiers: inputs.patientIdentifiers };
}

export function EvaluationCoverageSummary({ manifest }: { manifest?: ModelEvaluation['manifest'] | null }) {
  const coverage = manifest?.coverage;
  if (!coverage) return null;
  const selected = coverage.selectedSlideIds.length;
  return <div className="evaluation-coverage-summary">
    <p><strong>{selected - coverage.missingFeatureSlideIds.length} / {selected}</strong> selected slides have extracted features.</p>
    <p>{coverage.packChecked ? <><strong>{selected - coverage.missingPackSlideIds.length} / {selected}</strong> selected slides are in the feature pack.</> : 'Original feature files selected; packing is not required.'}</p>
    {manifest?.overlap ? <p>{manifest.overlap.slideIds.length} overlapping development slide IDs · {manifest.overlap.patientsComparable ? `${manifest.overlap.patientIds.length} overlapping patient IDs` : 'Patient IDs use separate naming systems; patient overlap cannot be checked by ID.'}</p> : null}
  </div>;
}

export function EvaluationInferenceFields({ value, target, onChange }: {
  value: EvaluationInference; target?: ProtocolSpec['target'] | null; onChange: (update: Partial<EvaluationInference>) => void;
}) {
  return <details className="evaluation-advanced"><summary>Inference settings</summary><div className="chain-fields">
    <NumericField label="Batch size" min={1} max={1024} value={value.batchSize} onChange={(batchSize) => onChange({ batchSize })} />
    <NumericField label="Data-loading workers" min={0} max={64} value={value.numWorkers} onChange={(numWorkers) => onChange({ numWorkers })} />
    <label className="label">Device<select className="field" value={value.device} onChange={(event) => onChange({ device: event.target.value as EvaluationInference['device'] })}><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label>
    <label className="label">Inference precision<select className="field" value={value.precision} onChange={(event) => onChange({ precision: event.target.value as EvaluationInference['precision'] })}><option value="float32">Float32</option><option value="float16">Float16</option><option value="bfloat16">BFloat16</option></select></label>
    {target?.unit === 'patient' || value.patientAggregation !== 'mean' ? <label className="label">Combine slides for each patient<select className="field" value={value.patientAggregation} onChange={(event) => onChange({ patientAggregation: event.target.value as EvaluationInference['patientAggregation'] })}>
      {value.patientAggregation === 'max' ? <option value="max" disabled>Maximum probabilities · unsupported; choose mean</option> : null}
      <option value="predictor">Use the predictor's patient scoring rule</option>
      <option value="mean">Mean probabilities</option>
      <option value="mean_logits">Mean logits</option>
    </select><small>Match the patient scoring rule saved with the predictor.</small></label> : null}
    {target?.task === 'binary_classification' ? <div><label className="label">Threshold policy<select className="field" value={value.decisionThreshold === 'predictor' ? 'predictor' : 'explicit'} onChange={(event) => onChange({ decisionThreshold: event.target.value === 'predictor' ? 'predictor' : 0.5 })}><option value="predictor">Use frozen predictor threshold</option><option value="explicit">Explicit threshold</option></select></label>{value.decisionThreshold !== 'predictor' ? <NumericField label="Decision threshold" integer={false} min={0} minExclusive max={1} maxExclusive value={value.decisionThreshold} onChange={(decisionThreshold) => onChange({ decisionThreshold })} /> : null}<small>New predictors require the threshold frozen during development. Historical predictors without a saved threshold default to 0.5.</small></div> : null}
  </div></details>;
}

/** Model-dependent feature and loading choices belong to each evaluation plan. */
export default function EvaluationInputSettings({ project, cohort, value, onChange, disabled = false }: {
  project: string; cohort?: EvaluationCohort; value: EvaluationExecutionInputs;
  onChange: (value: EvaluationExecutionInputs) => void; disabled?: boolean;
}) {
  const query = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project), enabled: Boolean(cohort) });
  if (!cohort) return null;
  const items = query.data?.items ?? [];
  const selected = items.find((item) => item.id === value.featureBundleId);
  const packBundles = selected ? [selected] : value.featureBundleId ? [] : items;
  const packs = [...new Map(packBundles.flatMap((bundle) => bundle.manifest.packs.map((pack) => [pack.id, { pack, bundle }] as const))).values()];
  const inference = (update: Partial<EvaluationInference>) => onChange({ ...value, inference: { ...value.inference, ...update } });
  return <fieldset className="evaluation-input-settings" disabled={disabled}>
    <legend>Test features and inference</legend>
    <ErrorNotice error={query.error} />
    <p className="muted">Review matches the prediction task and class encoding to each development model, then checks extracted features, encoder dimensions, packed-slide coverage and development overlap.</p>
    <div className="chain-fields">
      <label className="label">Test feature bundle<select className="field" value={value.featureBundleId} onChange={(event) => onChange({ ...value, featureBundleId: event.target.value, inference: { ...value.inference, packArtifactId: null } })}>
        <option value="">Find compatible features automatically</option>
        {value.featureBundleId && !selected ? <option value={value.featureBundleId}>Saved feature bundle unavailable</option> : null}
        {items.map((item) => <option key={item.id} value={item.id}>{versionLabelText(item, 'Feature bundle')} · {item.manifest.summary.slideCount} slides · {item.manifest.summary.dimensions ?? '?'} dimensions{!item.current ? ' · needs verification' : ''}</option>)}
      </select><small>Automatic selection requires one compatible feature inventory. Review explains missing or ambiguous features.</small></label>
      <label className="label">Load test features<select className="field" value={value.inference.loadingPolicy} onChange={(event) => inference({ loadingPolicy: event.target.value as EvaluationInference['loadingPolicy'], packArtifactId: null })}><option value="per_slide">Original feature files</option><option value="packed">Packed features</option></select></label>
      <label className="label">Patient identifiers across datasets<select className="field" value={value.patientIdentifiers} onChange={(event) => onChange({ ...value, patientIdentifiers: event.target.value as EvaluationExecutionInputs['patientIdentifiers'] })}><option value="shared">Shared IDs identify the same patients</option><option value="independent">IDs belong to separate naming systems</option></select><small>Separate naming systems apply when the same ID can mean different people. This choice does not establish whether patients overlap.</small></label>
      {value.inference.loadingPolicy === 'packed' ? <label className="label">Test feature pack<select className="field" value={value.inference.packArtifactId ?? ''} onChange={(event) => inference({ packArtifactId: event.target.value || null })}>
        <option value="">Choose a feature pack</option>
        {value.inference.packArtifactId && !packs.some(({ pack }) => pack.id === value.inference.packArtifactId) ? <option value={value.inference.packArtifactId}>Saved feature pack unavailable</option> : null}
        {packs.map(({ pack, bundle }) => <option key={pack.id} value={pack.id}>{versionLabelText(bundle, 'Feature bundle')} · {pack.outputDtype} · {pack.id}</option>)}
      </select></label> : null}
    </div>
    {!query.isPending && !query.isError && items.length === 0 ? <p className="callout">No frozen feature bundles are available. Prepare extracted features in <a href="#features">Slide features</a>, then review this evaluation again. The test cohort is already saved.</p> : null}
    <EvaluationInferenceFields value={value.inference} target={cohort.manifest.target} onChange={inference} />
  </fieldset>;
}
