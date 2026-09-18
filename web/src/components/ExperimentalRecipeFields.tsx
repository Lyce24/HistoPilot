import type { TrainingRecipe } from '../api/development';
import { withRecipeDefaults } from '../api/development';
import NumericField from './NumericField';
import { usesPatchFeatures } from '../lib/modelCapabilities';

type RecipeProps = { value: TrainingRecipe; onChange: (value: TrainingRecipe) => void };
export const lossLabels = { ce: 'Cross entropy', bce: 'Binary cross entropy', focal: 'Focal loss' };
export const samplingLabels = { slide_uniform: 'Uniform slides', patient_natural: 'Uniform patients', cohort_balanced: 'Balanced cohorts', cohort_label_balanced: 'Balanced cohorts and labels' };
export const scheduleLabels = { none: 'Constant learning rate', cosine: 'Cosine decay', plateau: 'Reduce on validation plateau', step: 'Step decay' };

function OptionalNumber({ label, value, fallback, onChange, integer = true, max = 1000000, min = 1 }: {
  label: string; value: number | null | undefined; fallback: number; onChange: (value: number | null) => void;
  integer?: boolean; min?: number; max?: number;
}) {
  return <div className="stack">
    <label className="development-check"><input type="checkbox" checked={value != null} onChange={(event) => onChange(event.target.checked ? fallback : null)} />{label}</label>
    {value != null ? <NumericField label={label} value={value} min={min} minExclusive={!integer && min === 0} max={max} integer={integer} onChange={onChange} /> : null}
  </div>;
}

export function ObjectiveFields({ value, onChange, classes = [] }: RecipeProps & { classes?: string[] }) {
  const resolved = withRecipeDefaults(value);
  const weighting = resolved.classWeights ? 'explicit' : resolved.classWeighting;
  return <details className="batch-settings-details setup-details"><summary><span>Loss &amp; patient predictions</span>{' '}<small>{lossLabels[resolved.lossType!]} · {resolved.patientAggregation === 'mean_logits' ? 'Mean logits per patient' : 'Mean probabilities per patient'}</small></summary>
    <div className="development-fields">
      <label className="label">Training loss<select className="field" value={resolved.lossType} onChange={(event) => onChange({ ...value, lossType: event.target.value as TrainingRecipe['lossType'], labelSmoothing: event.target.value === 'ce' ? resolved.labelSmoothing : 0 })}>
        <option value="ce">Cross entropy</option><option value="bce" disabled={classes.length > 2}>Binary cross entropy (two classes)</option><option value="focal">Focal loss</option>
      </select></label>
      <label className="label">Class loss weights<select className="field" value={weighting} onChange={(event) => onChange({ ...value,
        classWeighting: event.target.value === 'inverse_prevalence' ? 'inverse_prevalence' : 'none',
        classWeights: event.target.value === 'explicit' ? (classes.length ? classes : ['Negative', 'Positive']).map(() => 1) : null,
      })}><option value="none">Equal weights</option><option value="inverse_prevalence">Inverse prevalence in the training fold</option><option value="explicit">Set weights for each class</option></select><small>Automatic weights count training patients for patient targets or patient sampling; otherwise they count slides. Prevalence is measured before sampling adjustments.</small></label>
      {resolved.lossType === 'focal' ? <NumericField label="Focal gamma" value={resolved.focalGamma!} min={0} integer={false} onChange={(focalGamma) => onChange({ ...value, focalGamma })} /> : null}
      {resolved.lossType === 'ce' ? <NumericField label="Label smoothing" value={resolved.labelSmoothing!} min={0} max={1} maxExclusive integer={false} onChange={(labelSmoothing) => onChange({ ...value, labelSmoothing })} /> : null}
      <label className="label">Patient prediction aggregation<select className="field" value={resolved.patientAggregation} onChange={(event) => onChange({ ...value, patientAggregation: event.target.value as TrainingRecipe['patientAggregation'] })}>
        <option value="mean_probabilities">Average slide probabilities</option><option value="mean_logits">Average slide logits, then convert to probabilities</option>
      </select><small>Applied when reporting and selecting checkpoints by patient-level results.</small></label>
      <label className="label">Ensemble member aggregation<select className="field" value={resolved.ensembleAggregation} onChange={(event) => onChange({ ...value, ensembleAggregation: event.target.value as TrainingRecipe['ensembleAggregation'] })}>
        <option value="mean_probability">Average member probabilities</option><option value="mean_logit">Average member logits, then convert to probabilities</option>
      </select><small>Used when this batch creates an ensemble predictor.</small></label>
    </div>
    {resolved.classWeights ? <div className="development-fields">{resolved.classWeights.map((weight, index) => <NumericField key={index} label={`Loss weight · ${classes[index] ?? `class ${index + 1}`}`} value={weight} min={0} minExclusive integer={false} onChange={(number) => onChange({ ...value, classWeights: resolved.classWeights!.map((item, at) => at === index ? number : item) })} />)}</div> : null}
  </details>;
}

export function SamplingFields({ value, onChange }: RecipeProps) {
  const resolved = withRecipeDefaults(value);
  const patchFeatures = usesPatchFeatures(value.model, value.inputMode);
  const nnmilSampler = patchFeatures && value.model === 'nnmil' && resolved.nnmilBatchSampler !== 'patient_weighted';
  return <details className="batch-settings-details setup-details"><summary><span>Sampling &amp; augmentation</span>{' '}<small>{samplingLabels[resolved.samplingStrategy!]}{resolved.classWeightedSampling ? ' · Class weighted' : ''}{patchFeatures && resolved.bagCurriculum ? ' · Growing bags' : ''}</small></summary>
    <p>For patient targets, uniform slide sampling uses inverse slide-count loss weights so every patient has equal expected training weight. Uniform patient sampling draws one slide per patient each epoch. Class or cohort balancing deliberately changes that weighting.</p>
    <div className="development-fields">
      <label className="label">Training record sampling<select className="field" disabled={nnmilSampler} value={resolved.samplingStrategy} onChange={(event) => onChange({ ...value, samplingStrategy: event.target.value as TrainingRecipe['samplingStrategy'], classWeightedSampling: event.target.value === 'slide_uniform' ? resolved.classWeightedSampling : false })}>
        {Object.entries(samplingLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select><small>Sampling changes training exposure; dataset membership and held-out splits stay fixed.</small></label>
      {resolved.samplingStrategy === 'cohort_balanced' || resolved.samplingStrategy === 'cohort_label_balanced' ? <label className="label">Cohort column<input className="field" value={resolved.cohortColumn} required maxLength={128} onChange={(event) => onChange({ ...value, cohortColumn: event.target.value })} /></label> : null}
      {resolved.samplingStrategy === 'slide_uniform' ? <label className="development-check"><input type="checkbox" disabled={nnmilSampler} checked={resolved.classWeightedSampling} onChange={(event) => onChange({ ...value, classWeightedSampling: event.target.checked })} />Class-weighted sampling</label> : null}
      {resolved.samplingStrategy === 'cohort_label_balanced' ? <NumericField label="Target positive prevalence" value={resolved.samplingPositivePrevalence!} min={0} minExclusive max={1} maxExclusive integer={false} onChange={(samplingPositivePrevalence) => onChange({ ...value, samplingPositivePrevalence })} /> : null}
      {patchFeatures ? <NumericField label="Instance dropout" value={resolved.instanceDropout!} min={0} max={1} maxExclusive integer={false} onChange={(instanceDropout) => onChange({ ...value, instanceDropout })} /> : null}
      {value.inputMode !== 'clinical' ? <NumericField label="Feature noise standard deviation" value={resolved.featureNoiseStd!} min={0} integer={false} onChange={(featureNoiseStd) => onChange({ ...value, featureNoiseStd })} /> : null}
    </div>
    {nnmilSampler ? <p className="muted">The nnMIL sampler above composes training batches. Choose HistoPilot record sampling there to edit record sampling and class weighting here.</p> : null}
    {patchFeatures ? <><p className="muted">Instance dropout removes random patches during training. Feature noise adds random perturbations during training. Zero disables either augmentation.</p>
    <label className="development-check"><input type="checkbox" disabled={resolved.bagSizeMode === 'training_median'} checked={resolved.bagCurriculum} onChange={(event) => onChange({ ...value, bagCurriculum: event.target.checked })} />Grow the training bag over early epochs</label>
    {resolved.bagSizeMode === 'training_median' ? <p className="muted">Choose a fixed training bag to enable a bag curriculum.</p> : null}
    {resolved.bagCurriculum ? <><div className="development-fields">
      <NumericField label="Starting patches per bag" value={resolved.bagCurriculumStart!} min={1} max={resolved.bagCurriculumEnd} onChange={(bagCurriculumStart) => onChange({ ...value, bagCurriculumStart })} />
      <NumericField label="Final patches per bag" value={resolved.bagCurriculumEnd!} min={resolved.bagCurriculumStart} max={1000000} onChange={(bagCurriculumEnd) => onChange({ ...value, bagCurriculumEnd })} />
      <NumericField label="Bag curriculum warmup epochs" value={resolved.bagCurriculumWarmupEpochs!} min={1} max={100000} onChange={(bagCurriculumWarmupEpochs) => onChange({ ...value, bagCurriculumWarmupEpochs })} />
    </div><p className="muted">The curriculum replaces the training bag limit above and grows to its final patch count.</p></> : null}</> : value.inputMode !== 'clinical' ? <p className="muted">Feature noise perturbs the slide embedding during training. Zero disables it.</p> : null}
  </details>;
}

export function EvaluationBagFields({ value, onChange }: RecipeProps) {
  const resolved = withRecipeDefaults(value);
  const patchFeatures = usesPatchFeatures(value.model, value.inputMode);
  return <details className="batch-settings-details setup-details"><summary><span>Validation &amp; assessment loading</span>{' '}<small>{patchFeatures ? resolved.evalBagSize == null ? 'Whole bags' : `${resolved.evalBagSize} patches per bag` : 'One embedding per slide'} · Batch {resolved.evalBatchSize ?? resolved.batchSize}</small></summary>
    <div className="development-fields">
      {patchFeatures ? <OptionalNumber label="Limit validation and assessment patches per bag" value={resolved.evalBagSize} fallback={4096} onChange={(evalBagSize) => onChange({ ...value, evalBagSize })} /> : null}
      <OptionalNumber label="Use a separate evaluation batch size" value={resolved.evalBatchSize} fallback={resolved.batchSize} max={4096} onChange={(evalBatchSize) => onChange({ ...value, evalBatchSize })} />
    </div>{patchFeatures ? <p className="muted">By default, validation and assessment use every patch. An explicit limit uses a deterministic sample and can change reported metrics.</p> : <p className="muted">The complete slide embedding is used for each record.</p>}
  </details>;
}

export function OptimizerFields({ value, onChange }: RecipeProps) {
  const resolved = withRecipeDefaults(value);
  return <>
    <label className="label">Weight decay applies to<select className="field" value={resolved.weightDecayPolicy} onChange={(event) => onChange({ ...value, weightDecayPolicy: event.target.value as TrainingRecipe['weightDecayPolicy'] })}><option value="all">All trainable parameters</option><option value="weights_only">Weights only; exclude biases and vectors</option></select></label>
    {resolved.lrScheduler === 'cosine' ? <label className="label">Cosine schedule interval<select className="field" value={resolved.lrScheduleInterval} onChange={(event) => onChange({ ...value, lrScheduleInterval: event.target.value as TrainingRecipe['lrScheduleInterval'] })}><option value="epoch">Every epoch</option><option value="step">Every optimizer update</option></select></label> : null}
    {resolved.lrScheduler === 'plateau' ? <p className="muted">Refits use a constant learning rate because they have no validation partition. The refit records this schedule change.</p> : null}
    {resolved.lrScheduler === 'plateau' || resolved.lrScheduler === 'step' ? <>
      <NumericField label="LR decay factor" value={resolved.lrGamma!} min={0} minExclusive max={1} maxExclusive integer={false} onChange={(lrGamma) => onChange({ ...value, lrGamma })} />
      {resolved.lrScheduler === 'step' ? <NumericField label="Epochs between LR steps" value={resolved.lrStepSize!} min={1} max={100000} onChange={(lrStepSize) => onChange({ ...value, lrStepSize })} />
        : <NumericField label="LR plateau patience" value={resolved.lrPlateauPatience!} min={0} max={100000} onChange={(lrPlateauPatience) => onChange({ ...value, lrPlateauPatience })} />}
    </> : null}
    {resolved.optimizer !== 'sgd' ? <>
      <NumericField label="Adam beta 1" value={resolved.adamBetas![0]} min={0} max={1} maxExclusive integer={false} onChange={(beta) => onChange({ ...value, adamBetas: [beta, resolved.adamBetas![1]] })} />
      <NumericField label="Adam beta 2" value={resolved.adamBetas![1]} min={0} max={1} maxExclusive integer={false} onChange={(beta) => onChange({ ...value, adamBetas: [resolved.adamBetas![0], beta] })} />
      <NumericField label="Adam epsilon" value={resolved.adamEps!} min={0} minExclusive integer={false} onChange={(adamEps) => onChange({ ...value, adamEps })} />
    </> : null}
    <OptionalNumber label="Use a separate aggregator learning rate" value={resolved.aggregatorLearningRate} fallback={resolved.learningRate} min={0} integer={false} onChange={(aggregatorLearningRate) => onChange({ ...value, aggregatorLearningRate })} />
    <OptionalNumber label="Use a separate classifier learning rate" value={resolved.headLearningRate} fallback={resolved.learningRate} min={0} integer={false} onChange={(headLearningRate) => onChange({ ...value, headLearningRate })} />
  </>;
}

export function StoppingPolicyFields({ value, onChange }: RecipeProps) {
  const resolved = withRecipeDefaults(value);
  const policy = resolved.minValidationPositives != null ? 'fallback' : resolved.fixedEpochBudget != null ? 'fixed' : 'validation';
  return <div className="development-fields">
    <label className="label">Epoch selection policy<select className="field" value={policy} onChange={(event) => onChange({ ...value,
      minValidationPositives: event.target.value === 'fallback' ? resolved.minValidationPositives ?? 1 : null,
      fixedEpochBudget: event.target.value === 'validation' ? null : resolved.fixedEpochBudget ?? resolved.maxEpochs,
    })}><option value="validation">Use validation and early-stopping settings</option><option value="fixed">Always train an explicit epoch budget</option><option value="fallback">Use fixed epochs when validation positives are too few</option></select></label>
    {policy !== 'validation' ? <NumericField label="Fixed epoch budget" value={resolved.fixedEpochBudget!} min={Math.max(resolved.minEpochs!, resolved.warmupEpochs! + 1, 1)} max={resolved.maxEpochs} onChange={(fixedEpochBudget) => onChange({ ...value, fixedEpochBudget })} /> : null}
    {policy === 'fallback' ? <NumericField label="Minimum validation positives" value={resolved.minValidationPositives!} min={1} max={1000000} onChange={(minValidationPositives) => onChange({ ...value, minValidationPositives })} /> : null}
    {policy !== 'validation' ? <p className="muted">The reviewed budget selects the final epoch when this policy applies. A fallback requires the explicit budget shown here.</p> : null}
  </div>;
}

export function ExperimentalRecipeSummary({ recipe }: { recipe: TrainingRecipe }) {
  const value = withRecipeDefaults(recipe);
  const patchFeatures = usesPatchFeatures(recipe.model, recipe.inputMode);
  return <>
    <div><dt>Prediction inputs</dt><dd>{value.inputMode === 'clinical' ? 'Clinical-only logistic baseline' : value.inputMode === 'multimodal' ? 'Clinical + image, additive logit fusion' : 'Image only'}</dd></div>
    {value.clinicalFields?.length ? <div><dt>Clinical fields</dt><dd>{value.clinicalFields.map((item) => `${item.field} (${item.kind})`).join(', ')} · Preprocessing fitted on training patients only</dd></div> : null}
    {value.model === 'nnmil' ? <>
      <div><dt>nnMIL feature views</dt><dd>{value.nnmilFeatureSampling ? 'Random training feature views' : 'All feature coordinates during training'} · Stride divisor {value.nnmilWindowStrideDivisor} · {value.nnmilWindowShuffle ? `Shuffled coordinates, seed ${value.nnmilWindowSeed}` : 'Original coordinate order'}</dd></div>
      <div><dt>Feature-window aggregation</dt><dd>{value.nnmilWindowAggregation === 'mean_logits' ? 'Average window logits, then convert to probabilities' : 'Average window probabilities'}</dd></div>
      <div><dt>nnMIL sampler</dt><dd>{value.nnmilBatchSampler === 'class_balanced' ? 'Class-balanced batches' : value.nnmilBatchSampler === 'auc_stratified' ? 'AUC-stratified batches' : 'HistoPilot record sampling'}</dd></div>
      <div><dt>nnMIL evaluation checkpoint</dt><dd>{value.nnmilCheckpointSelection === 'latest' ? 'Latest completed epoch' : 'Best validation checkpoint'}</dd></div>
    </> : null}
    <div><dt>Optimizer protocol</dt><dd>Weight decay on {value.weightDecayPolicy === 'weights_only' ? 'weights only; biases and vectors excluded' : 'all trainable parameters'}{value.lrScheduler === 'cosine' ? ` · Cosine schedule every ${value.lrScheduleInterval === 'step' ? 'optimizer update' : 'epoch'}` : ''}</dd></div>
    {value.lrScheduler === 'plateau' ? <div><dt>Refit schedule</dt><dd>Constant learning rate; refits do not use validation.</dd></div> : null}
    <div><dt>Training loss</dt><dd>{lossLabels[value.lossType!]} · {value.classWeights ? `Weights ${value.classWeights.join(', ')}` : value.classWeighting === 'inverse_prevalence' ? 'Inverse training-fold prevalence weights' : 'Equal class weights'}{value.lossType === 'focal' ? ` · Gamma ${value.focalGamma}` : value.labelSmoothing ? ` · Label smoothing ${value.labelSmoothing}` : ''}</dd></div>
    <div><dt>Patient aggregation</dt><dd>{value.patientAggregation === 'mean_logits' ? 'Average logits, then convert to probabilities' : 'Average slide probabilities'}</dd></div>
    <div><dt>Ensemble aggregation</dt><dd>{value.ensembleAggregation === 'mean_logit' ? 'Average member logits, then convert to probabilities' : 'Average member probabilities'}</dd></div>
    <div><dt>Training sampling</dt><dd>{samplingLabels[value.samplingStrategy!]}{value.samplingStrategy?.startsWith('cohort') ? ` · Column ${value.cohortColumn}` : ''}{value.samplingStrategy === 'cohort_label_balanced' ? ` · Positive prevalence ${value.samplingPositivePrevalence}` : value.classWeightedSampling ? ' · Inverse-prevalence class sampling' : ''}</dd></div>
    {value.inputMode !== 'clinical' ? <div><dt>Training augmentation</dt><dd>{patchFeatures ? `Instance dropout ${value.instanceDropout} · ` : ''}Feature noise SD {value.featureNoiseStd}{patchFeatures && value.bagCurriculum ? ` · Bag curriculum ${value.bagCurriculumStart} → ${value.bagCurriculumEnd} patches over ${value.bagCurriculumWarmupEpochs} epochs` : ''}</dd></div> : null}
    <div><dt>Evaluation inputs</dt><dd>{patchFeatures ? value.evalBagSize == null ? 'All patches' : `${value.evalBagSize} patches maximum` : value.inputMode === 'clinical' ? 'Clinical variables only' : 'One embedding per slide'} · Batch size {value.evalBatchSize ?? value.batchSize}</dd></div>
    {value.fixedEpochBudget != null ? <div><dt>Explicit epoch policy</dt><dd>{value.fixedEpochBudget} epochs{value.minValidationPositives != null ? ` when validation has fewer than ${value.minValidationPositives} positives` : ' in every fold'}</dd></div> : null}
    {value.aggregatorLearningRate != null || value.headLearningRate != null ? <div><dt>Component learning rates</dt><dd>Aggregator {value.aggregatorLearningRate ?? value.learningRate} · Classifier {value.headLearningRate ?? value.learningRate}</dd></div> : null}
    {value.optimizer !== 'sgd' ? <div><dt>Adam settings</dt><dd>Betas {value.adamBetas!.join(', ')} · Epsilon {value.adamEps}</dd></div> : null}
  </>;
}
