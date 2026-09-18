import type { NnMILPlanningRow, TrainingRecipe } from '../api/development';
import { withRecipeDefaults } from '../api/development';
import NumericField from './NumericField';

type RecipeProps = { value: TrainingRecipe; onChange: (recipe: TrainingRecipe) => void };

export function applyNnMILPaperOptimizer(recipe: TrainingRecipe): TrainingRecipe {
  return { ...recipe, optimizer: 'adamw', learningRate: 0.0003, weightDecay: 0.0001,
    aggregatorLearningRate: null, headLearningRate: null, adamBetas: [0.9, 0.999], adamEps: 1e-8,
    weightDecayPolicy: 'weights_only', lrScheduler: 'cosine', lrScheduleInterval: 'step',
    warmupEpochs: 5, finalLrFraction: 0 };
}

export function NnMILRecipeFields({ value, onChange }: RecipeProps) {
  const resolved = withRecipeDefaults(value);
  return <section className="batch-editor-section" aria-label="nnMIL protocol">
    <div className="batch-section-heading"><h3>nnMIL training &amp; testing</h3><p>Gated attention samples feature coordinates and pools the original full-dimensional features. Every setting below remains editable.</p></div>
    <div className="development-fields">
      <label className="development-check"><input type="checkbox" checked={resolved.nnmilFeatureSampling} onChange={(event) => onChange({ ...value, nnmilFeatureSampling: event.target.checked })} />Use feature sampling and windowed testing</label>
      <label className="label">nnMIL training sampler<select className="field" value={resolved.nnmilBatchSampler} onChange={(event) => onChange({ ...value, nnmilBatchSampler: event.target.value as TrainingRecipe['nnmilBatchSampler'], samplingStrategy: 'slide_uniform', classWeightedSampling: false })}>
        <option value="patient_weighted">HistoPilot record sampling</option><option value="class_balanced">Class-balanced batches</option><option value="auc_stratified">AUC-stratified batches</option>
      </select><small>{resolved.nnmilBatchSampler === 'patient_weighted' ? 'Uses the record sampling settings below; the default gives patients equal expected training weight.' : 'Uses a slide-level loss objective and slide classes to compose batches; this changes patient exposure. Dataset membership and held-out splits remain fixed.'}</small></label>
      <label className="label">nnMIL evaluation checkpoint<select className="field" value={resolved.nnmilCheckpointSelection} onChange={(event) => onChange({ ...value, nnmilCheckpointSelection: event.target.value as TrainingRecipe['nnmilCheckpointSelection'] })}>
        <option value="best_validation">Best validation checkpoint</option><option value="latest">Latest completed epoch</option>
      </select><small>Latest follows the paper’s stated evaluation choice. Best validation preserves HistoPilot’s model selection protocol. Early stopping still uses the configured validation monitor.</small></label>
      <NumericField label="Feature-window stride divisor" value={resolved.nnmilWindowStrideDivisor!} min={1} max={256} disabled={!resolved.nnmilFeatureSampling} onChange={(nnmilWindowStrideDivisor) => onChange({ ...value, nnmilWindowStrideDivisor })} />
      <label className="development-check"><input type="checkbox" disabled={!resolved.nnmilFeatureSampling} checked={resolved.nnmilWindowShuffle} onChange={(event) => onChange({ ...value, nnmilWindowShuffle: event.target.checked })} />Shuffle feature coordinates before testing windows</label>
      <NumericField label="Feature-window shuffle seed" value={resolved.nnmilWindowSeed!} min={0} max={2 ** 32 - 1} disabled={!resolved.nnmilFeatureSampling || !resolved.nnmilWindowShuffle} onChange={(nnmilWindowSeed) => onChange({ ...value, nnmilWindowSeed })} />
      <label className="label">Feature-window aggregation<select className="field" disabled={!resolved.nnmilFeatureSampling} value={resolved.nnmilWindowAggregation} onChange={(event) => onChange({ ...value, nnmilWindowAggregation: event.target.value as TrainingRecipe['nnmilWindowAggregation'] })}>
        <option value="mean_logits">Average window logits, then convert to probabilities</option><option value="mean_probabilities">Average window probabilities</option>
      </select><small>Applied within each slide and checkpoint during validation, assessment and external testing. Patient and ensemble aggregation are configured separately.</small></label>
    </div>
    <p className="muted">{resolved.nnmilFeatureSampling ? 'The paper uses attention width 256 and stride divisor 4, giving a stride of 64 feature coordinates. Window counts are calculated from the selected feature bundle.' : 'Feature sampling is disabled: training and testing use one view with all feature coordinates. Window stride, shuffle and aggregation settings take effect when feature sampling is enabled.'}</p>
  </section>;
}

export function NnMILPlanning({ rows }: { rows?: NnMILPlanningRow[] }) {
  if (!rows?.length) return null;
  const hasMemoryEstimate = rows.some((row) => row.inputMemoryMiB !== undefined);
  return <details className="setup-details" open><summary>nnMIL fitting-fold plan</summary>
    <p>Patch counts use only each fold’s fitting slides in the selected dataset and feature bundle. Automatic limits round down the configured fraction of the median, with a minimum of one patch. Validation, assessment and external cohorts are excluded.</p>
    <div className="development-table"><table><thead><tr><th scope="col">Configuration / split</th><th scope="col">Fitting patients / slides</th><th scope="col">Median patches</th><th scope="col">Training patch limit</th><th scope="col">Feature dimensions / windows</th><th scope="col">Below / above limit</th><th scope="col">Patch range (5th–95th percentile)</th>{hasMemoryEstimate ? <th scope="col">Input tensors (MiB)</th> : null}</tr></thead>
      <tbody>{rows.map((row) => <tr key={`${row.candidateId}:${row.splitPlanId}`}><th scope="row">{row.candidateId} / {row.splitPlanId}</th><td>{row.trainingPatientCount.toLocaleString()} / {row.trainingSlideCount.toLocaleString()}</td><td>{row.medianPatchCount.toLocaleString()}</td><td>{row.bagSize === null ? 'All patches' : row.bagSize.toLocaleString()}</td><td>{row.featureDimension.toLocaleString()} / {row.windowCount}</td><td>{row.paddedSlides.toLocaleString()} / {row.truncatedSlides.toLocaleString()}</td><td>{row.minPatchCount.toLocaleString()}–{row.maxPatchCount.toLocaleString()} ({row.patchCountP05.toLocaleString()}–{row.patchCountP95.toLocaleString()})</td>{hasMemoryEstimate ? <td>{row.inputMemoryMiB?.toLocaleString(undefined, { maximumFractionDigits: 1 }) ?? '—'}</td> : null}</tr>)}</tbody>
    </table></div>
    <p className="muted">This patch limit is a dataset-based starting point. Batch size and memory settings remain under your control.</p>
    {hasMemoryEstimate ? <p className="muted">Memory estimates cover float32 input tensors per training batch. Activations, model parameters, optimizer state and data loading require additional memory.</p> : null}
  </details>;
}
