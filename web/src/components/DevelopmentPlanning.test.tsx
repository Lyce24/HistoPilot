import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DevelopmentBatches, { BatchPlanSettings, BatchPredictorSummary, ExperimentBatchOverview, batchConfigurationCount, batchTemplate, updateBatchPlans } from './DevelopmentBatches';
import { defaultRecipe } from '../api/development';
import type { ExperimentBatch, ModelExperiment } from '../api/experiments';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const plans = [{ id: 'baseline', spec: batchTemplate('baseline', inputs, 'Study') }, { id: 'comparison', spec: batchTemplate('learning-rate', inputs, 'Study') }];
const experiment: ModelExperiment = { id: 'study', key: 'draft:study', name: 'Study', notes: '', tags: [], revision: 2, state: 'active', stage: 'planning', status: 'planned', legacy: false, createdAt: '', updatedAt: '', inputs, batches: [], drafts: [], batchPlans: plans, predictorId: null };
function render(record = experiment, tab: 'batches' | 'runs' | 'results' = 'batches') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" record={record} experimentStage={record.stage} inputs={inputs} experimentName={record.name} experimentId={record.id} experimentRevision={record.revision} ownedBatches={record.batches} ownedDrafts={[]} tab={tab} onOpenSetup={() => {}} /></QueryClientProvider>);
  } finally { client.clear(); }
}

describe('editable experiment batch plans', () => {
  it('keeps adding a plan separate from submission and offers editable templates', () => {
    const html = render();
    expect(html).toContain('Add batch to plan');
    expect(html).toContain('Start from a template');
    expect(html).toContain('Quick check');
    expect(html).toContain('Start blank');
    expect(html).toContain('Configure predictors');
    expect(html).not.toContain('Save predictor choices');
    expect(html).toContain('Batch plans (2)');
    expect(html).toContain('Edit batch');
    expect(html).not.toContain('Launch batch');
    expect(html).not.toContain('Save draft');
    expect(html).not.toContain('Clone batch');
  });

  it('organizes search, training and compute with accessible mode choices and no invented fold count', () => {
    const html = render();
    expect(html).toContain('<legend>Configuration mode</legend>');
    expect(html).toMatch(/<input(?=[^>]*value="single")(?=[^>]*checked="")[^>]*>/);
    expect(html).toContain('Custom configurations');
    expect(html).toContain('Parameter search &amp; repeats');
    expect(html).toContain('Compute &amp; parallelism');
    expect(html).toContain('1 configuration × 1 training seed = 1 training group');
    expect(html).toContain('Check batch to confirm the total fold runs.');
    expect(html).not.toContain('5 planned runs');
  });

  it('counts a fifteen-configuration grid with three training seeds before multiplying by folds', () => {
    const spec = { ...plans[1].spec, trainingSeeds: [42, 43, 44], grid: { learningRates: [0.0001, 0.0002, 0.0003, 0.0004, 0.0005], weightDecays: [0, 0.0001, 0.001], maxEpochs: [100] } };
    expect(batchConfigurationCount(spec)).toBe(15);
    const html = renderToStaticMarkup(<BatchPlanSettings spec={spec} />);
    expect(html).toContain('Parameter grid · 15 configurations');
    expect(html).toContain('42, 43, 44');
    expect(html).toContain('ABMIL · Shared training settings');
    expect(html).toContain('Highest validation AUROC');
    expect(html).toContain('Patience 8');
    expect(html).not.toContain('225 predictors');
  });

  it('matches backend deduplication for explicit recipes including omitted model defaults', () => {
    const recipe = defaultRecipe();
    const { embedDim: _embedding, attentionDim: _attention, ...legacy } = recipe;
    expect(batchConfigurationCount({ ...plans[0].spec, mode: 'explicit', configurations: [recipe, legacy, { ...recipe, learningRate: 0.001 }] })).toBe(2);
  });

  it('separates each batch into configuration, training, compute and review pages while retaining form state', () => {
    const html = render();
    const editor = html.slice(html.indexOf('Add a training batch'));
    const sections = ['Start from a template', '>Batch name<', 'Parameter search &amp; repeats', 'aria-label="Settings"', 'Compute &amp; parallelism', 'Configure predictors', '>Add batch to plan<'];
    const positions = sections.map((label) => editor.indexOf(label));
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((left, right) => left - right));
    expect(editor).not.toContain('Save predictor choices');
    expect(html).toContain('aria-label="Batch configuration steps"');
    expect(html).toContain('data-batch-step="1"');
    expect(html).toContain('data-batch-step="2" hidden=""');
    expect(html).toContain('data-batch-step="3" hidden=""');
    expect(html).toContain('Continue to training settings');
  });

  it('keeps different saved batch predictor choices separate and visible when locked', () => {
    const record: ModelExperiment = { ...experiment, stage: 'running', configurationLocked: true, batchPlans: [
      { ...plans[0], spec: { ...plans[0].spec, predictorPolicy: { method: 'skip', refitPercentile: null } } },
      { ...plans[1], spec: { ...plans[1].spec, predictorPolicy: { method: 'both', refitPercentile: 75 } } },
    ] };
    const html = render(record);
    expect(html).toContain('<dt>Predictors</dt><dd>Skip</dd>');
    expect(html).toContain('<dt>Predictors</dt><dd>Both · P75 refit epochs</dd>');
    expect(html).toContain('Predictors: Skip');
    expect(html).toContain('Predictors: Both · P75 refit epochs');
    expect(html).not.toContain('Configure predictors');
    expect(html).not.toContain('Save predictor choices');
  });

  it('shows a compact batch predictor policy and count without expanding its settings', () => {
    const spec = { ...plans[0].spec, predictorPolicy: { method: 'both' as const, refitPercentile: 75 } };
    const html = renderToStaticMarkup(<BatchPredictorSummary spec={spec} count={90} />);
    expect(html).toContain('Predictors: Both · P75 refit epochs');
    expect(html).toContain('90 predictors planned');
    expect(html).not.toContain('<details');
  });

  it('protects configuration actions in running and finished experiments even without a readOnly prop', () => {
    for (const stage of ['running', 'finished'] as const) {
      const html = render({ ...experiment, stage, configurationLocked: true });
      expect(html).toContain('View settings');
      expect(html).toContain('Locked');
      for (const label of ['Add batch to plan', 'Edit batch', 'Duplicate', 'Remove', 'Start from a template', 'Launch batch']) expect(html).not.toContain(`>${label}<`);
    }
  });

  it('blocks planning runs and results for direct component navigation', () => {
    for (const tab of ['runs', 'results'] as const) expect(render(experiment, tab)).toContain('Runs unlock after submission. Results unlock when the experiment finishes.');
  });

  it('edits one recipe without changing sibling plans and preserves the ID across uncertain saves', () => {
    const changed = { ...plans[0], spec: { ...plans[0].spec, batchName: 'Adjusted', recipe: { ...plans[0].spec.recipe, learningRate: 0.001 } } };
    const once = updateBatchPlans(plans, changed);
    expect(updateBatchPlans(once, changed)).toEqual(once);
    expect(once[1]).toBe(plans[1]);
    expect(plans[0].spec.recipe.learningRate).toBe(0.0003);
    expect(once[0].spec.recipe.learningRate).toBe(0.001);
  });

  it('uses identical inputs and seeds for a learning-rate comparison and bounded quick-check epochs', () => {
    const comparison = batchTemplate('learning-rate', inputs, 'Study');
    expect(comparison.grid.learningRates).toHaveLength(3);
    expect(comparison.inputs).toEqual(inputs);
    expect(comparison.trainingSeeds).toEqual([42]);
    expect(comparison.recipe.maxEpochs).toBe(40);
    expect(comparison.recipe.patience).toBe(8);
    expect(comparison.predictorPolicy).toEqual({ method: 'ensemble', refitPercentile: null });
    const quick = batchTemplate('quick', inputs, 'Study');
    expect(quick.recipe.maxEpochs).toBe(5);
    expect(quick.recipe.bagSize).toBe(1024);
    expect(quick.resources.maxConcurrentRuns).toBe(1);
  });

  it('shows incomplete and unavailable batches without inventing completion', () => {
    const batches = [
      { id: 'a', status: 'running', manifest: { spec: { batchName: 'Baseline' }, summary: { runCount: 5 } }, execution: { runCounts: { completed: 2, failed: 1 } } },
      { id: 'b', status: 'unknown', manifest: { spec: { batchName: 'Comparison' }, summary: { runCount: 5 } } },
    ] as ExperimentBatch[];
    const html = renderToStaticMarkup(<ExperimentBatchOverview batches={batches} view="runs" onSelect={() => {}} />);
    expect(html).toContain('2 of 10 fold runs completed');
    expect(html).toContain('Status unavailable');
    expect(html).toContain('1 failed');
    expect(html.match(/<progress /g)).toHaveLength(1);
  });
});
