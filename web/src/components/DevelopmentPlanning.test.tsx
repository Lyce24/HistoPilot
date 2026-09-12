import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DevelopmentBatches, { BatchPlanSettings, ExperimentBatchOverview, batchConfigurationCount, batchTemplate, updateBatchPlans } from './DevelopmentBatches';
import { defaultRecipe } from '../api/development';
import type { ExperimentBatch, ModelExperiment } from '../api/experiments';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const plans = [{ id: 'baseline', spec: batchTemplate('baseline', inputs, 'Study') }, { id: 'comparison', spec: batchTemplate('learning-rate', inputs, 'Study') }];
const experiment: ModelExperiment = { id: 'study', key: 'draft:study', name: 'Study', notes: '', tags: [], revision: 2, state: 'active', stage: 'planning', status: 'planned', legacy: false, createdAt: '', updatedAt: '', inputs, batches: [], drafts: [], batchPlans: plans, predictorId: null };
function render(record = experiment, tab: 'batches' | 'runs' | 'results' = 'batches') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" record={record} experimentStage={record.stage} inputs={inputs} experimentName={record.name} experimentId={record.id} experimentRevision={record.revision} ownedBatches={record.batches} ownedDrafts={[]} tab={tab} onOpenSetup={() => {}} onRestoreInputs={() => {}} /></QueryClientProvider>);
  } finally { client.clear(); }
}

describe('editable experiment batch plans', () => {
  it('keeps adding a plan separate from submission and offers editable templates', () => {
    const html = render();
    expect(html).toContain('Add batch to plan');
    expect(html).toContain('Start from a template');
    expect(html).toContain('Quick check');
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
    expect(html).toContain('Lowest validation loss');
    expect(html).toContain('Patience 15');
    expect(html).not.toContain('225 predictors');
  });

  it('matches backend deduplication for explicit recipes including omitted model defaults', () => {
    const recipe = defaultRecipe();
    const { embedDim: _embedding, attentionDim: _attention, ...legacy } = recipe;
    expect(batchConfigurationCount({ ...plans[0].spec, mode: 'explicit', configurations: [recipe, legacy, { ...recipe, learningRate: 0.001 }] })).toBe(2);
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
