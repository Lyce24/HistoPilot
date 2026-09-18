import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { defaultRecipe, defaultResources, nnmilRecipe } from '../api/development';
import DevelopmentBatches from '../components/DevelopmentBatches';
import type { ModelExperiment } from '../api/experiments';
import { isBatchEditorDraft, type BatchEditorDraft } from './batchEditorDraft';
import { sessionDraftKey } from './sessionDraft';

const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native' as const, packArtifactId: null };
const draft = (): BatchEditorDraft => ({
  version: 1, editorRevision: 1, inputs, workingPlan: null, name: 'Unfinished recipe', editorOpen: true, batchPage: 2,
  templateId: 'blank', predictorPolicy: { method: 'ensemble', refitPercentile: null }, recipe: defaultRecipe(), resources: defaultResources(),
  mode: 'single', rows: [{ id: 0, recipe: defaultRecipe() }], explicitInitialized: false,
  seeds: '42,', lrs: '1e-', wds: '0', epochs: '40', gpus: '', notes: 'Keep this question',
  numericDrafts: { 'shared:Learning rate': { source: defaultRecipe().learningRate, text: '1e-' } },
});
afterEach(() => vi.unstubAllGlobals());

describe('batch editor recovery', () => {
  it('retains raw unfinished numeric/list entries without turning them into model values', () => {
    expect(isBatchEditorDraft(draft())).toBe(true);
    expect(isBatchEditorDraft({ ...draft(), recipe: { ...draft().recipe, bagSize: null } })).toBe(true);
    expect(draft().recipe.learningRate).toBe(0.0003);
  });

  it.each([
    { version: 2 }, { inputs: null }, { resources: { gpuIds: '0' } },
    { rows: [{ id: 0, recipe: defaultRecipe() }, { id: 0, recipe: defaultRecipe() }] },
    { recipe: { ...defaultRecipe(), maxEpochs: '40' } },
    { numericDrafts: { field: { source: null, text: '1e-' } } },
    { batchPage: 7 }, { name: 'x'.repeat(81) },
    { recipe: { ...defaultRecipe(), classWeights: '1,2' } },
    { recipe: { ...defaultRecipe(), classWeights: [1, null] } },
    { recipe: { ...defaultRecipe(), adamBetas: [0.9] } },
    { recipe: { ...defaultRecipe(), samplingStrategy: 'unknown' } },
    { recipe: { ...defaultRecipe(), evalBagSize: 'all' } },
    { recipe: { ...nnmilRecipe(), bagSizeMode: 'auto_test_median' } },
    { recipe: { ...nnmilRecipe(), nnmilWindowAggregation: 'average' } },
    { recipe: { ...nnmilRecipe(), nnmilFeatureSampling: 'yes' } },
    { recipe: { ...nnmilRecipe(), nnmilCheckpointSelection: 'assessment_best' } },
    { recipe: { ...nnmilRecipe(), lrScheduleInterval: 'batch_or_epoch' } },
  ])('rejects malformed recovery state: %j', (patch) => {
    expect(isBatchEditorDraft({ ...draft(), ...patch })).toBe(false);
  });

  it('retains optional experimental controls while accepting older copies without them', () => {
    expect(isBatchEditorDraft(draft())).toBe(true);
    expect(isBatchEditorDraft({ ...draft(), recipe: { ...draft().recipe, classWeights: [1, 2], adamBetas: [0.8, 0.99], evalBagSize: null, lossType: 'bce', minValidationPositives: 5, fixedEpochBudget: 20 } })).toBe(true);
  });

  it('retains every customized nnMIL parameter in recovered shared and explicit recipes', () => {
    const recipe = { ...nnmilRecipe(), attentionDim: 128, bagSizeFraction: 0.75, nnmilWindowSeed: 123,
      nnmilWindowAggregation: 'mean_probabilities' as const, nnmilFeatureSampling: false,
      nnmilBatchSampler: 'auc_stratified' as const, nnmilCheckpointSelection: 'latest' as const,
      weightDecayPolicy: 'weights_only' as const, lrScheduleInterval: 'step' as const };
    const value = { ...draft(), templateId: 'nnmil', recipe, rows: [{ id: 0, recipe }] };
    expect(isBatchEditorDraft(JSON.parse(JSON.stringify(value)))).toBe(true);
  });

  it('restores raw fields in the correct editor and preserves its stale baseline', () => {
    const key = sessionDraftKey('project', 'experiment', 'batch-editor');
    vi.stubGlobal('window', { sessionStorage: { getItem: (requested: string) => requested === key ? JSON.stringify({ version: 1, value: draft() }) : null } });
    const client = new QueryClient();
    const record = { id: 'experiment', revision: 2, batchPlans: [], state: 'active' } as unknown as ModelExperiment;
    const render = (project: string) => renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project={project} experimentId="experiment" experimentName="Study" experimentRevision={2} inputs={inputs} record={record} experimentStage="planning" ownedBatches={[]} ownedDrafts={[]} tab="batches" onOpenSetup={() => {}} /></QueryClientProvider>);
    try {
      const html = render('project');
      expect(html).toContain('Recovered unsaved batch edits');
      expect(html).toContain('Keep settings as a new batch');
      expect(html).toContain('value="1e-"');
      expect(html).toContain('value="42,"');
      expect(html).toContain('Keep this question');
      expect(html).toMatch(/<fieldset[^>]*disabled=""[^>]*class="development-editor"/);
      expect(render('different-project')).not.toContain('Unfinished recipe');
    } finally { client.clear(); }
  });

  it('does not crash when an unfinished recovered grid is on the review step', () => {
    const value = { ...draft(), mode: 'grid', batchPage: 4 };
    vi.stubGlobal('window', { sessionStorage: { getItem: () => JSON.stringify({ version: 1, value }) } });
    const client = new QueryClient();
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="project" experimentId="experiment" experimentName="Study" experimentRevision={1} inputs={inputs} experimentStage="planning" ownedBatches={[]} ownedDrafts={[]} tab="batches" onOpenSetup={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Some recovered parameter values are unfinished');
    } finally { client.clear(); }
  });
});
