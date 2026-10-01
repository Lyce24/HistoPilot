import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { defaultRecipe } from '../api/development';
import type { EvaluationCohort } from '../api/evaluation';
import type { FrozenPredictor, ModelEvaluation, PredictorChoice, RefitBuild } from '../api/predictors';
import type { Workspace } from '../api/types';
import PublicationConfirmation from '../components/PublicationConfirmation';
import LocalPostDevelopment, { RefitTraining } from './LocalPostDevelopment';
import { predictorSourceKey } from '../api/predictorBuilds';
import LocalApplyModels from './LocalApplyModels';

const workspace = { mode: 'local', project: { id: 'project', name: 'Test project', lifecycleState: 'active' } } as Workspace;
const target = { field: 'grade', task: 'binary_classification' as const, unit: 'patient' as const, classes: ['low', 'high'], labels: { low: 'low', high: 'high' }, positiveClass: 'high', missing: 'block' as const, unmapped: 'block' as const };
const recipe = defaultRecipe();
function predictor(id: string, experimentId = `experiment-${id}`, lifecycleState: FrozenPredictor['lifecycleState'] = 'active'): FrozenPredictor {
  return {
    id, contentHash: `hash-${id}`, createdAt: '2026-09-11', lifecycleState,
    manifest: {
      kind: 'frozen-predictor', experimentId, batchId: `batch-${id}`, candidateId: `candidate-${id}`, trainingSeed: 11, splitSeed: 42,
      name: `Predictor ${id}`, recipe, target, runIds: ['fold-1', 'fold-2'],
      checkpoints: [1, 2].map((fold) => ({ runId: `fold-${fold}`, path: `/fixtures/${id}/${fold}.ckpt`, sha256: 'a'.repeat(64), bytes: 123 })),
      aggregation: 'mean_probability', inputs: { protocol: { id: `protocol-${id}`, contentHash: `protocol-hash-${id}` }, features: { bundle: { id: `bundle-${id}`, contentHash: `bundle-hash-${id}` } }, loading: { protocolId: `protocol-${id}`, featureBundleId: `bundle-${id}`, loadingPolicy: 'native', packArtifactId: null } },
    },
  };
}
function choice(experimentId: string, changes: Partial<PredictorChoice> = {}): PredictorChoice {
  return { experimentId, experimentName: experimentId, batchId: `batch-${experimentId}`, batchName: `Batch ${experimentId}`, candidateId: 'candidate', candidateNumber: 1, trainingSeed: 11, splitSeed: 42, completedRuns: 2, totalRuns: 2, eligible: true, reason: null, existingPredictorId: null, recipe, ...changes };
}
function cohort(id: string, source: FrozenPredictor, current = true): EvaluationCohort {
  return {
    id, projectId: 'project', createdAt: '', contentHash: `hash-${id}`, current, findings: [],
    versionLabel: { tag: `Cohort ${id}`, note: '', revision: 1, createdAt: '', updatedAt: '' },
    manifest: {
      kind: 'evaluation-cohort', datasetId: 'test-data', target, findings: [],
      spec: { protocolId: source.manifest.inputs.protocol.id, developmentFeatureBundleId: source.manifest.inputs.features.bundle.id, datasetId: 'test-data', featureBundleId: 'test-features', target, eligibility: [], patientIdentifiers: 'shared', inference: { loadingPolicy: 'per_slide', packArtifactId: null, batchSize: 1, numWorkers: 0, device: 'cpu', precision: 'float32', patientAggregation: 'mean', decisionThreshold: 0.5 } },
      summary: { includedSlides: 20, includedPatients: 20, excludedSlides: 0, labeledSlides: 20, classCounts: { low: 10, high: 10 }, developmentSlideOverlap: 0, developmentPatientOverlap: 0 },
    },
  };
}
function evaluation(id: string, source: FrozenPredictor, test: EvaluationCohort, lifecycleState: ModelEvaluation['lifecycleState'] = 'active'): ModelEvaluation {
  return { id, createdAt: '', contentHash: `hash-${id}`, lifecycleState, manifest: { kind: 'model-evaluation', name: `Evaluation ${id}`, predictorId: source.id, experimentId: source.manifest.experimentId, cohortId: test.id, status: 'planned', results: null, executionEnabled: false } };
}

const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.unstubAllGlobals(); });
function render(module: 'freeze' | 'apply', options: { predictors?: FrozenPredictor[]; choices?: PredictorChoice[]; cohorts?: EvaluationCohort[]; evaluations?: ModelEvaluation[]; hash?: string } = {}) {
  vi.stubGlobal('window', { location: { hash: options.hash ?? '' } });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['predictors', 'project'], { items: options.predictors ?? [], executionEnabled: false });
  client.setQueryData(['predictor-choices', 'project'], { items: options.choices ?? [], executionEnabled: false });
  client.setQueryData(['evaluation-cohorts', 'project'], { items: options.cohorts ?? [] });
  client.setQueryData(['feature-bundles', 'project'], { items: [] });
  client.setQueryData(['model-evaluations', 'project'], { items: options.evaluations ?? [], executionEnabled: false });
  client.setQueryData(['evaluation-batches', 'project'], { items: [] });
  client.setQueryData(['model-experiment-summaries', 'project'], { items: [] });
  client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
  return renderToStaticMarkup(<QueryClientProvider client={client}>{module === 'freeze' ? <LocalPostDevelopment workspace={workspace} /> : <LocalApplyModels workspace={workspace} />}</QueryClientProvider>);
}

describe('model development predictor and Apply models chains', () => {
  it('opens the exact linked predictor in its library, including archived evidence', () => {
    const html = render('freeze', {
      predictors: [predictor('linked', 'experiment-linked', 'archived'), predictor('other')],
      hash: '#post-development?predictor=linked',
    });
    expect(html).toContain('Showing the linked predictor.');
    expect(html).toContain('Predictor linked');
    expect(html).not.toContain('Predictor other');
    expect(html).toContain('#interpretation?experiment=experiment-linked&amp;predictor=linked');
    expect(html).toContain('Interpret slides');
  });
  it('applies a linked predictor to a linked cohort and offers every cohort regardless of its original development bindings', () => {
    const source = predictor('linked');
    const selected = cohort('selected', source);
    const other = cohort('other', predictor('different'));
    const html = render('apply', { predictors: [source], cohorts: [selected, other], hash: '#apply?view=new&predictor=linked&cohort=selected' });
    // A linked predictor opens on its methods and cohort, selected alone with its own method.
    expect(html).toContain('2. Choose methods and cohort');
    expect(html).toMatch(/<input(?=[^>]*value="ensemble")(?=[^>]*checked="")[^>]*>/);
    expect(html).toMatch(/aria-label="Apply predictor Predictor linked \(linked\)"[^>]*checked=""/);
    expect(html).toContain('1 predictor to apply from 1 selected experiment');
    expect(html).toContain('<option value="selected" selected="">Cohort selected · Labeled · scored · 20 slides</option>');
    expect(html).toContain('<option value="other">Cohort other · Labeled · scored · 20 slides</option>');
    expect(html).toContain('Predicted without reading labels, then scored against the cohort’s labels.');
    const otherBinding = render('apply', { predictors: [source], cohorts: [selected, other], hash: '#apply?view=new&predictor=linked&cohort=other' });
    expect(otherBinding).toContain('<option value="other" selected="">');
    expect(otherBinding).toContain('Review matches the prediction task and class encoding');
    expect(otherBinding).toContain('No frozen feature bundles are available');
    expect(otherBinding).toContain('The cohort is already saved');
    expect(otherBinding).not.toContain('unavailable or incompatible');
  });
  it('distinguishes another published refit plan and offers recovery for a trashed predictor', () => {
    const ready = predictor('published-refit', 'experiment-refit', 'trashed');
    ready.manifest.method = 'refit';
    ready.manifest.refitId = 'original-plan';
    const build: RefitBuild = { ...ready, id: 'another-plan', lifecycleState: 'active', manifest: { ...ready.manifest, kind: 'predictor-refit' }, execution: { status: 'not_started' } };
    const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } });
    clients.push(client);
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><RefitTraining project="project" build={build} existing={ready} refresh={async () => {}} /></QueryClientProvider>);
    expect(html).toContain('already have a refit predictor from another plan');
    expect(html).toContain('Restore Predictor published-refit from Trash');
    expect(html).not.toContain('predictor=published-refit');
    expect(html).not.toContain('Train refit model');
    expect(html).not.toContain('Restore this record to run it.');
    expect(html).not.toContain('Publish refit predictor');
  });
  it('blocks incomplete folds and allows completed seeds to review reusable outputs', () => {
    const html = render('freeze', { choices: [choice('partial', { completedRuns: 1, eligible: false, reason: 'Every fold must finish.' }), choice('promoted', { eligible: false, existingPredictorId: 'predictor-one', reason: 'predictor already exists' })] });
    expect(html).toMatch(/<input[^>]*aria-label="Select partial[^>]*disabled=""/);
    expect(html).toMatch(/<input[^>]*aria-label="Select promoted[^>]*>/);
    expect(html).not.toMatch(/<input[^>]*aria-label="Select promoted[^>]*disabled=""/);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review predictors/);
    expect(html).toContain('Every fold must finish.');
    expect(html).toContain('Existing outputs are reused');
    expect(html).toMatch(/<input(?=[^>]*value="both")(?=[^>]*checked="")[^>]*>/);
    expect(html).toContain('Select all completed seeds');
  });

  it('keeps each completed training and split seed group separate within one experiment', () => {
    const options = [choice('same'), choice('same', { trainingSeed: 2024 }), choice('same', { splitSeed: 17 })];
    const html = render('freeze', { choices: options });
    expect(new Set(options.map(predictorSourceKey)).size).toBe(3);
    expect(html).toContain('configuration 1 training seed 2024 split seed 42');
    expect(html).toContain('configuration 1 training seed 11 split seed 17');
    expect(html).toContain('3 seed groups');
    expect(html).toMatch(/<input(?=[^>]*value="both")(?=[^>]*checked="")[^>]*>/);
  });

  it('displays independent predictors, keeps inactive records hidden, and exposes lifecycle recovery views', () => {
    const first = predictor('one'), second = predictor('two');
    const html = render('freeze', { hash: '#post-development?tab=library', predictors: [first, second, predictor('archived-name', undefined, 'archived'), predictor('deleted-name', undefined, 'trashed')] });
    expect(html).toContain('Predictor one');
    expect(html).toContain('Predictor two');
    expect(html).toContain('#experiments?experiment=experiment-one');
    expect(html).toContain('#experiments?experiment=experiment-two');
    expect(html).toContain('#apply?view=new&amp;predictor=one');
    expect(html).toContain('#apply?view=new&amp;predictor=two');
    expect(html).not.toContain('Predictor archived-name');
    expect(html).not.toContain('Predictor deleted-name');
    expect(html).toContain('value="archived"');
    expect(html).toContain('value="trashed"');
    expect(html).toContain('All records');
    expect(html).toContain('#cleanup?key=configuration%3Aone');
  });

  it('keeps a historical experiment link scoped to its exact batch instead of selecting another chain', () => {
    const legacyId = 'legacy-configuration-original';
    const html = render('freeze', { hash: `#post-development?experiment=${legacyId}&tab=library`, predictors: [predictor('legacy', legacyId), predictor('other')], choices: [choice(legacyId), choice('experiment-other')] });
    expect(html).toContain('Predictor legacy');
    expect(html).toContain(`#experiments?experiment=${legacyId}`);
    expect(html).not.toContain('Predictor other');
    expect(html).not.toContain('Batch experiment-other');
    expect(html).toContain('Show all experiments and predictors');
    expect(predictorSourceKey(choice(legacyId))).not.toBe(predictorSourceKey(choice(legacyId, { trainingSeed: 12 })));
  });

  it('lists saved runs by cohort, attached to both predictor chains, without inventing results', () => {
    const first = predictor('one'), second = predictor('two');
    const one = cohort('one', first), two = cohort('two', second);
    const html = render('apply', { predictors: [first, second], cohorts: [one, two], evaluations: [evaluation('first', first, one), evaluation('second', second, two), evaluation('archived-plan', first, one, 'archived'), evaluation('deleted-plan', first, one, 'trashed')] });
    expect(html).toContain('Evaluation first');
    expect(html).toContain('Evaluation second');
    expect(html).toContain('#experiments?experiment=experiment-one&amp;tab=predictors&amp;predictor=one');
    expect(html).toContain('#experiments?experiment=experiment-two&amp;tab=predictors&amp;predictor=two');
    expect(html).toContain('Cohort one · Labeled · scored');
    expect(html).toContain('Cohort two · Labeled · scored');
    expect(html).not.toContain('Evaluation archived-plan');
    expect(html).not.toContain('Evaluation deleted-plan');
    expect(html.match(/>Ready to run<\/span>/g)).toHaveLength(2);
    expect(html).toContain('Apply predictors');
    expect(html).toContain('Search runs');
    expect(html).toContain('Predictor method');
    expect(html).toContain('Sort runs');
    expect(html).toContain('Manage');
    expect(html).not.toContain('1. Select experiments');
    expect(html).not.toContain('>Run reviewed predictors');
  });

  it('scopes the runs to a linked predictor, and offers every frozen cohort when applying it', () => {
    const first = predictor('one'), second = predictor('two');
    const one = cohort('one', first), two = cohort('two', second), stale = cohort('stale', first, false);
    const runs = render('apply', { hash: '#apply?view=runs&predictor=one', predictors: [first, second], cohorts: [one, two, stale], evaluations: [evaluation('first', first, one), evaluation('second', second, two)] });
    expect(runs).toContain('Evaluation first');
    expect(runs).not.toContain('Evaluation second');
    expect(runs).toContain('Show all runs');
    const setup = render('apply', { hash: '#apply?view=new&predictor=one', predictors: [first, second], cohorts: [one, two, stale] });
    expect(setup).toContain('Cohort one');
    expect(setup).toContain('Cohort two');
    expect(setup).toMatch(/<option[^>]*value="stale"[^>]*>Cohort stale[^<]*needs verification/);
    expect(setup).not.toMatch(/<option[^>]*value="stale"[^>]*disabled/);
    expect(setup).toContain('Back to runs');
    expect(setup).not.toContain('Search runs');
  });

  it('does not substitute an active predictor when a link points to deleted or archived weights', () => {
    const deleted = predictor('deleted', undefined, 'trashed'), archived = predictor('archived', undefined, 'archived'), active = predictor('active');
    for (const id of ['deleted', 'archived']) {
      const html = render('apply', { hash: `#apply?view=new&predictor=${id}`, predictors: [deleted, archived, active], cohorts: [cohort('active', active)] });
      expect(html).toContain('The linked predictor is archived, in Trash or unavailable');
      expect(html).toContain('0 predictors to apply from 0 selected experiments');
      expect(html).not.toMatch(/aria-label="Apply predictor Predictor active/);
    }
  });

  it('offers only active predictors from a linked experiment', () => {
    const active = predictor('active'), archived = predictor('archived', 'experiment-active', 'archived');
    const html = render('apply', { hash: '#apply?view=new&experiment=experiment-active', predictors: [active, archived] });
    expect(html).toContain('1 ensemble / 0 refit ready');
    expect(html).toContain('1 predictor to apply from 1 selected experiment');
    expect(html).not.toContain('Predictor archived');
  });

  it('requires acknowledgement and clearly separates uncertain-save retry from a fresh publication', () => {
    const props = { busy: false, uncertain: false, acknowledged: false, onAcknowledge: () => {}, onConfirm: () => {}, label: 'Freeze predictor' };
    const initial = renderToStaticMarkup(<PublicationConfirmation {...props} />);
    expect(initial).toMatch(/<button[^>]*disabled=""[^>]*>Freeze predictor/);
    const uncertain = renderToStaticMarkup(<PublicationConfirmation {...props} uncertain />);
    expect(uncertain).toContain('record may already exist');
    expect(uncertain).toContain('identical reviewed request');
    expect(uncertain).toContain('Retry this save');
    expect(uncertain.match(/<button/g)).toHaveLength(1);
    expect(uncertain).not.toMatch(/Back to|Return to/);
    expect(uncertain).toContain('Resolve this save before changing its inputs');
    expect(uncertain).not.toContain('>Freeze predictor</button>');
  });
});
