import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { defaultRecipe } from '../api/development';
import type { EvaluationCohort } from '../api/evaluation';
import type { FrozenPredictor, ModelEvaluation, PredictorChoice, RefitBuild } from '../api/predictors';
import type { Workspace } from '../api/types';
import PublicationConfirmation from '../components/PublicationConfirmation';
import LocalPostDevelopment, { predictorChoiceKey, RefitTraining } from './LocalPostDevelopment';
import LocalModelEvaluation from './LocalModelEvaluation';

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
function render(module: 'freeze' | 'evaluate', options: { predictors?: FrozenPredictor[]; choices?: PredictorChoice[]; cohorts?: EvaluationCohort[]; evaluations?: ModelEvaluation[]; hash?: string } = {}) {
  vi.stubGlobal('window', { location: { hash: options.hash ?? '' } });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['predictors', 'project'], { items: options.predictors ?? [], executionEnabled: false });
  client.setQueryData(['predictor-choices', 'project'], { items: options.choices ?? [], executionEnabled: false });
  client.setQueryData(['evaluation-cohorts', 'project'], { items: options.cohorts ?? [] });
  client.setQueryData(['model-evaluations', 'project'], { items: options.evaluations ?? [], executionEnabled: false });
  return renderToStaticMarkup(<QueryClientProvider client={client}>{module === 'freeze' ? <LocalPostDevelopment workspace={workspace} /> : <LocalModelEvaluation workspace={workspace} />}</QueryClientProvider>);
}

describe('model development predictor and evaluation chains', () => {
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
  it('selects both linked evaluation inputs and keeps incompatible cohorts unavailable', () => {
    const source = predictor('linked');
    const selected = cohort('selected', source);
    const other = cohort('other', predictor('different'));
    const html = render('evaluate', { predictors: [source], cohorts: [selected, other], hash: '#evaluation?predictor=linked&cohort=selected' });
    expect(html).toContain('<option value="linked" selected="">');
    expect(html).toContain('<option value="selected" selected="">');
    expect(html).not.toContain('<option value="other"');
    const incompatible = render('evaluate', { predictors: [source], cohorts: [selected, other], hash: '#evaluation?predictor=linked&cohort=other' });
    expect(incompatible).toContain('<option value="other" disabled="" selected="">Selected test cohort unavailable or incompatible');
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
    expect(html).not.toContain('#evaluation?predictor=published-refit');
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
    expect(new Set(options.map(predictorChoiceKey)).size).toBe(3);
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
    expect(html).toContain('#evaluation?predictor=one');
    expect(html).toContain('#evaluation?predictor=two');
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
    expect(predictorChoiceKey(choice(legacyId))).not.toBe(predictorChoiceKey(choice(legacyId, { trainingSeed: 12 })));
  });

  it('keeps saved evaluation plans attached to both predictor chains without inventing results', () => {
    const first = predictor('one'), second = predictor('two');
    const one = cohort('one', first), two = cohort('two', second);
    const html = render('evaluate', { predictors: [first, second], cohorts: [one, two], evaluations: [evaluation('first', first, one), evaluation('second', second, two), evaluation('archived-plan', first, one, 'archived'), evaluation('deleted-plan', first, one, 'trashed')] });
    expect(html).toContain('Evaluation first');
    expect(html).toContain('Evaluation second');
    expect(html).toContain('Predictor one');
    expect(html).toContain('Predictor two');
    expect(html).toContain('Cohort one');
    expect(html).toContain('Cohort two');
    expect(html).not.toContain('Evaluation archived-plan');
    expect(html).not.toContain('Evaluation deleted-plan');
    expect(html.match(/>Ready to run<\/span>/g)).toHaveLength(2);
    expect(html).toContain('Accuracy');
    expect(html).toContain('AUROC');
    expect(html).toContain('Choose predictors');
    expect(html).not.toContain('>Launch evaluation');
    expect(html).not.toContain('>Run inference');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review all shown predictors/);
  });

  it('deep-links to the requested predictor and restricts cohorts by protocol and development features', () => {
    const first = predictor('one'), second = predictor('two');
    const one = cohort('one', first), two = cohort('two', second), stale = cohort('stale', first, false);
    const html = render('evaluate', { hash: '#evaluation?predictor=one', predictors: [first, second], cohorts: [one, two, stale], evaluations: [evaluation('first', first, one), evaluation('second', second, two)] });
    expect(html).toContain('Cohort one');
    expect(html).not.toContain('Cohort two');
    expect(html).toMatch(/<option[^>]*disabled=""[^>]*>Cohort stale[^<]*needs verification/);
    expect(html).toContain('Evaluation first');
    expect(html).not.toContain('Evaluation second');
    expect(html).toContain('Show all predictors');
  });

  it('does not substitute an active predictor when a link points to deleted weights', () => {
    const deleted = predictor('deleted', undefined, 'trashed'), active = predictor('active');
    const html = render('evaluate', { hash: '#evaluation?predictor=deleted', predictors: [deleted, active], cohorts: [cohort('active', active)] });
    expect(html).toContain('<option value="deleted" disabled="" selected="">Linked predictor unavailable');
    expect(html).toContain('Predictors appear automatically');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review evaluation/);
    expect(html).not.toContain('class target grade');
  });

  it('hides archived predictors from new choices while preserving an explicit retained-chain link', () => {
    const active = predictor('active'), archived = predictor('archived', undefined, 'archived');
    const defaultView = render('evaluate', { predictors: [active, archived] });
    expect(defaultView).toContain('<option value="active"');
    expect(defaultView).not.toMatch(/<option[^>]*value="archived"[^>]*>Predictor archived/);
    const linked = render('evaluate', { hash: '#evaluation?predictor=archived', predictors: [active, archived] });
    expect(linked).toMatch(/<option[^>]*value="archived"[^>]*>Configuration candidate-archived · Train 11 \/ split 42 · Fold ensemble · archived/);
    expect(linked).toContain('#experiments?experiment=experiment-archived');
    expect(linked).toContain('class target grade');
  });

  it('requires acknowledgement and clearly separates uncertain-save retry from a fresh publication', () => {
    const props = { busy: false, uncertain: false, acknowledged: false, onAcknowledge: () => {}, onConfirm: () => {}, onReset: () => {}, label: 'Freeze predictor' };
    const initial = renderToStaticMarkup(<PublicationConfirmation {...props} />);
    expect(initial).toMatch(/<button[^>]*disabled=""[^>]*>Freeze predictor/);
    const uncertain = renderToStaticMarkup(<PublicationConfirmation {...props} uncertain />);
    expect(uncertain).toContain('record may already exist');
    expect(uncertain).toContain('identical reviewed request');
    expect(uncertain).toContain('Retry this save');
    expect(uncertain).not.toContain('>Freeze predictor</button>');
  });
});
