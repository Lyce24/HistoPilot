import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { ClinicalAnalysis, ClinicalReport } from '../api/clinicalUtility';
import type { EvaluationMetrics, FrozenPredictor, ModelEvaluation } from '../api/predictors';
import { defaultRecipe } from '../api/development';
import { ClinicalReportView, RunClinicalUtility, clinicalThresholdNote } from '../components/ClinicalUtility';
import LocalInterpretation from './LocalInterpretation';
import CurveChart from '../components/CurveChart';
const workspace = { mode: 'local', project: { id: 'p', name: 'Evidence project' } } as Workspace;
const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.unstubAllGlobals(); });
const newClient = () => { const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } }); clients.push(client); return client; };
function render(options: { hash?: string; predictors?: FrozenPredictor[] } = {}) {
  vi.stubGlobal('window', { location: { hash: options.hash ?? '' } });
  const client = newClient();
  client.setQueryData(['interpretations', 'p'], { items: [] });
  client.setQueryData(['model-evaluations', 'p'], { items: [] });
  client.setQueryData(['predictors', 'p'], { items: options.predictors ?? [] });
  return renderToStaticMarkup(<QueryClientProvider client={client}><LocalInterpretation workspace={workspace} /></QueryClientProvider>);
}

const point = { threshold: .5, tp: 3, fp: 0, tn: 0, fn: 0, sensitivity: 1, specificity: null, ppv: 1, npv: null, accuracy: 1, balancedAccuracy: null, f1: 1, positiveLikelihoodRatio: null, negativeLikelihoodRatio: null, predictedPositive: 3, predictedNegative: 0, netBenefit: 1, treatAllNetBenefit: 1, treatNoneNetBenefit: 0, standardizedNetBenefit: 1, netInterventionsAvoidedPer100: 0, highRiskPer100: 100, truePositivePer100: 100, falsePositivePer100: 0, missedPositivePer100: 0 };
const report: ClinicalReport = { unit: 'patient', frozenUnit: 'patient', positiveClass: 'high', classOrder: ['low', 'high'], multiclass: false, decisionThreshold: .5, frozenDecisionThreshold: .4, thresholdSource: 'descriptive_override', counts: { total: 5, labeled: 3, unlabeled: 2, positive: 3, negative: 0, patients: 5, slides: 7, missingPatientIds: 0 }, metrics: { prevalence: 1, brierScore: .01, brierReference: 0, brierSkillScore: null, logLoss: .1, multiclassBrierScore: null, rocAuc: null, averagePrecision: null, ece: .1, mce: .1 }, operatingPoint: point, operatingCurve: [point], rocCurve: [], precisionRecallCurve: [], calibration: [{ lower: 0, upper: .5, count: 0, meanPredicted: null, observedFraction: null, absoluteError: null }], warnings: ['No negative outcomes.'], definitions: {}, sources: [], uncertainty: { method: 'unavailable', reason: 'Repeated patient observations.', operatingPoint: {} } };
const run = { id: 'run', lifecycleState: 'active', contentHash: 'run', createdAt: '', manifest: { kind: 'model-evaluation', predictorId: 'p1', cohortId: 'cohort', experimentId: 'experiment', status: 'planned', name: 'External validation' } } as ModelEvaluation;
const metrics = { unit: 'patient', classOrder: ['low', 'high'], positiveClass: 'high', decisionThreshold: .5 } as unknown as EvaluationMetrics;
const analysis = (id: string, evaluationId: string, lifecycleState: ClinicalAnalysis['lifecycleState'] = 'active', referenceId?: string) => ({ id, lifecycleState, contentHash: id, createdAt: `2026-09-2${id.length}T00:00:00Z`, manifest: { kind: 'clinical-analysis', name: `${id} report`, datasetId: 'd', experimentId: 'experiment', predictorId: 'p1', evaluationId, selection: { evaluationId, name: `${id} report`, unit: 'selected', bins: 10, thresholdMin: .01, thresholdMax: .99, thresholdSteps: 99, ...(referenceId ? { referenceId } : {}) }, target: null, report } }) as unknown as ClinicalAnalysis;
function renderSection(options: { record?: ModelEvaluation; analyses?: ClinicalAnalysis[] | null; reference?: { id: string; name: string } } = {}) {
  const client = newClient();
  if (options.analyses !== null) client.setQueryData(['clinical-analyses', 'p', 'all'], { items: options.analyses ?? [] });
  return renderToStaticMarkup(<QueryClientProvider client={client}><RunClinicalUtility project="p" record={options.record ?? run} metrics={metrics} reference={options.reference} /></QueryClientProvider>);
}
describe('clinical utility of a run', () => {
  it('keeps settings behind New analysis until the run has an analysis', () => {
    const html = renderSection();
    expect(html).toContain('Clinical utility');
    expect(html).toContain('No clinical utility analysis for this run yet.');
    expect(html).toMatch(/<button[^>]*>New analysis<\/button>/);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>New analysis/);
    expect(html).not.toContain('Operating threshold');
    expect(html).not.toContain('Analyze clinical utility');
  });
  it('reads the library that includes archived and trashed analyses, not the active-only cache', () => {
    const html = renderSection({ analyses: null });
    expect(html).toContain('Loading saved analyses…');
    expect(html).not.toContain('No clinical utility analysis for this run yet.');
    expect(clients.at(-1)!.getQueryState(['clinical-analyses', 'p', 'all'])?.status).toBe('pending');
  });
  it('opens only this run’s analyses and offers new ones only while the run is active', () => {
    const html = renderSection({ analyses: [analysis('mine', 'run'), analysis('other', 'another-run'), analysis('gone', 'run', 'trashed')] });
    expect(html).toContain('3 labeled patient records');
    expect(html).toContain('Copy into a new analysis');
    expect(html).not.toContain('other report'); expect(html).not.toContain('gone report');
    expect(renderSection({ record: { ...run, lifecycleState: 'archived' } })).toMatch(/<button[^>]*disabled=""[^>]*>New analysis/);
  });
  it('lists the analyses of the labels the run is scored against, cohort or reference', () => {
    const analyses = [analysis('cohort', 'run'), analysis('reader', 'run', 'active', 'reader-a')];
    const cohortLabels = renderSection({ analyses });
    expect(cohortLabels).toContain('cohort report'); expect(cohortLabels).not.toContain('reader report');
    const reader = renderSection({ analyses, reference: { id: 'reader-a', name: 'Reader A' } });
    expect(reader).toContain('reader report'); expect(reader).not.toContain('cohort report');
    expect(reader).toContain('with outcomes from Reader A');
    expect(renderSection({ analyses, reference: { id: 'reader-b', name: 'Reader B' } })).toContain('No clinical utility analysis for this run against Reader B yet.');
  });
  it('shows clinical unavailable values without rendering Infinity or invented confidence', () => {
    const html = renderToStaticMarkup(<ClinicalReportView report={report} />);
    expect(html).toContain('3 labeled patient records'); expect(html).toContain('2 unlabeled excluded');
    expect(html).toContain('frozen evaluation still uses 0.400'); expect(html).toContain('Confidence intervals unavailable');
    expect(html).toContain('Unavailable'); expect(html).not.toMatch(/Infinity|NaN/);
    expect(html).toContain('values below −0.10 are clipped');
  });
  it('names the threshold source, including multiclass reports that override nothing', () => {
    const base = { thresholdSource: 'frozen_evaluation' as const, decisionThreshold: .5, frozenDecisionThreshold: .5, multiclass: false, positiveClass: 'high' };
    expect(clinicalThresholdNote(base)).toBe('This is the frozen evaluation threshold.');
    expect(clinicalThresholdNote({ ...base, thresholdSource: 'descriptive_override', decisionThreshold: .3 })).toBe('Exploratory threshold; the frozen evaluation still uses 0.500.');
    const multiclass = clinicalThresholdNote({ ...base, multiclass: true });
    expect(multiclass).toContain('frozen evaluation threshold, applied to high versus the other classes');
    expect(multiclass).not.toContain('Exploratory');
    // Earlier multiclass reports were saved as overrides even though nothing was changed.
    expect(clinicalThresholdNote({ ...base, multiclass: true, thresholdSource: 'descriptive_override' })).toBe(multiclass);
  });
  it('does not mount thousands of hidden curve-table rows', () => {
    const html = renderToStaticMarkup(<CurveChart title="Operating curve" description="Available test evidence" xLabel="Threshold" yLabel="Sensitivity" series={[{ label: 'Model', points: Array.from({ length: 1000 }, (_, i) => ({ x: i / 1000, y: .75 })) }]} />);
    expect(html).not.toContain('<tbody>'); expect(html).toContain('View numeric curve data');
  });
});
describe('interpretation input provenance', () => {
  it('uses shared bundle and folder selection, keeps an unavailable model link and ignores evidence links', () => {
    const html = render({ hash: '#interpretation?predictor=missing&evaluation=eval&clinical=report' });
    expect(html).toContain('Feature bundle'); expect(html).toContain('Original slide features'); expect(html).not.toContain('Slide folder on the server'); expect(html).toContain('Slides come from the chosen frozen dataset');
    expect(html).toContain('Linked predictor unavailable'); expect(html).not.toContain('Linked evaluation'); expect(html).not.toContain('clinical report');
    expect(html).not.toContain('Review attention study'); expect(html).not.toContain('Separate coordinates file');
    expect(html).not.toContain('Compute slide attention');
  });
  it('rejects unsupported pooling models from predictor choices', () => {
    const source = { id: 'mean-model', lifecycleState: 'active', manifest: { name: 'Mean-pooling model', method: 'refit', recipe: { ...defaultRecipe(), model: 'mean_pool' }, checkpoints: [], inputs: { features: {} } } } as unknown as FrozenPredictor;
    const html = render({ predictors: [source] });
    expect(html).toMatch(/<option value="mean-model" disabled="">Mean-pooling model · [^<]* · Refit · attention unsupported/);
  });
});
