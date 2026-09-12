import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { ClinicalReport } from '../api/clinicalUtility';
import type { FrozenPredictor, ModelEvaluation } from '../api/predictors';
import { defaultRecipe } from '../api/development';
import LocalClinicalUtility, { ClinicalReportView } from './LocalClinicalUtility';
import LocalInterpretation from './LocalInterpretation';
import CurveChart from '../components/CurveChart';
const workspace = { mode: 'local', project: { id: 'p', name: 'Evidence project' } } as Workspace;
const clients: QueryClient[] = [];
afterEach(() => { clients.splice(0).forEach((client) => client.clear()); vi.unstubAllGlobals(); });
function render(page: 'clinical' | 'interpretation', options: { hash?: string; evaluations?: ModelEvaluation[]; predictors?: FrozenPredictor[] } = {}) {
  vi.stubGlobal('window', { location: { hash: options.hash ?? '' } });
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } }); clients.push(client);
  for (const key of ['clinical-analyses', 'interpretations']) client.setQueryData([key, 'p'], { items: [] });
  client.setQueryData(['model-evaluations', 'p'], { items: options.evaluations ?? [] });
  client.setQueryData(['predictors', 'p'], { items: options.predictors ?? [] });
  return renderToStaticMarkup(<QueryClientProvider client={client}>{page === 'clinical' ? <LocalClinicalUtility workspace={workspace} /> : <LocalInterpretation workspace={workspace} />}</QueryClientProvider>);
}
describe('clinical utility evidence selection', () => {
  it('cannot fabricate analysis from incomplete or deleted evaluation results', () => {
    const record = (id: string, status: 'completed' | 'running', lifecycleState: 'active' | 'trashed'): ModelEvaluation => ({ id, lifecycleState, contentHash: id, createdAt: '', manifest: { kind: 'model-evaluation', predictorId: 'p1', cohortId: 'cohort', experimentId: 'experiment', status: 'planned', name: id }, execution: { status } });
    const html = render('clinical', { hash: '#clinical-utility?evaluation=incomplete', evaluations: [record('incomplete', 'running', 'active'), record('deleted', 'completed', 'trashed')] });
    expect(html).toContain('Linked evaluation is unavailable or incomplete');
    expect(html).not.toContain('<option value="deleted"');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Analyze clinical utility/);
    expect(html).not.toContain('Mean squared probability error');
  });
  it('shows clinical unavailable values without rendering Infinity or invented confidence', () => {
    const point = { threshold: .5, tp: 3, fp: 0, tn: 0, fn: 0, sensitivity: 1, specificity: null, ppv: 1, npv: null, accuracy: 1, balancedAccuracy: null, f1: 1, positiveLikelihoodRatio: null, negativeLikelihoodRatio: null, predictedPositive: 3, predictedNegative: 0, netBenefit: 1, treatAllNetBenefit: 1, treatNoneNetBenefit: 0, standardizedNetBenefit: 1, netInterventionsAvoidedPer100: 0, highRiskPer100: 100, truePositivePer100: 100, falsePositivePer100: 0, missedPositivePer100: 0 };
    const report: ClinicalReport = { unit: 'patient', frozenUnit: 'patient', positiveClass: 'high', classOrder: ['low', 'high'], multiclass: false, decisionThreshold: .5, frozenDecisionThreshold: .4, thresholdSource: 'descriptive_override', counts: { total: 5, labeled: 3, unlabeled: 2, positive: 3, negative: 0, patients: 5, slides: 7, missingPatientIds: 0 }, metrics: { prevalence: 1, brierScore: .01, brierReference: 0, brierSkillScore: null, logLoss: .1, multiclassBrierScore: null, rocAuc: null, averagePrecision: null, ece: .1, mce: .1 }, operatingPoint: point, operatingCurve: [point], rocCurve: [], precisionRecallCurve: [], calibration: [{ lower: 0, upper: .5, count: 0, meanPredicted: null, observedFraction: null, absoluteError: null }], warnings: ['No negative outcomes.'], definitions: {}, sources: [], uncertainty: { method: 'unavailable', reason: 'Repeated patient observations.', operatingPoint: {} } };
    const html = renderToStaticMarkup(<ClinicalReportView report={report} />);
    expect(html).toContain('3 labeled patient records'); expect(html).toContain('2 unlabeled excluded');
    expect(html).toContain('frozen evaluation still uses 0.400'); expect(html).toContain('Confidence intervals unavailable');
    expect(html).toContain('Unavailable'); expect(html).not.toMatch(/Infinity|NaN/);
    expect(html).toContain('values below −0.10 are clipped');
  });
  it('does not mount thousands of hidden curve-table rows', () => {
    const html = renderToStaticMarkup(<CurveChart title="Operating curve" description="Available test evidence" xLabel="Threshold" yLabel="Sensitivity" series={[{ label: 'Model', points: Array.from({ length: 1000 }, (_, i) => ({ x: i / 1000, y: .75 })) }]} />);
    expect(html).not.toContain('<tbody>'); expect(html).toContain('View numeric curve data');
  });
});
describe('interpretation input provenance', () => {
  it('uses shared bundle and folder selection while preserving unavailable historical links', () => {
    const html = render('interpretation', { hash: '#interpretation?predictor=missing&evaluation=eval&clinical=report' });
    expect(html).toContain('Feature bundle'); expect(html).toContain('Original slide features'); expect(html).not.toContain('Slide folder on the server'); expect(html).toContain('Slides are loaded automatically from the dataset');
    expect(html).toContain('Linked predictor unavailable'); expect(html).toContain('Linked evaluation unavailable or incompatible');
    expect(html).not.toContain('Review attention study'); expect(html).not.toContain('Separate coordinates file');
    expect(html).not.toContain('Compute slide attention');
  });
  it('rejects unsupported pooling models from predictor choices', () => {
    const source = { id: 'mean-model', lifecycleState: 'active', manifest: { name: 'Mean-pooling model', method: 'refit', recipe: { ...defaultRecipe(), model: 'mean_pool' }, checkpoints: [], inputs: { features: {} } } } as unknown as FrozenPredictor;
    const html = render('interpretation', { predictors: [source] });
    expect(html).toMatch(/<option value="mean-model" disabled="">Mean-pooling model · Refit · attention unsupported/);
  });
});
