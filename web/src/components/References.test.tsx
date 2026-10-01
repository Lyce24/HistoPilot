import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import type { EvaluationCohort } from '../api/evaluation';
import type { ModelEvaluation } from '../api/predictors';
import type { ReferenceStandard, RunAgreement as Agreement } from '../api/references';
import type { DatasetVersion } from '../api/scientific';
import CohortReferences, { referenceClassSets } from './CohortReferences';
import ReferenceStandardEditor, { suggestedClass } from './ReferenceStandardEditor';
import RunAgreement, { agreementPair } from './RunAgreement';
import RunRecalibration, { recalibrationMap } from './RunRecalibration';
import type { CalibrationMetrics, Recalibration } from '../api/recalibration';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function withClient(seed: (client: QueryClient) => void, element: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  seed(client);
  return renderToStaticMarkup(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
}
const cohort = (target: string[] | null = null) => ({ id: 'cohort', current: true, versionLabel: { tag: 'Unlabeled slides' }, manifest: { kind: 'evaluation-cohort', datasetId: 'd1', spec: { purpose: target ? undefined : 'inference' }, target: target ? { classes: target } : null, summary: { includedSlides: 12, labeledSlides: 0 } } }) as unknown as EvaluationCohort;
const run = (id: string, classes: string[], cohortId = 'cohort') => ({ id, lifecycleState: 'active', createdAt: '', contentHash: id, manifest: { kind: 'model-evaluation', name: `Run ${id}`, cohortId, predictorId: 'p', experimentId: 'e', status: 'planned', target: { classes } }, execution: { status: 'completed' } }) as unknown as ModelEvaluation;
const dataset = (id: string, createdAt: string, columns: string[]) => ({ id, projectId: 'p', createdAt, contentHash: id, versionLabel: { tag: `Dataset ${id}` }, manifest: { dictionary: columns.map((key) => ({ key, sourceColumn: key.toUpperCase(), owner: 'slide', type: 'text' })) }, artifacts: {} }) as unknown as DatasetVersion;
const standard = (id: string, name: string, classes: string[]) => ({ id, createdAt: '', contentHash: id, lifecycleState: 'active',
  manifest: { kind: 'reference-standard', name, cohortId: 'cohort', cohort: { id: 'cohort', contentHash: 'c' }, datasetId: 'd2', datasets: [], field: 'reader_a', matchedBy: 'slideId', classes, labels: {}, summary: { slides: 12, labeledSlides: 10, missingSlides: 2, unmappedSlides: 0, unmatchedSlides: 0, classCounts: { ND: 4, LG: 3, HG: 3 }, conflictingPatients: 0, values: [], valuesTruncated: false }, findings: [] } }) as unknown as ReferenceStandard;

describe('reference standards', () => {
  it('maps a source value to the class it names, ignoring case and spaces', () => {
    expect(suggestedClass('  hg ', ['ND', 'LG', 'HG'])).toBe('HG');
    expect(suggestedClass('IND', ['ND', 'LG', 'HG'])).toBeNull();
  });

  it('starts a reference from the cohort’s dataset, offering only columns every chosen dataset has', () => {
    const html = withClient((client) => client.setQueryData(['scientific', 'p', 'datasets'], { datasets: [dataset('d2', '2026-09-20T00:00:00Z', ['reader_a', 'consensus']), dataset('d1', '2026-09-01T00:00:00Z', ['reader_a'])] }),
      <ReferenceStandardEditor project="p" cohort={cohort()} classes={['ND', 'LG', 'HG']} onSaved={() => {}} onCancel={() => {}} />);
    expect(html).toContain('map its values to ND, LG, HG');
    // The cohort's own dataset is listed first and chosen; its columns are the only ones offered.
    expect(html.indexOf('Dataset d1')).toBeLessThan(html.indexOf('Dataset d2'));
    expect(html).toContain('This cohort’s dataset');
    expect(html).toContain('<option value="reader_a">READER_A</option>');
    expect(html).not.toContain('<option value="consensus">');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Review reference standard<\/button>/);
  });

  it('takes a reference’s classes from the cohort’s target or the runs on it', () => {
    expect(referenceClassSets(cohort(['a', 'b']), [run('r1', ['b', 'a']), run('r2', ['a', 'b', 'c']), run('r3', ['x', 'y'], 'other')])).toEqual([['a', 'b'], ['a', 'b', 'c']]);
    expect(referenceClassSets(cohort(), [])).toEqual([]);
  });

  it('lists a cohort’s references with the runs each one scores', () => {
    const html = withClient((client) => {
      client.setQueryData(['reference-standards', 'p', 'cohort'], { items: [standard('ref-b', 'Reader B', ['ND', 'LG', 'HG']), standard('ref-a', 'Consensus', ['HG', 'LG', 'ND'])] });
      client.setQueryData(['model-evaluations', 'p'], { items: [run('r1', ['ND', 'LG', 'HG']), run('r2', ['a', 'b'])] });
    }, <CohortReferences project="p" cohort={cohort()} />);
    expect(html.indexOf('Consensus')).toBeLessThan(html.indexOf('Reader B'));
    expect(html).toContain('10 of 12');
    expect(html).toContain('ND 4 · LG 3 · HG 3');
    expect(html).toContain('href="#apply?run=r1&amp;tab=performance&amp;reference=ref-a"');
    expect(html).not.toContain('Run r2');
    expect(html).toMatch(/<button[^>]*>Add reference standard<\/button>/);
  });

  it('shows how the run and every label source agree, pair by pair', () => {
    const agreement: Agreement = {
      evaluationId: 'run', unit: 'slide', classOrder: ['ND', 'LG', 'HG'], developmentExcluded: 3, source: { predictionsSha256: 'x' },
      sources: [{ id: 'run', name: 'Seed ensemble', kind: 'run', labeled: 12 }, { id: 'ref-a', name: 'Consensus', kind: 'reference', labeled: 10 }, { id: 'ref-b', name: 'Reader B', kind: 'reference', labeled: 9 }, { id: 'ref-c', name: 'Reader C', kind: 'reference', reason: 'The reference standard covers different slides than this run.' }],
      pairs: [
        { left: 'run', right: 'ref-a', count: 10, agreement: 0.8, kappa: 0.7, weightedKappa: 0.75, disagreements: 2, matrix: [[4, 0, 0], [1, 2, 0], [0, 1, 2]] },
        { left: 'run', right: 'ref-b', count: 9, agreement: 0.6, kappa: 0.4, weightedKappa: 0.5, disagreements: 4, matrix: [[3, 1, 0], [1, 1, 1], [0, 1, 1]] },
        { left: 'ref-a', right: 'ref-b', count: 9, agreement: 0.9, kappa: 0.85, weightedKappa: 0.9, disagreements: 1, matrix: [[4, 0, 0], [0, 2, 1], [0, 0, 2]] },
      ],
    };
    expect(agreementPair(agreement.pairs, 'ref-b', 'ref-a')?.kappa).toBe(0.85);
    const html = withClient((client) => client.setQueryData(['run-agreement', 'p', 'run', 'selected'], agreement),
      <RunAgreement project="p" evaluationId="run" patient={null} targetUnit="slide" onReview={() => {}} />);
    expect(html).toContain('Linear weighted κ');
    expect(html).toContain('aria-label="Seed ensemble and Consensus: Cohen’s κ 0.700 over 10"');
    expect(html).toContain('aria-label="Consensus and Reader B: Cohen’s κ 0.850 over 9"');
    expect(html).toContain('on the order ND &lt; LG &lt; HG');
    expect(html).toContain('3 slides from development patients are left out');
    expect(html).toContain('Reader C cannot be compared: The reference standard covers different slides than this run.');
  });

  it('reads an expected reason for missing agreement or recalibration as a note, not an error', () => {
    // A failed query is rendered as it stands; without retryOnMount off, the first render would optimistically fetch.
    const withError = (queryKey: unknown[], code: string, message: string, element: React.ReactElement) => {
      const client = new QueryClient({ defaultOptions: { queries: { retry: false, retryOnMount: false, staleTime: Infinity } } });
      clients.push(client);
      const query = client.getQueryCache().build(client, { queryKey });
      query.setState({ status: 'error', error: new ApiError(message, 409, code), fetchStatus: 'idle' } as Partial<typeof query.state>);
      return renderToStaticMarkup(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
    };
    const agreement = withError(['run-agreement', 'p', 'run', 'selected'], 'AGREEMENT_UNAVAILABLE', 'The saved decisions cannot be paired yet.',
      <RunAgreement project="p" evaluationId="run" patient={null} targetUnit="slide" onReview={() => {}} />);
    expect(agreement).toContain('<p class="muted" role="status">The saved decisions cannot be paired yet.</p>');
    expect(agreement).not.toContain('role="alert"');
    const run = { id: 'run', manifest: { name: 'External run' } } as unknown as ModelEvaluation;
    const recalibration = withError(['run-recalibration', 'p', 'run', 'selected', null], 'RECALIBRATION_UNAVAILABLE', 'This predictor records no development predictions to fit a map on.',
      <RunRecalibration project="p" record={run} unit="selected" referenceId={null} />);
    expect(recalibration).toContain('<p class="muted" role="status">This predictor records no development predictions to fit a map on.</p>');
    expect(recalibration).not.toContain('role="alert"');
    // Anything else is still an error.
    const failure = withError(['run-recalibration', 'p', 'run', 'selected', null], 'INTERNAL', 'The service failed.',
      <RunRecalibration project="p" record={run} unit="selected" referenceId={null} />);
    expect(failure).toContain('role="alert"');
    expect(failure).toContain('The service failed.');
  });

  it('compares calibration as predicted and after a map fitted on development predictions', () => {
    const metrics = (brierScore: number, calibrationSlope: number | null): CalibrationMetrics => ({ count: 120, brierScore, logLoss: .5, ece: .08, meanPredictedRisk: .4, observedFraction: .45, observedExpectedRatio: 1.1, calibrationSlope, calibrationIntercept: .1,
      bins: [{ lower: 0, upper: .1, count: 10, meanPredicted: .05, observedFraction: .1 }, { lower: .9, upper: 1, count: 12, meanPredicted: .95, observedFraction: .8 }] });
    const report: Recalibration = {
      evaluationId: 'run', unit: 'patient', classOrder: ['a', 'b'], positiveClass: 'b', method: 'platt', parameters: { slope: .6, intercept: -.1 },
      development: { units: 360, seedGroups: 3, original: metrics(.2, .6), recalibrated: metrics(.18, 1) },
      cohort: { units: 120, original: metrics(.21, .55), recalibrated: metrics(.19, null) },
      developmentExcluded: 4, reference: { id: 'ref-a', name: 'Consensus' }, developmentSources: [],
    };
    expect(recalibrationMap(report)).toBe('Platt scaling, logit p′ = 0.600 · logit p − 0.100, fitted on 360 development patients (3 seed groups averaged).');
    expect(recalibrationMap({ ...report, method: 'temperature', parameters: { temperature: 1.8 }, development: { ...report.development, seedGroups: 1 } })).toBe('Temperature scaling, T = 1.800, fitted on 360 development patients.');
    const run = { id: 'run', manifest: { name: 'External run' } } as unknown as ModelEvaluation;
    const html = withClient((client) => client.setQueryData(['run-recalibration', 'p', 'run', 'selected', 'ref-a'], report),
      <RunRecalibration project="p" record={run} unit="selected" referenceId="ref-a" />);
    expect(html).toContain('never on this cohort');
    expect(html).toContain('Scored on 120 labeled patients from Consensus; 4 slides of development patients left out.');
    expect(html).toContain('<th scope="col">Risk of b</th>');
    expect(html).toMatch(/Brier score<small>[^<]*<\/small><\/th><td>0.210<\/td><td>0.190<\/td>/);
    // A slope that is not estimable reads as unavailable, never as a number.
    expect(html).toMatch(/Calibration slope<small>[^<]*<\/small><\/th><td>0.550<\/td><td>—<\/td>/);
    expect(html).toContain('Reliability');
  });
});

