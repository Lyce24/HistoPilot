import type { ReactNode } from 'react';
import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { ProtocolExploration } from '../api/scientific';
import type { EvaluationPreview } from '../api/evaluation';
import { parseConditionValue } from '../lib/conditions';
import LocalEvaluationSetup, { EvaluationInferenceFields, newEvaluationSpec, TestCohortSummary, cohortDatasetIds, mergeTestDistributions, independentCohortSpec, cohortIdentityNote, newInferenceSpec } from './LocalEvaluationSetup';

const preview: EvaluationPreview = {
  spec: newEvaluationSpec(),
  target: { field: 'grade', task: 'binary_classification', unit: 'patient', classes: ['low', 'high'], labels: { low: 'low', high: 'high' }, positiveClass: 'high', missing: 'block', unmapped: 'block' },
  summary: { includedSlides: 3, includedPatients: 2, excludedSlides: 7, labeledSlides: 3, classCounts: { low: 1, high: 2 }, developmentSlideOverlap: 1, developmentPatientOverlap: 1 },
  coverage: { selectedSlideIds: ['test-1', 'test-2', 'development-3'], featureSlideCount: 20, missingFeatureSlideIds: ['test-2'], missingPackSlideIds: ['test-2'], packChecked: true },
  overlap: { slideIds: ['development-3'], patientIds: ['patient-3'], patientsComparable: true },
  compatibility: { development: { dimensions: 1024, encoderId: 'uni' }, evaluation: { dimensions: 1024, encoderId: 'uni' } },
  findings: [{ severity: 'error', code: 'MISSING_TEST_FEATURES', message: 'One selected test slide has no features.' }, { severity: 'error', code: 'DEVELOPMENT_OVERLAP', message: 'One selected slide occurs in development.' }],
  canFreeze: false, executionEnabled: false, previewHash: 'preview',
};
// Apply models places the cohort library in its own page; this stands in for that frame.
const frame = (library: ReactNode, actions: ReactNode) => <section data-frame=""><div data-frame-actions="">{actions}</div>{library}</section>;

describe('cohorts in Apply models', () => {
  it('requires explicit numeric inference values, including zero workers and threshold boundaries', () => {
    const html = renderToStaticMarkup(<EvaluationInferenceFields value={{ ...newEvaluationSpec().inference, numWorkers: 0, decisionThreshold: 0 }} target={preview.target!} onChange={() => {}} />);
    expect(html.match(/<input[^>]*required=""/g)).toHaveLength(3);
    expect(html.match(/<input[^>]*inputMode="numeric"/g)).toHaveLength(2);
    expect(html).toMatch(/<input[^>]*inputMode="decimal"[^>]*value="0"/);
    expect(html).not.toContain('type="number"');
  });

  it('offers supported patient aggregation and identifies incompatible saved settings without changing them', () => {
    const current = renderToStaticMarkup(<EvaluationInferenceFields value={newEvaluationSpec().inference} target={preview.target!} onChange={() => {}} />);
    expect(current).toContain('Mean probabilities');
    expect(current).not.toContain('value="max"');
    const legacy = renderToStaticMarkup(<EvaluationInferenceFields value={{ ...newEvaluationSpec().inference, patientAggregation: 'max' }} target={preview.target!} onChange={() => {}} />);
    expect(legacy).toMatch(/<option value="max" disabled="" selected="">Maximum probabilities · unsupported; choose mean/);
    expect(legacy).toContain('Match the patient scoring rule saved with the predictor');
    expect(legacy).toContain('value="mean_logits"');
    expect(legacy).toContain('value="predictor"');
  });

  it('lists cohorts by their labels inside the Apply models frame and offers both kinds', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
    client.setQueryData(['evaluation-drafts', 'project'], { drafts: [{ id: 'unlabeled', name: 'Unlabeled slides', revision: 1, status: 'editable', payload: { type: 'evaluation-cohort', spec: newInferenceSpec() } }] });
    client.setQueryData(['evaluation-cohorts', 'project'], { items: [] });
    const workspace = { project: { id: 'project', name: 'Demo' } } as Workspace;
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><LocalEvaluationSetup workspace={workspace} frame={frame} /></QueryClientProvider>);
      expect(html).toContain('data-frame=""');
      // The module's page names the stage; the library adds no heading of its own.
      expect(html).not.toContain('class="eyebrow"');
      expect(html).toContain('Unlabeled slides');
      expect(html).toContain('>Unlabeled<');
      expect(html).toContain('Labels · target');
      expect(html).toMatch(/data-frame-actions="">.*Create labeled cohort.*Create unlabeled cohort/);
    } finally { client.clear(); }
  });

  it('opens a new cohort of the kind a link asks for, on a page of its own', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
    client.setQueryData(['evaluation-drafts', 'project'], { drafts: [] });
    client.setQueryData(['evaluation-cohorts', 'project'], { items: [] });
    const workspace = { project: { id: 'project', name: 'Demo' } } as Workspace;
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><LocalEvaluationSetup workspace={workspace} frame={frame} newCohort="unlabeled" /></QueryClientProvider>);
      expect(html).not.toContain('data-frame=""');
      expect(html).toContain('<div class="eyebrow">04 Apply models</div>');
      expect(html).toContain('Create unlabeled cohort');
      expect(html).toContain('Back to cohorts');
      expect(html).toContain('value="Demo unlabeled cohort"');
      expect(html).toContain('1. Choose the slides');
    } finally { client.clear(); }
  });

  it('summarizes inference cohorts without a labeled-slide count', () => {
    const preview = { summary: { includedSlides: 120, includedPatients: 80, excludedSlides: 240, labeledSlides: 0, classCounts: {}, developmentSlideOverlap: 0, developmentPatientOverlap: 0 }, findings: [], coverage: { selectedSlideIds: [] } };
    const inference = renderToStaticMarkup(<TestCohortSummary preview={preview} inference />);
    expect(inference).toContain('Slides to predict');
    expect(inference).not.toContain('Labeled slides');
    expect(renderToStaticMarkup(<TestCohortSummary preview={preview} />)).toContain('Labeled slides');
  });

  it('starts with only a cohort registry and create action, without development prerequisites', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    client.setQueryData(['evaluation-drafts', 'project'], { drafts: [{ id: 'saved-cohort', name: 'Existing test cohort', revision: 2, status: 'editable', payload: { type: 'evaluation-cohort', spec: newEvaluationSpec() } }] });
    client.setQueryData(['evaluation-cohorts', 'project'], { items: [] });
    const workspace = { project: { id: 'project', name: 'BLCA' } } as Workspace;
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><LocalEvaluationSetup workspace={workspace} frame={frame} /></QueryClientProvider>);
      expect(html).toContain('Existing test cohort');
      expect(html).toContain('Create labeled cohort');
      expect(html).toContain('Search cohorts');
      expect(html).toContain('Cohort status');
      expect(html).toContain('Sort cohorts');
      expect(html).toContain('Manage');
      expect(html).toContain('data-record-key="draft:saved-cohort');
      expect(html).not.toContain('Your test cohorts');
      expect(html).not.toContain('Stage 0 · Saved records');
      expect(html).toContain('Planned');
      expect(html).not.toContain('Development protocol');
      expect(html).not.toContain('Feature bundle');
      expect(html).not.toContain('Cohort name');
      expect(html).not.toContain('Cohort stages');
      expect(html).not.toContain('Save draft');
      expect(html).not.toContain('Cross-validation folds');
      expect(html).not.toContain('Split seed');
      expect(html).not.toContain('Start inference');
    } finally { client.clear(); }
  });

  it('summarizes selected slides and class distributions without model checks', () => {
    const html = renderToStaticMarkup(<TestCohortSummary preview={{ ...preview, findings: [] }} />);
    expect(html).toContain('Selected slides');
    expect(html).toContain('Prediction target · selected slides by class');
    expect(html).toContain('Patient / slide groups');
    expect(html).toContain('Excluded slides');
    expect(html).toContain('test-1');
    expect(html).not.toContain('Exact feature coverage');
    expect(html).not.toContain('Overlap with development');
    expect(html).not.toContain('Missing pack slide IDs');
  });

  it('counts supplied patient IDs separately from acknowledged slide groups using frozen provenance', () => {
    const memberships: NonNullable<EvaluationPreview['memberships']> = [
      { slideId: 's1', patientId: 'p1', patientIdSource: 'source' },
      { slideId: 's2', patientId: 'p1', patientIdSource: 'crosswalk' },
      { slideId: 's3', patientId: 's3', patientIdSource: 'slide_fallback' },
      { slideId: 's4', patientId: 'legacy-id' },
      { slideId: 's5', patientId: null, patientIdSource: 'unresolved' },
    ];
    const value = { ...preview, memberships, findings: [] };
    const html = renderToStaticMarkup(<TestCohortSummary preview={value} />);
    expect(html).toContain('1 supplied patient IDs · 1 acknowledged slide / case groups · 2 slides with unresolved or unrecorded patient-ID provenance');
    expect(html).not.toContain('Verified patients');
  });

  it('never calls fallback or legacy grouping IDs verified patients or infers counts from warning prose', () => {
    const findings = [{ code: 'SLIDE_ID_FALLBACK_GROUPING', severity: 'warning' as const, message: '76 slides use fallback.' }];
    expect(cohortIdentityNote({ findings })).toBe('Includes acknowledged slide / case groups. Patient independence is unverified.');
    expect(cohortIdentityNote({ findings: [] })).toContain('provenance counts are unavailable');
    const memberships = [{ slideId: 's1', patientId: 's1', patientIdSource: 'slide_fallback' as const }];
    expect(cohortIdentityNote({ findings, memberships })).toBe('0 supplied patient IDs · 1 acknowledged slide / case groups');
  });

  it('preserves single-dataset drafts and combines multi-dataset distributions with truncation evidence', () => {
    expect(cohortDatasetIds({ ...newEvaluationSpec(), datasetId: 'one' })).toEqual(['one']);
    expect(cohortDatasetIds({ ...newEvaluationSpec(), datasetId: 'one', datasetIds: ['one', 'two'] })).toEqual(['one', 'two']);
    const source = (values: { value: string | null; slides: number }[], distinctCount: number) => ({ target: { field: 'grade', values, distinctCount } }) as ProtocolExploration;
    const result = mergeTestDistributions([
      source([{ value: 'high', slides: 2 }, { value: null, slides: 1 }], 2),
      source([{ value: 'high', slides: 3 }, { value: 'low', slides: 2 }], 3),
    ]);
    expect(result.valueCounts).toEqual([{ value: 'high', count: 5 }, { value: 'low', count: 2 }, { value: null, count: 1 }]);
    expect(result.valuesTruncated).toBe(true);
    expect(newEvaluationSpec().protocolId).toBeFalsy();
    expect(newEvaluationSpec().featureBundleId).toBeFalsy();
    expect(newEvaluationSpec().developmentFeatureBundleId).toBeFalsy();
  });

  it('resumes old drafts as independent cohorts while preserving selected data and labels', () => {
    const legacy = { ...newEvaluationSpec(), datasetId: 'test-dataset', protocolId: 'old-protocol', developmentFeatureBundleId: 'missing-development-features', featureBundleId: 'missing-test-features', target: preview.target, eligibility: [{ field: 'partition', op: 'eq' as const, value: 'test' }] };
    const independent = independentCohortSpec(legacy);
    expect(independent.protocolId).toBeFalsy();
    expect(independent.developmentFeatureBundleId).toBeFalsy();
    expect(independent.featureBundleId).toBeFalsy();
    expect(independent.datasetId).toBe(legacy.datasetId);
    expect(independent.target).toEqual(legacy.target);
    expect(independent.eligibility).toEqual(legacy.eligibility);
    expect(legacy.protocolId).toBe('old-protocol');
  });

  it('keeps numeric and boolean filters typed while preserving exact categorical and regex values', () => {
    expect(parseConditionValue('4', 'gte', 'text')).toBe(4);
    expect(parseConditionValue('1 | 2', 'in', 'integer')).toEqual([1, 2]);
    expect(parseConditionValue('001', 'eq', 'text')).toBe('001');
    expect(parseConditionValue('false', 'eq', 'boolean')).toBe(false);
    expect(parseConditionValue('^1$', 'regex', 'integer')).toBe('^1$');
    expect(parseConditionValue('', 'gte', 'integer')).toBe('');
    expect(parseConditionValue('Infinity', 'gte', 'integer')).toBe('Infinity');
  });
});

it('continues earlier review cohorts and unlabeled cohorts as unlabeled cohorts', () => {
  const spec = independentCohortSpec({ ...newEvaluationSpec(), purpose: 'review', target: null });
  expect(spec.purpose).toBe('inference');
  expect(spec.patientIdentifiers).toBe('shared');
  expect(spec.target).toBeNull();
  expect(independentCohortSpec(newInferenceSpec())).toMatchObject({ purpose: 'inference', target: null });
  // A labeled cohort stays labeled, with its target.
  const labeled = independentCohortSpec(newEvaluationSpec());
  expect(labeled.purpose).toBeUndefined();
  expect(labeled.target).not.toBeNull();
});
