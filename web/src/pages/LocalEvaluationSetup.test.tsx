import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { EvaluationPreview } from '../api/evaluation';
import LocalEvaluationSetup, { EvaluationEvidence, EvaluationInferenceFields, EvaluationTargetMapping, evaluationConditionValue, evaluationLabelProblem, newEvaluationSpec } from './LocalEvaluationSetup';

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

describe('later test cohort setup', () => {
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
    expect(legacy).toContain('match the patient aggregation used by frozen predictors');
  });

  it('allows cohort preparation with no trained models and does not introduce split or run controls', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    client.setQueryData(['evaluation-drafts', 'project'], { drafts: [{ id: 'saved-cohort', name: 'Existing test cohort', revision: 2, status: 'editable', payload: { type: 'evaluation-cohort', spec: newEvaluationSpec() } }] });
    client.setQueryData(['evaluation-cohorts', 'project'], { items: [] });
    const workspace = { project: { id: 'project', name: 'BLCA' } } as Workspace;
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><LocalEvaluationSetup workspace={workspace} /></QueryClientProvider>);
      expect(html).toContain('Existing test cohort');
      expect(html).toContain('while models are being developed');
      expect(html).toContain('import and freeze it in Data');
      expect(html).toContain('A trained model is not required to prepare one.');
      expect(html).toContain('choose an ensemble or refit predictor in Evaluate models');
      expect(html).not.toContain('Cross-validation folds');
      expect(html).not.toContain('Split seed');
      expect(html).not.toContain('Start inference');
      expect(html.match(/<button[^>]*>Save draft<\/button>/)?.[0]).not.toContain('disabled');
      expect(html).toMatch(/<button[^>]*disabled=""[^>]*><svg[^]*?Review cohort<\/button>/);
    } finally { client.clear(); }
  });

  it('shows exact missing feature and pack IDs even when the bundle has many other slides', () => {
    const html = renderToStaticMarkup(<EvaluationEvidence preview={preview} />);
    expect(html).toContain('Missing feature slide IDs');
    expect(html).toContain('Missing pack slide IDs');
    expect(html).toContain('<pre>test-2</pre>');
    expect(html).toContain('One selected test slide has no features.');
    expect(html).toContain('Blocking');
    expect(html).toContain('Overlapping development patient IDs');
    expect(html).toContain('<pre>patient-3</pre>');
    expect(html).not.toContain('Every selected slide has a feature file.');
  });

  it('does not report patient disjointness when dataset naming systems are independent', () => {
    const html = renderToStaticMarkup(<EvaluationEvidence preview={{ ...preview, overlap: { slideIds: [], patientIds: [], patientsComparable: false }, summary: { ...preview.summary, developmentSlideOverlap: 0, developmentPatientOverlap: 0 } }} />);
    expect(html).toContain('Patient overlap cannot be checked across separate naming systems.');
    expect(html).not.toContain('0 overlapping patient IDs');
  });

  it('keeps failed packed verification visible instead of claiming original-file loading', () => {
    const html = renderToStaticMarkup(<EvaluationEvidence preview={{ ...preview, spec: { ...preview.spec, inference: { ...preview.spec.inference, loadingPolicy: 'packed', packArtifactId: 'unavailable' } }, coverage: { ...preview.coverage, packChecked: false } }} />);
    expect(html).toContain('Packed loading was requested, but pack coverage could not be verified.');
    expect(html).not.toContain('Original feature files selected');
  });

  it('allows separate-file raw values to map to fixed inherited classes and rejects ambiguous mappings', () => {
    const html = renderToStaticMarkup(<EvaluationTargetMapping rows={[["0", "low"], ["1", "high"]]} classes={['low', 'high']} onChange={() => {}} />);
    expect(html).toMatch(/<input[^>]*value="0"/);
    expect(html).toMatch(/<input[^>]*value="1"/);
    expect(html).toContain('Inherited class');
    expect(html).toContain('Add raw label value');
    expect(html).not.toContain('readonly');
    expect(evaluationLabelProblem([['0', 'low'], ['1', 'high']], ['low', 'high'])).toBeNull();
    expect(evaluationLabelProblem([['0', 'low'], ['0', 'high']], ['low', 'high'])).toContain('exactly one class');
    expect(evaluationLabelProblem([['0', 'low']], ['low', 'high'])).toContain('every inherited class');
  });

  it('keeps numeric and boolean filters typed while preserving exact categorical and regex values', () => {
    expect(evaluationConditionValue('4', 'gte', 'text')).toBe(4);
    expect(evaluationConditionValue('1 | 2', 'in', 'integer')).toEqual([1, 2]);
    expect(evaluationConditionValue('001', 'eq', 'text')).toBe('001');
    expect(evaluationConditionValue('false', 'eq', 'boolean')).toBe(false);
    expect(evaluationConditionValue('^1$', 'regex', 'integer')).toBe('^1$');
    expect(evaluationConditionValue('', 'gte', 'integer')).toBe('');
    expect(evaluationConditionValue('Infinity', 'gte', 'integer')).toBe('Infinity');
  });
});
