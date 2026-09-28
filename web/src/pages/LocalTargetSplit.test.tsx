import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { DatasetVersion } from '../api/scientific';
import type { Workspace } from '../api/types';
import { newTargetSplitSpec, targetSplitPartitionRequest, targetSplitTrainingTarget, targetSplitTestingIssue, targetSplitUnit, targetSplitWithUnit, targetDefinitionReady, targetSplitMethod, targetSplitTestingRemainder, targetSplitSetupLink, TARGET_SPLIT_STEPS, type TargetSplit } from '../api/targetSplits';
import { editorRecoveryKey } from '../lib/editorRecovery';
import LocalTargetSplit, { TargetSplitSummary, TargetSplitTestCohort, TargetSplitTestingRules, targetSplitKey } from './LocalTargetSplit';

const workspace = { project: { id: 'project', name: 'Study', config: { seed: 42 } }, dataset: { id: 'dataset' } } as Workspace;
const dataset = { id: 'dataset', projectId: 'project', createdAt: '', contentHash: 'dataset-hash', artifacts: {}, versionLabel: { tag: 'Reviewed slides' }, manifest: { dictionary: [] } } as unknown as DatasetVersion;
const spec = { ...newTargetSplitSpec('dataset'), target: { ...newTargetSplitSpec().target, field: 'grade', task: 'binary_classification' as const, classes: ['low', 'high'], positiveClass: 'high', labels: { '1': 'low', '2': 'high' } } };
const record: TargetSplit = { id: 'target-split-a', projectId: 'project', contentHash: 'hash', createdAt: '2026-09-25T12:00:00Z', versionLabel: { tag: 'Grade fixed holdout', note: 'Independent testing', revision: 1, createdAt: '', updatedAt: '' }, manifest: {
  kind: 'target-split', datasetId: 'dataset', spec, previewHash: 'preview', findings: [], memberships: [],
  summary: { totalSlides: 20, includedSlides: 20, includedPatients: 10, excludedSlides: 0, trainingSlides: 16, testingSlides: 4, trainingPatients: 8, testingPatients: 2, trainingGroups: 8, testingGroups: 2, classCounts: { low: 10, high: 10 }, trainingClassCounts: { low: 8, high: 8 }, testingClassCounts: { low: 2, high: 2 } },
} };
afterEach(() => vi.unstubAllGlobals());
function render(records: TargetSplit[] = [], recovered = false, detail = false, protocols: unknown[] = []) {
  const recovery = { version: 1, value: { version: 1, name: 'Recovered train/test', spec, draft: null, step: 3 } };
  vi.stubGlobal('window', { location: { hash: detail ? '#cohort?targetSplit=target-split-a' : '#cohort' }, sessionStorage: { getItem: (key: string) => recovered && key === editorRecoveryKey('project', 'target-split') ? JSON.stringify(recovery) : null } });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [dataset] });
  client.setQueryData(['scientific', 'project', 'drafts'], { drafts: [] });
  client.setQueryData(targetSplitKey('project'), { configurations: records });
  client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: protocols });
  if (detail) client.setQueryData([...targetSplitKey('project'), record.id], record);
  try { return renderToStaticMarkup(<QueryClientProvider client={client}><LocalTargetSplit workspace={workspace} /></QueryClientProvider>); }
  finally { client.clear(); }
}

describe('dataset target and train/test construction', () => {
  it('opens a library with one create action and no training configuration', () => {
    const html = render();
    expect(html).toContain('No targets and splits yet');
    expect(html).toContain('Create targets &amp; splits');
    expect(html.match(/data-stage-action="create"/g)).toHaveLength(1);
    expect(html).not.toContain('href="#legacy-protocol"');
    for (const control of ['Feature bundle', 'Folds', 'Validation fraction', 'Training seeds', 'Optimizer']) expect(html).not.toContain(control);
  });
  it('links historical combined protocols only when one exists, ignoring Setup-derived designs', () => {
    const derived = { id: 'derived', manifest: { kind: 'protocol', sourceTargetSplit: { id: 'target-split-a' }, spec: { sourceTargetSplitId: 'target-split-a' } } };
    expect(render([], false, false, [derived])).not.toContain('href="#legacy-protocol"');
    const historical = { id: 'historical', manifest: { kind: 'protocol', spec: { datasetId: 'dataset' } } };
    expect(render([], false, false, [derived, historical])).toContain('href="#legacy-protocol"');
  });
  it('lists fixed training, testing, target and label metadata', () => {
    const html = render([record]);
    for (const column of ['Name', 'Status', 'Dataset', 'Training', 'Testing', 'Target', 'Labels', 'Updated']) expect(html).toContain(`<th>${column}</th>`);
    expect(html).toContain('Grade fixed holdout');
    expect(html).toContain('Reviewed slides');
    expect(html).toContain('16 slides');
    expect(html).toContain('4 slides');
    expect(html).toContain('low, high');
  });
  it('opens a frozen record with a precise Experimental Setup link and fixed assignments', () => {
    const html = render([record], false, true);
    expect(html).toContain('Continue to Experimental Setup');
    expect(html).toContain('href="#experimental-setup?dataset=dataset&amp;targetSplit=target-split-a"');
    expect(html).toContain('Training and testing memberships are fixed in this version.');
    expect(html).toContain('Independent testing');
    expect(html).not.toContain('Training seeds');
  });
  describe('test cohort made from the testing set', () => {
    const cohort = (value: Partial<TargetSplit>, testTarget: TargetSplit['manifest']['spec']['testTarget'] = undefined) => {
      const client = new QueryClient();
      const item = { ...record, ...value, manifest: { ...record.manifest, spec: { ...spec, testTarget } } } as TargetSplit;
      return renderToStaticMarkup(<QueryClientProvider client={client}><TargetSplitTestCohort project="project" record={item} /></QueryClientProvider>);
    };
    it('links the cohort that freezing created', () => {
      const html = cohort({ evaluationCohortId: 'cohort/1', testCohort: { required: true, id: 'cohort/1', state: 'active' } });
      expect(html).toContain('saved its testing set as an evaluation cohort');
      expect(html).toContain('href="#evaluation?cohort=cohort%2F1"');
      expect(html).toContain('href="#test-data"');
      expect(html).not.toContain('Create test cohort');
      expect(cohort({ testCohort: { required: true, id: 'c', state: 'active' } }, null)).toContain('href="#inference?cohort=c"');
    });
    it('offers creation for an older version, and a retry after a failed cohort step', () => {
      expect(cohort({ evaluationCohortId: null, testCohort: { required: true, id: null, state: null } })).toContain('Create test cohort');
      const failed = cohort({ testCohort: { required: true, id: null, state: null }, testCohortError: { code: 'TARGET_TESTING_BLOCKED', message: 'Labels are missing.' } });
      expect(failed).toContain('This version is frozen, but its testing set could not be saved as an evaluation cohort: Labels are missing.');
      expect(failed).toContain('Retry test cohort');
    });
    it('says when the cohort is in Trash, and stays silent without a testing set', () => {
      expect(cohort({ testCohort: { required: true, id: 'c', state: 'trashed' } })).toContain('is in Trash');
      expect(cohort({ testCohort: { required: false, id: null, state: null } })).toBe('');
      // Older servers only report the cohort ID.
      expect(cohort({ evaluationCohortId: null })).toBe('');
    });
    it('shows the test cohort in the detail view', () => {
      const html = render([record], false, true);
      expect(html).not.toContain('aria-label="Test cohort"');
      const detail = { ...record, testCohort: { required: true, id: null, state: null } } as TargetSplit;
      vi.stubGlobal('window', { location: { hash: '#cohort?targetSplit=target-split-a' }, sessionStorage: { getItem: () => null } });
      const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
      client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [dataset] });
      client.setQueryData(['scientific', 'project', 'drafts'], { drafts: [] });
      client.setQueryData(targetSplitKey('project'), { configurations: [record] });
      client.setQueryData([...targetSplitKey('project'), record.id], detail);
      expect(renderToStaticMarkup(<QueryClientProvider client={client}><LocalTargetSplit workspace={workspace} /></QueryClientProvider>)).toContain('Create test cohort');
    });
    it('creates the cohort through the idempotent test-cohort route', async () => {
      const fetcher = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ token: 'session' })))
        .mockResolvedValueOnce(new Response(JSON.stringify({ evaluationCohortId: 'cohort-1', cohort: {} })));
      vi.stubGlobal('fetch', fetcher);
      vi.resetModules();
      const { targetSplits: api } = await import('../api/targetSplits');
      await expect(api.createTestCohort('project', 'target/1')).resolves.toMatchObject({ evaluationCohortId: 'cohort-1' });
      expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project/target-splits/target%2F1/test-cohort');
      expect(fetcher.mock.calls[1][1].method).toBe('POST');
    });
  });
  it('offers draft recovery in the library without claiming it was frozen', () => {
    const html = render([], true);
    expect(html).toContain('Return to current draft');
    expect(html).toContain('Unsaved targets and splits were recovered');
    expect(html).toContain('No targets and splits yet');
  });
  it('summarizes a zero-test split without implying testing results exist', () => {
    const html = renderToStaticMarkup(<TargetSplitSummary value={{ ...record.manifest, spec: { ...spec, split: { ...spec.split, testFraction: 0 } }, summary: { ...record.manifest.summary, trainingSlides: 20, testingSlides: 0, testingPatients: 0, testingGroups: 0, testingClassCounts: {} } }} />);
    expect(html).toContain('0% testing');
    expect(html).not.toMatch(/patients|patient groups/);
    expect(html).not.toContain('Accuracy');
  });
  it('defaults new drafts to slide splitting while preserving legacy patient semantics', () => {
    expect(newTargetSplitSpec('dataset')).toMatchObject({ splitUnit: 'slide', target: { unit: 'slide' } });
    expect(targetSplitUnit({})).toBe('patient');
    const legacy = { ...spec, splitUnit: undefined };
    expect(targetSplitPartitionRequest(legacy)).not.toHaveProperty('splitUnit');
    const html = renderToStaticMarkup(<TargetSplitSummary value={{ ...record.manifest, spec: legacy }} />);
    expect(html).toContain('<dt>Split unit</dt><dd>Patient</dd>');
    expect(html).toContain('Known patients stay in one set');
  });
  it('switches the full output definition without discarding filters or custom mappings', () => {
    const original = { ...spec, testTarget: { ...spec.target, field: 'external', labels: { L: 'low', H: 'high' } } };
    const changed = targetSplitWithUnit(original, 'patient');
    expect(changed.splitUnit).toBe('patient');
    expect(changed.target.unit).toBe('patient');
    expect(changed.testTarget?.unit).toBe('patient');
    expect(changed.testTarget?.labels).toEqual(original.testTarget.labels);
    expect(changed.split).toEqual(original.split);
    expect(targetSplitWithUnit({ ...original, testTarget: null }, 'slide').testTarget).toBeNull();
    expect(targetSplitPartitionRequest(targetSplitWithUnit(changed, 'slide')).splitUnit).toBe('slide');
  });
  it('clears incompatible assignments when changing the train/test selection method', () => {
    const imported = { ...spec.split, method: 'imported' as const, partitionField: 'partition', trainValues: ['dev'], testValues: ['heldout'] };
    const rules = targetSplitMethod(imported, 'rules');
    expect(rules.partitionField).toBeUndefined();
    expect(rules.trainValues).toEqual([]);
    expect(rules.testValues).toEqual([]);
    const random = targetSplitMethod({ ...rules, testRules: [{ field: 'site', op: 'eq', value: 'external' }] }, 'random');
    expect(random.testRules).toEqual([]);
    expect(random.seed).toBe(spec.split.seed);
  });
  it('lets testing take every eligible slide outside training, restoring set-aside testing conditions', () => {
    const train = [{ field: 'Requested_Split', op: 'eq' as const, value: 'train' }];
    const test = [{ field: 'Requested_Split', op: 'eq' as const, value: 'test' }];
    const rules = { ...spec.split, method: 'rules' as const, trainRules: train, testRules: test };
    const remaining = targetSplitTestingRemainder(rules, true);
    expect(remaining).toMatchObject({ testRemaining: true, testRules: [], trainRules: train });
    expect(targetSplitTestingRemainder(remaining, false, test)).toMatchObject({ testRemaining: undefined, testRules: test });
    expect(JSON.parse(JSON.stringify(targetSplitTestingRemainder(remaining, false)))).not.toHaveProperty('testRemaining');
    expect(targetSplitMethod(remaining, 'random').testRemaining).toBeUndefined();
    expect(targetSplitPartitionRequest({ ...spec, split: remaining }).split.testRemaining).toBe(true);
  });
  it('offers the testing remainder only once training has its own conditions', () => {
    const client = new QueryClient();
    const context = { project: 'project', datasetId: 'dataset', dictionary: [] };
    const html = (split: typeof spec.split, unit: 'slide' | 'patient' = 'slide') => renderToStaticMarkup(<QueryClientProvider client={client}><TargetSplitTestingRules split={split} unit={unit} columns={['Requested_Split']} fieldContext={context} onChange={() => {}} /></QueryClientProvider>);
    const rules = { ...spec.split, method: 'rules' as const };
    const withoutTraining = html(rules);
    expect(withoutTraining).toContain('Use all eligible slides outside the training set');
    expect(withoutTraining).toMatch(/<input type="checkbox" disabled=""/);
    expect(withoutTraining).toContain('Add training conditions first');
    expect(withoutTraining).toContain('Choose the slides reserved for evaluation or pure inference.');
    const trained = { ...rules, trainRules: [{ field: 'Requested_Split', op: 'eq' as const, value: 'train' }] };
    const available = html(trained);
    expect(available).not.toContain('disabled=""');
    expect(available).not.toContain('checked=""');
    expect(available).not.toContain('Add training conditions first');
    const remaining = html({ ...trained, testRemaining: true });
    expect(remaining).toContain('checked=""');
    expect(remaining).toContain('Testing uses every eligible slide that the training conditions do not select');
    expect(remaining).not.toContain('Choose the slides reserved for evaluation or pure inference.');
    expect(html({ ...trained, testRemaining: true }, 'patient')).toContain('Use all eligible patient groups outside the training set');
    client.clear();
  });
  it('supports constructing partitions before targets and does not send incomplete labels for live counts', () => {
    const initial = newTargetSplitSpec('saved-dataset');
    expect(TARGET_SPLIT_STEPS).toEqual(['Dataset & cohort', 'Training & Testing split', 'Prediction Targets', 'Review & Freeze']);
    const request = targetSplitPartitionRequest(initial);
    expect(request.datasetId).toBe('saved-dataset');
    expect(request.split).toEqual(initial.split);
    expect(request).not.toHaveProperty('target');
    expect(request).not.toHaveProperty('testTarget');
    expect(targetDefinitionReady(initial.target)).toBe(false);
  });
  it('keeps partition counts and value queries tied to all filters and both target fields', () => {
    const current = { ...spec, eligibility: [{ field: 'site', op: 'eq' as const, value: 'A' }], split: { ...spec.split, method: 'rules' as const, testRules: [{ field: 'source', op: 'eq' as const, value: 'external' }] }, testTarget: { ...spec.target, field: 'external_grade', labels: { negative: 'low', positive: 'high' } } };
    const request = targetSplitPartitionRequest(current);
    expect(request.targetFields).toEqual({ train: 'grade', test: 'external_grade' });
    expect(request.testTarget?.labels).toEqual({ negative: 'low', positive: 'high' });
    expect(request.eligibility).toEqual(current.eligibility);
    expect(request.split.testRules).toEqual(current.split.testRules);
    expect(targetSplitPartitionRequest(current, undefined, true)).not.toHaveProperty('target');
    expect(targetSplitPartitionRequest(current, undefined, true)).not.toHaveProperty('testTarget');
  });
  it('preserves absent testing labels without asking the preview to read a testing field', () => {
    const inference = { ...spec, testTarget: null };
    const request = targetSplitPartitionRequest(inference);
    expect(request.testTarget).toBeNull();
    expect(request.targetFields?.test).toBeUndefined();
    const html = renderToStaticMarkup(<TargetSplitSummary value={{ ...record.manifest, spec: inference, memberships: [{ slideId: 'unlabeled-slide', patientId: 'test-patient', patientIdSource: 'source', partition: 'test', label: null }] }} />);
    expect(html).toContain('None · Pure inference');
    expect(html).toContain('Not labeled');
    expect(html).toContain('retained without reading labels');
    expect(html).not.toContain('<td>0</td>');
  });
  it('explains conflicting mappings for the same target field before review', () => {
    const conflicting = { ...spec, testTarget: { ...spec.target, labels: { '1': 'high', '2': 'low' } } };
    expect(targetSplitTestingIssue(conflicting)).toContain('same target field must preserve training label mappings');
    expect(targetSplitPartitionRequest(conflicting)).not.toHaveProperty('target');
    expect(targetSplitTestingIssue({ ...conflicting, testTarget: { ...conflicting.testTarget, field: 'external_grade' } })).toBeNull();
  });
  it('keeps the testing output definition synchronized without overwriting its source mapping', () => {
    const testing = { ...spec.target, field: 'external_grade', labels: { negative: 'low', positive: 'high' } };
    const changed = targetSplitTrainingTarget({ ...spec, testTarget: testing }, { ...spec.target, unit: 'slide', positiveClass: 'low' });
    expect(changed.testTarget?.unit).toBe('slide');
    expect(changed.testTarget?.positiveClass).toBe('low');
    expect(changed.testTarget?.field).toBe('external_grade');
    expect(changed.testTarget?.labels).toEqual(testing.labels);
    expect(targetSplitTrainingTarget({ ...spec, testTarget: null }, spec.target)).not.toHaveProperty('testTarget');
  });
  it('keeps fixed dataset selection separate from fold and feature settings', () => {
    const initial = newTargetSplitSpec('saved-dataset', 123);
    expect(initial.split).toEqual({ method: 'random', testFraction: 0.2, seed: 123, stratify: false, trainRules: [], testRules: [], trainValues: [], testValues: [] });
    expect(initial).not.toHaveProperty('featureBundleId');
    expect(initial.split).not.toHaveProperty('folds');
    expect(targetSplitSetupLink(record)).toBe('#experimental-setup?dataset=dataset&targetSplit=target-split-a');
  });
});
