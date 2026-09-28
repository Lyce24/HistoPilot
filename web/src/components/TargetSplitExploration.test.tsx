import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { newTargetSplitSpec, targetSplitPartitionRequest, testingShareText, type TargetSplitPartitionPreview } from '../api/targetSplits';
import { TargetSplitDistributions, TargetSplitPartitionCounts, TargetSplitSelection, TargetSplitSelectionSummary, targetSplitExplorationKey } from './TargetSplitExploration';
import PredictionTargetEditor from './PredictionTargetEditor';
import { CohortStats } from './ProtocolExploration';

const target = { ...newTargetSplitSpec().target, field: 'grade', task: 'binary_classification' as const, unit: 'slide' as const, classes: ['low', 'high'], labels: { '1': 'low', '2': 'high' }, positiveClass: 'high' };
const live: TargetSplitPartitionPreview = {
  summary: { totalSlides: 16, eligibleSlides: 12, selectedSlides: 10, excludedSlides: 6, trainingSlides: 7, testingSlides: 3, trainingPatients: 3, testingPatients: 1 }, valid: true, findings: [],
  partitions: {
    train: { slides: 7, patients: 3, groups: 4, fallbackSlides: 1, unlinkedSlides: 0, target: { field: 'grade', values: [{ value: '1', slides: 4, patients: 2 }, { value: '2', slides: 3, patients: 1 }], distinctCount: 2, classCounts: { low: 4, high: 3 }, patientClassCounts: { low: 2, high: 1 }, missingSlides: 0, unmappedSlides: 0, mixedLabelPatients: 0, missingLabelPatients: 0, unmappedLabelPatients: 0, labeledPatients: 3, unlabeledPatients: 0 } },
    test: { slides: 3, patients: 1, groups: 2, fallbackSlides: 1, unlinkedSlides: 0, target: null },
  },
};

describe('target and split live exploration', () => {
  it('shows both selected populations before any target exists, separating fallback groups from patients', () => {
    const html = renderToStaticMarkup(<TargetSplitPartitionCounts value={live} />);
    expect(html).toContain('Training live selection');
    expect(html).toContain('Testing live selection');
    expect(html).toContain('7 selected · 5 outside this set');
    expect(html).toContain('3 selected · 9 outside this set');
    expect(html).toContain('aria-label="Training share of eligible slides" value="7" max="12"');
    expect(html).toContain('aria-label="Testing share of eligible slides" value="3" max="12"');
    expect(html).toContain('Training verified patients');
    expect(html).toContain('Testing verified patients');
    expect(html).toContain('Slide ID fallback groups');
    expect(html).not.toContain('metric-value-text');
    expect(html).not.toContain('target distribution');
  });
  it('shows mapped slide and verified-patient counts without inventing labels for inference', () => {
    const html = renderToStaticMarkup(<TargetSplitDistributions partitions={live.partitions} mapped unlabeledTest />);
    expect(html).toContain('grade · training slides by mapped label');
    expect(html).toContain('grade · training verified patients by mapped label');
    expect(html).not.toContain('Testing target distribution');
    expect(html).not.toContain('3 selected slides · 1 verified patients');
    expect(html).not.toContain('testing slides by mapped label');
    expect(html).toContain('Slide-ID fallback groups are not counted as patients');
  });
  it('distinguishes direct rule matches from expanded patients and assigned groups', () => {
    const stats = { totalSlides: 1, patientCount: 1, fallbackSlideCount: 0, groupCount: 1, unlinkedSlideCount: 0, sample: [] };
    const html = renderToStaticMarkup(<TargetSplitSelection label="Training" total={12} part={{ ...live.partitions.train, slides: 2, patients: 1, fallbackSlides: 0, selection: { mode: 'rules', directMatches: stats, expanded: { ...stats, totalSlides: 4 }, assigned: { ...stats, totalSlides: 2 } } }} />);
    expect(html).toContain('1 slide matches these conditions directly · 4 slides after including their eligible patient groups');
    expect(html).toContain('2 selected · 10 outside this set');
    expect(html).toContain('Conflicting groups are excluded from the assigned counts');
  });
  it('names the opposite selection when a role takes the eligible remainder', () => {
    const stats = { totalSlides: 3, patientCount: 0, fallbackSlideCount: 0, groupCount: 3, unlinkedSlideCount: 0, sample: [] };
    const part = (mode: 'remaining' | 'none') => ({ slides: mode === 'none' ? 0 : 3, patients: 0, groups: 3, fallbackSlides: 0, unlinkedSlides: 0, selection: { mode, directMatches: stats, expanded: stats, assigned: stats } });
    expect(renderToStaticMarkup(<TargetSplitSelection label="Testing" splitUnit="slide" total={12} part={part('remaining')} />)).toContain('All eligible slides outside the training selection.');
    expect(renderToStaticMarkup(<TargetSplitSelection label="Training" splitUnit="slide" total={12} part={part('remaining')} />)).toContain('All eligible slides outside the testing selection.');
    expect(renderToStaticMarkup(<TargetSplitSelection label="Training" total={12} part={part('none')} />)).toContain('No training conditions: no training slides selected.');
    expect(renderToStaticMarkup(<TargetSplitSelection label="Testing" total={12} part={part('none')} />)).toContain('No testing conditions: no testing slides selected.');
  });
  it('keeps empty selection feedback visible with a valid progress range', () => {
    const html = renderToStaticMarkup(<TargetSplitSelection label="Testing" total={0} part={{ slides: 0, patients: 0, groups: 0, fallbackSlides: 0, unlinkedSlides: 0 }} />);
    expect(html).toContain('0 selected · 0 outside this set');
    expect(html).toContain('aria-label="Testing share of eligible slides" value="0" max="1"');
  });
  it('reports only split exclusions against the already filtered cohort', () => {
    const html = renderToStaticMarkup(<TargetSplitSelectionSummary value={live} />);
    expect(html).toContain('10 of 12 eligible slides assigned · 2 excluded by split conditions');
    expect(html).not.toContain('6 excluded');
    expect(html).not.toContain('Testing holds');
  });
  it('says what share of the split units a random split put in testing', () => {
    const random = { ...live, summary: { ...live.summary, testingUnits: 3, splitUnits: 10, achievedTestFraction: 0.3 } };
    expect(renderToStaticMarkup(<TargetSplitSelectionSummary value={random} splitUnit="slide" />)).toContain('Testing holds 3 of 10 slides (30%).');
    expect(testingShareText({ testingUnits: 1, splitUnits: 3, achievedTestFraction: 1 / 3 }, 'patient')).toBe('1 of 3 patient groups (33.3%)');
    expect(testingShareText({}, 'slide')).toBeNull();
  });
  it('keeps role distributions adjacent to their selected target editor', () => {
    const html = renderToStaticMarkup(<TargetSplitDistributions partitions={live.partitions} partition="train" mapped />);
    expect(html).toContain('Training target distribution');
    expect(html).not.toContain('Testing target distribution');
  });
  it('shows only slide counts and distributions when slide splitting is selected', () => {
    const html = renderToStaticMarkup(<><CohortStats splitUnit="slide" stats={{ totalSlides: 12, patientCount: 4, fallbackSlideCount: 3, unlinkedSlideCount: 2, groupCount: 7, sample: [] }} total={16} /><TargetSplitPartitionCounts value={live} splitUnit="slide" /><TargetSplitDistributions partitions={live.partitions} splitUnit="slide" mapped /><TargetSplitSelectionSummary value={live} splitUnit="slide" /></>);
    expect(html).toContain('12 included · 4 excluded by eligibility rules');
    expect(html).toContain('Training slides');
    expect(html).toContain('grade · training slides by mapped label');
    expect(html).not.toMatch(/patient|fallback|unresolved/i);
  });
  it('locks labels to slide mode without displaying patient-link warnings', () => {
    const html = renderToStaticMarkup(<PredictionTargetEditor target={target} splitUnit="slide" showFieldProfile={false} unlinkedSlideCount={4} fallbackSlideCount={2} fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [] }} labelValues={{ isPending: false, error: null }} rawValues={[]} dataLabel="selected slides" onChooseTarget={() => {}} onChange={() => {}} />);
    expect(html).toContain('Label unit<strong>Slide</strong>');
    expect(html).not.toMatch(/patient|fallback|unresolved/i);
    expect(html).not.toContain('value="patient"');
  });
  it('invalidates live keys when partition conditions, stratification or target role changes', () => {
    const initial = { ...newTargetSplitSpec('dataset'), target };
    const key = (value: typeof initial) => targetSplitExplorationKey('project', targetSplitPartitionRequest(value));
    expect(key(initial)).not.toEqual(key({ ...initial, splitUnit: 'patient' }));
    expect(key(initial)).not.toEqual(key({ ...initial, eligibility: [{ field: 'site', op: 'eq', value: 'A' }] }));
    expect(key(initial)).not.toEqual(key({ ...initial, split: { ...initial.split, testFraction: 0.4 } }));
    expect(key(initial)).not.toEqual(key({ ...initial, split: { ...initial.split, stratify: true, stratifyField: 'grade' } }));
    const rules = { ...initial, split: { ...initial.split, method: 'rules' as const, trainRules: [{ field: 'Requested_Split', op: 'eq' as const, value: 'train' }] } };
    expect(key(rules)).not.toEqual(key({ ...rules, split: { ...rules.split, testRemaining: true } }));
    expect(targetSplitExplorationKey('project', targetSplitPartitionRequest({ ...initial, testTarget: null }))).not.toEqual(key(initial));
  });
  it('reuses the target editor with a locked model output definition and no dataset-wide profile', () => {
    const html = renderToStaticMarkup(<PredictionTargetEditor target={target} classDefinitionLocked showFieldProfile={false} fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [] }} labelValues={{ data: { valueCounts: [{ value: '1', count: 3 }], valuesTruncated: false }, isPending: false, error: null }} rawValues={['1']} dataLabel="selected testing slides" onChooseTarget={() => {}} onChange={() => {}} />);
    expect(html).toContain('grade · selected testing slides');
    expect(html).toContain('Testing uses the training task, class names, label unit and positive class');
    expect(html).not.toContain('protocol-field-profile');
    expect(html).not.toContain('This target has only one non-missing value');
    expect(html).toContain('Map the source values to the training classes');
  });
});
