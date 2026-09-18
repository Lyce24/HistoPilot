import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { ProtocolPreview, ProtocolSpec } from '../api/scientific';
import { newDevelopmentSplit } from '../lib/protocol';
import { FrozenProtocolSummary } from './LocalProtocol';

const spec: ProtocolSpec = {
  datasetId: 'dataset', predictors: [], eligibility: [], split: newDevelopmentSplit(),
  constraints: { minPatientsPerClass: 1, minPatientsPerPartition: 1 },
  target: { field: 'who2022_binary', task: 'binary_classification', unit: 'slide', classes: ['low_grade', 'high_grade'], labels: { '0': 'low_grade', '1': 'high_grade' }, positiveClass: 'high_grade', missing: 'block', unmapped: 'exclude' },
};
const summary: ProtocolPreview['summary'] = {
  datasetId: 'dataset', totalSlides: 138, eligibleSlides: 62, includedSlides: 62,
  includedPatients: 0, includedGroups: 62, fallbackSlideCount: 62, excludedSlides: 76,
  classCounts: { low_grade: 20, high_grade: 42 }, patientClassCounts: {}, grouping: 'Patient_ID', algorithm: 'grouped', splitVersion: 4,
};

describe('frozen protocol scientific summary', () => {
  it('exposes the exact target, unit, class mapping and positive class without opening saved JSON', () => {
    const html = renderToStaticMarkup(<FrozenProtocolSummary spec={spec} summary={summary} dictionary={[{ key: 'who2022_binary', sourceColumn: 'WHO 2022', owner: 'slide', type: 'categorical' }]} />);
    expect(html).toContain('who2022_binary · Source column: WHO 2022');
    expect(html).toContain('Binary classification');
    expect(html).toContain('Slide-level · One prediction per slide');
    expect(html).toContain('<dt>Positive class</dt><dd>high_grade</dd>');
    expect(html).toContain('0 → low_grade; 1 → high_grade');
    expect(html).toContain('<dt>Missing reference labels</dt><dd>Block until resolved</dd>');
    expect(html).toContain('<dt>Unmapped reference labels</dt><dd>Exclude affected slides</dd>');
    expect(html).not.toContain('<details');
  });

  it('distinguishes supplied patient IDs from slide groups and retains the independence limitation', () => {
    const html = renderToStaticMarkup(<FrozenProtocolSummary spec={spec} summary={summary} />);
    expect(html).toContain('Supplied patient IDs');
    expect(html).toContain('Acknowledged slide / case groups');
    expect(html).toContain('Patient independence cannot be verified');
    expect(html).not.toContain('Verified patients');
    const legacy = renderToStaticMarkup(<FrozenProtocolSummary spec={spec} summary={{ ...summary, fallbackSlideCount: undefined, includedGroups: undefined }} />);
    expect(legacy).not.toContain('Acknowledged slide / case groups');
  });

  it('does not suggest a positive class for multiclass targets or change stored settings', () => {
    const value = { ...spec, target: { ...spec.target, task: 'multiclass_classification' as const, unit: 'patient' as const, positiveClass: undefined } };
    const before = JSON.stringify(value);
    const html = renderToStaticMarkup(<FrozenProtocolSummary spec={value} summary={{ ...summary, includedPatients: 31, includedGroups: 31, fallbackSlideCount: 0 }} />);
    expect(html).toContain('Patient-level · One prediction per patient');
    expect(html).not.toContain('Positive class');
    expect(html).not.toContain('Patient independence cannot be verified');
    expect(JSON.stringify(value)).toBe(before);
  });
});
