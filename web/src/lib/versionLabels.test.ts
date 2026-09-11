import { describe, expect, it } from 'vitest';
import type { Configuration, DatasetVersion, VersionLabel } from '../api/scientific';
import { configurationVersionLabel, datasetVersionLabel, versionLabelText } from './versionLabels';

const label = (tag: string): VersionLabel => ({
  tag, note: 'A saved note', revision: 1, createdAt: '2026-09-10', updatedAt: '2026-09-10',
});
const dataset = (name?: string): DatasetVersion => ({
  id: 'dataset-12345678', projectId: 'project', createdAt: '2026-09-10', contentHash: 'hash',
  manifest: { name }, artifacts: {},
});
const configuration = (kind: 'protocol' | 'feature', spec: unknown, layout?: unknown) => ({
  id: `${kind}-12345678`, projectId: 'project', createdAt: '2026-09-10', contentHash: 'hash',
  manifest: { kind, datasetId: 'dataset', spec, layout, summary: {} },
}) as Configuration;

describe('scientific version display names', () => {
  it('uses personal tags as the primary name while leaving canonical IDs untouched', () => {
    const version = { ...dataset('Discovery cohort'), versionLabel: label('  baseline-v1  ') };
    expect(datasetVersionLabel(version)).toBe('baseline-v1');
    expect(version.id).toBe('dataset-12345678');
  });

  it('gives unlabelled and cleared dataset versions meaningful, distinguishable names', () => {
    expect(datasetVersionLabel(dataset('Discovery cohort'))).toBe('Discovery cohort · 12345678');
    expect(datasetVersionLabel(dataset())).toBe('Dataset · 12345678');
    expect(datasetVersionLabel(dataset('   '))).toBe('Dataset · 12345678');
    expect(datasetVersionLabel({ ...dataset('Discovery cohort'), versionLabel: label('') }))
      .toBe('Discovery cohort · 12345678');
    expect(versionLabelText({ id: 'config-87654321' }, '')).toBe('Saved version · 87654321');
  });

  it('uses the inferred native encoder before a legacy project default for feature versions', () => {
    const version = configuration('feature', { encoderId: 'uni' }, { encoderId: 'uni_v1' });
    expect(configurationVersionLabel(version)).toBe('uni_v1 features · 12345678');
    expect(configurationVersionLabel(configuration('feature', { encoderId: 'resnet50' })))
      .toBe('resnet50 features · 12345678');
    expect(configurationVersionLabel(configuration('feature', {}))).toBe('Features · 12345678');
    expect(configurationVersionLabel({ ...version, versionLabel: label('tiles-20x') })).toBe('tiles-20x');
  });

  it('names legacy protocols by their target while retaining an unambiguous fallback', () => {
    expect(configurationVersionLabel(configuration('protocol', { target: { field: 'Grade' } })))
      .toBe('Grade protocol · 12345678');
    expect(configurationVersionLabel(configuration('protocol', {})))
      .toBe('Cohort / protocol · 12345678');
  });
});
