import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { DatasetVersion, ScientificDraft } from '../api/scientific';
import LocalDataset from './LocalDataset';
import { editorRecoveryKey } from '../lib/editorRecovery';

// Render explicit page snapshots with real React hooks. Initial navigation remains the default
// in library tests; editor snapshots retain coverage of the scientific controls after opening.
const pageSnapshot = vi.hoisted(() => ({ view: '' as string, step: 1, filters: null as { search: string; status: string; sort: string } | null }));
vi.mock('react', async (importOriginal) => {
  const react = await importOriginal<typeof import('react')>();
  return { ...react, useState: (initial: unknown) => react.useState(
    typeof initial === 'object' && initial && 'search' in initial && 'status' in initial && 'sort' in initial && pageSnapshot.filters
      ? pageSnapshot.filters
      : initial === 'library' && pageSnapshot.view ? pageSnapshot.view : initial === 1 ? pageSnapshot.step : initial,
  ) };
});

const workspace = {
  project: { id: 'project', name: 'Study', config: { seed: 42, folds: 5 } },
  dataset: { id: 'dataset' },
  sources: [],
} as unknown as Workspace;
const dataset = {
  id: 'dataset', projectId: 'project', createdAt: '2026-09-12T12:00:00Z',
  versionLabel: { tag: 'Study slides' },
  manifest: { dictionary: [{ key: 'diagnosis', sourceColumn: 'diagnosis', owner: 'patient', type: 'categorical' }] },
} as DatasetVersion;
afterEach(() => { vi.unstubAllGlobals(); pageSnapshot.view = ''; pageSnapshot.step = 1; pageSnapshot.filters = null; });

/** Unsaved editor input kept by the browser tab, as the running editors write it. */
function stubRecovery(kind: 'dataset', value: unknown, hash = '') {
  const entries = new Map([[editorRecoveryKey('project', kind), JSON.stringify({ version: 1, value })]]);
  vi.stubGlobal('window', {
    location: { hash },
    sessionStorage: {
      getItem: (key: string) => entries.get(key) ?? null,
      setItem: (key: string, raw: string) => entries.set(key, raw),
      removeItem: (key: string) => entries.delete(key),
    },
  });
}

function render(datasets: DatasetVersion[] = [dataset], newImport = false, records: { drafts?: ScientificDraft[] } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets });
  client.setQueryData(['scientific', 'project', 'drafts'], { drafts: records.drafts ?? [] });
  client.setQueryData(['scientific', 'project', 'configurations', 'feature'], { configurations: [] });
  client.setQueryData(['feature-bundles', 'project'], { items: [] });
  client.setQueryData(['feature-packs', 'project'], { jobs: [], artifacts: [] });
  const current = newImport ? { ...workspace, dataset: { ...workspace.dataset, id: '' } } : workspace;
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}><LocalDataset workspace={current} /></QueryClientProvider>);
  } finally { client.clear(); }
}

describe('guided preparation', () => {
  it('keeps metadata-only dataset rows enabled when starting an import', () => {
    pageSnapshot.view = 'import';
    const html = render([], true);
    expect(html).toMatch(/<input type="checkbox" checked=""\/> Keep metadata rows/);
  });
  it('does not list already frozen drafts alongside their dataset versions', () => {
    const frozenImport = { id: 'old-import', name: 'Duplicated import', status: 'frozen', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    expect(render([dataset], false, { drafts: [frozenImport] })).not.toContain('Duplicated import');
  });
  it('offers direct next actions for a frozen dataset and keeps its version tools secondary', () => {
    pageSnapshot.view = 'dataset';
    const html = render();
    expect(html).toContain('href="#cohort?dataset=dataset"');
    expect(html).toContain('href="#features?dataset=dataset"');
    expect(html).toContain('<details class="setup-details"><summary>Edit version label and note</summary>');
    expect(html).toContain('Back to datasets');
    expect(html).not.toContain('Dataset library');
    expect(html).not.toContain('aria-label="Module workflow"');
  });
  it('starts imports at files and preserves patient-table capability behind explicit disclosure', () => {
    pageSnapshot.view = 'import';
    const html = render([], true);
    expect(html).toContain('aria-label="Choose dataset files"');
    expect(html).toContain('class="dataset-section dataset-mapping-section" hidden=""');
    expect(html).toContain('<details class="setup-details"><summary>Add a separate patient table (optional)</summary>');
    expect(html).toMatch(/<button(?=[^>]*data-stage-action="continue")(?=[^>]*disabled="")[^>]*><span>Preview dataset/);
  });
  it('starts with saved datasets and import drafts even when a workspace dataset is selected', () => {
    const draft = { id: 'draft-import', name: 'Patient table import', revision: 3, status: 'editable', updatedAt: '2026-09-12T12:00:00Z', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    const unrelated = { ...draft, id: 'other', name: 'Unrelated training draft', payload: { type: 'mil-experiment', spec: {} } } as unknown as ScientificDraft;
    const html = render([dataset], false, { drafts: [draft, unrelated] });
    expect(html).not.toContain('Stage 0 · Saved records');
    expect(html).toContain('Dataset library');
    expect(html).toContain('Study slides');
    expect(html).toContain('Patient table import');
    expect(html).toContain('Open dataset');
    expect(html).toContain('Open import');
    expect(html).toContain('Create dataset');
    expect(html).not.toContain('Unrelated training draft');
    expect(html).not.toContain('Choose dataset files');
    expect(html).not.toContain('Explore your dataset');
    expect(html).not.toContain('Dataset creation steps');
  });
  it('keeps the empty dataset stage on its library until a new import is requested', () => {
    const html = render([], true);
    expect(html).toContain('No datasets or import drafts yet');
    expect(html).toContain('Create dataset');
    expect(html).not.toContain('Dataset name');
    expect(html).not.toContain('Save draft');
    expect(html).not.toContain('Preview dataset');
  });
  it('moves dataset mapping to its own active page with files hidden and import state retained', () => {
    pageSnapshot.view = 'import';
    pageSnapshot.step = 2;
    const html = render([], true);
    expect(html).toContain('data-stage-page="import-2"');
    expect(html).toContain('class="dataset-section" hidden="" tabindex="-1" aria-label="Choose dataset files"');
    expect(html).toContain('class="dataset-section dataset-mapping-section" tabindex="-1"');
    expect(html).toContain('class="stage-steps" aria-label="Dataset creation steps"');
    expect(html).toContain('value="Study dataset"');
    expect(html).not.toContain('class="dataset-steps"');
  });
  it('provides consistent search, status, sort and exact record management destinations', () => {
    const draft = { id: 'draft:import', name: 'Patient table import', revision: 3, status: 'editable', updatedAt: '2026-09-12T12:00:00Z', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    const html = render([dataset], false, { drafts: [draft] });
    expect(html).toContain('Search datasets');
    expect(html).toContain('All statuses');
    expect(html).toContain('Last updated');
    expect(html).toContain('data-record-key="dataset:dataset"');
    expect(html).toContain('data-record-key="draft:draft:import"');
    expect(html).toContain('aria-label="Open dataset Study slides"');
    expect(html).toContain('aria-label="Open import Patient table import"');
    expect(html).not.toContain('<h2');
  });
  it('searches dataset notes and IDs while combining the frozen and draft status filter', () => {
    const noted = { ...dataset, id: 'dataset:bladder', versionLabel: { ...dataset.versionLabel!, note: 'Reviewed BLADDER cohort' } };
    const draft = { id: 'draft-import', name: 'Bladder mapping', revision: 1, status: 'editable', updatedAt: '2026-09-12T12:00:00Z', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    pageSnapshot.filters = { search: ' bladder ', status: 'frozen', sort: 'recent' };
    const frozen = render([noted], false, { drafts: [draft] });
    expect(frozen).toContain('Open dataset Study slides');
    expect(frozen).not.toContain('Bladder mapping');
    pageSnapshot.filters = { search: 'DRAFT-IMPORT', status: 'editable', sort: 'recent' };
    const editable = render([noted], false, { drafts: [draft] });
    expect(editable).toContain('Bladder mapping');
    expect(editable).not.toContain('Open dataset Study slides');
  });
  it('sorts datasets and imports as one list by last update, creation or name', () => {
    const older = { ...dataset, createdAt: '2026-08-01T12:00:00Z', versionLabel: { ...dataset.versionLabel!, updatedAt: '2026-09-12T14:00:00Z' } };
    const draft = { id: 'draft-import', name: 'Alpha import', revision: 1, status: 'editable', createdAt: '2026-09-01T12:00:00Z', updatedAt: '2026-09-12T13:00:00Z', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    const recent = render([older], false, { drafts: [draft] });
    expect(recent.indexOf('Open dataset Study slides')).toBeLessThan(recent.indexOf('Open import Alpha import'));
    pageSnapshot.filters = { search: '', status: 'all', sort: 'name' };
    const byName = render([older], false, { drafts: [draft] });
    expect(byName.indexOf('Open import Alpha import')).toBeLessThan(byName.indexOf('Open dataset Study slides'));
    pageSnapshot.filters = { search: '', status: 'all', sort: 'oldest' };
    const oldest = render([older], false, { drafts: [draft] });
    expect(oldest.indexOf('Open dataset Study slides')).toBeLessThan(oldest.indexOf('Open import Alpha import'));
  });
  it('distinguishes an empty filtered list from a project without saved records', () => {
    pageSnapshot.filters = { search: 'unmatched', status: 'all', sort: 'recent' };
    const html = render();
    expect(html).toContain('No matching datasets or imports');
    expect(html).not.toContain('No datasets or import drafts yet');
    expect(html).toContain('Clear filters');
    expect(html).not.toContain('Saved datasets and imports');
  });
  it('offers unsaved import input on the library after leaving the module', () => {
    stubRecovery('dataset', {
      version: 1, step: 2, draft: null, name: 'Recovered import',
      spec: { source: { path: '/data/recovered.csv' }, slideIdColumn: 'De ID', slideRoot: '/slides', recursive: true, includeMissingSlides: false, missingValues: [''], attributes: [], patientIdFallback: 'unresolved' },
    });
    const html = render([], true);
    // Every module still opens on its saved-record library.
    expect(html).toContain('data-stage-page="library-"');
    expect(html).toContain('Unsaved import input was recovered in this tab');
    expect(html).toContain('Return to current import');
  });
  it('restores the recovered import exactly when it is resumed', () => {
    stubRecovery('dataset', {
      version: 1, step: 2, draft: null, name: 'Recovered import',
      spec: { source: { path: '/data/recovered.csv' }, slideIdColumn: 'De ID', slideRoot: '/slides', recursive: true, includeMissingSlides: false, missingValues: [''], attributes: [], patientIdFallback: 'unresolved' },
    });
    pageSnapshot.view = 'import';
    const html = render([], true);
    expect(html).toContain('value="Recovered import"');
    expect(html).toContain('value="/data/recovered.csv"');
    // Column mapping needs the file read again, so the resumed import starts at the file step.
    expect(html).toContain('data-stage-page="import-1"');
    expect(html).toContain('Read file &amp; show columns');
  });
  it('ignores recovery entries that this editor could not have written', () => {
    stubRecovery('dataset', { version: 1, step: 1, draft: null, name: 'Broken', spec: 'not a specification' });
    const html = render([], true);
    expect(html).not.toContain('Recovered unsaved import input');
    expect(html).not.toContain('Broken');
    // An unusable entry leaves the saved-record library exactly as it was.
    expect(html).toContain('No datasets or import drafts yet');
  });
});
