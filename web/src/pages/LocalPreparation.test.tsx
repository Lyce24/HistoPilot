import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { Configuration, DatasetVersion, ScientificDraft } from '../api/scientific';
import LocalDataset from './LocalDataset';
import LocalProtocol, { TabularPredictorSelection } from './LocalProtocol';

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

function render(page: 'data' | 'targets', datasets: DatasetVersion[] = [dataset], newImport = false, records: { drafts?: ScientificDraft[]; protocols?: Configuration[] } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets });
  client.setQueryData(['scientific', 'project', 'drafts'], { drafts: records.drafts ?? [] });
  client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: records.protocols ?? [] });
  client.setQueryData(['scientific', 'project', 'configurations', 'feature'], { configurations: [] });
  client.setQueryData(['feature-packs', 'project'], { jobs: [], artifacts: [] });
  const current = newImport ? { ...workspace, dataset: { ...workspace.dataset, id: '' } } : workspace;
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}>{page === 'data' ? <LocalDataset workspace={current} /> : <LocalProtocol workspace={current} />}</QueryClientProvider>);
  } finally { client.clear(); }
}

describe('guided preparation', () => {
  it('does not offer spreadsheet covariates that ABMIL cannot consume', () => {
    const html = renderToStaticMarkup(<TabularPredictorSelection dictionary={dataset.manifest.dictionary ?? []} selected={[]} targetField="diagnosis" onChange={() => {}} />);
    expect(html).toContain('Current ABMIL training uses slide image features only');
    expect(html).toContain('Spreadsheet model inputs · unsupported');
    expect(html).toMatch(/<input type="checkbox" disabled=""/);
    expect(html).not.toContain('Choose extra columns for the model');
  });

  it('keeps unsupported legacy choices removable even when their columns are unavailable or are the target', () => {
    const html = renderToStaticMarkup(<TabularPredictorSelection dictionary={dataset.manifest.dictionary ?? []} selected={['diagnosis', 'old-age-column']} targetField="diagnosis" onChange={() => {}} />);
    expect(html).toContain('2 selected in this draft');
    expect(html).toContain('Remove these selections before training');
    expect(html).toContain('old-age-column');
    expect(html).toContain('Column unavailable');
    expect(html.match(/<input type="checkbox" checked=""/g)).toHaveLength(2);
    expect(html).not.toContain('disabled=""');
  });

  it('keeps the saved dataset selected when it is older than the workspace default', () => {
    vi.stubGlobal('window', { location: { hash: '#cohort?dataset=older&saved=dataset' } });
    const older = { ...dataset, id: 'older', versionLabel: { ...dataset.versionLabel!, tag: 'Older study slides' } };
    pageSnapshot.view = 'editor';
    const html = render('targets', [dataset, older]);
    expect(html).toContain('<option value="older" selected="">');
    expect(html).not.toContain('<option value="dataset" selected="">');
    expect(html).toContain('Dataset saved. Define the target and development splits for this dataset below.');
  });

  it('shows a missing linked dataset honestly instead of displaying a different dataset', () => {
    vi.stubGlobal('window', { location: { hash: '#cohort?dataset=missing' } });
    pageSnapshot.view = 'editor';
    const html = render('targets');
    expect(html).toContain('<option value="missing" disabled="" selected="">Selected dataset unavailable');
    expect(html).toContain('Choose a frozen dataset to continue.');
  });
  it('opens development data first and keeps target, split and review controls in their own steps', () => {
    pageSnapshot.view = 'editor';
    const html = render('targets');
    expect(html).toContain('id="protocol-cohort" class="protocol-section" tabindex="-1"');
    expect(html).toContain('id="protocol-target" class="protocol-section" hidden=""');
    expect(html).toContain('id="protocol-split" class="protocol-section" hidden=""');
    expect(html).toContain('id="protocol-review" class="science-savebar protocol-section" hidden=""');
    const firstStep = html.slice(html.indexOf('id="protocol-cohort"'), html.indexOf('id="protocol-split"'));
    expect(firstStep).toContain('Select the development data');
    expect(firstStep).toContain('Study slides');
    expect(firstStep).not.toContain('Choose how to compare models');
    expect(firstStep).not.toContain('Final test set');
    expect(html).toContain('Development data only. Select training records here; prepare test data later in Evaluate.');
    expect(html).toMatch(/<button type="button" class="btn btn-primary">Continue to target/);
  });

  it('explains a missing frozen dataset and prevents continuing with an unresolved dataset ID', () => {
    pageSnapshot.view = 'editor';
    const html = render('targets', []);
    expect(html).toContain('Choose a frozen dataset to continue.');
    expect(html).toMatch(/<button type="button" class="btn btn-primary" disabled="">Continue to target/);
    expect(html).toContain('No imported dataset yet');
  });

  it('keeps optional filters, source bindings and minimum constraints collapsed', () => {
    pageSnapshot.view = 'editor';
    const html = render('targets');
    expect(html).toContain('<details class="setup-details"><summary>Additional eligibility filters');
    expect(html).toContain('<details class="setup-details"><summary>Optional feature reference');
    expect(html).toContain('<details class="setup-details"><summary>Advanced minimum set sizes');
    expect(html).toContain('Development plan to review');
    expect(html).not.toContain('Execution unavailable');
  });

  it('offers direct next actions for a frozen dataset and keeps its version tools secondary', () => {
    pageSnapshot.view = 'dataset';
    const html = render('data');
    expect(html).toContain('href="#cohort?dataset=dataset"');
    expect(html).toContain('href="#features?dataset=dataset"');
    expect(html).toContain('<details class="setup-details"><summary>Edit version label and note</summary>');
    expect(html).toContain('Back to datasets');
    expect(html).not.toContain('Dataset library');
    expect(html).not.toContain('aria-label="Module workflow"');
  });

  it('starts imports at files and preserves patient-table capability behind explicit disclosure', () => {
    pageSnapshot.view = 'import';
    const html = render('data', [], true);
    expect(html).toContain('aria-label="Choose dataset files"');
    expect(html).toContain('class="dataset-section dataset-mapping-section" hidden=""');
    expect(html).toContain('<details class="setup-details"><summary>Add a separate patient table (optional)</summary>');
    expect(html).toMatch(/<button type="button" class="btn btn-primary" disabled="">Preview dataset/);
  });

  it('starts with saved datasets and import drafts even when a workspace dataset is selected', () => {
    const draft = { id: 'draft-import', name: 'Patient table import', revision: 3, status: 'editable', updatedAt: '2026-09-12T12:00:00Z', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    const unrelated = { ...draft, id: 'other', name: 'Unrelated training draft', payload: { type: 'mil-experiment', spec: {} } } as unknown as ScientificDraft;
    const html = render('data', [dataset], false, { drafts: [draft, unrelated] });
    expect(html).not.toContain('Stage 0 · Saved records');
    expect(html).toContain('Dataset library');
    expect(html).toContain('Study slides');
    expect(html).toContain('Patient table import');
    expect(html).toContain('Open dataset');
    expect(html).toContain('Open import');
    expect(html).toContain('New import');
    expect(html).not.toContain('Unrelated training draft');
    expect(html).not.toContain('Choose dataset files');
    expect(html).not.toContain('Explore your dataset');
    expect(html).not.toContain('Dataset creation steps');
  });

  it('keeps the empty dataset stage on its library until a new import is requested', () => {
    const html = render('data', [], true);
    expect(html).toContain('No datasets or import drafts yet');
    expect(html).toContain('New import');
    expect(html).not.toContain('Dataset name');
    expect(html).not.toContain('Save draft');
    expect(html).not.toContain('Preview dataset');
  });

  it('lists frozen protocols and their drafts before opening any scientific controls', () => {
    const protocol = { id: 'protocol', createdAt: '2026-09-12T12:00:00Z', versionLabel: { tag: 'Development baseline' }, manifest: { kind: 'protocol', datasetId: dataset.id, spec: { target: { field: 'diagnosis' } } } } as Configuration;
    const draft = { id: 'protocol-draft', name: 'Revised development cohort', revision: 2, status: 'editable', updatedAt: '2026-09-12T12:00:00Z', payload: { type: 'analysis-protocol', spec: { datasetId: dataset.id } } } as ScientificDraft;
    const html = render('targets', [dataset], false, { protocols: [protocol], drafts: [draft] });
    expect(html).not.toContain('Stage 0 · Saved records');
    expect(html).toContain('Development baseline');
    expect(html).toContain('Revised development cohort');
    expect(html).toContain('Open protocol');
    expect(html).toContain('Open protocol draft');
    expect(html).toContain('New protocol');
    expect(html).not.toContain('Protocol name');
    expect(html).not.toContain('Protocol sections');
    expect(html).not.toContain('protocol-cohort');
    expect(html).not.toContain('Preview &amp; preflight');
  });

  it('keeps empty protocols on their library without requiring a dataset before creating a draft', () => {
    const html = render('targets', []);
    expect(html).toContain('No protocols or drafts yet');
    expect(html).toContain('New protocol');
    expect(html).not.toContain('No imported dataset yet');
    expect(html).not.toContain('Protocol name');
    expect(html).not.toContain('Save draft');
  });

  it('shows the linked dataset on the protocol library while waiting for the user to create or open a record', () => {
    vi.stubGlobal('window', { location: { hash: '#cohort?dataset=older&saved=dataset' } });
    const older = { ...dataset, id: 'older', versionLabel: { ...dataset.versionLabel!, tag: 'Older study slides' } };
    const html = render('targets', [dataset, older]);
    expect(html).toContain('New protocols will start with <strong>Older study slides</strong>');
    expect(html).toContain('Protocol library');
    expect(html).not.toContain('Protocol name');
  });

  it('moves protocol targets, splits and review into separate active pages', () => {
    pageSnapshot.view = 'editor';
    for (const [step, section] of [[2, 'protocol-target'], [3, 'protocol-split'], [4, 'protocol-review']] as const) {
      pageSnapshot.step = step;
      const html = render('targets');
      expect(html).toContain(`data-stage-page="protocol-${step}"`);
      expect(html).toMatch(new RegExp(`id="${section}" class="[^"]+" tabindex="-1"`));
      expect(html).toContain('id="protocol-cohort" class="protocol-section" hidden=""');
      expect(html).toContain('class="stage-steps" aria-label="Protocol sections"');
      expect(html).not.toContain('class="protocol-route"');
    }
  });

  it('moves dataset mapping to its own active page with files hidden and import state retained', () => {
    pageSnapshot.view = 'import';
    pageSnapshot.step = 2;
    const html = render('data', [], true);
    expect(html).toContain('data-stage-page="import-2"');
    expect(html).toContain('class="dataset-section" hidden="" tabindex="-1" aria-label="Choose dataset files"');
    expect(html).toContain('class="dataset-section dataset-mapping-section" tabindex="-1"');
    expect(html).toContain('class="stage-steps" aria-label="Dataset creation steps"');
    expect(html).toContain('value="Study dataset"');
    expect(html).not.toContain('class="dataset-steps"');
  });

  it('provides consistent search, status, sort and exact record management destinations', () => {
    const draft = { id: 'draft:import', name: 'Patient table import', revision: 3, status: 'editable', updatedAt: '2026-09-12T12:00:00Z', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    const html = render('data', [dataset], false, { drafts: [draft] });
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
    const frozen = render('data', [noted], false, { drafts: [draft] });
    expect(frozen).toContain('Open dataset Study slides');
    expect(frozen).not.toContain('Bladder mapping');
    pageSnapshot.filters = { search: 'DRAFT-IMPORT', status: 'editable', sort: 'recent' };
    const editable = render('data', [noted], false, { drafts: [draft] });
    expect(editable).toContain('Bladder mapping');
    expect(editable).not.toContain('Open dataset Study slides');
  });

  it('sorts datasets and imports as one list by last update, creation or name', () => {
    const older = { ...dataset, createdAt: '2026-08-01T12:00:00Z', versionLabel: { ...dataset.versionLabel!, updatedAt: '2026-09-12T14:00:00Z' } };
    const draft = { id: 'draft-import', name: 'Alpha import', revision: 1, status: 'editable', createdAt: '2026-09-01T12:00:00Z', updatedAt: '2026-09-12T13:00:00Z', payload: { type: 'dataset-import', spec: {} } } as ScientificDraft;
    const recent = render('data', [older], false, { drafts: [draft] });
    expect(recent.indexOf('Open dataset Study slides')).toBeLessThan(recent.indexOf('Open import Alpha import'));
    pageSnapshot.filters = { search: '', status: 'all', sort: 'name' };
    const byName = render('data', [older], false, { drafts: [draft] });
    expect(byName.indexOf('Open import Alpha import')).toBeLessThan(byName.indexOf('Open dataset Study slides'));
    pageSnapshot.filters = { search: '', status: 'all', sort: 'oldest' };
    const oldest = render('data', [older], false, { drafts: [draft] });
    expect(oldest.indexOf('Open dataset Study slides')).toBeLessThan(oldest.indexOf('Open import Alpha import'));
  });

  it('distinguishes an empty filtered list from a project without saved records', () => {
    pageSnapshot.filters = { search: 'unmatched', status: 'all', sort: 'recent' };
    const html = render('data');
    expect(html).toContain('No matching datasets or imports');
    expect(html).not.toContain('No datasets or import drafts yet');
    expect(html).toContain('Clear filters');
    expect(html).not.toContain('Saved datasets and imports');
  });

  it('searches protocol source names, prediction targets and notes without opening scientific controls', () => {
    const protocol = { id: 'protocol', createdAt: '2026-09-12T12:00:00Z', versionLabel: { tag: 'Development baseline', note: 'Approved baseline' }, manifest: { kind: 'protocol', datasetId: dataset.id, spec: { target: { field: 'diagnosis' } } } } as Configuration;
    const draft = { id: 'protocol-draft', name: 'Revised cohort', revision: 2, status: 'editable', updatedAt: '2026-09-12T13:00:00Z', payload: { type: 'analysis-protocol', spec: { datasetId: dataset.id, target: { field: 'diagnosis' } } } } as ScientificDraft;
    for (const search of ['STUDY SLIDES', 'diagnosis', 'Approved baseline']) {
      pageSnapshot.filters = { search, status: 'frozen', sort: 'recent' };
      const html = render('targets', [dataset], false, { protocols: [protocol], drafts: [draft] });
      expect(html).toContain('Open protocol Development baseline');
      expect(html).not.toContain('Revised cohort');
      expect(html).toContain('data-record-key="configuration:protocol"');
      expect(html).not.toContain('protocol-cohort');
    }
    pageSnapshot.filters = { search: 'PROTOCOL-DRAFT', status: 'editable', sort: 'recent' };
    const html = render('targets', [dataset], false, { protocols: [protocol], drafts: [draft] });
    expect(html).toContain('Open protocol draft Revised cohort');
    expect(html).not.toContain('Development baseline');
    expect(html).toContain('data-record-key="draft:protocol-draft"');
  });

  it('sorts protocol versions and drafts together and shows a clear no-match state', () => {
    const protocol = { id: 'protocol', createdAt: '2026-08-12T12:00:00Z', versionLabel: { tag: 'Alpha baseline' }, manifest: { kind: 'protocol', datasetId: dataset.id, spec: { target: { field: 'diagnosis' } } } } as Configuration;
    const draft = { id: 'protocol-draft', name: 'Zeta revised cohort', revision: 2, status: 'editable', createdAt: '2026-09-01T12:00:00Z', updatedAt: '2026-09-12T13:00:00Z', payload: { type: 'analysis-protocol', spec: { datasetId: dataset.id } } } as ScientificDraft;
    const recent = render('targets', [dataset], false, { protocols: [protocol], drafts: [draft] });
    expect(recent.indexOf('Open protocol draft Zeta revised cohort')).toBeLessThan(recent.indexOf('Open protocol Alpha baseline'));
    for (const sort of ['name', 'oldest']) {
      pageSnapshot.filters = { search: '', status: 'all', sort };
      const sorted = render('targets', [dataset], false, { protocols: [protocol], drafts: [draft] });
      expect(sorted.indexOf('Open protocol Alpha baseline')).toBeLessThan(sorted.indexOf('Open protocol draft Zeta revised cohort'));
    }
    pageSnapshot.filters = { search: 'unmatched', status: 'all', sort: 'recent' };
    const filtered = render('targets', [dataset], false, { protocols: [protocol], drafts: [draft] });
    expect(filtered).toContain('No matching protocols or drafts');
    expect(filtered).toContain('Clear filters');
    expect(filtered).not.toContain('Saved protocols and drafts');
  });

});
