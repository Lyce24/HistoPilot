import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { DatasetVersion } from '../api/scientific';
import LocalDataset from './LocalDataset';
import LocalProtocol, { TabularPredictorSelection } from './LocalProtocol';

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
afterEach(() => vi.unstubAllGlobals());

function render(page: 'data' | 'targets', datasets: DatasetVersion[] = [dataset], newImport = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets });
  client.setQueryData(['scientific', 'project', 'drafts'], { drafts: [] });
  client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
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
    const html = render('targets', [dataset, older]);
    expect(html).toContain('<option value="older" selected="">');
    expect(html).not.toContain('<option value="dataset" selected="">');
    expect(html).toContain('Dataset saved. Define the target and development splits for this dataset below.');
  });

  it('shows a missing linked dataset honestly instead of displaying a different dataset', () => {
    vi.stubGlobal('window', { location: { hash: '#cohort?dataset=missing' } });
    const html = render('targets');
    expect(html).toContain('<option value="missing" disabled="" selected="">Selected dataset unavailable');
    expect(html).toContain('Choose a frozen dataset to continue.');
  });
  it('opens development data first and keeps target, split and review controls in their own steps', () => {
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
    const html = render('targets', []);
    expect(html).toContain('Choose a frozen dataset to continue.');
    expect(html).toMatch(/<button type="button" class="btn btn-primary" disabled="">Continue to target/);
    expect(html).toContain('No imported dataset yet');
  });

  it('keeps optional filters, source bindings and minimum constraints collapsed', () => {
    const html = render('targets');
    expect(html).toContain('<details class="setup-details"><summary>Additional eligibility filters');
    expect(html).toContain('<details class="setup-details"><summary>Optional feature reference');
    expect(html).toContain('<details class="setup-details"><summary>Advanced minimum set sizes');
    expect(html).toContain('Development plan to review');
    expect(html).not.toContain('Execution unavailable');
  });

  it('offers direct next actions for a frozen dataset and keeps its version tools secondary', () => {
    const html = render('data');
    expect(html).toContain('href="#cohort?dataset=dataset"');
    expect(html).toContain('href="#features?dataset=dataset"');
    expect(html).toContain('<details class="setup-details"><summary>Edit version label and note</summary>');
    expect(html).toContain('<details class="setup-details"><summary>Open a saved dataset or resume an import');
    expect(html).not.toContain('aria-label="Module workflow"');
  });

  it('starts imports at files and preserves patient-table capability behind explicit disclosure', () => {
    const html = render('data', [], true);
    expect(html).toContain('aria-label="Choose dataset files"');
    expect(html).toContain('class="dataset-section dataset-mapping-section" hidden=""');
    expect(html).toContain('<details class="setup-details"><summary>Add a separate patient table (optional)</summary>');
    expect(html).toMatch(/<button type="button" class="btn btn-primary" disabled="">Preview dataset/);
  });
});
