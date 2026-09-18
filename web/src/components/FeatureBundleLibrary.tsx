import { useState } from 'react';
import type { FeatureBundle } from '../api/bundles';
import type { Configuration, FeatureSpec } from '../api/scientific';
import { versionLabelText, configurationVersionLabel } from '../lib/versionLabels';
import VersionLabelEditor from './VersionLabelEditor';
import { Findings } from './ScientificUI';
import { Badge, EmptyState, Icon, Panel } from './ui';
import { preparationLink, type PreparationContext } from '../lib/preparationRoute';
import { StageLibrary, StageLibraryToolbar, StageRecordManageButton } from './StageWorkflow';
import VisualQualityExplorer from './VisualQualityExplorer';

type BundleStatus = '' | 'verified' | 'attention';
type BundleSort = 'recent' | 'oldest' | 'name';
export interface FeatureBundleLibraryFilters { search: string; status: BundleStatus; sort: BundleSort }
const bundleLabel = (item: FeatureBundle) => versionLabelText(item, 'Feature bundle');
const bundleUpdatedAt = (item: FeatureBundle) => item.versionLabel?.updatedAt || item.createdAt;

export function filterFeatureBundles(items: FeatureBundle[], features: Configuration[], search: string, status: BundleStatus, sort: BundleSort) {
  const query = search.trim().toLocaleLowerCase();
  const sourceNames = new Map(features.map((item) => [item.id, configurationVersionLabel(item)]));
  return items.filter((item) => {
    if (status === 'verified' && !item.current || status === 'attention' && item.current) return false;
    return !query || [bundleLabel(item), item.id, item.versionLabel?.note, item.manifest.datasetId,
      item.manifest.spec.featureSetId, sourceNames.get(item.manifest.spec.featureSetId)]
      .filter(Boolean).join(' ').toLocaleLowerCase().includes(query);
  }).sort((a, b) => {
    const order = sort === 'name'
      ? bundleLabel(a).localeCompare(bundleLabel(b))
      : sort === 'oldest'
        ? (Date.parse(a.createdAt) || 0) - (Date.parse(b.createdAt) || 0)
        : (Date.parse(bundleUpdatedAt(b)) || 0) - (Date.parse(bundleUpdatedAt(a)) || 0);
    return order || a.id.localeCompare(b.id);
  });
}

export default function FeatureBundleLibrary({ project, items, features, selectedId, onSelect, onPrepare, onRefresh, refreshBusy = false, filters: controlledFilters, onFiltersChange, context = {} }: {
  project: string;
  items: FeatureBundle[];
  features: Configuration[];
  selectedId: string;
  onSelect: (id: string) => void;
  onPrepare: (featureId?: string, packIds?: string[]) => void;
  onRefresh?: () => void;
  refreshBusy?: boolean;
  filters?: FeatureBundleLibraryFilters;
  onFiltersChange?: (filters: FeatureBundleLibraryFilters) => void;
  context?: PreparationContext;
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [localFilters, setLocalFilters] = useState<FeatureBundleLibraryFilters>({ search: '', status: '', sort: 'recent' });
  const filters = controlledFilters ?? localFilters;
  const setFilters = onFiltersChange ?? setLocalFilters;
  const { search, status, sort } = filters;
  const bundle = items.find((item) => item.id === selectedId);
  const visible = filterFeatureBundles(items, features, search, status, sort);
  if (!bundle) return <StageLibrary project={project} title="Feature bundles">
    <StageLibraryToolbar search={search} onSearch={(search) => setFilters({ ...filters, search })} searchLabel="Search feature bundles" placeholder="Name, ID, note or feature source" count={visible.length} total={items.length}
      actions={onRefresh ? <button type="button" className="btn btn-secondary btn-small" disabled={refreshBusy} onClick={onRefresh}>Refresh</button> : undefined}
      onReset={search || status || sort !== 'recent' ? () => setFilters({ search: '', status: '', sort: 'recent' }) : undefined}>
      <label className="label">Status<select className="field" value={status} onChange={(event) => setFilters({ ...filters, status: event.target.value as BundleStatus })}><option value="">All statuses</option><option value="verified">Verified inputs</option><option value="attention">Inputs need attention</option></select></label>
      <label className="label">Sort<select className="field" value={sort} onChange={(event) => setFilters({ ...filters, sort: event.target.value as BundleSort })}><option value="recent">Last updated</option><option value="oldest">Oldest first</option><option value="name">Name A–Z</option></select></label>
    </StageLibraryToolbar>
    {visible.length ? <div className="table-wrap"><table aria-label="Saved feature bundles"><thead><tr><th scope="col">Bundle</th><th scope="col">Contents</th><th scope="col">Slides</th><th scope="col">Status</th><th scope="col">Updated</th><th scope="col">Actions</th></tr></thead><tbody>
      {visible.map((item) => <tr key={item.id}>
        <th scope="row"><button type="button" className="text-button stage-record-name" aria-label={`Open ${bundleLabel(item)}`} onClick={() => { onSelect(item.id); setShowEvidence(false); }}>{bundleLabel(item)}</button>{item.versionLabel?.note ? <small className="muted">{item.versionLabel.note}</small> : null}</th>
        <td>{item.manifest.packs.length ? `Features + ${item.manifest.packs.length} pack(s)` : 'Features only'}</td>
        <td>{item.manifest.summary.slideCount.toLocaleString()}</td>
        <td><Badge tone={item.current ? 'green' : 'orange'}>{item.current ? 'Verified inputs' : 'Inputs need attention'}</Badge></td>
        <td><time dateTime={bundleUpdatedAt(item)}>{new Date(bundleUpdatedAt(item)).toLocaleDateString()}</time></td>
        <td><StageRecordManageButton type="configuration" id={item.id} name={bundleLabel(item)} /></td>
      </tr>)}
    </tbody></table></div> : <EmptyState
      icon="features"
      title={items.length ? 'No matching bundles' : 'No feature bundles yet'}
      description={items.length ? 'Try another search or clear the filters.' : 'Create a feature bundle to choose features, check their coverage and save them for experiments.'}
      action={items.length
        ? <button type="button" className="btn btn-secondary" onClick={() => setFilters({ search: '', status: '', sort: 'recent' })}>Clear filters</button>
        : undefined}
    />}
  </StageLibrary>;
  const source = features.find((item) => item.id === bundle.manifest.spec.featureSetId);
  const slideFeatures = (source?.manifest.spec as FeatureSpec | undefined)?.featureKind === 'slide';
  return <>
    <div className="pfm-version-detail"><Panel title={bundleLabel(bundle)} subtitle="Reuse this named bundle with any dataset. Targets & splits selects their shared slides." actions={<Badge tone={bundle.current ? 'green' : 'orange'}>{bundle.current ? 'Verified inputs' : 'Inputs need attention'}</Badge>}>
      <div className="stack">
        <dl className="pfm-version-facts"><div><dt>Feature source</dt><dd>{source ? configurationVersionLabel(source) : bundle.manifest.feature.id}</dd></div><div><dt>Contents</dt><dd>{bundle.manifest.packs.length ? `Features + ${bundle.manifest.packs.length} verified pack(s)` : 'Features alone'}</dd></div><div><dt>Coverage</dt><dd>{bundle.manifest.summary.slideCount.toLocaleString()} slides · {bundle.manifest.summary.patchCount.toLocaleString()} {slideFeatures ? 'slide embeddings' : 'patches'} · {bundle.manifest.summary.dimensions} dimensions</dd></div></dl>
        {bundle.versionLabel?.note ? <p>{bundle.versionLabel.note}</p> : null}
        {bundle.manifest.packs.length ? <div className="stack"><h3>Included packs</h3>{bundle.manifest.packs.map((pack) => <div key={pack.id} className="feature-bundle-pack"><Badge>{pack.outputDtype}</Badge><span className="mono">{pack.outputPath}</span></div>)}</div> : <p>No pack is included in this bundle.</p>}
        <Findings findings={bundle.findings} />
        <div className="inline-actions"><a className="btn btn-primary" href={preparationLink('cohort', { datasetId: context.datasetId, bundleId: bundle.id, protocolId: context.protocolId })}>Use in Targets &amp; splits <Icon name="arrow" /></a><button type="button" className="btn btn-secondary" onClick={() => onPrepare(bundle.manifest.spec.featureSetId, bundle.manifest.spec.packArtifactIds)}>Prepare another bundle</button></div>
        <p className="muted">{slideFeatures ? 'This bundle holds one embedding per slide; packing and patch attention are unavailable.' : 'Create another bundle to change the included packs. This bundle remains available to experiments that reference it.'}</p>
        <details><summary>Edit bundle name &amp; note</summary><VersionLabelEditor project={project} resourceType="configuration" resource={bundle} tagLabel="Bundle version tag" /></details>
        <details onToggle={(event) => setShowEvidence(event.currentTarget.open)}><summary>Frozen bundle evidence</summary>{showEvidence ? <pre className="code-block">{JSON.stringify(bundle.manifest, null, 2)}</pre> : null}</details>
      </div>
    </Panel></div>
    {context.datasetId || bundle.manifest.datasetId ? <VisualQualityExplorer key={`${context.datasetId || bundle.manifest.datasetId}:${bundle.id}`} project={project} datasetId={(context.datasetId || bundle.manifest.datasetId)!} initialBundleId={bundle.id} /> : <p className="muted">Open a frozen dataset to review its exact slides with this feature bundle.</p>}
  </>;
}
