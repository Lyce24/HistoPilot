import { useState } from 'react';
import type { FeatureBundle } from '../api/bundles';
import type { Configuration } from '../api/scientific';
import { versionLabelText, configurationVersionLabel } from '../lib/versionLabels';
import VersionLabelEditor from './VersionLabelEditor';
import { Findings } from './ScientificUI';
import { Badge, EmptyState, Icon, Panel } from './ui';

export default function FeatureBundleLibrary({ project, items, features, selectedId, onSelect, onPrepare }: {
  project: string;
  items: FeatureBundle[];
  features: Configuration[];
  selectedId: string;
  onSelect: (id: string) => void;
  onPrepare: (featureId?: string, packIds?: string[]) => void;
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const bundle = items.find((item) => item.id === selectedId) ?? items[0];
  if (!bundle) return <Panel title="Frozen feature bundles" subtitle="Keep your feature source and optional packs together for reuse.">
    <EmptyState title="No frozen bundles yet" description="Choose an existing feature source or extract features, decide whether to include a pack, then freeze the bundle." />
    <button type="button" className="btn btn-primary science-fit" onClick={() => onPrepare()}><Icon name="plus" /> Prepare a bundle</button>
  </Panel>;
  const source = features.find((item) => item.id === bundle.manifest.spec.featureSetId);
  const label = (item: FeatureBundle) => versionLabelText(item, 'Feature bundle');
  return <div className="pfm-library">
    <label className="label pfm-version-select">Frozen bundle<select className="field" value={bundle.id} onChange={(event) => onSelect(event.target.value)}>{items.map((item) => <option key={item.id} value={item.id}>{label(item)}</option>)}</select></label>
    <aside className="pfm-library-list" aria-label="Choose a frozen bundle"><div className="pfm-library-heading"><h2>Frozen bundles</h2><small>{items.length} saved</small></div>
      {items.map((item) => <button type="button" key={item.id} className={`pfm-version-card ${item.id === bundle.id ? 'is-selected' : ''}`} aria-pressed={item.id === bundle.id} onClick={() => { onSelect(item.id); setShowEvidence(false); }}>
        <strong>{label(item)}</strong><span>{item.manifest.packs.length ? `Features + ${item.manifest.packs.length} pack(s)` : 'Features only'}</span><small>{item.manifest.summary.slideCount.toLocaleString()} slides</small><Badge tone={item.current ? 'green' : 'orange'}>{item.current ? 'Verified inputs' : 'Inputs need attention'}</Badge>
      </button>)}
    </aside>
    <div className="pfm-version-detail"><Panel title={label(bundle)} subtitle="The feature source and included packs are frozen together." actions={<Badge tone="purple">Frozen bundle</Badge>}>
      <div className="stack">
        <dl className="pfm-version-facts"><div><dt>Feature source</dt><dd>{source ? configurationVersionLabel(source) : bundle.manifest.feature.id}</dd></div><div><dt>Contents</dt><dd>{bundle.manifest.packs.length ? `Features + ${bundle.manifest.packs.length} verified pack(s)` : 'Features alone'}</dd></div><div><dt>Coverage</dt><dd>{bundle.manifest.summary.slideCount.toLocaleString()} slides · {bundle.manifest.summary.patchCount.toLocaleString()} patches · {bundle.manifest.summary.dimensions} dimensions</dd></div></dl>
        {bundle.versionLabel?.note ? <p>{bundle.versionLabel.note}</p> : null}
        {bundle.manifest.packs.length ? <div className="stack"><h3>Included packs</h3>{bundle.manifest.packs.map((pack) => <div key={pack.id} className="feature-bundle-pack"><Badge>{pack.outputDtype}</Badge><span className="mono">{pack.outputPath}</span></div>)}</div> : <p>No pack is included in this bundle.</p>}
        <Findings findings={bundle.findings} />
        <div className="inline-actions"><a className="btn btn-primary" href="#experiments">Open MIL experiments <Icon name="arrow" /></a><button type="button" className="btn btn-secondary" onClick={() => onPrepare(bundle.manifest.spec.featureSetId, bundle.manifest.spec.packArtifactIds)}>Prepare another bundle</button></div>
        <p className="muted">Create another bundle to change the included packs. This bundle remains available to experiments that reference it.</p>
        <details><summary>Edit bundle name &amp; note</summary><VersionLabelEditor project={project} resourceType="configuration" resource={bundle} tagLabel="Bundle version tag" /></details>
        <details onToggle={(event) => setShowEvidence(event.currentTarget.open)}><summary>Frozen bundle evidence</summary>{showEvidence ? <pre className="code-block">{JSON.stringify(bundle.manifest, null, 2)}</pre> : null}</details>
      </div>
    </Panel></div>
  </div>;
}
