import { useEffect, useRef, useState } from 'react';
import { StagePage, StageSteps } from './StageWorkflow';
import { StageBackButton, StageContinueButton } from './StageActions';
import { useQueryClient } from '@tanstack/react-query';
import { bundles } from '../api/bundles';
import { ApiError } from '../api/client';
import type { FeatureBundle, FeatureBundlePreview } from '../api/bundles';
import type { Configuration, FeatureSpec, VersionLabelInput } from '../api/scientific';
import FeaturePacking from './FeaturePacking';
import FreezeVersionDialog from './FreezeVersionDialog';
import { Findings } from './ScientificUI';
import { Badge, ErrorNotice, Icon } from './ui';

export function bundleReviewInvalidated(error: unknown): error is ApiError {
  return error instanceof ApiError
    && (error.code === 'PREVIEW_STALE' || error.code === 'FEATURE_BUNDLE_INVALID');
}

export default function FeatureBundlePreparation({ project, configuration, configurations, initialPackIds = [], onSelectVersion, onFrozen, page, onPageChange, onReviewReadyChange, onBusyChange, showSteps = true }: {
  page?: 'packing' | 'review';
  onPageChange?: (page: 'packing' | 'review') => void;
  onReviewReadyChange?: (ready: boolean) => void;
  showSteps?: boolean;
  onBusyChange?: (busy: boolean) => void;
  project: string;
  configuration: Configuration;
  configurations: Configuration[];
  initialPackIds?: string[];
  onSelectVersion: (id: string) => void;
  onFrozen: (bundle: FeatureBundle) => void;
}) {
  const client = useQueryClient();
  const slideFeatures = (configuration.manifest.spec as FeatureSpec).featureKind === 'slide';
  const [packIds, setPackIds] = useState(initialPackIds);
  const [review, setReview] = useState<FeatureBundlePreview | null>(null);
  const [naming, setNaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [packingBusy, setPackingBusy] = useState(false);
  useEffect(() => { onBusyChange?.(busy || naming || packingBusy); }, [busy, naming, packingBusy, onBusyChange]);
  const [error, setError] = useState<Error | null>(null);
  const [label, setLabel] = useState<VersionLabelInput>({ tag: '', note: '' });
  const generation = useRef(0);
  const operation = useRef('');
  const [localPage, setLocalPage] = useState<'packing' | 'review'>('packing');
  const activePage = page ?? localPage;
  function goTo(next: 'packing' | 'review') { setLocalPage(next); onPageChange?.(next); }

  function changePacks(ids: string[]) {
    generation.current += 1;
    setPackIds([...new Set(ids)].sort());
    setReview(null); setError(null); setNaming(false); onReviewReadyChange?.(false);
    operation.current = '';
  }
  async function preview() {
    const token = generation.current;
    setBusy(true); setError(null); setReview(null); setNaming(false); onReviewReadyChange?.(false);
    operation.current = '';
    try {
      const result = await bundles.preview(project, { featureSetId: configuration.id, packArtifactIds: packIds });
      if (generation.current !== token) return;
      setReview(result);
      operation.current = `bundle:${crypto.randomUUID()}`;
      onReviewReadyChange?.(true); goTo('review');
    } catch (reason) {
      if (generation.current === token) setError(reason instanceof Error ? reason : new Error('Bundle review failed.'));
    } finally { setBusy(false); }
  }
  return <div className="stack">
    {showSteps ? <StageSteps label="Bundle validation steps" current={activePage} disabled={busy || naming || packingBusy} steps={[
      { id: 'packing', title: slideFeatures ? 'Validate slide embeddings' : 'Validate & choose packs', description: slideFeatures ? 'No patch packing' : 'Packing is optional' },
      { id: 'review', title: 'Review & freeze', description: 'Save the complete bundle', disabled: !review },
    ]} onChange={(step) => { if (step === 'packing' || step === 'review' && review) goTo(step); }} /> : null}
    <StagePage pageKey={activePage}>
    <div hidden={activePage !== 'packing'} className="pfm-content stack">
    <fieldset className="science-fieldset stack" disabled={busy || naming}>
    <FeaturePacking onBusyChange={setPackingBusy} project={project} configuration={configuration} configurations={configurations} onSelectVersion={onSelectVersion} selectedPackIds={packIds} onSelectedPackIdsChange={changePacks} />
    <section className="feature-bundle-freeze" aria-label="Freeze feature bundle">
      <div><h3>Continue to bundle review</h3><p>{slideFeatures ? 'Save the verified slide embeddings as one immutable input. Slide embeddings are not packed.' : 'Save the feature source and the packs included with it as one immutable input. A features-only bundle includes no pack.'}</p></div>
      <p className="muted">{slideFeatures ? 'Full validation checks one nonempty vector per slide and records its source identity.' : 'Full feature validation is required. Existing packs must match all feature values and coordinates; newly created packs must finish verification.'}</p>
      <ErrorNotice error={error} />
      <StageContinueButton className="science-fit" disabled={busy || packingBusy} onClick={() => void preview()}>{busy ? 'Checking bundle…' : 'Review bundle'}</StageContinueButton>
    </section>
    </fieldset>
    </div>
    {activePage === 'review' && review ? <section className="feature-bundle-review stack" aria-label="Bundle review">
      <StageBackButton className="science-fit" disabled={busy || naming} onClick={() => goTo('packing')}>{slideFeatures ? 'Back to validation' : 'Back to validation & packs'}</StageBackButton>
      <div className="feature-pack-section-heading"><h3>{review.summary.packCount ? `Features + ${review.summary.packCount} ${review.summary.packCount === 1 ? 'pack' : 'packs'}` : 'Features only'}</h3><Badge tone={review.canFreeze ? 'green' : 'orange'}>{review.canFreeze ? 'Verified · ready to freeze' : 'Verification required'}</Badge></div>
      <p>{review.summary.slideCount.toLocaleString()} slides · {review.summary.patchCount.toLocaleString()} {slideFeatures ? 'slide embeddings' : 'patches'} · {review.summary.dimensions ?? 'Unknown'} dimensions</p>
      {review.packs.map((pack) => <div key={pack.id} className="feature-bundle-pack"><Badge>{pack.outputDtype}</Badge><span className="mono">{pack.outputPath}</span></div>)}
      <Findings findings={review.findings} />
      <p className="muted">This saves bundle contents and verification evidence. Choose how to read these inputs in MIL experiments.</p>
      <button type="button" className="btn btn-primary science-fit" disabled={!review.canFreeze || busy} onClick={() => setNaming(true)}><Icon name="lock" /> Name &amp; freeze bundle</button>
    </section> : null}
    </StagePage>
    {naming && review?.canFreeze ? <FreezeVersionDialog kind="bundle" initialLabel={label} onLabelChange={setLabel} onClose={() => setNaming(false)} onFreeze={async (name) => {
      try {
        const result = await bundles.freeze(project, review.spec, review.previewHash, operation.current, name);
        await client.invalidateQueries({ queryKey: ['feature-bundles', project] });
        onFrozen(result);
      } catch (reason) {
        if (bundleReviewInvalidated(reason)) {
          generation.current += 1;
          operation.current = '';
          setReview(null); setNaming(false); onReviewReadyChange?.(false); goTo('packing');
          setError(new Error(`${reason.message} Review the bundle again. Your tag and note have been kept.`));
        }
        throw reason;
      }
    }}>
      <p>{review.summary.slideCount.toLocaleString()} slides · {review.summary.patchCount.toLocaleString()} {slideFeatures ? 'slide embeddings' : 'patches'}</p>
      <p>{review.summary.packCount ? `${review.summary.packCount} verified pack(s) included` : 'Features alone; no pack included'}</p>
      <p>Changing the included packs later creates another bundle.</p>
    </FreezeVersionDialog> : null}
  </div>;
}
