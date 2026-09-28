import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { experiments, type ModelExperiment, type SetupInput } from '../api/experiments';
import { bundles } from '../api/bundles';
import { targetSplits, targetSplitUnit } from '../api/targetSplits';
import { type ProtocolSpec } from '../api/scientific';
import { ApiError } from '../api/client';
import { newDevelopmentSplit } from '../lib/protocol';
import { splitSeedsError } from '../lib/split';
import { sameJSON } from '../lib/json';
import { preparationLink, type PreparationContext } from '../lib/preparationRoute';
import { readSessionDraft, sessionDraftKey, useSessionDraftBackup, writeSessionDraft } from '../lib/sessionDraft';
import { useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import { versionLabelText } from '../lib/versionLabels';
import { DatasetSelect, Findings, useDatasets } from './ScientificUI';
import { SplitStrategy } from './SplitStrategy';
import { Badge, ErrorNotice, Panel } from './ui';
import { BatchPlanSettings } from './DevelopmentBatches';
import { StageContinueButton } from './StageActions';

type Design = Omit<SetupInput, 'expectedRevision'>;
interface DesignDraft { design: Design; base: Design; seedsText: string }
const isDesign = (value: unknown): value is Design => Boolean(value && typeof value === 'object' && 'datasetId' in value && typeof value.datasetId === 'string' && 'targetSplitId' in value && typeof value.targetSplitId === 'string' && 'featureBundleId' in value && typeof value.featureBundleId === 'string' && 'trainingSplit' in value && value.trainingSplit && typeof value.trainingSplit === 'object' && 'seeds' in value.trainingSplit && Array.isArray(value.trainingSplit.seeds));
const isDraft = (value: unknown): value is DesignDraft => Boolean(value && typeof value === 'object' && 'design' in value && isDesign(value.design) && 'base' in value && isDesign(value.base) && 'seedsText' in value && typeof value.seedsText === 'string');
export const setupDesign = (record: ModelExperiment, context: PreparationContext = {}): Design => ({
  datasetId: record.setupDesign?.datasetId ?? context.datasetId ?? '',
  targetSplitId: record.setupDesign?.targetSplitId ?? context.targetSplitId ?? '',
  featureBundleId: record.inputs?.featureBundleId ?? context.bundleId ?? '',
  loadingPolicy: record.inputs?.loadingPolicy ?? 'auto',
  packArtifactId: record.inputs?.packArtifactId ?? null,
  trainingSplit: record.setupDesign?.trainingSplit ?? newDevelopmentSplit(),
});

/** Only the frozen training population is split into folds and early-stop validation. */
export function ExperimentalSetupInputs({ project, record, context, readOnly, onVerified, onDirtyChange, onBusyChange }: {
  project: string; record: ModelExperiment; context: PreparationContext; readOnly: boolean;
  onVerified: (record: ModelExperiment) => void; onDirtyChange: (value: boolean) => void; onBusyChange: (value: boolean) => void;
}) {
  const client = useQueryClient();
  const datasets = useDatasets(project);
  const partitions = useQuery({ queryKey: ['scientific', project, 'configurations', 'target-split'], queryFn: () => targetSplits.list(project) });
  const features = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const recoveryKey = sessionDraftKey(project, record.id, 'setup-design');
  const [recovered] = useState(() => readSessionDraft(recoveryKey, isDraft));
  const [base, setBase] = useState(recovered?.base ?? setupDesign(record, context));
  const [design, setDesign] = useState(recovered?.design ?? setupDesign(record, context));
  const [seedsText, setSeedsText] = useState(recovered?.seedsText ?? design.trainingSplit.seeds.join(', '));
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const source = setupDesign(record, context);
  const stale = !sameJSON(base, source);
  const dirty = !readOnly && (!record.setupDesign || !sameJSON(design, source) || seedsText !== design.trainingSplit.seeds.join(', '));
  const recovery = useSessionDraftBackup(recoveryKey, !readOnly && (dirty || stale) ? { design, base, seedsText } : null, isDraft);
  useWorkspaceNavigationGuard(busy ? 'Setup input verification is still pending.' : dirty && recovery.error ? recovery.error : null);
  useEffect(() => { onDirtyChange(dirty || stale); }, [dirty, stale, onDirtyChange]);
  useEffect(() => { onBusyChange(busy); }, [busy, onBusyChange]);
  useEffect(() => () => { onDirtyChange(false); onBusyChange(false); }, [onDirtyChange, onBusyChange]);
  const current = readOnly ? source : design;
  const dataset = datasets.data?.datasets.find((item) => item.id === current.datasetId);
  const targetSplit = partitions.data?.configurations.find((item) => item.id === current.targetSplitId);
  const bundle = features.data?.items.find((item) => item.id === current.featureBundleId);
  const selectedPacks = bundle?.manifest.packs ?? [];
  const seedsError = splitSeedsError(seedsText);
  const validSeeds = !seedsError;
  function edit(update: Partial<Design>) { setDesign((value) => ({ ...value, ...update })); setError(null); }
  function split(update: Partial<ProtocolSpec['split']>) { edit({ trainingSplit: { ...design.trainingSplit, ...update } }); }
  async function verify() {
    if (inFlight.current || readOnly || stale || !validSeeds || design.trainingSplit.mode !== 'kfold') return;
    inFlight.current = true; setBusy(true); setError(null);
    try {
      const result = await experiments.setupInputs(project, record.id, { ...design, expectedRevision: record.revision });
      const saved = setupDesign(result, context);
      setBase(saved); setDesign(saved); setSeedsText(saved.trainingSplit.seeds.join(', '));
      client.setQueryData(['model-experiment', project, record.id], result);
      await client.invalidateQueries({ queryKey: ['scientific', project, 'configurations', 'protocol'] });
      onVerified(result);
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Setup inputs could not be verified.')); }
    finally { inFlight.current = false; setBusy(false); }
  }
  return <>
    <ErrorNotice error={error ?? datasets.error ?? partitions.error ?? features.error} />
    {stale && !readOnly ? <p className="callout" role="alert">Saved setup inputs changed. <button type="button" className="text-button" onClick={() => { setBase(source); setDesign(source); setSeedsText(source.trainingSplit.seeds.join(', ')); }}>Reload saved design</button></p> : null}
    {recovered && dirty ? <p className="callout">Recovered your unfinished setup input changes.</p> : null}
    <fieldset className="science-fieldset" disabled={readOnly || busy || stale}>
      <Panel title="1. Dataset, features, targets and splits" subtitle="Choose the saved training/testing population and the features to use for training.">
        <div className="science-grid-two">
          <DatasetSelect versions={datasets.data?.datasets ?? []} value={current.datasetId} onChange={(datasetId) => edit({ datasetId, targetSplitId: '' })} />
          <label className="label">Targets &amp; splits<select className="field" value={current.targetSplitId} onChange={(event) => edit({ targetSplitId: event.target.value })}>
            <option value="">Choose frozen targets and splits</option>
            {current.targetSplitId && !targetSplit ? <option value={current.targetSplitId} disabled>{readOnly ? `Saved targets and splits · ${current.targetSplitId}` : 'Selected targets and splits unavailable'}</option> : null}
            {(partitions.data?.configurations ?? []).filter((item) => item.manifest.spec.datasetId === current.datasetId).map((item) => <option key={item.id} value={item.id}>{versionLabelText(item, 'Targets & splits')}</option>)}
          </select></label>
          <label className="label">Feature bundle<select className="field" value={current.featureBundleId} onChange={(event) => edit({ featureBundleId: event.target.value, packArtifactId: null })}>
            <option value="">Choose verified features</option>
            {current.featureBundleId && !bundle ? <option value={current.featureBundleId} disabled>{readOnly ? `Saved bundle · ${current.featureBundleId}` : 'Selected bundle unavailable'}</option> : null}
            {(features.data?.items ?? []).map((item) => <option key={item.id} value={item.id} disabled={!item.current || item.findings.some((finding) => finding.severity === 'error')}>{versionLabelText(item, 'Features')}{!item.current ? ' · needs verification' : ''}</option>)}
          </select></label>
          <label className="label">Feature loading<select className="field" value={current.loadingPolicy} onChange={(event) => edit({ loadingPolicy: event.target.value as Design['loadingPolicy'], packArtifactId: null })}><option value="auto">Auto</option><option value="native">Original feature files</option><option value="mmap" disabled={!selectedPacks.length}>Packed mmap</option></select></label>
          {current.loadingPolicy !== 'native' && selectedPacks.length ? <label className="label">Feature pack<select className="field" value={current.packArtifactId ?? ''} onChange={(event) => edit({ packArtifactId: event.target.value || null })}><option value="">Choose automatically when unambiguous</option>{selectedPacks.map((pack) => <option key={pack.id} value={pack.id}>{pack.outputDtype} · {pack.outputPath}</option>)}</select></label> : null}
        </div>
        {targetSplit ? <p className="callout"><strong>{targetSplit.manifest.spec.target.field}</strong> · {targetSplitUnit(targetSplit.manifest.spec) === 'slide' ? 'Slide split' : 'Patient split'} · {targetSplit.manifest.spec.target.classes.join(' / ')} · {targetSplit.manifest.summary.trainingSlides} training slides · {targetSplit.manifest.summary.testingSlides} {targetSplit.manifest.spec.testTarget === null ? 'inference-only testing slides (no labels or evaluation metrics)' : 'reserved testing slides'}.</p> : readOnly ? <p className="muted">Saved input identities and training settings are retained. Exact specifications remain available in the input history below.</p> : <p className="muted">Create a dataset and <a href="#cohort">Targets &amp; splits</a>, and prepare <a href="#features">Slide features</a> before reviewing this setup.</p>}
        {bundle?.findings.length ? <Findings findings={bundle.findings} /> : null}
      </Panel>
      <Panel title="2. Training design" subtitle="Folds, validation and model selection use training records only. The testing set remains reserved for evaluation or inference.">
        <label className="label">Early-stop validation (% of fitting data)<input className="field" type="number" min="1" max="90" value={Number(((current.trainingSplit.validationFraction ?? 0.15) * 100).toFixed(6))} onChange={(event) => split({ validationFraction: Number(event.target.value) / 100 })} /></label>
        <SplitStrategy splitUnit={targetSplit ? targetSplitUnit(targetSplit.manifest.spec) : record.setupDesign ? record.setupDesign.splitUnit ?? 'patient' : 'unknown'} supportedModes={['kfold']} split={current.trainingSplit} onChange={split} seedsText={readOnly ? current.trainingSplit.seeds.join(', ') : seedsText} seedsError={readOnly ? '' : seedsError} onSeedsChange={(value) => { setSeedsText(value); if (value.split(',').every((seed) => /^\d+$/.test(seed.trim()))) split({ seeds: value.split(',').map(Number) }); }} fieldContext={{ project, datasetId: current.datasetId, dictionary: dataset?.manifest.dictionary ?? [] }} />
      </Panel>
    </fieldset>
    {!readOnly ? <Panel title="Check setup inputs" subtitle="Verify training membership, feature coverage, and fold feasibility before adding hyperparameters.">
      <p>Feature bundles can originate from another dataset. Every training slide must be covered. Testing slides are excluded from all folds and validation sets.</p>
      <StageContinueButton disabled={busy || stale || !dataset || !targetSplit || !bundle || !validSeeds || current.trainingSplit.mode !== 'kfold' || targetSplit.manifest.spec.datasetId !== current.datasetId || !bundle.current || (current.trainingSplit.validationFraction ?? 0) <= 0 || (current.trainingSplit.validationFraction ?? 1) >= 1} onClick={() => void verify()}>{busy ? 'Checking setup…' : 'Check & continue to hyperparameters'}</StageContinueButton>
    </Panel> : null}
  </>;
}

export function FreezeSetupControl({ project, record, disabledReason, onFrozen, onBusyChange }: { project: string; record: ModelExperiment; disabledReason: string | null; onFrozen: (record: ModelExperiment) => void; onBusyChange?: (busy: boolean) => void }) {
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const recoveryKey = sessionDraftKey(project, record.id, 'freeze-setup');
  const [pending, setPending] = useState(() => readSessionDraft(recoveryKey, (value): value is { expectedRevision: number; operationId: string } => Boolean(value && typeof value === 'object' && 'expectedRevision' in value && Number.isSafeInteger(value.expectedRevision) && 'operationId' in value && typeof value.operationId === 'string')));
  useSessionDraftBackup(recoveryKey, record.frozenSetupId ? null : pending);
  useWorkspaceNavigationGuard(busy ? 'Setup freezing is still pending.' : null);
  useEffect(() => { onBusyChange?.(busy); return () => onBusyChange?.(false); }, [busy, onBusyChange]);
  async function freeze() {
    if (busy || inFlight.current || (disabledReason && !pending)) return;
    inFlight.current = true; setBusy(true); setError(null);
    const input = pending ?? { expectedRevision: record.revision, operationId: crypto.randomUUID() };
    setPending(input);
    writeSessionDraft(recoveryKey, input);
    try { const saved = await experiments.freezeSetup(project, record.id, input); setPending(null); writeSessionDraft(recoveryKey, null); onFrozen(saved); }
    catch (reason) { if (reason instanceof ApiError && reason.status < 500 && reason.status !== 408) { setPending(null); writeSessionDraft(recoveryKey, null); } setError(reason instanceof Error ? reason : new Error('The setup could not be frozen.')); }
    finally { inFlight.current = false; setBusy(false); }
  }
  return <Panel title={record.frozenSetupId ? 'Frozen experimental setup' : 'Review & freeze setup'} subtitle="Freezing saves this complete design. Training starts separately in Experiments.">
    {record.frozenSetupId ? <><Badge tone="green">Frozen</Badge><p>This setup is ready to run. Its inputs, training design, hyperparameters and predictor choices are fixed.</p><StageContinueButton href={preparationLink('experiments', {}, { experiment: record.id })}>Continue to Experiments</StageContinueButton></> : <>
      <p><strong>{record.name}</strong> · {record.batchPlans?.length ?? 0} saved training batches. Review your inputs, validation design, configurations, training seeds and predictor settings before freezing.</p>
      {record.setupDesign ? <dl className="science-summary"><div><dt>Dataset</dt><dd>{record.setupDesign.datasetId}</dd></div><div><dt>Targets &amp; splits</dt><dd>{record.setupDesign.targetSplitId}</dd></div><div><dt>Feature bundle</dt><dd>{record.inputs?.featureBundleId}</dd></div><div><dt>Split unit</dt><dd>{record.setupDesign.splitUnit === 'slide' ? record.setupDesign.trainingSplit.groupByPatient ? 'Slide labels · cases kept together in folds' : 'Slide' : 'Patient'}</dd></div><div><dt>Training design</dt><dd>{record.setupDesign.trainingSplit.folds} folds · {record.setupDesign.trainingSplit.seeds.length} split seeds · {Number(((record.setupDesign.trainingSplit.validationFraction ?? 0.15) * 100).toFixed(6))}% early-stop validation</dd></div></dl> : null}
      {record.batchPlans?.map((plan) => <details key={plan.id}><summary>{plan.spec.batchName} · {plan.spec.trainingSeeds.length} training seeds</summary><BatchPlanSettings spec={plan.spec} fallbackPredictorPolicy={record.predictorPolicy ?? undefined} /></details>)}
      {disabledReason ? <p className="callout">{disabledReason}</p> : null}
      <ErrorNotice error={error} />
      <StageContinueButton disabled={busy || (Boolean(disabledReason) && !pending)} onClick={() => void freeze()}>{busy ? 'Freezing setup…' : pending ? 'Retry setup freeze' : 'Freeze experimental setup'}</StageContinueButton>
    </>}
  </Panel>;
}
