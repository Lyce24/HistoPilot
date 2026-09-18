import { StageBackButton, StageContinueButton } from '../components/StageActions';
import { useEffect, useRef, useState } from 'react';
import { splitModeLabel, taskLabel, unitLabel } from '../lib/labels';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { ApiError } from '../api/client';
import { type Configuration, type ProtocolSpec } from '../api/scientific';
import { experiments, experimentStage, experimentStageLabel, type ExperimentStage, type ModelExperiment } from '../api/experiments';
import { lifecycleLabel } from '../api/lifecycle';
import ExperimentRegistry, { ExperimentMetadata, newExperimentLibraryFilters } from '../components/ExperimentRegistry';
import ExperimentLifecycle from '../components/ExperimentLifecycle';
import { bundles, type FeatureBundle } from '../api/bundles';
import { mil, type LoadingPolicy, type MILExperimentPreview, type MILExperimentSpec } from '../api/mil';
import { sameJSON } from '../lib/json';
import { configurationVersionLabel, versionLabelText } from '../lib/versionLabels';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import { Findings, SavedNotice, useConfigurations } from '../components/ScientificUI';
import DevelopmentBatches, { developmentTabs, type DevelopmentTab } from '../components/DevelopmentBatches';
import { protocolBundleCompatible } from '../lib/roadmap';
import { preparationLink, usePreparationContext, type PreparationContext } from '../lib/preparationRoute';
import PreparationNotice from '../components/PreparationNotice';
import ExperimentPredictors from '../components/ExperimentPredictors';
import { StagePage, useStageLibrary } from '../components/StageWorkflow';
import ExperimentNavigation, { type ExperimentPage } from '../components/ExperimentNavigation';
import { readSessionDraft, sessionDraftKey, useSessionDraftBackup } from '../lib/sessionDraft';
import { confirmWorkspaceNavigation, rememberWorkspaceLocation, useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import { batchPredictorPolicy, includesRefit, predictorPolicyLabel, experimentPredictorCount } from '../lib/experimentPredictors';
import './LocalExperiments.css';

const initialSpec = (): MILExperimentSpec => ({
  protocolId: '', featureBundleId: '', loadingPolicy: 'auto', packArtifactId: null,
});

/** Suggest a unique compatible pair, retaining explicit prepared inputs when the other choice is ambiguous. */
export function suggestedExperimentInputs(protocols: Configuration[], featureBundles: FeatureBundle[], context: PreparationContext = {}): MILExperimentSpec {
  const pairs = protocols.filter((protocol) => (!context.protocolId || protocol.id === context.protocolId) && (!context.datasetId || protocol.manifest.datasetId === context.datasetId)).flatMap((protocol) => featureBundles
    .filter((bundle) => (!context.bundleId || bundle.id === context.bundleId) && bundle.current && !bundle.findings.some((finding) => finding.severity === 'error') && protocolBundleCompatible(protocol, bundle))
    .map((bundle) => ({ protocolId: protocol.id, featureBundleId: bundle.id })));
  if (pairs.length === 1) return { ...initialSpec(), ...pairs[0] };
  return pairs.length > 1 ? { ...initialSpec(), protocolId: context.protocolId ?? '', featureBundleId: context.bundleId ?? '' } : initialSpec();
}

const loadingOptions: { value: LoadingPolicy; title: string; description: string }[] = [
  { value: 'auto', title: 'Auto', description: 'Use the original files for a bundle without packs. Automatically use a single pack that preserves source precision. Choose explicitly when several packs are included or the pack changes precision.' },
  { value: 'native', title: 'Original files', description: 'Read each slide from its original feature file.' },
  { value: 'mmap', title: 'Packed mmap', description: 'Read selected rows from a verified pack. The operating system caches pages as memory allows; the whole pack need not fit in RAM.' },
];

export function LoadingOptions({ value, hasPacks, onChange }: {
  value: LoadingPolicy; hasPacks: boolean; onChange: (value: LoadingPolicy) => void;
}) {
  return <fieldset className="mil-loading-options">
    <legend>Feature loading</legend>
    {loadingOptions.map((option) => <label key={option.value} className={`mil-loading-option${value === option.value ? ' is-selected' : ''}`}>
      <input type="radio" name="mil-loading-policy" value={option.value} checked={value === option.value} disabled={option.value === 'mmap' && !hasPacks} onChange={() => onChange(option.value)} />
      <span><strong>{option.title}</strong><small>{option.description}</small></span>
    </label>)}
  </fieldset>;
}

interface InputDraft { spec: MILExperimentSpec; baseInputs: MILExperimentSpec | null }
const isInputSpec = (value: unknown): value is MILExperimentSpec => Boolean(value && typeof value === 'object'
  && 'protocolId' in value && typeof value.protocolId === 'string'
  && 'featureBundleId' in value && typeof value.featureBundleId === 'string'
  && 'loadingPolicy' in value && ['auto', 'native', 'mmap'].includes(String(value.loadingPolicy))
  && 'packArtifactId' in value && (value.packArtifactId === null || typeof value.packArtifactId === 'string'));
export function isInputDraft(value: unknown): value is InputDraft {
  return Boolean(value && typeof value === 'object' && 'spec' in value && isInputSpec(value.spec)
    && 'baseInputs' in value && (value.baseInputs === null || isInputSpec(value.baseInputs)));
}

const stages: { id: ExperimentStage; description: string }[] = [
  { id: 'planning', description: 'Plan inputs, batches and predictors' },
  { id: 'running', description: 'Track folds and predictor creation' },
  { id: 'finished', description: 'Review results and evaluate predictors' },
];

export function ExperimentStages({ stage }: { stage: ExperimentStage }) {
  return <ol className="experiment-stages" aria-label="Experiment stages">{stages.map((item, index) => <li key={item.id} aria-current={item.id === stage ? 'step' : undefined} className={item.id === stage ? 'current' : stages.findIndex((value) => value.id === stage) > index ? 'complete' : ''}>
    <span className="experiment-stage-number" aria-hidden="true">{index + 1}</span><span><strong>{experimentStageLabel[item.id]}</strong><small>{item.description}</small></span>
  </li>)}</ol>;
}

export function availableExperimentTab(stage: ExperimentStage, tab: DevelopmentTab): DevelopmentTab {
  if (stage === 'planning' && (tab === 'runs' || tab === 'results')) return 'batches';
  if (stage === 'running' && tab === 'results') return 'runs';
  return tab;
}

export function ExperimentSubmissionControl({ project, record, disabledReason, onSubmitted, onSubmissionPendingChange, protocol, initialReview = false }: {
  initialReview?: boolean; project: string; record: ModelExperiment; disabledReason: string | null; onSubmitted: () => void; onSubmissionPendingChange?: (busy: boolean) => void; protocol?: ProtocolSpec;
}) {
  const client = useQueryClient();
  const [reviewing, setReviewing] = useState(initialReview);
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const recoveryKey = sessionDraftKey(project, record.id, 'submission');
  const [pending, setPending] = useState(() => readSessionDraft(recoveryKey, (value): value is { operationId: string; expectedRevision: number } => Boolean(value && typeof value === 'object' && 'operationId' in value && typeof value.operationId === 'string' && 'expectedRevision' in value && Number.isSafeInteger(value.expectedRevision))));
  const stage = experimentStage(record);
  const submission = record.submission;
  const acknowledged = submission?.status === 'submitted' && !submission.error;
  useSessionDraftBackup(recoveryKey, acknowledged ? null : pending);
  useWorkspaceNavigationGuard(busy ? 'Experiment submission is still pending. Leaving now may hide its outcome.' : null);
  useEffect(() => {
    if (acknowledged) { setPending(null); setError(null); onSubmissionPendingChange?.(false); }
  }, [acknowledged, onSubmissionPendingChange]);
  const retryable = !acknowledged && Boolean(pending || submission?.retryable);
  const canSubmit = record.state === 'active' && !record.legacy && (stage === 'planning' || retryable);
  const batchCount = (record.batchPlans?.length ?? 0) + record.batches.filter((batch) => batch.state === 'active').length;
  const batchPolicies = [
    ...record.batches.filter((batch) => batch.state === 'active').map((batch) => ({ id: batch.id, name: batch.manifest.spec.batchName, policy: record.predictorPolicies?.[batch.id] ?? batchPredictorPolicy(batch.manifest.spec, record.predictorPolicy) })),
    ...(record.batchPlans ?? []).map((plan) => ({ id: plan.id, name: plan.spec.batchName, policy: batchPredictorPolicy(plan.spec, record.predictorPolicy) })),
  ];
  const count = experimentPredictorCount(record, record.predictorPolicy, protocol);
  async function submit() {
    if (inFlight.current || !canSubmit || (disabledReason && !retryable)) return;
    inFlight.current = true; setBusy(true); setError(null); onSubmissionPendingChange?.(true);
    let uncertain = false;
    const input = submission ? { operationId: submission.operationId, expectedRevision: submission.expectedRevision } : pending ?? { expectedRevision: record.revision, operationId: crypto.randomUUID() };
    setPending(input);
    try {
      const saved = await experiments.submit(project, record.id, input);
      setPending(null);
      client.setQueryData(['model-experiment', project, record.id], saved);
      if (!saved.submission?.error) { setReviewing(false); onSubmitted(); }
    } catch (reason) {
      uncertain = !(reason instanceof ApiError) || reason.status >= 500 || reason.status === 408;
      setPending(uncertain ? input : null);
      setError(reason instanceof Error ? reason : new Error('The experiment could not be submitted.'));
    } finally {
      inFlight.current = false; setBusy(false); onSubmissionPendingChange?.(uncertain);
      void client.invalidateQueries({ queryKey: ['model-experiment', project, record.id] });
      void client.invalidateQueries({ queryKey: ['model-experiments', project] });
    }
  }
  if (stage === 'finished' || record.legacy) return null;
  return <section className="experiment-submission" aria-label="Experiment submission">
    {stage === 'planning' && !initialReview ? <div className="experiment-submit-summary"><div><strong>Plan before you submit</strong><p>Verify inputs, then add as many batches as you need. Submission freezes all batches and starts training.</p></div><button type="button" className="btn btn-primary" disabled={busy || !canSubmit || Boolean(disabledReason)} aria-expanded={reviewing} onClick={() => setReviewing((value) => !value)}>Review &amp; submit</button></div> : null}
    {stage === 'planning' && disabledReason ? <p className="muted">{disabledReason}</p> : null}
    {reviewing && stage === 'planning' ? <div className="experiment-submit-review"><h2>Review &amp; submit</h2><p>{batchCount} saved {batchCount === 1 ? 'batch' : 'batches'} will be submitted together. Inputs, batch settings and predictor choices become permanently read-only. To change a submitted plan, create a new experiment using it as a template.</p><ul className="experiment-submit-batches">{batchPolicies.map((batch) => <li key={batch.id}><strong>{batch.name}</strong><span>{predictorPolicyLabel(batch.policy)}{includesRefit(batch.policy) ? ` · refit P${batch.policy.refitPercentile}` : ''}{batch.policy.method === 'skip' ? ' · cross-validation only' : ''}</span></li>)}</ul>{count ? <p><strong>{count.foldRuns} fold runs · {count.total} predictors</strong> ({count.ensembles} ensembles, {count.refits} refits).</p> : null}<p>Batches create their selected predictors automatically after their folds finish. Batches set to cross-validation only do not create predictors.</p>{!retryable ? <button type="button" className="btn btn-primary" disabled={busy || !canSubmit || Boolean(disabledReason)} onClick={() => void submit()}>{busy ? 'Submitting experiment…' : 'Freeze & submit experiment'}</button> : null}</div> : null}
    <ErrorNotice error={error} />
    {submission?.error ? <p className="callout" role="status">Submission needs attention: {submission.error.message} The saved plan remains locked.</p> : null}
    {pending && !busy && !acknowledged ? <p className="callout" role="status">The submission response was not confirmed. Retry safely continues this submission and keeps the saved plan unchanged.</p> : null}
    {retryable ? <button type="button" className="btn btn-primary" disabled={busy || !canSubmit} onClick={() => void submit()}>{busy ? 'Submitting experiment…' : 'Retry submission'}</button> : null}
  </section>;
}

export function ExperimentDetail({ workspace: w, record, initialTab, onBack, context = {} }: { workspace: Workspace; record: ModelExperiment; initialTab?: ExperimentPage; onBack: () => void; onOpen: (id: string) => void; context?: PreparationContext }) {
  const project = w.project.id;
  const protocols = useConfigurations(project, 'protocol');
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const client = useQueryClient();
  const draftKey = sessionDraftKey(project, record.id, 'inputs');
  const [recovered, setRecovered] = useState(() => readSessionDraft(draftKey, isInputDraft));
  const [selectedSpec, setSpec] = useState<MILExperimentSpec | null>(recovered?.spec ?? record.inputs);
  const [baseInputs, setBaseInputs] = useState(recovered ? recovered.baseInputs : record.inputs);
  const [planDirty, setPlanDirty] = useState(false);
  const [planBusy, setPlanBusy] = useState(false);
  const [planEditorOpen, setPlanEditorOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const stage = experimentStage(record);
  const readOnly = record.legacy || record.state !== 'active' || record.configurationLocked === true || stage !== 'planning';
  const refresh = () => client.invalidateQueries({ predicate: (query) => query.queryKey.includes(project) });
  const spec = readOnly ? record.inputs ?? initialSpec() : selectedSpec ?? suggestedExperimentInputs(protocols.data?.configurations ?? [], featureBundles.data?.items ?? [], context);
  const name = record.name;
  const [preview, setPreview] = useState<MILExperimentPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const defaultTab = stage === 'planning' ? 'setup' : stage === 'running' ? 'runs' : 'results';
  const [requestedTab, setTab] = useState<ExperimentPage>(initialTab ?? defaultTab);
  const availableTab = requestedTab === 'review' ? stage === 'planning' ? 'review' : stage === 'running' ? 'runs' : 'results' : availableExperimentTab(stage, requestedTab);
  useEffect(() => { if (initialTab) setTab(initialTab); }, [initialTab]);
  function changeTab(next: ExperimentPage) {
    if (busy || planBusy || submitting || (next === 'review' ? stage !== 'planning' : availableExperimentTab(stage, next) !== next)) return;
    setTab(next);
    if (typeof window !== 'undefined') window.history.replaceState(null, '', preparationLink('experiments', context, { experiment: record.id, tab: next === 'setup' ? 'inputs' : next }));
    rememberWorkspaceLocation();
  }
  const protocol = protocols.data?.configurations.find((item) => item.id === spec.protocolId);
  const protocolSpec = protocol?.manifest.spec as ProtocolSpec | undefined;
  const bundle = featureBundles.data?.items.find((item) => item.id === spec.featureBundleId);
  const packs = bundle?.manifest.packs ?? [];
  const packIds = bundle?.manifest.spec.packArtifactIds ?? [];
  const dirty = !sameJSON(record.inputs, spec);
  const stale = !sameJSON(baseInputs, record.inputs);
  const draft = useSessionDraftBackup(draftKey, !readOnly && selectedSpec && (dirty || stale) ? { spec: selectedSpec, baseInputs } : readOnly && recovered ? recovered : null, isInputDraft);
  useWorkspaceNavigationGuard(busy || planBusy || submitting ? 'An experiment request is still pending. Leaving now may hide its outcome.' : !readOnly && dirty && draft.error ? draft.error : null);
  const inputsNeedVerification = stage === 'planning' && (!record.inputs || dirty || stale);
  const tab = inputsNeedVerification && (availableTab === 'batches' || availableTab === 'review') ? 'setup' : availableTab;
  const ready = Boolean(name.trim() && spec.protocolId && spec.featureBundleId);
  function backToExperiments() {
    if (busy || planBusy || submitting || !confirmWorkspaceNavigation()) return;
    onBack();
  }
  useStageLibrary(backToExperiments);
  const loadingNeedsChoice = (spec.loadingPolicy !== 'native' && !spec.packArtifactId && (packs.length > 1 || spec.loadingPolicy === 'mmap')) || Boolean(preview && !preview.canPlan);

  function edit(update: Partial<MILExperimentSpec>) {
    setRecovered(null);
    setSpec((current) => ({ ...(current ?? spec), ...update }));
    setPreview(null);
    setMessage('');
    setError(null);
  }
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setMessage('');
    try { await action(); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('The MIL plan could not be saved.')); }
    finally { setBusy(false); }
  }
  async function reviewAndSave(save: boolean) {
    setPreview(null);
    const result = await mil.preview(project, spec);
    setSpec(spec);
    setPreview(result);
    if (!save || !result.canPlan) return result.canPlan;
    if (sameJSON(record.inputs, spec)) {
      setBaseInputs(record.inputs); setRecovered(null);
      setMessage('Inputs verified. Continue to batches.');
      return true;
    }
    const saved = await experiments.update(project, record.id, { name: record.name, notes: record.notes, tags: record.tags, inputs: spec, expectedRevision: record.revision });
    setBaseInputs(saved.inputs);
    setRecovered(null);
    client.setQueryData(['model-experiment', project, record.id], saved);
    setMessage('Inputs verified and saved. Add training batches and choose their predictors.');
    await refresh();
    return true;
  }

  return <div className="clinical-workspace mil-workspace">
    <PageHeader eyebrow="02 DEVELOP · EXPERIMENT" title={name} description={record.notes || 'Plan your training, follow its progress, and review the resulting models.'} actions={<StageBackButton disabled={busy || planBusy || submitting} onClick={backToExperiments}>Back to experiments</StageBackButton>} />
    <div className="experiment-detail-heading"><Badge>{experimentStageLabel[stage]}</Badge><Badge>{lifecycleLabel[record.state]}</Badge>{record.tags.map((tag) => <Badge key={tag}>{tag}</Badge>)}<span className="experiment-detail-id">{record.id} · revision {record.revision}</span></div>
    {record.legacy ? <p className="callout">This legacy record retains its original batch or draft identity. Use it as a template from the experiments list to organize new work; historical runs stay here.</p> : null}
    {record.state !== 'active' ? <p className="callout">This experiment is {lifecycleLabel[record.state].toLowerCase()}. Its saved history remains visible. Restore it to Active to manage it; submitted configurations remain locked.</p> : null}
    <section className="experiment-context" aria-label={dirty && !readOnly ? 'Draft input context' : 'Saved input context'}>
      <div className="experiment-context-heading"><strong>{dirty && !readOnly ? 'Draft inputs' : 'Saved inputs'}</strong><span>{readOnly ? 'Locked at submission' : dirty ? 'Verify to save' : record.inputs ? 'Shared by all batches' : 'Choose your inputs below'}</span></div>
      <dl><div><dt>Targets &amp; splits</dt><dd>{protocol ? configurationVersionLabel(protocol) : spec.protocolId || 'Not selected'}</dd></div><div><dt>Target</dt><dd>{protocolSpec ? `${protocolSpec.target.field} · ${unitLabel(protocolSpec.target.unit)}` : spec.protocolId ? 'See saved input history' : 'Not selected'}</dd></div><div><dt>Split strategy</dt><dd>{protocolSpec ? `${splitModeLabel(protocolSpec.split.mode)} · ${protocolSpec.split.seeds.length} split seed${protocolSpec.split.seeds.length === 1 ? '' : 's'}` : spec.protocolId ? 'See saved input history' : 'Not selected'}</dd></div><div><dt>Features</dt><dd>{bundle ? versionLabelText(bundle, 'Feature bundle') : spec.featureBundleId || 'Not selected'}</dd></div></dl>
    </section>
    {stage !== 'planning' ? <p className="experiment-stage-notice"><Icon name="lock" size={16} />{stage === 'running' ? 'Training and predictor progress is in Runs. Results open when the experiment finishes.' : 'Inputs, batches and runs are read-only. Results contain the completed evidence and available predictors.'}</p> : null}
    {draft.error && !readOnly && dirty ? <p role="alert" className="callout">{draft.error}</p> : !readOnly && selectedSpec && dirty ? <p className="experiment-draft-notice" role="status">{recovered ? 'Recovered input edits.' : 'Input edits kept in this browser tab.'} You can visit another module and return. Verify inputs to save them to the experiment.</p> : null}
    {readOnly && recovered ? <p className="callout">An earlier input draft is still kept in this browser tab. This record is now locked; its saved inputs take precedence. <button className="text-button" onClick={() => { setRecovered(null); setSpec(record.inputs); setBaseInputs(record.inputs); }}>Discard earlier input draft</button></p> : null}
    <ExperimentNavigation stage={stage} current={tab} disabled={busy || planBusy || submitting} inputsReady={!inputsNeedVerification} hasBatches={Boolean(record.batchPlans?.length || record.batches.length) && !planDirty} onChange={changeTab} />
    <StagePage pageKey={tab}>
    <div hidden={tab !== 'setup'}>
    <p className="experiment-input-intro">Choose the development protocol. Its saved feature bundle is selected automatically; verify the inputs, then configure one or more batches.</p>
    {context.bundleId ? <p className="muted">Prepared inputs are suggested only for experiments without saved inputs. Existing experiment inputs are retained. Review the selected protocol and feature bundle below.</p> : null}
    <ErrorNotice error={error ?? protocols.error ?? featureBundles.error} />
    {stale && !readOnly ? <p className="callout" role="alert">The saved inputs changed while these edits were open. Your draft has not replaced them. <button className="text-button" onClick={() => { setSpec(record.inputs); setBaseInputs(record.inputs); setRecovered(null); setPreview(null); setError(null); }}>Reload saved inputs</button></p> : null}
    {readOnly ? <p className="muted">These are the saved inputs used by this experiment. Archived versions remain visible in the exact input history.</p> : null}
    {planDirty && !readOnly ? <p className="callout">Save or discard your batch edits before changing the shared inputs. <button className="text-button" onClick={() => changeTab('batches')}>Return to batch editor</button></p> : null}
    <SavedNotice>{message}</SavedNotice>
    <fieldset className="mil-plan-fields" disabled={busy || planBusy || submitting || readOnly || stale || planDirty}>
      <legend className="sr-only">MIL experiment plan</legend>
      <Panel title="Training inputs" subtitle="The protocol supplies the dataset, feature bundle, target and splits.">
        <div className="experiment-input-choices">
          <label className="label">Development protocol
            <select className="field" value={spec.protocolId} onChange={(event) => {
              const id = event.target.value;
              const selected = protocols.data?.configurations.find((item) => item.id === id);
              const compatible = bundle && selected && protocolBundleCompatible(selected, bundle);
              const pinnedBundleId = (selected?.manifest.spec as ProtocolSpec | undefined)?.featureBundleId;
              edit({ protocolId: id, ...(pinnedBundleId ? { featureBundleId: pinnedBundleId, packArtifactId: pinnedBundleId === spec.featureBundleId ? spec.packArtifactId : null } : !compatible ? { featureBundleId: '', packArtifactId: null } : {}) });
            }}>
              <option value="">{protocols.isPending ? 'Loading targets…' : 'Choose saved targets and splits'}</option>
              {spec.protocolId && !protocol ? <option value={spec.protocolId} disabled>Retained protocol (archived or unavailable to new selections)</option> : null}
              {(protocols.data?.configurations ?? []).map((item) => <option key={item.id} value={item.id}>{configurationVersionLabel(item)}</option>)}
            </select>
          </label>
          {protocolSpec?.featureBundleId ? <div className="label">
            <span>Feature bundle from protocol</span>
            <strong>{bundle ? versionLabelText(bundle, 'Feature bundle') : versionLabelText({ id: protocolSpec.featureBundleId }, 'Feature bundle')}</strong>
            <small>Saved with this protocol. Choose another protocol to change the bundle.</small>
          </div> : <label className="label">Feature bundle
            <select className="field" value={spec.featureBundleId} disabled={!protocol} onChange={(event) => edit({ featureBundleId: event.target.value, packArtifactId: null })}>
              <option value="">{featureBundles.isPending ? 'Loading features…' : 'Choose verified features'}</option>
              {spec.featureBundleId && !bundle ? <option value={spec.featureBundleId} disabled>Retained feature bundle (archived or unavailable to new selections)</option> : null}
              {(featureBundles.data?.items ?? []).map((item) => {
                const incompatible = Boolean(protocol && !protocolBundleCompatible(protocol, item));
                return <option key={item.id} value={item.id} disabled={item.current === false || incompatible}>{versionLabelText(item, 'Feature bundle')} · {item.manifest.summary.packCount} pack(s){item.current === false ? ' · needs verification' : incompatible ? ' · different protocol inputs' : ''}</option>;
              })}
            </select>
          </label>}
        </div>
        <div className="stack experiment-input-summary">
          {!protocol || !bundle ? <p className="callout">Prepare missing inputs in <a href="#cohort">Targets &amp; splits</a> or <a href="#features">Features</a>, then return here.</p> : null}
          {protocolSpec ? <div className="mil-inherited-protocol"><strong>Selected development protocol</strong><div className="mil-protocol-summary"><Badge>{taskLabel(protocolSpec.target.task)}</Badge><Badge>Target: {protocolSpec.target.field}</Badge><Badge>{unitLabel(protocolSpec.target.unit)} predictions</Badge><Badge>{splitModeLabel(protocolSpec.split.mode)}</Badge><Badge>Split seeds: {protocolSpec.split.seeds.join(', ')}</Badge></div><p className="muted">The protocol supplies the dataset, feature bundle, target and splits. <a href="#cohort">View protocol</a></p></div> : null}
          {bundle ? <div className="mil-bundle-summary"><Badge>{bundle.manifest.summary.slideCount.toLocaleString()} slides</Badge><Badge>{bundle.manifest.summary.patchCount.toLocaleString()} patches</Badge><Badge>{bundle.manifest.summary.dimensions ?? '?'} dimensions</Badge><Badge>{Array.isArray(bundle.manifest.summary.dtype) ? bundle.manifest.summary.dtype.join(', ') : bundle.manifest.summary.dtype ?? 'Unknown dtype'}</Badge></div> : null}
          {bundle?.findings?.length ? <Findings findings={bundle.findings} /> : null}
          {protocolSpec?.featurePackId ? <p className="callout">This older protocol pins pack <code>{protocolSpec.featurePackId}</code>. The selected bundle must include that pack, and this plan must retain its loading source.</p> : null}
        </div>
      </Panel>
      <details className="setup-details" open={loadingNeedsChoice}>
        <summary>Advanced: feature loading <span className="muted">· {spec.loadingPolicy === 'auto' ? 'Automatic' : spec.loadingPolicy === 'mmap' ? 'Packed mmap' : 'Original files'}</span></summary>
        <LoadingOptions value={spec.loadingPolicy} hasPacks={packIds.length > 0} onChange={(loadingPolicy) => edit({ loadingPolicy, ...(loadingPolicy === 'native' ? { packArtifactId: null } : {}) })} />
        {spec.loadingPolicy !== 'native' && packIds.length > 0 ? <label className="label mil-pack-select">Pack in this bundle
          <select className="field" value={spec.packArtifactId ?? ''} onChange={(event) => edit({ packArtifactId: event.target.value || null })}>
            <option value="">{spec.loadingPolicy === 'mmap' ? 'Choose a pack' : packIds.length === 1 ? 'Auto-select when source precision is preserved' : 'Choose a pack — multiple versions are included'}</option>
            {spec.packArtifactId && !packIds.includes(spec.packArtifactId) ? <option value={spec.packArtifactId} disabled>Saved pack is not in this bundle</option> : null}
            {packIds.map((id) => {
              const pack = packs.find((item) => item.id === id);
              return <option key={id} value={id}>{pack ? `${pack.outputDtype} · ${pack.outputPath}` : id}</option>;
            })}
          </select>
        </label> : null}
        <p className="muted mil-memory-note">Packed mmap can read a pack larger than available RAM. Full RAM/GPU preloading and resource-based tuning are not enabled.</p>
      </details>
    </fieldset>
    {!readOnly ? <Panel title="Verify inputs" subtitle="Check feature coverage and protocol compatibility before adding batches.">
      {preview ? <div className="mil-plan-review">
        <Findings findings={preview.findings} />
        {preview.canPlan ? <p className="science-success">Resolved source: <strong>{preview.resolvedLoadingPolicy === 'mmap' ? 'Packed mmap' : 'Original feature files'}</strong>{preview.packArtifactId ? <> · <code>{preview.packArtifactId}</code></> : null}</p> : null}
      </div> : <p className="muted">This check uses the current feature files and the saved protocol and bundle versions.</p>}
      <div className="inline-actions">
        <StageContinueButton type="button" disabled={busy || planBusy || submitting || !ready || readOnly || stale || planDirty} onClick={() => void run(async () => { if (await reviewAndSave(true)) changeTab('batches'); })}>{busy ? 'Checking inputs…' : 'Check & continue to batches'}</StageContinueButton>
        {dirty ? <button type="button" className="text-button" disabled={busy} onClick={() => { setSpec(record.inputs); setBaseInputs(record.inputs); setRecovered(null); setPreview(null); setError(null); }}>Discard input edits</button> : null}
      </div>
      <p className="muted">Verified inputs are shared by every batch. You can change them during planning; training starts only when you submit the experiment.</p>
    </Panel> : null}
    </div>
    {tab === 'results' && stage !== 'planning' ? <ExperimentPredictors project={project} record={record} /> : null}
    <div hidden={tab === 'setup' || tab === 'review'}><DevelopmentBatches protocol={protocolSpec} record={record} experimentStage={stage} onPlanDirtyChange={setPlanDirty} onPlanBusyChange={setPlanBusy} onEditorOpenChange={setPlanEditorOpen} project={project} inputs={record.inputs ?? initialSpec()} experimentName={name} experimentId={record.id} experimentRevision={record.revision} ownedBatches={record.batches} ownedDrafts={record.drafts} executionImplemented={record.executionImplemented === true} readOnly={readOnly || submitting || inputsNeedVerification || busy} tab={tab === 'setup' || tab === 'review' ? 'batches' : tab} onOpenSetup={() => changeTab('setup')} /></div>
    {tab === 'runs' && stage !== 'planning' ? <ExperimentPredictors project={project} record={record} /> : null}
    {tab === 'batches' && !planEditorOpen ? <div className="stage-actions"><StageBackButton disabled={busy || planBusy || submitting} onClick={() => changeTab('setup')}>Back to inputs</StageBackButton>{stage === 'planning' ? <StageContinueButton disabled={busy || planBusy || submitting || dirty || planDirty || stale} onClick={() => changeTab('review')}>Continue to review &amp; submit </StageContinueButton> : <StageContinueButton onClick={() => changeTab('runs')}>Continue to runs </StageContinueButton>}</div> : null}
    {tab === 'setup' && readOnly ? <div className="stage-actions"><StageContinueButton onClick={() => changeTab('batches')}>Continue to batches </StageContinueButton></div> : null}
    {tab === 'runs' && stage === 'finished' ? <div className="stage-actions"><StageBackButton onClick={() => changeTab('batches')}>Back to batches</StageBackButton><StageContinueButton onClick={() => changeTab('results')}>Continue to results </StageContinueButton></div> : null}
    {(tab === 'review' || (tab === 'runs' && stage === 'running')) ? <ExperimentSubmissionControl initialReview={tab === 'review'} project={project} record={record} protocol={protocolSpec} onSubmissionPendingChange={setSubmitting} disabledReason={!record.inputs ? 'Verify experiment inputs before submitting.' : stale ? 'Reload saved inputs before submitting this experiment.' : dirty || planDirty ? 'Save or discard your input and batch edits before submitting.' : !(record.batchPlans?.length || record.batches.some((batch) => batch.state === 'active')) ? 'Add at least one batch before submitting.' : null} onSubmitted={() => { setTab('runs'); if (typeof window !== 'undefined') window.history.replaceState(null, '', preparationLink('experiments', context, { experiment: record.id, tab: 'runs' })); rememberWorkspaceLocation(); }} /> : null}
    {tab === 'review' ? <div className="stage-actions"><StageBackButton disabled={busy || planBusy || submitting} onClick={() => changeTab('batches')}>Back to batches</StageBackButton></div> : null}
    </StagePage>
    <details className="setup-details experiment-management"><summary>Manage experiment</summary><ExperimentMetadata project={project} record={record} /><ExperimentLifecycle key={record.id} project={project} recordKey={record.key} state={record.state} name={name} /></details>
    <details className="setup-details"><summary>Exact input history &amp; configuration snapshots</summary>
      <p className="muted">These saved values belong to this experiment and its batches. Archived inputs remain inspectable; current project defaults are never substituted.</p>
      <h3>Saved experiment inputs</h3><pre className="experiment-snapshot">{JSON.stringify(record.inputSnapshot ?? record.inputs, null, 2)}</pre>
      {record.batches.map((batch) => <details key={batch.id}><summary>{batch.manifest.spec.batchName} · {batch.id} · {lifecycleLabel[batch.state]}</summary><pre className="experiment-snapshot">{JSON.stringify({ inputSnapshot: batch.inputSnapshot, spec: batch.manifest.spec, configurations: batch.manifest.configurations, splitPlans: batch.manifest.splitPlans }, null, 2)}</pre></details>)}
      {record.legacy && record.drafts.map((draft) => <pre key={draft.id} className="experiment-snapshot">{JSON.stringify(draft, null, 2)}</pre>)}
    </details>
  </div>;
}

export function experimentRoute(hash: string, fallback: DevelopmentTab = 'setup') {
  const params = new URLSearchParams(hash.split('?')[1] ?? '');
  const value = params.get('tab');
  const tab: ExperimentPage = value === 'review' ? 'review' : value === 'inputs' ? 'setup' : value === 'predictors' ? 'results' : developmentTabs.some((item) => item.id === value) ? value as DevelopmentTab : fallback;
  return { id: params.get('experiment') ?? '', tab };
}
export default function LocalExperiments({ workspace, initialTab = 'setup' }: { workspace: Workspace; initialTab?: DevelopmentTab }) {
  const context = usePreparationContext();
  function readRoute() {
    const hash = typeof window === 'undefined' ? '' : window.location.hash;
    return { ...experimentRoute(hash, initialTab), explicitTab: new URLSearchParams(hash.split('?')[1] ?? '').has('tab') };
  }
  const [route, setRoute] = useState(readRoute);
  const [libraryFilters, setLibraryFilters] = useState(newExperimentLibraryFilters);
  useEffect(() => {
    const update = () => {
      const hash = window.location.hash;
      setRoute({ ...experimentRoute(hash, initialTab), explicitTab: new URLSearchParams(hash.split('?')[1] ?? '').has('tab') });
    };
    window.addEventListener('hashchange', update);
    window.addEventListener('popstate', update);
    return () => { window.removeEventListener('hashchange', update); window.removeEventListener('popstate', update); };
  }, [initialTab]);
  // A submitted experiment refreshes quickly; a planning or finished record does not.
  const detail = useQuery({ queryKey: ['model-experiment', workspace.project.id, route.id], queryFn: () => experiments.get(workspace.project.id, route.id), enabled: Boolean(route.id), refetchIntervalInBackground: false, refetchInterval: (query) => query.state.data && experimentStage(query.state.data) === 'running' ? 5000 : 30000 });
  function open(id: string) { window.location.hash = preparationLink('experiments', context, id ? { experiment: id } : {}); setRoute({ id, tab: 'setup', explicitTab: false }); }
  if (!route.id) return <><PreparationNotice context={context} /><ExperimentRegistry project={workspace.project.id} onOpen={open} filters={libraryFilters} onFiltersChange={setLibraryFilters} /></>;
  if (!detail.data) return <div className="clinical-workspace"><StageBackButton onClick={() => open('')}>Back to experiments</StageBackButton><ErrorNotice error={detail.error} />{detail.isPending ? <p role="status">Loading experiment…</p> : null}</div>;
  return <><PreparationNotice context={context} /><ErrorNotice error={detail.error} /><ExperimentDetail key={`${route.id}:${context.datasetId ?? ''}:${context.protocolId ?? ''}:${context.bundleId ?? ''}`} workspace={workspace} record={detail.data} initialTab={route.explicitTab ? route.tab : undefined} onBack={() => open('')} onOpen={open} context={context} /></>;
}
