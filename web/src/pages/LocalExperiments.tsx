import { StageBackButton, StageContinueButton } from '../components/StageActions';
import { useEffect, useRef, useState } from 'react';
import { splitModeLabel, unitLabel } from '../lib/labels';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { ApiError } from '../api/client';
import type { ProtocolSpec } from '../api/scientific';
import { experiments, experimentStage, experimentStageLabel, experimentStatusLabel, experimentStatusTone, type ExperimentStage, type ModelExperiment } from '../api/experiments';
import { lifecycleLabel } from '../api/lifecycle';
import ExperimentRegistry, { ExperimentMetadata, newExperimentLibraryFilters } from '../components/ExperimentRegistry';
import ExperimentLifecycle from '../components/ExperimentLifecycle';
import { targetSplits } from '../api/targetSplits';
import { bundles } from '../api/bundles';
import type { MILExperimentSpec } from '../api/mil';
import { sameJSON } from '../lib/json';
import { configurationVersionLabel, versionLabelText } from '../lib/versionLabels';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import { useConfigurations } from '../components/ScientificUI';
import DevelopmentBatches, { developmentTabs, type DevelopmentTab } from '../components/DevelopmentBatches';
import { preparationLink, usePreparationContext, type PreparationContext } from '../lib/preparationRoute';
import PreparationNotice from '../components/PreparationNotice';
import ExperimentPredictors from '../components/ExperimentPredictors';
import ExperimentResults from '../components/ExperimentResults';
import ExperimentQueueBar from '../components/ExperimentQueueBar';
import { managedByTaskCenter } from '../api/development';
import { StagePage, useStageLibrary } from '../components/StageWorkflow';
import ExperimentNavigation, { type ExperimentPage } from '../components/ExperimentNavigation';
import { readSessionDraft, sessionDraftKey, useSessionDraftBackup } from '../lib/sessionDraft';
import { confirmWorkspaceNavigation, rememberWorkspaceLocation, useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import { batchPredictorPolicy, includesRefit, predictorPolicyLabel, experimentPredictorCount } from '../lib/experimentPredictors';
import { ExperimentalSetupInputs, FreezeSetupControl } from '../components/ExperimentalSetupInputs';
import './LocalExperiments.css';

export type ExperimentWorkspaceMode = 'setup' | 'execution';

const initialSpec = (): MILExperimentSpec => ({
  protocolId: '', featureBundleId: '', loadingPolicy: 'auto', packArtifactId: null,
});

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

/** Runs, results and predictors open once an experiment is submitted; results fill in seed by seed. */
export function availableExperimentTab<T extends DevelopmentTab | 'predictors'>(stage: ExperimentStage, tab: T): T | 'batches' {
  if (stage === 'planning' && (tab === 'runs' || tab === 'results' || tab === 'predictors')) return 'batches';
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
    {stage === 'planning' && disabledReason ? <p className="muted">{disabledReason}</p> : null}
    {reviewing && stage === 'planning' ? <div className="experiment-submit-review"><h2>{record.frozenSetupId ? 'Run frozen setup' : 'Review & submit'}</h2><p>{batchCount} saved {batchCount === 1 ? 'batch' : 'batches'} will run together. {record.frozenSetupId ? 'The frozen setup fixes inputs, splits, hyperparameters and predictor choices. Starting checks the current features and compute environment before queuing training.' : 'Inputs, batch settings and predictor choices become permanently read-only. To change a submitted plan, create a new experiment using it as a template.'}</p><ul className="experiment-submit-batches">{batchPolicies.map((batch) => <li key={batch.id}><strong>{batch.name}</strong><span>{predictorPolicyLabel(batch.policy)}{includesRefit(batch.policy) ? ` · refit P${batch.policy.refitPercentile}` : ''}{batch.policy.method === 'skip' ? ' · cross-validation only' : ''}</span></li>)}</ul>{count ? <p><strong>{count.foldRuns} fold runs · {count.total} predictors</strong> ({count.ensembles} ensembles, {count.refits} refits).</p> : null}<p>Batches create their selected predictors automatically after their folds finish. Batches set to cross-validation only do not create predictors.</p>{!retryable ? <button type="button" className="btn btn-primary" disabled={busy || !canSubmit || Boolean(disabledReason)} onClick={() => void submit()}>{busy ? 'Submitting experiment…' : record.frozenSetupId ? 'Start experiment' : 'Freeze & submit experiment'}</button> : null}</div> : null}
    <ErrorNotice error={error} />
    {submission?.error ? <p className="callout" role="status">Submission needs attention: {submission.error.message} The saved plan remains locked.</p> : null}
    {pending && !busy && !acknowledged ? <p className="callout" role="status">The submission response was not confirmed. Retry safely continues this submission and keeps the saved plan unchanged.</p> : null}
    {retryable ? <button type="button" className="btn btn-primary" disabled={busy || !canSubmit} onClick={() => void submit()}>{busy ? 'Submitting experiment…' : 'Retry submission'}</button> : null}
  </section>;
}

export function ExperimentDetail({ workspace: w, record, initialTab, onBack, context = {}, mode }: { mode: ExperimentWorkspaceMode; workspace: Workspace; record: ModelExperiment; initialTab?: ExperimentPage; onBack: () => void; onOpen: (id: string) => void; context?: PreparationContext }) {
  const project = w.project.id;
  const protocols = useConfigurations(project, 'protocol');
  const targetVersions = useQuery({ queryKey: ['scientific', project, 'configurations', 'target-split'], queryFn: () => targetSplits.list(project), enabled: Boolean(record.setupDesign) });
  const targetVersion = targetVersions.data?.configurations.find((item) => item.id === record.setupDesign?.targetSplitId);
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const client = useQueryClient();
  const draftKey = sessionDraftKey(project, record.id, 'inputs');
  const [recovered, setRecovered] = useState(() => readSessionDraft(draftKey, isInputDraft));
  const [selectedSpec, setSpec] = useState<MILExperimentSpec | null>(recovered?.spec ?? record.inputs);
  const [baseInputs, setBaseInputs] = useState(recovered ? recovered.baseInputs : record.inputs);
  const [planDirty, setPlanDirty] = useState(false);
  const [designDirty, setDesignDirty] = useState(false);
  const [planBusy, setPlanBusy] = useState(false);
  const [planEditorOpen, setPlanEditorOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const stage = experimentStage(record);
  const readOnly = mode === 'execution' || record.legacy || record.state !== 'active' || record.configurationLocked === true || stage !== 'planning';
  const refresh = () => client.invalidateQueries({ predicate: (query) => query.queryKey.includes(project) });
  // Execution is read-only and Experimental Setup edits its design in ExperimentalSetupInputs,
  // so both show the saved inputs.
  const spec = record.inputs ?? initialSpec();
  const name = record.name;
  const [busy, setBusy] = useState(false);
  const defaultTab = mode === 'setup' ? 'setup' : stage === 'planning' ? 'review' : stage === 'running' ? 'runs' : 'results';
  const [requestedTab, setTab] = useState<ExperimentPage>(initialTab ?? defaultTab);
  const availableTab: ExperimentPage = mode === 'setup' ? (['setup', 'batches', 'review'].includes(requestedTab) ? requestedTab : 'setup') : stage === 'planning' ? 'review' : requestedTab === 'results' || requestedTab === 'predictors' ? requestedTab : 'runs';
  useEffect(() => { if (initialTab) setTab(initialTab); }, [initialTab]);
  // The Experiments page shows runs and results; inputs and batches of a submitted plan are
  // read on its frozen setup, so a link asking for them goes there instead of landing on Runs.
  const setupTab = mode === 'execution' && (initialTab === 'setup' || initialTab === 'batches') ? initialTab : null;
  const setupLink = setupTab ? preparationLink('experimental-setup', context, { experiment: record.id, tab: setupTab === 'setup' ? 'inputs' : 'batches' }) : null;
  useEffect(() => { if (setupLink && typeof window !== 'undefined') window.location.replace(setupLink); }, [setupLink]);
  function changeTab(next: ExperimentPage) {
    if (busy || planBusy || submitting || (mode !== 'setup' && (next === 'review' ? stage !== 'planning' : availableExperimentTab(stage, next) !== next))) return;
    setTab(next);
    if (typeof window !== 'undefined') window.history.replaceState(null, '', preparationLink(mode === 'setup' ? 'experimental-setup' : 'experiments', context, { experiment: record.id, tab: next === 'setup' ? 'inputs' : next }));
    rememberWorkspaceLocation();
  }
  const protocol = protocols.data?.configurations.find((item) => item.id === spec.protocolId);
  const savedProtocol = record.inputSnapshot?.protocol as { spec?: ProtocolSpec } | undefined;
  const protocolSpec = (protocol?.manifest.spec ?? (readOnly ? savedProtocol?.spec : undefined)) as ProtocolSpec | undefined;
  const bundle = featureBundles.data?.items.find((item) => item.id === spec.featureBundleId);
  const dirty = mode === 'setup' ? designDirty : !sameJSON(record.inputs, spec);
  const stale = mode === 'setup' ? false : !sameJSON(baseInputs, record.inputs);
  const draft = useSessionDraftBackup(draftKey, !readOnly && selectedSpec && (dirty || stale) ? { spec: selectedSpec, baseInputs } : readOnly && recovered ? recovered : null, isInputDraft);
  useWorkspaceNavigationGuard(busy || planBusy || submitting ? 'An experiment request is still pending. Leaving now may hide its outcome.' : !readOnly && dirty && draft.error ? draft.error : null);
  const inputsNeedVerification = mode === 'setup' && !readOnly && stage === 'planning' && (!record.inputs || dirty || stale || !record.setupDesign);
  const tab = inputsNeedVerification && (availableTab === 'batches' || availableTab === 'review') ? 'setup' : availableTab;
  function backToExperiments() {
    if (busy || planBusy || submitting || !confirmWorkspaceNavigation()) return;
    onBack();
  }
  useStageLibrary(backToExperiments);
  return <div className="clinical-workspace mil-workspace">
    <PageHeader eyebrow={mode === 'setup' ? '03 EXPERIMENTAL SETUP' : '04 EXPERIMENTS'} title={name} description={record.notes || (mode === 'setup' ? 'Prepare and freeze the full training design before running it.' : 'Run your frozen setup, follow progress, and review the resulting models.')} actions={<StageBackButton disabled={busy || planBusy || submitting} onClick={backToExperiments}>Back to {mode === 'setup' ? 'setups' : 'experiments'}</StageBackButton>} />
    <div className="experiment-detail-heading"><Badge tone={mode === 'setup' || stage === 'planning' ? 'neutral' : experimentStatusTone(record.status)}>{mode === 'setup' ? record.frozenSetupId ? 'Frozen setup' : 'Setup draft' : record.frozenSetupId && stage === 'planning' ? 'Ready to run' : stage === 'planning' ? experimentStageLabel[stage] : experimentStatusLabel(record.status)}</Badge><Badge>{lifecycleLabel[record.state]}</Badge>{record.tags.map((tag) => <Badge key={tag}>{tag}</Badge>)}<span className="experiment-detail-id">{record.id} · revision {record.revision}</span></div>
    {record.legacy ? <p className="callout">This legacy record retains its original batch or draft identity. Use it as a template from the experiments list to organize new work; historical runs stay here.</p> : null}
    {record.state !== 'active' ? <p className="callout">This experiment is {lifecycleLabel[record.state].toLowerCase()}. Its saved history remains visible. Restore it to Active to manage it; submitted configurations remain locked.</p> : null}
    <section className="experiment-context" aria-label={dirty && !readOnly ? 'Draft input context' : 'Saved input context'}>
      <div className="experiment-context-heading"><strong>{dirty && !readOnly ? 'Draft inputs' : 'Saved inputs'}</strong><span>{readOnly ? 'Frozen design' : dirty ? 'Verify to save' : record.inputs ? 'Shared by all batches' : 'Choose your inputs below'}</span></div>
      <dl><div><dt>Targets &amp; splits</dt><dd>{record.setupDesign ? targetVersion ? versionLabelText(targetVersion, 'Targets & splits') : versionLabelText({ id: record.setupDesign.targetSplitId }, 'Targets & splits') : protocol ? configurationVersionLabel(protocol) : spec.protocolId || 'Not selected'}</dd></div><div><dt>Target</dt><dd>{protocolSpec ? `${protocolSpec.target.field} · ${unitLabel(protocolSpec.target.unit)}` : spec.protocolId ? 'See saved input history' : 'Not selected'}</dd></div><div><dt>{record.setupDesign ? 'Training design' : 'Split strategy'}</dt><dd>{protocolSpec ? `${splitModeLabel(protocolSpec.split.mode)} · ${protocolSpec.split.seeds.length} split seed${protocolSpec.split.seeds.length === 1 ? '' : 's'}` : spec.protocolId ? 'See saved input history' : 'Not selected'}</dd></div><div><dt>Features</dt><dd>{bundle ? versionLabelText(bundle, 'Feature bundle') : spec.featureBundleId || 'Not selected'}</dd></div></dl>
    </section>
    {mode !== 'setup' && stage !== 'planning' ? <p className="experiment-stage-notice"><Icon name="lock" size={16} />{stage === 'running' ? 'Training and predictor progress is in Runs. Results fill in as test folds and training seeds finish.' : 'Inputs, batches and runs are read-only. Results compare the batches across every seed and fold; ready predictors are under Predictors.'}</p> : null}
    {draft.error && !readOnly && dirty ? <p role="alert" className="callout">{draft.error}</p> : !readOnly && selectedSpec && dirty ? <p className="experiment-draft-notice" role="status">{recovered ? 'Recovered input edits.' : 'Input edits kept in this browser tab.'} You can visit another module and return. Verify inputs to save them to the experiment.</p> : null}
    {readOnly && recovered ? <p className="callout">An earlier input draft is still kept in this browser tab. This record is now locked; its saved inputs take precedence. <button className="text-button" onClick={() => { setRecovered(null); setSpec(record.inputs); setBaseInputs(record.inputs); }}>Discard earlier input draft</button></p> : null}
    <ExperimentNavigation mode={mode} frozen={Boolean(record.frozenSetupId)} stage={stage} current={tab} disabled={busy || planBusy || submitting} inputsReady={!inputsNeedVerification} hasBatches={Boolean(record.batchPlans?.length || record.batches.length) && !planDirty} onChange={changeTab} />
    <StagePage pageKey={tab}>
    <div hidden={tab !== 'setup'}>
    {mode === 'setup' ? <ExperimentalSetupInputs project={project} record={record} context={context} readOnly={readOnly} onDirtyChange={setDesignDirty} onBusyChange={setBusy} onVerified={(saved) => { setSpec(saved.inputs); setBaseInputs(saved.inputs); setRecovered(null); setDesignDirty(false); setTab('batches'); void refresh(); }} /> : null}
    </div>
    {tab === 'runs' && stage !== 'planning' && mode !== 'setup' && !record.legacy ? <ExperimentQueueBar project={project} experimentId={record.id} ownerKey={record.batches.map((batch) => batch.execution?.taskCenter?.ownerKey).find(Boolean)} managed={record.batches.some((batch) => managedByTaskCenter(batch.execution)) || record.predictorExecution?.executor === 'task-center'} batchIds={[...record.batches.filter((batch) => batch.state !== 'trashed').map((batch) => batch.id), ...(record.submission?.batchIds ?? [])]} /> : null}
    {/* Planning keeps the batch editor mounted across Inputs and Review so unsaved edits survive;
        a submitted record mounts it only where batches or runs are shown. */}
    {stage === 'planning' || mode === 'setup' || tab === 'batches' || tab === 'runs' ? <div hidden={tab === 'setup' || tab === 'review'}><DevelopmentBatches protocol={protocolSpec} record={record} experimentStage={stage} onPlanDirtyChange={setPlanDirty} onPlanBusyChange={setPlanBusy} onEditorOpenChange={setPlanEditorOpen} project={project} inputs={record.inputs ?? initialSpec()} experimentName={name} experimentId={record.id} experimentRevision={record.revision} ownedBatches={record.batches} ownedDrafts={record.drafts} executionImplemented={record.executionImplemented === true} readOnly={readOnly || submitting || inputsNeedVerification || busy} tab={tab === 'runs' ? 'runs' : 'batches'} onOpenSetup={() => changeTab('setup')} /></div> : null}
    {tab === 'runs' && stage !== 'planning' ? <ExperimentPredictors project={project} record={record} view="progress" /> : null}
    {tab === 'results' && stage !== 'planning' ? <ExperimentResults project={project} record={record} /> : null}
    {tab === 'predictors' && stage !== 'planning' ? <ExperimentPredictors project={project} record={record} view="library" /> : null}
    {tab === 'batches' && !planEditorOpen ? <div className="stage-actions"><StageBackButton disabled={busy || planBusy || submitting} onClick={() => changeTab('setup')}>Back to inputs</StageBackButton>{stage === 'planning' ? <StageContinueButton disabled={busy || planBusy || submitting || dirty || planDirty || stale} onClick={() => changeTab('review')}>{mode === 'setup' ? 'Continue to review & freeze' : 'Continue to review & submit'} </StageContinueButton> : <StageContinueButton onClick={() => changeTab('runs')}>Continue to runs </StageContinueButton>}</div> : null}
    {tab === 'setup' && readOnly ? <div className="stage-actions"><StageContinueButton onClick={() => changeTab('batches')}>Continue to batches </StageContinueButton></div> : null}
    {tab === 'runs' && stage === 'finished' ? <div className="stage-actions"><StageContinueButton onClick={() => changeTab('results')}>Continue to results </StageContinueButton></div> : null}
    {tab === 'results' && stage !== 'planning' ? <div className="stage-actions"><StageBackButton onClick={() => changeTab('runs')}>Back to runs</StageBackButton><StageContinueButton onClick={() => changeTab('predictors')}>Continue to predictors </StageContinueButton></div> : null}
    {mode === 'setup' && (tab === 'review' || Boolean(record.frozenSetupId)) ? <FreezeSetupControl onBusyChange={setSubmitting} project={project} record={record} disabledReason={!record.setupDesign || !record.inputs ? 'Check setup inputs first.' : dirty || planDirty || busy || planBusy ? 'Save your input and batch changes before freezing.' : !record.batchPlans?.length ? 'Add at least one training batch before freezing.' : null} onFrozen={(saved) => { client.setQueryData(['model-experiment', project, record.id], saved); setTab('review'); void refresh(); }} /> : null}
    {mode !== 'setup' && (tab === 'review' || (tab === 'runs' && stage === 'running')) ? <ExperimentSubmissionControl initialReview={tab === 'review'} project={project} record={record} protocol={protocolSpec} onSubmissionPendingChange={setSubmitting} disabledReason={!record.inputs ? 'Verify experiment inputs before submitting.' : stale ? 'Reload saved inputs before submitting this experiment.' : dirty || planDirty ? 'Save or discard your input and batch edits before submitting.' : !(record.batchPlans?.length || record.batches.some((batch) => batch.state === 'active')) ? 'Add at least one batch before submitting.' : null} onSubmitted={() => { setTab('runs'); if (typeof window !== 'undefined') window.history.replaceState(null, '', preparationLink('experiments', context, { experiment: record.id, tab: 'runs' })); rememberWorkspaceLocation(); }} /> : null}
    {mode === 'execution' && record.frozenSetupId ? <div className="stage-actions"><a className="btn btn-secondary" href={preparationLink('experimental-setup', {}, { experiment: record.id, tab: 'review' })}>View frozen setup</a></div> : null}
    {tab === 'review' && mode === 'setup' ? <div className="stage-actions"><StageBackButton disabled={busy || planBusy || submitting} onClick={() => changeTab('batches')}>Back to batches</StageBackButton></div> : null}
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
  const tab: ExperimentPage = value === 'review' ? 'review' : value === 'inputs' ? 'setup' : value === 'predictors' ? 'predictors' : developmentTabs.some((item) => item.id === value) ? value as DevelopmentTab : fallback;
  return { id: params.get('experiment') ?? '', tab };
}
export default function LocalExperiments({ workspace, initialTab = 'setup', mode }: { workspace: Workspace; initialTab?: DevelopmentTab; mode: ExperimentWorkspaceMode }) {
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
  function open(id: string) { window.location.hash = preparationLink(mode === 'setup' ? 'experimental-setup' : 'experiments', context, id ? { experiment: id } : {}); setRoute({ id, tab: 'setup', explicitTab: false }); }
  if (!route.id) return <><PreparationNotice context={context} /><ExperimentRegistry mode={mode} project={workspace.project.id} onOpen={open} filters={libraryFilters} onFiltersChange={setLibraryFilters} /></>;
  if (mode === 'execution' && detail.data && !detail.data.frozenSetupId && experimentStage(detail.data) === 'planning') return <div className="clinical-workspace"><StageBackButton onClick={() => open('')}>Back to experiments</StageBackButton><Panel title={detail.data.name} subtitle="Finish preparing and freeze this setup before running it."><a className="btn btn-primary" href={preparationLink('experimental-setup', context, { experiment: detail.data.id })}>Open Experimental Setup</a></Panel></div>;
  if (!detail.data) return <div className="clinical-workspace"><StageBackButton onClick={() => open('')}>Back to {mode === 'setup' ? 'setups' : 'experiments'}</StageBackButton><ErrorNotice error={detail.error} />{detail.isPending ? <p role="status">Loading experiment…</p> : null}</div>;
  return <><PreparationNotice context={context} /><ErrorNotice error={detail.error} /><ExperimentDetail mode={mode} key={`${mode}:${route.id}:${context.datasetId ?? ''}:${context.protocolId ?? ''}:${context.targetSplitId ?? ''}:${context.bundleId ?? ''}`} workspace={workspace} record={detail.data} initialTab={route.explicitTab ? route.tab : undefined} onBack={() => open('')} onOpen={open} context={context} /></>;
}
