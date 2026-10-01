import { useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { bulkEvaluationActive, bulkEvaluationPollInterval, bulkEvaluations, type EvaluationBatch } from '../api/bulkEvaluations';
import { clinicalAnalyses } from '../api/clinicalUtility';
import { evaluation, type EvaluationCohort } from '../api/evaluation';
import { experimentPollInterval, experiments, type ModelExperimentSummary } from '../api/experiments';
import { computePollInterval, modelEvaluations, predictors, type FrozenPredictor } from '../api/predictors';
import { references } from '../api/references';
import type { Workspace } from '../api/types';
import ApplyMethods from '../components/ApplyMethods';
import ApplyRunDetail from '../components/ApplyRunDetail';
import ApplyRunsTable, { newRunFilters } from '../components/ApplyRunsTable';
import BulkEvaluationRunner, { EvaluationBatchStatus, batchMemberStatus } from '../components/BulkEvaluationRunner';
import { StageBackButton, StageCreateButton, StageLibrary, StageLibraryTabs, StageLibraryToolbar, StageRecordManageButton, useStageLibrary } from '../components/StageWorkflow';
import { Badge, EmptyState, ErrorNotice, PageHeader, Panel } from '../components/ui';
import { cohortKindText, cohortName, runLabeled } from '../lib/applyModels';
import { applyHref, readApplyLink, type ApplyLink, type ApplyView, type RunTab } from '../lib/applyRoutes';
import { useHashVisit } from '../lib/hashRoute';
import { shortRecordId } from '../lib/recordLabels';
import { stageEyebrow } from '../lib/roadmap';
import { rememberWorkspaceLocation, useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import LocalEvaluationSetup from './LocalEvaluationSetup';
import './ModelChains.css';

/** What a new setup starts from: the experiment, predictor and cohort a link named. */
interface SetupSeed { key: number; experiment?: string; predictor?: string; cohort?: string }
const seedOf = (link: ApplyLink, key: number): SetupSeed => ({ key, experiment: link.experiment, predictor: link.predictor, cohort: link.cohort });
const hasContext = (link: ApplyLink) => Boolean(link.experiment || link.predictor || link.cohort);
/** A runs-library link naming an experiment or predictor lists only its runs. */
const runScope = (link: ApplyLink) => !link.run && !link.batch && !link.clinical && (link.view ?? 'runs') === 'runs' ? { experiment: link.experiment, predictor: link.predictor } : {};

const DESCRIPTION = 'Apply ready predictors to a cohort. Runs on a labeled cohort are scored against its labels; runs on an unlabeled cohort predict only.';

/**
 * Apply models: ready predictors applied to cohorts, labeled or not. The link names the view
 * (a library, the setup, a run and its tab, or a batch), so every view can be shared and
 * reopened; the setup in progress and the library filters survive moving between views.
 */
export default function LocalApplyModels({ workspace }: { workspace: Workspace }) {
  const project = workspace.project.id;
  const { parameters, visit } = useHashVisit();
  const link = readApplyLink(parameters);
  // Lists refresh quickly while their own jobs run and slowly once they finish.
  const batches = useQuery({ queryKey: ['evaluation-batches', project], queryFn: () => bulkEvaluations.list(project, true), refetchIntervalInBackground: false, refetchInterval: (query) => bulkEvaluationPollInterval(query.state.data) });
  const running = Boolean(batches.data?.items.some(bulkEvaluationActive));
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), refetchIntervalInBackground: false, refetchInterval: () => running ? 30000 : 120000 });
  const experimentRegistry = useQuery({ queryKey: ['model-experiment-summaries', project], queryFn: () => experiments.summaries(project), refetchIntervalInBackground: false, refetchInterval: (query) => experimentPollInterval(query.state.data) });
  const cohorts = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project) });
  const records = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project), refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  // Reference standards score runs on unlabeled cohorts, so the library says which cohorts have them.
  const standards = useQuery({ queryKey: ['reference-standards', project], queryFn: () => references.list(project), staleTime: 60000 });
  // A saved clinical analysis opens inside the run it analyzes.
  const analysis = useQuery({ queryKey: ['clinical-analysis', project, link.clinical], queryFn: () => clinicalAnalyses.get(project, link.clinical!), enabled: Boolean(link.clinical && !link.run) });
  const [setup, setSetup] = useState<SetupSeed | null>(() => link.view === 'new' ? seedOf(link, 0) : null);
  const [runFilters, setRunFilters] = useState(() => newRunFilters(runScope(link)));
  const [tab, setTab] = useState<RunTab | undefined>(link.tab);
  const [reference, setReference] = useState<string | undefined>(link.reference);
  const [locked, setLocked] = useState(false);
  const [seenVisit, setSeenVisit] = useState(visit);
  if (seenVisit !== visit) {
    setSeenVisit(visit);
    setTab(link.tab);
    setReference(link.reference);
    // A link naming what to apply starts a new setup; one without resumes the setup in progress.
    if (link.view === 'new' && !locked && (hasContext(link) || !setup)) setSetup(seedOf(link, (setup?.key ?? 0) + 1));
    const scope = runScope(link);
    if (scope.experiment || scope.predictor) setRunFilters(newRunFilters(scope));
  }
  // The runner locks while predictors are under review or being submitted.
  useWorkspaceNavigationGuard(locked ? 'Predictors are being reviewed or submitted. Leaving now may hide the outcome.' : null);

  const runs = records.data?.items ?? [];
  const cohortItems = cohorts.data?.items ?? [];
  const predictorItems = registry.data?.items ?? [];
  const experimentItems = experimentRegistry.data?.items ?? [];
  const batchRows = batches.data?.items ?? [];
  const view = link.run || link.clinical ? 'run' : link.batch ? 'batch' : link.view === 'new' ? 'setup' : 'library';
  const library: ApplyView = link.view && link.view !== 'new' ? link.view : 'runs';
  const runId = link.run ?? analysis.data?.manifest.evaluationId;
  const resolving = Boolean(link.clinical && !link.run) && analysis.isPending;
  // A saved clinical analysis opens against the labels it was computed with.
  const [seenAnalysis, setSeenAnalysis] = useState<string | undefined>();
  if (analysis.data && seenAnalysis !== analysis.data.id) {
    setSeenAnalysis(analysis.data.id);
    if (!link.reference) setReference(analysis.data.manifest.selection.referenceId ?? undefined);
  }
  const record = runs.find((item) => item.id === runId);
  const batch = batchRows.find((item) => item.id === link.batch);

  const go = (next: ApplyLink) => { if (!locked) window.location.hash = applyHref(next); };
  const openRun = (id: string) => { go({ run: id }); void records.refetch(); };
  function create() {
    if (locked) return;
    setSetup(seedOf({}, (setup?.key ?? 0) + 1));
    go({ view: 'new' });
  }
  // The tab and the labels a run is scored against are part of its link.
  function showRun(next: { tab?: RunTab; reference?: string }) {
    setTab(next.tab); setReference(next.reference);
    window.history.replaceState(window.history.state, '', applyHref({ run: runId, ...next }));
    rememberWorkspaceLocation();
  }
  // A repeated click on Apply models returns to its runs; an open cohort returns to Cohorts itself.
  useStageLibrary(() => { if (view !== 'library' || library !== 'cohorts') go({}); });
  const refresh = () => void Promise.all([records.refetch(), batches.refetch(), cohorts.refetch(), registry.refetch(), experimentRegistry.refetch()]);
  const errors = <ErrorNotice error={records.error ?? registry.error ?? cohorts.error ?? experimentRegistry.error ?? batches.error ?? analysis.error} />;
  const libraryActions = <>
    {setup && view !== 'setup' ? <button type="button" className="btn btn-secondary btn-small" onClick={() => go({ view: 'new' })}>Resume setup</button> : null}
    <button type="button" className="btn btn-secondary btn-small" disabled={records.isFetching || batches.isFetching} onClick={refresh}>Refresh</button>
  </>;
  const active = <T extends { lifecycleState?: string }>(items: readonly T[]) => items.filter((item) => (item.lifecycleState ?? 'active') === 'active').length;
  const frame = (body: ReactNode, actions: ReactNode = <StageCreateButton onClick={create}>Apply predictors</StageCreateButton>) => <div className="clinical-workspace model-chains apply-models">
    <PageHeader eyebrow={stageEyebrow('apply')} title="Apply models" description={DESCRIPTION} actions={<div className="inline-actions">{actions}</div>} />
    {errors}
    <StageLibrary project={project} title="Apply models">
      <StageLibraryTabs label="Apply models records" current={library} onChange={(id) => go({ view: id === 'runs' ? undefined : id })} tabs={[
        { id: 'runs', title: 'Runs', count: records.data ? active(runs) : undefined },
        { id: 'batches', title: 'Batches', count: batches.data ? active(batchRows) : undefined },
        { id: 'cohorts', title: 'Cohorts', count: cohorts.data ? cohortItems.length : undefined },
        { id: 'methods', title: 'Compare methods' },
      ]} />
      {body}
    </StageLibrary>
  </div>;

  let content: ReactNode = null;
  if (view === 'library') {
    content = library === 'cohorts'
      ? <LocalEvaluationSetup key={visit} workspace={workspace} newCohort={link.newCohort} frame={frame} />
      : frame(library === 'batches'
        ? <BatchLibrary batches={batchRows} cohorts={cohortItems} loading={batches.isPending} onOpen={(id) => go({ batch: id })} actions={libraryActions} />
        : library === 'methods'
          ? <ApplyMethods project={project} runs={runs} predictors={predictorItems} experiments={experimentItems} cohorts={cohortItems} loading={records.isPending} onOpen={openRun} />
          : <ApplyRunsTable runs={runs} predictors={predictorItems} experiments={experimentItems} cohorts={cohortItems} references={standards.data?.items ?? []} loading={records.isPending} onOpen={openRun} onCreate={create} filters={runFilters} onFiltersChange={setRunFilters} actions={libraryActions} />);
  } else if (view === 'run') {
    content = <div className="clinical-workspace model-chains apply-models">
      <PageHeader eyebrow={stageEyebrow('apply')} title={record?.manifest.name ?? 'Run'}
        description={!record ? 'One predictor applied to one cohort.' : runLabeled(record) ? 'Predicted without reading labels, then scored against the cohort’s frozen labels or a reference standard added later.' : 'Predictions for an unlabeled cohort, read without labels. A reference standard added when labels arrive scores them.'}
        actions={<div className="inline-actions"><StageBackButton onClick={() => go({})}>Back to runs</StageBackButton>{record ? <StageRecordManageButton type="configuration" id={record.id} name={record.manifest.name} /> : null}</div>} />
      {errors}
      {record ? <ApplyRunDetail key={record.id} project={project} record={record} runs={runs} predictors={predictorItems} experiments={experimentItems} cohorts={cohortItems} tab={tab} onTabChange={(next) => showRun({ tab: next, reference })} reference={reference} onReferenceChange={(next, nextTab = tab) => showRun({ tab: nextTab, reference: next })} initialAnalysis={link.clinical} />
        : <Panel title="Run"><p role="status">{records.isPending || resolving ? 'Loading run…' : records.isError ? 'The run could not be loaded. Retry to check this record.' : link.clinical && analysis.isError ? 'This clinical analysis is unavailable. Its run may be in Trash.' : 'This run is unavailable. Return to the runs to choose a saved record.'}</p>{records.isError ? <button type="button" className="btn btn-secondary" onClick={() => void records.refetch()}>Retry loading run</button> : null}</Panel>}
    </div>;
  } else if (view === 'batch') {
    const cohort = cohortItems.find((item) => item.id === batch?.cohortId);
    content = <div className="clinical-workspace model-chains apply-models">
      <PageHeader eyebrow={stageEyebrow('apply')} title={batch?.name ?? 'Batch'}
        description={batch ? `${batch.items.length} ${batch.items.length === 1 ? 'predictor' : 'predictors'} applied to ${cohortName(cohort, batch.cohortId)}${cohort ? ` · ${cohortKindText(cohort)}` : ''}. Each predictor has its own run.` : 'Predictors applied together to one cohort; each has its own run.'}
        actions={<StageBackButton onClick={() => go({ view: 'batches' })}>Back to batches</StageBackButton>} />
      {errors}
      <Panel title="Runs in this batch"><EvaluationBatchStatus project={project} id={link.batch!} onOpen={openRun} /></Panel>
    </div>;
  }
  return <div className="apply-workspace">
    {setup ? <div className="clinical-workspace model-chains apply-models" hidden={view !== 'setup'}>
      <PageHeader eyebrow={stageEyebrow('apply')} title="Apply predictors" description="Choose experiments and their predictors, then one cohort. Review compatibility, then run: one job per predictor." actions={<StageBackButton disabled={locked} onClick={() => go({})}>Back to runs</StageBackButton>} />
      {view === 'setup' ? errors : null}
      <ApplySetup key={setup.key} project={project} seed={setup} predictors={predictorItems} predictorsLoading={registry.isPending} experiments={experimentItems} experimentsLoading={experimentRegistry.isPending} cohorts={cohortItems} onLockChange={setLocked} onOpenRun={openRun} />
    </div> : null}
    {content}
  </div>;
}

function ApplySetup({ project, seed, predictors, predictorsLoading, experiments, experimentsLoading, cohorts, onLockChange, onOpenRun }: {
  project: string; seed: SetupSeed; predictors: FrozenPredictor[]; predictorsLoading: boolean; experiments: ModelExperimentSummary[];
  experimentsLoading: boolean; cohorts: EvaluationCohort[]; onLockChange: (locked: boolean) => void; onOpenRun: (id: string) => void;
}) {
  const [chosen, setChosen] = useState<string[] | null>(() => seed.experiment ? [seed.experiment] : null);
  // Only an active predictor can run; a linked one brings its experiment once the registry loads.
  const linked = predictors.find((item) => item.id === seed.predictor && item.lifecycleState === 'active');
  const experimentIds = chosen ?? (linked ? [linked.manifest.experimentId] : []);
  return <Panel title="Apply predictors to a cohort" subtitle="Each ready predictor you select runs on the cohort and keeps its own results.">
    <p className="callout">No job reads a label. On a labeled cohort the predictions are then scored against its frozen labels; on an unlabeled cohort they are predictions only. Development slides are never predicted, and patients seen in development are allowed only for slide-level predictors, flagged, and left out of every metric.</p>
    {seed.predictor && !predictorsLoading && !linked ? <p className="callout">The linked predictor is archived, in Trash or unavailable, so it cannot be applied. Restore it, or choose an experiment and its ready predictors below.</p> : null}
    <BulkEvaluationRunner project={project} predictors={predictors} experiments={experiments} experimentsLoading={experimentsLoading} cohorts={cohorts} linkedCohort={seed.cohort} linkedPredictor={seed.predictor} experimentIds={experimentIds} onExperimentsChange={setChosen} onLockChange={onLockChange} onOpenEvaluation={onOpenRun} />
  </Panel>;
}

/** Predictors applied together to one cohort, labeled or not. */
export function BatchLibrary({ batches, cohorts, loading, onOpen, actions }: {
  batches: EvaluationBatch[]; cohorts: EvaluationCohort[]; loading: boolean; onOpen: (id: string) => void; actions?: ReactNode;
}) {
  const [search, setSearch] = useState('');
  const [state, setState] = useState('active');
  const [status, setStatus] = useState('all');
  const [sort, setSort] = useState('recent');
  const cohortById = new Map(cohorts.map((item) => [item.id, item]));
  const nameOf = (id: string) => cohortName(cohortById.get(id), id);
  const visible = batches.filter((item) => (state === 'all' || (item.lifecycleState ?? 'active') === state)
    && (status === 'all' || item.status === status)
    && `${item.name ?? ''} ${item.id} ${nameOf(item.cohortId)} ${item.items.map((source) => source.predictorName ?? source.predictorId).join(' ')}`.toLowerCase().includes(search.trim().toLowerCase()))
    .sort((a, b) => (sort === 'name' ? (a.name ?? a.id).localeCompare(b.name ?? b.id) : sort === 'oldest' ? (a.createdAt ?? '').localeCompare(b.createdAt ?? '') : (b.createdAt ?? '').localeCompare(a.createdAt ?? '')) || a.id.localeCompare(b.id));
  const filtered = Boolean(search || state !== 'active' || status !== 'all' || sort !== 'recent');
  const reset = () => { setSearch(''); setState('active'); setStatus('all'); setSort('recent'); };
  return <>
    <StageLibraryToolbar search={search} onSearch={setSearch} searchLabel="Search batches" placeholder="Name, ID, cohort or predictor" count={loading ? undefined : visible.length} total={batches.length} actions={actions} onReset={filtered ? reset : undefined}>
      <label className="label">State<select className="field" aria-label="Batch state" value={state} onChange={(event) => setState(event.target.value)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label>
      <label className="label">Status<select className="field" aria-label="Batch status" value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All statuses</option>{[...new Set(batches.map((item) => item.status))].sort().map((value) => <option key={value} value={value}>{batchMemberStatus(value)}</option>)}</select></label>
      <label className="label">Sort<select className="field" aria-label="Sort batches" value={sort} onChange={(event) => setSort(event.target.value)}><option value="recent">Newest first</option><option value="oldest">Oldest first</option><option value="name">Name</option></select></label>
    </StageLibraryToolbar>
    {loading ? <p role="status">Loading batches…</p> : visible.length ? <div className="table-wrap"><table className="chain-table"><thead><tr><th scope="col">Batch</th><th scope="col">Status</th><th scope="col">Cohort</th><th scope="col">Runs</th><th scope="col">Actions</th></tr></thead><tbody>
      {visible.map((item) => { const cohort = cohortById.get(item.cohortId); return <tr key={item.id}>
        <th scope="row">{item.lifecycleState === 'trashed' ? item.name ?? shortRecordId(item.id) : <button type="button" className="text-button stage-record-name" onClick={() => onOpen(item.id)}>{item.name ?? shortRecordId(item.id)}</button>}{item.lifecycleState === 'archived' || item.lifecycleState === 'trashed' ? <small>{item.lifecycleState === 'trashed' ? 'Trash' : 'Archived'}</small> : null}</th>
        <td><Badge>{batchMemberStatus(item.status)}</Badge></td>
        <td>{nameOf(item.cohortId)}{cohort ? <small>{cohortKindText(cohort)}</small> : null}</td>
        <td>{item.items.length.toLocaleString()}</td>
        <td><StageRecordManageButton type="configuration" id={item.id} name={item.name ?? item.id} /></td>
      </tr>; })}
    </tbody></table></div> : <EmptyState title={batches.length ? 'No matching batches' : 'No batches yet'} description={batches.length ? 'Try another search or clear the filters.' : 'A batch is saved each time predictors are applied to a cohort together.'} action={batches.length ? <button type="button" className="btn btn-secondary" onClick={reset}>Clear filters</button> : undefined} />}
  </>;
}
