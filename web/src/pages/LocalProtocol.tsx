import { StageBackButton, StageContinueButton, StageCreateButton } from '../components/StageActions';
import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { scientific } from '../api/scientific';
import { packing } from '../api/packing';
import { bundles } from '../api/bundles';
import ConditionEditor from '../components/ConditionEditor';
export { default as ConditionEditor } from '../components/ConditionEditor';
import { conditionFields } from '../lib/conditions';
import { resetImportedForDataset, validationFractionDefault } from '../lib/split';
import { sameJSON } from '../lib/json';
import {
  configurationVersionLabel,
  datasetVersionLabel,
  versionLabelText,
} from '../lib/versionLabels';
import VersionLabelEditor from '../components/VersionLabelEditor';
import FreezeVersionDialog from '../components/FreezeVersionDialog';
import SetupContext from '../components/SetupContext';
import { StageLibrary, StageLibraryToolbar, StageRecordManageButton, StagePage, StageSteps, useStageLibrary } from '../components/StageWorkflow';
import { SplitStrategy, newSplit, strategyNames } from '../components/SplitStrategy';
import { SplitPools } from '../components/SplitPools';
import { inferTargetSettings } from '../lib/protocol';
import { taskLabel, unitLabel } from '../lib/labels';
import PredictionTargetEditor from '../components/PredictionTargetEditor';
import type {
  AttributeMapping,
  ExecutionPreflight,
  ProtocolPreview,
  ProtocolSpec,
  ScientificDraft,
  VersionLabelInput,
} from '../api/scientific';
import {
  Badge,
  EmptyState,
  ErrorNotice,
  Icon,
  Metric,
  PageHeader,
  Panel,
} from '../components/ui';
import {
  CohortSample,
  CohortStats,
  DistributionBars,
  FieldProfile,
  PartitionLive,
  useProtocolExploration,
  type ProtocolFieldContext,
} from '../components/ProtocolExploration';
import {
  DatasetSelect,
  Findings,
  SavedNotice,
  scienceKey,
  useConfigurations,
  useDatasets,
  useDrafts,
  useRefreshScientific,
} from '../components/ScientificUI';
import './protocol-workflow.css';
import { scientificReviewInvalidated } from '../lib/scientificReview';
import { preparationLink, usePreparationContext, type PreparationContext } from '../lib/preparationRoute';
import { readEditorRecovery, recoveredStep, useEditorRecoveryBackup, type EditorRecovery } from '../lib/editorRecovery';
import { useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import PreparationNotice from '../components/PreparationNotice';

/** Declare allowed covariates; each recipe chooses image, clinical or combined inputs. */
export function TabularPredictorSelection({ dictionary, selected, targetField, onChange }: {
  dictionary: AttributeMapping[]; selected: string[]; targetField: string; onChange: (fields: string[]) => void;
}) {
  const fields = [...new Set([...dictionary.map((item) => item.key), ...selected])];
  return <details className="protocol-extra-inputs" open={selected.length > 0 ? true : undefined}>
    <summary>Extra spreadsheet inputs{selected.length ? ` · ${selected.length} declared` : ''}</summary>
    <p className="muted">Declare patient-level variables available at prediction time. Training recipes can use these in a clinical-only baseline or together with image features. The target, identifiers and split columns cannot be predictors. All slides belonging to one patient must agree on these values.</p>
    <div className="science-checkbox-grid">{fields.map((field) => {
      const item = dictionary.find((column) => column.key === field);
      const checked = selected.includes(field);
      return <label className="science-check" key={field}>
        <input type="checkbox" checked={checked} disabled={!checked && (!item || field === targetField)} onChange={(event) => onChange(event.target.checked ? [...selected, field] : selected.filter((value) => value !== field))} />
        <span>{field}<small>{item ? `${item.owner === 'patient' ? 'Patient' : 'Slide / case'} · ${item.type.replaceAll('_', ' ')}` : 'Column unavailable'}{field === targetField ? ' · prediction target' : ''}</small></span>
      </label>;
    })}</div>
  </details>;
}

const initialSpec = (workspace: Workspace): ProtocolSpec => ({
  datasetId: workspace.dataset.id,
  target: {
    field: '',
    task: '',
    unit: 'patient',
    classes: [],
    labels: {},
    positiveClass: undefined,
    missing: 'block',
    unmapped: 'block',
  },
  predictors: [],
  eligibility: [],
  split: newSplit([workspace.project.config.seed ?? 42], workspace.project.config.folds ?? 5),
  constraints: { minPatientsPerClass: 1, minPatientsPerPartition: 1 },
  featureBundleId: null,
});
export default function LocalProtocol({ workspace: w }: { workspace: Workspace }) {
  const context = usePreparationContext();
  return <ProtocolWorkspace key={`${context.datasetId ?? ''}:${context.bundleId ?? ''}`} workspace={w} context={context} />;
}
function ProtocolWorkspace({ workspace: w, context }: { workspace: Workspace; context: PreparationContext }) {
  const project = w.project.id;
  const queryClient = useQueryClient();
  const targetRequest = useRef(0);
  const datasets = useDatasets(project);
  const drafts = useDrafts(project);
  const configurations = useConfigurations(project, 'protocol');
  const features = useConfigurations(project, 'feature');
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project) });
  const refresh = useRefreshScientific(project);
  // Unsaved target and split input from this tab is retained across a module
  // change or a reload. The module still opens on its library; Return to current
  // protocol reopens the work.
  const [recovered] = useState(() => readEditorRecovery<ProtocolSpec>(project, 'protocol'));
  const [view, setView] = useState<'library' | 'editor'>('library');
  const [libraryFilters, setLibraryFilters] = useState({ search: '', status: 'all', sort: 'recent' });
  const [resumeAvailable, setResumeAvailable] = useState(Boolean(recovered));
  const [spec, setSpec] = useState<ProtocolSpec>(() => recovered?.spec ?? { ...initialSpec(w), datasetId: context.datasetId ?? w.dataset.id, featureBundleId: context.bundleId ?? null });
  const featurePacks = useQuery({
    queryKey: ['feature-packs', project],
    queryFn: () => packing.jobs(project),
  });
  const legacyPack = featurePacks.data?.artifacts.find((artifact) => artifact.id === spec.featurePackId);
  const [step, setStep] = useState<1 | 2 | 3 | 4>((recoveredStep(recovered?.step, 4) || 1) as 1 | 2 | 3 | 4);
  function showStep(next: 1 | 2 | 3 | 4) {
    setStep(next);
  }
  const [name, setName] = useState(recovered?.name ?? `${w.project.name} protocol`);
  const [draft, setDraft] = useState<ScientificDraft<ProtocolSpec> | null>(recovered?.draft ?? null);
  const [seedsText, setSeedsText] = useState((recovered?.spec ?? initialSpec(w)).split.seeds.join(', '));
  const seedsValid = seedsText
    .split(',')
    .every((value) => /^\d+$/.test(value.trim()) && Number(value.trim()) <= 4294967295);
  const [preview, setPreview] = useState<ProtocolPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState(recovered ? 'Unsaved protocol input was recovered in this tab. Choose Return to current protocol to continue, or save it as a draft in your project folder.' : '');
  const [preflight, setPreflight] = useState<ExecutionPreflight | null>(null);
  const [showSaved, setShowSaved] = useState<string | null>(null);
  const [freezeReview, setFreezeReview] = useState<{
    draftId: string; revision: number; preview: ProtocolPreview; name: string; operationId: string;
  } | null>(null);
  const [freezeLabel, setFreezeLabel] = useState<VersionLabelInput>({ tag: '', note: '' });
  useStageLibrary(() => {
    if (!busy && !freezeReview && view !== 'library') { setResumeAvailable(true); setView('library'); }
  });
  const dataset = datasets.data?.datasets.find((item) => item.id === spec.datasetId);
  const linkedDataset = datasets.data?.datasets.find((item) => item.id === context.datasetId);
  const selectedBundle = featureBundles.data?.items.find((item) => item.id === spec.featureBundleId);
  const bundleReady = Boolean(selectedBundle?.current && !selectedBundle.findings.some((finding) => finding.severity === 'error'));
  const dictionary = dataset?.manifest.dictionary ?? [];
  const columns = dictionary.map((item) => item.key);
  const fieldContext = { project, datasetId: spec.datasetId, dictionary };
  const live = useProtocolExploration(project, {
    datasetId: view === 'editor' ? spec.datasetId : '',
    targetField: spec.target.field || undefined,
    eligibility: spec.eligibility,
    featureBundleId: spec.featureBundleId ?? undefined,
    featureSetId: spec.featureSetId ?? undefined,
    featureCoverage: spec.featureCoverage,
    rules: spec.split.pools?.rules ?? spec.split.rules,
    splitMode: spec.split.mode,
    split: spec.split,
  });
  const targetValuesKey = (field: string) => [
    ...scienceKey(project), 'target-values', spec.datasetId, spec.featureBundleId, spec.featureSetId, spec.featureCoverage, field,
    ...(spec.split.version === 4 ? [spec.eligibility, spec.split.pools] : []),
  ];
  async function readTargetValues(field: string) {
    if (spec.split.version !== 4) {
      return scientific.queryDataset(project, spec.datasetId, {
        field, search: '', filters: [], offset: 0, limit: 1,
      });
    }
    const result = await scientific.exploreProtocol(project, {
      datasetId: spec.datasetId, targetField: field, eligibility: spec.eligibility,
      featureBundleId: spec.featureBundleId ?? undefined, featureSetId: spec.featureSetId ?? undefined, featureCoverage: spec.featureCoverage,
      rules: spec.split.pools!.rules, splitMode: spec.split.mode, split: spec.split,
    });
    if (!result.valid) {
      throw new Error(result.findings.find((item) => item.severity === 'error')?.message ?? 'Complete the development cohort selection to read its target values.');
    }
    const values = result.target?.values ?? [];
    return {
      valueCounts: values.map(({ value, slides }) => ({ value, count: slides })),
      valuesTruncated: (result.target?.distinctCount ?? 0) > values.length,
    };
  }
  const labelValues = useQuery({
    queryKey: targetValuesKey(spec.target.field),
    queryFn: () => readTargetValues(spec.target.field),
    enabled: view === 'editor' && Boolean(spec.datasetId) && columns.includes(spec.target.field),
  });
  const rawValues = (labelValues.data?.valueCounts ?? [])
    .map((item) => item.value)
    .filter((value): value is string => value !== null && value.trim() !== '');
  const dirty = !draft || draft.name !== name || !sameJSON(draft.payload.spec, spec);
  const frozen = draft?.status === 'frozen';
  const pristine = { ...initialSpec(w), datasetId: context.datasetId ?? w.dataset.id, featureBundleId: context.bundleId ?? null };
  // A pristine new protocol is not work worth recovering; a saved draft is, as
  // soon as it differs from its saved revision.
  const unsaved = !frozen && (draft
    ? dirty
    : name !== `${w.project.name} protocol` || !sameJSON(spec, pristine));
  const recovery: EditorRecovery<ProtocolSpec> | null = unsaved
    ? { version: 1, name, spec, draft, step: view === 'editor' ? step : 0 }
    : null;
  const backup = useEditorRecoveryBackup(project, 'protocol', recovery);
  useWorkspaceNavigationGuard(busy
    ? 'A protocol request is still pending. Leaving now may hide its outcome.'
    : recovery && backup.error
      ? 'Unsaved protocol input cannot be recovered in this browser. Save the draft before leaving Targets & splits.'
      : null);
  const protocolDrafts = (drafts.data?.drafts ?? []).filter((item) => item.payload.type === 'analysis-protocol' && item.status !== 'frozen');
  const datasetName = (id: string | undefined) => {
    const source = datasets.data?.datasets.find((item) => item.id === id);
    return source ? datasetVersionLabel(source) : id ? versionLabelText({ id }, 'Dataset') : 'Not selected';
  };
  const bundleName = (id: string | null | undefined) => {
    const item = featureBundles.data?.items.find((bundle) => bundle.id === id);
    return item ? versionLabelText(item, 'Feature bundle') : id ? versionLabelText({ id }, 'Feature bundle') : 'Choose in draft';
  };
  const libraryRows = [
    ...(configurations.data?.configurations ?? []).map((item) => ({
      kind: 'configuration' as const, item, name: configurationVersionLabel(item), status: 'frozen',
      datasetName: datasetName(item.manifest.datasetId),
      bundleName: bundleName((item.manifest.spec as ProtocolSpec).featureBundleId),
      updatedAt: item.versionLabel?.updatedAt ?? item.createdAt, createdAt: item.createdAt,
      search: [configurationVersionLabel(item), item.id, item.versionLabel?.note, item.manifest.datasetId,
        datasetName(item.manifest.datasetId), bundleName((item.manifest.spec as ProtocolSpec).featureBundleId), (item.manifest.spec as ProtocolSpec).target?.field].join(' ').toLowerCase(),
    })),
    ...protocolDrafts.map((item) => ({
      kind: 'draft' as const, item, name: item.name, status: item.status,
      datasetName: datasetName((item.payload.spec as ProtocolSpec).datasetId),
      bundleName: bundleName((item.payload.spec as ProtocolSpec).featureBundleId),
      updatedAt: item.updatedAt, createdAt: item.createdAt ?? item.updatedAt,
      search: [item.name, item.id, (item.payload.spec as ProtocolSpec).datasetId,
        datasetName((item.payload.spec as ProtocolSpec).datasetId), bundleName((item.payload.spec as ProtocolSpec).featureBundleId), (item.payload.spec as ProtocolSpec).target?.field].join(' ').toLowerCase(),
    })),
  ];
  const visibleRows = libraryRows.filter((row) =>
    (libraryFilters.status === 'all' || row.status === libraryFilters.status) &&
    row.search.includes(libraryFilters.search.trim().toLowerCase()),
  ).sort((left, right) => {
    const order = libraryFilters.sort === 'name' ? left.name.localeCompare(right.name)
      : libraryFilters.sort === 'oldest' ? left.createdAt.localeCompare(right.createdAt)
        : right.updatedAt.localeCompare(left.updatedAt);
    return order || left.item.id.localeCompare(right.item.id);
  });
  const resetLibraryFilters = () => setLibraryFilters({ search: '', status: 'all', sort: 'recent' });
  const saved = configurations.data?.configurations.find((item) => item.id === showSaved);
  const savedDataset = datasets.data?.datasets.find(
    (item) => item.id === saved?.manifest.datasetId,
  );
  const savedBundleId = (saved?.manifest.spec as ProtocolSpec | undefined)?.featureBundleId;
  const savedBundle = featureBundles.data?.items.find((item) => item.id === savedBundleId);
  const savedFeatureId = (saved?.manifest.spec as ProtocolSpec | undefined)?.featureSetId;
  const savedPackId = (saved?.manifest.spec as ProtocolSpec | undefined)?.featurePackId;
  const savedPack = featurePacks.data?.artifacts.find((artifact) => artifact.id === savedPackId);
  const savedFeature = features.data?.configurations.find((item) => item.id === savedFeatureId);
  function edit(update: Partial<ProtocolSpec>) {
    if (update.target || update.datasetId !== undefined || update.featureBundleId !== undefined || update.eligibility || update.split) targetRequest.current += 1;
    setSpec((current) => ({ ...current, ...update }));
    setPreview(null);
    setMessage('');
    setError(null);
    setShowSaved(null);
  }
  function chooseBundle(featureBundleId: string) {
    if ((spec.featureBundleId ?? '') !== featureBundleId) {
      edit({ featureBundleId: featureBundleId || null, featureSetId: null, featurePackId: null, featureCoverage: 'restrict' });
    }
  }
  function target(update: Partial<ProtocolSpec['target']>) {
    edit({ target: { ...spec.target, ...update } });
  }
  async function chooseTarget(field: string) {
    const datasetId = spec.datasetId;
    edit({ target: { ...spec.target, field, ...inferTargetSettings([]) } });
    const requestId = targetRequest.current;
    if (!field) return;
    try {
      const source = await queryClient.fetchQuery({
        queryKey: targetValuesKey(field),
        queryFn: () => readTargetValues(field),
      });
      if (requestId !== targetRequest.current) return;
      const inferred = inferTargetSettings(
        source.valueCounts.map((item) => item.value),
        source.valuesTruncated,
      );
      setSpec((current) =>
        current.datasetId === datasetId && current.target.field === field
          ? { ...current, target: { ...current.target, ...inferred } }
          : current,
      );
      setPreview(null);
    } catch {
      // The shared query displays its error below the target selector.
    }
  }
  function split(update: Partial<ProtocolSpec['split']>) {
    edit({ split: { ...spec.split, ...update } });
  }
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setMessage('');
    try {
      await action();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason : new Error('The protocol could not be saved.'),
      );
    } finally {
      setBusy(false);
    }
  }
  /** Never discard entered input to start or open another record: save it first. */
  async function keepCurrentWork() {
    if (unsaved) await save();
  }
  function reset() {
    setView('editor');
    setStep(1);
    setFreezeReview(null);
    setPreflight(null);
    setFreezeLabel({ tag: '', note: '' });
    targetRequest.current += 1;
    setSpec({ ...initialSpec(w), datasetId: context.datasetId ?? w.dataset.id, featureBundleId: context.bundleId ?? null });
    setSeedsText(initialSpec(w).split.seeds.join(', '));
    setName(`${w.project.name} protocol`);
    setDraft(null);
    setPreview(null);
    setShowSaved(null);
    setMessage('');
    setError(null);
  }
  async function loadDraft(id: string) {
    const reloading = draft?.id === id;
    targetRequest.current += 1;
    setPreview(null);
    setFreezeReview(null);
    setPreflight(null);
    const next = await scientific.draft<ProtocolSpec>(project, id);
    setDraft(next);
    if (!reloading) setFreezeLabel({ tag: '', note: '' });
    setSpec(next.payload.spec);
    setSeedsText(next.payload.spec.split.seeds.join(', '));
    setName(next.name);
    setView('editor');
    setStep(1);
    setShowSaved(null);
    if (reloading) setMessage(`Reloaded saved draft revision ${next.revision}.`);
    await drafts.refetch();
  }
  async function save() {
    if (draft && !dirty) return draft;
    const next = await scientific.saveDraft(
      project,
      {
        kind: 'experiment',
        name,
        payload: { type: 'analysis-protocol', spec },
      },
      draft ?? undefined,
    );
    setDraft(next);
    setName(next.name);
    await refresh();
    return next;
  }
  const dataSources = spec.split.pools ? (
    <SplitPools
      development={spec.split.version === 4}
      pools={spec.split.pools}
      validationFraction={
        spec.split.validationFraction ??
        validationFractionDefault(spec.split.version)
      }
      onFractionChange={(validationFraction) =>
        split({ validationFraction })
      }
      onChange={(update, validationFraction) =>
        split({
          pools: { ...spec.split.pools!, ...update },
          ...(validationFraction === undefined
            ? {}
            : { validationFraction }),
        })
      }
      live={live}
      targetField={spec.target.field}
      imported={
        spec.split.pools.imported ? (
          <ImportedSplit
            spec={{
              ...spec,
              split: { ...spec.split, imported: spec.split.pools.imported },
            }}
            columns={columns}
            fieldContext={fieldContext}
            heldOutOnly
            onChange={(imported) =>
              split({ pools: { ...spec.split.pools!, imported } })
            }
          />
        ) : null
      }
      renderConditions={(role) => (
        <ConditionEditor
          title={
            role === 'train'
              ? 'Training slide filters'
              : role === 'test'
                ? 'Test set conditions (required)'
                : 'Fixed validation conditions (required)'
          }
          description={
            role === 'train'
              ? 'Select the groups available for fitting and cross-validation.'
              : role === 'test'
                ? 'Reserve these groups for final evaluation.'
                : 'Select separate groups for early stopping.'
          }
          emptyMessage={role === 'train' ? 'Use all shared slides outside fixed validation. Add a filter to choose specific cohorts or records.' : 'Add a condition to define this set.'}
          columns={columns}
          fieldContext={fieldContext}
          conditions={spec.split.pools!.rules[role]}
          onChange={(conditions) =>
            split({
              pools: {
                ...spec.split.pools!,
                ...(role === 'train' ? { trainSelection: conditions.length ? 'rules' as const : 'remaining' as const } : {}),
                rules: { ...spec.split.pools!.rules, [role]: conditions },
              },
            })
          }
        />
      )}
    />
  ) : null;
  const reviewControls = (
    <>
          <div>
            <dl className="protocol-review-facts" aria-label="Development plan to review">
              <div><dt>Dataset</dt><dd>{dataset ? datasetVersionLabel(dataset) : 'Choose a frozen dataset'}</dd></div>
              <div><dt>Feature bundle</dt><dd>{selectedBundle ? versionLabelText(selectedBundle, 'Feature bundle') : 'Choose a named feature bundle'}</dd></div>
              <div><dt>Target</dt><dd>{spec.target.field ? `${spec.target.field} · ${spec.target.classes.length} classes · ${spec.target.unit}` : 'Choose a prediction target'}</dd></div>
              <div><dt>Split design</dt><dd>{strategyNames[spec.split.mode] ?? spec.split.mode}{spec.split.mode === 'kfold' ? ` · ${spec.split.folds} folds` : ''}</dd></div>
              <div><dt>Split seeds</dt><dd>{seedsText || 'Enter valid seeds'}</dd></div>
            </dl>
            <strong>
              {draft
                ? `Revision ${draft.revision} · ${frozen ? 'Frozen protocol' : dirty ? 'Unsaved changes' : 'Saved draft'}`
                : 'New protocol draft'}
            </strong>
            <p>
              Save your progress, or check labels, group overlap and set sizes. Changes require
              a new preview.
            </p>
          </div>
          <div className="inline-actions">
            <button
              type="button"
              className="btn btn-secondary"
              disabled={!name.trim() || !seedsValid}
              onClick={() =>
                void run(async () => {
                  await save();
                  setMessage('Protocol draft saved. It remains editable.');
                })
              }
            >
              Save draft
            </button>
            <StageContinueButton
              type="button"
              disabled={!dataset || !bundleReady || !spec.target.field || !name.trim() || !seedsValid}
              onClick={() =>
                void run(async () => {
                  setPreview(null);
                  const current = await save();
                  setPreview(
                    await scientific.protocolPreview(project, current.id, current.revision),
                  );
                })
              }
            >
              {busy ? 'Validating…' : 'Preview & preflight'}
            </StageContinueButton>
          </div>
    </>
  );
  return (
    <div className="clinical-workspace protocol-workspace">
      <PageHeader
        eyebrow="01 PREPARE"
        title={view === 'library' ? 'Targets & splits' : saved ? configurationVersionLabel(saved) : draft ? name : 'Create protocol'}
        description={view === 'library' ? 'Combine a dataset and feature bundle, then define targets and development splits.'
          : saved ? 'Review this frozen protocol, its target and its split memberships.'
            : 'Choose a dataset and feature bundle, filter their shared slides, then define the target and splits.'}
        actions={view === 'library'
          ? <StageCreateButton disabled={busy} onClick={() => void run(async () => { await keepCurrentWork(); reset(); })}>Create protocol</StageCreateButton>
          : <StageBackButton disabled={busy} onClick={() => { setResumeAvailable(true); setView('library'); }}>Back to protocols</StageBackButton>}
      />
      <StagePage pageKey={view === 'library' ? 'library' : showSaved ?? `protocol-${step}${step === 4 && preview ? `-preview-${preview.previewHash}` : ''}`}>
      {view !== 'library' ? <SetupContext input={saved ? savedDataset ? datasetVersionLabel(savedDataset) : versionLabelText({ id: saved.manifest.datasetId }, 'Dataset') : dataset ? datasetVersionLabel(dataset) : 'Choose a frozen dataset'} output="A dataset, named bundle, target and development splits with recorded identity groups">
        {(saved ? (saved.manifest.spec as ProtocolSpec).split.version : spec.split.version) === 4 ? 'Development data only. Select training records here; prepare test data later in Evaluate.' : 'This saved design retains its original split behavior. Review its assignments before creating a new version.'}
      </SetupContext> : null}
      <PreparationNotice context={context} />
      <ErrorNotice
        error={
          error ?? datasets.error ?? drafts.error ?? configurations.error ?? (view === 'editor' ? featureBundles.error ?? features.error ?? (!saved ? live.error ?? labelValues.error : null) : null)
        }
      />
      <SavedNotice>{message}</SavedNotice>
      {view === 'library' ? (
        <StageLibrary project={project} title="Protocol library">
          <StageLibraryToolbar search={libraryFilters.search} onSearch={(search) => setLibraryFilters((current) => ({ ...current, search }))} searchLabel="Search protocols" placeholder="Name, dataset, bundle, target or note" count={configurations.isPending || drafts.isPending ? undefined : visibleRows.length} total={libraryRows.length}
            onReset={libraryFilters.search || libraryFilters.status !== 'all' || libraryFilters.sort !== 'recent' ? resetLibraryFilters : undefined}
            actions={<>
              {resumeAvailable ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => setView('editor')}>Return to current protocol</button> : null}
              <button type="button" className="btn btn-secondary btn-small" disabled={configurations.isFetching || drafts.isFetching || datasets.isFetching} onClick={() => void refresh()}>Refresh</button>
            </>}>
            <label className="label">Status<select className="field" value={libraryFilters.status} onChange={(event) => setLibraryFilters((current) => ({ ...current, status: event.target.value }))}>
              <option value="all">All statuses</option><option value="editable">Draft</option><option value="frozen">Frozen</option>
            </select></label>
            <label className="label">Sort<select className="field" value={libraryFilters.sort} onChange={(event) => setLibraryFilters((current) => ({ ...current, sort: event.target.value }))}>
              <option value="recent">Last updated</option><option value="oldest">Oldest first</option><option value="name">Name A–Z</option>
            </select></label>
          </StageLibraryToolbar>
          {context.datasetId ? <p className={linkedDataset || datasets.isPending ? 'muted' : 'callout callout-warning'}>{linkedDataset ? <>New protocols will start with <strong>{datasetVersionLabel(linkedDataset)}</strong>.</> : datasets.isPending ? 'Loading the linked dataset…' : 'The linked dataset is unavailable. Create a protocol to choose another frozen dataset.'}</p> : null}
          {configurations.isPending || drafts.isPending ? <p role="status">Loading protocols and drafts…</p> : null}
          {visibleRows.length ? (
            <div className="table-wrap"><table className="stage-library-table" aria-label="Saved protocols and drafts">
              <thead><tr><th>Name</th><th>Status</th><th>Dataset</th><th>Feature bundle</th><th>Updated</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>{visibleRows.map((row) => row.kind === 'configuration' ? <tr key={`protocol-${row.item.id}`}>
                <td><button type="button" className="text-button stage-record-name" disabled={busy} aria-label={`Open protocol ${row.name}`} onClick={() => { setShowSaved(row.item.id); setView('editor'); setPreflight(null); setError(null); setMessage(''); }}>{row.name}</button>{row.item.versionLabel?.note ? <small>{row.item.versionLabel.note}</small> : null}</td>
                <td><Badge tone="green">Frozen protocol</Badge></td>
                <td>{row.datasetName}</td>
                <td>{row.bundleName}</td>
                <td>{new Date(row.updatedAt).toLocaleDateString()}</td>
                <td><StageRecordManageButton type="configuration" id={row.item.id} name={row.name} /></td>
              </tr> : <tr key={`draft-${row.item.id}`}>
                <td><button type="button" className="text-button stage-record-name" disabled={busy} aria-label={`Open protocol draft ${row.name}`} onClick={() => { if (draft?.id === row.item.id && resumeAvailable) { setShowSaved(null); setView('editor'); } else void run(async () => { await keepCurrentWork(); await loadDraft(row.item.id); }); }}>{row.name}</button><small>Revision {row.item.revision}</small></td>
                <td><Badge tone={row.item.status === 'frozen' ? 'frozen' : 'neutral'}>{row.item.status === 'frozen' ? 'Frozen draft' : 'Draft'}</Badge></td>
                <td>{row.datasetName}</td>
                <td>{row.bundleName}</td>
                <td>{new Date(row.updatedAt).toLocaleDateString()}</td>
                <td><StageRecordManageButton type="draft" id={row.item.id} name={row.name} /></td>
              </tr>)}</tbody>
            </table></div>
          ) : !configurations.isPending && !drafts.isPending && !configurations.error && !drafts.error ? <EmptyState
            icon="cohort"
            title={libraryRows.length ? 'No matching protocols or drafts' : 'No protocols or drafts yet'}
            description={libraryRows.length ? 'Try another search or clear the filters.' : 'Create a protocol to choose development records, define a prediction target and review the patient or acknowledged slide groups used for splitting.'}
            action={libraryRows.length
              ? <button type="button" className="btn btn-secondary" onClick={resetLibraryFilters}>Clear filters</button>
              : <StageCreateButton disabled={busy} onClick={() => void run(async () => { await keepCurrentWork(); reset(); })}>Create protocol</StageCreateButton>}
          /> : null}
        </StageLibrary>
      ) : <>
      {!saved && live.data?.findings.some((finding) => finding.severity === 'error') ? <Findings findings={live.data.findings.filter((finding) => finding.severity === 'error')} /> : null}
      {!saved && preview && step !== 4 && !preview.canFreeze ? <div className="callout callout-warning"><strong>The current design has blocking findings.</strong> <button type="button" className="text-button" onClick={() => showStep(4)}>Review findings</button><Findings findings={preview.findings.filter((finding) => finding.severity === 'error')} /></div> : null}
      {!saved ? <StageSteps label="Protocol sections" current={String(step)} onChange={(id) => showStep(Number(id) as 1 | 2 | 3 | 4)} disabled={busy} steps={[
        { id: '1', title: 'Development data', description: dataset && selectedBundle ? `${datasetVersionLabel(dataset)} · ${versionLabelText(selectedBundle, 'Bundle')}` : 'Choose a dataset and feature bundle', complete: Boolean(dataset && bundleReady) },
        { id: '2', title: 'Prediction target', description: spec.target.field || 'Choose the label to predict', complete: Boolean(spec.target.field && spec.target.task && spec.target.classes.length >= 2) },
        { id: '3', title: 'Split design', description: strategyNames[spec.split.mode] || 'Review the saved split strategy', complete: Boolean(preview) },
        { id: '4', title: 'Review & freeze', description: preview ? preview.canFreeze ? 'Ready to freeze' : 'Resolve findings' : 'Check assignments before saving' },
      ]} /> : null}
      {!saved && draft ? <div className="stage-actions">
        <span className="muted">{name} · revision {draft.revision}{dirty ? ' · Unsaved changes' : ''}</span>
        <button type="button" className="btn btn-secondary btn-small" disabled={busy} title="Reload the latest saved revision and discard unsaved local edits. Your version tag and note are kept." onClick={() => void run(() => loadDraft(draft.id))}><Icon name="reset" size={15} /> Reload saved draft</button>
      </div> : null}
      {saved ? (
        <Panel
          title={configurationVersionLabel(saved)}
          subtitle="Assignments and scientific settings are immutable. Copy this configuration to create another draft."
          actions={<Badge tone="green">Frozen</Badge>}
        >
          <VersionLabelEditor
            key={saved.id}
            project={project}
            resourceType="configuration"
            resource={saved}
            tagLabel="Development protocol tag"
            description="Name this frozen cohort, prediction target and split together. Add a note explaining what changed in this version."
          />
          <ul className="detail-list" aria-label="Frozen protocol inputs">
            <li>
              <span>Dataset version</span>
              <strong title={saved.manifest.datasetId}>
                {savedDataset
                  ? datasetVersionLabel(savedDataset)
                  : versionLabelText({ id: saved.manifest.datasetId }, 'Dataset')}
              </strong>
            </li>
            <li>
              <span>{savedBundleId ? 'Feature bundle' : 'Legacy feature version'}</span>
              <strong title={savedBundleId ?? savedFeatureId ?? undefined}>
                {savedBundleId ? savedBundle ? versionLabelText(savedBundle, 'Feature bundle') : versionLabelText({ id: savedBundleId }, 'Feature bundle') :
                savedFeature
                  ? configurationVersionLabel(savedFeature)
                  : savedFeatureId
                    ? versionLabelText({ id: savedFeatureId }, 'Features')
                    : 'Not selected'}
              </strong>
            </li>
            {savedPackId ? <li><span>Legacy pack binding</span><strong title={savedPack?.outputPath ?? savedPackId}>{savedPack ? `${savedPack.outputDtype} pack · ${savedPack.outputPath}` : `Saved pack · ${savedPackId.slice(-12)}`}</strong></li> : null}
          </ul>
          <FrozenProtocolSummary spec={saved.manifest.spec as ProtocolSpec} summary={saved.manifest.summary as ProtocolPreview['summary']} dictionary={savedDataset?.manifest.dictionary} />
          <Findings findings={saved.manifest.findings ?? []} />
          {saved.manifest.partitions ? (
            <>
              <PlanSummary summary={saved.manifest.summary as ProtocolPreview['summary']} />
              <PartitionTable partitions={saved.manifest.partitions} />
            </>
          ) : null}
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              setStep(1);
              targetRequest.current += 1;
              setFreezeLabel({ tag: '', note: '' });
              setSpec(saved.manifest.spec as ProtocolSpec);
              setSeedsText((saved.manifest.spec as ProtocolSpec).split.seeds.join(', '));
              setDraft(null);
              setPreview(null);
              setShowSaved(null);
              setName(`${configurationVersionLabel(saved)} copy`);
            }}
          >
            Copy into a new draft
          </button>
          <div className="science-crosswalk">
            <div className="inline-actions">
              <button
                type="button"
                className="btn btn-secondary"
                disabled={busy}
                onClick={() =>
                  void run(async () =>
                    setPreflight(await scientific.protocolPreflight(project, saved.id)),
                  )
                }
              >
                Recheck input preflight
              </button>
              <StageContinueButton size="small" href={preparationLink('experiments', { protocolId: saved.id, bundleId: savedBundleId ?? undefined })}>Continue to experiments </StageContinueButton>
            </div>
            {preflight?.protocolId === saved.id ? (
              <Findings findings={preflight.findings} />
            ) : null}
            <p className="muted">
              Rechecks the saved bundle and eligible-slide coverage before training in Experiments.
            </p>
          </div>
          <details>
            <summary>View saved configuration</summary>
            <pre className="code-block">{JSON.stringify(saved, null, 2)}</pre>
          </details>
        </Panel>
      ) : null}
      <div className="protocol-editor" hidden={Boolean(saved)}>
      {!datasets.data?.datasets.length && !datasets.isPending ? (
        <Panel title="Start with a frozen dataset">
          <EmptyState
            title="No imported dataset yet"
            description="Import and freeze a dataset before configuring labels or patient assignments."
          />
          <StageContinueButton href="#dataset">
            Go to Dataset
          </StageContinueButton>
        </Panel>
      ) : null}
      {frozen ? (
        <div className="callout">
          This draft is frozen.{' '}
          <button
            type="button"
            className="text-button"
            onClick={() => {
              setDraft(null);
              setPreview(null);
              setMessage('Copied protocol into a new editable draft.');
            }}
          >
            Copy into a new draft
          </button>
        </div>
      ) : null}
      <fieldset className="science-fieldset" disabled={busy || frozen}>
        <div id="protocol-target" className="protocol-section" hidden={step !== 2} tabIndex={-1}>
          <Panel
            title="2. What should the model predict?"
            subtitle="Choose the target column and review the classes found in your selected development data."
            actions={<Badge>Prediction target</Badge>}
          >
            <PredictionTargetEditor
              target={spec.target}
              fieldContext={fieldContext}
              unlinkedSlideCount={live.data?.cohort?.unlinkedSlideCount ?? dataset?.manifest.summary?.unlinkedSlideCount}
              fallbackSlideCount={live.data?.cohort?.fallbackSlideCount ?? dataset?.manifest.summary?.fallbackSlideCount}
              labelValues={labelValues}
              rawValues={rawValues}
              dataLabel={spec.split.version === 4 ? 'selected development records' : 'source values across all dataset slides'}
              onChooseTarget={chooseTarget}
              onChange={target}
            />
          </Panel>
        </div>
        <div id="protocol-cohort" className="protocol-section" hidden={step !== 1} tabIndex={-1}>
          <Panel
            title="1. Choose the development data"
            subtitle="Development uses slides present in both the dataset and the named feature bundle. Add filters to narrow that shared population."
            actions={<Badge>{spec.split.version === 4 ? 'Development data only' : 'Saved cohort design'}</Badge>}
          >
            <div className="stack">
              <div className="science-grid-two">
                <label className="label">
                  Protocol name
                  <input
                    className="field"
                    value={name}
                    maxLength={120}
                    onChange={(event) => {
                      setName(event.target.value);
                      setPreview(null);
                      setMessage('');
                    }}
                  />
                </label>
                <DatasetSelect
                  versions={datasets.data?.datasets ?? []}
                  value={spec.datasetId}
                  onChange={(datasetId) =>
                    edit({
                      datasetId,
                      target: { ...spec.target, field: '', ...inferTargetSettings([]) },
                      predictors: [],
                      eligibility: [],
                      split: {
                        ...spec.split,
                        pools: (spec.split.version ?? 1) >= 3 ? newSplit().pools : undefined,
                        rules: { train: [], val: [], test: [] },
                        imported: resetImportedForDataset(spec.split),
                        domainField: undefined,
                        heldOutDomains: [],
                      },
                    })
                  }
                />
              </div>
              <div className="protocol-bundle-source">
                <label className="label">
                  Feature bundle
                  <select className="field" value={spec.featureBundleId ?? ''} onChange={(event) => chooseBundle(event.target.value)}>
                    <option value="">{featureBundles.isPending ? 'Loading feature bundles…' : 'Choose a named feature bundle'}</option>
                    {spec.featureBundleId && !selectedBundle ? <option value={spec.featureBundleId} disabled>Selected bundle unavailable</option> : null}
                    {(featureBundles.data?.items ?? []).map((item) => {
                      const available = item.current && !item.findings.some((finding) => finding.severity === 'error');
                      return <option key={item.id} value={item.id} disabled={!available}>{versionLabelText(item, 'Feature bundle')} · {item.manifest.summary.slideCount.toLocaleString()} slides{available ? '' : ' · needs verification'}</option>;
                    })}
                  </select>
                </label>
                <p className="muted">Bundles can cover slides from any dataset. The shared Slide IDs define this protocol’s starting population. <a href={preparationLink('features', { datasetId: spec.datasetId || undefined })}>Prepare a feature bundle</a>.</p>
                {selectedBundle ? <div className="science-metrics" aria-label="Dataset and bundle coverage">
                  <Metric label="Dataset slides" value={live.data?.dataset.totalSlides ?? dataset?.manifest.summary?.slideCount ?? '—'} />
                  <Metric label="Bundle slides" value={selectedBundle.manifest.summary.slideCount} />
                  <Metric label="Shared slides" value={live.data?.matchedSlides ?? '—'} note="Before cohort filters and target exclusions" />
                </div> : null}
                {selectedBundle?.findings.length ? <Findings findings={selectedBundle.findings} /> : null}
                {!spec.featureBundleId && (spec.featureSetId || spec.featurePackId) ? <p className="callout">This older draft uses a feature version{spec.featurePackId ? ` and pack ${legacyPack?.outputPath ?? spec.featurePackId}` : ''}. Choose a named bundle to use this draft for a new protocol.</p> : null}
              </div>
              {spec.split.version === 4 ? dataSources : null}
              <details className="setup-details" open={spec.split.version !== 4 || spec.eligibility.length > 0 ? true : undefined}>
                <summary>{spec.split.version !== 4 || spec.eligibility.length ? 'Additional eligibility filters and record preview' : 'Explore shared slide records'}{spec.eligibility.length ? ` · ${spec.eligibility.length} conditions` : ' (optional)'}</summary>
                <div className="stack">
              {spec.split.version !== 4 || spec.eligibility.length ? <ConditionEditor
                title="Which slides should be included?"
                description="Optional. Combine conditions to narrow the dataset and bundle’s shared slides."
                emptyMessage="All shared slides are included. Add filters only to narrow the cohort."
                conditions={spec.eligibility}
                columns={columns}
                fieldContext={fieldContext}
                onChange={(eligibility) => edit({ eligibility })}
              /> : null}
              {!spec.datasetId ? (
                <p className="protocol-live-status">
                  Choose a frozen dataset to see available slides and patients.
                </p>
              ) : live.loading ? (
                <p className="protocol-live-status" role="status">
                  Updating cohort and split counts…
                </p>
              ) : live.error ? (
                <ErrorNotice error={live.error} />
              ) : live.data?.cohort ? (
                <div className="stack" aria-live="polite">
                  <CohortStats stats={live.data.cohort} total={live.data.dataset.totalSlides} />
                  <p className="muted">
                    {['dataset_and_features', 'dataset_and_bundle'].includes(live.data.populationSource ?? '')
                      ? `These counts are the eligible slides the selected bundle covers${live.data.featureExclusions ? `, after excluding ${live.data.featureExclusions.toLocaleString()} without features` : ''}. Missing labels, label mappings and final assignment constraints are checked in Preview & preflight.`
                      : 'These counts apply eligibility conditions only. Missing labels, label mappings and final assignment constraints are checked in Preview & preflight.'}
                  </p>
                  {live.data.target ? (
                    <DistributionBars
                      values={live.data.target.values.map((item) => ({
                        value: item.value,
                        count: item.slides,
                      }))}
                      caption={`${live.data.target.field} · values in the eligible cohort (slides)`}
                      distinctCount={live.data.target.distinctCount}
                    />
                  ) : null}
                  <CohortSample
                    stats={live.data.cohort}
                    fields={[...conditionFields(spec.eligibility), spec.target.field]}
                  />
                </div>
              ) : (
                <p className="protocol-live-status">
                  Counts are unavailable until the eligibility conditions are valid.
                </p>
              )}
              {live.data?.findings.length && live.data.selectionBasis !== 'pools' ? (
                <Findings findings={live.data.findings} />
              ) : null}
                </div>
              </details>
              <TabularPredictorSelection dictionary={dictionary} selected={spec.predictors} targetField={spec.target.field} onChange={(predictors) => edit({ predictors })} />
            </div>
          </Panel>
        </div>
        <div id="protocol-split" className="protocol-section" hidden={step !== 3} tabIndex={-1}>
          <Panel
            title="3. How will models be developed and compared?"
            subtitle="Select training records and create development splits for fitting, early stopping and assessment."
            actions={<Badge>Patient groups stay together</Badge>}
          >
            <div className="stack">
              {(spec.split.version ?? 1) >= 2 ? (
                <>
                  {spec.split.version !== 4 ? (
                    <div className="callout">
                      <p>
                        This legacy design keeps its original assignments. Start a development-only draft and review its cohort selection before freezing.
                      </p>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => {
                          edit({
                            split: {
                              ...newSplit(spec.split.seeds, spec.split.folds),
                            },
                          });
                          setDraft(null);
                        }}
                      >
                        Create a development-only draft
                      </button>
                    </div>
                  ) : null}
                  <SplitStrategy
                    split={spec.split}
                    onChange={split}
                    seedsText={seedsText}
                    seedsValid={seedsValid}
                    onSeedsChange={(text) => {
                      setSeedsText(text);
                      setPreview(null);
                      setMessage('');
                      if (text.split(',').every((value) => /^\d+$/.test(value.trim())))
                        split({ seeds: text.split(',').map((value) => Number(value.trim())) });
                    }}
                    fieldContext={fieldContext}
                    pools={spec.split.version === 4 ? null : dataSources}
                    imported={
                      spec.split.imported ? (
                        <ImportedSplit
                          spec={spec}
                          columns={columns}
                          fieldContext={fieldContext}
                          heldOutOnly
                          onChange={(imported) => split({ imported })}
                        />
                      ) : null
                    }
                    rules={
                      <>
                        {(['test', 'val', 'train'] as const).map((role) => (
                          <div className="stack" key={role}>
                            <ConditionEditor
                              title={`${role === 'val' ? 'Early-stop validation' : role === 'train' ? 'Training' : 'Test'} selection${role === 'test' ? '' : ' (optional)'}`}
                              description={
                                role === 'test'
                                  ? 'Reserve groups for the reported test set.'
                                  : role === 'val'
                                    ? 'Optionally choose fixed groups for early stopping.'
                                    : 'Optionally limit the training pool with conditions.'
                              }
                              emptyMessage={
                                role === 'train'
                                  ? 'Use every eligible group outside test and fixed validation. Early-stop validation is taken from this pool when no validation rules are set.'
                                  : role === 'val'
                                    ? 'Use the selected early-stop percentage of the remaining training pool.'
                                    : 'Add a test rule to reserve the reported test set.'
                              }
                              columns={columns}
                              fieldContext={fieldContext}
                              conditions={spec.split.rules[role]}
                              onChange={(conditions) =>
                                split({ rules: { ...spec.split.rules, [role]: conditions } })
                              }
                            />
                            {role === 'val' && !spec.split.rules.val.length ? (
                              <p className="muted">
                                The exact early-stop selection appears in Preview & preflight.
                              </p>
                            ) : live.loading ? (
                              <p className="protocol-live-status" role="status">
                                Updating selection…
                              </p>
                            ) : live.data?.partitions && live.data.cohort ? (
                              <PartitionLive
                                partition={live.data.partitions[role]}
                                label={
                                  role === 'train' && !spec.split.rules.val.length
                                    ? 'training pool before early-stop validation'
                                    : role === 'val'
                                      ? 'early-stop validation'
                                      : role
                                }
                                fields={[
                                  ...conditionFields(spec.split.rules[role]),
                                  spec.target.field,
                                ]}
                                total={live.data.cohort.totalSlides}
                              />
                            ) : (
                              <p className="muted">
                                Set counts are unavailable until the rules are valid.
                              </p>
                            )}
                          </div>
                        ))}
                        <p className="muted">
                          Each condition group uses its selected AND/OR rule. A matching slide selects its whole group.
                          Overlapping selections block freezing.
                        </p>
                        {live.data?.unassigned?.totalSlides ? (
                          <p className="callout callout-warning">
                            {live.data.unassigned.totalSlides} eligible slides remain
                            unassigned. Broaden the training rules or leave them empty.
                          </p>
                        ) : null}
                      </>
                    }
                  />
                </>
              ) : (
                <>
                  <div className="callout">
                    <strong>Earlier split design</strong>
                    <p>
                      This saved protocol keeps its original split behavior. Its K-fold
                      validation folds are not reported test folds.
                    </p>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => {
                        edit({ split: newSplit(spec.split.seeds, spec.split.folds) });
                        setDraft(null);
                        setMessage(
                          'Created a development draft with assessment folds and early-stop validation. Review the training cohort before freezing.',
                        );
                      }}
                    >
                      Create a draft with the new strategies
                    </button>
                  </div>
                  <div className="science-grid-two">
                    <label className="label">
                      Split strategy
                      <select
                        className="field"
                        value={spec.split.mode}
                        onChange={(event) => {
                          const mode = event.target.value as ProtocolSpec['split']['mode'];
                          const seeds =
                            mode === 'rules' ? [spec.split.seeds[0] ?? 42] : spec.split.seeds;
                          if (mode === 'rules') setSeedsText(seeds.join(', '));
                          split({
                            mode,
                            seeds,
                            ratios: { train: 0.8, val: 0.2, test: 0 },
                            imported:
                              event.target.value === 'imported'
                                ? {
                                    partitionLabels: {},
                                    foldLabels: {},
                                    testFoldLabels: [],
                                  }
                                : undefined,
                          });
                        }}
                      >
                        <option value="rules">
                          Choose sets with rules — train is the remainder
                        </option>
                        <option value="kfold">Generate reproducible K-fold assignments</option>
                        <option value="holdout">
                          Generate reproducible train / validation / test ratios
                        </option>
                        <option value="imported">Validate existing split columns</option>
                      </select>
                    </label>
                    {spec.split.mode !== 'rules' ? (
                      <label className="label">
                        Seeds, separated by commas
                        <input
                          className="field"
                          value={seedsText}
                          onChange={(event) => {
                            const text = event.target.value;
                            setSeedsText(text);
                            setPreview(null);
                            setMessage('');
                            if (text.split(',').every((value) => /^\d+$/.test(value.trim())))
                              split({
                                seeds: text.split(',').map((value) => Number(value.trim())),
                              });
                          }}
                        />
                      </label>
                    ) : null}
                    {['kfold', 'imported'].includes(spec.split.mode) ? (
                      <label className="label">
                        Number of folds
                        <input
                          className="field"
                          type="number"
                          min={2}
                          max={10}
                          value={spec.split.folds}
                          onChange={(event) => split({ folds: Number(event.target.value) })}
                        />
                      </label>
                    ) : null}
                  </div>
                  {spec.split.mode === 'rules' && spec.split.seeds.length > 1 ? (
                    <p className="muted">
                      This saved draft has multiple seeds. Rules determine the same assignments
                      for every seed; seeds do not randomize this strategy.
                    </p>
                  ) : null}
                  <div className="callout">
                    {spec.split.mode === 'rules'
                      ? 'Choose test slides first. Validation rules are optional. Leave training rules empty to use every eligible group outside test and validation for training. No validation set is created unless you select one.'
                      : spec.split.mode === 'kfold'
                        ? 'Fixed rules reserve groups first. The remaining groups are divided into training and validation for each fold using the listed seeds. Add a fixed test rule if you want a separate test set.'
                        : spec.split.mode === 'holdout'
                          ? 'Fixed rules reserve groups first. Ratios apply to the remaining groups and must add up to 1. Preview & preflight shows the final generated assignments.'
                          : 'Existing partition and fold columns are checked against patient grouping. Map only the columns you use. Preview & preflight validates the final imported assignments.'}
                  </div>
                  {!seedsValid ? (
                    <p className="callout callout-warning" role="alert">
                      Enter one or more integer seeds from 0 to 4294967295, separated by commas.
                    </p>
                  ) : null}
                  {spec.split.mode === 'holdout' ? (
                    <div className="science-grid-three">
                      {(['train', 'val', 'test'] as const).map((role) => (
                        <label className="label" key={role}>
                          {role === 'val'
                            ? 'Validation'
                            : role === 'train'
                              ? 'Training'
                              : 'Test'}{' '}
                          ratio
                          <input
                            className="field"
                            type="number"
                            min={0}
                            max={1}
                            step={0.01}
                            value={spec.split.ratios[role]}
                            onChange={(event) =>
                              split({
                                ratios: {
                                  ...spec.split.ratios,
                                  [role]: Number(event.target.value),
                                },
                              })
                            }
                          />
                        </label>
                      ))}
                    </div>
                  ) : null}
                  {spec.split.mode === 'imported' && spec.split.imported ? (
                    <ImportedSplit
                      spec={spec}
                      columns={columns}
                      fieldContext={fieldContext}
                      onChange={(imported) => split({ imported })}
                    />
                  ) : null}
                  {(['test', 'val', 'train'] as const).map((role) => (
                    <div className="stack" key={role}>
                      <ConditionEditor
                        title={`${role === 'val' ? 'Validation' : role === 'train' ? 'Training' : 'Test'} selection${role === 'test' ? '' : ' (optional)'}`}
                        description={
                          role === 'test'
                            ? 'Reserve slides for the final evaluation.'
                            : role === 'val'
                              ? 'Optionally reserve a separate set for choosing model settings.'
                              : 'Optionally limit the training set with explicit conditions.'
                        }
                        emptyMessage={
                          role === 'train' && spec.split.mode === 'rules'
                            ? 'Every eligible group outside test and validation belongs to training.'
                            : role === 'val' && spec.split.mode === 'rules'
                              ? 'No validation set selected. This is optional.'
                              : spec.split.mode === 'rules'
                                ? 'No test set selected.'
                                : 'No fixed groups selected. The split strategy controls generated or imported assignments.'
                        }
                        columns={columns}
                        fieldContext={fieldContext}
                        conditions={spec.split.rules[role]}
                        onChange={(conditions) =>
                          split({
                            rules: { ...spec.split.rules, [role]: conditions },
                          })
                        }
                      />
                      {live.loading ? (
                        <p className="protocol-live-status" role="status">
                          Updating {role === 'val' ? 'validation' : role} selection…
                        </p>
                      ) : live.data?.partitions && live.data.cohort ? (
                        <PartitionLive
                          partition={live.data.partitions[role]}
                          label={
                            role === 'val'
                              ? 'validation'
                              : role === 'train'
                                ? 'training'
                                : 'test'
                          }
                          fields={[
                            ...conditionFields(spec.split.rules[role]),
                            spec.target.field,
                          ]}
                          total={live.data.cohort.totalSlides}
                        />
                      ) : (
                        <p className="protocol-live-status">
                          Set counts are unavailable. Choose a dataset and resolve any rule
                          findings above.
                        </p>
                      )}
                    </div>
                  ))}
                  <p className="muted">
                    Each condition group uses its selected AND/OR rule. A matching eligible slide
                    reserves its whole group. Overlapping test, validation and training
                    selections block freezing. Counts above are before target-label exclusions
                    and final feasibility checks.
                  </p>
                  {live.data?.unassigned && live.data.unassigned.totalSlides > 0 ? (
                    <p className="callout callout-warning">
                      {live.data.unassigned.totalSlides} eligible slides remain outside the
                      fixed sets.
                      {spec.split.mode === 'rules'
                        ? ' Broaden the training conditions or leave them empty to include the remainder.'
                        : ' Their final roles are determined by the selected split strategy in Preview & preflight.'}
                    </p>
                  ) : null}
                </>
              )}
              <details className="setup-details"><summary>Advanced minimum set sizes · {spec.constraints.minPatientsPerClass} groups per class / {spec.constraints.minPatientsPerPartition} per set</summary>
              <div className="protocol-constraints-heading">
                <Icon name="check" />
                <div>
                  <h3>Minimum set sizes</h3>
                  <p className="muted">
                    Check that every required set has enough groups and classes.
                  </p>
                </div>
              </div>
              <div className="science-grid-two">
                <label className="label">
                  Minimum groups per class in each set
                  <input
                    className="field"
                    type="number"
                    min={1}
                    value={spec.constraints.minPatientsPerClass}
                    onChange={(event) =>
                      edit({
                        constraints: {
                          ...spec.constraints,
                          minPatientsPerClass: Number(event.target.value),
                        },
                      })
                    }
                  />
                </label>
                <label className="label">
                  Minimum groups in each set
                  <input
                    className="field"
                    type="number"
                    min={1}
                    value={spec.constraints.minPatientsPerPartition}
                    onChange={(event) =>
                      edit({
                        constraints: {
                          ...spec.constraints,
                          minPatientsPerPartition: Number(event.target.value),
                        },
                      })
                    }
                  />
                </label>
              </div>
              <p className="muted">
                A group is one verified patient or one confirmed Slide ID fallback. These
                minimums check feasibility in each required training, early-stop validation,
                tuning and {spec.split.version === 4 ? 'development assessment' : 'test'} set.
              </p>
              </details>
            </div>
          </Panel>
        </div>
        <div id="protocol-review" className="science-savebar protocol-section" hidden={step !== 4 || Boolean(preview)} tabIndex={-1}>
          {!preview ? reviewControls : null}
        </div>
        <div className="setup-step-actions" aria-label="Protocol step actions" hidden={step === 4 && Boolean(preview)}>
          {step > 1 ? <StageBackButton type="button" onClick={() => showStep((step - 1) as 1 | 2 | 3)}>Back</StageBackButton> : null}
          <p>{step === 1 ? !dataset ? 'Choose a frozen dataset to continue.' : !bundleReady ? 'Choose a verified feature bundle to continue.' : 'Development uses shared dataset and bundle slides that match your selection.' : step === 2 ? !spec.target.field || !spec.target.task || spec.target.classes.length < 2 ? 'Choose a target with at least two mapped classes to continue.' : 'Review the class mapping and positive class before continuing.' : step === 3 ? !seedsValid ? 'Enter valid split seeds before continuing.' : 'The review checks patient overlap, labels and fold sizes before freezing.' : 'Changes to any step invalidate the reviewed assignments. Run the checks again before freezing.'}</p>
          {step < 4 ? <>
            <button type="button" className="btn btn-secondary" disabled={!name.trim() || !seedsValid} onClick={() => void run(async () => { await save(); setMessage('Protocol draft saved. It remains editable.'); })}>Save draft</button>
            <StageContinueButton type="button" disabled={step === 1 ? !dataset || !bundleReady : step === 2 ? !spec.target.field || !spec.target.task || spec.target.classes.length < 2 : !seedsValid} onClick={() => showStep((step + 1) as 2 | 3 | 4)}>Continue to {step === 1 ? 'target' : step === 2 ? 'split design' : 'review'} </StageContinueButton>
          </> : null}
        </div>
      </fieldset>
      {preview && step === 4 ? (
        <div id="protocol-preflight" className="protocol-section" tabIndex={-1}>
          <Panel
            title="Review data and splits"
            subtitle="Assignments are derived by the service and checked for patient overlap, label validity and feasibility."
            actions={
              <Badge tone={preview.canFreeze ? 'green' : 'orange'}>
                {preview.canFreeze ? 'Protocol can freeze' : 'Blocked'}
              </Badge>
            }
          >
            <div className="science-metrics">
              <Metric
                label="Verified patients"
                value={preview.summary.includedPatients}
                note="Distinct supplied patient IDs"
              />
              <Metric
                label="Included slides"
                value={preview.summary.includedSlides}
                note={['dataset_and_features', 'dataset_and_bundle'].includes(preview.summary.populationSource ?? '')
                  ? `${preview.summary.totalSlides} dataset slides · ${(preview.summary.featureExclusions ?? 0).toLocaleString()} without features`
                  : `${preview.summary.totalSlides} total dataset slides`}
              />
              <Metric
                label="Excluded slides"
                value={preview.summary.excludedSlides}
                note="Reasons retained in the protocol"
              />
              <Metric
                label="Assignment groups"
                value={preview.summary.includedGroups ?? preview.summary.includedPatients}
                note={
                  preview.summary.fallbackSlideCount
                    ? `${preview.summary.fallbackSlideCount} Slide ID fallback groups included`
                    : 'Known patients stay together'
                }
              />
            </div>
            <Findings findings={preview.findings} />
            <PlanSummary summary={preview.summary} />
            <PartitionTable partitions={preview.partitions} />
            <div className="callout">
              Freezing preserves this development design and its assignments. Configure and
              launch training separately from Experiments.
            </div>
            <details className="setup-details">
              <summary>Review settings and rerun checks</summary>
              <fieldset className="science-fieldset" disabled={busy || frozen}>
                <div className="science-savebar">{reviewControls}</div>
              </fieldset>
            </details>
            <div className="stage-actions">
              <StageBackButton type="button" disabled={busy} onClick={() => showStep(3)}>Back to split design</StageBackButton>
              <p>Next, name this development protocol version. Your required tag, optional note and reviewed design are saved together.</p>
              <button
                type="button"
                className="btn btn-primary"
                disabled={busy || dirty || frozen || !preview.canFreeze}
                onClick={() => setFreezeReview({ draftId: draft!.id, revision: draft!.revision, preview, name, operationId: `protocol:${crypto.randomUUID()}` })}
              >
                <Icon name="lock" /> Name & freeze protocol
              </button>
            </div>
          </Panel>
        </div>
      ) : null}
      </div>
      </>}
      </StagePage>
      {freezeReview ? (
        <FreezeVersionDialog
          kind="protocol"
          initialLabel={{ ...freezeLabel, tag: freezeLabel.tag || freezeReview.name.slice(0, 80) }}
          onLabelChange={setFreezeLabel}
          onClose={() => setFreezeReview(null)}
          onFreeze={async (versionLabel) => {
            setBusy(true);
            try {
              const result = await scientific.protocolFreeze(project, freezeReview.draftId, freezeReview.revision, freezeReview.preview.previewHash, versionLabel, freezeReview.operationId);
              setDraft((current) => current?.id === freezeReview.draftId ? { ...current, status: 'frozen', revision: freezeReview.revision + 1 } : current);
              await refresh();
              setShowSaved(result.id);
              setPreview(null);
              setMessage(`Development protocol “${result.versionLabel?.tag || versionLabel.tag}” frozen with its commit note.`);
              window.location.hash = preparationLink('experiments', { datasetId: result.manifest.datasetId, protocolId: result.id, bundleId: (result.manifest.spec as ProtocolSpec).featureBundleId ?? undefined, saved: 'protocol' });
              window.scrollTo({ top: 0 });
            } catch (reason) {
              if (scientificReviewInvalidated(reason)) {
                setFreezeReview(null);
                setPreview(null);
                setError(new Error(`${reason.message} Your tag and note have been kept. Review the protocol again before freezing.`));
              }
              throw reason;
            } finally { setBusy(false); }
          }}
        >
          <p><strong>{freezeReview.name}</strong></p>
          <p>{freezeReview.preview.summary.includedSlides.toLocaleString()} included slides · {freezeReview.preview.summary.includedPatients.toLocaleString()} supplied patient IDs · {(freezeReview.preview.summary.includedGroups ?? freezeReview.preview.summary.includedPatients).toLocaleString()} assignment groups</p>
          <p>Cohort, prediction target and split assignments are kept as one version.</p>
        </FreezeVersionDialog>
      ) : null}
    </div>
  );
}
function ImportedSplit({
  spec,
  columns,
  fieldContext,
  onChange,
  heldOutOnly = false,
}: {
  spec: ProtocolSpec;
  columns: string[];
  fieldContext: ProtocolFieldContext;
  onChange: (value: NonNullable<ProtocolSpec['split']['imported']>) => void;
  heldOutOnly?: boolean;
}) {
  const imported = spec.split.imported!;
  return (
    <div className="stack">
      <div className="science-grid-two">
        <label className="label">
          {heldOutOnly ? 'Predefined split column' : 'Partition column (optional)'}
          <select
            className="field"
            value={imported.partitionField ?? ''}
            onChange={(event) =>
              onChange({
                ...imported,
                partitionField: event.target.value || undefined,
                partitionLabels: event.target.value
                  ? {
                      train: 'train',
                      val: 'val',
                      ...(spec.split.version === 4 ? {} : { test: 'test' as const }),
                      trainval: 'trainval',
                    }
                  : {},
              })
            }
          >
            <option value="">No partition column</option>
            {columns.map((column) => (
              <option key={column}>{column}</option>
            ))}
          </select>
        </label>
        {!heldOutOnly ? (
          <label className="label">
            Fold column (optional)
            <select
              className="field"
              value={imported.foldField ?? ''}
              onChange={(event) =>
                onChange({
                  ...imported,
                  foldField: event.target.value || undefined,
                  foldLabels: event.target.value
                    ? Object.fromEntries(
                        Array.from({ length: spec.split.folds }, (_, fold) => [
                          String(fold),
                          fold,
                        ]),
                      )
                    : {},
                  testFoldLabels: event.target.value ? ['-1'] : [],
                })
              }
            >
              <option value="">No fold column</option>
              {columns.map((column) => (
                <option key={column}>{column}</option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
      {imported.partitionField ? (
        <FieldProfile {...fieldContext} field={imported.partitionField} />
      ) : null}
      {imported.foldField ? (
        <FieldProfile {...fieldContext} field={imported.foldField} />
      ) : null}
      {imported.foldField ? (
        <>
          <label className="label">
            Fixed-test fold values, separated by |
            <input
              className="field"
              value={imported.testFoldLabels.join(' | ')}
              onChange={(event) =>
                onChange({
                  ...imported,
                  testFoldLabels: event.target.value
                    .split('|')
                    .map((value) => value.trim())
                    .filter(Boolean),
                })
              }
              placeholder="-1"
            />
          </label>
        </>
      ) : null}
      {imported.partitionField ? (
        <MappingEditor
          label="Partition value mapping"
          development={spec.split.version === 4}
          value={imported.partitionLabels}
          numeric={false}
          onChange={(partitionLabels) =>
            onChange({
              ...imported,
              partitionLabels: partitionLabels as typeof imported.partitionLabels,
            })
          }
        />
      ) : null}
      {imported.foldField ? (
        <MappingEditor
          label="Fold value mapping (zero-based)"
          value={imported.foldLabels}
          numeric
          onChange={(foldLabels) =>
            onChange({
              ...imported,
              foldLabels: foldLabels as Record<string, number>,
            })
          }
        />
      ) : null}
    </div>
  );
}
function MappingEditor({
  development = false,
  label,
  value,
  numeric,
  onChange,
}: {
  development?: boolean;
  label: string;
  value: Record<string, string | number>;
  numeric: boolean;
  onChange: (value: Record<string, unknown>) => void;
}) {
  return (
    <div className="stack">
      <div className="science-subheading">
        <h3>{label}</h3>
        <button
          type="button"
          className="btn btn-secondary btn-small"
          onClick={() =>
            onChange({
              ...value,
              [`value_${Object.keys(value).length + 1}`]: numeric ? 0 : 'train',
            })
          }
        >
          Add mapping
        </button>
      </div>
      {Object.entries(value).map(([raw, mapped], index) => (
        <div className="science-label-row" key={index}>
          <label className="label">
            Source value
            <input
              className="field"
              value={raw}
              onChange={(event) =>
                onChange(
                  Object.fromEntries(
                    Object.entries(value).map(([key, entry]) => [
                      key === raw ? event.target.value : key,
                      entry,
                    ]),
                  ),
                )
              }
            />
          </label>
          <Icon name="arrow" />
          <label className="label">
            {numeric ? 'Fold index' : 'Partition'}
            {numeric ? (
              <input
                className="field"
                type="number"
                min={0}
                value={mapped}
                onChange={(event) => onChange({ ...value, [raw]: Number(event.target.value) })}
              />
            ) : (
              <select
                className="field"
                value={mapped}
                onChange={(event) => onChange({ ...value, [raw]: event.target.value })}
              >
                {(development ? ['train', 'val', 'trainval'] : ['train', 'val', 'test', 'trainval']).map((role) => (
                  <option key={role}>{role}</option>
                ))}
              </select>
            )}
          </label>
          <button
            type="button"
            className="icon-button"
            aria-label={`Remove ${label} ${raw}`}
            onClick={() =>
              onChange(Object.fromEntries(Object.entries(value).filter(([key]) => key !== raw)))
            }
          >
            <Icon name="close" />
          </button>
        </div>
      ))}
    </div>
  );
}
export function FrozenProtocolSummary({ spec, summary, dictionary = [] }: {
  spec: ProtocolSpec; summary: ProtocolPreview['summary']; dictionary?: AttributeMapping[];
}) {
  const target = spec.target;
  const source = dictionary.find((field) => field.key === target.field);
  return <section className="stack" aria-label="Frozen prediction target and identity groups">
    <h3>Prediction target</h3>
    <dl className="protocol-review-facts">
      <div><dt>Reference label field</dt><dd>{target.field}{source && source.sourceColumn !== target.field ? ` · Source column: ${source.sourceColumn}` : ''}</dd></div>
      <div><dt>Prediction task</dt><dd>{taskLabel(target.task)}</dd></div>
      <div><dt>Prediction unit</dt><dd>{unitLabel(target.unit)} · One prediction per {target.unit === 'patient' ? 'patient' : 'slide'}</dd></div>
      <div><dt>Class order</dt><dd>{target.classes.join(' → ')}</dd></div>
      {target.task === 'binary_classification' ? <div><dt>Positive class</dt><dd>{target.positiveClass || 'Not recorded'}</dd></div> : null}
      <div><dt>Label mapping</dt><dd>{Object.entries(target.labels).map(([raw, mapped]) => `${raw} → ${mapped}`).join('; ') || 'Not recorded'}</dd></div>
      <div><dt>Missing reference labels</dt><dd>{target.missing === 'exclude' ? 'Exclude affected slides' : 'Block until resolved'}</dd></div>
      <div><dt>Unmapped reference labels</dt><dd>{target.unmapped === 'exclude' ? 'Exclude affected slides' : 'Block until resolved'}</dd></div>
    </dl>
    <h3>Identity groups used for splitting</h3>
    <div className="science-metrics protocol-metrics">
      <Metric label="Supplied patient IDs" value={summary.includedPatients.toLocaleString()} note="Distinct recorded patient identifiers" />
      <Metric label="Assignment groups" value={(summary.includedGroups ?? summary.includedPatients).toLocaleString()} note="Slides in the same group stay together" />
      {summary.fallbackSlideCount !== undefined ? <Metric label="Acknowledged slide / case groups" value={summary.fallbackSlideCount.toLocaleString()} note="Slide_ID used where patient identity is unavailable" /> : null}
    </div>
    {summary.fallbackSlideCount ? <p className="callout callout-warning">Some groups identify individual slides or cases without supplied patient IDs. Patient independence cannot be verified for those groups.</p> : null}
  </section>;
}

function PlanSummary({ summary }: { summary: ProtocolPreview['summary'] }) {
  if ((summary.splitVersion ?? 1) < 2) return null;
  const coverage = summary.oofCoverage;
  const development = summary.splitVersion === 4;
  return (
    <div className="stack">
      <div className="split-plan-summary">
        <Badge tone="purple">
          {summary.strategy
            ? (strategyNames[summary.strategy] ?? summary.strategy)
            : 'Split plans'}
        </Badge>
        <Badge>
          {summary.evaluationPlanCount ?? 0}{' '}
          {development ? 'development assessment' : summary.splitVersion === 3 ? 'CV assessment' : 'evaluation'}{' '}
          {summary.evaluationPlanCount === 1 ? 'plan' : 'plans'}
        </Badge>
        {summary.finalPlanCount ? (
          <Badge>
            {summary.finalPlanCount} final test{' '}
            {summary.finalPlanCount === 1 ? 'plan' : 'plans'}
          </Badge>
        ) : null}
        {summary.innerPlanCount ? (
          <Badge>{summary.innerPlanCount} inner tuning plans</Badge>
        ) : null}
      </div>
      {summary.poolCounts && (summary.finalPlanCount || development) ? (
        <div className="science-metrics">
          {(development ? (['train', 'val'] as const) : (['train', 'val', 'test'] as const)).map((role) => (
            <Metric
              key={role}
              label={
                role === 'train'
                  ? 'Selected training set'
                  : role === 'test'
                    ? 'Reserved final test set'
                    : summary.validationSource === 'training_fraction'
                      ? 'Early-stop validation'
                      : 'Fixed validation set'
              }
              value={
                role === 'val' && summary.validationSource === 'training_fraction'
                  ? 'Sampled from training'
                  : `${summary.poolCounts![role].slides} slides`
              }
              note={
                role === 'val' && summary.validationSource === 'training_fraction'
                  ? `${(summary.validationFraction ?? validationFractionDefault(summary.splitVersion)) * 100}% of each training set or fold`
                  : `${summary.poolCounts![role].patients} verified patients · ${summary.poolCounts![role].groups} groups`
              }
            />
          ))}
        </div>
      ) : null}
      {coverage && !(summary.splitVersion === 3 && !summary.evaluationPlanCount) ? (
        <div className="protocol-coverage">
          <progress
            aria-label={
              (summary.splitVersion ?? 1) >= 3
                ? 'Planned CV assessment coverage'
                : 'Planned test coverage'
            }
            value={coverage.testedGroups}
            max={Math.max(1, coverage.groups)}
          />
          <span>
            <strong>
              {(summary.splitVersion ?? 1) >= 3
                ? 'CV assessment coverage within training:'
                : 'Planned test coverage:'}
            </strong>{' '}
            {coverage.testedGroups} of {coverage.groups} groups · {coverage.minTestAppearances}–
            {coverage.maxTestAppearances} {(summary.splitVersion ?? 1) >= 3 ? 'assessment' : 'test'}{' '}
            appearances per group per seed.
          </span>
        </div>
      ) : null}
      <p className="muted">
        These are planned assignments. Predictions and model selection are performed later in
        Experiments.
      </p>
    </div>
  );
}
export function PartitionTable({ partitions }: { partitions: ProtocolPreview['partitions'] }) {
  const development = partitions.some((partition) => partition.pool === 'development');
  const modern = partitions.some((partition) => Boolean(partition.planId));
  const explicitPools = partitions.some((partition) => partition.phase === 'final');
  const nested = partitions.some((partition) => partition.phase === 'inner');
  const roles = nested
    ? (['train', 'val', 'tune', 'test'] as const)
    : (['train', 'val', 'test'] as const);
  return partitions.length ? (
    <div className="table-wrap">
      <table>
        <caption>Exact set sizes and class counts by assignment group</caption>
        <thead>
          <tr>
            <th>{modern ? 'Plan' : 'Seed / fold'}</th>
            <th>Training</th>
            <th>{modern ? 'Early-stop validation' : 'Validation'}</th>
            {nested ? <th>Inner tuning</th> : null}
            <th>
              {development ? 'Development assessment' : explicitPools ? 'CV assessment / final test' : modern ? 'Reported test' : 'Test'}
            </th>
          </tr>
        </thead>
        <tbody>
          {partitions.map((partition) => {
            const total =
              partition.train.slides +
              partition.val.slides +
              partition.test.slides +
              (partition.tune?.slides ?? 0);
            return (
              <tr key={partition.planId ?? `${partition.seed}-${partition.fold}`}>
                <th className="split-plan-name">
                  {partition.phase === 'final'
                    ? 'Final test evaluation'
                    : partition.phase === 'inner'
                      ? `Outer ${(partition.outerFold ?? 0) + 1} · inner ${(partition.innerFold ?? 0) + 1}`
                      : partition.phase === 'outer'
                        ? `Outer ${(partition.outerFold ?? 0) + 1} · refit`
                        : partition.domain !== undefined
                          ? `${development || explicitPools ? 'Assessment' : 'Test'}: ${partition.domain}`
                          : partition.repeat !== undefined
                            ? `Repeat ${partition.repeat + 1}`
                            : partition.fold === null
                              ? 'Held-out evaluation'
                              : `${development || explicitPools ? 'Assessment fold' : modern ? 'Test fold' : 'Fold'} ${partition.fold + 1}`}
                  <small className="science-block">Seed {partition.seed}</small>
                  {explicitPools ? (
                    <small className="science-block">
                      {partition.phase === 'final'
                        ? 'Uses the reserved test set'
                        : 'Within the selected training set'}
                    </small>
                  ) : null}
                  {partition.excludedValidation?.groups ? (
                    <small className="science-block">
                      {partition.excludedValidation.groups} fixed validation groups from the
                      held-out site omitted.
                    </small>
                  ) : null}
                </th>
                {roles.map((role) => {
                  const counts = partition[role];
                  return (
                    <td key={role} className="protocol-partition-bars">
                      {!counts || (partition.phase === 'inner' && role === 'test') ? (
                        <small className="muted">
                          {role === 'test'
                            ? development || explicitPools
                              ? 'Outer assessment stays untouched'
                              : 'Outer test stays untouched'
                            : 'Settings selected using inner folds'}
                        </small>
                      ) : (
                        <>
                          <strong>{counts.slides} slides</strong>
                          <div className="science-bar-track" aria-hidden="true">
                            <span
                              style={{
                                width: `${(counts.slides / Math.max(1, total)) * 100}%`,
                              }}
                            />
                          </div>
                          <small className="science-block">
                            {counts.patients} verified patients
                          </small>
                          {counts.fallbackSlides ? (
                            <small className="science-block">
                              {counts.fallbackSlides} Slide ID fallback groups
                            </small>
                          ) : null}
                          <small className="muted">
                            {Object.entries(counts.classes)
                              .map(([label, count]) => `${label}: ${count}`)
                              .join(' · ')}
                          </small>
                        </>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  ) : (
    <p className="muted">
      No valid assignments yet. Resolve the findings above and preview again.
    </p>
  );
}
