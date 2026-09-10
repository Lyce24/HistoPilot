import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { scientific } from '../api/scientific';
import { resetImportedForDataset, validationFractionDefault } from '../lib/split';
import { sameJSON } from '../lib/json';
import { SplitStrategy, newSplit, strategyNames } from '../components/SplitStrategy';
import { SplitPools } from '../components/SplitPools';
import { inferTargetSettings, preservePositiveClass } from '../lib/protocol';
import type {
  Condition,
  ConditionValue,
  ExecutionPreflight,
  ProtocolPreview,
  ProtocolSpec,
  ScientificDraft,
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
  DraftSelect,
  Findings,
  SavedNotice,
  scienceKey,
  useConfigurations,
  useDatasets,
  useDrafts,
  useRefreshScientific,
} from '../components/ScientificUI';

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
  featureSetId: null,
});
export default function LocalProtocol({ workspace: w }: { workspace: Workspace }) {
  const project = w.project.id;
  const queryClient = useQueryClient();
  const targetRequest = useRef(0);
  const datasets = useDatasets(project);
  const drafts = useDrafts(project);
  const configurations = useConfigurations(project, 'protocol');
  const features = useConfigurations(project, 'feature');
  const refresh = useRefreshScientific(project);
  const [spec, setSpec] = useState<ProtocolSpec>(() => initialSpec(w));
  const [name, setName] = useState(`${w.project.name} protocol`);
  const [draft, setDraft] = useState<ScientificDraft<ProtocolSpec> | null>(null);
  const [seedsText, setSeedsText] = useState(initialSpec(w).split.seeds.join(', '));
  const seedsValid = seedsText
    .split(',')
    .every((value) => /^\d+$/.test(value.trim()) && Number(value.trim()) <= 4294967295);
  const [preview, setPreview] = useState<ProtocolPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const [preflight, setPreflight] = useState<ExecutionPreflight | null>(null);
  const [showSaved, setShowSaved] = useState<string | null>(null);
  const dataset = datasets.data?.datasets.find((item) => item.id === spec.datasetId);
  const dictionary = dataset?.manifest.dictionary ?? [];
  const columns = dictionary.map((item) => item.key);
  const fieldContext = { project, datasetId: spec.datasetId, dictionary };
  const live = useProtocolExploration(project, {
    datasetId: spec.datasetId,
    targetField: spec.target.field || undefined,
    eligibility: spec.eligibility,
    rules: spec.split.pools?.rules ?? spec.split.rules,
    splitMode: spec.split.mode,
    split: spec.split,
  });
  const labelQuery = {
    field: spec.target.field,
    search: '',
    filters: [],
    offset: 0,
    limit: 1,
  };
  const labelValues = useQuery({
    queryKey: [...scienceKey(project), 'target-values', spec.datasetId, spec.target.field],
    queryFn: () => scientific.queryDataset(project, spec.datasetId, labelQuery),
    enabled: Boolean(spec.datasetId) && columns.includes(spec.target.field),
  });
  const rawValues = (labelValues.data?.valueCounts ?? [])
    .map((item) => item.value)
    .filter((value): value is string => value !== null && value.trim() !== '');
  const dirty = !draft || draft.name !== name || !sameJSON(draft.payload.spec, spec);
  const frozen = draft?.status === 'frozen';
  const saved = configurations.data?.configurations.find((item) => item.id === showSaved);
  function edit(update: Partial<ProtocolSpec>) {
    if (update.target || update.datasetId !== undefined) targetRequest.current += 1;
    setSpec((current) => ({ ...current, ...update }));
    setPreview(null);
    setMessage('');
    setError(null);
    setShowSaved(null);
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
        queryKey: [...scienceKey(project), 'target-values', datasetId, field],
        queryFn: () =>
          scientific.queryDataset(project, datasetId, {
            field,
            search: '',
            filters: [],
            offset: 0,
            limit: 1,
          }),
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
  function reset() {
    targetRequest.current += 1;
    setSpec(initialSpec(w));
    setSeedsText(initialSpec(w).split.seeds.join(', '));
    setName(`${w.project.name} protocol`);
    setDraft(null);
    setPreview(null);
    setShowSaved(null);
    setMessage('');
    setError(null);
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
  return (
    <>
      <PageHeader
        eyebrow="ANALYSIS DESIGN"
        title="Target & split"
        description="Choose what to predict, which slides to include, and how to divide them into training, validation and test sets."
        actions={
          <button type="button" className="btn btn-primary" disabled={busy} onClick={reset}>
            <Icon name="plus" /> New protocol
          </button>
        }
      />
      <div className="science-toolbar">
        <DraftSelect
          drafts={(drafts.data?.drafts ?? []).filter(
            (item) => item.payload.type === 'analysis-protocol',
          )}
          value={draft?.id ?? ''}
          disabled={busy}
          onChange={(id) => {
            if (!id) {
              reset();
              return;
            }
            targetRequest.current += 1;
            void run(async () => {
              const next = await scientific.draft<ProtocolSpec>(project, id);
              setDraft(next);
              setSpec(next.payload.spec);
              setSeedsText(next.payload.spec.split.seeds.join(', '));
              setName(next.name);
              setPreview(null);
              setShowSaved(null);
            });
          }}
        />
        <label className="label">
          Frozen protocols
          <select
            className="field"
            value={showSaved ?? ''}
            disabled={busy}
            onChange={(event) => setShowSaved(event.target.value || null)}
          >
            <option value="">Choose a saved protocol</option>
            {configurations.data?.configurations.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id.slice(-10)} · {new Date(item.createdAt).toLocaleString()}
              </option>
            ))}
          </select>
        </label>
      </div>
      <ErrorNotice
        error={
          error ?? datasets.error ?? drafts.error ?? configurations.error ?? features.error
        }
      />
      <SavedNotice>{message}</SavedNotice>
      {saved ? (
        <Panel
          title="Frozen protocol"
          subtitle="Assignments and scientific settings are immutable. Copy this configuration to create another draft."
          actions={<Badge tone="purple">Frozen</Badge>}
        >
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
              targetRequest.current += 1;
              setSpec(saved.manifest.spec as ProtocolSpec);
              setSeedsText((saved.manifest.spec as ProtocolSpec).split.seeds.join(', '));
              setDraft(null);
              setPreview(null);
              setShowSaved(null);
              setName(`${w.project.name} protocol copy`);
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
              <Badge>Execution unavailable</Badge>
            </div>
            {preflight?.protocolId === saved.id ? (
              <Findings findings={preflight.findings} />
            ) : null}
            <p className="muted">
              Rechecks the selected feature files and eligible-slide coverage. Full tensor
              validation and model execution remain unavailable.
            </p>
          </div>
          <details>
            <summary>View saved configuration</summary>
            <pre className="code-block">{JSON.stringify(saved, null, 2)}</pre>
          </details>
        </Panel>
      ) : null}
      {!datasets.data?.datasets.length && !datasets.isPending ? (
        <Panel title="Start with a frozen dataset">
          <EmptyState
            title="No imported dataset yet"
            description="Import and freeze a dataset before configuring labels or patient assignments."
          />
          <a className="btn btn-primary" href="#dataset">
            Go to Dataset <Icon name="arrow" />
          </a>
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
        <Panel
          title="1. Dataset and target"
          subtitle="Original attributes stay intact. The label map defines their meaning for this experiment."
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
                    featureSetId: null,
                    target: { ...spec.target, field: '', ...inferTargetSettings([]) },
                    predictors: [],
                    eligibility: [],
                    split: {
                      ...spec.split,
                      pools: spec.split.version === 3 ? newSplit().pools : undefined,
                      rules: { train: [], val: [], test: [] },
                      imported: resetImportedForDataset(spec.split),
                      domainField: undefined,
                      heldOutDomains: [],
                    },
                  })
                }
              />
            </div>
            {dataset?.manifest.summary?.unlinkedSlideCount ? (
              <div className="callout callout-warning">
                <strong>
                  {dataset.manifest.summary.unlinkedSlideCount} slides have unresolved
                  Patient_ID.
                </strong>{' '}
                Revise this dataset to supply a patient mapping or explicitly confirm Slide ID
                fallback. Fallback creates one group per unresolved slide; it cannot establish
                which slides belong to the same patient.
              </div>
            ) : null}
            <div className="science-grid-two">
              <label className="label">
                Target attribute
                <select
                  className="field"
                  value={spec.target.field}
                  onChange={(event) => void chooseTarget(event.target.value)}
                >
                  <option value="">Choose a target</option>
                  {columns.map((column) => (
                    <option key={column}>{column}</option>
                  ))}
                </select>
              </label>
              <label className="label">
                Task
                <select
                  className="field"
                  value={spec.target.task}
                  onChange={(event) =>
                    target({
                      task: event.target.value as ProtocolSpec['target']['task'],
                      positiveClass:
                        event.target.value === 'binary_classification'
                          ? (preservePositiveClass(
                              spec.target.positiveClass,
                              spec.target.classes,
                            ) ?? spec.target.classes[0])
                          : undefined,
                    })
                  }
                >
                  <option value="">Choose a task</option>
                  <option value="binary_classification">Binary classification</option>
                  <option value="multiclass_classification">Multiclass classification</option>
                </select>
              </label>
              <label className="label">
                Label unit
                <select
                  className="field"
                  value={spec.target.unit}
                  onChange={(event) =>
                    target({ unit: event.target.value as 'patient' | 'slide' })
                  }
                >
                  <option value="patient">
                    Patient — consistent label across their slides
                  </option>
                  <option value="slide">Slide / case — keep known patients together</option>
                </select>
              </label>
              <label className="label">
                Class names, separated by |
                <input
                  className="field"
                  value={spec.target.classes.join(' | ')}
                  onChange={(event) =>
                    target({
                      classes:
                        event.target.value === ''
                          ? []
                          : event.target.value.split('|').map((value) => value.trim()),
                      positiveClass: preservePositiveClass(
                        spec.target.positiveClass,
                        event.target.value.split('|').map((value) => value.trim()),
                      ),
                    })
                  }
                  placeholder="Filled from the selected target"
                />
              </label>
              <label className="label">
                Positive class
                <select
                  className="field"
                  value={
                    spec.target.task === 'binary_classification'
                      ? (spec.target.positiveClass ?? '')
                      : ''
                  }
                  disabled={spec.target.task !== 'binary_classification'}
                  onChange={(event) =>
                    target({ positiveClass: event.target.value || undefined })
                  }
                >
                  <option value="">
                    {spec.target.task === 'multiclass_classification'
                      ? 'Not used for multiclass'
                      : 'Choose positive class'}
                  </option>
                  {spec.target.task === 'binary_classification'
                    ? spec.target.classes.map((value, index) => (
                        <option key={index}>{value}</option>
                      ))
                    : null}
                </select>
              </label>
            </div>
            {spec.target.field ? (
              <div className="stack">
                <FieldProfile {...fieldContext} field={spec.target.field} />
                {labelValues.isPending ? (
                  <p className="protocol-live-status" role="status">
                    Reading target values…
                  </p>
                ) : labelValues.data ? (
                  <DistributionBars
                    values={labelValues.data.valueCounts}
                    caption={`${spec.target.field} · source values across all dataset slides`}
                    distinctCount={
                      labelValues.data.valuesTruncated
                        ? undefined
                        : labelValues.data.valueCounts.length
                    }
                  />
                ) : null}
              </div>
            ) : null}
            <div className="science-subheading">
              <h3>Explicit label mapping</h3>
              <button
                type="button"
                className="btn btn-secondary btn-small"
                disabled={!rawValues.length || labelValues.data?.valuesTruncated}
                onClick={() =>
                  target({
                    ...inferTargetSettings(rawValues),
                    positiveClass:
                      rawValues.length === 2
                        ? (preservePositiveClass(spec.target.positiveClass, rawValues) ??
                          rawValues[0])
                        : undefined,
                  })
                }
              >
                Use observed source values
              </button>
            </div>
            <p className="muted">
              Selecting a target fills the task, class names and label mapping from its source
              values. For binary classification, check the suggested positive class. You can
              edit these settings before previewing.
            </p>
            <ErrorNotice error={labelValues.error} />
            {labelValues.data?.valuesTruncated ? (
              <p className="callout callout-warning">
                This field has more than 200 distinct source values. Choose a categorical target
                or enter the task, classes and label mapping yourself.
              </p>
            ) : null}
            {spec.target.field &&
            labelValues.data &&
            !labelValues.data.valuesTruncated &&
            rawValues.length < 2 ? (
              <p className="callout callout-warning">
                {rawValues.length
                  ? 'This target has only one non-missing value.'
                  : 'This target has no non-missing values.'}{' '}
                Choose a target with at least two classes for classification.
              </p>
            ) : null}
            <div className="science-label-map">
              {Object.entries(spec.target.labels).map(([raw, mapped], index) => (
                <div className="science-label-row" key={index}>
                  <label className="label">
                    Source value
                    <input
                      className="field"
                      value={raw}
                      onChange={(event) =>
                        target({
                          labels: Object.fromEntries(
                            Object.entries(spec.target.labels).map(([key, value]) => [
                              key === raw ? event.target.value : key,
                              value,
                            ]),
                          ),
                        })
                      }
                    />
                  </label>
                  <Icon name="arrow" />
                  <label className="label">
                    Class
                    <select
                      className="field"
                      value={mapped}
                      onChange={(event) =>
                        target({
                          labels: {
                            ...spec.target.labels,
                            [raw]: event.target.value,
                          },
                        })
                      }
                    >
                      <option value="">Choose class</option>
                      {spec.target.classes.map((value, at) => (
                        <option key={at}>{value}</option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`Remove mapping ${raw}`}
                    onClick={() =>
                      target({
                        labels: Object.fromEntries(
                          Object.entries(spec.target.labels).filter(([key]) => key !== raw),
                        ),
                      })
                    }
                  >
                    <Icon name="close" />
                  </button>
                </div>
              ))}
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-small science-fit"
              onClick={() => {
                const raw =
                  rawValues.find((value) => !(value in spec.target.labels)) ??
                  `value_${Object.keys(spec.target.labels).length + 1}`;
                target({
                  labels: {
                    ...spec.target.labels,
                    [raw]: spec.target.classes[0] ?? '',
                  },
                });
              }}
            >
              <Icon name="plus" size={15} /> Add label mapping
            </button>
            <div className="science-grid-two">
              <label className="label">
                Missing target values
                <select
                  className="field"
                  value={spec.target.missing}
                  onChange={(event) =>
                    target({
                      missing: event.target.value as 'block' | 'exclude',
                    })
                  }
                >
                  <option value="block">Block until resolved</option>
                  <option value="exclude">Exclude and record the reason</option>
                </select>
              </label>
              <label className="label">
                Unmapped target values
                <select
                  className="field"
                  value={spec.target.unmapped}
                  onChange={(event) =>
                    target({
                      unmapped: event.target.value as 'block' | 'exclude',
                    })
                  }
                >
                  <option value="block">Block until resolved</option>
                  <option value="exclude">Exclude and record the reason</option>
                </select>
              </label>
            </div>
          </div>
        </Panel>
        <Panel
          title="2. Choose slides and optional extra inputs"
          subtitle="The cohort is the set of slides included in this experiment. Start with all slides, then add conditions only if you want a smaller group."
        >
          <div className="stack">
            <ConditionEditor
              title="Which slides should be included?"
              description="Optional. Keep every condition true on the same slide. For example, Age is at least 18 includes only slides whose linked age meets that condition."
              emptyMessage="All dataset slides are included. Add a condition only to narrow the cohort."
              conditions={spec.eligibility}
              columns={columns}
              fieldContext={fieldContext}
              onChange={(eligibility) => edit({ eligibility })}
            />
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
                  These counts apply eligibility conditions only. Missing labels, label mappings
                  and final assignment constraints are checked in Preview & preflight.
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
                  fields={[...spec.eligibility.map((item) => item.field), spec.target.field]}
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
            <details
              className="protocol-extra-inputs"
              open={spec.predictors.length > 0 ? true : undefined}
            >
              <summary>
                Extra spreadsheet inputs (optional)
                {spec.predictors.length
                  ? ` · ${spec.predictors.length} selected`
                  : ' · none selected'}
              </summary>
              <p className="muted">
                Choose extra columns for the model, such as age. Leave empty to use slide image
                features only.
              </p>
              <p className="muted">
                These choices do not filter slides. Avoid IDs, split columns, and columns that
                reveal the target.
              </p>
              <div className="science-checkbox-grid">
                {dictionary.map((item) => (
                  <label className="science-check" key={item.key}>
                    <input
                      type="checkbox"
                      checked={spec.predictors.includes(item.key)}
                      disabled={item.key === spec.target.field}
                      onChange={(event) =>
                        edit({
                          predictors: event.target.checked
                            ? [...spec.predictors, item.key]
                            : spec.predictors.filter((value) => value !== item.key),
                        })
                      }
                    />
                    <span>
                      {item.key}
                      <small>
                        {item.owner === 'patient' ? 'Patient' : 'Slide / case'} ·{' '}
                        {item.type.replaceAll('_', ' ')}
                        {item.key === spec.target.field ? ' · prediction target' : ''}
                      </small>
                    </span>
                  </label>
                ))}
              </div>
              {spec.predictors.map((field) => (
                <FieldProfile key={field} {...fieldContext} field={field} />
              ))}
            </details>
            <label className="label">
              Slide image features (optional for saving this protocol)
              <select
                className="field"
                value={spec.featureSetId ?? ''}
                onChange={(event) => edit({ featureSetId: event.target.value || null })}
              >
                <option value="">Choose later — protocol can be saved without features</option>
                {features.data?.configurations
                  .filter((item) => item.manifest.datasetId === spec.datasetId)
                  .map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.id.slice(-10)} ·{' '}
                      {(item.manifest.spec as { encoderId?: string }).encoderId ??
                        'Existing HDF5 features'}
                    </option>
                  ))}
              </select>
            </label>
            <p className="muted">
              Choose the image embeddings attached in PFM & features, such as UNI patch
              features. This is separate from the optional spreadsheet inputs above.
            </p>
          </div>
        </Panel>
        <Panel
          title="3. Assign training, validation and test sets"
          subtitle="Known patients stay together. A matching slide selects all eligible slides in its group. Confirmed Slide ID fallbacks each form a separate group."
        >
          <div className="stack">
            {(spec.split.version ?? 1) >= 2 ? (
              <>
                {spec.split.version === 2 ? (
                  <div className="callout">
                    <p>
                      This saved design keeps its existing assignments. Create a new draft to
                      define separate training and final test sets.
                    </p>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => {
                        edit({
                          split: {
                            ...spec.split,
                            version: 3,
                            pools: newSplit().pools,
                            rules: { train: [], val: [], test: [] },
                            imported: undefined,
                          },
                        });
                        setDraft(null);
                      }}
                    >
                      Create a draft with training and test selections
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
                  pools={
                    spec.split.pools ? (
                      <SplitPools
                        pools={spec.split.pools}
                        validationFraction={
                          spec.split.validationFraction ??
                          validationFractionDefault(spec.split.version)
                        }
                        onFractionChange={(validationFraction) => split({ validationFraction })}
                        onChange={(update, validationFraction) =>
                          split({
                            pools: { ...spec.split.pools!, ...update },
                            ...(validationFraction === undefined ? {} : { validationFraction }),
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
                                ? 'Training set conditions (required)'
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
                            emptyMessage="Add a condition to define this set."
                            columns={columns}
                            fieldContext={fieldContext}
                            conditions={spec.split.pools!.rules[role]}
                            onChange={(conditions) =>
                              split({
                                pools: {
                                  ...spec.split.pools!,
                                  rules: { ...spec.split.pools!.rules, [role]: conditions },
                                },
                              })
                            }
                          />
                        )}
                      />
                    ) : null
                  }
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
                                ...spec.split.rules[role].map((item) => item.field),
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
                        All conditions must match on a slide; its whole group is selected.
                        Overlapping selections block freezing.
                      </p>
                      {live.data?.unassigned?.totalSlides ? (
                        <p className="callout callout-warning">
                          {live.data.unassigned.totalSlides} eligible slides remain unassigned.
                          Broaden the training rules or leave them empty.
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
                    This saved protocol keeps its original split behavior. Its K-fold validation
                    folds are not reported test folds.
                  </p>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => {
                      edit({ split: newSplit(spec.split.seeds, spec.split.folds) });
                      setDraft(null);
                      setMessage(
                        'Created a new draft with K-fold test rotation and early-stop validation. Review the settings before freezing.',
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
                        {role === 'val' ? 'Validation' : role === 'train' ? 'Training' : 'Test'}{' '}
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
                          role === 'val' ? 'validation' : role === 'train' ? 'training' : 'test'
                        }
                        fields={[
                          ...spec.split.rules[role].map((item) => item.field),
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
                  Within a rule group, all conditions must match. A matching eligible slide
                  reserves its whole group. Overlapping test, validation and training selections
                  block freezing. Counts above are before target-label exclusions and final
                  feasibility checks.
                </p>
                {live.data?.unassigned && live.data.unassigned.totalSlides > 0 ? (
                  <p className="callout callout-warning">
                    {live.data.unassigned.totalSlides} eligible slides remain outside the fixed
                    sets.
                    {spec.split.mode === 'rules'
                      ? ' Broaden the training conditions or leave them empty to include the remainder.'
                      : ' Their final roles are determined by the selected split strategy in Preview & preflight.'}
                  </p>
                ) : null}
              </>
            )}
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
              A group is one verified patient or one confirmed Slide ID fallback. These minimums
              check feasibility in each required training, early-stop validation, tuning and
              test set.
            </p>
          </div>
        </Panel>
        <div className="science-savebar">
          <div>
            <strong>
              {draft
                ? `Revision ${draft.revision} · ${frozen ? 'Frozen protocol' : dirty ? 'Unsaved changes' : 'Saved draft'}`
                : 'New protocol draft'}
            </strong>
            <p>Changing any setting invalidates the current preview.</p>
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
            <button
              type="button"
              className="btn btn-primary"
              disabled={!spec.datasetId || !spec.target.field || !name.trim() || !seedsValid}
              onClick={() =>
                void run(async () => {
                  const current = await save();
                  setPreview(
                    await scientific.protocolPreview(project, current.id, current.revision),
                  );
                })
              }
            >
              {busy ? 'Validating…' : 'Preview & preflight'} <Icon name="arrow" />
            </button>
          </div>
        </div>
      </fieldset>
      {preview ? (
        <Panel
          title="Preflight and partition review"
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
              note={`${preview.summary.totalSlides} total dataset slides`}
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
            Training execution is not connected. Freezing preserves this analysis design and its
            assignments; it does not start a job.
          </div>
          <div className="science-savebar">
            <p>Freeze after reviewing every blocking finding and partition.</p>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || dirty || frozen || !preview.canFreeze}
              onClick={() =>
                void run(async () => {
                  const result = await scientific.protocolFreeze(
                    project,
                    draft!.id,
                    draft!.revision,
                    preview.previewHash,
                  );
                  setDraft(await scientific.draft(project, draft!.id));
                  await refresh();
                  setShowSaved(result.id);
                  setPreview(null);
                  setMessage(
                    'Protocol frozen. Group assignments and target settings are preserved.',
                  );
                  window.scrollTo({ top: 0 });
                })
              }
            >
              <Icon name="lock" /> {busy ? 'Freezing…' : 'Freeze protocol'}
            </button>
          </div>
        </Panel>
      ) : null}
    </>
  );
}
function ConditionEditor({
  title,
  description,
  emptyMessage,
  conditions,
  columns,
  fieldContext,
  onChange,
}: {
  title: string;
  description: string;
  emptyMessage: string;
  conditions: Condition[];
  columns: string[];
  fieldContext: ProtocolFieldContext;
  onChange: (conditions: Condition[]) => void;
}) {
  const update = (index: number, changed: Partial<Condition>) =>
    onChange(
      conditions.map((condition, at) =>
        at === index ? { ...condition, ...changed } : condition,
      ),
    );
  const fields = ['Slide_ID', 'Patient_ID', ...columns];
  return (
    <div className="condition-editor">
      <div className="science-subheading">
        <h3>{title}</h3>
        <button
          type="button"
          className="btn btn-secondary btn-small"
          onClick={() =>
            onChange([...conditions, { field: columns[0] ?? 'Slide_ID', op: 'eq', value: '' }])
          }
        >
          <Icon name="plus" size={14} /> Add condition
        </button>
      </div>
      <p className="muted">{description}</p>
      {conditions.length ? (
        conditions.map((condition, index) => (
          <div className="protocol-condition" key={index}>
            <div className="condition-row">
              <label className="label">
                Field
                <select
                  className="field"
                  value={condition.field}
                  onChange={(event) => update(index, { field: event.target.value })}
                >
                  {fields.map((field) => (
                    <option key={field}>{field}</option>
                  ))}
                </select>
              </label>
              <label className="label">
                Condition
                <select
                  className="field"
                  value={condition.op}
                  onChange={(event) => {
                    const op = event.target.value as Condition['op'];
                    update(index, {
                      op,
                      value: op === 'exists' ? true : ['in', 'not_in'].includes(op) ? [] : '',
                    });
                  }}
                >
                  {[
                    ['eq', 'equals'],
                    ['ne', 'does not equal'],
                    ['in', 'is one of'],
                    ['not_in', 'is not one of'],
                    ['regex', 'matches regex'],
                    ['lt', 'less than'],
                    ['lte', 'at most'],
                    ['gt', 'greater than'],
                    ['gte', 'at least'],
                    ['exists', 'is present / missing'],
                  ].map(([op, label]) => (
                    <option key={op} value={op}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="label">
                {['in', 'not_in'].includes(condition.op) ? 'Values, separated by |' : 'Value'}
                {condition.op === 'exists' ? (
                  <select
                    className="field"
                    value={String(condition.value)}
                    onChange={(event) =>
                      update(index, { value: event.target.value === 'true' })
                    }
                  >
                    <option value="true">Is present</option>
                    <option value="false">Is missing</option>
                  </select>
                ) : (
                  <input
                    className="field"
                    value={
                      Array.isArray(condition.value)
                        ? condition.value.join(' | ')
                        : String(condition.value ?? '')
                    }
                    placeholder={condition.op === 'regex' ? '^pattern$' : ''}
                    onChange={(event) => {
                      const text = event.target.value;
                      let value: ConditionValue = text;
                      if (['in', 'not_in'].includes(condition.op))
                        value = text.split('|').map((item) => item.trim());
                      else if (['lt', 'lte', 'gt', 'gte'].includes(condition.op))
                        value = text === '' ? '' : Number(text);
                      update(index, { value });
                    }}
                  />
                )}
              </label>
              <button
                type="button"
                className="icon-button"
                aria-label={`Remove ${title} condition ${index + 1}`}
                onClick={() => onChange(conditions.filter((_, at) => at !== index))}
              >
                <Icon name="close" />
              </button>
            </div>
            <FieldProfile {...fieldContext} field={condition.field} />
            {condition.op === 'regex' ? (
              <small className="muted">
                Regular expression, matched as text. For example, ^2$ matches only 2; ^T matches
                values beginning with T.
              </small>
            ) : ['lt', 'lte', 'gt', 'gte'].includes(condition.op) ? (
              <small className="muted">
                Enter a number. Non-numeric source values must be corrected before using this
                comparison.
              </small>
            ) : ['in', 'not_in'].includes(condition.op) ? (
              <small className="muted">
                Separate exact values with |, for example low | high.
              </small>
            ) : null}
          </div>
        ))
      ) : (
        <p className="muted">{emptyMessage}</p>
      )}
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
                      test: 'test',
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
  label,
  value,
  numeric,
  onChange,
}: {
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
                {['train', 'val', 'test', 'trainval'].map((role) => (
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
function PlanSummary({ summary }: { summary: ProtocolPreview['summary'] }) {
  if ((summary.splitVersion ?? 1) < 2) return null;
  const coverage = summary.oofCoverage;
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
          {summary.splitVersion === 3 ? 'CV assessment' : 'evaluation'}{' '}
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
      {summary.poolCounts && summary.finalPlanCount ? (
        <div className="science-metrics">
          {(['train', 'val', 'test'] as const).map((role) => (
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
              summary.splitVersion === 3
                ? 'Planned CV assessment coverage'
                : 'Planned test coverage'
            }
            value={coverage.testedGroups}
            max={Math.max(1, coverage.groups)}
          />
          <span>
            <strong>
              {summary.splitVersion === 3
                ? 'CV assessment coverage within training:'
                : 'Planned test coverage:'}
            </strong>{' '}
            {coverage.testedGroups} of {coverage.groups} groups · {coverage.minTestAppearances}–
            {coverage.maxTestAppearances} {summary.splitVersion === 3 ? 'assessment' : 'test'}{' '}
            appearances per group per seed.
          </span>
        </div>
      ) : null}
      <p className="muted">
        These are planned assignments. Predictions and model selection are performed later in
        MIL experiments.
      </p>
    </div>
  );
}
export function PartitionTable({ partitions }: { partitions: ProtocolPreview['partitions'] }) {
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
              {explicitPools ? 'CV assessment / final test' : modern ? 'Reported test' : 'Test'}
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
                          ? `${explicitPools ? 'Assessment' : 'Test'}: ${partition.domain}`
                          : partition.repeat !== undefined
                            ? `Repeat ${partition.repeat + 1}`
                            : partition.fold === null
                              ? 'Held-out evaluation'
                              : `${explicitPools ? 'Assessment fold' : modern ? 'Test fold' : 'Fold'} ${partition.fold + 1}`}
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
                            ? explicitPools
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
