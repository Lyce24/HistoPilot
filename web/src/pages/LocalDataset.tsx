import { useRef, useState } from 'react';
import type { Workspace } from '../api/types';
import type {
  AttributeMapping,
  ImportPreview,
  ImportSpec,
  Inspection,
  ScientificDraft,
} from '../api/scientific';
import { scientific } from '../api/scientific';
import { sameJSON } from '../lib/json';
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
  DatasetSelect,
  DictionaryEditor,
  DraftSelect,
  Findings,
  RecordExplorer,
  SavedNotice,
  SourceFields,
  useDatasets,
  useDrafts,
  useRefreshScientific,
} from '../components/ScientificUI';
import DatasetExplorer from '../components/DatasetExplorer';
import ServerFolderPicker from '../components/ServerFolderPicker';
import PatientTableOption from '../components/PatientTableOption';
import PatientFallbackDialog from '../components/PatientFallbackDialog';

const newSpec = (workspace: Workspace): ImportSpec => ({
  source: { path: '' },
  slideIdColumn: '',
  slideRoot: workspace.sources.find((source) => source.role === 'slides')?.path,
  recursive: true,
  includeMissingSlides: false,
  missingValues: [''],
  attributes: [],
  patientIdFallback: 'unresolved',
});
export default function LocalDataset({ workspace: w }: { workspace: Workspace }) {
  const project = w.project.id;
  const versions = useDatasets(project);
  const savedDrafts = useDrafts(project);
  const refresh = useRefreshScientific(project);
  const [view, setView] = useState<'import' | 'dataset'>(w.dataset.id ? 'dataset' : 'import');
  const [versionId, setVersionId] = useState(w.dataset.id);
  const [name, setName] = useState(`${w.project.name} dataset`);
  const [spec, setSpec] = useState<ImportSpec>(() => newSpec(w));
  const [draft, setDraft] = useState<ScientificDraft<ImportSpec> | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [patientInspection, setPatientInspection] = useState<Inspection | null>(null);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [fallbackCount, setFallbackCount] = useState<number | null>(null);
  const [readingSource, setReadingSource] = useState<'main' | 'patient' | null>(null);
  const sourcesSection = useRef<HTMLDivElement>(null);
  const mappingSection = useRef<HTMLDivElement>(null);
  const reviewSection = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const datasetId = versionId || versions.data?.datasets[0]?.id || '';
  const version = versions.data?.datasets.find((item) => item.id === datasetId);
  const dirty = !draft || draft.name !== name || !sameJSON(draft.payload.spec, spec);
  const frozen = draft?.status === 'frozen';
  function goTo(section: HTMLDivElement | null) {
    section?.scrollIntoView({ block: 'start' });
    section?.focus({ preventScroll: true });
  }
  function showStep(next: 1 | 2 | 3) {
    setStep(next);
    requestAnimationFrame(() =>
      goTo(
        next === 1
          ? sourcesSection.current
          : next === 2
            ? mappingSection.current
            : reviewSection.current,
      ),
    );
  }
  const columns = inspection?.headers ?? [
    ...new Set(
      [
        spec.slideIdColumn,
        spec.patientIdColumn,
        ...spec.attributes.map((item) => item.sourceColumn),
      ].filter((v): v is string => Boolean(v)),
    ),
  ];
  const patientColumns = patientInspection?.headers ?? [
    ...new Set(
      [
        spec.patientSourceSlideIdColumn,
        spec.patientSourcePatientIdColumn,
        ...(spec.patientAttributes ?? []).map((item) => item.sourceColumn),
      ].filter((v): v is string => Boolean(v)),
    ),
  ];
  function edit(update: Partial<ImportSpec>) {
    const identityChanges = [
      'source',
      'slideIdColumn',
      'patientIdColumn',
      'patientSource',
      'patientSourceKind',
      'patientSourceSlideIdColumn',
      'patientSourcePatientIdColumn',
      'slideRoot',
      'recursive',
      'includeMissingSlides',
      'missingValues',
    ];
    setSpec((current) => ({
      ...current,
      ...update,
      ...(identityChanges.some(
        (key) =>
          key in update &&
          !sameJSON(current[key as keyof ImportSpec], update[key as keyof ImportSpec]),
      )
        ? { patientIdFallback: 'unresolved' as const }
        : {}),
    }));
    setPreview(null);
    setMessage('');
    setError(null);
  }
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setMessage('');
    try {
      await action();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason : new Error('This operation could not be completed.'),
      );
    } finally {
      setBusy(false);
      setReadingSource(null);
    }
  }
  function reset() {
    setDraft(null);
    setSpec(newSpec(w));
    setName(`${w.project.name} dataset`);
    setInspection(null);
    setPatientInspection(null);
    setPreview(null);
    setError(null);
    setMessage('');
    setView('import');
    setFallbackCount(null);
    showStep(1);
  }
  async function save(nextSpec = spec) {
    if (draft && draft.name === name && sameJSON(draft.payload.spec, nextSpec)) return draft;
    const next = await scientific.saveDraft(
      project,
      { kind: 'import', name, payload: { type: 'dataset-import', spec: nextSpec } },
      draft ?? undefined,
    );
    setDraft(next);
    setName(next.name);
    await refresh();
    return next;
  }
  async function previewDataset(nextSpec = spec) {
    const current = await save(nextSpec);
    const next = await scientific.importPreview(project, current.id, current.revision);
    if (next.summary.unlinkedSlideCount > 0 && nextSpec.patientIdFallback !== 'slide_id') {
      setPreview(null);
      setFallbackCount(next.summary.unlinkedSlideCount);
      return;
    }
    setFallbackCount(null);
    setPreview(next);
    showStep(3);
  }
  async function inspect() {
    setInspection(null);
    setPreview(null);
    const next = await scientific.inspect(project, spec.source);
    setInspection(next);
    const slide = next.headers.includes(spec.slideIdColumn)
      ? spec.slideIdColumn
      : next.headers.includes('Slide_ID')
        ? 'Slide_ID'
        : next.headers.includes('De ID')
          ? 'De ID'
          : '';
    const patient =
      spec.patientIdColumn && next.headers.includes(spec.patientIdColumn)
        ? spec.patientIdColumn
        : next.headers.includes('Patient_ID')
          ? 'Patient_ID'
          : undefined;
    const attributes: AttributeMapping[] = next.headers
      .filter((column) => column !== slide && column !== patient)
      .map(
        (column) =>
          spec.attributes.find((item) => item.sourceColumn === column) ?? {
            key: column,
            sourceColumn: column,
            owner: 'slide',
            type: 'text',
          },
      );
    edit({
      slideIdColumn: slide,
      patientIdColumn: patient,
      attributes,
      ...(inspection && inspection.fingerprint !== next.fingerprint
        ? { patientIdFallback: 'unresolved' as const }
        : {}),
    });
  }
  return (
    <>
      <PageHeader
        eyebrow="DATA FOUNDATION"
        title="Dataset workspace"
        description="Connect your tables and slides, review identities and attributes, then preserve a dataset version."
        actions={
          <button type="button" className="btn btn-primary" disabled={busy} onClick={reset}>
            <Icon name="plus" /> New import
          </button>
        }
      />
      <div className="science-toolbar">
        <DatasetSelect
          versions={versions.data?.datasets ?? []}
          value={view === 'dataset' ? datasetId : ''}
          allowEmpty
          disabled={busy}
          onChange={(id) => {
            if (id) {
              setVersionId(id);
              setView('dataset');
            }
          }}
        />
        <DraftSelect
          drafts={(savedDrafts.data?.drafts ?? []).filter(
            (item) => item.payload.type === 'dataset-import',
          )}
          value={draft?.id ?? ''}
          disabled={busy}
          onChange={(id) => {
            if (!id) {
              reset();
              return;
            }
            void run(async () => {
              const loaded = await scientific.draft<ImportSpec>(project, id);
              setDraft(loaded);
              setSpec({
                ...newSpec(w),
                ...loaded.payload.spec,
                attributes: loaded.payload.spec.attributes ?? [],
              });
              setName(loaded.name);
              setInspection(null);
              setPatientInspection(null);
              setPreview(null);
              setView('import');
              setFallbackCount(null);
              showStep(1);
            });
          }}
        />
      </div>
      <ErrorNotice error={error ?? versions.error ?? savedDrafts.error} />
      <SavedNotice>{message}</SavedNotice>
      {view === 'dataset' ? (
        version ? (
          <>
            <div className="science-version">
              <Badge tone="purple">
                <Icon name="lock" size={12} /> Frozen dataset
              </Badge>
              <strong>{version.manifest.name ?? 'Dataset version'}</strong>
              <span className="mono">{version.id}</span>
              <small>{new Date(version.createdAt).toLocaleString()}</small>
            </div>
            {version.manifest.summary ? (
              <ImportMetrics summary={version.manifest.summary} />
            ) : null}
            <Panel
              title="Explore your dataset"
              subtitle="Attributes remain separate from targets and model inputs."
            >
              <DatasetExplorer
                key={datasetId}
                project={project}
                datasetId={datasetId}
                dictionary={version.manifest.dictionary ?? []}
              />
            </Panel>
            <Panel
              title="Dataset provenance"
              subtitle="A frozen version retains source fingerprints, mapping decisions and artifact hashes."
            >
              <button
                type="button"
                className="btn btn-secondary btn-small"
                disabled={busy}
                onClick={() => {
                  const provenance = version.manifest.provenance as
                    { mapping?: ImportSpec } | undefined;
                  const mapping = provenance?.mapping;
                  setSpec({
                    ...newSpec(w),
                    ...mapping,
                    patientIdFallback: 'unresolved',
                    parentId: version.id,
                    source: mapping?.source.path
                      ? mapping.source
                      : { filename: mapping?.source.filename ?? '', contentBase64: '' },
                    attributes: mapping?.attributes ?? [],
                    patientSource: mapping?.patientSource
                      ? mapping.patientSource.path
                        ? mapping.patientSource
                        : { filename: mapping.patientSource.filename ?? '', contentBase64: '' }
                      : undefined,
                  });
                  setDraft(null);
                  setPreview(null);
                  setInspection(null);
                  setPatientInspection(null);
                  setName(`${w.project.name} revised dataset`);
                  setView('import');
                  setFallbackCount(null);
                  showStep(1);
                  setMessage(
                    'New import based on this dataset. Review mappings and reselect uploaded metadata if needed; the original version remains frozen.',
                  );
                }}
              >
                Revise as a new dataset version
              </button>
              <details>
                <summary>View saved provenance</summary>
                <pre className="code-block">
                  {JSON.stringify(
                    {
                      id: version.id,
                      createdAt: version.createdAt,
                      contentHash: version.contentHash,
                      provenance: version.manifest.provenance,
                      artifacts: version.artifacts,
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            </Panel>
            <a className="btn btn-primary" href="#cohort">
              Configure target & split <Icon name="arrow" />
            </a>
          </>
        ) : (
          <EmptyState
            title="No dataset version selected"
            description="Create an import or choose a frozen version above."
          />
        )
      ) : (
        <>
          <nav className="dataset-steps" aria-label="Dataset creation steps">
            <button
              type="button"
              disabled={busy}
              aria-current={step === 1 ? 'step' : undefined}
              onClick={() => showStep(1)}
            >
              <span>1</span>
              <strong>Choose files</strong>
              <small>{inspection ? 'File read' : 'Start here'}</small>
            </button>
            <button
              type="button"
              disabled={busy || !columns.length}
              aria-current={step === 2 ? 'step' : undefined}
              onClick={() => showStep(2)}
            >
              <span>2</span>
              <strong>Map columns</strong>
              <small>
                {columns.length ? 'Review IDs and attributes' : 'Read your file first'}
              </small>
            </button>
            <button
              type="button"
              disabled={busy || !preview}
              aria-current={step === 3 ? 'step' : undefined}
              onClick={() => showStep(3)}
            >
              <span>3</span>
              <strong>Preview & freeze</strong>
              <small>{preview ? 'Review your dataset' : 'Preview after mapping'}</small>
            </button>
          </nav>
          {frozen ? (
            <div className="callout">
              This import draft is frozen.{' '}
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  setDraft(null);
                  setPreview(null);
                  setSpec((current) => ({ ...current, patientIdFallback: 'unresolved' }));
                  setMessage('Copied into a new editable import. Save when ready.');
                }}
              >
                Copy into a new draft
              </button>
            </div>
          ) : null}
          <fieldset className="science-fieldset" disabled={busy || frozen}>
            <div
              className="dataset-section"
              ref={sourcesSection}
              hidden={step !== 1}
              tabIndex={-1}
              aria-label="Choose dataset files"
            >
              <Panel
                title="1. Choose your metadata and slides"
                subtitle="Select a CSV or XLSX, then read it to see its columns. Add a slide folder here or keep rows for an existing feature store."
              >
                <div className="stack">
                  <label className="label">
                    Dataset name
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
                  <SourceFields
                    label="Metadata"
                    source={spec.source}
                    inspection={inspection}
                    busy={busy}
                    reading={readingSource === 'main'}
                    onChange={(source) => {
                      edit({ source });
                      setInspection(null);
                    }}
                    onInspect={() => {
                      setReadingSource('main');
                      void run(inspect);
                    }}
                  />
                  {inspection?.findings.length ? (
                    <Findings findings={inspection.findings} />
                  ) : null}
                  <div className="science-grid-two">
                    <label className="label">
                      Slide folder (optional)
                      <input
                        className="field mono"
                        value={spec.slideRoot ?? ''}
                        placeholder="/path/to/slides"
                        onChange={(event) =>
                          edit({ slideRoot: event.target.value || undefined })
                        }
                      />
                    </label>
                    <div className="science-field-actions">
                      <ServerFolderPicker
                        label="Browse slide folders"
                        onSelect={(slideRoot) => edit({ slideRoot })}
                      />
                    </div>
                  </div>
                  <label className="science-check">
                    <input
                      type="checkbox"
                      checked={spec.recursive}
                      onChange={(event) => edit({ recursive: event.target.checked })}
                    />{' '}
                    Include subfolders in the slide scan
                  </label>
                  <label className="science-check">
                    <input
                      type="checkbox"
                      checked={spec.includeMissingSlides}
                      onChange={(event) => edit({ includeMissingSlides: event.target.checked })}
                    />{' '}
                    Keep metadata rows without matching slide files (for example, existing
                    features)
                  </label>
                  <p className="muted">
                    When this option is off, unmatched metadata rows are listed as exclusions.
                    Exact filenames are joined using the declared slide ID and a recognized file
                    extension.
                  </p>
                  <div className="dataset-source-next">
                    <p className="muted">
                      {spec.slideRoot || spec.includeMissingSlides
                        ? 'Next, review the ID columns and attributes from your metadata.'
                        : 'Choose a slide folder, or select Keep metadata rows above if you are starting from existing features.'}
                    </p>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      disabled={!name.trim()}
                      onClick={() =>
                        void run(async () => {
                          await save();
                          setMessage('Draft saved. Continue here or reopen it later.');
                        })
                      }
                    >
                      Save draft
                    </button>
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={
                        !inspection ||
                        inspection.findings.some((finding) => finding.severity === 'error')
                      }
                      onClick={() => showStep(2)}
                    >
                      Continue to column mapping <Icon name="arrow" size={16} />
                    </button>
                  </div>
                </div>
              </Panel>
            </div>
            <div
              className="dataset-section dataset-mapping-section"
              ref={mappingSection}
              hidden={step !== 2}
              tabIndex={-1}
              aria-label="Map dataset columns"
            >
              <Panel
                title="2. Tell us what each column means"
                subtitle="Choose the slide and patient identifiers, then review the attributes and their example values. Labels and model inputs are chosen later in Target & split."
              >
                {columns.length ? (
                  <div className="stack">
                    <div className="science-grid-two">
                      <label className="label">
                        Slide_ID source column
                        <select
                          className="field"
                          value={spec.slideIdColumn}
                          onChange={(event) =>
                            edit({
                              slideIdColumn: event.target.value,
                              attributes: spec.attributes.filter(
                                (item) => item.sourceColumn !== event.target.value,
                              ),
                            })
                          }
                        >
                          <option value="">Select a slide identifier</option>
                          {columns.map((column) => (
                            <option key={column}>{column}</option>
                          ))}
                        </select>
                      </label>
                      <label className="label">
                        Patient_ID source column (optional)
                        <select
                          className="field"
                          value={spec.patientIdColumn ?? ''}
                          onChange={(event) =>
                            edit({
                              patientIdColumn: event.target.value || undefined,
                              attributes: spec.attributes.filter(
                                (item) => item.sourceColumn !== event.target.value,
                              ),
                            })
                          }
                        >
                          <option value="">Unresolved — no patient mapping yet</option>
                          {columns
                            .filter((column) => column !== spec.slideIdColumn)
                            .map((column) => (
                              <option key={column}>{column}</option>
                            ))}
                        </select>
                      </label>
                    </div>
                    <div className="callout">
                      {spec.patientIdFallback === 'slide_id'
                        ? 'Slide_ID fallback confirmed. Slides without a patient ID will each form a separate split group; known patients still stay together.'
                        : 'Map Patient_ID if available. If any patient IDs remain missing, the next step asks whether to use Slide_ID for those slides or return here to change the mapping.'}
                      {spec.patientIdFallback === 'slide_id' ? (
                        <button
                          type="button"
                          className="text-button"
                          onClick={() => edit({ patientIdFallback: 'unresolved' })}
                        >
                          Change this choice
                        </button>
                      ) : null}
                    </div>
                    <label className="label">
                      Treat these values as missing (optional)
                      <input
                        className="field"
                        value={spec.missingValues.filter(Boolean).join(' | ')}
                        placeholder="Empty cells are missing; optionally NA | unknown"
                        onChange={(event) =>
                          edit({
                            missingValues: [
                              '',
                              ...event.target.value
                                .split('|')
                                .map((value) => value.trim())
                                .filter(Boolean),
                            ],
                          })
                        }
                      />
                      <small>
                        Empty cells are missing by default. Separate additional values with |,
                        for example unknown | not recorded.
                      </small>
                    </label>
                    <DictionaryEditor
                      attributes={spec.attributes}
                      inspection={inspection}
                      columns={columns.filter(
                        (column) =>
                          column !== spec.slideIdColumn && column !== spec.patientIdColumn,
                      )}
                      onChange={(attributes) => edit({ attributes })}
                    />
                  </div>
                ) : (
                  <div className="dataset-read-first">
                    <EmptyState
                      title="Your column mapping will appear here"
                      description="Choose a metadata file in step 1 and click Read file & show columns. Then choose the ID columns and review the suggested attributes."
                    />
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => showStep(1)}
                    >
                      Back to choose a file
                    </button>
                  </div>
                )}
                <div className="science-crosswalk">
                  <PatientTableOption
                    checked={Boolean(spec.patientSource)}
                    onChange={(checked) => {
                      edit({
                        patientSource: checked ? { path: '' } : undefined,
                        patientSourceKind: 'crosswalk',
                        patientSourceSlideIdColumn: 'Slide_ID',
                        patientSourcePatientIdColumn: 'Patient_ID',
                        patientAttributes: [],
                      });
                      setPatientInspection(null);
                    }}
                  />
                  {spec.patientSource ? (
                    <div className="stack">
                      <label className="label">
                        What does this extra file contain?
                        <select
                          className="field"
                          value={spec.patientSourceKind ?? 'crosswalk'}
                          onChange={(event) =>
                            edit({
                              patientSourceKind: event.target.value as 'crosswalk' | 'patients',
                            })
                          }
                        >
                          <option value="crosswalk">
                            Patient IDs for my slides (Slide_ID → Patient_ID)
                          </option>
                          <option value="patients">
                            Patient attributes (match by Patient_ID)
                          </option>
                        </select>
                      </label>
                      <p className="muted">
                        {spec.patientSourceKind === 'patients'
                          ? 'Use one row per patient, with Patient_ID and attributes such as age. Map Patient_ID in your main table above first.'
                          : 'Use one row per slide with its Slide_ID and verified Patient_ID. Several slides may share one patient.'}
                      </p>
                      <SourceFields
                        label="Patient table"
                        source={spec.patientSource}
                        inspection={patientInspection}
                        busy={busy}
                        reading={readingSource === 'patient'}
                        onChange={(patientSource) => {
                          edit({ patientSource });
                          setPatientInspection(null);
                        }}
                        onInspect={() => {
                          setReadingSource('patient');
                          void run(async () => {
                            setPatientInspection(null);
                            setPreview(null);
                            const inspected = await scientific.inspect(
                              project,
                              spec.patientSource!,
                            );
                            setPatientInspection(inspected);
                            const slide =
                              spec.patientSourceKind === 'patients'
                                ? undefined
                                : spec.patientSourceSlideIdColumn &&
                                    inspected.headers.includes(spec.patientSourceSlideIdColumn)
                                  ? spec.patientSourceSlideIdColumn
                                  : inspected.headers.includes('Slide_ID')
                                    ? 'Slide_ID'
                                    : inspected.headers.includes('De ID')
                                      ? 'De ID'
                                      : '';
                            const patient =
                              spec.patientSourcePatientIdColumn &&
                              inspected.headers.includes(spec.patientSourcePatientIdColumn)
                                ? spec.patientSourcePatientIdColumn
                                : inspected.headers.includes('Patient_ID')
                                  ? 'Patient_ID'
                                  : '';
                            edit({
                              ...(patientInspection &&
                              patientInspection.fingerprint !== inspected.fingerprint
                                ? { patientIdFallback: 'unresolved' as const }
                                : {}),
                              patientSourceSlideIdColumn: slide,
                              patientSourcePatientIdColumn: patient,
                              patientAttributes: inspected.headers
                                .filter((column) => column !== slide && column !== patient)
                                .map(
                                  (column) =>
                                    spec.patientAttributes?.find(
                                      (item) => item.sourceColumn === column,
                                    ) ?? {
                                      key: column,
                                      sourceColumn: column,
                                      owner: 'patient',
                                      type: 'text',
                                    },
                                ),
                            });
                          });
                        }}
                      />
                      {patientInspection ? (
                        <Findings findings={patientInspection.findings} />
                      ) : null}
                      <div className="science-grid-two">
                        {spec.patientSourceKind !== 'patients' ? (
                          <label className="label">
                            Crosswalk slide column
                            <select
                              className="field"
                              value={spec.patientSourceSlideIdColumn ?? ''}
                              onChange={(event) =>
                                edit({ patientSourceSlideIdColumn: event.target.value })
                              }
                            >
                              <option value="">Select column</option>
                              {patientColumns.map((column) => (
                                <option key={column}>{column}</option>
                              ))}
                            </select>
                          </label>
                        ) : null}
                        <label className="label">
                          Patient table Patient_ID column
                          <select
                            className="field"
                            value={spec.patientSourcePatientIdColumn ?? ''}
                            onChange={(event) =>
                              edit({ patientSourcePatientIdColumn: event.target.value })
                            }
                          >
                            <option value="">Select column</option>
                            {patientColumns.map((column) => (
                              <option key={column}>{column}</option>
                            ))}
                          </select>
                        </label>
                      </div>
                      <DictionaryEditor
                        attributes={spec.patientAttributes ?? []}
                        inspection={patientInspection}
                        columns={patientColumns}
                        onChange={(patientAttributes) =>
                          edit({
                            patientAttributes: patientAttributes.map((item) => ({
                              ...item,
                              owner: 'patient',
                            })),
                          })
                        }
                      />
                    </div>
                  ) : null}
                </div>
              </Panel>
            </div>
            <div className="science-savebar" hidden={step !== 2}>
              <div>
                <strong>
                  {draft
                    ? `Revision ${draft.revision} · ${frozen ? 'Frozen import' : dirty ? 'Unsaved changes' : 'Saved draft'}`
                    : 'New import draft'}
                </strong>
                <p>
                  Save draft keeps your setup for later. Preview dataset saves the draft and
                  checks slide matches, patient IDs and attribute values before you freeze.
                </p>
              </div>
              <div className="inline-actions">
                <button type="button" className="btn btn-secondary" onClick={() => showStep(1)}>
                  Back to files
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={!name.trim()}
                  onClick={() =>
                    void run(async () => {
                      await save();
                      setMessage(
                        'Import draft saved. You can reopen it after restarting the service.',
                      );
                    })
                  }
                >
                  Save draft
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!name.trim() || !spec.slideIdColumn}
                  onClick={() => void run(() => previewDataset())}
                >
                  {busy ? 'Working…' : 'Preview dataset'} <Icon name="arrow" />
                </button>
              </div>
            </div>
          </fieldset>
          {preview && step === 3 ? (
            <div
              className="dataset-section"
              ref={reviewSection}
              tabIndex={-1}
              aria-label="Preview dataset"
            >
              <Panel
                title="3. Preview your dataset"
                subtitle="Check the included slides, file matches and patient links. Review any exclusions or warnings, then freeze to save a fixed dataset version."
                actions={
                  <div className="inline-actions">
                    <button
                      type="button"
                      className="btn btn-secondary btn-small"
                      disabled={busy}
                      onClick={() => showStep(2)}
                    >
                      Back to mapping
                    </button>
                    <Badge tone={preview.canFreeze ? 'green' : 'orange'}>
                      {preview.canFreeze ? 'Ready to freeze' : 'Resolve blocking findings'}
                    </Badge>
                  </div>
                }
              >
                <ImportMetrics summary={preview.summary} preview />
                <Findings findings={preview.findings} />
                <RecordExplorer
                  records={preview.records}
                  total={preview.summary.slideCount}
                  dictionary={preview.dictionary}
                />
                <div className="science-savebar">
                  <div>
                    <strong>Freeze this dataset version</strong>
                    <p>
                      Save this reviewed dataset as a fixed version you can reopen. To change it
                      later, create a revised version. Training is configured separately.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={busy || dirty || !preview.canFreeze || frozen}
                    onClick={() =>
                      void run(async () => {
                        const version = await scientific.importFreeze(
                          project,
                          draft!.id,
                          draft!.revision,
                          preview.previewHash,
                        );
                        setVersionId(version.id);
                        setView('dataset');
                        setDraft(await scientific.draft(project, draft!.id));
                        setPreview(null);
                        await refresh();
                        setMessage(
                          'Dataset version frozen. Its records and mapping will reload from this experiment folder.',
                        );
                        window.scrollTo({ top: 0 });
                      })
                    }
                  >
                    <Icon name="lock" /> {busy ? 'Freezing…' : 'Freeze dataset'}
                  </button>
                </div>
              </Panel>
            </div>
          ) : null}
        </>
      )}
      <PatientFallbackDialog
        count={fallbackCount}
        busy={busy}
        error={error}
        onBack={() => {
          setFallbackCount(null);
          setPreview(null);
          showStep(2);
        }}
        onContinue={() =>
          void run(async () => {
            const next = { ...spec, patientIdFallback: 'slide_id' as const };
            setSpec(next);
            await previewDataset(next);
          })
        }
      />
    </>
  );
}
function ImportMetrics({
  summary,
  preview = false,
}: {
  summary: ImportPreview['summary'];
  preview?: boolean;
}) {
  return (
    <div className="science-metrics">
      <Metric
        label={preview ? 'Slides to include' : 'Imported slides'}
        value={summary.slideCount.toLocaleString()}
        note={`${summary.sourceRowCount} source rows · ${summary.excludedRowCount} excluded`}
      />
      <Metric
        label="Known patients"
        value={(summary.verifiedPatientCount ?? summary.mappedPatientCount).toLocaleString()}
        note={
          summary.fallbackSlideCount
            ? `${summary.fallbackSlideCount} additional Slide_ID fallback groups`
            : `${summary.unlinkedSlideCount} slides have unresolved patients`
        }
      />
      <Metric
        label="Matched slide files"
        value={summary.matchedSlideCount.toLocaleString()}
        note={`${summary.missingSlideCount} metadata rows without files`}
      />
      <Metric
        label="Unmatched files"
        value={summary.unmatchedFileCount.toLocaleString()}
        note={`${summary.scannedFileCount} slide files scanned`}
      />
    </div>
  );
}
