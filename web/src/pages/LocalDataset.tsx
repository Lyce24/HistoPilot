import { useRef, useState } from 'react';
import { preparationLink } from '../lib/preparationRoute';
import type { Workspace } from '../api/types';
import type {
  ImportPreview,
  ImportSpec,
  Inspection,
  ScientificDraft,
  VersionLabelInput,
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
  DictionaryEditor,
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
import VersionLabelEditor from '../components/VersionLabelEditor';
import FreezeVersionDialog from '../components/FreezeVersionDialog';
import SetupContext from '../components/SetupContext';
import { StageLibrary, StageLibraryToolbar, StageRecordManageButton, StagePage, StageSteps, useStageLibrary } from '../components/StageWorkflow';
import { datasetVersionLabel } from '../lib/versionLabels';
import { scientificReviewInvalidated } from '../lib/scientificReview';
import { canReuseImportMapping, inspectedAttributes } from '../lib/datasetImport';
import './dataset-workflow.css';

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
  const [view, setView] = useState<'library' | 'import' | 'dataset'>('library');
  const [libraryFilters, setLibraryFilters] = useState({ search: '', status: 'all', sort: 'recent' });
  const [resumeView, setResumeView] = useState<'import' | 'dataset' | null>(null);
  const [versionId, setVersionId] = useState(w.dataset.id);
  const [freezeReview, setFreezeReview] = useState<{
    draftId: string; revision: number; preview: ImportPreview; name: string; operationId: string;
  } | null>(null);
  const [freezeLabel, setFreezeLabel] = useState<VersionLabelInput>({ tag: '', note: '' });
  const [name, setName] = useState(`${w.project.name} dataset`);
  const [spec, setSpec] = useState<ImportSpec>(() => newSpec(w));
  const [draft, setDraft] = useState<ScientificDraft<ImportSpec> | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [patientInspection, setPatientInspection] = useState<Inspection | null>(null);
  const mappedMainSource = useRef<{ source: ImportSpec['source']; fingerprint: string } | null>(null);
  const mappedPatientSource = useRef<{ source: ImportSpec['source']; fingerprint: string } | null>(null);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [fallbackCount, setFallbackCount] = useState<number | null>(null);
  const [readingSource, setReadingSource] = useState<'main' | 'patient' | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  useStageLibrary(() => {
    if (!busy && !freezeReview && fallbackCount === null && view !== 'library') { setResumeView(view); setView('library'); }
  });
  const datasetId = versionId || versions.data?.datasets[0]?.id || '';
  const version = versions.data?.datasets.find((item) => item.id === datasetId);
  const dirty = !draft || draft.name !== name || !sameJSON(draft.payload.spec, spec);
  const frozen = draft?.status === 'frozen';
  const sourceReady =
    Boolean(inspection?.headers.length) &&
    !inspection?.findings.some((finding) => finding.severity === 'error');
  function showStep(next: 1 | 2 | 3) {
    setStep(next);
  }
  const importDrafts = (savedDrafts.data?.drafts ?? []).filter((item) => item.payload.type === 'dataset-import');
  const libraryRows = [
    ...(versions.data?.datasets ?? []).map((item) => ({
      kind: 'dataset' as const, item, name: datasetVersionLabel(item), status: 'frozen',
      updatedAt: item.versionLabel?.updatedAt ?? item.createdAt, createdAt: item.createdAt,
      search: [datasetVersionLabel(item), item.id, item.manifest.name, item.versionLabel?.note].join(' ').toLowerCase(),
    })),
    ...importDrafts.map((item) => ({
      kind: 'draft' as const, item, name: item.name, status: item.status,
      updatedAt: item.updatedAt, createdAt: item.createdAt ?? item.updatedAt,
      search: [item.name, item.id].join(' ').toLowerCase(),
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
    setFreezeLabel({ tag: '', note: '' });
    setFreezeReview(null);
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
  async function loadDraft(id: string) {
    const reloading = draft?.id === id;
    setPreview(null);
    setFreezeReview(null);
    const loaded = await scientific.draft<ImportSpec>(project, id);
    setDraft(loaded);
    if (!reloading) setFreezeLabel({ tag: '', note: '' });
    setSpec({
      ...newSpec(w),
      ...loaded.payload.spec,
      attributes: loaded.payload.spec.attributes ?? [],
    });
    setName(loaded.name);
    setInspection(null);
    setPatientInspection(null);
    setView('import');
    setFallbackCount(null);
    if (reloading) setMessage(`Reloaded saved draft revision ${loaded.revision}.`);
    showStep(1);
    await savedDrafts.refetch();
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
    setPreview(null);
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
    const prior = mappedMainSource.current;
    const parent = versions.data?.datasets.find((item) => item.id === spec.parentId);
    const savedMapping = draft?.payload.spec ?? (parent?.manifest.provenance as { mapping?: ImportSpec } | undefined)?.mapping;
    const preserve = canReuseImportMapping(spec.source, prior?.source, savedMapping?.attributes != null ? savedMapping.source : undefined);
    mappedMainSource.current = { source: spec.source, fingerprint: next.fingerprint };
    const slide = preserve ? spec.slideIdColumn : next.headers.includes(spec.slideIdColumn)
      ? spec.slideIdColumn
      : next.headers.includes('Slide_ID')
        ? 'Slide_ID'
        : next.headers.includes('De ID')
          ? 'De ID'
          : '';
    const patient = preserve ? spec.patientIdColumn :
      spec.patientIdColumn && next.headers.includes(spec.patientIdColumn)
        ? spec.patientIdColumn
        : next.headers.includes('Patient_ID')
          ? 'Patient_ID'
          : undefined;
    const attributes = inspectedAttributes(next.headers, spec.attributes, [slide, patient], preserve, 'slide');
    edit({
      slideIdColumn: slide,
      patientIdColumn: patient,
      attributes,
      ...(prior && prior.source === spec.source && prior.fingerprint !== next.fingerprint
        ? { patientIdFallback: 'unresolved' as const }
        : {}),
    });
  }
  return (
    <div className="clinical-workspace dataset-workspace">
      <PageHeader
        eyebrow="PROJECT INPUTS · DATA"
        title="Datasets"
        description="Open a dataset or import slide and patient records."
        actions={
          <div className="inline-actions">
            {view !== 'library' ? <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => { setResumeView(view); setView('library'); }}>Back to datasets</button> : null}
            <button type="button" className="btn btn-primary" disabled={busy} onClick={reset}>
              <Icon name="plus" /> New import
            </button>
          </div>
        }
      />
      <StagePage pageKey={`${view}-${view === 'import' ? step : versionId}`}>
      {view !== 'library' ? <SetupContext input="Slide table and optional slide images" output="A fixed dataset for targets and features">
        Recommended: keep development and test rows in one file with a cohort column. Select development rows in Targets and test rows in Evaluate. Separate files are also supported.
      </SetupContext> : null}
      <ErrorNotice error={error ?? versions.error ?? savedDrafts.error} />
      <SavedNotice>{message}</SavedNotice>
      {view === 'library' ? (
        <StageLibrary project={project} title="Dataset library">
          <StageLibraryToolbar search={libraryFilters.search} onSearch={(search) => setLibraryFilters((current) => ({ ...current, search }))} searchLabel="Search datasets" placeholder="Name, ID or note" count={versions.isPending || savedDrafts.isPending ? undefined : visibleRows.length} total={libraryRows.length}
            onReset={libraryFilters.search || libraryFilters.status !== 'all' || libraryFilters.sort !== 'recent' ? resetLibraryFilters : undefined}
            actions={<>
              {resumeView ? <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => setView(resumeView)}>Return to current {resumeView === 'import' ? 'import' : 'dataset'}</button> : null}
              <button type="button" className="btn btn-secondary btn-small" disabled={versions.isFetching || savedDrafts.isFetching} onClick={() => void refresh()}>Refresh</button>
            </>}>
            <label className="label">Status<select className="field" value={libraryFilters.status} onChange={(event) => setLibraryFilters((current) => ({ ...current, status: event.target.value }))}>
              <option value="all">All statuses</option><option value="editable">Draft</option><option value="frozen">Frozen</option>
            </select></label>
            <label className="label">Sort<select className="field" value={libraryFilters.sort} onChange={(event) => setLibraryFilters((current) => ({ ...current, sort: event.target.value }))}>
              <option value="recent">Last updated</option><option value="oldest">Oldest first</option><option value="name">Name A–Z</option>
            </select></label>
          </StageLibraryToolbar>
          {versions.isPending || savedDrafts.isPending ? <p role="status">Loading datasets and import drafts…</p> : null}
          {visibleRows.length ? (
            <div className="table-wrap"><table className="stage-library-table" aria-label="Saved datasets and imports">
              <thead><tr><th>Name</th><th>Status</th><th>Records</th><th>Updated</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>{visibleRows.map((row) => row.kind === 'dataset' ? <tr key={`dataset-${row.item.id}`}>
                <td><button type="button" className="text-button stage-record-name" disabled={busy} aria-label={`Open dataset ${row.name}`} onClick={() => { setVersionId(row.item.id); setView('dataset'); setError(null); setMessage(''); }}>{row.name}</button>{row.item.versionLabel?.note ? <small>{row.item.versionLabel.note}</small> : null}</td>
                <td><Badge tone="green">Frozen dataset</Badge></td>
                <td>{row.item.manifest.summary?.slideCount?.toLocaleString() ?? '—'} slides</td>
                <td>{new Date(row.updatedAt).toLocaleDateString()}</td>
                <td><StageRecordManageButton type="dataset" id={row.item.id} name={row.name} /></td>
              </tr> : <tr key={`draft-${row.item.id}`}>
                <td><button type="button" className="text-button stage-record-name" disabled={busy} aria-label={`Open import ${row.name}`} onClick={() => { if (draft?.id === row.item.id) setView('import'); else void run(() => loadDraft(row.item.id)); }}>{row.name}</button><small>Revision {row.item.revision}</small></td>
                <td><Badge tone={row.item.status === 'frozen' ? 'green' : 'neutral'}>{row.item.status === 'frozen' ? 'Frozen import' : 'Draft'}</Badge></td>
                <td>Import mapping</td>
                <td>{new Date(row.updatedAt).toLocaleDateString()}</td>
                <td><StageRecordManageButton type="draft" id={row.item.id} name={row.name} /></td>
              </tr>)}</tbody>
            </table></div>
          ) : !versions.isPending && !savedDrafts.isPending && !versions.error && !savedDrafts.error ? <EmptyState
            title={libraryRows.length ? 'No matching datasets or imports' : 'No datasets or import drafts yet'}
            description={libraryRows.length ? 'Try another search or clear the filters.' : 'Start a new import to connect your metadata, patient IDs and slide files.'}
          /> : null}
        </StageLibrary>
      ) : view === 'dataset' ? (
        version ? (
          <>
            <div className="science-version">
              <Badge tone="green">
                <Icon name="lock" size={12} /> Frozen dataset
              </Badge>
              <strong title={version.id}>{datasetVersionLabel(version)}</strong>
              <small>{new Date(version.createdAt).toLocaleString()}</small>
            </div>
            {version.versionLabel?.note ? <p className="version-tag-note">{version.versionLabel.note}</p> : null}
            <details className="setup-details"><summary>Edit version label and note</summary><VersionLabelEditor
              project={project}
              resourceType="dataset"
              resource={version}
              tagLabel="Dataset version tag"
            /></details>
            {version.manifest.summary ? (
              <ImportMetrics summary={version.manifest.summary} />
            ) : null}
            <Panel
              title="Explore your dataset"
              subtitle="Browse slide records and explore the values in your table. Choose the prediction target later."
            >
              <DatasetExplorer
                key={datasetId}
                project={project}
                datasetId={datasetId}
                dictionary={version.manifest.dictionary ?? []}
              />
            </Panel>
            <Panel
              title="Saved dataset details"
              subtitle="The source files and column mapping are recorded with this version."
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
                  setFreezeLabel({ tag: '', note: '' });
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
            <div className="setup-next-actions" aria-label="Continue with this dataset">
              <a className="btn btn-primary" href={preparationLink('cohort', { datasetId: version.id })}>Define targets &amp; splits <Icon name="arrow" /></a>
              <a className="btn btn-secondary" href={preparationLink('features', { datasetId: version.id })}>Prepare features <Icon name="arrow" /></a>
            </div>
          </>
        ) : (
          <EmptyState
            title="No dataset version selected"
            description="Create an import or choose a frozen version above."
          />
        )
      ) : (
        <>
          <StageSteps label="Dataset creation steps" current={String(step)} onChange={(id) => showStep(Number(id) as 1 | 2 | 3)} disabled={busy} steps={[
            { id: '1', title: 'Choose files', description: sourceReady ? 'File read · ready to map' : 'Table and slide folder', complete: sourceReady },
            { id: '2', title: 'Map columns', description: preview ? 'Mapping reviewed' : 'IDs and attributes', disabled: !columns.length, complete: Boolean(preview) },
            { id: '3', title: 'Review & freeze', description: preview ? 'Check your dataset' : 'Save a fixed version', disabled: !preview },
          ]} />
          {draft ? <div className="stage-actions">
            <span className="muted">{name} · revision {draft.revision}{dirty ? ' · Unsaved changes' : ''}</span>
            <button type="button" className="btn btn-secondary btn-small" disabled={busy} title="Reload the latest saved revision and discard unsaved local edits. Your version tag and note are kept." onClick={() => void run(() => loadDraft(draft.id))}><Icon name="reset" size={15} /> Reload saved draft</button>
          </div> : null}
          {frozen ? (
            <div className="callout">
              This import draft is frozen.{' '}
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  setDraft(null);
                  setFreezeLabel({ tag: '', note: '' });
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
              hidden={step !== 1}
              tabIndex={-1}
              aria-label="Choose dataset files"
            >
              <Panel
                title="1. Choose your metadata and slides"
                subtitle="Start with your spreadsheet. Slide images stay in their existing folder."
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
                  <div className="dataset-file-layout">
                    <section className="dataset-file-card" aria-label="Metadata table">
                      <div className="dataset-file-heading">
                        <span className="dataset-file-icon">
                          <Icon name="dataset" size={20} />
                        </span>
                        <div>
                          <h3>Metadata table</h3>
                          <p>A CSV or XLSX with one row per slide.</p>
                        </div>
                        <span className="dataset-requirement">Required</span>
                      </div>
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
                    </section>
                    <section
                      className="dataset-file-card dataset-slide-card"
                      aria-label="Slide images"
                    >
                      <div className="dataset-file-heading">
                        <span className="dataset-file-icon">
                          <Icon name="folder" size={20} />
                        </span>
                        <div>
                          <h3>Slide images</h3>
                          <p>Connect the folder containing your slide files.</p>
                        </div>
                        <span className="dataset-requirement is-optional">Optional</span>
                      </div>
                      <div className="dataset-slide-path">
                        <label className="label">
                          Slide folder
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
                          onChange={(event) =>
                            edit({ includeMissingSlides: event.target.checked })
                          }
                        />{' '}
                        Keep metadata rows without matching slide files (for example, existing
                        features)
                      </label>
                      <p className="dataset-field-note">
                        Files are matched by Slide_ID and file extension. Without the option
                        above, rows with no matching file are excluded from the dataset.
                      </p>
                    </section>
                  </div>
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
              hidden={step !== 2}
              tabIndex={-1}
              aria-label="Map dataset columns"
            >
              <Panel
                title="2. Tell us what each column means"
                subtitle="First identify each slide and patient. Then choose which other information to keep."
              >
                {columns.length ? (
                  <div className="stack">
                    <div className="dataset-subsection-heading">
                      <span className="dataset-subsection-marker">
                        <Icon name="patient" size={17} />
                      </span>
                      <div>
                        <h3>Link slides to patients</h3>
                        <p>Slides from the same patient should share the same Patient_ID.</p>
                      </div>
                    </div>
                    <div className="science-grid-two dataset-identity-fields">
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
                        <small>Identifies each slide or case. Required.</small>
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
                        <small>Identifies the person the slide belongs to.</small>
                      </label>
                    </div>
                    <div className="callout dataset-identity-note">
                      <Icon name="info" size={17} />
                      <div>
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
                    </div>
                    <details className="setup-details">
                      <summary>Review retained columns and missing-value rules · {spec.attributes.length} columns</summary>
                      <p className="muted">The suggested columns are retained below. Open to change their names, types or ownership before reviewing the dataset.</p>
                    <label className="label dataset-missing-values">
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
                    </details>
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
                <details className="setup-details" open={spec.patientSource ? true : undefined}>
                  <summary>Add a separate patient table (optional){spec.patientSource ? ' · configured' : ''}</summary>
                  <div className="stack">
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
                            const prior = mappedPatientSource.current;
                            const parent = versions.data?.datasets.find((item) => item.id === spec.parentId);
                            const savedMapping = draft?.payload.spec ?? (parent?.manifest.provenance as { mapping?: ImportSpec } | undefined)?.mapping;
                            const preserve = canReuseImportMapping(spec.patientSource!, prior?.source, savedMapping?.patientAttributes != null ? savedMapping.patientSource : undefined);
                            mappedPatientSource.current = { source: spec.patientSource!, fingerprint: inspected.fingerprint };
                            const slide =
                              spec.patientSourceKind === 'patients'
                                ? undefined
                                : preserve ? spec.patientSourceSlideIdColumn : spec.patientSourceSlideIdColumn &&
                                    inspected.headers.includes(spec.patientSourceSlideIdColumn)
                                  ? spec.patientSourceSlideIdColumn
                                  : inspected.headers.includes('Slide_ID')
                                    ? 'Slide_ID'
                                    : inspected.headers.includes('De ID')
                                      ? 'De ID'
                                      : '';
                            const patient =
                              preserve ? spec.patientSourcePatientIdColumn : spec.patientSourcePatientIdColumn &&
                              inspected.headers.includes(spec.patientSourcePatientIdColumn)
                                ? spec.patientSourcePatientIdColumn
                                : inspected.headers.includes('Patient_ID')
                                  ? 'Patient_ID'
                                  : '';
                            edit({
                              ...(prior && prior.source === spec.patientSource &&
                              prior.fingerprint !== inspected.fingerprint
                                ? { patientIdFallback: 'unresolved' as const }
                                : {}),
                              patientSourceSlideIdColumn: slide,
                              patientSourcePatientIdColumn: patient,
                              patientAttributes: inspectedAttributes(inspected.headers, spec.patientAttributes ?? [], [slide, patient], preserve, 'patient'),
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
                </details>
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
                  Save your progress, or preview to check slide matches, patient IDs and values.
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
              tabIndex={-1}
              aria-label="Preview dataset"
            >
              <Panel
                title="3. Preview your dataset"
                subtitle="Check the counts and explore your data. Resolve any blocking findings before saving this version."
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
                <div className="science-savebar dataset-freeze-bar">
                  <div>
                    <strong>Ready to keep this dataset?</strong>
                    <p>
                      Next, choose a required version tag and an optional commit note. Then
                      freeze this reviewed dataset in one save.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={busy || dirty || !preview.canFreeze || frozen}
                    onClick={() => setFreezeReview({ draftId: draft!.id, revision: draft!.revision, preview, name, operationId: `import:${crypto.randomUUID()}` })}
                  >
                    <Icon name="lock" /> Name & freeze dataset
                  </button>
                </div>
              </Panel>
            </div>
          ) : null}
        </>
      )}
      </StagePage>
      {freezeReview ? (
        <FreezeVersionDialog
          kind="dataset"
          initialLabel={freezeLabel}
          onLabelChange={setFreezeLabel}
          onClose={() => setFreezeReview(null)}
          onFreeze={async (versionLabel) => {
            setBusy(true);
            try {
              const version = await scientific.importFreeze(project, freezeReview.draftId, freezeReview.revision, freezeReview.preview.previewHash, versionLabel, freezeReview.operationId);
              setVersionId(version.id);
              setView('dataset');
              setDraft((current) => current?.id === freezeReview.draftId ? { ...current, status: 'frozen', revision: freezeReview.revision + 1 } : current);
              setPreview(null);
              await refresh();
              setMessage(`Dataset “${version.versionLabel?.tag || versionLabel.tag}” frozen. Its tag and commit note were saved with it.`);
              window.location.hash = preparationLink('cohort', { datasetId: version.id, saved: 'dataset' });
              window.scrollTo({ top: 0 });
            } catch (reason) {
              if (scientificReviewInvalidated(reason)) {
                setFreezeReview(null);
                setPreview(null);
                showStep(2);
                setError(new Error(`${reason.message} Your tag and note have been kept. Review the dataset again before freezing.`));
              }
              throw reason;
            } finally { setBusy(false); }
          }}
        >
          <p><strong>{freezeReview.name}</strong></p>
          <p>{freezeReview.preview.summary.slideCount.toLocaleString()} slides · {freezeReview.preview.summary.mappedPatientCount.toLocaleString()} mapped patients</p>
        </FreezeVersionDialog>
      ) : null}
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
    </div>
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
