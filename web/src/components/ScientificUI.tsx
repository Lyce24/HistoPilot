import { useId, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { scientific } from '../api/scientific';
import type {
  AttributeMapping,
  DataRecord,
  DatasetVersion,
  Finding,
  Inspection,
  ScientificDraft,
  TableSource,
} from '../api/scientific';
import { Badge, ErrorNotice, Icon } from './ui';
import ServerFolderPicker from './ServerFolderPicker';
import './dataset-fields.css';

export const scienceKey = (project: string) => ['scientific', project];
export function useDatasets(project: string) {
  return useQuery({
    queryKey: [...scienceKey(project), 'datasets'],
    queryFn: () => scientific.datasets(project),
  });
}
export function useDrafts(project: string) {
  return useQuery({
    queryKey: [...scienceKey(project), 'drafts'],
    queryFn: () => scientific.drafts(project),
  });
}
export function useConfigurations(project: string, kind: 'protocol' | 'feature') {
  return useQuery({
    queryKey: [...scienceKey(project), 'configurations', kind],
    queryFn: () => scientific.configurations(project, kind),
  });
}
export function useRefreshScientific(project: string) {
  const client = useQueryClient();
  return () =>
    Promise.all([
      client.invalidateQueries({ queryKey: scienceKey(project) }),
      client.invalidateQueries({ queryKey: ['workspace', project] }),
    ]);
}
export function Findings({ findings }: { findings: Finding[] }) {
  return findings.length ? (
    <div className="finding-list" aria-label="Validation findings">
      {findings.map((item, index) => (
        <div className={`finding finding-${item.severity}`} key={`${item.code}-${index}`}>
          <Icon name={item.severity === 'error' ? 'lock' : 'info'} size={17} />
          <div>
            <strong>{item.message}</strong>
            <small>
              {item.code}
              {item.count !== undefined ? ` · ${item.count} affected` : ''}
            </small>
          </div>
          <Badge
            tone={
              item.severity === 'error'
                ? 'red'
                : item.severity === 'warning'
                  ? 'orange'
                  : 'neutral'
            }
          >
            {item.severity === 'error' ? 'Blocking' : item.severity}
          </Badge>
        </div>
      ))}
    </div>
  ) : (
    <p className="science-success">
      <Icon name="check" size={16} /> No blocking findings in this check.
    </p>
  );
}
export function SavedNotice({ children }: { children: ReactNode }) {
  return children ? (
    <div className="callout science-success" role="status">
      {children}
    </div>
  ) : null;
}
export function DatasetSelect({
  versions,
  value,
  onChange,
  disabled = false,
  allowEmpty = false,
}: {
  versions: DatasetVersion[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  allowEmpty?: boolean;
}) {
  return (
    <label className="label">
      Dataset version
      <select
        className="field"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {allowEmpty || !value ? (
          <option value="">
            {versions.length ? 'Choose a frozen dataset' : 'No frozen datasets yet'}
          </option>
        ) : null}
        {versions.map((version, index) => (
          <option key={version.id} value={version.id}>
            {version.manifest.name ?? `Dataset ${versions.length - index}`} ·{' '}
            {version.manifest.summary?.slideCount ?? '?'} slides · {version.id.slice(-8)}
          </option>
        ))}
      </select>
    </label>
  );
}
export function DraftSelect({
  drafts,
  value,
  onChange,
  disabled,
}: {
  drafts: ScientificDraft[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <label className="label">
      Saved drafts
      <select
        className="field"
        disabled={disabled}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">Start a new draft</option>
        {drafts.map((draft) => (
          <option key={draft.id} value={draft.id}>
            {draft.name} · revision {draft.revision} · {draft.status}
          </option>
        ))}
      </select>
    </label>
  );
}
export function SourceFields({
  label,
  source,
  onChange,
  inspection,
  onInspect,
  onContinue,
  busy,
  reading = false,
}: {
  label: string;
  source: TableSource;
  onChange: (source: TableSource) => void;
  inspection?: Inspection | null;
  onInspect: () => void;
  onContinue?: () => void;
  busy: boolean;
  reading?: boolean;
}) {
  const [error, setError] = useState<Error | null>(null);
  const upload = source.contentBase64 !== undefined;
  const sheetsId = useId();
  const pathId = useId();
  const readHelpId = useId();
  const workbook = /\.xlsx$/i.test((upload ? source.filename : source.path)?.trim() ?? '');
  const ready =
    inspection &&
    inspection.headers.length > 0 &&
    !inspection.findings.some((finding) => finding.severity === 'error');
  async function readFile(file?: File) {
    if (!file) return;
    setError(null);
    onChange({ filename: file.name, contentBase64: '' });
    if (file.size > 256 * 1024) {
      setError(
        new Error(
          'Browser metadata uploads are limited to 256 KB. Use a server file path for larger tables (up to 16 MB).',
        ),
      );
      return;
    }
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let text = '';
      for (let offset = 0; offset < bytes.length; offset += 8192)
        text += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
      onChange({ filename: file.name, contentBase64: btoa(text) });
    } catch {
      setError(new Error('This metadata file could not be read.'));
    }
  }
  return (
    <div className="source-fields dataset-source-fields">
      <label className="label">
        {label} location
        <select
          className="field"
          value={upload ? 'upload' : 'server'}
          onChange={(event) => {
            setError(null);
            onChange(
              event.target.value === 'upload'
                ? { filename: '', contentBase64: '' }
                : { path: '' },
            );
          }}
        >
          <option value="server">File on the server</option>
          <option value="upload">Upload metadata from this computer</option>
        </select>
      </label>
      {upload ? (
        <label className="label">
          CSV or XLSX file · maximum 256 KB
          <input
            className="field"
            type="file"
            accept=".csv,.xlsx"
            onChange={(event) => void readFile(event.target.files?.[0])}
          />
          <small>{source.filename || 'Select a metadata file'}</small>
        </label>
      ) : (
        <div className="dataset-source-path">
          <label className="label" htmlFor={pathId}>
            {label} file path
          </label>
          <div className="dataset-source-path-controls">
            <input
              id={pathId}
              className="field mono"
              value={source.path ?? ''}
              placeholder="/path/to/metadata.csv or workbook.xlsx"
              onChange={(event) => {
                const path = event.target.value;
                onChange({
                  ...source,
                  path,
                  sheet: /\.xlsx$/i.test(path.trim()) ? source.sheet : undefined,
                });
              }}
            />
            <ServerFolderPicker
              selection="table"
              label="Browse metadata files"
              title="Choose a metadata file"
              onSelect={(path) => onChange({ path })}
            />
          </div>
        </div>
      )}
      {workbook ? (
        <label className="label">
          Workbook sheet (optional)
          <input
            className="field"
            value={source.sheet ?? ''}
            list={sheetsId}
            placeholder="First sheet"
            onChange={(event) =>
              onChange({ ...source, sheet: event.target.value || undefined })
            }
          />
          <small>
            Leave blank to read the first sheet. Read the file again after choosing another
            sheet.
          </small>
          <datalist id={sheetsId}>
            {inspection?.sheets.map((sheet) => (
              <option key={sheet} value={sheet} />
            ))}
          </datalist>
        </label>
      ) : null}
      <div className="dataset-read-action">
        <div>
          <strong>
            {inspection ? 'Review or refresh this file' : 'Next: read your metadata file'}
          </strong>
          <p id={readHelpId}>
            Read the column names and example values, then choose what each column means. Your
            dataset is saved and frozen in the following steps.
          </p>
        </div>
        <button
          type="button"
          className={`btn ${ready ? 'btn-secondary' : 'btn-primary'}`}
          disabled={busy || reading || !(source.path?.trim() || source.contentBase64)}
          aria-describedby={readHelpId}
          aria-busy={reading}
          onClick={onInspect}
        >
          <Icon name={reading ? 'clock' : 'search'} size={17} />
          {reading
            ? 'Reading file…'
            : inspection
              ? 'Read file again'
              : 'Read file & show columns'}
        </button>
      </div>
      <ErrorNotice error={error} />
      {inspection ? (
        <div className={`dataset-source-result ${ready ? 'is-ready' : ''}`} role="status">
          <div>
            {ready ? <Icon name="check" size={18} /> : <Icon name="info" size={18} />}
            <strong>{ready ? 'Columns ready to map' : 'Review the file findings below'}</strong>
            <span>
              {inspection.rowCount.toLocaleString()} rows · {inspection.headers.length} columns
              {inspection.sheet ? ` · ${inspection.sheet}` : ''}
            </span>
          </div>
          {ready ? (
            onContinue ? (
              <button
                type="button"
                className="btn btn-secondary btn-small"
                disabled={busy}
                onClick={onContinue}
              >
                Continue to column mapping <Icon name="arrow" size={15} />
              </button>
            ) : (
              <p>Next, review the identity columns and attributes below.</p>
            )
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function ColumnExamples({
  column,
  inspection,
}: {
  column: string;
  inspection?: Inspection | null;
}) {
  if (!inspection) return <span className="muted">Read the file to see examples</span>;
  if (!inspection.headers.includes(column))
    return <span className="muted">Column not found in this file</span>;
  const summary = inspection.columnSummaries?.[column];
  const examples = summary?.examples ?? [
    ...new Set(
      inspection.rows
        .map((row) => row[column])
        .filter(
          (value): value is string => value !== null && value !== undefined && value !== '',
        ),
    ),
  ];
  return (
    <div className="dataset-column-examples">
      {examples.length ? (
        <div className="dataset-example-chips" aria-label={`Examples for ${column}`}>
          {examples.slice(0, 4).map((value) => (
            <span className="dataset-example-chip" key={value} title={value}>
              {value}
            </span>
          ))}
          {(summary?.distinctCount ?? examples.length) > 4 ? (
            <span className="dataset-example-more">…</span>
          ) : null}
        </div>
      ) : (
        <span className="muted">
          {summary ? 'All cells are empty' : 'No non-empty values in the preview sample'}
        </span>
      )}
      {summary ? (
        <small>
          {summary.distinctCount.toLocaleString()} distinct
          {summary.missingCount ? ` · ${summary.missingCount.toLocaleString()} empty` : ''}
        </small>
      ) : (
        <small>From {inspection.rows.length} preview rows</small>
      )}
    </div>
  );
}

export function DictionaryEditor({
  attributes,
  columns,
  onChange,
  inspection,
}: {
  attributes: AttributeMapping[];
  columns: string[];
  onChange: (value: AttributeMapping[]) => void;
  inspection?: Inspection | null;
}) {
  const change = (index: number, update: Partial<AttributeMapping>) =>
    onChange(attributes.map((item, at) => (at === index ? { ...item, ...update } : item)));
  return (
    <div className="dictionary-editor dataset-dictionary-editor">
      <p className="dataset-dictionary-guide muted">
        {inspection?.columnSummaries
          ? `Examples show the most frequent source values across all ${inspection.rowCount.toLocaleString()} rows.`
          : inspection
            ? `Examples come from ${inspection.rows.length} preview rows, so other values may exist.`
            : 'Read the metadata file to see example values beside each column.'}{' '}
        Choose who each attribute describes and how its values should be checked.
      </p>
      <div className="table-wrap dataset-dictionary-wrap">
        <table className="dataset-dictionary-table" role="table" aria-label="Attribute mapping">
          <colgroup>
            <col />
            <col />
            <col />
            <col />
            <col />
            <col />
            <col />
          </colgroup>
          <thead>
            <tr role="row">
              <th scope="col" role="columnheader">
                Source column
              </th>
              <th scope="col" role="columnheader">
                Attribute name
              </th>
              <th scope="col" role="columnheader">
                Belongs to
              </th>
              <th scope="col" role="columnheader">
                Data type
              </th>
              <th scope="col" role="columnheader">
                Examples
              </th>
              <th scope="col" role="columnheader">
                Values
              </th>
              <th scope="col" role="columnheader">
                <span className="sr-only">Remove</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {attributes.map((item, index) => (
              <tr key={index} role="row">
                <td data-label="Source column" role="cell">
                  <select
                    className="field"
                    aria-label={`Source column ${index + 1}`}
                    value={item.sourceColumn}
                    onChange={(event) => change(index, { sourceColumn: event.target.value })}
                  >
                    {columns.map((column) => (
                      <option key={column}>{column}</option>
                    ))}
                  </select>
                </td>
                <td data-label="Attribute name" role="cell">
                  <input
                    className="field"
                    aria-label={`Attribute name ${index + 1}`}
                    value={item.key}
                    onChange={(event) => change(index, { key: event.target.value })}
                  />
                </td>
                <td data-label="Belongs to" role="cell">
                  <select
                    className="field"
                    aria-label={`Attribute owner ${index + 1}`}
                    value={item.owner}
                    onChange={(event) =>
                      change(index, { owner: event.target.value as AttributeMapping['owner'] })
                    }
                  >
                    <option value="slide">Slide / case</option>
                    <option value="patient">Patient</option>
                  </select>
                </td>
                <td data-label="Data type" role="cell">
                  <select
                    className="field"
                    aria-label={`Attribute type ${index + 1}`}
                    value={item.type}
                    onChange={(event) =>
                      change(index, { type: event.target.value as AttributeMapping['type'] })
                    }
                  >
                    {[
                      'text',
                      'categorical',
                      'ordered_categorical',
                      'integer',
                      'decimal',
                      'boolean',
                      'date',
                    ].map((type) => (
                      <option key={type} value={type}>
                        {type.replaceAll('_', ' ')}
                      </option>
                    ))}
                  </select>
                </td>
                <td data-label="Examples" role="cell">
                  <ColumnExamples column={item.sourceColumn} inspection={inspection} />
                </td>
                <td data-label="Values" role="cell">
                  {item.type.includes('categorical') ? (
                    <>
                      <input
                        className="field"
                        aria-label={`Allowed categories ${index + 1}`}
                        placeholder="Any source value"
                        value={(item.categories ?? []).join(' | ')}
                        onChange={(event) =>
                          change(index, {
                            categories: event.target.value
                              ? event.target.value.split('|').map((v) => v.trim())
                              : undefined,
                          })
                        }
                      />
                      <small>
                        {item.type === 'ordered_categorical'
                          ? 'Optional order, e.g. low | high. Unlisted values are flagged.'
                          : 'Optional allowed list, e.g. low | high. Unlisted values are flagged.'}
                      </small>
                    </>
                  ) : (
                    <>
                      <span className="dataset-preserve-values">Preserve source values</span>
                      <small>
                        {item.type === 'text'
                          ? 'Keep original text; no recoding.'
                          : `Checked as ${item.type}; no label recoding.`}
                      </small>
                    </>
                  )}
                </td>
                <td className="dataset-attribute-remove" role="cell">
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`Remove ${item.sourceColumn} attribute`}
                    onClick={() => onChange(attributes.filter((_, at) => at !== index))}
                  >
                    <Icon name="close" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        type="button"
        className="btn btn-secondary btn-small"
        disabled={!columns.length}
        onClick={() =>
          onChange([
            ...attributes,
            { key: columns[0], sourceColumn: columns[0], owner: 'slide', type: 'text' },
          ])
        }
      >
        <Icon name="plus" size={15} /> Add attribute
      </button>
      <p className="muted">
        Targets and model inputs are chosen later in Target & split. Numeric columns are not
        automatically predictors. Patient attributes require consistent values within each
        mapped patient.
      </p>
    </div>
  );
}
export function RecordExplorer({
  records,
  total,
  dictionary,
  onPage,
  offset = 0,
}: {
  records: DataRecord[];
  total: number;
  dictionary: AttributeMapping[];
  onPage?: (offset: number) => void;
  offset?: number;
}) {
  const [field, setField] = useState('');
  const [compare, setCompare] = useState('');
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<string | null>(null);
  const columns = dictionary.map((item) => item.key);
  const selectedField = columns.includes(field) ? field : (columns[0] ?? '');
  const valueOf = (record: DataRecord, key: string) =>
    String(record.attributes[key] ?? '(missing)');
  const counts = useMemo(() => {
    const result = new Map<string, number>();
    for (const record of records) {
      const value = String(record.attributes[selectedField] ?? '(missing)');
      result.set(value, (result.get(value) ?? 0) + 1);
    }
    return [...result.entries()].sort((a, b) => b[1] - a[1]);
  }, [records, selectedField]);
  const maximum = Math.max(1, ...counts.map(([, count]) => count));
  const visible = records.filter(
    (record) =>
      (!filter || valueOf(record, selectedField) === filter) &&
      (!search ||
        [
          record.slideId,
          record.patientId ?? '',
          ...Object.values(record.attributes).map(String),
        ]
          .join(' ')
          .toLowerCase()
          .includes(search.toLowerCase())),
  );
  const secondValues = [...new Set(records.map((record) => valueOf(record, compare)))]
    .sort()
    .slice(0, 12);
  return (
    <div className="record-explorer">
      <div className="science-grid-two">
        <label className="label">
          Explore an attribute
          <select
            className="field"
            value={selectedField}
            onChange={(event) => {
              setField(event.target.value);
              setFilter(null);
            }}
          >
            <option value="">No attribute selected</option>
            {columns.map((column) => (
              <option key={column}>{column}</option>
            ))}
          </select>
        </label>
        <label className="label">
          Compare with (optional)
          <select
            className="field"
            value={compare}
            onChange={(event) => setCompare(event.target.value)}
          >
            <option value="">Distribution only</option>
            {columns
              .filter((column) => column !== selectedField)
              .map((column) => (
                <option key={column}>{column}</option>
              ))}
          </select>
        </label>
      </div>
      <p className="muted">
        Distributions describe these {records.length} displayed slide rows
        {total > records.length ? ` of ${total.toLocaleString()} total` : ''}. Counts are not
        patient counts. Select a bar to filter the table.
      </p>
      {selectedField ? (
        <div
          className="science-chart"
          role="group"
          aria-label={`${selectedField} distribution`}
        >
          {counts.slice(0, 16).map(([value, count]) => (
            <button
              type="button"
              className={`science-bar ${filter === value ? 'selected' : ''}`}
              key={value}
              aria-pressed={filter === value}
              onClick={() => setFilter(filter === value ? null : value)}
            >
              <span title={value}>{value}</span>
              <span className="science-bar-track">
                <span style={{ width: `${(count / maximum) * 100}%` }} />
              </span>
              <strong>{count}</strong>
            </button>
          ))}
          {counts.length > 16 ? (
            <small className="muted">Showing the 16 most frequent values.</small>
          ) : null}
        </div>
      ) : null}
      {compare && selectedField ? (
        <div className="table-wrap">
          <table className="cross-tab">
            <caption>
              {selectedField} × {compare} · displayed slide rows
            </caption>
            <thead>
              <tr>
                <th>{selectedField}</th>
                {secondValues.map((value) => (
                  <th key={value}>{value}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {counts.slice(0, 12).map(([value]) => (
                <tr key={value}>
                  <th>{value}</th>
                  {secondValues.map((second) => (
                    <td key={second}>
                      {
                        records.filter(
                          (record) =>
                            valueOf(record, selectedField) === value &&
                            valueOf(record, compare) === second,
                        ).length
                      }
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <div className="inline-actions">
        <label className="label science-search">
          Search displayed rows
          <input
            className="field"
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Slide, patient or attribute…"
          />
        </label>
        {filter ? (
          <button
            type="button"
            className="btn btn-secondary btn-small"
            onClick={() => setFilter(null)}
          >
            Clear {filter} filter <Icon name="close" size={13} />
          </button>
        ) : null}
        <span className="muted">{visible.length} visible rows</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Slide_ID</th>
              <th>Patient_ID</th>
              <th>Slide file</th>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((record) => (
              <tr key={record.slideId}>
                <td className="mono">{record.slideId}</td>
                <td className="mono">
                  {record.patientId ?? <span className="muted">Unresolved</span>}
                  {record.patientIdSource === 'slide_fallback' ? (
                    <small className="science-block muted">Slide ID fallback</small>
                  ) : null}
                </td>
                <td>
                  {record.slidePath ? (
                    <span title={record.slidePath}>Matched</span>
                  ) : (
                    <span className="muted">Unavailable</span>
                  )}
                </td>
                {columns.map((column) => (
                  <td key={column}>
                    {record.attributes[column] === null ||
                    record.attributes[column] === undefined ? (
                      <span className="muted">Missing</span>
                    ) : (
                      String(record.attributes[column])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {onPage && total > 200 ? (
        <div className="inline-actions">
          <button
            type="button"
            className="btn btn-secondary btn-small"
            disabled={offset === 0}
            onClick={() => {
              setFilter(null);
              onPage(Math.max(0, offset - 200));
            }}
          >
            Previous 200
          </button>
          <span>
            {offset + 1}–{offset + records.length} of {total}
          </span>
          <button
            type="button"
            className="btn btn-secondary btn-small"
            disabled={offset + records.length >= total}
            onClick={() => {
              setFilter(null);
              onPage(offset + 200);
            }}
          >
            Next 200
          </button>
        </div>
      ) : null}
    </div>
  );
}
