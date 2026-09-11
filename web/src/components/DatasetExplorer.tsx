import { useDeferredValue, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { scientific } from '../api/scientific';
import type { AttributeMapping, DatasetQuery } from '../api/scientific';
import { ErrorNotice, Icon } from './ui';
import { scienceKey } from './ScientificUI';
import { categoryLabel } from '../lib/datasetValues';

export default function DatasetExplorer({
  project,
  datasetId,
  dictionary,
}: {
  project: string;
  datasetId: string;
  dictionary: AttributeMapping[];
}) {
  const [field, setField] = useState(
    dictionary.find((item) => item.type !== 'text')?.key ?? dictionary[0]?.key ?? '',
  );
  const [compare, setCompare] = useState('');
  const [search, setSearch] = useState('');
  const deferredSearch = useDeferredValue(search);
  const [filters, setFilters] = useState<DatasetQuery['filters']>([]);
  const [offset, setOffset] = useState(0);
  const query: DatasetQuery = {
    field: field || undefined,
    compare: compare || undefined,
    search: deferredSearch,
    filters,
    offset,
    limit: 200,
  };
  const results = useQuery({
    queryKey: [...scienceKey(project), 'dataset-query', datasetId, query],
    queryFn: () => scientific.queryDataset(project, datasetId, query),
  });
  const data = results.data;
  const maximum = Math.max(1, ...(data?.distribution.counts.map((item) => item.count) ?? []));
  const selected = filters.find((filter) => filter.field === field)?.values ?? [];
  const availableQuery: DatasetQuery = {
    field, search: deferredSearch, filters: filters.filter((filter) => filter.field !== field),
    offset: 0, limit: 1,
  };
  const available = useQuery({
    queryKey: [...scienceKey(project), 'dataset-query', datasetId, availableQuery],
    queryFn: () => scientific.queryDataset(project, datasetId, availableQuery),
    enabled: Boolean(field && selected.length),
  });
  function toggle(value: string | null) {
    const values = selected.includes(value)
      ? selected.filter((item) => item !== value)
      : [...selected, value];
    setFilters([
      ...filters.filter((filter) => filter.field !== field),
      ...(values.length ? [{ field, values }] : []),
    ]);
    setOffset(0);
  }
  return (
    <div className="record-explorer">
      <div className="science-grid-three">
        <label className="label">
          Explore an attribute
          <select
            className="field"
            value={field}
            onChange={(event) => {
              setField(event.target.value);
              if (compare === event.target.value) setCompare('');
              setOffset(0);
            }}
          >
            <option value="">Choose an attribute</option>
            {dictionary.map((item) => (
              <option key={item.key} value={item.key}>
                {item.key} · {item.owner}
              </option>
            ))}
          </select>
        </label>
        <label className="label">
          Compare with
          <select
            className="field"
            value={compare}
            onChange={(event) => {
              setCompare(event.target.value);
              setOffset(0);
            }}
          >
            <option value="">Distribution only</option>
            {dictionary
              .filter((item) => item.key !== field)
              .map((item) => (
                <option key={item.key} value={item.key}>
                  {item.key}
                </option>
              ))}
          </select>
        </label>
        <label className="label">
          Search the complete dataset
          <input
            className="field"
            type="search"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setOffset(0);
            }}
            placeholder="Slide, patient or attribute…"
          />
        </label>
      </div>
      {filters.length ? (
        <div className="science-filter-chips" aria-label="Active dataset filters">
          {filters.map((filter) => (
            <button
              type="button"
              className="btn btn-secondary btn-small"
              key={filter.field}
              onClick={() => {
                setFilters(filters.filter((item) => item.field !== filter.field));
                setOffset(0);
              }}
            >
              {filter.field}: {filter.values.map(categoryLabel).join(', ')}{' '}
              <Icon name="close" size={13} />
            </button>
          ))}
          <button
            type="button"
            className="text-button"
            onClick={() => {
              setFilters([]);
              setOffset(0);
            }}
          >
            Clear all filters
          </button>
        </div>
      ) : null}
      {selected.length ? <fieldset className="science-filter-values">
        <legend>Values to include for {field} (OR)</legend>
        <p className="muted">Add another value without clearing this filter. Choices respect your search and other attribute filters; charts and the table show the selected records.</p>
        <ErrorNotice error={available.error} />
        {available.isPending ? <p role="status">Loading available values…</p> : null}
        <div>
          {[...new Set([...selected, ...(available.data?.valueCounts.map((item) => item.value) ?? [])])].map((value) => <label key={JSON.stringify(value)} className="science-check">
            <input type="checkbox" checked={selected.includes(value)} disabled={!selected.includes(value) && selected.length >= 100} onChange={() => toggle(value)} />
            {categoryLabel(value)}
          </label>)}
        </div>
        {available.data?.valuesTruncated ? <p className="muted">Showing the 200 most frequent available values and your current selections.</p> : null}
        {selected.length >= 100 ? <p className="muted">Up to 100 values can be selected per attribute.</p> : null}
      </fieldset> : null}
      <ErrorNotice error={results.error} />
      {results.isFetching ? (
        <p className="muted" role="status">
          Querying the full dataset…
        </p>
      ) : null}
      {data ? (
        <>
          <div className="source-inspection">
            <strong>
              {data.total.toLocaleString()} / {data.totalSlides.toLocaleString()} slides
            </strong>
            <span>
              {data.summary.verifiedPatientCount ?? data.summary.mappedPatientCount} known
              patients
            </span>
            {data.summary.fallbackSlideCount ? (
              <span>{data.summary.fallbackSlideCount} Slide ID fallback groups</span>
            ) : null}
            <span>{data.summary.unlinkedSlideCount} unlinked slides</span>
          </div>
          <p className="muted">
            Charts and filters use the complete dataset.{' '}
            {data.distribution.unit === 'patient'
              ? 'This attribute counts distinct linked patients.'
              : data.distribution.unit === 'group'
                ? 'This attribute counts distinct groups, including Slide ID fallbacks. Fallback groups are not verified patients.'
                : 'This attribute counts slide rows.'}{' '}
            Values within a filter are combined with OR; different attribute filters use AND.
          </p>
          {data.distribution.unlinkedSlideCount ? (
            <p className="callout">
              {data.distribution.unlinkedSlideCount} unlinked slides are excluded from this
              grouped chart. They remain visible in the slide table.
            </p>
          ) : null}
          {data.distribution.truncated ? (
            <p className="callout">
              Showing the 100 most frequent categories. {data.distribution.omittedCategoryCount}{' '}
              other categories account for {data.distribution.otherCount} additional{' '}
              {data.distribution.unit === 'patient'
                ? 'patients'
                : data.distribution.unit === 'group'
                  ? 'groups'
                  : 'slides'}
              .
            </p>
          ) : null}
          {data.distribution.kind !== 'none' ? (
            <div className="science-chart" role="group" aria-label={`${field} distribution`}>
              {data.distribution.counts.map((item, index) => (
                <button
                  type="button"
                  className={`science-bar ${selected.includes(item.value) ? 'selected' : ''} ${data.distribution.kind === 'numeric' ? 'histogram-bin' : ''}`}
                  key={index}
                  disabled={data.distribution.kind === 'numeric'}
                  aria-pressed={
                    data.distribution.kind === 'categorical'
                      ? selected.includes(item.value)
                      : undefined
                  }
                  onClick={() => toggle(item.value)}
                >
                  <span>{data.distribution.kind === 'categorical' ? categoryLabel(item.value) : item.value ?? '(missing value)'}</span>
                  <span className="science-bar-track">
                    <span style={{ width: `${(item.count / maximum) * 100}%` }} />
                  </span>
                  <strong>{item.count}</strong>
                </button>
              ))}
              <small className="muted">
                {data.distribution.total.toLocaleString()}{' '}
                {data.distribution.unit === 'patient'
                  ? 'patients'
                  : data.distribution.unit === 'group'
                    ? 'groups'
                    : 'slides'}{' '}
                · {data.distribution.missingCount} missing values
                {data.distribution.kind === 'categorical' ? ' · Select bars to filter' : ''}
              </small>
            </div>
          ) : (
            <p className="muted">
              Text attributes can be searched. Define a categorical or numeric data type in an
              import draft to enable its chart.
            </p>
          )}
          {data.crossTab ? (
            <div className="table-wrap">
              <table className="cross-tab">
                <caption>
                  {field} × {compare} · {data.crossTab.unit} counts
                </caption>
                <thead>
                  <tr>
                    <th>{field}</th>
                    {data.crossTab.columns.map((value, index) => (
                      <th key={index}>{categoryLabel(value)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.crossTab.rows.map((value, row) => (
                    <tr key={row}>
                      <th>{categoryLabel(value)}</th>
                      {data.crossTab!.columns.map((_, column) => (
                        <td key={column}>{data.crossTab!.counts[row]?.[column] ?? 0}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Slide_ID</th>
                  <th>Patient_ID</th>
                  <th>Slide file</th>
                  {dictionary.map((item) => (
                    <th key={item.key}>{item.key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.records.map((record) => (
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
                    {dictionary.map((item) => (
                      <td key={item.key}>
                        {record.attributes[item.key] === null ||
                        record.attributes[item.key] === undefined ? (
                          <span className="muted">Missing</span>
                        ) : (
                          String(record.attributes[item.key])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.crossTab?.truncated ? (
            <p className="callout">
              This cross-tab is limited to 20 rows and 20 columns. {data.crossTab.omittedCount}{' '}
              observations are outside the displayed categories.
            </p>
          ) : null}
          {data.crossTab?.unlinkedSlideCount ? (
            <p className="muted">
              {data.crossTab.unlinkedSlideCount} unlinked slides are excluded from the
              patient-level cross-tab.
            </p>
          ) : null}
          {data.total === 0 ? <p className="muted">No records match these filters.</p> : null}
          <div className="inline-actions">
            <button
              type="button"
              className="btn btn-secondary btn-small"
              disabled={offset === 0 || results.isFetching}
              onClick={() => setOffset(Math.max(0, offset - 200))}
            >
              Previous
            </button>
            <span>
              {data.total ? offset + 1 : 0}–{offset + data.records.length} of{' '}
              {data.total.toLocaleString()} filtered slides
            </span>
            <button
              type="button"
              className="btn btn-secondary btn-small"
              disabled={offset + data.records.length >= data.total || results.isFetching}
              onClick={() => setOffset(offset + 200)}
            >
              Next
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}
