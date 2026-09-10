import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { scientific } from '../api/scientific';
import type {
  AttributeMapping,
  DataRecord,
  ProtocolCohortStats,
  ProtocolExploreRequest,
  ProtocolPartitionStats,
} from '../api/scientific';
import { Badge, ErrorNotice, Metric } from './ui';
import { scienceKey } from './ScientificUI';
import './ProtocolExploration.css';

export function useProtocolExploration(project: string, request: ProtocolExploreRequest) {
  const serialized = JSON.stringify(request);
  const [settled, setSettled] = useState(serialized);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(serialized), 300);
    return () => window.clearTimeout(timer);
  }, [serialized]);
  const changing = settled !== serialized;
  const query = useQuery({
    queryKey: [...scienceKey(project), 'protocol-exploration', settled],
    queryFn: () => scientific.exploreProtocol(project, JSON.parse(settled)),
    enabled: Boolean(request.datasetId) && !changing,
    retry: false,
  });
  return {
    data: changing ? undefined : query.data,
    error: changing ? null : query.error,
    loading: Boolean(request.datasetId) && (changing || query.isPending),
  };
}

export interface ProtocolFieldContext {
  project: string;
  datasetId: string;
  dictionary: AttributeMapping[];
}

export function FieldProfile({
  project,
  datasetId,
  dictionary,
  field,
}: ProtocolFieldContext & { field: string }) {
  const attribute = dictionary.find((item) => item.key === field);
  const identity = field === 'Slide_ID' || field === 'Patient_ID';
  const query = useQuery({
    queryKey: [...scienceKey(project), 'field-profile', datasetId, field],
    queryFn: () =>
      scientific.queryDataset(project, datasetId, {
        ...(attribute ? { field } : {}),
        search: '',
        filters: [],
        offset: 0,
        limit: 5,
      }),
    enabled: Boolean(datasetId && field && (attribute || identity)),
    staleTime: 60_000,
  });
  if (!field) return null;
  if (!attribute && !identity)
    return (
      <p className="callout callout-warning">
        {field} is not a column in this dataset. Choose an available field.
      </p>
    );
  const examples = identity
    ? [
        ...new Set(
          (query.data?.records ?? [])
            .map((row) => (field === 'Slide_ID' ? row.slideId : row.patientId))
            .filter((value) => value !== null),
        ),
      ]
    : (query.data?.valueCounts ?? [])
        .filter((item) => item.value !== null)
        .slice(0, 5)
        .map((item) => item.value);
  return (
    <div className="protocol-field-profile">
      <span>
        <strong>Format:</strong> {attribute?.type.replaceAll('_', ' ') ?? 'text identifier'}
        {attribute ? ` · ${attribute.owner === 'patient' ? 'Patient' : 'Slide / case'}` : ''}
      </span>
      <span>
        <strong>Typical values:</strong>{' '}
        {query.isPending && !query.error
          ? 'Reading values…'
          : examples.length
            ? examples.map((value, index) => <code key={index}>{String(value)}</code>)
            : 'No non-missing examples available'}
      </span>
      {identity ? (
        <small>Examples from the first five dataset slides; IDs are matched exactly.</small>
      ) : (
        <small>
          Most frequent values across the frozen dataset, before these rules. Matching preserves
          spelling and case.
        </small>
      )}
      <ErrorNotice error={query.error} />
    </div>
  );
}

export function DistributionBars({
  values,
  caption,
  limit = 8,
  distinctCount,
}: {
  values: { value: string | null; count: number }[];
  caption: string;
  limit?: number;
  distinctCount?: number;
}) {
  const displayed = values.slice(0, limit);
  const maximum = Math.max(1, ...displayed.map((item) => item.count));
  return (
    <figure className="protocol-distribution">
      <figcaption>{caption}</figcaption>
      {displayed.length ? (
        displayed.map((item) => (
          <div
            className="science-bar"
            key={item.value === null ? 'missing' : `value:${item.value}`}
          >
            <span title={item.value ?? 'Missing'}>{item.value ?? 'Missing'}</span>
            <span className="science-bar-track" aria-hidden="true">
              <span style={{ width: `${(item.count / maximum) * 100}%` }} />
            </span>
            <strong>{item.count.toLocaleString()}</strong>
          </div>
        ))
      ) : (
        <p className="muted">No values in this population.</p>
      )}
      {(distinctCount ?? values.length) > displayed.length ? (
        <small className="muted">
          Showing {displayed.length} of {distinctCount ?? values.length} distinct values.
        </small>
      ) : null}
    </figure>
  );
}

export function CohortStats({ stats, total }: { stats: ProtocolCohortStats; total?: number }) {
  return (
    <div className="protocol-population">
      <div className="science-metrics protocol-metrics">
        <Metric
          label="Eligible slides"
          value={stats.totalSlides.toLocaleString()}
          note={total === undefined ? undefined : `of ${total.toLocaleString()} dataset slides`}
        />
        <Metric
          label="Verified patients"
          value={stats.patientCount.toLocaleString()}
          note="Distinct supplied patient IDs"
        />
        {stats.fallbackSlideCount > 0 ? (
          <Metric
            label="Slide ID fallback groups"
            value={stats.fallbackSlideCount.toLocaleString()}
            note="One group per unresolved slide; not verified patients"
          />
        ) : null}
        {stats.unlinkedSlideCount > 0 ? (
          <Metric
            label="Unresolved slides"
            value={stats.unlinkedSlideCount.toLocaleString()}
            note="Revise the dataset to link patients or confirm fallback"
          />
        ) : null}
      </div>
      {total !== undefined && total > 0 ? (
        <div className="protocol-coverage">
          <progress
            aria-label="Slides included by eligibility rules"
            value={stats.totalSlides}
            max={total}
          />
          <span>
            {stats.totalSlides.toLocaleString()} included ·{' '}
            {(total - stats.totalSlides).toLocaleString()} excluded by eligibility rules
          </span>
        </div>
      ) : null}
    </div>
  );
}

function SampleRows({
  rows,
  fields,
  label,
  expanded = false,
}: {
  rows: DataRecord[];
  fields: string[];
  label: string;
  expanded?: boolean;
}) {
  const attributes = [...new Set(fields)]
    .filter((field) => field && !['Slide_ID', 'Patient_ID'].includes(field))
    .slice(0, 4);
  return rows.length ? (
    <details className="protocol-sample" open={expanded}>
      <summary>
        Typical rows in {label} ({rows.length} examples)
      </summary>
      <div className="table-wrap">
        <table>
          <caption>
            Illustrative rows from this set; counts above describe the complete set.
          </caption>
          <thead>
            <tr>
              <th>Slide_ID</th>
              <th>Patient_ID / grouping</th>
              {attributes.map((field) => (
                <th key={field}>{field}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.slideId}>
                <td>{row.slideId}</td>
                <td>
                  {row.patientId ?? 'Unresolved'}
                  {row.patientIdSource === 'slide_fallback' ? (
                    <small className="science-block">Slide ID fallback</small>
                  ) : null}
                </td>
                {attributes.map((field) => (
                  <td key={field}>
                    {row.attributes[field] === null || row.attributes[field] === undefined
                      ? 'Missing'
                      : String(row.attributes[field])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  ) : null;
}

export function CohortSample({
  stats,
  fields,
}: {
  stats: ProtocolCohortStats;
  fields: string[];
}) {
  return <SampleRows rows={stats.sample} fields={fields} label="the eligible cohort" />;
}

export function PartitionLive({
  partition,
  label,
  fields,
  total,
}: {
  partition: ProtocolPartitionStats;
  label: string;
  fields: string[];
  total: number;
}) {
  const stats = partition.expanded;
  return (
    <div className="protocol-partition-live">
      <div className="science-filter-chips">
        <Badge tone="purple">{stats.totalSlides.toLocaleString()} slides</Badge>
        <Badge>{stats.patientCount.toLocaleString()} verified patients</Badge>
        {stats.fallbackSlideCount > 0 ? (
          <Badge>{stats.fallbackSlideCount.toLocaleString()} Slide ID fallback groups</Badge>
        ) : null}
        {stats.unlinkedSlideCount > 0 ? (
          <Badge tone="orange">
            {stats.unlinkedSlideCount.toLocaleString()} unresolved slides
          </Badge>
        ) : null}
      </div>
      {total > 0 ? (
        <progress
          aria-label={`${label} share of eligible slides`}
          max={total}
          value={stats.totalSlides}
        />
      ) : null}
      <p className="muted">
        {partition.selection === 'remaining'
          ? 'All eligible groups remaining after the test and validation selections.'
          : partition.selection === 'none'
            ? 'No fixed selection for this set.'
            : `${partition.directMatches.totalSlides.toLocaleString()} slides match these conditions directly. The set includes all eligible slides from their groups.`}
      </p>
      <SampleRows rows={stats.sample} fields={fields} label={label} expanded />
    </div>
  );
}
