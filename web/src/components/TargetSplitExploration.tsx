import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { targetSplits, testingShareText, type TargetSplitPartition, type TargetSplitPartitionPreview, type TargetSplitPartitionRequest, type TargetSplitUnit } from '../api/targetSplits';
import { DistributionBars } from './ProtocolExploration';
import { Metric } from './ui';
import { scienceKey } from './ScientificUI';

export const targetSplitExplorationKey = (project: string, request: TargetSplitPartitionRequest) => [...scienceKey(project), 'target-split-exploration', JSON.stringify(request)];

/** Hide old counts as soon as conditions change; only the latest settled request is shown. */
export function useTargetSplitExploration(project: string, request: TargetSplitPartitionRequest, enabled: boolean) {
  const serialized = JSON.stringify(request);
  const [settled, setSettled] = useState(serialized);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(serialized), 300);
    return () => window.clearTimeout(timer);
  }, [serialized]);
  const changing = settled !== serialized;
  const query = useQuery({ queryKey: [...scienceKey(project), 'target-split-exploration', settled],
    queryFn: () => targetSplits.partitionPreview(project, JSON.parse(settled) as TargetSplitPartitionRequest),
    enabled: enabled && Boolean(request.datasetId) && !changing, retry: false });
  return { data: changing ? undefined : query.data, error: changing ? null : query.error,
    loading: enabled && Boolean(request.datasetId) && (changing || query.isPending) };
}

export function TargetSplitSelection({ part, total, label, splitUnit = 'patient' }: { part: TargetSplitPartition; total: number; label: 'Training' | 'Testing'; splitUnit?: TargetSplitUnit }) {
  const selection = part.selection;
  const outside = Math.max(0, total - part.slides);
  return <section className="protocol-population target-split-selection" aria-label={`${label} live selection`} aria-live="polite">
    <div className="protocol-population-heading"><span>{label.toUpperCase()} SELECTION</span><small>Counts update as you edit</small></div>
    <div className="science-metrics protocol-metrics">
      <Metric label={`${label} slides`} value={part.slides.toLocaleString()} note={`of ${total.toLocaleString()} eligible slides`} />
      {splitUnit === 'patient' ? <Metric label={`${label} verified patients`} value={part.patients.toLocaleString()} note="Distinct supplied patient IDs" /> : null}
      {splitUnit === 'patient' && part.fallbackSlides > 0 ? <Metric label="Slide ID fallback groups" value={part.fallbackSlides.toLocaleString()} note="One group per slide; not verified patients" /> : null}
      {splitUnit === 'patient' && part.unlinkedSlides > 0 ? <Metric label="Unresolved slides" value={part.unlinkedSlides.toLocaleString()} note="Resolve patient links before freezing" /> : null}
    </div>
    <div className="protocol-coverage">
      <progress aria-label={`${label} share of eligible slides`} value={part.slides} max={Math.max(1, total)} />
      <span>{part.slides.toLocaleString()} selected · {outside.toLocaleString()} outside this set</span>
    </div>
    {selection?.mode === 'remaining' ? <p className="muted">All eligible {splitUnit === 'slide' ? 'slides' : 'patient groups'} outside the {label === 'Training' ? 'testing' : 'training'} selection.</p>
      : selection?.mode === 'none' ? <p className="muted">No {label.toLowerCase()} conditions: no {label.toLowerCase()} slides selected.</p>
      : selection && ['rules', 'imported'].includes(selection.mode) ? <p className="muted">{selection.directMatches.totalSlides.toLocaleString()} {selection.directMatches.totalSlides === 1 ? 'slide matches' : 'slides match'} {selection.mode === 'rules' ? 'these conditions' : 'the mapped values'} directly{splitUnit === 'patient' ? ` · ${selection.expanded.totalSlides.toLocaleString()} slides after including their eligible patient groups.` : '.'}{selection.expanded.totalSlides !== part.slides ? ` Conflicting ${splitUnit === 'slide' ? 'slides' : 'groups'} are excluded from the assigned counts until the overlap is resolved.` : ''}</p> : null}
  </section>;
}

export function TargetSplitPartitionCounts({ value, splitUnit = 'patient' }: { value: TargetSplitPartitionPreview; splitUnit?: TargetSplitUnit }) {
  return <div className="target-split-partition-grid target-split-live">
    <TargetSplitSelection part={value.partitions.train} total={value.summary.eligibleSlides} splitUnit={splitUnit} label="Training" />
    <TargetSplitSelection part={value.partitions.test} total={value.summary.eligibleSlides} splitUnit={splitUnit} label="Testing" />
  </div>;
}

export function TargetSplitSelectionSummary({ value, splitUnit = 'patient' }: { value: TargetSplitPartitionPreview; splitUnit?: TargetSplitUnit }) {
  const unassigned = Math.max(0, value.summary.eligibleSlides - value.summary.selectedSlides);
  return <div className="target-split-selection-summary" aria-live="polite">
    <p>{value.summary.selectedSlides.toLocaleString()} of {value.summary.eligibleSlides.toLocaleString()} eligible slides assigned · {unassigned.toLocaleString()} excluded by split conditions.</p>
    {testingShareText(value.summary, splitUnit) ? <p>Testing holds {testingShareText(value.summary, splitUnit)}.</p> : null}
    <p className="muted">{value.membershipStatus === 'provisional' ? 'These assignments are provisional. Resolve the findings below before continuing.' : splitUnit === 'slide' ? 'Each slide is assigned independently. Defining targets does not redraw these assignments.' : 'Patient groups stay in one set. Defining targets does not redraw these assignments.'}</p>
  </div>;
}

export function TargetSplitDistributions({ partitions, mapped = false, unlabeledTest = false, partition, splitUnit = 'patient' }: { partitions: Record<'train' | 'test', TargetSplitPartition>; mapped?: boolean; unlabeledTest?: boolean; partition?: 'train' | 'test'; splitUnit?: TargetSplitUnit }) {
  const visible = (partition ? [partition] : ['train', 'test'] as const).filter((key) => !(key === 'test' && unlabeledTest));
  return <div className={`target-split-distributions${visible.length > 1 ? ' target-split-partition-grid' : ''}`} aria-live="polite">
    {visible.map((key) => {
      const part = partitions[key];
      const title = key === 'train' ? 'Training' : 'Testing';
      const distribution = part.target;
      const slideValues = distribution ? mapped
        ? Object.entries(distribution.classCounts).map(([value, count]) => ({ value, count }))
        : distribution.values.map(({ value, slides }) => ({ value, count: slides })) : [];
      const patientValues = distribution && splitUnit === 'patient' ? mapped
        ? Object.entries(distribution.patientClassCounts).map(([value, count]) => ({ value, count }))
        : distribution.values.map(({ value, patients }) => ({ value, count: patients })) : [];
      return <section className="target-split-partition-card" key={key} aria-label={`${title} target distributions`}>
        <h3>{title} target distribution</h3>
        <p className="muted">{part.slides.toLocaleString()} selected slides{splitUnit === 'patient' ? ` · ${part.patients.toLocaleString()} verified patients` : ''}</p>
        {distribution ? <>
          <DistributionBars values={slideValues} caption={`${distribution.field} · ${title.toLowerCase()} slides${mapped ? ' by mapped label' : ' by source value'}`} distinctCount={mapped ? undefined : distribution.distinctCount} />
          {splitUnit === 'patient' ? <DistributionBars values={patientValues} caption={`${distribution.field} · ${title.toLowerCase()} verified patients${mapped ? ' by mapped label' : ' by source value'}`} distinctCount={mapped ? undefined : distribution.distinctCount} /> : null}
          {mapped ? <p className="muted">{distribution.missingSlides.toLocaleString()} missing-label slides · {distribution.unmappedSlides.toLocaleString()} unmapped-label slides.{splitUnit === 'patient' ? ` Patients: ${distribution.mixedLabelPatients.toLocaleString()} with mixed labels, ${distribution.missingLabelPatients.toLocaleString()} with missing labels, ${distribution.unmappedLabelPatients.toLocaleString()} with unmapped labels.` : ''}</p> : splitUnit === 'patient' && distribution.mixedValuePatients ? <p className="muted">{distribution.mixedValuePatients.toLocaleString()} patients have more than one source value.</p> : null}
          {splitUnit === 'patient' ? <p className="muted">Patient bars count verified patients with a consistent {mapped ? 'mapped label' : 'source value'}. Slide-ID fallback groups are not counted as patients.</p> : null}
        </> : <p className="muted">Choose a target field to inspect this partition.</p>}
      </section>;
    })}
  </div>;
}
