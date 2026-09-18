import { useEffect, useId, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { development, trainingActive } from '../api/development';
import type { TrainingExecution, TrainingGPU, TrainingResourceHistory, TrainingResourceSample } from '../api/development';
import './RunResourceUsage.css';

const measured = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const percent = (value: number | null | undefined) => measured(value) && value <= 100 ? value : null;
const memory = (value: number | null | undefined) => measured(value) ? `${value.toFixed(2)} GiB` : 'Not recorded';
const unavailableHistory = (error: unknown) => error instanceof ApiError && error.status === 404;
const usedRam = (sample?: TrainingResourceSample) => sample && measured(sample.host.totalRamGb) && measured(sample.host.availableRamGb) && sample.host.totalRamGb >= sample.host.availableRamGb
  ? sample.host.totalRamGb - sample.host.availableRamGb : null;
const dateLabel = (at: string) => Number.isFinite(Date.parse(at)) ? new Date(at).toLocaleString() : 'Unknown time';

export function resourceHistoryPollInterval(active: boolean, error: unknown, intervalSeconds = 15) {
  return active && !unavailableHistory(error) ? Math.max(3000, intervalSeconds * 1000 || 15000) : false;
}

/** A recorded worker sample always wins over unrelated current machine metrics. */
export function resourceSamples(execution: TrainingExecution, history?: TrainingResourceHistory): TrainingResourceSample[] {
  const samples = new Map<number, TrainingResourceSample>();
  for (const sample of [...(history?.rows ?? []), ...(execution.telemetry?.latest ? [execution.telemetry.latest] : [])]) {
    const at = Date.parse(sample.at);
    if (Number.isFinite(at)) samples.set(at, sample);
  }
  return [...samples.entries()].sort(([a], [b]) => a - b).map(([, sample]) => sample);
}

export function ResourceCards({ project, execution }: { project?: string; execution: TrainingExecution }) {
  return project ? <RecordedResources project={project} execution={execution} /> : <ResourceDashboard execution={execution} />;
}

function RecordedResources({ project, execution }: { project: string; execution: TrainingExecution }) {
  const active = trainingActive(execution);
  const history = useQuery({
    // A terminal status gets one final read even if the last active read was fresh.
    queryKey: ['training-resource-history', project, execution.batchId, execution.status, active ? 'active' : execution.updatedAt],
    queryFn: ({ signal }) => development.resourceHistory(project, execution.batchId, signal),
    refetchInterval: (query) => resourceHistoryPollInterval(active, query.state.error, execution.telemetry?.intervalSeconds),
    refetchOnWindowFocus: (query) => active && !unavailableHistory(query.state.error),
    staleTime: (query) => !active || unavailableHistory(query.state.error) ? Infinity : 3000,
    retry: false,
  });
  return <ResourceDashboard execution={execution} history={history.data} error={history.error} loading={history.isPending} refreshing={history.isFetching} onRefresh={() => void history.refetch()} />;
}

export function ResourceDashboard({ execution, history, error, loading = false, refreshing = false, onRefresh }: {
  execution: TrainingExecution; history?: TrainingResourceHistory; error?: unknown; loading?: boolean;
  refreshing?: boolean; onRefresh?: () => void;
}) {
  const id = useId();
  const active = trainingActive(execution);
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 15000);
    return () => window.clearInterval(timer);
  }, [active]);
  const samples = resourceSamples(execution, history);
  const latest = samples.at(-1) ?? execution.telemetry?.latest;
  const interval = execution.telemetry?.intervalSeconds || 15;
  const stale = Boolean(active && latest && (!Number.isFinite(Date.parse(latest.at)) || now - Date.parse(latest.at) > Math.max(45000, interval * 3000)));
  const gpus = new Map<string, TrainingGPU>();
  for (const sample of samples) for (const gpu of sample.gpus) gpus.set(gpu.uuid || String(gpu.index), gpu);
  if (!samples.length) for (const gpu of latest?.gpus ?? []) gpus.set(gpu.uuid || String(gpu.index), gpu);
  const findGPU = (sample: TrainingResourceSample | undefined, gpu: TrainingGPU) => sample?.gpus.find((item) => (item.uuid || String(item.index)) === (gpu.uuid || String(gpu.index)));
  const cpu = percent(latest?.host.cpuUtilizationPercent);
  const ram = usedRam(latest);
  return <section className="run-resource-usage" aria-labelledby={`${id}-title`}>
    <div className="run-resource-heading"><div><h3 id={`${id}-title`}>Resource usage</h3><p>CPU and memory on the training host · recorded during this batch</p></div>
      <div className="run-resource-refresh">{latest ? <span className={`run-resource-state${stale || error ? ' is-stale' : ''}`}>{!active ? 'Recorded' : stale || error ? 'Last recorded' : 'Updating'}</span> : null}{onRefresh ? <button type="button" className="btn btn-secondary btn-small" disabled={refreshing} onClick={onRefresh}>{refreshing ? 'Refreshing…' : error ? 'Retry resource history' : 'Refresh'}</button> : null}</div>
    </div>
    {error ? <p className="run-resource-notice" role="status">{unavailableHistory(error) ? 'Resource history is unavailable from this service. Restart HistoPilot manually after updating to enable history, then retry. Saved resource snapshots remain available.' : 'Resource history could not refresh. Showing the last recorded measurements.'}</p> : null}
    {history?.warning ? <p className="run-resource-notice" role="status">{history.warning}</p> : null}
    {stale ? <p className="run-resource-notice" role="status">The resource sample is stale. These values may no longer reflect current usage.</p> : null}
    {!latest ? <p className="run-resource-empty" role="status">{loading ? 'Loading recorded resource usage…' : 'Resource usage has not been recorded yet.'}</p> : <>
      <div className="run-resource-grid">
        <ResourceMetric title="CPU" detail={cpu === null ? 'Not recorded by this worker' : 'Host utilization'} value={cpu === null ? 'Not recorded' : `${cpu.toFixed(0)}%`} tone="cpu" suffix={`${latest.host.cpuCount} logical CPUs`} samples={samples} interval={interval} read={(sample) => percent(sample.host.cpuUtilizationPercent)} unit="%" ceiling={100} />
        <ResourceMetric title="RAM" detail="Host memory used" value={memory(ram)} tone="ram" suffix={`${memory(latest.host.totalRamGb)} total RAM`} samples={samples} interval={interval} read={usedRam} unit="GiB" ceiling={measured(latest.host.totalRamGb) ? latest.host.totalRamGb : undefined} />
        {[...gpus.values()].map((gpu) => {
          const current = findGPU(latest, gpu);
          const utilization = percent(current?.utilizationPercent);
          const peaks = samples.map((sample) => findGPU(sample, gpu)?.usedMemoryGb).filter(measured);
          // Legacy peaks are indexed by slot, so they cannot identify hardware after a host change.
          const slotHasOneDevice = [...gpus.values()].filter((device) => device.index === gpu.index).length === 1;
          const legacyPeak = slotHasOneDevice ? execution.telemetry?.peak.gpuUsedMemoryGb[String(gpu.index)] : undefined;
          if (measured(legacyPeak)) peaks.push(legacyPeak);
          const peak = peaks.length ? Math.max(...peaks) : null;
          return <div className="run-resource-gpu" key={gpu.uuid || gpu.index}>
            <ResourceMetric title={`GPU ${gpu.index}`} detail={gpu.name} value={utilization === null ? 'Utilization unavailable' : `${utilization.toFixed(0)}% utilization`} tone="gpu" suffix="Device-wide GPU usage" samples={samples} interval={interval} read={(sample) => percent(findGPU(sample, gpu)?.utilizationPercent)} unit="%" ceiling={100} />
            <ResourceMetric title={`VRAM · GPU ${gpu.index}`} detail="Device memory used" value={memory(current?.usedMemoryGb)} tone="vram" suffix={`${memory(current?.totalMemoryGb)} total VRAM${peak !== null ? ` · Peak ${memory(peak)}` : ''}`} samples={samples} interval={interval} read={(sample) => { const value = findGPU(sample, gpu)?.usedMemoryGb; return measured(value) ? value : null; }} unit="GiB" ceiling={measured(gpu.totalMemoryGb) ? gpu.totalMemoryGb : undefined} />
          </div>;
        })}
      </div>
      {!gpus.size ? <p className="run-resource-empty">GPU and VRAM measurements were not recorded for this batch.</p> : null}
      {latest.gpuProbeError ? <p className="run-resource-notice" role="status">GPU measurement unavailable: {latest.gpuProbeError}</p> : null}
      <div className="run-resource-caption"><span>{active ? 'Recorded' : 'Last resource sample'} <time dateTime={latest.at}>{dateLabel(latest.at)}</time>{!active ? ' · recorded measurements, not live device usage' : ''}</span><span>Samples every {interval}s · {samples.length} recorded {samples.length === 1 ? 'sample' : 'samples'}</span></div>
      <p className="run-resource-scope">CPU and RAM are host-wide. GPU usage includes other programs. Gaps indicate missing samples; per-run memory appears below.</p>
      {history?.truncated ? <p className="run-resource-scope">Showing the latest {history.rows.length} of {history.totalRowsIsLowerBound ? 'at least ' : ''}{history.totalRows} saved samples.</p> : null}
    </>}
  </section>;
}

interface ResourcePoint { at: number; value: number | null }
/** Do not join unavailable samples or gaps when a worker paused/stopped. */
export function resourceSegments(points: ResourcePoint[], maxGap: number): { at: number; value: number }[][] {
  const segments: { at: number; value: number }[][] = [];
  let current: { at: number; value: number }[] = [];
  let previous: number | undefined;
  for (const point of points) {
    if (!Number.isFinite(point.at) || !measured(point.value)) { current = []; previous = undefined; continue; }
    if (previous === undefined || point.at - previous > maxGap) { current = []; }
    if (!current.length) segments.push(current);
    current.push({ at: point.at, value: point.value });
    previous = point.at;
  }
  return segments;
}

function ResourceMetric({ title, detail, value, suffix, tone, samples, interval, read, unit, ceiling }: {
  title: string; detail: string; value: string; suffix: string; tone: 'cpu' | 'ram' | 'gpu' | 'vram';
  samples: TrainingResourceSample[]; interval: number; read: (sample: TrainingResourceSample) => number | null | undefined;
  unit: '%' | 'GiB'; ceiling?: number;
}) {
  const id = useId();
  const [dataOpen, setDataOpen] = useState(false);
  const points = samples.map((sample) => { const value = read(sample); return { at: Date.parse(sample.at), value: measured(value) ? value : null }; });
  const segments = resourceSegments(points, interval * 2500);
  const minTime = points[0]?.at ?? 0;
  const maxTime = Math.max(points.at(-1)?.at ?? 0, minTime + interval * 1000);
  const maxValue = Math.max(ceiling ?? 0, ...points.map((point) => point.value ?? 0), 1);
  const x = (at: number) => 35 + (at - minTime) / (maxTime - minTime) * 265;
  const y = (value: number) => 102 - value / maxValue * 84;
  const time = (at: number) => new Date(at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  return <article className={`run-resource-metric tone-${tone}`} aria-labelledby={`${id}-title`}>
    <h4 id={`${id}-title`}>{title}</h4><p className="run-resource-detail">{detail}</p><strong className="run-resource-value">{value}</strong><p className="run-resource-detail run-resource-capacity">{suffix}</p>
    {segments.length ? <svg className="run-resource-chart" viewBox="0 0 316 128" role="img" aria-labelledby={`${id}-chart-title ${id}-chart-description`}>
      <title id={`${id}-chart-title`}>{`${title} history`}</title><desc id={`${id}-chart-description`}>{unit === '%' ? 'Utilization in percent' : 'Memory used in GiB'} over recorded time. Missing measurements and time gaps are not connected. Exact timestamps and values are available below.</desc>
      {[0, 0.5, 1].map((fraction) => <g key={fraction}><line className="run-resource-gridline" x1="35" x2="300" y1={y(maxValue * fraction)} y2={y(maxValue * fraction)} /><text x="29" y={y(maxValue * fraction) + 3} textAnchor="end">{(maxValue * fraction).toFixed(maxValue >= 10 ? 0 : 1)}</text></g>)}
      <text className="run-resource-unit" x="35" y="10">{unit}</text>
      {segments.map((segment, index) => segment.length === 1 ? <circle key={index} className="run-resource-point" cx={x(segment[0].at)} cy={y(segment[0].value)} r="3" /> : <path key={index} className="run-resource-line" d={segment.map((point, number) => `${number ? 'L' : 'M'}${x(point.at).toFixed(2)} ${y(point.value).toFixed(2)}`).join(' ')} />)}
      <text x="35" y="121">{time(minTime)}</text>{points.length > 1 ? <text x="300" y="121" textAnchor="end">{time(points.at(-1)!.at)}</text> : null}
    </svg> : <div className="run-resource-chart-empty">No recorded {title.toLowerCase()} history</div>}
    {samples.length ? <details className="run-resource-data" onToggle={(event) => setDataOpen(event.currentTarget.open)}><summary>View {title} data</summary>{dataOpen ? <div className="table-wrap"><table><caption>{title} · {unit}</caption><thead><tr><th scope="col">Recorded time</th><th scope="col">{unit === '%' ? 'Utilization (%)' : 'Used (GiB)'}</th></tr></thead><tbody>{points.map((point) => <tr key={point.at}><td><time dateTime={new Date(point.at).toISOString()}>{new Date(point.at).toLocaleString()}</time></td><td>{point.value === null ? 'Not recorded' : point.value.toFixed(2)}</td></tr>)}</tbody></table></div> : null}</details> : null}
  </article>;
}
