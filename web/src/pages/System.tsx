import { useState } from 'react';
import { ApiError } from '../api/client';
import { useSystem, useSystemCompute } from '../api/queries';
import type { SystemCompute, SystemStatus } from '../api/types';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import './System.css';

const unavailable = 'Unavailable';
const validNumber = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function formatComputeBytes(value: number | null | undefined): string {
  if (!validNumber(value)) return unavailable;
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  const exponent = value === 0 ? 0 : Math.max(0, Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1));
  const amount = value / 1024 ** exponent;
  return `${amount.toLocaleString(undefined, { maximumFractionDigits: exponent === 0 ? 0 : 1 })} ${units[exponent]}`;
}

export function formatUptime(value: number | null | undefined): string {
  if (!validNumber(value)) return unavailable;
  const minutes = Math.floor(value / 60);
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor(minutes / 60) % 24;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  return `${minutes}m`;
}

function percentage(value: number | null): string {
  return validNumber(value) ? `${value.toFixed(1)}%` : unavailable;
}

function metricValue(value: number | null, unit = ''): string {
  return validNumber(value) ? `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}${unit}` : unavailable;
}

function UsageMeter({ label, value, pending = false }: { label: string; value: number | null; pending?: boolean }) {
  const measured = validNumber(value);
  return <div className="system-usage">
    <div className="system-usage-label"><span>{label}</span><strong>{measured ? percentage(value) : pending ? 'Collecting sample' : unavailable}</strong></div>
    {measured ? <progress max={100} value={Math.min(100, value)} aria-label={label}>{percentage(value)}</progress> : <div className="system-usage-unavailable" aria-hidden="true" />}
  </div>;
}

export function ComputeDashboard({ data, stale = false, live = true }: { data: SystemCompute; stale?: boolean; live?: boolean }) {
  const { cpu, memory, gpu, host, disks } = data;
  const sampledDate = new Date(data.sampledAt);
  const sampledLabel = Number.isNaN(sampledDate.valueOf()) ? 'Unknown sample time' : sampledDate.toLocaleString();
  return <section className="system-compute" aria-label="Host compute and storage">
    <div className="system-sample">
      <div><strong>{host.hostname || 'Python host'}</strong><span>{[host.platform, host.release].filter(Boolean).join(' · ')}</span></div>
      <div><Badge tone={stale ? 'amber' : live ? 'green' : 'neutral'}>{stale ? 'Last known sample' : live ? 'Live · every 5 seconds' : 'Updates paused'}</Badge><span>Updated <time dateTime={data.sampledAt}>{sampledLabel}</time> · Uptime {formatUptime(host.uptimeSeconds)}</span></div>
    </div>
    <div className="grid-2">
      <Panel title="CPU" subtitle={cpu.model ?? 'Processor details unavailable'}>
        <UsageMeter label="CPU utilization" value={cpu.utilizationPercent} pending={cpu.status === 'available'} />
        <dl className="detail-list">
          <div><dt>Physical / logical cores</dt><dd>{metricValue(cpu.physicalCores)} / {metricValue(cpu.logicalCores)}</dd></div>
          <div><dt>CPUs available to this service</dt><dd>{metricValue(cpu.availableCores)}</dd></div>
          <div><dt>Load average · 1 / 5 / 15 min</dt><dd>{cpu.loadAverage ? cpu.loadAverage.map((load) => metricValue(load)).join(' / ') : unavailable}</dd></div>
        </dl>
        {cpu.message ? <p className="system-metric-note">{cpu.message}</p> : null}
      </Panel>
      <Panel title="RAM" subtitle="System memory">
        <UsageMeter label="RAM utilization" value={memory.utilizationPercent} />
        <dl className="detail-list">
          <div><dt>Used / total</dt><dd>{formatComputeBytes(memory.usedBytes)} / {formatComputeBytes(memory.totalBytes)}</dd></div>
          <div><dt>Available</dt><dd>{formatComputeBytes(memory.availableBytes)}</dd></div>
          <div><dt>Swap used / total</dt><dd>{memory.swapTotalBytes === 0 ? 'No swap configured' : `${formatComputeBytes(memory.swapUsedBytes)} / ${formatComputeBytes(memory.swapTotalBytes)}`}</dd></div>
        </dl>
        {memory.message ? <p className="system-metric-note">{memory.message}</p> : null}
      </Panel>
    </div>
    <Panel title="GPU & VRAM" subtitle="NVIDIA devices visible to the Python host" actions={<Badge tone={gpu.status === 'available' ? 'neutral' : 'amber'}>{gpu.status === 'available' ? `${gpu.devices.length} ${gpu.devices.length === 1 ? 'GPU' : 'GPUs'}` : 'Telemetry unavailable'}</Badge>}>
      {gpu.devices.length > 0 ? <div className="system-gpus">{gpu.devices.map((device) => <article key={device.uuid ?? device.index} className="system-gpu" aria-label={`GPU ${device.index}: ${device.name}`}>
        <div className="system-gpu-title"><h3>GPU {device.index} · {device.name}</h3><span>Driver {device.driverVersion ?? unavailable}</span></div>
        <div className="grid-2">
          <UsageMeter label={`GPU ${device.index} utilization`} value={device.utilizationPercent} />
          <UsageMeter label={`GPU ${device.index} VRAM used`} value={device.memoryUtilizationPercent} />
        </div>
        <dl className="system-gpu-details">
          <div><dt>VRAM used / total</dt><dd>{formatComputeBytes(device.memoryUsedBytes)} / {formatComputeBytes(device.memoryTotalBytes)}</dd></div>
          <div><dt>VRAM free</dt><dd>{formatComputeBytes(device.memoryFreeBytes)}</dd></div>
          <div><dt>Temperature</dt><dd>{metricValue(device.temperatureCelsius, ' °C')}</dd></div>
          <div><dt>Power / limit</dt><dd>{metricValue(device.powerWatts, ' W')} / {metricValue(device.powerLimitWatts, ' W')}</dd></div>
        </dl>
      </article>)}</div> : <p className="system-empty">{gpu.status === 'available' ? 'No NVIDIA GPUs detected.' : 'GPU statistics could not be read on this host.'}</p>}
      {gpu.message ? <p className="system-metric-note">{gpu.message}</p> : null}
    </Panel>
    <Panel title="Disk capacity" subtitle="Workspace and allowed data folders; folders on the same filesystem share this capacity.">
      {disks.length > 0 ? <div className="system-disk-table"><table><thead><tr><th scope="col">Location</th><th scope="col">Used / total</th><th scope="col">Free</th><th scope="col">Usage</th></tr></thead><tbody>
        {disks.map((disk) => <tr key={`${disk.role}:${disk.path}`}><th scope="row"><span>{disk.role === 'workspace' ? 'Workspace' : 'Data'}</span><code>{disk.path}</code>{disk.message ? <small>{disk.message}</small> : null}</th><td>{formatComputeBytes(disk.usedBytes)} / {formatComputeBytes(disk.totalBytes)}</td><td>{formatComputeBytes(disk.freeBytes)}</td><td><UsageMeter label={`${disk.path} disk utilization`} value={disk.utilizationPercent} /></td></tr>)}
      </tbody></table></div> : <p className="system-empty">No storage locations were reported.</p>}
    </Panel>
  </section>;
}

export function computeErrorMessage(error: Error): string {
  return error instanceof ApiError && error.status === 404
    ? 'The running HistoPilot service does not support compute statistics yet. Restart HistoPilot manually, then select Refresh status.'
    : `Compute statistics could not be refreshed. ${error.message}`;
}

function ServiceDetails({ data, stale }: { data: SystemStatus; stale: boolean }) {
  return <details className="system-service-details">
    <summary>Service & workspace details</summary>
    <div className="grid-2">
      <Panel title="Control service" actions={<Badge tone={stale ? 'amber' : 'green'}>{stale ? 'Last known state' : 'Connected'}</Badge>}>
        <ul className="detail-list">
          <li><span>Application mode</span><strong>{data.mode}</strong></li>
          <li><span>Process role</span><strong>{data.control.process}</strong></li>
          <li><span>CUDA models in control process</span><strong>{data.control.cudaModelsLoaded ? 'Loaded' : 'None'}</strong></li>
          <li><span>Browser connection</span><strong className="mono break-all">{typeof window === 'undefined' ? 'Browser connection' : window.location.origin}</strong></li>
        </ul>
      </Panel>
      <Panel title="Local workspace">
        <ul className="detail-list">
          <li><span>Workspace location</span><strong className="mono break-all">{data.workspace}</strong></li>
          <li><span>Application database</span><strong>{data.storage.engine} · {data.storage.journalMode.toUpperCase()}</strong></li>
          <li><span>Schema version</span><strong>{data.storage.schemaVersion}</strong></li>
          <li><span>Original WSI policy</span><strong>{data.sourcesReadOnly ? 'Referenced read-only' : 'See server configuration'}</strong></li>
        </ul>
      </Panel>
      <Panel title="Compute workers" actions={<Badge tone={data.workers.nativeExecutionImplemented ? 'green' : 'neutral'}>{data.workers.nativeExecutionImplemented ? 'Native execution implemented' : 'Check module runtimes'}</Badge>}>
        <ul className="detail-list">
          <li><span>ABMIL training, evaluation & attention</span><strong>{data.workers.nativeExecutionImplemented ? 'Implemented · check module runtime' : 'Check availability in each module'}</strong></li>
          <li><span>TRIDENT feature extraction</span><strong>{data.workers.executionEnabled ? 'Runtime ready' : 'Extraction runtime setup required'}</strong></li>
          <li><span>Persistent tmux jobs</span><strong>{data.workers.tmuxAvailable === undefined ? 'Availability not reported' : data.workers.tmuxAvailable ? 'tmux available' : 'tmux unavailable'}</strong></li>
        </ul>
        <p className="system-metric-note">{data.workers.status}</p>
        <div className="inline-actions"><a className="text-link" href="#features">Prepare features →</a><a className="text-link" href="#experiments">Open experiments →</a></div>
      </Panel>
      <Panel title="Storage formats">
        <ul className="detail-list">
          <li><span>SQLite</span><strong>Application metadata</strong></li>
          <li><span>Parquet</span><strong>Dataset and analytical tables</strong></li>
          <li><span>HDF5</span><strong>Feature tensors & coordinates</strong></li>
          <li><span>Memory-mapped packs</span><strong>Verified feature loading for workers</strong></li>
          <li><span>JSON / YAML</span><strong>Experiment manifests</strong></li>
        </ul>
        <a className="text-link" href="#dataset">Manage server folder references →</a>
      </Panel>
    </div>
    <p className="system-metric-note">Workers run model jobs separately from the control service. Folder paths and these statistics belong to the Python host, including when you connect from another computer.</p>
  </details>;
}

export default function System() {
  const [live, setLive] = useState(true);
  const system = useSystem();
  const compute = useSystemCompute(live);
  const sampledAt = compute.data ? Date.parse(compute.data.sampledAt) : null;
  const computeStale = compute.isError || (sampledAt !== null && (!Number.isFinite(sampledAt) || Date.now() - sampledAt > 15_000));
  const refreshing = system.isFetching || compute.isFetching;
  return <div className="system-page">
    <PageHeader eyebrow="LOCAL WORKSPACE" title="System & storage" description="Monitor CPU, memory, GPUs and disk capacity on your HistoPilot host." actions={<>
      <button className="btn btn-secondary" aria-pressed={!live} onClick={() => setLive((value) => !value)}>{live ? 'Pause live updates' : 'Resume live updates'}</button>
      <button className="btn btn-secondary" disabled={refreshing} onClick={() => { void system.refetch(); void compute.refetch(); }}><Icon name="reset" />{refreshing ? 'Refreshing…' : 'Refresh status'}</button>
    </>} />
    {compute.error ? <div className="callout callout-warning" role="alert">{computeErrorMessage(compute.error)}{compute.data ? ' The last available sample is shown below.' : ''}</div> : null}
    {compute.isPending ? <p role="status">Reading host compute statistics…</p> : null}
    {compute.data ? <ComputeDashboard data={compute.data} stale={computeStale} live={live} /> : null}
    <ErrorNotice error={system.error} />
    {system.isPending ? <p role="status">Reading control service status…</p> : null}
    {system.data ? <ServiceDetails data={system.data} stale={system.isError} /> : null}
  </div>;
}
