import type { ResourcePolicy, TrainingRuntime } from '../api/development';

export function trainingCapacity(resources: ResourcePolicy, runtime?: TrainingRuntime) {
  const gpuSlots = resources.gpuIds.length ? resources.gpuIds.length * resources.runsPerGpu : null;
  const configured = Math.min(resources.maxConcurrentRuns, gpuSlots ?? Infinity);
  const cpuSlotsPerRun = resources.cpuThreadsPerRun + 2 * resources.dataLoaderWorkers;
  const host = runtime?.host;
  const cpuLimit = host ? Math.floor(host.cpuCount / cpuSlotsPerRun) : undefined;
  const ramLimit = host ? Math.floor(host.availableRamGb / resources.ramGbPerRun) : undefined;
  const missingGpu = Boolean(runtime && resources.gpuIds.length && (!runtime.cudaAvailable || resources.gpuIds.some((id) => id >= runtime.gpuCount)));
  const available = runtime?.available === false || missingGpu ? 0 : host ? Math.max(0, Math.min(configured, cpuLimit!, ramLimit!)) : undefined;
  return { configured, available, gpuSlots, cpuSlotsPerRun, cpuLimit, ramLimit, missingGpu };
}

export default function TrainingCapacity({ resources, runtime }: { resources: ResourcePolicy; runtime?: TrainingRuntime }) {
  const capacity = trainingCapacity(resources, runtime);
  return <aside className="development-capacity" aria-label="Effective training concurrency" aria-live="polite">
    <strong>Configured limit: {capacity.configured} concurrent {capacity.configured === 1 ? 'run' : 'runs'}</strong>
    <p>{resources.gpuIds.length ? `${resources.gpuIds.length} selected GPU${resources.gpuIds.length === 1 ? '' : 's'} × ${resources.runsPerGpu} run${resources.runsPerGpu === 1 ? '' : 's'} per GPU, capped at ${resources.maxConcurrentRuns} for this batch.` : `CPU execution, capped at ${resources.maxConcurrentRuns} runs for this batch.`}</p>
    {capacity.available !== undefined ? <p>Current CPU and RAM capacity: up to <strong>{capacity.available}</strong> concurrent {capacity.available === 1 ? 'run' : 'runs'}.</p> : <p>CPU and RAM availability are checked when the batch launches.</p>}
    {capacity.missingGpu ? <p role="status">A selected GPU is unavailable. Choose an available GPU or CPU execution.</p> : null}
    <small>GPU slots count processes; they do not guarantee enough VRAM. Other jobs can reduce availability.</small>
  </aside>;
}
