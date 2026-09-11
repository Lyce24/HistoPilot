import type { ExtractionJob, ExtractionState } from '../api/trident';
import { Icon } from './ui';
import './ExtractionProgress.css';

export const extractionStateLabel: Record<ExtractionState, string> = {
  starting: 'Preparing',
  running: 'Running',
  cancelling: 'Stopping',
  succeeded: 'Complete',
  failed: 'Failed',
  cancelled: 'Cancelled',
  interrupted: 'Interrupted',
};

export function extractionTaskLabel(job: ExtractionJob): string {
  switch (job.spec.options.task) {
    case 'seg': return 'Tissue segmentation';
    case 'coords': return 'Patch coordinates';
    case 'feat': return 'Feature extraction';
    default: return 'Full extraction';
  }
}

export function extractionModelLabel(job: ExtractionJob): string {
  const task = job.spec.options.task;
  const value = task === 'seg'
    ? job.spec.options.segmenter
    : task === 'coords'
      ? null
      : job.spec.options.slide_encoder || job.spec.options.patch_encoder;
  if (!value) return task === 'coords' ? 'Tissue patches' : 'TRIDENT';
  const name = String(value);
  return ({ uni_v1: 'UNI v1', uni_v2: 'UNI v2', hest: 'HEST', grandqc: 'GrandQC', otsu: 'Otsu', resnet50: 'ResNet-50' } as Record<string, string>)[name] ?? name;
}

export function formatExtractionDuration(seconds: number | null | undefined): string | null {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return null;
  const whole = Math.floor(seconds);
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return `${minutes}m ${whole % 60}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

function progressTitle(job: ExtractionJob) {
  switch (job.state) {
    case 'succeeded': return 'Extraction complete';
    case 'failed': return 'Extraction needs attention';
    case 'cancelled': return 'Extraction cancelled';
    case 'interrupted': return 'Extraction interrupted';
    case 'cancelling': return 'Stopping extraction';
    default: return job.progress?.label || (job.state === 'starting' ? 'Preparing extraction' : 'Processing slides');
  }
}

function progressDescription(job: ExtractionJob) {
  switch (job.state) {
    case 'succeeded': return 'Outputs passed validation and are ready to inspect.';
    case 'failed': return 'Review the message below before resuming this extraction.';
    case 'cancelled': return 'Review existing outputs or reuse these settings to resume.';
    case 'interrupted': return 'The worker stopped before confirming completion. Review the run details before resuming.';
    case 'cancelling': return 'Waiting for the worker to stop safely.';
    default: return job.progress?.detail || 'Progress details will appear when the current stage reports them.';
  }
}

export default function ExtractionProgress({ job }: { job: ExtractionJob }) {
  const progress = job.progress;
  const working = job.state === 'starting' || job.state === 'running';
  const successful = job.state === 'succeeded';
  const stopped = !working && !successful;
  const stages = progress?.stages ?? [];
  const activeIndex = stages.findIndex((stage) => stage.id === progress?.stage);
  const scope = progress?.scope;
  const scopeName = scope === 'batch' ? 'Current batch' : 'Current stage';
  const percent = scope && progress?.percent !== null && progress?.percent !== undefined && Number.isFinite(progress.percent)
    ? Math.min(100, Math.max(0, progress.percent))
    : null;
  const unit = progress?.unit ?? 'slides';
  const reportedCount = progress?.completed;
  const total = progress?.total;
  const validatedCoverage = !working && progress?.stage === 'validation' && job.result?.completedSlides !== undefined;
  const countLabel = validatedCoverage ? 'validated' : progress?.stage === 'validation' ? 'inspected' : 'processed';
  const counts = reportedCount !== null && reportedCount !== undefined
    ? `${reportedCount.toLocaleString()}${total !== null && total !== undefined ? ` of ${total.toLocaleString()}` : ''} ${unit} ${countLabel}`
    : null;
  const elapsed = formatExtractionDuration(progress?.elapsedSeconds);
  const stageElapsed = formatExtractionDuration(progress?.stageElapsedSeconds);
  const eta = working && scope ? formatExtractionDuration(progress?.etaSeconds) : null;
  const rate = working && scope && progress?.ratePerSecond !== null && progress?.ratePerSecond !== undefined && Number.isFinite(progress.ratePerSecond) && progress.ratePerSecond > 0
    ? `${(unit === 'slides' ? progress.ratePerSecond * 60 : progress.ratePerSecond).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${unit}/${unit === 'slides' ? 'min' : 'sec'}`
    : null;
  const stepLabel = working && activeIndex >= 0 ? `Step ${activeIndex + 1} of ${stages.length}` : 'Extraction status';

  return (
    <section className={`extraction-progress extraction-progress-${job.state}`} aria-label="Extraction progress">
      {stages.length ? (
        <ol className="extraction-stepper" aria-label="Processing stages">
          {stages.map((stage, index) => {
            const status = stopped && stage.status === 'active' ? 'stopped' : stage.status;
            return (
              <li key={stage.id} className={`extraction-step extraction-step-${status}`} aria-current={working && stage.id === progress?.stage ? 'step' : undefined}>
                <span className="extraction-step-number" aria-hidden="true">{status === 'complete' ? <Icon name="check" size={14} /> : status === 'stopped' ? <Icon name="close" size={13} /> : index + 1}</span>
                <span>{stage.label}<span className="sr-only"> — {status}</span></span>
              </li>
            );
          })}
        </ol>
      ) : null}

      <div className="extraction-status-heading">
        <span className="extraction-status-icon"><Icon name={successful ? 'check' : stopped ? 'info' : progress?.stage === 'segmentation' ? 'cohort' : progress?.stage === 'coordinates' ? 'overview' : progress?.stage === 'validation' ? 'check' : 'features'} size={24} /></span>
        <div aria-live="polite">
          <small>{stepLabel}</small>
          <h3>{progressTitle(job)}</h3>
          <p>{progressDescription(job)}</p>
        </div>
      </div>

      {successful ? (
        <div className="extraction-success-summary">
          <Icon name="check" size={18} />
          <span>{job.result?.completedSlides !== undefined ? `${job.result.completedSlides.toLocaleString()} slides validated` : 'Output validation complete'}</span>
        </div>
      ) : (
        <div className="extraction-stage-progress">
          <div className="extraction-progress-caption">
            <span>{validatedCoverage ? 'Validated outputs' : stopped ? `Last reported ${scope === 'batch' ? 'batch' : 'stage'} progress` : scopeName}</span>
            <strong>{percent !== null ? `${percent.toLocaleString(undefined, { maximumFractionDigits: 1 })}%` : working ? 'In progress' : 'Stopped'}</strong>
          </div>
          <div
            className={`extraction-progress-track ${percent === null && working ? 'is-indeterminate' : ''}`}
            role="progressbar"
            aria-label={`${scopeName} progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent ?? undefined}
            aria-valuetext={counts ?? (working ? 'Progress estimate not yet available' : 'Extraction stopped')}
          >
            <span style={percent !== null ? { width: `${percent}%` } : undefined} />
          </div>
          <div className="extraction-progress-explanation">
            <strong>{counts ?? (working ? 'Waiting for the next progress update…' : 'No further progress updates')}</strong>
            {scope === 'batch' ? <small>This batch only; other slides or stages may follow.</small> : progress?.stage === 'validation' ? <small>Each slide needs usable outputs before the extraction can complete.</small> : <small>Stage counts can include skipped slides. Outputs are validated before completion.</small>}
          </div>
        </div>
      )}

      <dl className="extraction-time-stats">
        <div>
          <dt>Run time</dt>
          <dd>{elapsed ?? 'Not reported'}</dd>
          <small>{stageElapsed && !successful ? `${scope === 'batch' ? 'This batch' : 'This stage'}: ${stageElapsed}` : 'This extraction'}</small>
        </div>
        <div>
          <dt>{scope === 'batch' ? 'Batch time remaining' : 'Stage time remaining'}</dt>
          <dd>{eta ? `~${eta}` : successful ? 'Complete' : working ? 'Estimating…' : '—'}</dd>
          <small>{successful ? 'Outputs validated' : stopped ? 'Extraction stopped' : scope ? 'Estimate for this step only' : 'Available after processing begins'}</small>
        </div>
        <div>
          <dt>Processing rate</dt>
          <dd className="extraction-rate">{rate ?? '—'}</dd>
          <small>{working ? scope ? `${scopeName.toLowerCase()} only` : 'Available after processing begins' : 'Final run status'}</small>
        </div>
      </dl>

      {progress?.currentSlide ? (
        <div className="extraction-current-slide"><Icon name="dataset" size={17} /><span>{working ? 'Current slide' : 'Last reported slide'}</span><strong className="mono">{progress.currentSlide}</strong></div>
      ) : null}
      {progress?.warnings?.length ? (
        <div className="extraction-progress-warnings" aria-label="Processing notices">
          {progress.warnings.map((warning, index) => <p key={`${index}-${warning}`}><Icon name="info" size={15} />{warning}</p>)}
        </div>
      ) : null}
    </section>
  );
}
