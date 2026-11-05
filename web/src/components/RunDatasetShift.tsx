import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import type { ModelEvaluation } from '../api/predictors';
import { shift, type ShiftGroupMetrics, type ShiftReport, type ShiftSlide } from '../api/shift';
import { formatStatistic } from '../api/statistics';
import { downloadJSON } from '../lib/download';
import { useReadOnly } from '../lib/readOnly';
import { ErrorNotice } from './ui';
import './ApplyRunDetail.css';

const ACTIVE = new Set(['queued', 'running']);
/** Codes that read as a note about this run rather than as an error. */
const EXPECTED = new Set(['SHIFT_INPUTS_UNAVAILABLE', 'PERFORMANCE_REQUIRES_LABELS', 'SHIFT_NOT_COMPLETED']);

const percent = (value: number | null | undefined) => (typeof value === 'number' ? `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%` : '—');

/** Both sets on the development slides' first two principal components; beyond-range slides ringed. */
export function ShiftPlane({ slides, label }: { slides: ShiftSlide[]; label: string }) {
  const width = 640, height = 360, pad = 24;
  const xs = slides.map((row) => row.x), ys = slides.map((row) => row.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (value: number) => pad + ((value - minX) / (maxX - minX || 1)) * (width - 2 * pad);
  const sy = (value: number) => height - pad - ((value - minY) / (maxY - minY || 1)) * (height - 2 * pad);
  const development = slides.filter((row) => row.set === 'development');
  const cohort = slides.filter((row) => row.set === 'cohort');
  return <figure className="run-figure shift-plane">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} preserveAspectRatio="xMidYMid meet">
      <rect x="0" y="0" width={width} height={height} className="shift-plane-frame" />
      {development.map((row) => <circle key={row.slideId} cx={sx(row.x)} cy={sy(row.y)} r={2.2} className="shift-point-development" />)}
      {cohort.map((row) => <circle key={row.slideId} cx={sx(row.x)} cy={sy(row.y)} r={row.beyond ? 3.4 : 2.6} className={row.beyond ? 'shift-point-beyond' : 'shift-point-cohort'} />)}
    </svg>
    <figcaption className="shift-legend">
      <span><i className="shift-swatch shift-point-development" />Development slides</span>
      <span><i className="shift-swatch shift-point-cohort" />Cohort slides</span>
      <span><i className="shift-swatch shift-point-beyond" />Beyond the development range</span>
    </figcaption>
  </figure>;
}

function GroupRows({ rows }: { rows: ShiftGroupMetrics[] }) {
  return <>{rows.map((row) => <tr key={row.group}>
    <th scope="row">{row.group}{row.percentileRange ? <small>Percentile {percent(row.percentileRange[0])}–{percent(row.percentileRange[1])}</small> : null}</th>
    <td>{row.count.toLocaleString()}</td>
    <td>{row.available ? formatStatistic(row.auroc) : '—'}</td>
    <td>{row.available ? formatStatistic(row.balancedAccuracy) : '—'}</td>
    <td>{row.available ? formatStatistic(row.accuracy) : row.reason ?? '—'}</td>
  </tr>)}</>;
}

function Report({ project, record, report, unit, referenceId }: {
  project: string; record: ModelEvaluation; report: ShiftReport; unit: 'selected' | 'slide' | 'patient'; referenceId: string | null;
}) {
  const summary = report.summary;
  const performance = useQuery({
    queryKey: ['run-shift-performance', project, record.id, unit, referenceId],
    queryFn: ({ signal }) => shift.performance(project, record.id, unit, referenceId, signal),
    staleTime: 300000,
  });
  const note = performance.error instanceof ApiError && EXPECTED.has(performance.error.code ?? '') ? performance.error.message : null;
  const domain = summary.domainAuroc;
  return <>
    {summary.verdictText ? <p className={`callout shift-verdict shift-verdict-${summary.verdict}`}>{summary.verdictText}</p> : null}
    <div className="chain-metrics">
      <div><strong>{formatStatistic(domain.value, { lower: domain.lower, upper: domain.upper })}</strong><span>Domain AUROC (95% CI)</span></div>
      <div><strong>{percent(summary.beyond.fraction)}</strong><span>Cohort slides beyond the development range ({summary.beyond.count.toLocaleString()} of {summary.cohortSlides.toLocaleString()})</span></div>
      <div><strong>{summary.developmentSlides.toLocaleString()}</strong><span>Development slides</span></div>
      <div><strong>{summary.cohortSlides.toLocaleString()}</strong><span>Cohort slides</span></div>
    </div>
    <p className="muted">Domain AUROC: how well a cross-validated linear classifier tells cohort slides from development slides by their mean patch features; 0.5 means indistinguishable. A slide is beyond the development range when its distance to its nearest development slides exceeds what 99% of development slides show among themselves.</p>
    <ShiftPlane slides={report.slides} label={`Development and cohort slides on the development data's first two principal components; ${summary.beyond.count} cohort slides beyond the development range.`} />
    <h4>Performance by atypicality</h4>
    <ErrorNotice error={note ? null : performance.error} />
    {note ? <p className="muted" role="status">{note}</p> : null}
    {performance.data ? <>
      <div className="table-wrap"><table className="chain-table shift-performance-table"><thead><tr><th scope="col">Group</th><th scope="col">Units</th><th scope="col">AUROC</th><th scope="col">Balanced accuracy</th><th scope="col">Accuracy</th></tr></thead>
        <tbody><GroupRows rows={performance.data.thirds} /><GroupRows rows={performance.data.range} /></tbody></table></div>
      <p className="muted">Units ordered by their slides&rsquo; mean atypicality percentile and cut in thirds. Descriptive: small groups give unstable estimates, and shift explains a drop rather than proving it.</p>
    </> : null}
  </>;
}

/**
 * Dataset shift: this run's cohort against the slides its predictor was developed on, in
 * feature space. The analysis is shared by every run with the same development batch and
 * cohort features, and its job never reads a label.
 */
export default function RunDatasetShift({ project, record, unit, referenceId }: {
  project: string; record: ModelEvaluation; unit: 'selected' | 'slide' | 'patient'; referenceId: string | null;
}) {
  const client = useQueryClient();
  const publicDemo = useReadOnly();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const status = useQuery({
    queryKey: ['run-shift', project, record.id],
    queryFn: ({ signal }) => shift.status(project, record.id, signal),
    refetchInterval: (query) => (ACTIVE.has(query.state.data?.execution?.status ?? '') ? 4000 : false),
  });
  const data = status.data;
  const state = data?.execution?.status ?? 'not_started';
  const unavailable = status.error instanceof ApiError && EXPECTED.has(status.error.code ?? '') ? status.error.message : null;
  async function analyze() {
    setBusy(true); setError(null);
    try {
      const preview = await shift.preview(project, record.id);
      if (!preview.canRun) throw new Error(preview.findings.find((row) => row.severity === 'error')?.message ?? 'This analysis cannot run.');
      await shift.start(project, record.id, preview.previewHash, `shift:${crypto.randomUUID()}`);
      await client.invalidateQueries({ queryKey: ['run-shift', project, record.id] });
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The analysis could not start.'));
    } finally { setBusy(false); }
  }
  const progress = data?.execution?.progress;
  return <section className="run-section" aria-labelledby={`shift-${record.id}`}>
    <div className="run-section-heading"><h3 id={`shift-${record.id}`}>Dataset shift</h3>
      {data?.report ? <button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${record.manifest.name} dataset shift.json`, data.report)}>Shift report (JSON)</button> : null}</div>
    <p className="muted">How far this cohort sits from the slides its predictor was developed on, in the feature space the model reads. A shifted cohort is one where its scores may not transfer.</p>
    <ErrorNotice error={unavailable ? null : status.error ?? error} />
    {unavailable ? <p className="muted" role="status">{unavailable}</p> : null}
    {data?.findings.filter((row) => row.severity === 'warning').map((row) => <p key={row.code} className="muted">{row.message}</p>)}
    {status.isPending ? <p role="status">Checking for an analysis…</p> : null}
    {data && !data.report && ACTIVE.has(state) ? <p role="status">Analyzing slides{progress?.total ? `: ${(progress.completed ?? 0).toLocaleString()} of ${progress.total.toLocaleString()} read` : '…'}</p> : null}
    {data && !data.report && !ACTIVE.has(state) ? publicDemo ? <p className="muted">No dataset shift analysis for this run.</p>
      : <div className="inline-actions">
        {state !== 'not_started' ? <p className="muted">The last analysis {state === 'failed' ? `failed: ${data.execution?.error ?? 'see its task'}` : `was ${state}`}.</p> : null}
        <button type="button" className="btn btn-secondary" disabled={busy || !data.canRun} onClick={() => void analyze()}>{busy ? 'Starting…' : state === 'not_started' ? 'Analyze dataset shift' : 'Run the analysis again'}</button>
        {!data.canRun ? <p className="muted">{data.findings.find((row) => row.severity === 'error')?.message}</p> : null}
      </div> : null}
    {data?.report ? <Report project={project} record={record} report={data.report} unit={unit} referenceId={referenceId} /> : null}
  </section>;
}
