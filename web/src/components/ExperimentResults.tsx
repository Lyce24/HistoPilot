import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { assessesEveryUnit, designLabel, experimentResults, foldLabel, lowerIsBetter, type ArmContrast, type BatchResult, type ConfigurationResult, type ExperimentResults as Results, type MetricStats, type ResultMetric, type SeedResult } from '../api/experimentResults';
import type { TrainingRecipe } from '../api/development';
import { experimentStage, type ModelExperiment } from '../api/experiments';
import type { Finding, ProtocolSpec } from '../api/scientific';
import { download, downloadJSON } from '../lib/download';
import { batchModel, csv, fixed, hasResults, interval, leadingBatch, meanSd, metricLabel, metricPhrase, metricShortLabel, niceTicks, orient, orientedPairs, pValue, tickDigits, percent, range, reportedConfiguration, resultRows, shade, signed, signedInterval, verdict, verdictLabel, type OrientedComparison } from '../lib/experimentResults';
import { inputModeLabel } from './ClinicalInputFields';
import { modelLabel } from '../lib/modelCapabilities';
import DevelopmentExecution, { OOFPredictionDownloads } from './DevelopmentExecution';
import ApplyConfiguration from './ApplyConfiguration';
import { EmptyState, ErrorNotice, Icon } from './ui';
import './ExperimentResults.css';

type DesignStrategy = NonNullable<Results['design']>['strategy'];
/** One assessment run of the design, as its runs and fold tables name it. */
const planTitle = (strategy?: DesignStrategy) => strategy === 'held_out' ? 'Held-out assessment' : strategy === 'leave_one_domain_out' ? 'Held-out site' : 'Test fold';
/** The design's assessment runs, in words: test folds, held-out sites, or one held-out run per seed. */
const planRuns = (strategy?: DesignStrategy) => strategy === 'held_out' ? 'held-out runs' : strategy === 'leave_one_domain_out' ? 'held-out sites' : 'test folds';

/** Batch identity: one validated categorical slot per batch, in experiment order. */
const seriesVar = (index: number) => `var(--results-series-${(index % 8) + 1})`;
const pickable: ResultMetric[] = ['auroc', 'auprc', 'balancedAccuracy', 'macroF1', 'accuracy'];

/**
 * The Results view of an experiment. It answers three questions in order: how well did
 * each model do (seed-averaged OOF with its interval, and the fold spread), is one model
 * clearly better (paired differences on the same draws and folds), and how robust is a
 * model (every training seed, every test fold, every class).
 */
export default function ExperimentResults({ project, record }: { project: string; record: ModelExperiment }) {
  const stage = experimentStage(record);
  const query = useQuery({
    queryKey: ['experiment-results', project, record.id],
    queryFn: () => experimentResults.get(project, record.id),
    // Seeds complete one at a time; the summary only changes when a fold or seed finishes.
    refetchInterval: stage === 'running' ? 15000 : false,
    refetchIntervalInBackground: false,
  });
  const [metric, setMetric] = useState<ResultMetric>('auroc');
  const [selected, setSelected] = useState<string | null>(null);
  const data = query.data;
  if (!data) {
    return <div className="exp-results">
      <ErrorNotice error={query.error} />
      {query.isError ? <button type="button" className="btn btn-secondary btn-small" onClick={() => void query.refetch()}>Retry results</button> : <p role="status" className="muted">Loading results…</p>}
    </div>;
  }
  const shown = data.batches.filter(hasResults);
  if (!shown.length) {
    return <div className="exp-results">
      <EmptyState icon="evaluation" title="No results yet" description={stage === 'running'
        ? `${planTitle(data.design?.strategy)} results appear as each ${data.design?.strategy === 'held_out' ? 'run' : data.design?.strategy === 'leave_one_domain_out' ? 'site' : 'fold'} finishes. Out-of-fold results for a training seed appear once all of its runs are done.`
        : 'This experiment has no completed runs. Runs shows what happened to its training.'} />
    </div>;
  }
  const batch = shown.find((item) => item.batchId === selected) ?? shown[0];
  const colors = new Map(data.batches.map((item, index) => [item.batchId, seriesVar(index)]));
  const task = data.target?.task;
  return <div className={`exp-results${query.isFetching ? ' is-refreshing' : ''}`}>
    <ErrorNotice error={query.isError ? query.error : null} />
    <ResultsIntro results={data} shown={shown} running={stage === 'running'} split={record.setupDesign?.trainingSplit} />
    <MetricPicker value={metric} onChange={setMetric} task={task} />
    <Notes findings={data.findings} batches={data.batches} />
    <section className="exp-section" aria-labelledby="exp-overview-title">
      <header><h2 id="exp-overview-title">{shown.length > 1 ? 'Model comparison' : 'Headline results'}</h2>
        <p>{shown.length > 1 ? 'The validation-selected configuration of each batch' : 'The validation-selected configuration'}{shown.some((item) => item.comparison) ? ', or the reference of a controlled comparison' : ''}. Out-of-fold (OOF) values are the mean ± SD across training seeds, with the 95% interval of that mean.</p></header>
      <Takeaway results={data} shown={shown} metric={metric} task={task} />
      <OverviewTable batches={shown} metric={metric} task={task} colors={colors} onSelect={setSelected} strategy={data.design?.strategy} />
      <SpreadChart batches={shown} metric={metric} task={task} colors={colors} unit={data.target?.unit ?? 'slide'} strategy={data.design?.strategy} />
      {shown.length > 1 ? <PairedTable results={data} shown={shown} metric={metric} task={task} /> : null}
    </section>
    {shown.filter((item) => item.comparison).map((item) => <ArmComparison key={item.batchId} results={data} batch={item} task={task} multiple={shown.length > 1} />)}
    <section className="exp-section" aria-labelledby="exp-detail-title">
      <header className="exp-detail-header">
        <div><h2 id="exp-detail-title">Seeds, {data.design?.strategy === 'held_out' ? 'runs' : data.design?.strategy === 'leave_one_domain_out' ? 'sites' : 'folds'} and classes</h2><p>Every training seed and every {planTitle(data.design?.strategy).toLowerCase()} of one batch, so you can see how much a single run can move.</p></div>
        {shown.length > 1 ? <BatchPicker batches={shown} value={batch.batchId} colors={colors} onChange={setSelected} /> : null}
      </header>
      <BatchDetail key={batch.batchId} project={project} record={record} results={data} batch={batch} metric={metric} task={task} />
    </section>
    <FileActions project={project} record={record} results={data} batch={batch} />
  </div>;
}

function ResultsIntro({ results, shown, running, split }: { results: Results; shown: BatchResult[]; running: boolean; split?: ProtocolSpec['split'] }) {
  const design = results.design;
  const seedCounts = [...new Set(shown.map((batch) => reportedConfiguration(batch)?.plannedSeedCount ?? 0))].sort((a, b) => a - b);
  const seeds = seedCounts.length === 1 ? `${seedCounts[0]} training seed${seedCounts[0] === 1 ? '' : 's'}` : `${seedCounts[0]}–${seedCounts.at(-1)} training seeds`;
  const classes = results.target?.classes ?? [];
  const unit = results.target?.unit ?? 'slide';
  const partial = shown.some((batch) => !reportedConfiguration(batch)?.complete);
  return <div className="exp-intro">
    {design ? <p className="exp-design"><strong>{designLabel(design, split)}</strong><span>{design.splitSeeds.length} split seed{design.splitSeeds.length === 1 ? '' : 's'}</span><span>{seeds}</span>{design.slideCount ? <span>{design.slideCount.toLocaleString()} slides</span> : null}<span>{classes.length} classes ({classes.join(', ')})</span><span>{unit === 'patient' ? 'Patient-level scoring' : 'Slide-level scoring'}</span></p> : null}
    {partial ? <p className="callout exp-partial" role="status"><Icon name="info" size={16} />{running ? 'Partial results. ' : 'Some training seeds are incomplete. '}A training seed joins the averages once all of its runs finish; {planRuns(design?.strategy)} appear as they finish.</p> : null}
    <details className="exp-help"><summary>How to read these results</summary>
      <dl>
        <div><dt>OOF (out-of-fold)</dt><dd>{design && !assessesEveryUnit(design, split) ? `Every assessed ${unit} is scored once, by the model that never trained on it; ${design.strategy === 'held_out' ? 'only the held-out set is assessed' : 'only the held-out sites are assessed'}.` : `Every ${unit} is scored once, by the fold model that never trained on it.`} The metric is computed over all of them pooled, so it is the most stable single estimate.</dd></div>
        {design?.strategy === 'held_out' ? <div><dt>Held-out assessment</dt><dd>One run on the held-out set, assessed once per training seed. It is one draw of the training set, so its value moves with the units it holds.</dd></div>
          : design?.strategy === 'leave_one_domain_out' ? <div><dt>Held-out site</dt><dd>One run’s held-out site or cohort. Sites differ in size and case mix, so a site far below the rest points to a shift between sites, not only to hard cases.</dd></div>
          : <div><dt>Test fold</dt><dd>One run’s held-out fold ({design ? `one of ${design.folds} folds` : 'one of the folds'}). Fold values spread more because each fold is small; a fold far below the rest points to hard cases or a small validation set.</dd></div>}
        <div><dt>Mean ± SD</dt><dd>Across training seeds: the same {design?.strategy === 'held_out' ? 'split' : planRuns(design?.strategy)} trained again with different initialisation and sampling. A large SD means a single run is not representative.</dd></div>
        <div><dt>95% interval</dt><dd>{results.policy.resamples.toLocaleString()} bootstrap resamples of the {results.design?.resamplingUnit ?? unit}s (seed {results.policy.seed}); each resample scores every seed and takes their mean. It covers sampling of {results.design?.resamplingUnit ?? unit}s, not retraining or configuration choice.</dd></div>
        <div><dt>Seed ensemble</dt><dd>The seeds’ predictions averaged per {unit}, then scored: what combining the seeds would give.</dd></div>
        <div><dt>Paired difference</dt><dd>Two batches share folds and {unit}s, so they are compared on the same resamples (OOF) and on the same folds. “No clear difference” means the 95% interval includes zero.</dd></div>
      </dl>
    </details>
  </div>;
}

/** One plain sentence: who leads on the picked metric, and whether the lead is clear. */
function Takeaway({ results, shown, metric, task }: { results: Results; shown: BatchResult[]; metric: ResultMetric; task?: string }) {
  const label = metricPhrase(metric, task);
  if (shown.length === 1) {
    const configuration = reportedConfiguration(shown[0])!;
    const ci = configuration.intervals.seedAverage?.available ? configuration.intervals.seedAverage.intervals?.[metric] : undefined;
    const folds = configuration.foldAverage[metric];
    return <p className="exp-takeaway"><strong>{shown[0].name}</strong>: {label} {meanSd(configuration.seedAverage[metric])} {configuration.seedCount === 1 ? 'from 1 training seed' : `across ${configuration.seedCount} training seeds`}{ci ? ` (95% interval ${interval(ci)})` : ''}{folds && folds.n > 1 ? `; its ${planRuns(results.design?.strategy)} ranged ${range(folds)}` : ''}.</p>;
  }
  const ranked = shown.map((batch) => ({ batch, stats: reportedConfiguration(batch)?.seedAverage[metric] ?? null }))
    .filter((row): row is { batch: BatchResult; stats: MetricStats } => Boolean(row.stats))
    .sort((a, b) => lowerIsBetter(metric) ? a.stats.mean - b.stats.mean : b.stats.mean - a.stats.mean);
  if (ranked.length < 2) return null;
  const [first, second] = ranked;
  const pair = results.comparisons.find((row) => new Set([row.leftBatchId, row.rightBatchId]).size === 2
    && [row.leftBatchId, row.rightBatchId].includes(first.batch.batchId) && [row.leftBatchId, row.rightBatchId].includes(second.batch.batchId));
  const value = pair?.available ? orient(pair, second.batch.batchId).metric(metric) : null;
  const result = verdict(value?.interval);
  return <p className="exp-takeaway">
    <strong>{first.batch.name}</strong> has the highest {label} ({meanSd(first.stats)}){ranked.length > 2 ? `, ahead of ${second.batch.name}` : ''}.{' '}
    {value && value.difference !== null ? <>Against {second.batch.name} the difference is {signed(value.difference)}{value.interval ? ` (95% interval ${signedInterval(value.interval)})` : ''}: {result === 'unclear' ? <strong>not a clear difference</strong> : result === 'unavailable' ? 'no interval is available' : <strong>a clear difference</strong>}{value.folds.n ? `, better in ${value.folds.better} of ${value.folds.n} shared folds` : ''}.</> : null}
  </p>;
}

/** Arrow keys, Home and End move the choice within a radio group, like the view tabs. */
function radioKeys(event: KeyboardEvent<HTMLDivElement>) {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
  const options = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button[role="radio"]')];
  const index = options.indexOf(event.target as HTMLButtonElement);
  if (index < 0) return;
  event.preventDefault();
  const step = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? options.length - 1 : (index + step + options.length) % options.length;
  options[next]?.focus(); options[next]?.click();
}

function MetricPicker({ value, onChange, task }: { value: ResultMetric; onChange: (metric: ResultMetric) => void; task?: string }) {
  return <div className="exp-controls" role="radiogroup" aria-label="Metric shown in charts and fold tables" onKeyDown={radioKeys}>
    <span className="exp-controls-label">Metric</span>
    {pickable.map((metric) => <button type="button" role="radio" aria-checked={metric === value} tabIndex={metric === value ? 0 : -1} key={metric} className={metric === value ? 'is-selected' : undefined} onClick={() => onChange(metric)}>{metricLabel(metric, task)}</button>)}
  </div>;
}

function Notes({ findings, batches }: { findings: (Finding & { batchId?: string })[]; batches: BatchResult[] }) {
  const referenced = new Set(batches.filter((batch) => batch.selection?.source === 'reference').map((batch) => batch.batchId));
  // A declared comparison reports its reference arm by design, not for want of a validation choice.
  const relevant = findings.filter((item) => !(item.code === 'SELECTION_UNAVAILABLE' && item.batchId && referenced.has(item.batchId)));
  if (!relevant.length) return null;
  const ordered = [...relevant].sort((a, b) => (a.severity === 'warning' ? 0 : 1) - (b.severity === 'warning' ? 0 : 1));
  const known = new Set(batches.map((batch) => batch.batchId));
  return <section className="exp-notes" aria-label="Things to know about these results">
    <h3>Things to know</h3>
    <ul>{ordered.filter((item) => !item.batchId || known.has(item.batchId)).map((item, index) => <li key={`${item.code}-${item.batchId ?? ''}-${index}`} className={`exp-note exp-note-${item.severity}`}>
      <Icon name={item.severity === 'warning' ? 'alert' : 'info'} size={16} /><span>{item.message}</span>
    </li>)}</ul>
  </section>;
}

function BatchName({ batch, color, detail = true }: { batch: BatchResult; color?: string; detail?: boolean }) {
  const configuration = reportedConfiguration(batch);
  const configurations = batch.configurations.length;
  return <span className="exp-batch-name">
    <span className="exp-swatch" style={{ background: color }} aria-hidden="true" />
    <span><strong>{batch.name}</strong>{detail ? <small>{batchModel(batch)}{configurations > 1 && configuration ? ` · configuration ${configuration.number} of ${configurations}${batch.selection?.source === 'validation' ? ', chosen on validation' : batch.selection?.source === 'reference' ? ', the comparison’s reference' : ''}` : ''}</small> : null}</span>
  </span>;
}

function StatCell({ stats, ci, leading }: { stats?: MetricStats | null; ci?: { lower: number; upper: number } | null; leading?: boolean }) {
  if (!stats) return <td className="exp-empty">—</td>;
  return <td className={leading ? 'is-leading' : undefined}>
    <span className="exp-value"><strong>{fixed(stats.mean)}</strong>{stats.sd !== null && stats.n > 1 ? <span className="exp-sd"> ± {fixed(stats.sd)}</span> : null}</span>
    {ci ? <small>95% CI {interval(ci)}</small> : null}
    {leading ? <span className="sr-only"> (highest in this column)</span> : null}
  </td>;
}

function OverviewTable({ batches, metric, task, colors, onSelect, strategy }: { batches: BatchResult[]; metric: ResultMetric; task?: string; colors: Map<string, string>; onSelect: (id: string) => void; strategy?: DesignStrategy }) {
  const leaders = new Map(pickable.map((name) => [name, leadingBatch(batches, name)]));
  return <><div className="exp-table-wrap"><table className="exp-table exp-overview">
    <caption className="sr-only">OOF results by batch: mean ± SD across training seeds and the 95% interval of the mean.</caption>
    <thead><tr><th scope="col">Batch</th>{pickable.map((name) => <th scope="col" key={name} className={name === metric ? 'is-active' : undefined}>{metricLabel(name, task)}</th>)}<th scope="col">Evidence</th></tr></thead>
    <tbody>{batches.map((batch) => {
      const configuration = reportedConfiguration(batch)!;
      const cis = configuration.intervals.seedAverage?.available ? configuration.intervals.seedAverage.intervals : undefined;
      return <tr key={batch.batchId}>
        <th scope="row"><button type="button" className="exp-row-link" onClick={() => onSelect(batch.batchId)} title="Show this batch’s seeds and folds"><BatchName batch={batch} color={colors.get(batch.batchId)} /></button></th>
        {pickable.map((name) => <StatCell key={name} stats={configuration.seedAverage[name]} ci={cis?.[name]} leading={batches.length > 1 && leaders.get(name) === batch.batchId} />)}
        <td className="exp-evidence">{configuration.seedCount} of {configuration.plannedSeedCount} seed{configuration.plannedSeedCount === 1 ? '' : 's'}<small>{configuration.foldCount} of {configuration.plannedFoldCount} {planRuns(strategy)}</small></td>
      </tr>;
    })}</tbody>
  </table></div>
  {batches.length > 1 ? <p className="exp-footnote">Underlined: the highest seed mean in a column. Whether a gap is real is shown under Paired differences.</p> : null}
  </>;
}

interface Mark { value: number; label: string; kind: 'fold' | 'seed' | 'mean' }
/** Chart labels stay inside their margin; the full name is in the tooltip and table view. */
const short = (text: string, limit: number) => text.length > limit ? `${text.slice(0, limit - 1)}…` : text;

function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    // Layout width: a page-entrance transform must not shrink the chart.
    const measure = () => { const next = element.clientWidth; if (next > 0) setWidth(Math.round(next)); };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/**
 * One row per batch on a shared axis: test folds (small), each seed's OOF (large) and
 * the seed mean with its 95% interval (ink tick and bar). Rows are direct-labeled, so
 * colour is never the only identity; the table view carries every value.
 */
function SpreadChart({ batches, metric, task, colors, unit, strategy }: { batches: BatchResult[]; metric: ResultMetric; task?: string; colors: Map<string, string>; unit: string; strategy?: DesignStrategy }) {
  const id = useId();
  const [box, measured] = useWidth<HTMLDivElement>(720);
  const [hover, setHover] = useState<{ x: number; y: number; mark: Mark; batch: string } | null>(null);
  const rows = batches.map((batch) => {
    const configuration = reportedConfiguration(batch)!;
    const seeds = configuration.splitSeeds.flatMap((split) => split.seeds.map((seed) => ({ seed, split: split.splitSeed, splits: configuration.splitSeeds.length })));
    const folds: Mark[] = seeds.flatMap(({ seed, split, splits }) => seed.folds.filter((fold) => typeof fold.metrics?.[metric] === 'number').map((fold) => ({ kind: 'fold' as const, value: fold.metrics![metric]!, label: `Seed ${seed.trainingSeed}${splits > 1 ? ` · split ${split}` : ''} · ${foldLabel(fold, strategy)}${fold.testCount ? ` · ${fold.testCount} ${unit}s` : ''}` })));
    const oof: Mark[] = seeds.filter(({ seed }) => typeof seed.oof?.[metric] === 'number').map(({ seed, split, splits }) => ({ kind: 'seed' as const, value: seed.oof![metric]!, label: `Seed ${seed.trainingSeed}${splits > 1 ? ` · split ${split}` : ''} · OOF` }));
    const mean = configuration.seedAverage[metric];
    const ci = configuration.intervals.seedAverage?.available ? configuration.intervals.seedAverage.intervals?.[metric] ?? null : null;
    return { batch, folds, oof, mean, ci };
  });
  const values = rows.flatMap((row) => [...row.folds.map((mark) => mark.value), ...row.oof.map((mark) => mark.value), ...(row.ci ? [row.ci.lower, row.ci.upper] : [])]);
  if (!values.length) return null;
  let low = Math.min(...values), high = Math.max(...values);
  const pad = Math.max((high - low) * 0.08, 0.005);
  low -= pad; high += pad;
  if (metric !== 'loss') { low = Math.max(0, low); high = Math.min(1, high); }
  const ticks = niceTicks(low, high, measured < 480 ? 3 : 5);
  const digits = tickDigits(ticks);
  const compact = measured < 560;
  const width = Math.max(300, measured), left = compact ? 12 : 168, right = 18, rowHeight = compact ? 70 : 58, top = 8, bottom = 38;
  const labelHeight = compact ? 20 : 0;
  const height = top + rows.length * rowHeight + bottom;
  const x = (value: number) => left + ((value - low) / (high - low)) * (width - left - right);
  const lanes = (index: number) => ({ base: top + index * rowHeight + labelHeight, fold: top + index * rowHeight + labelHeight + 14, seed: top + index * rowHeight + labelHeight + 34 });
  const show = (mark: Mark, batch: string, cx: number, cy: number) => setHover({ x: cx, y: cy, mark, batch });
  const markProps = (mark: Mark, batch: string, cx: number, cy: number) => ({
    tabIndex: 0, role: 'img' as const, 'aria-label': `${batch}, ${mark.label}: ${fixed(mark.value)}`,
    onPointerEnter: () => show(mark, batch, cx, cy), onFocus: () => show(mark, batch, cx, cy), onBlur: () => setHover(null),
  });
  return <figure className="exp-figure" aria-labelledby={`${id}-title`}>
    <figcaption id={`${id}-title`}>{metricLabel(metric, task)}: every {planTitle(strategy).toLowerCase()}, every seed, and the seed mean</figcaption>
    <ul className="exp-legend" aria-label="Chart marks">
      <li><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4" className="exp-legend-fold" /></svg>{planTitle(strategy)} (one run)</li>
      <li><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5.5" className="exp-legend-seed" /></svg>Seed OOF</li>
      <li><svg width="22" height="14" aria-hidden="true"><line x1="2" x2="20" y1="7" y2="7" className="exp-legend-ci" /><line x1="11" x2="11" y1="1" y2="13" className="exp-legend-mean" /></svg>Seed mean and 95% interval</li>
    </ul>
    <div className="exp-plot" ref={box} onPointerLeave={() => setHover(null)}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <desc id={`${id}-desc`}>{`${metricLabel(metric, task)} per batch. ${rows.map((row) => `${row.batch.name}: seed mean ${fixed(row.mean?.mean)}${row.ci ? `, 95% interval ${interval(row.ci)}` : ''}, ${planRuns(strategy)} ${range(reportedConfiguration(row.batch)?.foldAverage[metric])}`).join('. ')}. Exact values are in the table view.`}</desc>
        {ticks.map((tick) => <g key={tick} className="exp-grid"><line x1={x(tick)} x2={x(tick)} y1={top} y2={height - bottom + 4} /><text x={x(tick)} y={height - bottom + 18} textAnchor="middle">{tick.toFixed(digits)}</text></g>)}
        <text className="exp-axis-title" x={left + (width - left - right) / 2} y={height - 6} textAnchor="middle">{metricLabel(metric, task)}{lowerIsBetter(metric) ? ' (lower is better)' : ''}</text>
        {rows.map((row, index) => {
          const lane = lanes(index);
          const color = colors.get(row.batch.batchId);
          return <g key={row.batch.batchId} style={{ color }}>
            <text className="exp-row-label" x={compact ? left : left - 14} y={compact ? lane.base : lane.base + 28} textAnchor={compact ? 'start' : 'end'}>{short(row.batch.name, compact ? 40 : 20)}<title>{row.batch.name}</title></text>
            <line className="exp-lane" x1={left} x2={width - right} y1={lane.seed} y2={lane.seed} />
            {row.folds.map((mark, position) => {
              const cx = x(mark.value), cy = lane.fold + ((position % 3) - 1) * 4;
              return <circle key={`f${position}`} className="exp-mark-fold" cx={cx} cy={cy} r={4} {...markProps(mark, row.batch.name, cx, cy)} />;
            })}
            {row.ci ? <g className="exp-mark-ci"><line x1={x(row.ci.lower)} x2={x(row.ci.upper)} y1={lane.seed} y2={lane.seed} /><line x1={x(row.ci.lower)} x2={x(row.ci.lower)} y1={lane.seed - 5} y2={lane.seed + 5} /><line x1={x(row.ci.upper)} x2={x(row.ci.upper)} y1={lane.seed - 5} y2={lane.seed + 5} /></g> : null}
            {row.oof.map((mark, position) => {
              const cx = x(mark.value);
              return <circle key={`s${position}`} className="exp-mark-seed" cx={cx} cy={lane.seed} r={5.5} {...markProps(mark, row.batch.name, cx, lane.seed)} />;
            })}
            {row.mean ? (() => {
              const mark: Mark = { kind: 'mean', value: row.mean.mean, label: `Seed mean${row.ci ? ` · 95% interval ${interval(row.ci)}` : ''}` };
              const cx = x(row.mean.mean);
              return <g className="exp-mark-mean" {...markProps(mark, row.batch.name, cx, lane.seed)}><rect x={cx - 8} y={lane.seed - 12} width={16} height={24} fill="transparent" /><line x1={cx} x2={cx} y1={lane.seed - 10} y2={lane.seed + 10} /></g>;
            })() : null}
          </g>;
        })}
      </svg>
      {hover ? <div className="exp-tooltip" role="status" style={{ left: Math.min(Math.max(hover.x, 90), width - 90), top: hover.y }}><strong>{fixed(hover.mark.value)}</strong><span>{hover.batch}</span><small>{hover.mark.label}</small></div> : null}
    </div>
    <details className="exp-table-view"><summary>Table view</summary>
      <div className="exp-table-wrap"><table className="exp-table"><thead><tr><th scope="col">Batch</th><th scope="col">Seed mean</th><th scope="col">95% interval</th><th scope="col">Seed OOF values</th><th scope="col">{planRuns(strategy).replace(/^./, (letter) => letter.toUpperCase())}</th></tr></thead>
        <tbody>{rows.map((row) => <tr key={row.batch.batchId}><th scope="row">{row.batch.name}</th><td>{fixed(row.mean?.mean)}</td><td>{interval(row.ci)}</td><td>{row.oof.map((mark) => fixed(mark.value)).join(', ') || '—'}</td><td>{row.folds.map((mark) => fixed(mark.value)).join(', ') || '—'}</td></tr>)}</tbody></table></div>
    </details>
  </figure>;
}

function PairedTable({ results, shown, metric, task }: { results: Results; shown: BatchResult[]; metric: ResultMetric; task?: string }) {
  const names = new Map(results.batches.map((batch) => [batch.batchId, batch.name]));
  const visible = new Set(shown.map((batch) => batch.batchId));
  const pairs = orientedPairs(results).filter((pair) => visible.has(pair.otherId) && visible.has(pair.referenceId));
  if (!pairs.length) return null;
  return <div className="exp-paired">
    <h3>Paired differences</h3>
    <p className="muted">Each row compares the batch with the higher {metricPhrase(results.primaryMetric, task)} against the other, on the same {results.design?.resamplingUnit ?? 'slide'}s and folds: the difference of their OOF seed means, its 95% interval, and in how many shared folds the first batch did better.</p>
    <div className="exp-table-wrap"><table className="exp-table">
      <caption className="sr-only">Paired OOF differences between batches with 95% intervals and fold agreement.</caption>
      <thead><tr><th scope="col">Comparison</th>{pickable.map((name) => <th scope="col" key={name} className={name === metric ? 'is-active' : undefined}>{metricLabel(name, task)}</th>)}</tr></thead>
      <tbody>{pairs.map((pair) => <PairRow key={`${pair.otherId}-${pair.referenceId}`} pair={pair} names={names} metric={metric} />)}</tbody>
    </table></div>
    {pairs.some((pair) => pair.intervalReason) ? <p className="exp-footnote">{pairs.find((pair) => pair.intervalReason)?.intervalReason}</p> : null}
    <p className="exp-footnote">Intervals hold the trained models fixed, so they do not include the chance that retraining would reorder close models. Treat a clear difference in one metric as a lead to confirm on external data.</p>
  </div>;
}

const armModel = (row: Pick<ConfigurationResult, 'model' | 'inputMode'>) => row.inputMode === 'clinical' ? '—' : modelLabel(row.model ?? undefined) || row.model || '—';
const armInputs = (row: Pick<ConfigurationResult, 'inputMode'>) => inputModeLabel(row.inputMode as TrainingRecipe['inputMode']) ?? row.inputMode;
const armName = (row: Pick<ConfigurationResult, 'model' | 'inputMode'>) => row.inputMode === 'clinical' ? 'Clinical only' : `${armModel(row)} · ${armInputs(row)}`;
/** The service words a missing pairing for batches; say it for arms. */
const armReason = (reason?: string) => reason === 'Both batches need complete OOF results.' ? 'The arm and the reference both need complete out-of-fold results.' : reason ?? 'Not available yet.';

/**
 * A declared controlled comparison: every arm of one batch against its reference arm on
 * the primary metric, with paired intervals on shared draws, bootstrap p-values adjusted
 * by Holm for the planned contrasts, and fold-by-fold agreement.
 */
function ArmComparison({ results, batch, task, multiple }: { results: Results; batch: BatchResult; task?: string; multiple: boolean }) {
  const id = useId();
  const comparison = batch.comparison!;
  const metric = comparison.primaryMetric;
  const label = metricLabel(metric, task);
  const arms = [...batch.configurations].sort((a, b) => a.number - b.number);
  const reference = arms.find((row) => row.candidateId === comparison.referenceId);
  const unit = comparison.contrasts.find((row) => row.oofInterval?.unit)?.oofInterval?.unit ?? results.design?.resamplingUnit ?? results.target?.unit ?? 'slide';
  const partial = arms.some((row) => !row.complete);
  const noInterval = comparison.contrasts.find((row) => row.available && row.oofInterval && !row.oofInterval.available)?.oofInterval?.reason;
  return <section className="exp-section exp-comparison" aria-labelledby={`${id}-title`}>
    <header><h2 id={`${id}-title`}>Controlled comparison{multiple ? `: ${batch.name}` : ''}</h2>
      <p>Every arm trained on the same folds and training seeds and differs from the reference only in its model or inputs. Reference: configuration {comparison.referenceNumber}{reference ? ` (${armName(reference)})` : ''}. Primary metric: {label}.</p></header>
    {partial ? <p className="callout exp-partial" role="status"><Icon name="info" size={16} />Some arms have unfinished training seeds. Their values cover completed seeds only and will change as training finishes.</p> : null}
    <figure className="exp-figure">
      <figcaption>Arms · {label}</figcaption>
      <div className="exp-table-wrap"><table className="exp-table">
        <caption className="sr-only">Each arm’s out-of-fold {label}: mean ± SD across training seeds and the 95% interval of the mean.</caption>
        <thead><tr><th scope="col">Arm</th><th scope="col">Model</th><th scope="col">Inputs</th><th scope="col">{label}<small>seed mean ± SD</small></th><th scope="col">95% interval</th><th scope="col">Evidence</th></tr></thead>
        <tbody>{arms.map((row) => {
          const ci = row.intervals.seedAverage?.available ? row.intervals.seedAverage.intervals?.[metric] : undefined;
          return <tr key={row.candidateId} className={row.candidateId === comparison.referenceId ? 'is-current' : undefined}>
            <th scope="row">Configuration {row.number}{row.candidateId === comparison.referenceId ? <span className="exp-chip">Reference</span> : null}</th>
            <td>{armModel(row)}</td><td>{armInputs(row)}</td>
            <td><span className="exp-value"><strong>{meanSd(row.seedAverage[metric])}</strong></span></td>
            <td>{ci ? interval(ci) : '—'}</td>
            <td className="exp-evidence">{row.seedCount} of {row.plannedSeedCount} seed{row.plannedSeedCount === 1 ? '' : 's'}<small>{row.foldCount} of {row.plannedFoldCount} {planRuns(results.design?.strategy)}</small></td>
          </tr>;
        })}</tbody>
      </table></div>
    </figure>
    {comparison.contrasts.length ? <figure className="exp-figure">
      <figcaption>Each arm against the reference · {label}</figcaption>
      <p className="muted">Reference − arm: a positive difference means the reference scored higher. The interval and p-values come from the same resamples for both, so they compare the arms directly.</p>
      <div className="exp-table-wrap"><table className="exp-table exp-contrasts">
        <caption className="sr-only">Reference minus each arm on {label}, with paired 95% intervals, p-values, Holm-adjusted p-values and test-fold agreement.</caption>
        <thead><tr><th scope="col">Arm</th><th scope="col">Reference − arm</th><th scope="col">Paired 95% interval</th><th scope="col">p</th><th scope="col">Holm-adjusted p</th><th scope="col">{planRuns(results.design?.strategy).replace(/^./, (letter) => letter.toUpperCase())}<small>reference better</small></th></tr></thead>
        <tbody>{comparison.contrasts.map((row) => <ContrastRow key={row.armId} row={row} metric={metric} />)}</tbody>
      </table></div>
      <p className="exp-footnote">Differences use the same resampled {unit}s for every arm; Holm adjusts p for the number of planned contrasts.{noInterval ? ` ${noInterval}` : ''}</p>
      <p className="exp-footnote">Like every interval here, these hold the trained models fixed. A clear difference is a lead to confirm on an independent labeled cohort.</p>
    </figure> : null}
  </section>;
}

function ContrastRow({ row, metric }: { row: ArmContrast; metric: Exclude<ResultMetric, 'loss'> }) {
  const name = <th scope="row">Configuration {row.armNumber}<small>{armName(row)}</small></th>;
  if (!row.available) return <tr>{name}<td colSpan={5} className="exp-empty exp-wrap">{armReason(row.reason)}</td></tr>;
  const point = row.oof?.[metric];
  const ci = row.oofInterval?.available ? row.oofInterval.intervals?.[metric] ?? null : null;
  const result = verdict(ci);
  const folds = row.folds?.[metric];
  return <tr>
    {name}
    <td><span className="exp-value"><strong>{signed(point?.difference)}</strong></span>{point ? <small>{fixed(point.left)} vs {fixed(point.right)}</small> : null}</td>
    <td>{ci ? signedInterval(ci) : 'no interval'}{result !== 'unavailable' ? <span className={`exp-verdict exp-verdict-${result}`}>{result === 'higher' ? 'Reference better' : result === 'lower' ? 'Arm better' : 'No clear difference'}</span> : null}</td>
    <td>{pValue(row.pValue)}</td>
    <td><strong>{pValue(row.pValueHolm)}</strong></td>
    <td>{folds?.n ? `better in ${folds.better ?? 0} of ${folds.n} folds` : '—'}</td>
  </tr>;
}

function PairRow({ pair, names, metric }: { pair: OrientedComparison; names: Map<string, string>; metric: ResultMetric }) {
  return <tr>
    <th scope="row"><span className="exp-pair"><strong>{names.get(pair.otherId)}</strong><span className="muted"> vs </span><span>{names.get(pair.referenceId)}</span></span></th>
    {pickable.map((name) => {
      if (!pair.available) return <td key={name} className="exp-empty">{name === pickable[0] ? pair.reason ?? '—' : '—'}</td>;
      const value = pair.metric(name);
      const result = verdict(value.interval);
      return <td key={name} className={name === metric ? 'is-active' : undefined}>
        <span className="exp-value"><strong>{signed(value.difference)}</strong></span>
        <small>{value.interval ? signedInterval(value.interval) : 'no interval'}</small>
        <span className={`exp-verdict exp-verdict-${result}`}>{verdictLabel(result, name)}</span>
        {value.folds.n ? <small>better in {value.folds.better} of {value.folds.n} folds</small> : null}
      </td>;
    })}
  </tr>;
}

function BatchPicker({ batches, value, colors, onChange }: { batches: BatchResult[]; value: string; colors: Map<string, string>; onChange: (id: string) => void }) {
  return <div className="exp-controls exp-batch-picker" role="radiogroup" aria-label="Batch shown in detail" onKeyDown={radioKeys}>
    {batches.map((batch) => <button type="button" role="radio" aria-checked={batch.batchId === value} tabIndex={batch.batchId === value ? 0 : -1} key={batch.batchId} className={batch.batchId === value ? 'is-selected' : undefined} onClick={() => onChange(batch.batchId)}>
      <span className="exp-swatch" style={{ background: colors.get(batch.batchId) }} aria-hidden="true" />{batch.name}
    </button>)}
  </div>;
}

function BatchDetail({ project, record, results, batch, metric, task }: { project: string; record: ModelExperiment; results: Results; batch: BatchResult; metric: ResultMetric; task?: string }) {
  const [candidate, setCandidate] = useState(batch.selectedCandidateId);
  const configuration = batch.configurations.find((row) => row.candidateId === candidate) ?? reportedConfiguration(batch)!;
  const classes = results.target?.classes ?? [];
  return <div className="exp-detail">
    {batch.configurations.length > 1 ? <ConfigurationTable batch={batch} record={record} metric={metric} task={task} value={configuration.candidateId} onChange={setCandidate} /> : null}
    <SeedTable configuration={configuration} metric={metric} task={task} strategy={results.design?.strategy} />
    <ApplyConfiguration project={project} record={record} batchId={batch.batchId} candidateId={configuration.candidateId} number={configuration.number} />
    {configuration.splitSeeds.map((split) => <FoldTable key={split.splitSeed} split={split} configuration={configuration} metric={metric} task={task} unit={results.target?.unit ?? 'slide'} multiple={configuration.splitSeeds.length > 1} strategy={results.design?.strategy} />)}
    <ClassTable configuration={configuration} unit={results.target?.unit ?? 'slide'} binary={task === 'binary_classification'} />
    <ConfusionTable configuration={configuration} classes={classes} />
    <PredictionFiles project={project} batch={batch} configuration={configuration} splitUnit={results.design?.splitUnit} />
  </div>;
}

function ConfigurationTable({ batch, record, metric, task, value, onChange }: { batch: BatchResult; record: ModelExperiment; metric: ResultMetric; task?: string; value: string; onChange: (id: string) => void }) {
  const manifest = record.batches.find((item) => item.id === batch.batchId)?.manifest;
  const recipes = new Map(manifest?.configurations.map((item) => [item.id, item.recipe]) ?? []);
  const selectionMetric = batch.selection?.metric?.replace('validation_', '') ?? null;
  const arms = Boolean(batch.comparison);
  // Best validation first (lowest for loss); configurations without a score go last. Comparison arms keep their order.
  const rank = (row: ConfigurationResult) => arms ? 0 : typeof row.validationScore === 'number' ? (selectionMetric === 'loss' ? row.validationScore : -row.validationScore) : Infinity;
  const ordered = [...batch.configurations].sort((a, b) => rank(a) - rank(b) || a.number - b.number);
  return <figure className="exp-figure">
    <figcaption>{arms ? 'Comparison arms' : 'Configurations, ranked by validation'}</figcaption>
    <p className="muted">{arms ? 'Choose an arm to see its seeds, folds and classes below. Validation does not choose between arms; each is compared with the reference above.'
      : <>The batch reports the configuration with the best mean validation {selectionMetric ? selectionMetric.replace('auroc', 'AUROC') : 'score'} across folds and seeds. OOF values are shown for context only: choosing a configuration by OOF would overstate its performance.</>}</p>
    <div className="exp-table-wrap"><table className="exp-table"><thead><tr><th scope="col">Configuration</th><th scope="col">Settings</th><th scope="col">Validation {selectionMetric ?? 'score'}</th><th scope="col">OOF {metricShortLabel(metric)} (seed mean)</th><th scope="col"><span className="sr-only">Show</span></th></tr></thead>
      <tbody>{ordered.map((row) => {
        const recipe = recipes.get(row.candidateId);
        return <tr key={row.candidateId} className={row.candidateId === value ? 'is-current' : undefined}>
          <th scope="row">{row.number}{row.selected ? <span className="exp-chip">{arms ? 'Reference' : 'Selected'}</span> : null}</th>
          <td>{arms ? armName(row) : recipe ? `${modelLabel(recipe.model)} · LR ${recipe.learningRate} · WD ${recipe.weightDecay} · ${recipe.maxEpochs} epochs` : row.model ?? '—'}</td>
          <td>{fixed(row.validationScore)}</td>
          <td>{meanSd(row.seedAverage[metric])}</td>
          <td><button type="button" className="text-button" aria-pressed={row.candidateId === value} onClick={() => onChange(row.candidateId)}>{row.candidateId === value ? 'Shown below' : 'Show'}</button></td>
        </tr>;
      })}</tbody></table></div>
    <p className="exp-footnote">{metricLabel(metric, task)} is the metric picked above.</p>
  </figure>;
}

const seedLabel = (seed: SeedResult, multiple: boolean) => `Seed ${seed.trainingSeed}${multiple ? ` · split ${seed.splitSeed}` : ''}`;

function epochSummary(seeds: SeedResult[]) {
  const epochs = seeds.flatMap((seed) => seed.folds.map((fold) => fold.bestEpoch)).filter((value): value is number => typeof value === 'number').sort((a, b) => a - b);
  if (!epochs.length) return '—';
  const middle = epochs.length / 2;
  const median = epochs.length % 2 ? epochs[Math.floor(middle)] : (epochs[middle - 1] + epochs[middle]) / 2;
  return epochs.length === 1 ? String(median) : `${median} (${epochs[0]}–${epochs.at(-1)})`;
}

function SeedTable({ configuration, metric, task, strategy }: { configuration: ConfigurationResult; metric: ResultMetric; task?: string; strategy?: DesignStrategy }) {
  const multiple = configuration.splitSeeds.length > 1;
  const seeds = configuration.splitSeeds.flatMap((split) => split.seeds);
  const columns: ResultMetric[] = ['auroc', 'auprc', 'balancedAccuracy', 'macroF1', 'accuracy', 'loss'];
  const cis = configuration.intervals.seedAverage?.available ? configuration.intervals.seedAverage.intervals : undefined;
  const cell = (name: ResultMetric) => name === metric ? 'is-active' : undefined;
  return <figure className="exp-figure">
    <figcaption>Training seeds (OOF)</figcaption>
    <p className="muted">One row per training seed: the out-of-fold result over all {configuration.intervals.units ? `${configuration.intervals.units.toLocaleString()} ` : ''}{configuration.intervals.unit === 'patient' ? 'patients' : 'slides'}. Then their mean ± SD, the 95% interval of that mean, and the seed ensemble.</p>
    <div className="exp-table-wrap"><table className="exp-table exp-seeds">
      <thead><tr><th scope="col">Seed</th>{columns.map((name) => <th scope="col" key={name} className={cell(name)}>{metricLabel(name, task)}</th>)}</tr></thead>
      <tbody>
        {seeds.map((seed) => <tr key={`${seed.splitSeed}-${seed.trainingSeed}`}>
          <th scope="row">{seedLabel(seed, multiple)}</th>
          {seed.oof ? columns.map((name) => <td key={name} className={cell(name)}>{fixed(seed.oof?.[name])}</td>)
            : <td colSpan={columns.length} className="exp-empty exp-wrap">{seed.completedRuns} of {seed.totalRuns} {planRuns(strategy)} finished; OOF waits for all of them</td>}
        </tr>)}
      </tbody>
      <tbody className="exp-summary-rows">
        {configuration.seedCount > 1 ? <tr><th scope="row">Mean ± SD<small>{configuration.seedCount} seeds</small></th>{columns.map((name) => <td key={name} className={cell(name)}><strong>{meanSd(configuration.seedAverage[name])}</strong></td>)}</tr> : null}
        {cis ? <tr className="exp-interval-row"><th scope="row">95% interval<small>{configuration.intervals.resamples.toLocaleString()} {configuration.intervals.unit} resamples</small></th>{columns.map((name) => <td key={name} className={cell(name)}>{cis[name] ? interval(cis[name]) : '—'}</td>)}</tr> : null}
        {configuration.ensemble ? <tr><th scope="row">Seed ensemble<small>seeds’ mean prediction</small></th>{columns.map((name) => <td key={name} className={cell(name)}>{fixed(configuration.ensemble?.[name])}</td>)}</tr> : null}
      </tbody>
    </table></div>
    <p className="exp-footnote">{metricLabel(metric, task)} is highlighted. Loss is the mean cross-entropy; it is not resampled.{!cis && configuration.intervals.reason ? ` ${configuration.intervals.reason}` : ''}</p>
  </figure>;
}

function FoldTable({ split, configuration, metric, task, unit, multiple, strategy }: { split: ConfigurationResult['splitSeeds'][number]; configuration: ConfigurationResult; metric: ResultMetric; task?: string; unit: string; multiple: boolean; strategy?: DesignStrategy }) {
  const values = split.seeds.flatMap((seed) => seed.folds.map((fold) => fold.metrics?.[metric])).filter((value): value is number => typeof value === 'number');
  const low = Math.min(...values), high = Math.max(...values);
  const better = lowerIsBetter(metric);
  const tone = (value?: number | null) => typeof value === 'number' ? shade(better ? high - (value - low) : value, low, high) : undefined;
  const foldMeans = split.folds.map((fold) => {
    const scores = split.seeds.map((seed) => seed.folds.find((row) => row.splitPlanId === fold.splitPlanId)?.metrics?.[metric]).filter((value): value is number => typeof value === 'number');
    const mean = scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null;
    const sd = scores.length > 1 && mean !== null ? Math.sqrt(scores.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (scores.length - 1)) : null;
    return { mean, sd, n: scores.length };
  });
  return <figure className="exp-figure">
    <figcaption>{strategy === 'held_out' ? 'Held-out assessment' : strategy === 'leave_one_domain_out' ? 'Held-out sites' : 'Test folds'} · {metricLabel(metric, task)}{multiple ? ` · split seed ${split.splitSeed}` : ''}</figcaption>
    <p className="muted">{strategy === 'held_out' ? `Each cell is one run on the held-out set, scored by the checkpoint its validation set chose. Every seed assesses the same ${unit}s, so the spread across seeds comes from training alone.` : strategy === 'leave_one_domain_out' ? 'Each cell is one run: its held-out site, scored by the checkpoint that run’s validation set chose. Rows share the site across seeds, so a row that is low for every seed points at that site, not the training.' : `Each cell is one run: its held-out test fold, scored by the checkpoint that fold’s validation set chose. Rows share test ${unit}s across seeds, so a row that is low for every seed points at the data, not the training.`}</p>
    <div className="exp-table-wrap"><table className="exp-table exp-folds">
      <thead><tr><th scope="col">{planTitle(strategy)}</th>{split.seeds.map((seed) => <th scope="col" key={seed.trainingSeed}>Seed {seed.trainingSeed}</th>)}<th scope="col">Across seeds</th></tr></thead>
      <tbody>{split.folds.map((fold, index) => <tr key={fold.splitPlanId}>
        <th scope="row">{foldLabel(fold, strategy)}{fold.testCount ? <small>{fold.testCount.toLocaleString()} {unit}s</small> : null}</th>
        {split.seeds.map((seed) => {
          const run = seed.folds.find((row) => row.splitPlanId === fold.splitPlanId);
          const value = run?.metrics?.[metric];
          const background = tone(value);
          return <td key={seed.trainingSeed} className="exp-heat" style={background ? { background } : undefined}>
            {typeof value === 'number' ? <><strong>{fixed(value)}</strong>{run?.bestEpoch ? <small>epoch {run.bestEpoch}{run.epochsCompleted ? ` of ${run.epochsCompleted}` : ''}</small> : null}</> : <span className="muted">{run?.status && run.status !== 'completed' ? run.status : '—'}</span>}
          </td>;
        })}
        <td>{foldMeans[index].mean !== null ? <>{fixed(foldMeans[index].mean)}{foldMeans[index].sd !== null ? <span className="exp-sd"> ± {fixed(foldMeans[index].sd)}</span> : null}</> : '—'}</td>
      </tr>)}</tbody>
      <tbody className="exp-summary-rows">
        {split.folds.length > 1 ? <tr><th scope="row">{strategy === 'leave_one_domain_out' ? 'Site mean ± SD' : 'Fold mean ± SD'}</th>{split.seeds.map((seed) => <td key={seed.trainingSeed}>{meanSd(seed.foldStats[metric])}</td>)}<td><strong>{meanSd(configuration.foldAverage[metric])}</strong></td></tr> : null}
        <tr><th scope="row">Pooled OOF</th>{split.seeds.map((seed) => <td key={seed.trainingSeed}>{fixed(seed.oof?.[metric])}</td>)}<td><strong>{meanSd(configuration.seedAverage[metric])}</strong></td></tr>
        <tr className="exp-interval-row"><th scope="row">Checkpoint epoch<small>median (range)</small></th>{split.seeds.map((seed) => <td key={seed.trainingSeed}>{epochSummary([seed])}</td>)}<td>{epochSummary(split.seeds)}</td></tr>
      </tbody>
    </table></div>
    {values.length > 1 ? <p className="exp-footnote"><span className="exp-ramp" aria-hidden="true" />Shading runs from the {better ? 'highest' : 'lowest'} to the {better ? 'lowest' : 'highest'} {planTitle(strategy).toLowerCase()} value in this table ({fixed(low)}–{fixed(high)}); darker is better.{split.folds.length > 1 ? ` Pooled OOF can sit below the ${strategy === 'leave_one_domain_out' ? 'site' : 'fold'} mean: pooling mixes ${strategy === 'leave_one_domain_out' ? 'site' : 'fold'} models whose scores are on slightly different scales.` : ''}</p> : null}
  </figure>;
}

function ClassTable({ configuration, unit, binary }: { configuration: ConfigurationResult; unit: string; binary: boolean }) {
  const hasRanking = configuration.perClass.some((row) => row.auroc);
  return <figure className="exp-figure">
    <figcaption>Per class (OOF, mean ± SD across seeds)</figcaption>
    <p className="muted">Recall is the share of each true class the model calls correctly{binary ? ' (sensitivity for the positive class, specificity for the other)' : ''}. A class with low recall is folded into its neighbours even when AUROC looks good.</p>
    <div className="exp-table-wrap"><table className="exp-table exp-classes">
      <thead><tr><th scope="col">Class</th><th scope="col">Recall</th><th scope="col">Precision</th><th scope="col">F1</th>{hasRanking ? <th scope="col">AUROC (vs rest)</th> : null}</tr></thead>
      <tbody>{configuration.perClass.map((row) => {
        const recall = row.recall?.mean;
        const low = typeof recall === 'number' && recall < 0.5;
        return <tr key={row.label}>
          <th scope="row">{row.label}{row.support !== null ? <small>{row.support.toLocaleString()} {unit}s</small> : null}</th>
          <td className="exp-recall"><span className="exp-recall-value">{meanSd(row.recall)}</span><span className="exp-bar" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(1, recall ?? 0)) * 100}%` }} /></span>{low ? <span className="exp-flag"><Icon name="alert" size={13} />Low</span> : null}</td>
          <td>{meanSd(row.precision)}</td>
          <td>{meanSd(row.f1)}</td>
          {hasRanking ? <td>{meanSd(row.auroc)}</td> : null}
        </tr>;
      })}</tbody>
    </table></div>
  </figure>;
}

function ConfusionTable({ configuration, classes }: { configuration: ConfigurationResult; classes: string[] }) {
  const confusion = configuration.confusion;
  if (!confusion) return null;
  return <figure className="exp-figure">
    <figcaption>Confusion (OOF)</figcaption>
    <p className="muted">Rows are the true class, columns the predicted class: the share of each true class{confusion.seeds > 1 ? `, averaged over ${confusion.seeds} seeds` : ''}, with the mean count.</p>
    <div className="exp-table-wrap"><table className="exp-table exp-confusion">
      <thead><tr><th scope="col"><span className="sr-only">True class</span>True ↓ · Predicted →</th>{classes.map((label) => <th scope="col" key={label}>{label}</th>)}</tr></thead>
      <tbody>{classes.map((label, row) => <tr key={label}><th scope="row">{label}</th>{classes.map((predicted, column) => {
        const rate = confusion.rowRates[row]?.[column];
        const background = shade(rate, 0, 1);
        return <td key={predicted} className={`exp-heat${row === column ? ' is-diagonal' : ''}`} style={background ? { background } : undefined}>
          <strong>{percent(rate)}</strong><small>{fixed(confusion.meanCounts[row]?.[column], confusion.seeds > 1 ? 1 : 0)}</small>
        </td>;
      })}</tr>)}</tbody>
    </table></div>
  </figure>;
}

function PredictionFiles({ project, batch, configuration, splitUnit }: { project: string; batch: BatchResult; configuration: ConfigurationResult; splitUnit?: string }) {
  const seeds = configuration.splitSeeds.flatMap((split) => split.seeds).filter((seed) => seed.oof);
  if (!seeds.length) return null;
  return <details className="exp-files"><summary>OOF prediction files for {batch.name}</summary>
    <p className="muted">One CSV per training seed: every {splitUnit === 'slide' ? 'slide' : 'slide or patient'} with its label and predicted probabilities.</p>
    {seeds.map((seed) => <div key={`${seed.splitSeed}-${seed.trainingSeed}`} className="exp-file-row"><strong>Seed {seed.trainingSeed}{configuration.splitSeeds.length > 1 ? ` · split ${seed.splitSeed}` : ''}</strong>
      <OOFPredictionDownloads project={project} batchId={batch.batchId} candidate={{ candidateId: configuration.candidateId, trainingSeed: seed.trainingSeed, splitSeed: seed.splitSeed, complete: true, metrics: null, completedRuns: seed.completedRuns, totalRuns: seed.totalRuns }} units={splitUnit === 'slide' ? ['slide'] : undefined} />
    </div>)}
  </details>;
}

function FileActions({ project, record, results, batch }: { project: string; record: ModelExperiment; results: Results; batch: BatchResult }) {
  const [legacy, setLegacy] = useState(false);
  const frozen = record.batches.find((item) => item.id === batch.batchId);
  const stage = experimentStage(record);
  useEffect(() => { setLegacy(false); }, [batch.batchId]);
  return <section className="exp-section exp-exports" aria-label="Export results">
    <div className="inline-actions">
      <button type="button" className="btn btn-secondary btn-small" onClick={() => download(`${record.name}-folds-and-seeds.csv`, csv(resultRows(results)), 'text/csv')}><Icon name="download" size={15} />Folds and seeds (CSV)</button>
      <button type="button" className="btn btn-secondary btn-small" onClick={() => downloadJSON(`${record.name}-results.json`, results)}><Icon name="download" size={15} />Full summary (JSON)</button>
    </div>
    {frozen ? <details className="exp-files" onToggle={(event) => setLegacy(event.currentTarget.open)}><summary>Scoring details by unit for {batch.name}</summary>
      {legacy ? <DevelopmentExecution project={project} batch={frozen} implemented={frozen.state !== 'trashed'} stage={stage} readOnly knownExecution={frozen.execution ?? undefined} view="results" /> : null}
    </details> : null}
  </section>;
}
