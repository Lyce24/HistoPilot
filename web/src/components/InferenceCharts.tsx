import { useId, useState } from 'react';
import type { InferenceComparison, InferencePredictedClass, InferenceUnitSummary } from '../api/inference';
import { decimal, percent } from '../lib/inference';
import './InferenceResults.css';

/** Class identity uses one categorical slot per class, in frozen class order, on every chart. */
export const seriesVar = (index: number) => `var(--inference-series-${(index % 8) + 1})`;

export function ClassLegend({ classes }: { classes: string[] }) {
  return <ul className="inference-legend" aria-label="Class colors">{classes.map((label, index) => <li key={label}><span className="inference-swatch" style={{ background: seriesVar(index) }} aria-hidden="true" />{label}</li>)}</ul>;
}

function niceMaximum(value: number) {
  if (value <= 1) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 2.5, 5, 10].find((candidate) => candidate * power >= value) ?? 10;
  return step * power;
}

/** Magnitude per class with every value labeled at the bar tip. */
export function PredictedClassBars({ predicted, unit }: { predicted: InferencePredictedClass[]; unit: string }) {
  const maximum = Math.max(1, ...predicted.map((item) => item.count));
  return <figure className="inference-figure" aria-label={`Predicted class per ${unit}`}>
    <figcaption>Predicted class</figcaption>
    <div className="inference-class-bars" role="list">{predicted.map((item, index) => <div role="listitem" className="inference-class-bar" key={item.label} tabIndex={0} aria-label={`${item.label}: ${item.count} ${unit}s, ${percent(item.fraction)}, mean confidence ${percent(item.meanConfidence)}`}>
      <span className="inference-class-name"><span className="inference-swatch" style={{ background: seriesVar(index) }} aria-hidden="true" />{item.label}</span>
      <span className="inference-class-track" aria-hidden="true"><span style={{ width: `${(item.count / maximum) * 100}%`, background: seriesVar(index) }} /></span>
      <span className="inference-class-value"><strong>{item.count.toLocaleString()}</strong> {percent(item.fraction)}</span>
      <span className="inference-class-note">{item.count ? `mean confidence ${percent(item.meanConfidence)}` : 'none predicted'}</span>
    </div>)}</div>
  </figure>;
}

export interface HistogramSeries { label: string; counts: number[] }
interface HoverSegment { bin: number; series: number }

/**
 * Unit-interval histogram. Several series stack in class order with a 2px surface
 * gap; the table view carries every value for readers who cannot rely on hover.
 */
export function Histogram({ title, description, edges, series, xLabel, colorOffset = 0, marker }: {
  title: string; description: string; edges: number[]; series: HistogramSeries[]; xLabel: string;
  colorOffset?: number; marker?: { value: number; label: string };
}) {
  const id = useId();
  const [hover, setHover] = useState<HoverSegment | null>(null);
  const bins = edges.length - 1;
  const totals = Array.from({ length: bins }, (_, bin) => series.reduce((sum, item) => sum + (item.counts[bin] ?? 0), 0));
  const maximum = niceMaximum(Math.max(...totals, 1));
  const width = 560, height = 230, left = 46, right = 14, top = 14, bottom = 40;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const band = plotWidth / bins, barWidth = Math.min(24, band - 2);
  const x = (value: number) => left + value * plotWidth;
  const y = (value: number) => top + plotHeight - (value / maximum) * plotHeight;
  const ticks = [0, maximum / 2, maximum];
  const range = (bin: number) => `${decimal(edges[bin], 2)}–${decimal(edges[bin + 1], 2)}`;
  const active = hover ? { bin: hover.bin, label: series[hover.series].label, count: series[hover.series].counts[hover.bin] ?? 0 } : null;
  const total = totals.reduce((sum, value) => sum + value, 0);
  return <figure className="inference-figure inference-histogram" aria-labelledby={`${id}-title`}>
    <figcaption id={`${id}-title`}>{title}</figcaption>
    <p className="muted">{description}</p>
    {series.length > 1 ? <ClassLegend classes={series.map((item) => item.label)} /> : null}
    <div className="inference-plot">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`} onPointerLeave={() => setHover(null)}>
        <desc id={`${id}-desc`}>{`${total} records across ${bins} bins of ${xLabel}. Exact counts are in the table view.`}</desc>
        {ticks.map((tick) => <g key={tick} className="inference-grid"><line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} /><text x={left - 8} y={y(tick) + 4} textAnchor="end">{Number.isInteger(tick) ? tick.toLocaleString() : tick.toFixed(1)}</text></g>)}
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => <text key={tick} className="inference-axis-label" x={x(tick)} y={height - bottom + 16} textAnchor="middle">{tick}</text>)}
        <text className="inference-axis-title" x={left + plotWidth / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
        {Array.from({ length: bins }, (_, bin) => {
          let base = top + plotHeight;
          const visible = series.map((item, index) => ({ index, count: item.counts[bin] ?? 0 })).filter((item) => item.count > 0);
          return <g key={bin}>{visible.map((item, position) => {
            const heightValue = (item.count / maximum) * plotHeight;
            const gap = position ? 2 : 0;
            const segmentTop = base - heightValue;
            const barHeight = Math.max(1, heightValue - gap);
            const x0 = left + bin * band + (band - barWidth) / 2;
            const last = position === visible.length - 1;
            const radius = last ? Math.min(4, barHeight / 2, barWidth / 2) : 0;
            base = segmentTop;
            const path = `M${x0} ${segmentTop + barHeight}V${segmentTop + radius}${radius ? `Q${x0} ${segmentTop} ${x0 + radius} ${segmentTop}` : ''}H${x0 + barWidth - radius}${radius ? `Q${x0 + barWidth} ${segmentTop} ${x0 + barWidth} ${segmentTop + radius}` : ''}V${segmentTop + barHeight}Z`;
            const selected = hover?.bin === bin && hover.series === item.index;
            return <path key={item.index} d={path} className={selected ? 'is-active' : undefined} style={{ fill: seriesVar(item.index + colorOffset) }} tabIndex={0}
              aria-label={`${series[item.index].label}, ${xLabel} ${range(bin)}: ${item.count}`}
              onPointerEnter={() => setHover({ bin, series: item.index })} onFocus={() => setHover({ bin, series: item.index })} onBlur={() => setHover(null)} />;
          })}</g>;
        })}
        {marker ? <g className="inference-marker"><line x1={x(marker.value)} x2={x(marker.value)} y1={top} y2={top + plotHeight} /><text x={x(marker.value) + 6} y={top + 12}>{marker.label}</text></g> : null}
      </svg>
      {active && hover ? <div className="inference-tooltip" role="status" style={{ left: `${((left + (hover.bin + 0.5) * band) / width) * 100}%` }}><strong>{active.count.toLocaleString()}</strong><span><span className="inference-tooltip-key" style={{ background: seriesVar(hover.series + colorOffset) }} aria-hidden="true" />{active.label}</span><small>{xLabel} {range(hover.bin)}</small></div> : null}
    </div>
    <details className="inference-table-view"><summary>Table view</summary><div className="table-wrap"><table className="chain-table"><thead><tr><th scope="col">{xLabel}</th>{series.map((item) => <th scope="col" key={item.label}>{item.label}</th>)}{series.length > 1 ? <th scope="col">Total</th> : null}</tr></thead><tbody>{totals.map((value, bin) => value ? <tr key={bin}><th scope="row">{range(bin)}</th>{series.map((item) => <td key={item.label}>{(item.counts[bin] ?? 0).toLocaleString()}</td>)}{series.length > 1 ? <td>{value.toLocaleString()}</td> : null}</tr> : null)}</tbody></table></div></details>
  </figure>;
}

/** How many fold members vote for the ensemble decision. */
export function AgreementBars({ ensemble, unit }: { ensemble: NonNullable<InferenceUnitSummary['ensemble']>; unit: string }) {
  const maximum = Math.max(1, ...ensemble.agreement.map((item) => item.count));
  return <figure className="inference-figure" aria-label="Fold member agreement">
    <figcaption>Fold members agreeing with the ensemble</figcaption>
    <p className="muted">{ensemble.unanimous.toLocaleString()} of {ensemble.records.toLocaleString()} {unit}s are unanimous · mean spread of the chosen class probability {decimal(ensemble.meanSpread)}</p>
    <div className="inference-class-bars" role="list">{ensemble.agreement.map((item) => <div role="listitem" className="inference-class-bar" key={item.agree} tabIndex={0} aria-label={`${item.agree} of ${ensemble.memberCount} members agree: ${item.count} ${unit}s`}>
      <span className="inference-class-name">{item.agree} of {ensemble.memberCount}</span>
      <span className="inference-class-track" aria-hidden="true"><span style={{ width: `${(item.count / maximum) * 100}%`, background: seriesVar(0) }} /></span>
      <span className="inference-class-value"><strong>{item.count.toLocaleString()}</strong> {percent(ensemble.records ? item.count / ensemble.records : null)}</span>
      <span className="inference-class-note">{item.agree === ensemble.memberCount ? 'unanimous' : item.agree * 2 > ensemble.memberCount ? 'majority' : 'minority'}</span>
    </div>)}</div>
  </figure>;
}

export interface CompositionRow { value: string; count: number; counts: Record<string, number>; meanConfidence?: number }

/** Counts per group, with a 100% bar per row; the numbers are always visible. */
export function CompositionTable({ caption, groupLabel, rows, classes, unit }: {
  caption: string; groupLabel: string; rows: CompositionRow[]; classes: string[]; unit: string;
}) {
  return <figure className="inference-figure">
    <figcaption>{caption}</figcaption>
    <ClassLegend classes={classes} />
    <div className="table-wrap"><table className="chain-table inference-composition"><thead><tr><th scope="col">{groupLabel}</th><th scope="col">{unit[0].toUpperCase() + unit.slice(1)}s</th><th scope="col">Predicted classes</th>{classes.map((label) => <th scope="col" key={label}>{label}</th>)}{rows.some((row) => row.meanConfidence !== undefined) ? <th scope="col">Mean confidence</th> : null}</tr></thead>
      <tbody>{rows.map((row) => <tr key={row.value}><th scope="row">{row.value}</th><td>{row.count.toLocaleString()}</td>
        <td><span className="inference-composition-bar" aria-hidden="true">{classes.map((label, index) => row.counts[label] ? <span key={label} style={{ flexGrow: row.counts[label], background: seriesVar(index) }} /> : null)}</span></td>
        {classes.map((label) => <td key={label}>{(row.counts[label] ?? 0).toLocaleString()}<small>{percent(row.count ? (row.counts[label] ?? 0) / row.count : null, 0)}</small></td>)}
        {row.meanConfidence !== undefined ? <td>{percent(row.meanConfidence)}</td> : rows.some((item) => item.meanConfidence !== undefined) ? <td>—</td> : null}</tr>)}</tbody></table></div>
  </figure>;
}

/** Predicted positives across thresholds; the frozen threshold row is marked. */
export function ThresholdTable({ binary, total }: { binary: NonNullable<InferenceUnitSummary['binary']>; total: number }) {
  const rows = binary.sweep.filter((item) => Math.abs(item.threshold * 10 - Math.round(item.threshold * 10)) < 1e-9 || item.threshold === binary.threshold);
  return <figure className="inference-figure">
    <figcaption>Predicted {binary.positiveClass} across decision thresholds</figcaption>
    <p className="muted">The frozen threshold is {binary.threshold}. {binary.nearThreshold.map((item) => `${item.count.toLocaleString()} within ±${item.band}`).join(' · ')} of it. Other thresholds are descriptive; the saved decisions never change.</p>
    <div className="table-wrap"><table className="chain-table inference-threshold"><thead><tr><th scope="col">Threshold</th><th scope="col">Predicted {binary.positiveClass}</th><th scope="col">Share</th></tr></thead><tbody>{rows.map((item) => <tr key={item.threshold} className={item.threshold === binary.threshold ? 'is-frozen' : undefined}><th scope="row">{item.threshold}{item.threshold === binary.threshold ? ' · frozen' : ''}</th><td>{item.positive.toLocaleString()}</td><td><span className="inference-inline-bar" aria-hidden="true"><span style={{ width: `${total ? (item.positive / total) * 100 : 0}%`, background: seriesVar(1) }} /></span>{percent(total ? item.positive / total : null)}</td></tr>)}</tbody></table></div>
  </figure>;
}

/** Label-free cross-tab of two runs' decisions on the same cases. */
export function ComparisonMatrix({ comparison, classes, name }: { comparison: InferenceComparison; classes: string[]; name: string }) {
  return <figure className="inference-figure">
    <figcaption>{name} versus {comparison.name}</figcaption>
    <p className="muted">Agreement {percent(comparison.agreement)} · Cohen&rsquo;s κ {decimal(comparison.kappa)} · {comparison.disagreements.toLocaleString()} disagreements of {comparison.count.toLocaleString()}. Agreement is not accuracy: both runs can be wrong together.</p>
    <div className="table-wrap"><table className="chain-table inference-matrix"><thead><tr><th scope="col">{name} ↓ / {comparison.name} →</th>{classes.map((label) => <th scope="col" key={label}>{label}</th>)}</tr></thead><tbody>{comparison.matrix.map((row, index) => <tr key={classes[index]}><th scope="row">{classes[index]}</th>{row.map((value, column) => <td key={classes[column]} className={index === column ? 'is-agreement' : undefined}>{value.toLocaleString()}</td>)}</tr>)}</tbody></table></div>
  </figure>;
}
