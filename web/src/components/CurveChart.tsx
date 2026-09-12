import { useId, useState } from 'react';
import { chartRange, curveSegments, formatStatistic, type CurvePoint, type Range } from '../lib/evidenceCharts';
export interface CurveSeries { label: string; points: CurvePoint[]; dashed?: boolean }
const colors = ['#087f8c', '#ae4c13', '#7454ad', '#3768ac', '#a33764', '#52656e'];
export default function CurveChart({ title, description, xLabel, yLabel, series, xRange = [0, 1], yRange, referenceDiagonal = false, integerX = false }: {
  title: string; description: string; xLabel: string; yLabel: string; series: CurveSeries[]; xRange?: Range; yRange?: Range; referenceDiagonal?: boolean; integerX?: boolean;
}) {
  const id = useId();
  const [tableOpen, setTableOpen] = useState(false);
  const yr = chartRange(series.flatMap((item) => item.points.map((point) => point.y)), yRange);
  const xr = chartRange([], xRange);
  const xTicks = [...new Set([0, .25, .5, .75, 1].map((fraction) => { const x = xr[0] + fraction * (xr[1] - xr[0]); return integerX ? Math.round(x) : x; }))];
  const px = (value: number) => 65 + (value - xr[0]) / (xr[1] - xr[0]) * 430;
  const py = (value: number) => 242 - (value - yr[0]) / (yr[1] - yr[0]) * 206;
  const available = series.some((item) => curveSegments(item.points).length);
  return <section className="evidence-chart" aria-labelledby={`${id}-title`}>
    <h3 id={`${id}-title`}>{title}</h3><p className="muted">{description}</p>
    {available ? <><svg viewBox="0 0 520 300" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <desc id={`${id}-desc`}>{`${yLabel} versus ${xLabel}. ${series.map((item) => item.label).join(', ')}. Numeric data are available in the table below.`}</desc>
      <defs><clipPath id={`${id}-clip`}><rect x="65" y="36" width="430" height="206" /></clipPath></defs>
      {[0, .25, .5, .75, 1].map((fraction) => { const y = yr[0] + fraction * (yr[1] - yr[0]); return <g key={fraction}><line x1="65" x2="495" y1={py(y)} y2={py(y)} className="evidence-grid" /><text x="58" y={py(y) + 4} textAnchor="end">{formatStatistic(y, 2)}</text></g>; })}
      {xTicks.map((x) => <text key={x} x={px(x)} y="263" textAnchor="middle">{formatStatistic(x, integerX ? 0 : 2)}</text>)}
      <path d="M65 36 V242 H495" className="evidence-axis" />
      <g clipPath={`url(#${id}-clip)`}>{referenceDiagonal ? <path d={`M${px(0)} ${py(0)} L${px(1)} ${py(1)}`} fill="none" stroke="#82948a" strokeDasharray="4 5" /> : null}{series.map((item, index) => <g key={item.label} fill="none" stroke={colors[index % colors.length]} strokeWidth="2.5" strokeDasharray={item.dashed ? '7 5' : undefined}>{curveSegments(item.points).map((segment, number) => segment.length === 1 ? <circle key={number} cx={px(segment[0].x)} cy={py(segment[0].y)} r="3" /> : <path key={number} d={segment.map((point, pointIndex) => `${pointIndex ? 'L' : 'M'}${px(point.x).toFixed(2)} ${py(point.y).toFixed(2)}`).join(' ')} />)}</g>)}</g>
      <text x="280" y="288" textAnchor="middle">{xLabel}</text><text transform="translate(16 139) rotate(-90)" textAnchor="middle">{yLabel}</text>
    </svg><ul className="evidence-legend">{series.map((item, index) => <li key={item.label}><span aria-hidden="true" style={{ borderColor: colors[index % colors.length], borderTopStyle: item.dashed ? 'dashed' : 'solid' }} />{item.label}</li>)}</ul>
      <details onToggle={(event) => setTableOpen(event.currentTarget.open)}><summary>View numeric curve data</summary>{tableOpen ? <>{series.some((item) => item.points.length > 500) ? <p className="muted">Showing the first 500 points per series. Download the report for all returned curve points.</p> : null}<div className="table-wrap evidence-curve-data"><table><thead><tr><th>Series</th><th>{xLabel}</th><th>{yLabel}</th></tr></thead><tbody>{series.flatMap((item) => item.points.slice(0, 500).map((point, index) => <tr key={`${item.label}-${index}`}><th scope="row">{item.label}</th><td>{formatStatistic(point.x, integerX ? 0 : 4)}</td><td>{formatStatistic(point.y, 4)}</td></tr>))}</tbody></table></div></> : null}</details>
    </> : <p className="callout">This curve cannot be estimated from the available labeled records.</p>}
  </section>;
}
