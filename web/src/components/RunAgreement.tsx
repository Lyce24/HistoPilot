import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { CaseQuery } from '../api/caseReview';
import { ApiError } from '../api/client';
import { references, type AgreementPair, type RunAgreement as Agreement } from '../api/references';
import { formatStatistic } from '../api/statistics';
import { percent } from '../lib/inference';
import { ErrorNotice } from './ui';

type Unit = CaseQuery['unit'];
type Measure = 'kappa' | 'weightedKappa' | 'agreement';
const MEASURES: Record<Measure, string> = { kappa: 'Cohen’s κ', weightedKappa: 'Linear weighted κ', agreement: 'Percent agreement' };
/** A run whose decisions cannot be paired yet reads as a note rather than an error. */
const EXPECTED_CODES = new Set(['AGREEMENT_UNAVAILABLE']);

/** The pair of two sources, in either order. */
export function agreementPair(pairs: readonly AgreementPair[], left: string, right: string) {
  return pairs.find((item) => (item.left === left && item.right === right) || (item.left === right && item.right === left));
}

function measured(pair: AgreementPair, measure: Measure) {
  return measure === 'agreement' ? percent(pair.agreement) : formatStatistic(measure === 'kappa' ? pair.kappa : pair.weightedKappa);
}

/**
 * How often this run and every label source of its cohort agree, pair by pair: the cohort's
 * labels and each reference standard, such as individual readers and their consensus. Each
 * pair covers the units both sides label; development patients are left out, as from every
 * metric.
 */
export default function RunAgreement({ project, evaluationId, patient, targetUnit, onReview }: {
  project: string; evaluationId: string;
  /** Why a patient view is unavailable, or null. */
  patient: string | null; targetUnit?: string;
  /** Review the run's disagreements with one label source (null: the cohort's labels). */
  onReview: (source: string | null, query: Partial<CaseQuery>) => void;
}) {
  const [unit, setUnit] = useState<Unit>('selected');
  const [measure, setMeasure] = useState<Measure>('kappa');
  const [chosen, setChosen] = useState<[string, string] | null>(null);
  const query = useQuery({ queryKey: ['run-agreement', project, evaluationId, unit], queryFn: ({ signal }) => references.agreement(project, evaluationId, unit, signal), staleTime: 60000 });
  const data: Agreement | undefined = query.isError ? undefined : query.data;
  const unavailable = query.error instanceof ApiError && EXPECTED_CODES.has(query.error.code ?? '') ? query.error.message : null;
  const ordered = data ? data.classOrder.length > 2 : false;
  const shown = measure === 'weightedKappa' && !ordered ? 'kappa' : measure;
  const usable = data?.sources.filter((item) => item.labeled !== undefined) ?? [];
  const unusable = data?.sources.filter((item) => item.labeled === undefined) ?? [];
  const pair = chosen && data ? agreementPair(data.pairs, ...chosen) : undefined;
  const names = new Map(data?.sources.map((item) => [item.id, item.name]));
  return <div className="run-tab-body">
    <p className="muted">How often this run&rsquo;s decisions and each label source agree: the cohort&rsquo;s labels and every reference standard of it, such as individual readers and their consensus. Each pair covers the units both sides label.</p>
    <div className="inference-controls" role="group" aria-label="Agreement settings">
      <label className="label">Unit<select className="field" value={unit} onChange={(event) => { setUnit(event.target.value as Unit); setChosen(null); }}>
        <option value="selected">Target unit ({targetUnit ?? 'configured'})</option><option value="slide">Slide</option>
        <option value="patient" disabled={Boolean(patient)}>Patient{patient ? ' · unavailable' : ''}</option>
      </select></label>
      <label className="label">Measure<select className="field" value={shown} onChange={(event) => setMeasure(event.target.value as Measure)}>{(Object.keys(MEASURES) as Measure[]).filter((key) => key !== 'weightedKappa' || ordered).map((key) => <option key={key} value={key}>{MEASURES[key]}</option>)}</select></label>
    </div>
    <ErrorNotice error={unavailable ? null : query.error} />
    {unavailable ? <p className="muted" role="status">{unavailable}</p> : null}
    {query.isPending ? <p role="status">Pairing the run&rsquo;s decisions with each label source…</p> : null}
    {data ? <>
      {usable.length > 1 ? <div className="table-wrap"><table className="chain-table agreement-matrix" aria-label={`${MEASURES[shown]} between every pair`}>
        <thead><tr><th scope="col"><span className="sr-only">Source</span></th>{usable.map((item) => <th key={item.id} scope="col">{item.name}</th>)}</tr></thead>
        <tbody>{usable.map((row) => <tr key={row.id}><th scope="row">{row.name}<small>{row.kind === 'run' ? 'This run' : `${(row.labeled ?? 0).toLocaleString()} labeled`}</small></th>{usable.map((column) => {
          if (column.id === row.id) return <td key={column.id} className="is-self" aria-label="Same source">—</td>;
          const item = agreementPair(data.pairs, row.id, column.id);
          const selected = Boolean(chosen && agreementPair(item ? [item] : [], ...chosen));
          return <td key={column.id}>{item ? <button type="button" className="text-button" aria-pressed={selected} aria-label={`${row.name} and ${column.name}: ${MEASURES[shown]} ${measured(item, shown)} over ${item.count}`} onClick={() => setChosen([row.id, column.id])}>{measured(item, shown)}<small>n = {item.count.toLocaleString()}</small></button> : '—'}</td>;
        })}</tr>)}</tbody>
      </table></div> : <p className="callout">No label source can be compared with this run yet. Score it against labels to compare them.</p>}
      <p className="muted">Select a pair to see where the two disagree. κ corrects agreement for what each side&rsquo;s class frequencies would produce by chance.{ordered ? ` Linear weighted κ counts near misses on the order ${data.classOrder.join(' < ')} as partial agreement; read it only when the classes are ordered.` : ''}{data.developmentExcluded ? ` ${data.developmentExcluded.toLocaleString()} slides from development patients are left out, as from every metric.` : ''}</p>
      {unusable.map((item) => <p key={item.id} className="callout">{item.name} cannot be compared: {item.reason}</p>)}
      {pair ? <figure className="run-figure">
        <figcaption>{names.get(pair.left)} and {names.get(pair.right)}</figcaption>
        <p className="muted">{pair.count.toLocaleString()} {data.unit}s labeled by both · agreement {percent(pair.agreement)} · κ {formatStatistic(pair.kappa)}{ordered ? ` · weighted κ ${formatStatistic(pair.weightedKappa)}` : ''} · {pair.disagreements.toLocaleString()} disagreements. Rows: {names.get(pair.left)}. Columns: {names.get(pair.right)}.</p>
        <div className="table-wrap"><table className="chain-table"><thead><tr><th scope="col">{names.get(pair.left)} / {names.get(pair.right)}</th>{data.classOrder.map((name) => <th key={name} scope="col">{name}</th>)}</tr></thead>
          <tbody>{pair.matrix.map((row, index) => <tr key={index}><th scope="row">{data.classOrder[index]}</th>{row.map((value, column) => <td key={column} className={column === index ? 'is-agreement' : undefined}>{value.toLocaleString()}</td>)}</tr>)}</tbody></table></div>
        {pair.left === 'run' && pair.disagreements ? <div className="inline-actions"><button type="button" className="btn btn-secondary" onClick={() => onReview(pair.right === 'cohort' ? null : pair.right, { unit, outcome: 'error' })}>Review the run&rsquo;s errors against {names.get(pair.right)}</button></div> : null}
      </figure> : null}
    </> : null}
  </div>;
}
