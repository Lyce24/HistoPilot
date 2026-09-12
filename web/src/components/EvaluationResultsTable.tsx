import { Fragment, useState } from 'react';
import { computeStatusLabel, predictorMethodLabel, type ModelEvaluation, type FrozenPredictor } from '../api/predictors';
import type { EvaluationCohort } from '../api/evaluation';
import { lifecycleLabel } from '../api/lifecycle';
import { cleanupLink } from '../lib/hashRoute';
import { versionLabelText } from '../lib/versionLabels';
import { shortRecordId } from '../lib/recordLabels';
import { Badge } from './ui';
import { predictorConfigurationLabel, experimentPredictorLink } from '../lib/predictorGroups';
import './RunWorkspace.css';

export default function EvaluationResultsTable({ records, predictors, cohorts, loading, onOpen }: { records: ModelEvaluation[]; predictors: FrozenPredictor[]; cohorts: EvaluationCohort[]; loading: boolean; onOpen: (id: string) => void }) {
  const [search, setSearch] = useState('');
  const [state, setState] = useState('active');
  const [method, setMethod] = useState('all');
  const [cohortId, setCohortId] = useState('');
  const [sort, setSort] = useState('recent');
  const [groupBy, setGroupBy] = useState('cohort');
  const predictorById = new Map(predictors.map((item) => [item.id, item]));
  const cohortById = new Map(cohorts.map((item) => [item.id, versionLabelText(item, 'Test cohort')]));
  const experimentName = (item: ModelEvaluation) => predictorById.get(item.manifest.predictorId)?.manifest.experiment?.name ?? shortRecordId(item.manifest.experimentId);
  const cohortName = (id: string) => cohortById.get(id) ?? shortRecordId(id);
  const metric = (item: ModelEvaluation, key: 'auroc' | 'accuracy') => {
    const selected = item.execution?.result?.metrics?.selected;
    const value = selected?.[key];
    return item.execution?.status === 'completed' && selected?.available && typeof value === 'number' && Number.isFinite(value) ? value : null;
  };
  const format = (value: number | null) => value === null ? '—' : value.toFixed(4);
  const visible = records.filter((item) => (state === 'all' || item.lifecycleState === state) && (!cohortId || item.manifest.cohortId === cohortId) && (method === 'all' || (predictorById.get(item.manifest.predictorId)?.manifest.method ?? 'ensemble') === method) && `${item.manifest.name} ${predictorById.get(item.manifest.predictorId)?.manifest.name ?? ''} ${experimentName(item)} ${predictorById.get(item.manifest.predictorId)?.manifest.candidateId ?? ''} ${cohortName(item.manifest.cohortId)}`.toLowerCase().includes(search.toLowerCase())).sort((a,b) => sort === 'name' ? a.manifest.name.localeCompare(b.manifest.name) : sort === 'auroc' ? (metric(b,'auroc') ?? -1) - (metric(a,'auroc') ?? -1) : b.createdAt.localeCompare(a.createdAt));
  const groups = groupBy === 'none' ? [['all', { name: '', items: visible }]] as const : [...visible.reduce((groups, item) => {
    const id = groupBy === 'experiment' ? item.manifest.experimentId : item.manifest.cohortId;
    const group = groups.get(id) ?? { name: groupBy === 'experiment' ? experimentName(item) : cohortName(id), items: [] };
    group.items.push(item); groups.set(id, group); return groups;
  }, new Map<string, { name: string; items: ModelEvaluation[] }>()).entries()];
  return <div className="run-workspace"><div className="run-toolbar"><label className="label run-search">Search evaluations<input className="field" type="search" value={search} onChange={(event) => setSearch(event.target.value)} /></label><label className="label">Evaluation visibility<select className="field" value={state} onChange={(event) => setState(event.target.value)}><option value="active">Active</option><option value="archived">Archived</option><option value="trashed">Trash</option><option value="all">All records</option></select></label><label className="label">Test cohort filter<select className="field" value={cohortId} onChange={(event) => setCohortId(event.target.value)}><option value="">All cohorts</option>{[...new Set(records.map((item) => item.manifest.cohortId))].map((id) => <option key={id} value={id}>{cohortName(id)}</option>)}</select></label><label className="label">Evaluation method<select className="field" value={method} onChange={(event) => setMethod(event.target.value)}><option value="all">All methods</option><option value="ensemble">Ensemble</option><option value="refit">Refit</option></select></label><label className="label">Sort evaluations<select className="field" value={sort} onChange={(event) => setSort(event.target.value)}><option value="recent">Recently created</option><option value="auroc">Highest AUROC</option><option value="name">Name</option></select></label><label className="label">Group evaluations<select className="field" value={groupBy} onChange={(event) => setGroupBy(event.target.value)}><option value="cohort">Test cohort</option><option value="experiment">Experiment</option><option value="none">No grouping</option></select></label></div>
    <div className="run-kpis"><span><strong>{visible.length}</strong>evaluations</span><span><strong>{visible.filter((item) => item.execution?.status === 'completed').length}</strong>completed</span></div>
    {loading ? <p role="status">Loading evaluations…</p> : !visible.length ? <p>No evaluation records in this view.</p> : <div className="run-table-scroll"><table className="run-table"><thead><tr><th className="run-name">Evaluation / predictor</th><th>Experiment / configuration</th><th>Method / seeds</th><th>Test cohort</th><th>Status</th><th className="run-number">Accuracy</th><th className="run-number">AUROC</th><th>Actions</th></tr></thead><tbody>{groups.map(([id,group]) => <Fragment key={id}>{group.name ? <tr className="run-group"><th colSpan={8}>{group.name}</th></tr> : null}{group.items.map((item) => { const source = predictorById.get(item.manifest.predictorId)?.manifest; return <tr key={item.id}><td><button className="text-button" onClick={() => onOpen(item.id)}>{item.manifest.name}</button><small>{source?.name ?? shortRecordId(item.manifest.predictorId)}</small></td><td><a href={experimentPredictorLink(item.manifest.experimentId, item.manifest.predictorId)}>{source?.experiment?.name ?? shortRecordId(item.manifest.experimentId)}</a>{source ? <small title={source.candidateId}>{predictorConfigurationLabel(source)}</small> : null}</td><td>{source ? predictorMethodLabel(source.method) : 'Unavailable'}<small>Train {source?.trainingSeed ?? '—'} / split {source?.splitSeed ?? '—'}</small></td><td>{cohortName(item.manifest.cohortId)}</td><td><Badge>{computeStatusLabel(item.execution)}</Badge><small>{lifecycleLabel[item.lifecycleState]}</small></td><td>{format(metric(item,'accuracy'))}</td><td>{format(metric(item,'auroc'))}</td><td><a href={cleanupLink(item.id)}>{item.lifecycleState === 'active' ? 'Archive / delete' : 'Restore / manage'}</a></td></tr>; })}</Fragment>)}</tbody></table></div>}
  </div>;
}
