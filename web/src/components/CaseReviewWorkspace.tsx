import { useDeferredValue, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { caseReviews, defaultCaseQuery, type CaseQuery, type CasePage, type ReviewedCase, type CaseSlide } from '../api/caseReview';
import { interpretations, type Interpretation } from '../api/interpretation';
import type { ModelEvaluation } from '../api/predictors';
import { reviewStatusLabels } from '../api/slideReviews';
import { ErrorNotice, EmptyState } from './ui';
import SlideReviewEditor from './SlideReviewEditor';
import AttentionSlideViewer from './AttentionSlideViewer';
import { QualitySlide } from './VisualQualityExplorer';
import './CaseReview.css';

interface Props { project: string; evaluation: ModelEvaluation; comparisons: ModelEvaluation[]; initial?: Partial<CaseQuery> }
export const caseOutcomeLabels: Record<string, string> = { all: 'All cases', error: 'All errors', false_positive: 'False positives', false_negative: 'False negatives', correct: 'Correct predictions', unlabeled: 'Unlabeled', disagreement: 'Model disagreements' };
export function compatibleComparisons(record: ModelEvaluation, others: ModelEvaluation[]) {
  return others.filter((item) => item.id !== record.id && item.manifest.cohortId === record.manifest.cohortId && item.execution?.status === 'completed' && item.lifecycleState !== 'trashed');
}
export default function CaseReviewWorkspace({ project, evaluation, comparisons, initial }: Props) {
  const [query, setQuery] = useState<CaseQuery>(() => ({ ...defaultCaseQuery(), ...initial }));
  const deferredSearch = useDeferredValue(query.search);
  const selection = { ...query, search: deferredSearch };
  const response = useQuery({ queryKey: ['case-review', project, evaluation.id, selection], queryFn: ({ signal }) => caseReviews.query(project, evaluation.id, selection, signal), staleTime: 15000 });
  const [chosen, setChosen] = useState('');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<Error | null>(null);
  const data = !response.isError ? response.data : undefined;
  const classes = data?.classOrder ?? [];
  const active = data?.items.find((item) => item.id === chosen) ?? data?.items[0];
  function update(next: Partial<CaseQuery>) { setQuery((current) => ({ ...current, ...next, offset: 0 })); setChosen(''); }
  async function exportCases() {
    if (!data || exporting) return;
    setExporting(true); setExportError(null);
    try { await caseReviews.export(project, evaluation.id, query); }
    catch (error) { setExportError(error instanceof Error ? error : new Error('Export failed.')); }
    finally { setExporting(false); }
  }
  const available = compatibleComparisons(evaluation, comparisons);
  return <section aria-label="Case and error review">
    <h3>Case and error review</h3>
    <p className="muted">Inspect saved predictions, original slides, and reviewer findings. Filters describe this evaluation and do not change its cohort or scores.</p>
    <div className="case-review-controls">
      <label className="label">Review unit<select className="field" value={query.unit} onChange={(event) => update({ unit: event.target.value as CaseQuery['unit'] })}><option value="selected">Target unit</option><option value="patient">Patient</option><option value="slide">Slide</option></select></label>
      <label className="label">Cases<select className="field" value={query.outcome} onChange={(event) => update({ outcome: event.target.value as CaseQuery['outcome'], actualClass: null, predictedClass: null })}>{Object.entries(caseOutcomeLabels).filter(([key]) => key !== 'disagreement' || query.comparisonId).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
      <label className="label">Compare with<select className="field" value={query.comparisonId ?? ''} onChange={(event) => update({ comparisonId: event.target.value || null, outcome: query.outcome === 'disagreement' ? 'all' : query.outcome })}><option value="">One evaluation</option>{available.map((item) => <option key={item.id} value={item.id}>{item.manifest.name}</option>)}</select></label>
      <label className="label">Search cases<input className="field" value={query.search} maxLength={200} placeholder="Patient or slide ID" onChange={(event) => update({ search: event.target.value })} /></label>
    </div>
    <details><summary>Class, metadata, and probability filters</summary><div className="case-review-controls">
      <label className="label">Actual class<select className="field" value={query.actualClass ?? ''} onChange={(event) => update({ actualClass: event.target.value === '' ? null : Number(event.target.value) })}><option value="">All classes</option>{classes.map((label, index) => <option key={label} value={index}>{label}</option>)}</select></label>
      <label className="label">Predicted class<select className="field" value={query.predictedClass ?? ''} onChange={(event) => update({ predictedClass: event.target.value === '' ? null : Number(event.target.value) })}><option value="">All classes</option>{classes.map((label, index) => <option key={label} value={index}>{label}</option>)}</select></label>
      <label className="label">Metadata attribute<select className="field" value={query.attribute ?? ''} onChange={(event) => update({ attribute: event.target.value || null, attributeValue: null })}><option value="">All attributes</option>{data?.attributes.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
      {query.attribute ? <label className="label">Attribute value<input className="field" list="case-attribute-values" maxLength={2000} value={query.attributeValue ?? ''} onChange={(event) => update({ attributeValue: event.target.value || null })} /><datalist id="case-attribute-values">{data?.attributes.find((item) => item.key === query.attribute)?.values.map((value) => <option key={value} value={value} />)}</datalist></label> : null}
      <label className="label">Minimum predicted probability<select className="field" value={query.minConfidence} onChange={(event) => update({ minConfidence: Number(event.target.value) })}>{[0, .5, .7, .8, .9, .95].map((value) => <option key={value} value={value}>{value ? `${Math.round(value * 100)}%` : 'Any probability'}</option>)}</select></label>
    </div></details>
    <ErrorNotice error={response.error ?? exportError} />
    {response.isError ? <button type="button" className="btn btn-secondary" onClick={() => void response.refetch()}>Retry case evidence</button> : null}
    {response.isPending ? <p role="status">Verifying saved predictions and loading cases…</p> : null}
    {data ? <>
      <div className="inline-actions"><p role="status">{data.total} matching {data.unit} cases of {data.summary.total}. Ordered by predicted probability.</p><button type="button" className="btn btn-secondary" disabled={exporting || response.isFetching || !data.total} onClick={() => void exportCases()}>{exporting ? 'Exporting…' : 'Export filtered cases and reviews'}</button></div>
      {(query.actualClass !== null || query.predictedClass !== null) ? <p className="callout">Confusion-matrix selection: {query.actualClass === null ? 'any actual class' : data.classOrder[query.actualClass]} → {query.predictedClass === null ? 'any prediction' : data.classOrder[query.predictedClass]}. <button type="button" className="text-button" onClick={() => update({ actualClass: null, predictedClass: null })}>Clear class selection</button></p> : null}
      {data.items.length ? <div className="case-review-layout"><div><div className="table-wrap case-review-list"><table className="chain-table"><thead><tr><th scope="col">Case</th><th scope="col">Actual → predicted</th><th scope="col"><abbr title="Probability of the predicted class">Prob.</abbr></th></tr></thead><tbody>{data.items.map((item) => <tr key={item.id}><th scope="row"><button type="button" className="text-button" aria-pressed={active?.id === item.id} onClick={() => setChosen(item.id)}>{item.id}</button><small>{caseOutcomeLabels[item.outcome] ?? item.outcome}</small>{item.comparison?.disagrees ? <small>Models disagree</small> : null}</th><td>{item.label ?? 'Unlabeled'} → {item.predictedLabel}</td><td>{(item.confidence * 100).toFixed(1)}%</td></tr>)}</tbody></table></div><div className="inline-actions"><button type="button" className="btn btn-secondary" disabled={!query.offset} onClick={() => setQuery((current) => ({ ...current, offset: Math.max(0, current.offset - current.limit) }))}>Previous cases</button><button type="button" className="btn btn-secondary" disabled={!data.hasMore} onClick={() => setQuery((current) => ({ ...current, offset: current.offset + current.limit }))}>Next cases</button></div></div>{active ? <CaseDetail key={`${data.evaluationId}:${data.comparison?.id ?? ''}:${active.id}`} project={project} record={active} page={data} /> : null}</div> : <EmptyState title="No matching cases" description="Adjust the filters to inspect other predictions from this evaluation." />}
    </> : null}
  </section>;
}

export function CaseDetail({ project, record, page }: { project: string; record: ReviewedCase; page: CasePage }) {
  const [slideId, setSlideId] = useState(record.slides[0]?.slideId ?? '');
  const slide = record.slides.find((item) => item.slideId === slideId) ?? record.slides[0];
  return <article className="case-review-detail" aria-label={`Case ${record.id}`}>
    <h4>{record.id}</h4><p>Actual label: <strong>{record.label ?? 'Unlabeled'}</strong>{record.patientId ? ` · ${page.unit === 'patient' ? 'Patient' : 'Case/group ID'} ${record.patientId}` : ''}</p>
    <div className="case-predictions"><Prediction name={page.name} probabilities={record.probabilities} predicted={record.predictedLabel} classes={page.classOrder} />{record.comparison && page.comparison ? <Prediction name={page.comparison.name} probabilities={record.comparison.probabilities} predicted={record.comparison.predictedLabel} classes={page.classOrder} /> : null}</div>
    {page.positiveClass ? <p className="muted">Frozen {page.positiveClass} threshold: {page.decisionThreshold}{page.comparison ? `; comparison: ${page.comparison.decisionThreshold}` : ''}.</p> : null}
    <details><summary>Frozen case metadata</summary><dl className="case-metadata">{Object.entries(record.attributes).map(([key, values]) => <div key={key} style={{ display: 'contents' }}><dt>{page.attributes.find((field) => field.key === key)?.label ?? key}</dt><dd>{values.map((value) => value ?? 'Missing').join(' / ')}</dd></div>)}</dl>{page.unit === 'patient' ? <p className="muted">Multiple values indicate differences between this patient's slides.</p> : null}</details>
    {record.slides.length > 1 ? <label className="label">Patient slide<select className="field" value={slideId} onChange={(event) => setSlideId(event.target.value)}>{record.slides.map((item) => <option key={item.slideId} value={item.slideId}>{item.slideId} · {reviewStatusLabels[item.review.status]}</option>)}</select></label> : null}
    {slide ? <CaseSlideDetail key={slide.slideId} project={project} slide={slide} page={page} /> : null}
  </article>;
}
function Prediction({ name, probabilities, predicted, classes }: { name: string; probabilities: number[]; predicted: string; classes: string[] }) {
  return <div><small>{name}</small><strong>{predicted}</strong>{probabilities.map((value, index) => <small key={classes[index]}>{classes[index]}: {(value * 100).toFixed(1)}%</small>)}</div>;
}
export function CaseSlideDetail({ project, slide, page }: { project: string; slide: CaseSlide; page: CasePage }) {
  const [showAttention, setShowAttention] = useState(false);
  const attentionModels = [{ predictorId: page.predictorId, name: page.name, supportsAttention: page.supportsAttention }, ...(page.comparison ? [page.comparison] : [])].filter((item) => item.supportsAttention === true);
  const [selectedAttentionPredictor, setAttentionPredictor] = useState('');
  const attentionPredictor = attentionModels.some((item) => item.predictorId === selectedAttentionPredictor) ? selectedAttentionPredictor : attentionModels[0]?.predictorId;
  const attention = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project), enabled: showAttention && Boolean(attentionPredictor), staleTime: 30000 });
  const available = attention.data?.items.filter((item) => item.manifest.predictorId === attentionPredictor && item.manifest.slides.some((entry) => entry.slideId === slide.slideId && entry.slidePath === slide.slidePath) && item.execution?.status === 'completed' && item.lifecycleState !== 'trashed') ?? [];
  const study = available[0];
  const [region, setRegion] = useState({ x: '0', y: '0', width: '512', height: '512' });
  const [markRegion, setMarkRegion] = useState(false);
  const validRegion = Object.values(region).every((value) => value.trim() && Number.isFinite(Number(value)) && Number(value) >= 0) && Number(region.width) > 0 && Number(region.height) > 0;
  const regionValue = markRegion && validRegion ? { x: Number(region.x), y: Number(region.y), width: Number(region.width), height: Number(region.height) } : undefined;
  return <>
    <h4>Slide {slide.slideId}</h4>
    {slide.hasImage ? <QualitySlide project={project} datasetId={slide.datasetId} slideId={slide.slideId} featureBundleId={page.featureBundleId ?? undefined} showReview={false} onRegion={(value) => { setMarkRegion(true); setRegion({ x: String(value.x), y: String(value.y), width: String(value.width), height: String(value.height) }); }} /> : <p className="callout">This frozen dataset has no linked image for this slide. Predictions and notes remain available.</p>}
    {attentionPredictor ? <div className="inline-actions"><button type="button" className="btn btn-secondary" aria-expanded={showAttention} onClick={() => setShowAttention(!showAttention)}>{showAttention ? 'Hide model attention' : 'Inspect model attention'}</button><a className="btn btn-secondary" href={`#interpretation?${new URLSearchParams({ predictor: attentionPredictor, evaluation: attentionPredictor === page.predictorId ? page.evaluationId : page.comparison!.id, slide: slide.slideId, search: slide.slideId })}`}>Open interpretation workspace</a></div> : <p className="muted">Patch attention is unavailable for these model inputs. Original slide review and notes remain available.</p>}
    {showAttention && attentionPredictor ? <><label className="label">Attention model<select className="field" value={attentionPredictor} onChange={(event) => setAttentionPredictor(event.target.value)}>{attentionModels.map((item) => <option key={item.predictorId} value={item.predictorId}>{item.name}</option>)}</select></label><ErrorNotice error={attention.error} />{attention.isPending ? <p role="status">Checking saved attention…</p> : study ? <SavedAttention project={project} record={study} slideId={slide.slideId} /> : <p className="callout">No completed attention map for this model and slide. Open the interpretation workspace to prepare it.</p>}</> : null}
    <details className="case-review-region"><summary>Mark a review region</summary><label><input type="checkbox" checked={markRegion} onChange={(event) => setMarkRegion(event.target.checked)} /> Record a rectangle in full-resolution slide pixels</label>{markRegion ? <div className="case-review-controls">{(['x', 'y', 'width', 'height'] as const).map((key) => <label key={key} className="label">{key}<input className="field" type="number" min={key === 'x' || key === 'y' ? 0 : 1} value={region[key]} onChange={(event) => setRegion((current) => ({ ...current, [key]: event.target.value }))} /></label>)}</div> : null}</details>
    <SlideReviewEditor project={project} datasetId={slide.datasetId} slideId={slide.slideId} evaluationId={page.evaluationId} selectedRegion={regionValue} />
  </>;
}
function SavedAttention({ project, record, slideId }: { project: string; record: Interpretation; slideId: string }) {
  const slide = record.manifest.slides.find((item) => item.slideId === slideId);
  const result = record.execution?.result?.slides?.find((item) => item.slideId === slideId);
  return slide && result ? <AttentionSlideViewer project={project} record={record} slide={slide} result={result} /> : <p>Saved attention is incomplete.</p>;
}
