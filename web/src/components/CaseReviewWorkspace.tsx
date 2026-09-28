import { useDeferredValue, useId, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { caseReviews, defaultCaseQuery, updateCaseQuery, type CaseQuery, type CasePage, type CaseSort, type ReviewedCase, type CaseSlide } from '../api/caseReview';
import { inferenceRuns } from '../api/inference';
import { interpretations, type Interpretation } from '../api/interpretation';
import { computeActive, type ModelEvaluation } from '../api/predictors';
import { reviewStatusLabels } from '../api/slideReviews';
import { isInferenceRun, percent } from '../lib/inference';
import { ErrorNotice, EmptyState } from './ui';
import SlideReviewEditor from './SlideReviewEditor';
import AttentionSlideViewer from './AttentionSlideViewer';
import { QualitySlide } from './VisualQualityExplorer';
import './CaseReview.css';

interface Props { project: string; evaluation: ModelEvaluation; comparisons: ModelEvaluation[]; initial?: Partial<CaseQuery> }
export const caseOutcomeLabels: Record<string, string> = { all: 'All cases', error: 'All errors', false_positive: 'False positives', false_negative: 'False negatives', correct: 'Correct predictions', unlabeled: 'Unlabeled', disagreement: 'Model disagreements' };
export const caseSortLabels: Record<CaseSort, string> = { margin_asc: 'Closest to a decision boundary', confidence_asc: 'Least confident first', agreement_asc: 'Most fold-member disagreement', confidence_desc: 'Most confident first' };
/** Attention requests are bounded; larger selections are split by the reviewer. */
export const MAX_ATTENTION_SLIDES = 32;
export function compatibleComparisons(record: ModelEvaluation, others: ModelEvaluation[]) {
  return others.filter((item) => item.id !== record.id && item.manifest.cohortId === record.manifest.cohortId && item.execution?.status === 'completed' && item.lifecycleState !== 'trashed');
}
/** Slides on the current page that have an image, in review order, within the request bound. */
export function attentionSlides(items: ReviewedCase[], limit = MAX_ATTENTION_SLIDES) {
  return [...new Set(items.flatMap((item) => item.slides.filter((slide) => slide.hasImage).map((slide) => slide.slideId)))].slice(0, limit);
}
interface AttentionRunContext { evaluationId: string; predictorId: string; featureBundleId?: string | null; packArtifactId?: string | null }
/** Inference attention must come from this run's exact frozen feature source. */
export function matchesRunAttention(item: Interpretation, slide: CaseSlide, run: AttentionRunContext, inference: boolean) {
  return item.lifecycleState !== 'trashed' && item.manifest.predictorId === run.predictorId
    && item.manifest.slides.some((entry) => entry.slideId === slide.slideId && entry.slidePath === slide.slidePath)
    && (!inference || (item.manifest.evaluationId === run.evaluationId
      && item.manifest.featureBundleId === run.featureBundleId
      && (item.manifest.packArtifactId ?? null) === (run.packArtifactId ?? null)));
}
export default function CaseReviewWorkspace({ project, evaluation, comparisons, initial }: Props) {
  const inference = isInferenceRun(evaluation);
  // Inference review starts with borderline predictions: the margin respects the frozen threshold.
  const [query, setQuery] = useState<CaseQuery>(() => ({ ...defaultCaseQuery(), ...(inference ? { sort: 'margin_asc' as const } : {}), ...initial }));
  const attributeListId = useId();
  const deferredSearch = useDeferredValue(query.search);
  const selection = { ...query, search: deferredSearch };
  const response = useQuery({ queryKey: ['case-review', project, evaluation.id, selection], queryFn: ({ signal }) => caseReviews.query(project, evaluation.id, selection, signal), staleTime: 15000 });
  const [chosen, setChosen] = useState('');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<Error | null>(null);
  const data = !response.isError ? response.data : undefined;
  const classes = data?.classOrder ?? [];
  const listedAttentionSlides = attentionSlides(data?.items ?? []);
  const imageSlideCount = attentionSlides(data?.items ?? [], Infinity).length;
  const active = data?.items.find((item) => item.id === chosen) ?? data?.items[0];
  function update(next: Partial<CaseQuery>) { setQuery((current) => updateCaseQuery(current, next)); setChosen(''); }
  async function exportCases() {
    if (!data || exporting) return;
    setExporting(true); setExportError(null);
    try { await caseReviews.export(project, evaluation.id, query); }
    catch (error) { setExportError(error instanceof Error ? error : new Error('Export failed.')); }
    finally { setExporting(false); }
  }
  const available = compatibleComparisons(evaluation, comparisons);
  const sort = query.sort ?? 'confidence_desc';
  return <section aria-label={inference ? 'Prediction review' : 'Case and error review'}>
    <h3>{inference ? 'Review predictions' : 'Case and error review'}</h3>
    <p className="muted">{inference ? 'Inspect predicted slides with their original images, attention and reviewer notes. No labels are used, and filters never change the saved predictions.' : 'Inspect saved predictions, original slides, and reviewer findings. Filters describe this evaluation and do not change its cohort or scores.'}</p>
    <div className="case-review-controls">
      <label className="label">Review unit<select className="field" value={query.unit} onChange={(event) => update({ unit: event.target.value as CaseQuery['unit'] })}><option value="selected">Target unit</option><option value="patient">Patient</option><option value="slide">Slide</option></select></label>
      {inference
        ? <label className="label">Order<select className="field" value={sort} onChange={(event) => update({ sort: event.target.value as CaseSort })}>{Object.entries(caseSortLabels).filter(([key]) => key !== 'agreement_asc' || data?.memberCount).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
        : <label className="label">Cases<select className="field" value={query.outcome} onChange={(event) => update({ outcome: event.target.value as CaseQuery['outcome'], actualClass: null, predictedClass: null })}>{Object.entries(caseOutcomeLabels).filter(([key]) => key !== 'disagreement' || query.comparisonId).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>}
      <label className="label">Compare with<select className="field" value={query.comparisonId ?? ''} onChange={(event) => update({ comparisonId: event.target.value || null, outcome: query.outcome === 'disagreement' ? 'all' : query.outcome })}><option value="">{inference ? 'One run' : 'One evaluation'}</option>{available.map((item) => <option key={item.id} value={item.id}>{item.manifest.name}</option>)}</select></label>
      <label className="label">Search cases<input className="field" value={query.search} maxLength={200} placeholder="Patient or slide ID" onChange={(event) => update({ search: event.target.value })} /></label>
    </div>
    <details open={inference && Boolean(initial && (initial.memberDisagreement || initial.developmentPatients || initial.maxConfidence !== undefined || initial.maxMargin != null))}><summary>{inference ? 'Class, confidence, agreement and metadata filters' : 'Class, metadata, and probability filters'}</summary><div className="case-review-controls">
      {inference ? null : <label className="label">Actual class<select className="field" value={query.actualClass ?? ''} onChange={(event) => update({ actualClass: event.target.value === '' ? null : Number(event.target.value) })}><option value="">All classes</option>{classes.map((label, index) => <option key={label} value={index}>{label}</option>)}</select></label>}
      <label className="label">Predicted class<select className="field" value={query.predictedClass ?? ''} onChange={(event) => update({ predictedClass: event.target.value === '' ? null : Number(event.target.value) })}><option value="">All classes</option>{classes.map((label, index) => <option key={label} value={index}>{label}</option>)}</select></label>
      <label className="label">Metadata attribute<select className="field" value={query.attribute ?? ''} onChange={(event) => update({ attribute: event.target.value || null, attributeValue: null })}><option value="">All attributes</option>{data?.attributes.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
      {query.attribute ? <label className="label">Attribute value<input className="field" list={attributeListId} maxLength={2000} value={query.attributeValue ?? ''} onChange={(event) => update({ attributeValue: event.target.value || null })} /><datalist id={attributeListId}>{data?.attributes.find((item) => item.key === query.attribute)?.values.map((value) => <option key={value} value={value} />)}</datalist></label> : null}
      <label className="label">Minimum predicted probability<select className="field" value={query.minConfidence} onChange={(event) => update({ minConfidence: Number(event.target.value) })}>{[0, .5, .6, .7, .8, .9, .95].map((value) => <option key={value} value={value}>{value ? `${Math.round(value * 100)}%` : 'Any probability'}</option>)}</select></label>
      {inference ? <label className="label">Maximum predicted probability<select className="field" value={query.maxConfidence ?? 1} onChange={(event) => update({ maxConfidence: Number(event.target.value) })}>{[1, .95, .9, .8, .7, .6, .5].map((value) => <option key={value} value={value}>{value < 1 ? `${Math.round(value * 100)}%` : 'Any probability'}</option>)}</select></label> : null}
      {inference ? <label className="label">Decision margin below<select className="field" value={query.maxMargin ?? ''} onChange={(event) => update({ maxMargin: event.target.value === '' ? null : Number(event.target.value) })}><option value="">Any margin</option>{[.1, .2, .3, .4].map((value) => <option key={value} value={value}>{value}</option>)}</select><small>Distance from the frozen decision boundary.</small></label> : null}
      {inference && data?.developmentComparable ? <label className="label">Development patients<select className="field" value={query.developmentPatients ?? 'all'} onChange={(event) => update({ developmentPatients: event.target.value as CaseQuery['developmentPatients'] })}><option value="all">All patients</option><option value="new">New patients only</option><option value="shared">Patients seen in development</option></select></label> : null}
      {inference && data?.memberCount ? <label className="science-check"><input type="checkbox" checked={Boolean(query.memberDisagreement)} onChange={(event) => update({ memberDisagreement: event.target.checked })} /><span>Only fold-member disagreements<small>At least one of {data.memberCount} fold models votes for another class.</small></span></label> : null}
      {inference && query.comparisonId ? <label className="science-check"><input type="checkbox" checked={query.outcome === 'disagreement'} onChange={(event) => update({ outcome: event.target.checked ? 'disagreement' : 'all' })} /><span>Only disagreements with the compared run</span></label> : null}
    </div></details>
    {inference ? <button type="button" className="text-button" onClick={() => { setQuery({ ...defaultCaseQuery(), unit: query.unit, sort: 'margin_asc' }); setChosen(''); }}>Reset review filters</button> : null}
    <ErrorNotice error={response.error ?? exportError} />
    {response.isError ? <button type="button" className="btn btn-secondary" onClick={() => void response.refetch()}>Retry case evidence</button> : null}
    {response.isPending ? <p role="status">Verifying saved predictions and loading cases…</p> : null}
    {data ? <>
      <div className="inline-actions"><p role="status">{data.total} matching {data.unit} cases of {data.summary.total}. {inference ? `${caseSortLabels[sort]}.` : 'Ordered by predicted probability.'}</p><button type="button" className="btn btn-secondary" disabled={exporting || response.isFetching || !data.total} onClick={() => void exportCases()}>{exporting ? 'Exporting…' : 'Export filtered cases and reviews'}</button>{inference && data.supportsAttention ? <AttentionRequest key={JSON.stringify(listedAttentionSlides)} project={project} evaluationId={evaluation.id} slideIds={listedAttentionSlides} label={`Compute attention for ${imageSlideCount > listedAttentionSlides.length ? `first ${listedAttentionSlides.length} of ${imageSlideCount}` : listedAttentionSlides.length} listed slides`} /> : null}</div>
      {(query.actualClass !== null || query.predictedClass !== null) && !inference ? <p className="callout">Confusion-matrix selection: {query.actualClass === null ? 'any actual class' : data.classOrder[query.actualClass]} → {query.predictedClass === null ? 'any prediction' : data.classOrder[query.predictedClass]}. <button type="button" className="text-button" onClick={() => update({ actualClass: null, predictedClass: null })}>Clear class selection</button></p> : null}
      {data.items.length ? <div className="case-review-layout"><div><div className="table-wrap case-review-list"><table className={`chain-table${inference ? ' case-review-inference' : ''}`}><thead><tr><th scope="col">Case</th><th scope="col">{inference ? <abbr title="Predicted class">Class</abbr> : 'Actual → predicted'}</th><th scope="col"><abbr title="Probability of the predicted class">Prob.</abbr></th>{inference && data.memberCount ? <th scope="col"><abbr title="Fold members voting for the ensemble decision">Members</abbr></th> : null}</tr></thead><tbody>{data.items.map((item) => <tr key={item.id}><th scope="row"><button type="button" className="text-button" aria-pressed={active?.id === item.id} onClick={() => setChosen(item.id)}>{item.id}</button>{inference ? (item.developmentPatient ? <small>Development patient</small> : null) : <small>{caseOutcomeLabels[item.outcome] ?? item.outcome}</small>}{item.comparison?.disagrees ? <small>Models disagree</small> : null}</th><td>{inference ? item.predictedLabel : `${item.label ?? 'Unlabeled'} → ${item.predictedLabel}`}</td><td>{(item.confidence * 100).toFixed(1)}%</td>{inference && data.memberCount ? <td>{item.memberAgreement ? `${item.memberAgreement.agree}/${item.memberAgreement.total}` : '—'}</td> : null}</tr>)}</tbody></table></div><div className="inline-actions"><button type="button" className="btn btn-secondary" disabled={!query.offset} onClick={() => setQuery((current) => ({ ...current, offset: Math.max(0, current.offset - current.limit) }))}>Previous cases</button><button type="button" className="btn btn-secondary" disabled={!data.hasMore} onClick={() => setQuery((current) => ({ ...current, offset: current.offset + current.limit }))}>Next cases</button></div></div>{active ? <CaseDetail key={`${data.evaluationId}:${data.comparison?.id ?? ''}:${active.id}`} project={project} record={active} page={data} /> : null}</div> : <EmptyState title="No matching cases" description={inference ? 'Adjust the filters to inspect other predictions from this run.' : 'Adjust the filters to inspect other predictions from this evaluation.'} />}
    </> : null}
  </section>;
}

export function CaseDetail({ project, record, page }: { project: string; record: ReviewedCase; page: CasePage }) {
  const [slideId, setSlideId] = useState(record.slides[0]?.slideId ?? '');
  const slide = record.slides.find((item) => item.slideId === slideId) ?? record.slides[0];
  const inference = page.purpose === 'inference';
  const identity = record.patientId ? `${page.unit === 'patient' ? 'Patient' : 'Case/group ID'} ${record.patientId}` : '';
  return <article className="case-review-detail" aria-label={`Case ${record.id}`}>
    <h4>{record.id}</h4>
    {inference
      ? <p>Predicted <strong>{record.predictedLabel}</strong> · {percent(record.confidence)} · margin {percent(record.margin ?? null)}{record.memberAgreement ? ` · ${record.memberAgreement.agree} of ${record.memberAgreement.total} fold members agree` : ''}{identity ? ` · ${identity}` : ''}{record.developmentPatient ? <span className="inference-flag">Patient seen in development</span> : null}</p>
      : <p>Actual label: <strong>{record.label ?? 'Unlabeled'}</strong>{identity ? ` · ${identity}` : ''}</p>}
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

/**
 * Queue attention through the run's frozen features and slide folder. The server
 * reuses completed or running studies, so repeating the request is safe.
 */
export function AttentionRequest({ project, evaluationId, slideIds, label, onQueued }: { project: string; evaluationId: string; slideIds: string[]; label: string; onQueued?: () => void }) {
  const client = useQueryClient();
  // A retry reuses its operation only for the same run and slides; new inputs need a new operation.
  const inputs = JSON.stringify([evaluationId, slideIds]);
  const [pending, setPending] = useState<{ id: string; inputs: string } | null>(null);
  const operation = pending?.inputs === inputs ? pending.id : null;
  const setOperation = (id: string | null) => setPending(id ? { id, inputs } : null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  async function request() {
    if (busy || !slideIds.length) return;
    const id = operation ?? crypto.randomUUID(); setOperation(id); setBusy(true); setError(null); setMessage('');
    try {
      const result = await inferenceRuns.attention(project, evaluationId, slideIds, id);
      const failed = result.items.filter((item) => item.error);
      setMessage(failed.length ? `${result.items.length - failed.length} queued or reused · ${failed.length} unavailable: ${failed[0].error?.message ?? 'unknown error'}` : `${result.items.length} attention ${result.items.length === 1 ? 'map' : 'maps'} queued or reused. They appear here when complete.`);
      setOperation(null);
      await client.invalidateQueries({ queryKey: ['interpretations', project] });
      onQueued?.();
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Attention could not be requested.')); }
    finally { setBusy(false); }
  }
  return <span className="attention-request"><button type="button" className="btn btn-secondary" disabled={busy || !slideIds.length} onClick={() => void request()}>{busy ? 'Requesting attention…' : operation ? 'Retry attention request' : label}</button>{message ? <small role="status">{message}</small> : null}<ErrorNotice error={error} /></span>;
}

export function CaseSlideDetail({ project, slide, page }: { project: string; slide: CaseSlide; page: CasePage }) {
  const inference = page.purpose === 'inference';
  const [showAttention, setShowAttention] = useState(false);
  const attentionModels = [{ predictorId: page.predictorId, name: page.name, supportsAttention: page.supportsAttention, evaluationId: page.evaluationId, featureBundleId: page.featureBundleId, packArtifactId: page.packArtifactId }, ...(page.comparison ? [{ ...page.comparison, evaluationId: page.comparison.id }] : [])].filter((item) => item.supportsAttention === true);
  const [selectedAttentionRun, setAttentionRun] = useState('');
  const attentionModel = attentionModels.find((item) => item.evaluationId === selectedAttentionRun) ?? attentionModels[0];
  const attentionPredictor = attentionModel?.predictorId;
  const matches = (item: Interpretation) => Boolean(attentionModel && matchesRunAttention(item, slide, attentionModel, inference));
  const attention = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project), enabled: showAttention && Boolean(attentionPredictor), staleTime: 30000,
    refetchInterval: (query) => showAttention && query.state.data?.items.some((item) => matches(item) && computeActive(item.execution)) ? 3000 : false });
  const study = attention.data?.items.find((item) => matches(item) && item.execution?.status === 'completed');
  const pending = attention.data?.items.find((item) => matches(item) && computeActive(item.execution));
  const [region, setRegion] = useState({ x: '0', y: '0', width: '512', height: '512' });
  const [markRegion, setMarkRegion] = useState(false);
  const validRegion = Object.values(region).every((value) => value.trim() && Number.isFinite(Number(value)) && Number(value) >= 0) && Number(region.width) > 0 && Number(region.height) > 0;
  const regionValue = markRegion && validRegion ? { x: Number(region.x), y: Number(region.y), width: Number(region.width), height: Number(region.height) } : undefined;
  return <>
    <h4>Slide {slide.slideId}</h4>
    {slide.hasImage ? <QualitySlide project={project} datasetId={slide.datasetId} slideId={slide.slideId} featureBundleId={page.featureBundleId ?? undefined} showReview={false} onRegion={(value) => { setMarkRegion(true); setRegion({ x: String(value.x), y: String(value.y), width: String(value.width), height: String(value.height) }); }} /> : <p className="callout">This frozen dataset has no linked image for this slide. Predictions and notes remain available.</p>}
    {attentionModel ? <div className="inline-actions"><button type="button" className="btn btn-secondary" aria-expanded={showAttention} onClick={() => setShowAttention(!showAttention)}>{showAttention ? 'Hide model attention' : 'Inspect model attention'}</button><a className="btn btn-secondary" href={`#interpretation?${new URLSearchParams({ predictor: attentionModel.predictorId, evaluation: attentionModel.evaluationId, slide: slide.slideId, search: slide.slideId })}`}>Open interpretation workspace</a></div> : <p className="muted">Patch attention is unavailable for these model inputs. Original slide review and notes remain available.</p>}
    {showAttention && attentionModel ? <><label className="label">Attention model<select className="field" value={attentionModel.evaluationId} onChange={(event) => setAttentionRun(event.target.value)}>{attentionModels.map((item) => <option key={item.evaluationId} value={item.evaluationId}>{item.name}</option>)}</select></label><ErrorNotice error={attention.error} />
      {attention.isError ? <button type="button" className="btn btn-secondary" onClick={() => void attention.refetch()}>Retry loading attention</button> : attention.isPending ? <p role="status">Checking saved attention…</p> : study ? <SavedAttention project={project} record={study} slideId={slide.slideId} /> : pending ? <p className="callout" role="status">Attention for this slide is {pending.execution?.status === 'running' ? 'being computed' : 'queued'}. It appears here when complete.</p>
        : inference && slide.hasImage ? <div className="callout"><p>No attention map yet for this model and slide. Attention visualizes the predictor&rsquo;s pooling weights; it needs no labels.</p><AttentionRequest key={attentionModel.evaluationId} project={project} evaluationId={attentionModel.evaluationId} slideIds={[slide.slideId]} label="Compute attention for this slide" /></div>
        : <p className="callout">No completed attention map for this model and slide. Open the interpretation workspace to prepare it.</p>}</> : null}
    <details className="case-review-region"><summary>Mark a review region</summary><label><input type="checkbox" checked={markRegion} onChange={(event) => setMarkRegion(event.target.checked)} /> Record a rectangle in full-resolution slide pixels</label>{markRegion ? <div className="case-review-controls">{(['x', 'y', 'width', 'height'] as const).map((key) => <label key={key} className="label">{key}<input className="field" type="number" min={key === 'x' || key === 'y' ? 0 : 1} value={region[key]} onChange={(event) => setRegion((current) => ({ ...current, [key]: event.target.value }))} /></label>)}</div> : null}</details>
    <SlideReviewEditor project={project} datasetId={slide.datasetId} slideId={slide.slideId} evaluationId={page.evaluationId} selectedRegion={regionValue} />
  </>;
}
function SavedAttention({ project, record, slideId }: { project: string; record: Interpretation; slideId: string }) {
  const slide = record.manifest.slides.find((item) => item.slideId === slideId);
  const result = record.execution?.result?.slides?.find((item) => item.slideId === slideId);
  return slide && result ? <AttentionSlideViewer project={project} record={record} slide={slide} result={result} /> : <p>Saved attention is incomplete.</p>;
}
