import { useState, type KeyboardEvent, type ReactNode } from 'react';
import type { GallerySlide, Interpretation, InterpretationSlideResult, SlidePrediction, VisualizeItem } from '../api/interpretation';
import { computeActive, computeStatusLabel, type ComputeExecution } from '../api/predictors';
import { completedSlideResult, predictionCounts, slideProgress, type SelectedStudyState, type SlideProgress } from '../lib/interpretationWizard';
import AttentionSlideViewer from './AttentionSlideViewer';
import ComputeJobControls from './ComputeJobControls';
import './VisualQualityExplorer.css';
import './InterpretationReview.css';

const PALETTE = ['#205e63', '#ac4d22', '#715298', '#387636', '#9c3e67', '#3866a3', '#74672d', '#595959'];
const PROGRESS: Record<SlideProgress, string> = { ready: 'Ready', computing: 'Computing attention', failed: 'Needs attention', waiting: 'Waiting to start' };
const percent = (value: number) => `${(value * 100).toFixed(value >= .995 || value < .005 ? 1 : 0)}%`;
const classColor = (label: string, classOrder: readonly string[]) => PALETTE[Math.max(0, classOrder.indexOf(label)) % PALETTE.length];

export function PredictionChip({ prediction, classOrder, large = false }: { prediction: SlidePrediction; classOrder: readonly string[]; large?: boolean }) {
  return <span className={`interpretation-prediction-chip${large ? ' is-large' : ''}`}><i aria-hidden="true" style={{ background: classColor(prediction.predictedLabel, classOrder) }} />
    <strong>{prediction.predictedLabel}</strong><span>{percent(prediction.confidence)}</span></span>;
}

/** Class probabilities in frozen class order, with the model's decision highlighted. */
export function PredictionDetail({ result, classOrder, memberCount }: { result: InterpretationSlideResult; classOrder: readonly string[]; memberCount: number }) {
  const prediction = result.prediction;
  const agreement = prediction?.memberAgreement;
  return <div className="interpretation-prediction" aria-label="Predicted label and class probabilities">
    <div className="interpretation-prediction-bars">{classOrder.map((label, index) => {
      const value = result.probabilities[index] ?? 0;
      const chosen = prediction?.predictedIndex === index;
      return <div key={label} className={`interpretation-prediction-bar${chosen ? ' is-predicted' : ''}`}>
        <span>{label}</span><span className="bar" aria-hidden="true"><i style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: classColor(label, classOrder) }} /></span><span>{percent(value)}</span>
      </div>;
    })}</div>
    <p className="muted">{prediction ? `Predicted ${prediction.predictedLabel} · decision margin ${prediction.margin.toFixed(2)}` : 'Prediction pending.'}
      {agreement ? ` · ${agreement.agree} of ${agreement.total} fold models agree` : memberCount > 1 ? '' : ' · single refit model'}.</p>
  </div>;
}

interface ReviewProps {
  project: string;
  selected: GallerySlide[];
  items: VisualizeItem[];
  states: Map<string, SelectedStudyState>;
  records: Map<string, Interpretation>;
  classOrder: readonly string[];
  activeSlideId?: string;
  busy: boolean;
  canControlJobs: boolean;
  actions?: ReactNode;
  onSelect: (slide: GallerySlide) => void;
}

/** Datasets-style visual review: a slide list beside the selected slide's attention and prediction. */
export default function InterpretationReview({ project, selected, items, states, records, classOrder, activeSlideId, busy, canControlJobs, actions, onSelect }: ReviewProps) {
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const progress = new Map(selected.map((slide) => [slide.slidePath, slideProgress(slide, states.get(slide.slidePath))]));
  const results = new Map(selected.map((slide) => [slide.slidePath, completedSlideResult(slide, states.get(slide.slidePath))]));
  const counts = predictionCounts(selected, states, classOrder);
  const tally = (value: SlideProgress) => selected.filter((slide) => progress.get(slide.slidePath) === value).length;
  const ready = tally('ready'), computing = tally('computing'), failed = tally('failed'), waiting = tally('waiting');
  const search = query.trim().toLocaleLowerCase();
  const visible = selected.filter((slide) => {
    if (search && !`${slide.name} ${slide.slideId}`.toLocaleLowerCase().includes(search)) return false;
    if (filter === 'pending') return progress.get(slide.slidePath) !== 'ready';
    if (filter === 'failed') return progress.get(slide.slidePath) === 'failed';
    if (filter.startsWith('label:')) return results.get(slide.slidePath)?.prediction?.predictedLabel === filter.slice(6);
    return true;
  });
  // Open the requested slide, else the first ready one, so review starts before every job finishes.
  const active = selected.find((slide) => slide.slideId === activeSlideId)
    ?? visible.find((slide) => progress.get(slide.slidePath) === 'ready') ?? visible[0] ?? selected[0];
  const position = active ? visible.findIndex((slide) => slide.slidePath === active.slidePath) : -1;
  function step(offset: number) {
    const next = visible[position < 0 ? 0 : position + offset];
    if (next) onSelect(next);
    return next;
  }
  function listKey(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    const next = step(event.key === 'ArrowDown' ? 1 : -1);
    if (next) event.currentTarget.querySelector<HTMLButtonElement>(`[data-slide-path="${CSS.escape(next.slidePath)}"]`)?.focus();
  }
  const activeState = active ? states.get(active.slidePath) : undefined;
  const activeItem = active ? items.find((item) => item.slidePath === active.slidePath) : undefined;
  const activeRecord = activeItem?.interpretationId ? records.get(activeItem.interpretationId) : undefined;
  return <div className="interpretation-review">
    <div className="interpretation-review-summary" role="status">
      <span><strong>{ready}</strong> of {selected.length} slides ready</span>
      {computing ? <span>{computing} computing</span> : null}
      {waiting ? <span>{busy ? `${waiting} being checked` : `${waiting} waiting`}</span> : null}
      {failed ? <span className="is-warning">{failed} need attention</span> : null}
      {counts.length ? <span className="interpretation-review-counts">Predicted {counts.map(([label, count]) => <span key={label} className="interpretation-prediction-chip"><i aria-hidden="true" style={{ background: classColor(label, classOrder) }} /><strong>{label}</strong><span>{count}</span></span>)}</span> : null}
      {actions ? <span className="interpretation-review-actions">{actions}</span> : null}
    </div>
    <div className="morphology-workspace interpretation-explorer">
      <aside aria-label="Selected slides">
        <div className="interpretation-review-filters">
          <label className="label">Show<select className="field" value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="all">All slides ({selected.length})</option>
            {counts.map(([label, count]) => <option key={label} value={`label:${label}`}>Predicted {label} ({count})</option>)}
            {ready < selected.length ? <option value="pending">Not ready yet ({selected.length - ready})</option> : null}
            {failed ? <option value="failed">Needs attention ({failed})</option> : null}
          </select></label>
          <label className="label">Find slide<input className="field" type="search" value={query} maxLength={200} onChange={(event) => setQuery(event.target.value)} placeholder="Slide name" /></label>
        </div>
        <div className="morphology-slide-list interpretation-slide-list" role="list" aria-label="Slides in this attention batch" onKeyDown={listKey}>
          {visible.map((slide) => {
            const state = progress.get(slide.slidePath)!;
            const result = results.get(slide.slidePath);
            const isActive = active?.slidePath === slide.slidePath;
            return <div role="listitem" key={slide.slidePath}><button type="button" data-slide-path={slide.slidePath} className={`morphology-slide-button ${isActive ? 'is-selected' : ''}`} aria-pressed={isActive} onClick={() => onSelect(slide)}>
              <strong>{slide.name}</strong>
              {result?.prediction ? <PredictionChip prediction={result.prediction} classOrder={classOrder} /> : <small className={`interpretation-progress-${state}`}>{PROGRESS[state]}</small>}
            </button></div>;
          })}
          {!visible.length ? <p className="muted interpretation-empty">No slides match this view.</p> : null}
        </div>
        <div className="interpretation-review-stepper"><button type="button" className="btn btn-secondary btn-small" aria-label="Previous slide" disabled={position <= 0} onClick={() => step(-1)}>← Prev</button><span>{position >= 0 ? position + 1 : 0} / {visible.length}</span><button type="button" className="btn btn-secondary btn-small" aria-label="Next slide" disabled={position < 0 || position >= visible.length - 1} onClick={() => step(1)}>Next →</button></div>
        <p className="muted interpretation-review-hint">↑ / ↓ move through the list.</p>
      </aside>
      <div>{active ? <SlideAttention key={active.slidePath} project={project} slide={active} state={activeState} item={activeItem} record={activeRecord} classOrder={classOrder} canControlJobs={canControlJobs} /> : <p>No slides are selected.</p>}</div>
    </div>
  </div>;
}

/**
 * A slide's own job controls: its Resume when it stopped short, "Compute slide attention" for
 * a saved record never launched (a job created before the Task Center shows its saved status,
 * read-only). A job in progress is followed by the batch chip.
 */
export const slideJobControlsShown = (execution?: ComputeExecution) => Boolean(execution && ['failed', 'cancelled', 'interrupted', 'not_started'].includes(execution.status));

function SlideAttention({ project, slide, state, item, record, classOrder, canControlJobs }: { project: string; slide: GallerySlide; state?: SelectedStudyState; item?: VisualizeItem; record?: Interpretation; classOrder: readonly string[]; canControlJobs: boolean }) {
  const result = completedSlideResult(slide, state);
  const recordSlide = record?.manifest.slides.find((entry) => entry.slideId === slide.slideId);
  const order = (record?.execution?.result?.classOrder as string[] | undefined) ?? classOrder;
  const execution = state?.execution;
  const error = state?.error?.message ?? (!computeActive(execution) && execution?.status !== 'completed' ? item?.error?.message ?? execution?.error : null);
  return <section className="interpretation-slide-detail" aria-label={`Attention for ${slide.name}`}>
    <header className="interpretation-slide-header">
      <div><h3>{slide.name}</h3>{recordSlide ? <small>{`${recordSlide.patchCount.toLocaleString()} patches · ${recordSlide.width.toLocaleString()} × ${recordSlide.height.toLocaleString()} level-0 pixels`}</small> : slide.relativePath !== slide.name ? <small>{slide.relativePath}</small> : null}</div>
      {result?.prediction ? <PredictionChip prediction={result.prediction} classOrder={order} large /> : null}
    </header>
    {result && record && recordSlide ? <>
      <PredictionDetail result={result} classOrder={order} memberCount={record.manifest.memberCount} />
      <AttentionSlideViewer project={project} record={record} slide={recordSlide} result={result} />
    </> : <div className="interpretation-slide-progress" role="status">
      <p><strong>{result ? 'Loading this slide…' : execution ? computeStatusLabel(execution) : item?.error ? 'Could not start' : 'Waiting for its attention job'}</strong>
        {execution?.progress?.completedPairs !== undefined ? ` · ${execution.progress.completedPairs} / ${execution.progress.totalPairs} checkpoint passes` : ''}</p>
      {error ? <p className="slide-gallery-error">{error}</p> : null}
      {!result ? <p className="muted">Attention, top patches and the predicted label appear here as soon as this slide finishes; other slides can be reviewed meanwhile.</p> : null}
      {canControlJobs && item?.interpretationId && slideJobControlsShown(execution) ? <ComputeJobControls project={project} id={item.interpretationId} kind="interpretation" initial={execution} variant="chip" /> : null}
    </div>}
  </section>;
}
