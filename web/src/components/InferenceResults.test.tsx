import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { CasePage, ReviewedCase } from '../api/caseReview';
import type { InferenceSummary } from '../api/inference';
import { attentionSlides, CaseDetail, MAX_ATTENTION_SLIDES } from './CaseReviewWorkspace';
import { ComparisonMatrix, CompositionTable, Histogram, PredictedClassBars, ThresholdTable } from './InferenceCharts';
import { DistributionFigure, InferenceOverview, marginNote } from './InferenceResults';

const edges = Array.from({ length: 21 }, (_, index) => index / 20);
const bins = (entries: Record<number, number>) => Array.from({ length: 20 }, (_, index) => entries[index] ?? 0);
const classes = ['ND', 'IND', 'LG', 'HG'];

function summary(extra: Partial<InferenceSummary> = {}): InferenceSummary {
  const quantiles = { p10: 0.4, p25: 0.5, median: 0.62, p75: 0.8, p90: 0.9 };
  return {
    evaluationId: 'run', name: 'Ensemble', purpose: 'inference', predictorId: 'predictor', cohortId: 'cohort',
    unit: 'slide', task: 'multiclass_classification', classOrder: classes, positiveClass: null, decisionThreshold: 0.5,
    patientAggregation: 'mean', patients: 80, source: { predictionsSha256: 'a'.repeat(64) }, count: 120,
    predicted: [
      { label: 'ND', count: 60, fraction: 60 / 120, meanConfidence: 0.8 }, { label: 'IND', count: 5, fraction: 5 / 120, meanConfidence: 0.45 },
      { label: 'LG', count: 20, fraction: 20 / 120, meanConfidence: 0.6 }, { label: 'HG', count: 35, fraction: 35 / 120, meanConfidence: 0.7 },
    ],
    confidence: { mean: 0.64, quantiles, edges, counts: { ND: bins({ 16: 55, 9: 5 }), IND: bins({ 8: 5 }), LG: bins({ 11: 20 }), HG: bins({ 14: 35 }) } },
    margin: { mean: 0.3, quantiles, edges, counts: { ND: bins({ 10: 60 }), IND: bins({ 1: 5 }), LG: bins({ 3: 20 }), HG: bins({ 6: 35 }) } },
    ensemble: { memberCount: 5, records: 120, unanimous: 90, disagreements: 30, meanSpread: 0.07, agreement: [5, 4, 3, 2, 1, 0].map((agree) => ({ agree, count: agree === 5 ? 90 : agree === 4 ? 20 : agree === 3 ? 10 : 0 })) },
    development: { comparable: true, patients: 30, records: 50, shared: { ND: 25, IND: 2, LG: 8, HG: 15 }, new: { ND: 35, IND: 3, LG: 12, HG: 20 } },
    attributes: [{ key: 'site', label: 'site' }],
    ...extra,
  };
}

describe('inference results', () => {
  it('describes predictions, confidence, agreement and development patients without metrics', () => {
    const html = renderToStaticMarkup(<InferenceOverview summary={summary()} />);
    expect(html).toContain('Slides predicted');
    expect(html).toContain('120');
    expect(html).toContain('Fold members disagree');
    expect(html).toContain('From development patients');
    expect(html).toContain('New patients');
    expect(html).toContain('Seen in development');
    expect(html).toContain('Confidence by predicted class');
    expect(html).toContain('Decision margin by predicted class');
    // Margin below 0.2 by default: bins 0-3 hold 5 IND and 20 LG slides.
    expect(html).toContain('<strong>25</strong>');
    expect(html).toContain('near a decision boundary');
    for (const metric of ['AUROC', 'AUPRC', 'Accuracy', 'accuracy', 'F1', 'Log loss', 'Actual']) expect(html).not.toContain(metric);
  });

  it('shows the binary threshold histogram and marks the frozen threshold', () => {
    const binary = summary({
      classOrder: ['benign', 'tumor'], positiveClass: 'tumor', decisionThreshold: 0.35, ensemble: undefined, development: { comparable: false },
      predicted: [{ label: 'benign', count: 3, fraction: 0.6, meanConfidence: 0.8 }, { label: 'tumor', count: 2, fraction: 0.4, meanConfidence: 0.7 }],
      confidence: { ...summary().confidence, counts: { benign: bins({ 16: 3 }), tumor: bins({ 14: 2 }) } },
      binary: { positiveClass: 'tumor', threshold: 0.35, positiveProbability: { edges, counts: bins({ 3: 3, 14: 2 }) },
        sweep: [0.1, 0.2, 0.3, 0.35, 0.4, 0.5].map((threshold) => ({ threshold, positive: threshold <= 0.35 ? 5 : 2 })), nearThreshold: [{ band: 0.05, count: 0 }, { band: 0.1, count: 1 }] },
      count: 5,
    });
    const html = renderToStaticMarkup(<InferenceOverview summary={binary} />);
    expect(html).toContain('Probability of tumor');
    expect(html).toContain('threshold 0.35');
    expect(html).toContain('0.35 · frozen');
    expect(html).toContain('Member probabilities were not recorded for this run');
    expect(renderToStaticMarkup(<InferenceOverview summary={binary} memberNote="Single model: a refit has no fold members" />)).toContain('a refit has no fold members');
    expect(html).not.toContain('From development patients');
    const table = renderToStaticMarkup(<ThresholdTable binary={binary.binary!} total={5} />);
    expect(table.match(/<tr class="is-frozen">/g)).toHaveLength(1);
    // Binary targets can switch to the margin, which runs from the frozen threshold.
    expect(html).toContain('aria-label="Distribution to show"');
    expect(html).toContain('aria-pressed="true">P(tumor)</button>');
    expect(html).toContain('aria-pressed="false">Decision margin</button>');
    const margin = renderToStaticMarkup(<DistributionFigure summary={binary} view="margin" onViewChange={() => {}} />);
    expect(margin).toContain('Decision margin by predicted class');
    expect(margin).toContain('frozen threshold 0.35');
    expect(margin).toContain('aria-pressed="true">Decision margin</button>');
    expect(margin).not.toContain('Probability of tumor');
  });

  it('describes the margin for each kind of target', () => {
    expect(marginNote({ binary: summary({ binary: { positiveClass: 'tumor', threshold: 0.35, positiveProbability: { edges, counts: bins({}) }, sweep: [], nearThreshold: [] } }).binary })).toContain('How far P(tumor) lies from the frozen threshold 0.35');
    expect(marginNote({})).toContain('predicted class minus the runner-up');
    // Non-binary targets show only the margin, without a switch.
    const multiclass = renderToStaticMarkup(<DistributionFigure summary={summary()} view="probability" onViewChange={() => {}} />);
    expect(multiclass).toContain('Decision margin by predicted class');
    expect(multiclass).not.toContain('Distribution to show');
  });

  it('lists every computed threshold row and marks the frozen one', () => {
    const sweep = [...Array.from({ length: 19 }, (_, step) => Math.round(0.05 * (step + 1) * 100) / 100), 0.37].sort((a, b) => a - b).map((threshold) => ({ threshold, positive: Math.round((1 - threshold) * 10) }));
    const table = renderToStaticMarkup(<ThresholdTable binary={{ positiveClass: 'tumor', threshold: 0.37, positiveProbability: { edges, counts: bins({}) }, sweep, nearThreshold: [{ band: 0.05, count: 1 }] }} total={10} />);
    expect(table.match(/<tr[ >]/g)).toHaveLength(21);
    for (const threshold of ['0.05', '0.10', '0.15', '0.35', '0.95']) expect(table).toContain(`<th scope="row">${threshold}</th>`);
    expect(table).toContain('0.37 · frozen');
    expect(table.match(/<tr class="is-frozen">/g)).toHaveLength(1);
  });

  it('draws stacked bins with a table view and labeled class bars', () => {
    const html = renderToStaticMarkup(<Histogram title="Confidence" description="d" edges={edges} xLabel="Confidence" series={[{ label: 'ND', counts: bins({ 10: 3 }) }, { label: 'HG', counts: bins({ 10: 2, 18: 1 }) }]} />);
    // One path per non-empty class segment; the legend and table carry identity and values.
    expect(html.match(/<path /g)).toHaveLength(3);
    expect(html).toContain('Table view');
    expect(html).toContain('0.50–0.55');
    expect(html).toContain('aria-label="HG, Confidence 0.90–0.95: 1"');
    const bars = renderToStaticMarkup(<PredictedClassBars predicted={summary().predicted} unit="slide" />);
    expect(bars).toContain('ND: 60 slides');
    expect(bars.match(/role="listitem"/g)).toHaveLength(4);
  });

  it('compares runs without calling agreement accuracy and shows composition counts', () => {
    const matrix = renderToStaticMarkup(<ComparisonMatrix name="Ensemble" classes={['a', 'b']} comparison={{ evaluationId: 'other', name: 'Refit', predictorId: 'p', decisionThreshold: 0.5, predictionsSha256: 'x', count: 4, agreement: 0.75, kappa: 0.5, disagreements: 1, matrix: [[1, 1], [0, 2]] }} />);
    expect(matrix).toContain('Agreement 75.0%');
    expect(matrix).toContain('Agreement is not accuracy');
    expect(matrix.match(/class="is-agreement"/g)).toHaveLength(2);
    const table = renderToStaticMarkup(<CompositionTable caption="By part" groupLabel="Part" unit="slide" classes={classes} rows={[{ value: 'A', count: 4, counts: { ND: 3, HG: 1 }, meanConfidence: 0.7 }]} />);
    expect(table).toContain('75%');
    expect(table).toContain('70.0%');
  });

  it('shows inference cases as predictions with flags, never as unlabeled outcomes', () => {
    const record = { id: 'CASE1A', patientId: 'p1', slideIds: ['CASE1A'], label: null, labelIndex: null, probabilities: [0.7, 0.1, 0.1, 0.1], predictedIndex: 0, predictedLabel: 'ND', confidence: 0.7, margin: 0.6,
      memberAgreement: { agree: 4, total: 5, spread: 0.05 }, developmentPatient: true, outcome: 'unlabeled', comparison: null, attributes: {}, slides: [] } as ReviewedCase;
    const page = { purpose: 'inference', unit: 'slide', classOrder: classes, name: 'Ensemble', positiveClass: null, attributes: [] } as unknown as CasePage;
    const html = renderToStaticMarkup(<CaseDetail project="project" record={record} page={page} />);
    expect(html).toContain('Predicted <strong>ND</strong>');
    expect(html).toContain('4 of 5 fold members agree');
    expect(html).toContain('Patient seen in development');
    expect(html).not.toContain('Actual label');
  });

  it('bounds attention requests to listed slides that have images', () => {
    const slide = (slideId: string, hasImage = true) => ({ slideId, hasImage }) as ReviewedCase['slides'][number];
    const items = Array.from({ length: 40 }, (_, index) => ({ slides: [slide(`s${index}`), slide(`missing${index}`, false)] }) as ReviewedCase);
    const chosen = attentionSlides(items);
    expect(chosen).toHaveLength(MAX_ATTENTION_SLIDES);
    expect(chosen[0]).toBe('s0');
    expect(chosen.some((id) => id.startsWith('missing'))).toBe(false);
  });
});
