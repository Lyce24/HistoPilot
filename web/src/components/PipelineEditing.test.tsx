import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { defaultRecipe, nnmilRecipe, type CandidateResult, type FrozenBatch } from '../api/development';
import { defaultPatientAnalysis } from '../api/statistics';
import { EditingRecipeFields } from './DevelopmentBatches';
import { applyNnMILPaperOptimizer } from './NnMILRecipeFields';
import { candidateDisplayMetrics, OOFPredictionDownloads, ResultsTable } from './DevelopmentExecution';
import PredictionTargetEditor from './PredictionTargetEditor';
import { newDatasetImportSpec } from '../pages/LocalDataset';
import type { Workspace } from '../api/types';

describe('pipeline editing and scoring contracts', () => {
  it('keeps metadata-only imports eligible for existing feature bundles without a slide folder', () => {
    const spec = newDatasetImportSpec({ sources: [] } as unknown as Workspace);
    expect(spec.slideRoot).toBeUndefined();
    expect(spec.includeMissingSlides).toBe(true);
    expect(spec.patientIdFallback).toBe('unresolved');
  });
  it.each(['paper optimizer', 'patient analysis'])('invalidates review and enables recovery for the %s button without a native change event', (action) => {
    const recipe = { ...nnmilRecipe(), learningRate: 1e-4, analysis: null };
    const state: { dirty: boolean; preview: string | null } = { dirty: false, preview: 'reviewed-plan' };
    const changed = vi.fn();
    const editor = EditingRecipeFields({ value: recipe, onChange: changed,
      onEdit: () => { state.dirty = true; state.preview = null; } });
    const next = action === 'paper optimizer' ? applyNnMILPaperOptimizer(recipe) : { ...recipe, analysis: defaultPatientAnalysis() };
    // Both recipe buttons invoke this callback directly; clicks do not bubble a form change.
    editor.props.onChange(next);
    expect(state).toEqual({ dirty: true, preview: null });
    expect(changed).toHaveBeenCalledExactlyOnceWith(next);
    expect(recipe.analysis).toBeNull();
    expect(recipe.learningRate).toBe(1e-4);
  });

  const candidate: CandidateResult = { candidateId: 'c1', trainingSeed: 42, splitSeed: 42, complete: true,
    completedRuns: 5, totalRuns: 5, metrics: { available: true, auroc: 0.91 },
    metricDetails: { unit: 'patient', classOrder: ['no', 'yes'], positiveClass: 'yes', patientAggregation: 'mean_probabilities',
      selected: { available: true, auroc: 0.91 }, patient: { available: true, auroc: 0.91, count: 50 },
      slide: { available: true, auroc: 0.72, count: 80 } } };

  it('uses the requested scoring unit without substituting primary scores or exposing partial folds', () => {
    expect(candidateDisplayMetrics(candidate, 'patient')?.auroc).toBe(0.91);
    expect(candidateDisplayMetrics(candidate, 'slide')?.auroc).toBe(0.72);
    expect(candidateDisplayMetrics({ ...candidate, metricDetails: undefined }, 'patient')).toBeNull();
    for (const unit of ['selected', 'patient', 'slide'] as const) expect(candidateDisplayMetrics({ ...candidate, complete: false }, unit)).toBeNull();
  });

  it('offers both patient and slide OOF results while retaining the frozen selection policy and complete exports', () => {
    const batch = { manifest: { spec: { batchName: 'nnMIL' }, configurations: [{ id: 'c1', number: 1, recipe: { ...nnmilRecipe(), nnmilCheckpointSelection: 'latest' } }] } } as FrozenBatch;
    const html = renderToStaticMarkup(<ResultsTable batch={batch} loading={false} results={{ status: 'completed', oof: [], candidates: [candidate] }} />);
    for (const text of ['OOF metrics by prediction unit', '<option value="patient">Patient', '<option value="slide">Slide', 'frozen evaluation policy', 'both scoring units', 'Export results']) expect(html).toContain(text);
    expect(html).not.toContain('Validation selects each run');
    expect(html).toContain('frozen scoring unit');
    expect(html).toContain('Patient-level intervals require verified patient IDs');
    expect(html).not.toContain('slide-level metrics are secondary');
    expect(html).toContain('0.910');
  });

  it('offers patient and slide prediction exports only for completed fold groups', () => {
    const html = renderToStaticMarkup(<OOFPredictionDownloads project="p" batchId="b" candidate={candidate} />);
    expect(html).toContain('Download patient OOF predictions');
    expect(html).toContain('Download slide OOF predictions');
    expect(renderToStaticMarkup(<OOFPredictionDownloads project="p" batchId="b" candidate={{ ...candidate, complete: false }} />)).toBe('');
  });

  it('explains slide training and prevents patient targets from implying Slide ID fallback is sufficient', () => {
    const client = new QueryClient();
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><PredictionTargetEditor
        target={{ field: '', unit: 'patient', task: '', classes: [], labels: {}, missing: 'exclude', unmapped: 'exclude' }}
        fieldContext={{ project: 'p', datasetId: '', dictionary: [] }} unlinkedSlideCount={2} fallbackSlideCount={3}
        labelValues={{ isPending: false, error: null }} rawValues={[]} dataLabel="eligible records"
        onChooseTarget={() => {}} onChange={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Training uses individual slides');
      expect(html).toContain('including validation');
      expect(html).toContain('Patient-level analysis requires verified patient IDs');
      expect(html).toContain('3 slides use Slide ID fallback');
      expect(html).not.toContain('or explicitly confirm Slide ID');
    } finally { client.clear(); }
    expect(defaultRecipe()).toMatchObject({ model: 'abmil', checkpointMetric: 'validation_auroc' });
  });
});
