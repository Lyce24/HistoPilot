import type { GallerySlide, Interpretation, InterpretationExecution, VisualizeItem, VisualizeSelection } from '../api/interpretation';
import type { ResourcePolicy } from '../api/development';
export type InterpretationStage = 'select' | 'compute' | 'results' | 'viewer';
export interface ResourceDraft { device: 'cpu' | 'gpu'; gpu: string; threads: string; ram: string; width: string; height: string }
export interface InterpretationBatch { id: string; selected: GallerySlide[]; request: VisualizeSelection | null; items: VisualizeItem[]; pending: { selection: VisualizeSelection; operationId: string } | null; uncertain?: boolean; error?: string | null }
export interface InterpretationWizardDraft {
  version: 1; predictorId: string; bundleId: string | null; packChoice: string | null; evaluationId: string; clinicalId: string;
  selected: GallerySlide[]; search: string; offset: number; resources: ResourceDraft; batch: InterpretationBatch | null;
}
export const defaultResourceDraft = (): ResourceDraft => ({ device: 'cpu', gpu: '0', threads: '4', ram: '8', width: '', height: '' });
export function wizardStage(parameters: URLSearchParams): InterpretationStage {
  const value = parameters.get('stage');
  if (value === 'compute' || value === 'results' || value === 'viewer') return value;
  return value === 'select' ? 'select' : parameters.get('interpretation') ? 'viewer' : 'select';
}
export function initialWizardDraft(parameters: URLSearchParams): InterpretationWizardDraft {
  return { version: 1, predictorId: parameters.get('predictor') ?? '', bundleId: null, packChoice: null, evaluationId: parameters.get('evaluation') ?? '', clinicalId: parameters.get('clinical') ?? '', selected: [], search: (parameters.get('search') ?? '').slice(0, 200), offset: 0, resources: defaultResourceDraft(), batch: null };
}
const storageKey = (project: string) => `histopilot:interpretation-wizard:v1:${project}`;
const objectValue = (value: unknown): value is Record<string, unknown> => Boolean(value && typeof value === 'object' && !Array.isArray(value));
const nonemptyString = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
function validSlides(value: unknown): value is GallerySlide[] {
  return Array.isArray(value) && value.length <= 128 && value.every((slide) => objectValue(slide) && nonemptyString(slide.slidePath) && nonemptyString(slide.slideId) && typeof slide.name === 'string' && typeof slide.relativePath === 'string' && typeof slide.available === 'boolean');
}
function validSelection(value: unknown): value is VisualizeSelection {
  return objectValue(value) && nonemptyString(value.predictorId) && nonemptyString(value.featureBundleId) && typeof value.slideFolder === 'string'
    && Array.isArray(value.slidePaths) && value.slidePaths.length > 0 && value.slidePaths.length <= 128 && value.slidePaths.every(nonemptyString);
}
function validBatch(value: unknown): value is InterpretationBatch {
  if (!objectValue(value) || !nonemptyString(value.id) || !validSlides(value.selected) || !Array.isArray(value.items) || value.items.length > 128) return false;
  if (value.items.some((item) => !objectValue(item) || !nonemptyString(item.slidePath) || typeof item.slideId !== 'string' || typeof item.status !== 'string' || typeof item.reused !== 'boolean' || (item.interpretationId != null && typeof item.interpretationId !== 'string') || (item.error != null && (!objectValue(item.error) || typeof item.error.message !== 'string')))) return false;
  if (value.request != null && !validSelection(value.request)) return false;
  return value.pending == null || (objectValue(value.pending) && nonemptyString(value.pending.operationId) && validSelection(value.pending.selection));
}
export function restoreWizardDraft(project: string, parameters: URLSearchParams): InterpretationWizardDraft {
  const fallback = initialWizardDraft(parameters);
  try {
    const raw = typeof window === 'undefined' ? null : window.sessionStorage?.getItem(storageKey(project));
    if (!raw || raw.length > 2_000_000) return fallback;
    const value: unknown = JSON.parse(raw);
    if (!objectValue(value) || value.version !== 1 || typeof value.predictorId !== 'string' || !validSlides(value.selected) || typeof value.search !== 'string' || !Number.isInteger(value.offset) || (value.offset as number) < 0 || (value.batch != null && !validBatch(value.batch))) return fallback;
    const storedResources = objectValue(value.resources) ? value.resources : {};
    const resources = defaultResourceDraft();
    if (storedResources.device === 'gpu') resources.device = 'gpu';
    for (const key of ['gpu', 'threads', 'ram', 'width', 'height'] as const) if (typeof storedResources[key] === 'string') resources[key] = storedResources[key];
    const restored: InterpretationWizardDraft = { ...fallback, predictorId: value.predictorId, selected: value.selected, search: value.search, offset: value.offset as number, resources, batch: value.batch as InterpretationBatch | null ?? null,
      bundleId: typeof value.bundleId === 'string' ? value.bundleId : null, packChoice: typeof value.packChoice === 'string' ? value.packChoice : null,
      evaluationId: typeof value.evaluationId === 'string' ? value.evaluationId : '', clinicalId: typeof value.clinicalId === 'string' ? value.clinicalId : '' };
    if (restored.batch?.pending) restored.batch.uncertain = true;
    else if ((fallback.predictorId && fallback.predictorId !== restored.predictorId) || (fallback.evaluationId && fallback.evaluationId !== restored.evaluationId) || (fallback.clinicalId && fallback.clinicalId !== restored.clinicalId)) return fallback;
    else if (parameters.has('search') && fallback.search !== restored.search) return { ...restored, search: fallback.search, offset: 0 };
    return restored;
  } catch { return fallback; }
}
export function persistWizardDraft(project: string, draft: InterpretationWizardDraft) {
  try {
    const compact = { ...draft, batch: draft.batch ? { ...draft.batch, items: draft.batch.items.map(({ interpretation: _record, ...item }) => item) } : null };
    window.sessionStorage?.setItem(storageKey(project), JSON.stringify(compact));
  } catch { /* The in-memory wizard remains usable when storage is full or disabled. */ }
}
export function wizardRoute(parameters: URLSearchParams, stage: InterpretationStage, context: { predictorId: string; evaluationId: string; clinicalId: string }, studyId?: string, slideId?: string) {
  const next = new URLSearchParams(parameters);
  next.set('stage', stage);
  for (const [key, value] of [['predictor', context.predictorId], ['evaluation', context.evaluationId], ['clinical', context.clinicalId]] as const) if (value) next.set(key, value); else next.delete(key);
  if (studyId) next.set('interpretation', studyId); else next.delete('interpretation');
  if (slideId) next.set('slide', slideId); else next.delete('slide');
  return `#interpretation?${next}`;
}
/** Saved evidence owns its source; legacy manual slides are kept out of a new gallery selection. */
export function adoptSavedInterpretation(current: InterpretationWizardDraft, record: Interpretation): InterpretationWizardDraft {
  if (current.batch?.pending || current.batch?.items.some((item) => item.interpretationId === record.id)) return current;
  const selected: GallerySlide[] = record.manifest.slides.map((slide) => ({ slideId: slide.slideId, slidePath: slide.slidePath, name: slide.slideId, relativePath: slide.slidePath.split('/').at(-1) ?? slide.slideId, available: true, reason: null, patchCount: slide.patchCount }));
  const bundleId = nonemptyString(record.manifest.featureBundleId) ? record.manifest.featureBundleId : null;
  const packChoice = bundleId ? nonemptyString(record.manifest.packArtifactId) ? record.manifest.packArtifactId : '' : null;
  const selection = record.manifest.selection;
  const inputs = objectValue(selection) && Array.isArray(selection.slides) ? selection.slides : [];
  const first = inputs[0];
  const commonOverride = objectValue(first) && typeof first.patchWidthLevel0 === 'number' && Number.isFinite(first.patchWidthLevel0) && first.patchWidthLevel0 > 0
    && typeof first.patchHeightLevel0 === 'number' && Number.isFinite(first.patchHeightLevel0) && first.patchHeightLevel0 > 0
    && inputs.every((input) => objectValue(input) && input.patchWidthLevel0 === first.patchWidthLevel0 && input.patchHeightLevel0 === first.patchHeightLevel0);
  return { ...current, predictorId: record.manifest.predictorId, evaluationId: record.manifest.evaluationId ?? '', clinicalId: record.manifest.clinicalAnalysisId ?? '', bundleId, packChoice,
    selected: bundleId ? selected : [], search: '', offset: 0,
    resources: { ...current.resources, width: commonOverride ? String(first.patchWidthLevel0) : '', height: commonOverride ? String(first.patchHeightLevel0) : '' },
    batch: { id: `saved-${record.id}`, selected, request: null, pending: null, items: selected.map((slide) => ({ slidePath: slide.slidePath, slideId: slide.slideId, interpretationId: record.id, interpretation: record, status: record.execution?.status ?? 'not_started', reused: true })) } };
}
/** Keep selection edits and their URL in the same reloadable context. */
export function replaceWizardContext(project: string, draft: InterpretationWizardDraft, parameters: URLSearchParams, stage: InterpretationStage, studyId?: string, slideId?: string) {
  persistWizardDraft(project, draft);
  const route = wizardRoute(parameters, stage, draft, studyId, slideId);
  window.history.replaceState(window.history.state, '', route);
  window.dispatchEvent(new Event('hashchange'));
}
export function wizardResources(draft: ResourceDraft): ResourcePolicy {
  return { maxConcurrentRuns: 1, gpuIds: draft.device === 'gpu' ? [draft.gpu.trim() ? Number(draft.gpu) : NaN] : [], runsPerGpu: 1, cpuThreadsPerRun: Number(draft.threads), dataLoaderWorkers: 0, ramGbPerRun: Number(draft.ram) };
}
export interface SelectedStudyState { item?: VisualizeItem; execution?: InterpretationExecution; error?: Error | null; hasRecord: boolean }
/** Only the requested selection determines readiness; cached extras and missing/failed jobs cannot advance. */
export function selectedBatchReady(selected: GallerySlide[], states: Map<string, SelectedStudyState>) {
  return selected.length > 0 && selected.every((slide) => { const state = states.get(slide.slidePath); return Boolean(state?.hasRecord && !state.error && state.execution?.status === 'completed' && state.execution.result?.slides?.some((item) => item.slideId === slide.slideId)); });
}
