import type { GallerySlide, InterpretationSource, VisualizeItem, VisualizeSelection } from '../api/interpretation';
import type { ResourcePolicy } from '../api/development';
export function selectedGallerySlides(selected: Map<string, GallerySlide>, item: GallerySlide, checked: boolean) {
  const next = new Map(selected);
  if (checked && item.available && next.size < 128) next.set(item.slidePath, item);
  if (!checked) next.delete(item.slidePath);
  return next;
}
export function representationCompatible(dtype: unknown, expectedDtype: unknown) {
  return typeof dtype === 'string' && typeof expectedDtype === 'string' && dtype === expectedDtype;
}
export function sourceCompatible(source: InterpretationSource, encoderId: unknown, dimensions: unknown, expectedDtype: unknown) {
  return source.current && !source.findings.some((finding) => finding.severity === 'error') && typeof encoderId === 'string' && source.encoderId === encoderId && (typeof dimensions !== 'number' || source.dimensions === dimensions) && (representationCompatible(source.dtype, expectedDtype) || source.packs.some((pack) => representationCompatible(pack.outputDtype, expectedDtype)));
}
/** Prefer the predictor's exact pack, then compatible originals, then a stable matching pack. */
export function defaultRepresentation(source: InterpretationSource | undefined, expectedDtype: unknown, preferredPackId?: string | null): string | null {
  if (!source) return null;
  if (preferredPackId && source.packs.some((pack) => pack.id === preferredPackId && representationCompatible(pack.outputDtype, expectedDtype))) return preferredPackId;
  if (representationCompatible(source.dtype, expectedDtype)) return '';
  return source.packs.filter((pack) => representationCompatible(pack.outputDtype, expectedDtype)).map((pack) => pack.id).sort()[0] ?? null;
}
export function validInterpretationResources(value: ResourcePolicy) {
  return Number.isInteger(value.cpuThreadsPerRun) && value.cpuThreadsPerRun >= 1 && value.cpuThreadsPerRun <= 256 && Number.isFinite(value.ramGbPerRun) && value.ramGbPerRun > 0 && value.gpuIds.length <= 1 && value.gpuIds.every((id) => Number.isInteger(id) && id >= 0 && id <= 127);
}
export function visualizationRequest(selection: VisualizeSelection, operationId: string) {
  return { selection: structuredClone({ ...selection, slidePaths: [...new Set(selection.slidePaths)].sort() }), operationId };
}
export function mergeVisualizationItems(previous: VisualizeItem[], incoming: VisualizeItem[]) {
  const items = new Map(previous.map((item) => [item.slidePath, item]));
  for (const item of incoming) items.set(item.slidePath, item);
  return [...items.values()];
}
