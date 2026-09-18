/**
 * What each supported architecture can do, mirroring the service's catalog.
 *
 * `modelCatalog.json` is generated from `histopilot/models/catalog.py` and is
 * checked against it by `tests/test_model_catalog_fixture.py`, so the browser
 * and the service cannot disagree about which models exist, which read patch
 * bags and which have attention to show.
 */
import catalog from './modelCatalog.json';

export type FeatureKind = 'patch' | 'slide';

export interface ModelSpec {
  name: string;
  label: string;
  summary: string;
  featureKind: FeatureKind;
  supportsAttention: boolean;
  structuredOutput: boolean;
  honorsCheckpointSelection: boolean;
  windowedSampling: boolean;
  optionPrefix: string | null;
}

export const MODELS = catalog as ModelSpec[];

export const modelSpec = (model?: string): ModelSpec | undefined =>
  MODELS.find((item) => item.name === model?.toLowerCase());

/**
 * Architectures that can read the extraction output a bundle holds. With no
 * resolved bundle yet, offer every architecture rather than silently hiding the
 * ones the eventual bundle might need; the service rejects a real mismatch.
 */
export const modelsForFeatureKind = (featureKind?: FeatureKind): ModelSpec[] =>
  featureKind ? MODELS.filter((item) => item.featureKind === featureKind) : MODELS;

export const modelLabel = (model?: string): string => modelSpec(model)?.label ?? model ?? '';

export const featureKindOf = (model?: string): FeatureKind => modelSpec(model)?.featureKind ?? 'patch';

export const usesPatchFeatures = (model?: string, inputMode?: string): boolean =>
  inputMode !== 'clinical' && featureKindOf(model) === 'patch';

/** Clinical-only predictors have no image attention whatever their architecture. */
export const supportsAttention = (model?: string, inputMode?: string) =>
  inputMode !== 'clinical' && Boolean(modelSpec(model)?.supportsAttention);
