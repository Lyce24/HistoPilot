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

/** A shared architecture setting; each model lists the ones it reads. */
export type SharedField = 'embedDim' | 'attentionDim' | 'numFcLayers' | 'gatedAttention' | 'dropout' | 'inputDropout';

/** One setting an architecture owns, named with its option prefix (for example `transmilLayers`). */
export interface ModelOption {
  key: string;
  label: string;
  /** `text` is a path or short string; its `maximum` is the length limit. */
  kind: 'integer' | 'number' | 'boolean' | 'choice' | 'text';
  default: number | boolean | string;
  help: string;
  minimum: number | null;
  maximum: number | null;
  maximumExclusive: boolean;
  choices: { value: string; label: string }[];
  /** Changes tensor shapes, so the Task Center keys measured memory on it. */
  memory: boolean;
  /** Read only while another option of the same model has one of these values. */
  enabledWhen: { key: string; values: (string | number | boolean)[] } | null;
}

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
  reads: SharedField[];
  /** Shared fields read only while one of the model's options has one of these values. */
  readsEnabledWhen: Partial<Record<SharedField, { key: string; values: (string | number | boolean)[] }>>;
  options: ModelOption[];
  /** Fitting adds the architecture's own loss terms to the slide loss. */
  auxiliaryLoss: boolean;
  /** Needs one logit per class, so one-logit BCE is unavailable. */
  requiresClassLogits: boolean;
  attentionNote: string | null;
  reference: string | null;
}

export const MODELS = catalog as ModelSpec[];

export const modelSpec = (model?: string): ModelSpec | undefined =>
  MODELS.find((item) => item.name === model?.toLowerCase());

/** Every catalogued option once; CLAM-SB and CLAM-MB share theirs. */
export const MODEL_OPTIONS: ModelOption[] = [...new Map(MODELS.flatMap((item) => item.options).map((option) => [option.key, option])).values()];

export const modelOptions = (model?: string): ModelOption[] => modelSpec(model)?.options ?? [];

/** An architecture missing from this catalog reads the patch projection, as pooling does. */
const UNKNOWN_READS: SharedField[] = ['embedDim', 'numFcLayers', 'dropout', 'inputDropout'];

export const readsField = (model: string | undefined, field: SharedField): boolean =>
  (modelSpec(model)?.reads ?? UNKNOWN_READS).includes(field);

/** False when the model reads a shared field only under an option this recipe leaves off. */
export function sharedFieldEnabled(recipe: { model: string } & object, field: SharedField): boolean {
  const condition = modelSpec(recipe.model)?.readsEnabledWhen?.[field];
  if (!condition) return true;
  const option = modelOptions(recipe.model).find((item) => item.key === condition.key);
  const value = (recipe as Record<string, unknown>)[condition.key] ?? option?.default;
  return condition.values.includes(value as string | number | boolean);
}

/**
 * The recipe for a newly chosen model: other architectures' options that the recipe sets
 * return to their defaults, and options it never set stay unset. Mirrors
 * `resolvers.with_arm_model` in the service.
 */
export function withoutForeignOptions<T extends object>(recipe: T, model: string): T {
  const own = new Set(modelOptions(model).map((option) => option.key));
  const reset = Object.fromEntries(MODEL_OPTIONS
    .filter((option) => !own.has(option.key) && option.key in recipe)
    .map((option) => [option.key, option.default]));
  return { ...recipe, ...reset };
}

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
