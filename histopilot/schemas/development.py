"""Resolved training recipes and reproducible development batch intent."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_serializer, model_validator

from histopilot.models import catalog
from histopilot.schemas.analysis import PatientAnalysisSettings
from histopilot.schemas.mil import MILInputSpec
from histopilot.schemas.predictor_policy import ExperimentPredictorPolicy
from histopilot.schemas.version_labels import FreezeVersionLabel
from histopilot.schemas.workspace import RequestModel, Seed

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Epochs = Annotated[StrictInt, Field(ge=1, le=100000)]
Probability = Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)]

# These defaults were implicit in previously frozen recipes. Omit them from the
# serialized recipe so adding controls does not rewrite historical experiment hashes.
EXPERIMENTAL_DEFAULTS = {
    "lossType": "ce", "classWeighting": "none", "classWeights": None,
    "focalGamma": 2.0, "labelSmoothing": 0.0,
    "patientAggregation": "mean_probabilities", "ensembleAggregation": "mean_probability", "adamBetas": (0.9, 0.999),
    "adamEps": 1e-8, "lrStepSize": 10, "lrGamma": 0.5, "lrPlateauPatience": 5,
    "aggregatorLearningRate": None, "headLearningRate": None,
    "samplingStrategy": "slide_uniform", "classWeightedSampling": False,
    "samplingPositivePrevalence": 0.4, "cohortColumn": "cohort",
    "instanceDropout": 0.0, "featureNoiseStd": 0.0, "bagCurriculum": False,
    "bagCurriculumStart": 512, "bagCurriculumEnd": 8000, "bagCurriculumWarmupEpochs": 5,
    "evalBagSize": None, "evalBatchSize": None,
    "minValidationPositives": None, "fixedEpochBudget": None,
    "bagSizeMode": "fixed", "bagSizeFraction": 0.5,
    "nnmilFeatureSampling": True, "nnmilWindowStrideDivisor": 4,
    "nnmilWindowShuffle": True, "nnmilWindowSeed": 42,
    "nnmilWindowAggregation": "mean_logits", "nnmilBatchSampler": "patient_weighted",
    "nnmilCheckpointSelection": "best_validation",
    "weightDecayPolicy": "all", "lrScheduleInterval": "epoch",
    "inputMode": "image", "clinicalFields": [],
}


class ClinicalField(RequestModel):
    field: str = Field(min_length=1, max_length=128)
    kind: Literal["numeric", "categorical"]


class TrainingRecipe(RequestModel):
    model: str = Field(default="abmil", min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    inputMode: Literal["image", "clinical", "multimodal"] = "image"
    clinicalFields: list[ClinicalField] = Field(default_factory=list, max_length=30)
    embedDim: Annotated[StrictInt, Field(ge=1, le=8192)] = 512
    attentionDim: Annotated[StrictInt, Field(ge=1, le=8192)] = 384
    numFcLayers: Annotated[StrictInt, Field(ge=1, le=8)] = 1
    gatedAttention: bool = True
    dropout: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)] = 0.25
    inputDropout: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)] = 0
    gradientCheckpointing: bool = False
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    gradientClipNorm: NonnegativeFloat = 0
    accumulateGradBatches: Annotated[StrictInt, Field(ge=1, le=4096)] = 1
    learningRate: PositiveFloat = 0.0003
    weightDecay: NonnegativeFloat = 0.0001
    weightDecayPolicy: Literal["all", "weights_only"] = "all"
    maxEpochs: Epochs = 40
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lrScheduler: Literal["none", "cosine", "plateau", "step"] = "none"
    lrScheduleInterval: Literal["epoch", "step"] = "epoch"
    adamBetas: tuple[Probability, Probability] = (0.9, 0.999)
    adamEps: PositiveFloat = 1e-8
    aggregatorLearningRate: PositiveFloat | None = None
    headLearningRate: PositiveFloat | None = None
    lrStepSize: Epochs = 10
    lrGamma: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.5
    lrPlateauPatience: Annotated[StrictInt, Field(ge=0, le=100000)] = 5
    warmupEpochs: Annotated[StrictInt, Field(ge=0, le=99999)] = 0
    finalLrFraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.01
    batchSize: Annotated[StrictInt, Field(ge=1, le=4096)] = 1
    bagSize: Annotated[StrictInt, Field(ge=1, le=1000000)] | None = Field(
        default=4096, description="Maximum training patches per slide; null uses the whole bag."
    )
    lossType: Literal["ce", "bce", "focal"] = "ce"
    bagSizeMode: Literal["fixed", "training_median"] = "fixed"
    bagSizeFraction: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.5
    nnmilFeatureSampling: bool = True
    nnmilWindowStrideDivisor: Annotated[StrictInt, Field(ge=1, le=256)] = 4
    nnmilWindowShuffle: bool = True
    nnmilWindowSeed: Seed = 42
    nnmilWindowAggregation: Literal["mean_logits", "mean_probabilities"] = "mean_logits"
    nnmilBatchSampler: Literal[
        "patient_weighted", "class_balanced", "auc_stratified"
    ] = "patient_weighted"
    nnmilCheckpointSelection: Literal["best_validation", "latest"] = "best_validation"
    classWeighting: Literal["none", "inverse_prevalence"] = "none"
    classWeights: list[PositiveFloat] | None = Field(default=None, min_length=2, max_length=50)
    focalGamma: NonnegativeFloat = 2
    labelSmoothing: Probability = 0
    patientAggregation: Literal["mean_probabilities", "mean_logits"] = "mean_probabilities"
    ensembleAggregation: Literal["mean_probability", "mean_logit"] = "mean_probability"
    samplingStrategy: Literal[
        "slide_uniform", "patient_natural", "cohort_balanced", "cohort_label_balanced"
    ] = "slide_uniform"
    classWeightedSampling: bool = False
    samplingPositivePrevalence: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.4
    cohortColumn: str = Field(default="cohort", min_length=1, max_length=128)
    instanceDropout: Probability = 0
    featureNoiseStd: NonnegativeFloat = 0
    bagCurriculum: bool = False
    bagCurriculumStart: Annotated[StrictInt, Field(ge=1, le=1000000)] = 512
    bagCurriculumEnd: Annotated[StrictInt, Field(ge=1, le=1000000)] = 8000
    bagCurriculumWarmupEpochs: Epochs = 5
    evalBagSize: Annotated[StrictInt, Field(ge=1, le=1000000)] | None = None
    evalBatchSize: Annotated[StrictInt, Field(ge=1, le=4096)] | None = None
    earlyStopping: bool = True
    patience: Annotated[StrictInt, Field(ge=1, le=10000)] = 8
    earlyStoppingMinDelta: NonnegativeFloat = 0
    minEpochs: Annotated[StrictInt, Field(ge=0, le=100000)] = 1
    minValidationPositives: Annotated[StrictInt, Field(ge=1, le=1000000)] | None = None
    fixedEpochBudget: Epochs | None = None
    checkpointMetric: Literal["validation_loss", "validation_auroc", "validation_accuracy"] = (
        "validation_auroc"
    )
    analysis: PatientAnalysisSettings | None = Field(default_factory=PatientAnalysisSettings)
    decisionThreshold: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = 0.5

    @model_validator(mode="before")
    @classmethod
    def historical_defaults(cls, values, info):
        if isinstance(values, dict) and (info.context or {}).get("legacy"):
            return {
                "learningRate": 0.0003, "weightDecay": 0.0001,
                "maxEpochs": 100, "patience": 15, **values,
                "checkpointMetric": values.get("checkpointMetric", "validation_loss"),
                "analysis": values.get("analysis"),
                "decisionThreshold": values.get("decisionThreshold"),
            }
        return values

    @model_validator(mode="after")
    def coherent_epoch_budget(self):
        if len({item.field for item in self.clinicalFields}) != len(self.clinicalFields):
            raise ValueError("Clinical fields must be distinct.")
        if self.inputMode != "image" and not self.clinicalFields:
            raise ValueError("Choose explicitly typed clinical fields for clinical or combined modeling.")
        if self.inputMode == "image" and self.clinicalFields:
            raise ValueError("Image-only models do not consume clinical fields.")
        if self.inputMode == "clinical" and self.model != "abmil":
            raise ValueError("Clinical-only uses a linear clinical head; retain the default model setting.")
        if self.minEpochs > self.maxEpochs:
            raise ValueError("Minimum epochs cannot exceed maximum epochs.")
        if self.warmupEpochs >= self.maxEpochs:
            raise ValueError("Warmup epochs must be fewer than maximum epochs.")
        if self.warmupEpochs and self.lrScheduler != "cosine":
            raise ValueError("Warmup epochs require the cosine learning-rate schedule.")
        if self.lrScheduleInterval == "step" and self.lrScheduler != "cosine":
            raise ValueError("Optimizer-step scheduling requires the cosine schedule.")
        if self.bagSizeMode == "training_median" and self.bagCurriculum:
            raise ValueError("Choose automatic training bags or a bag curriculum.")
        if not catalog.owns_options(self.model, "nnmil") and (
            self.nnmilBatchSampler != "patient_weighted"
            or self.nnmilCheckpointSelection != "best_validation"
        ):
            raise ValueError("nnMIL batch samplers and checkpoint selection require nnMIL.")
        if self.nnmilBatchSampler != "patient_weighted" and (
            self.samplingStrategy != "slide_uniform" or self.classWeightedSampling
        ):
            raise ValueError("Choose an nnMIL batch sampler or the existing slide/patient sampler.")
        if self.minValidationPositives is not None and self.fixedEpochBudget is None:
            raise ValueError("A validation-positive threshold requires an explicit fixed epoch budget.")
        if self.fixedEpochBudget is not None and not (
            self.minEpochs <= self.fixedEpochBudget <= self.maxEpochs
            and self.warmupEpochs < self.fixedEpochBudget
        ):
            raise ValueError(
                "The fixed epoch budget must be between minimum and maximum epochs, after warmup."
            )
        if self.classWeights is not None and self.classWeighting != "none":
            raise ValueError("Choose explicit class weights or inverse-prevalence weighting.")
        if self.labelSmoothing and self.lossType != "ce":
            raise ValueError("Label smoothing is supported only with cross entropy.")
        if self.classWeightedSampling and self.samplingStrategy != "slide_uniform":
            raise ValueError("Class-weighted sampling requires the slide-uniform sampling strategy.")
        if self.bagCurriculum and self.bagCurriculumStart > self.bagCurriculumEnd:
            raise ValueError("The bag curriculum must start at or below its final bag size.")
        return self

    @model_serializer(mode="wrap")
    def retain_legacy_shape(self, handler):
        values = handler(self)
        for key, default in EXPERIMENTAL_DEFAULTS.items():
            if getattr(self, key) == default:
                values.pop(key, None)
        for key in ("analysis", "decisionThreshold"):
            if getattr(self, key) is None:
                values.pop(key, None)
        return values


class SearchGrid(RequestModel):
    learningRates: list[PositiveFloat] = Field(
        default_factory=lambda: [0.0003], min_length=1, max_length=100
    )
    weightDecays: list[NonnegativeFloat] = Field(
        default_factory=lambda: [0.0001], min_length=1, max_length=100
    )
    maxEpochs: list[Epochs] = Field(default_factory=lambda: [40], min_length=1, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def historical_defaults(cls, values, info):
        if isinstance(values, dict) and (info.context or {}).get("legacy"):
            return {"learningRates": [0.0003], "weightDecays": [0.0001], "maxEpochs": [100], **values}
        return values

    @field_validator("learningRates", "weightDecays", "maxEpochs")
    @classmethod
    def unique_values(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Sweep values must be distinct.")
        return value


class ResourcePolicy(RequestModel):
    maxConcurrentRuns: Annotated[StrictInt, Field(ge=1, le=128)] = 1
    gpuIds: list[Annotated[StrictInt, Field(ge=0, le=127)]] = Field(
        default_factory=lambda: [0], max_length=128
    )
    runsPerGpu: Annotated[StrictInt, Field(ge=1, le=16)] = 1
    cpuThreadsPerRun: Annotated[StrictInt, Field(ge=1, le=256)] = 2
    dataLoaderWorkers: Annotated[StrictInt, Field(ge=0, le=64)] = 2
    ramGbPerRun: PositiveFloat = 8

    @field_validator("gpuIds")
    @classmethod
    def unique_gpus(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("GPU IDs must be distinct.")
        return value


class DevelopmentBatchSpec(RequestModel):
    version: Literal[1] = 1
    experimentId: str | None = Field(default=None, min_length=1, max_length=128)
    experimentRevision: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)] | None = None
    experimentName: str = Field(min_length=1, max_length=120)
    batchName: str = Field(min_length=1, max_length=120)
    inputs: MILInputSpec
    recipe: TrainingRecipe = Field(default_factory=TrainingRecipe)
    mode: Literal["single", "grid", "explicit"] = "single"
    grid: SearchGrid = Field(default_factory=SearchGrid)
    configurations: list[TrainingRecipe] = Field(default_factory=list, max_length=512)
    trainingSeeds: list[Seed] = Field(default_factory=lambda: [42], min_length=1, max_length=100)
    resources: ResourcePolicy = Field(default_factory=ResourcePolicy)
    notes: str = Field(default="", max_length=2000)
    predictorPolicy: ExperimentPredictorPolicy | None = None
    selectionMetric: Literal["validation_auroc", "validation_loss", "validation_accuracy"] | None = (
        "validation_auroc"
    )
    candidateSelection: Literal["best_validation", "all"] | None = "best_validation"

    @model_validator(mode="before")
    @classmethod
    def historical_defaults(cls, values, info):
        if isinstance(values, dict) and (info.context or {}).get("legacy"):
            return {"recipe": {}, "grid": {}, "selectionMetric": None,
                    "candidateSelection": None, **values}
        return values

    @field_validator("experimentName", "batchName")
    @classmethod
    def readable_name(cls, value):
        if not value.strip():
            raise ValueError("Enter a name.")
        return value.strip()

    @field_validator("trainingSeeds")
    @classmethod
    def distinct_seeds(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Training seeds must be distinct.")
        return value

    @model_validator(mode="after")
    def explicit_rows(self):
        if self.candidateSelection == "best_validation" and self.selectionMetric is None:
            raise ValueError("Best-configuration selection requires a validation metric.")
        if (self.experimentId is None) != (self.experimentRevision is None):
            raise ValueError("An experiment ID and its reviewed revision are required together.")
        if (self.mode == "explicit") != bool(self.configurations):
            raise ValueError(
                "Explicit mode requires configuration rows; other modes use the recipe."
            )
        if self.mode == "grid":
            for epochs in self.grid.maxEpochs:
                TrainingRecipe.model_validate({**self.recipe.model_dump(), "maxEpochs": epochs})
        return self

    @model_serializer(mode="wrap")
    def retain_legacy_shape(self, handler):
        values = handler(self)
        # Older frozen manifests and publication receipts have neither field.
        # Preserve their exact request shape for replay and pinned workers.
        if self.experimentId is None:
            values.pop("experimentId", None)
            values.pop("experimentRevision", None)
        if self.predictorPolicy is None:
            values.pop("predictorPolicy", None)
        if self.selectionMetric is None:
            values.pop("selectionMetric", None)
        if self.candidateSelection is None:
            values.pop("candidateSelection", None)
        return values


class FreezeDevelopmentBatch(RequestModel):
    spec: DevelopmentBatchSpec
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
    versionLabel: FreezeVersionLabel


class TrainingAction(RequestModel):
    operationId: str = Field(min_length=1, max_length=200)
