"""Resolved training recipes and reproducible development batch intent."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_serializer, model_validator

from histopilot.schemas.mil import MILInputSpec
from histopilot.schemas.version_labels import FreezeVersionLabel
from histopilot.schemas.workspace import RequestModel, Seed

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Epochs = Annotated[StrictInt, Field(ge=1, le=100000)]


class TrainingRecipe(RequestModel):
    model: str = Field(default="abmil", min_length=1, max_length=100, pattern=r"^[\w.-]+$")
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
    maxEpochs: Epochs = 100
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lrScheduler: Literal["none", "cosine"] = "none"
    warmupEpochs: Annotated[StrictInt, Field(ge=0, le=99999)] = 0
    finalLrFraction: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.01
    batchSize: Annotated[StrictInt, Field(ge=1, le=4096)] = 1
    bagSize: Annotated[StrictInt, Field(ge=1, le=1000000)] | None = Field(
        default=4096, description="Maximum training patches per slide; null uses the whole bag."
    )
    earlyStopping: bool = True
    patience: Annotated[StrictInt, Field(ge=1, le=10000)] = 15
    earlyStoppingMinDelta: NonnegativeFloat = 0
    minEpochs: Epochs = 1
    checkpointMetric: Literal["validation_loss", "validation_auroc", "validation_accuracy"] = (
        "validation_loss"
    )

    @model_validator(mode="after")
    def coherent_epoch_budget(self):
        if self.minEpochs > self.maxEpochs:
            raise ValueError("Minimum epochs cannot exceed maximum epochs.")
        if self.warmupEpochs >= self.maxEpochs:
            raise ValueError("Warmup epochs must be fewer than maximum epochs.")
        if self.warmupEpochs and self.lrScheduler != "cosine":
            raise ValueError("Warmup epochs require the cosine learning-rate schedule.")
        return self


class SearchGrid(RequestModel):
    learningRates: list[PositiveFloat] = Field(
        default_factory=lambda: [0.0003], min_length=1, max_length=100
    )
    weightDecays: list[NonnegativeFloat] = Field(
        default_factory=lambda: [0.0001], min_length=1, max_length=100
    )
    maxEpochs: list[Epochs] = Field(default_factory=lambda: [100], min_length=1, max_length=100)

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
        return values


class FreezeDevelopmentBatch(RequestModel):
    spec: DevelopmentBatchSpec
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
    versionLabel: FreezeVersionLabel


class TrainingAction(RequestModel):
    operationId: str = Field(min_length=1, max_length=200)
