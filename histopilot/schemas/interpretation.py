"""Reviewed whole-slide attention inputs and explicit coordinate provenance."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from histopilot.schemas.development import ResourcePolicy
from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.workspace import RequestModel

PathText = Annotated[str, Field(min_length=1, max_length=4096)]
DatasetKey = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")]
Dimension = Annotated[float, Field(gt=0, le=1000000, allow_inf_nan=False, strict=True)]


class InterpretationSlide(RequestModel):
    slideId: str = Field(min_length=1, max_length=512)
    slidePath: PathText
    sourceFormat: Literal["h5", "packed"] = "h5"
    featurePath: PathText | None = None
    packPath: PathText | None = None
    packSlideId: str | None = Field(default=None, min_length=1, max_length=512)
    coordinatesPath: PathText | None = None
    featureKey: DatasetKey = "features"
    coordinatesKey: DatasetKey = "coords"
    coordinateSpace: Literal["level0"] = "level0"
    patchWidthLevel0: Dimension | None = None
    patchHeightLevel0: Dimension | None = None
    confirmRowAlignment: Literal[True]

    @field_validator("slideId", "packSlideId")
    @classmethod
    def logical_identity(cls, value):
        if value is not None and (
            value in {".", ".."}
            or not value.strip()
            or any(character in value for character in ("/", "\\"))
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("Slide IDs cannot contain slashes, traversal or control characters.")
        return value

    @field_validator("confirmRowAlignment", mode="before")
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError("Explicitly confirm feature and coordinate row alignment.")
        return value

    @model_validator(mode="after")
    def geometry_pair(self):
        if (self.patchWidthLevel0 is None) != (self.patchHeightLevel0 is None):
            raise ValueError("Provide both patch width and height in level-0 pixels.")
        if self.sourceFormat == "h5" and (
            not self.featurePath or self.packPath or self.packSlideId
        ):
            raise ValueError("HDF5 inputs require a feature file and cannot select a pack.")
        if self.sourceFormat == "packed" and (
            not self.packPath or not self.packSlideId or self.featurePath or self.coordinatesPath
        ):
            raise ValueError(
                "Packed inputs require a pack and slide identity, without separate feature or coordinate files."
            )
        return self


class InterpretationSelection(RequestModel):
    name: str = Field(min_length=1, max_length=120)
    predictorId: ConfigurationId
    evaluationId: ConfigurationId | None = None
    clinicalAnalysisId: ConfigurationId | None = None
    featureBundleId: ConfigurationId | None = None
    packArtifactId: str | None = Field(default=None, pattern=r"^pack-[a-f0-9]{64}$")
    slideFolder: PathText | None = None
    encoderId: str = Field(min_length=1, max_length=200)
    slides: list[InterpretationSlide] = Field(min_length=1, max_length=128)
    resources: ResourcePolicy = Field(
        default_factory=lambda: ResourcePolicy(gpuIds=[], dataLoaderWorkers=0, ramGbPerRun=8.0)
    )

    @field_validator("resources")
    @classmethod
    def canonical_resources(cls, value):
        return ResourcePolicy.model_validate(value.model_dump())

    @field_validator("name", "encoderId")
    @classmethod
    def nonempty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a nonempty value.")
        return value

    @field_validator("slides")
    @classmethod
    def unique_slides(cls, values):
        for key in ("slideId", "slidePath"):
            entries = [getattr(row, key) for row in values]
            if len(set(entries)) != len(entries):
                raise ValueError("Select each slide once and give each slide a distinct ID.")
        return values

    @model_validator(mode="after")
    def bound_source(self):
        if bool(self.featureBundleId) != bool(self.slideFolder):
            raise ValueError("A selected bundle requires its slide folder.")
        if self.packArtifactId and not self.featureBundleId:
            raise ValueError("Select a frozen bundle for this pack artifact.")
        return self


class SaveInterpretation(InterpretationSelection):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)


class SlideInspection(RequestModel):
    path: PathText


class AttentionQuery(RequestModel):
    offset: Annotated[StrictInt, Field(ge=0)] = 0
    limit: Annotated[StrictInt, Field(ge=1, le=50000)] = 10000


class InterpretationGallerySource(RequestModel):
    slideFolder: PathText | None = None
    featureBundleId: ConfigurationId
    packArtifactId: str | None = Field(default=None, pattern=r"^pack-[a-f0-9]{64}$")
    predictorId: ConfigurationId | None = None


class InterpretationGalleryQuery(InterpretationGallerySource):
    search: str = Field(default="", max_length=200)
    offset: Annotated[StrictInt, Field(ge=0, le=20000)] = 0
    limit: Annotated[StrictInt, Field(ge=1, le=100)] = 24


class VisualizeInterpretation(InterpretationGallerySource):
    predictorId: ConfigurationId
    evaluationId: ConfigurationId | None = None
    clinicalAnalysisId: ConfigurationId | None = None
    slidePaths: list[PathText] = Field(min_length=1, max_length=128)
    patchWidthLevel0: Dimension | None = None
    patchHeightLevel0: Dimension | None = None
    resources: ResourcePolicy = Field(
        default_factory=lambda: ResourcePolicy(gpuIds=[], dataLoaderWorkers=0, ramGbPerRun=8.0)
    )
    operationId: str = Field(min_length=1, max_length=128)

    @field_validator("slidePaths")
    @classmethod
    def distinct_paths(cls, values):
        if len(set(values)) != len(values):
            raise ValueError("Select each slide only once.")
        return values

    @model_validator(mode="after")
    def paired_geometry(self):
        if (self.patchWidthLevel0 is None) != (self.patchHeightLevel0 is None):
            raise ValueError("Provide both patch width and height in level-0 pixels.")
        return self
