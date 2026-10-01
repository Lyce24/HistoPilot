"""Reference standards: labels attached to a frozen cohort after it was frozen."""

from pydantic import Field, field_validator, model_validator

from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.inference import Unit
from histopilot.schemas.morphology import DatasetId
from histopilot.schemas.workspace import RequestModel


class ReferenceSelection(RequestModel):
    """One dataset column mapped to a class set, for every slide of one cohort.

    Values the mapping leaves out, and missing values, stay unlabeled: those slides are
    never scored against this reference.
    """

    cohortId: ConfigurationId
    name: str = Field(min_length=1, max_length=120)
    # One or more frozen datasets, matched to the cohort's slides by slide ID.
    datasetIds: list[DatasetId] = Field(min_length=1, max_length=20)
    field: str = Field(min_length=1, max_length=200)
    classes: list[str] = Field(min_length=2, max_length=64)
    labels: dict[str, str] = Field(default_factory=dict, max_length=1000)

    @field_validator("name")
    @classmethod
    def readable_name(cls, value):
        if not value.strip():
            raise ValueError("Enter a name for the reference standard.")
        return value.strip()

    @field_validator("field")
    @classmethod
    def named_column(cls, value):
        # Kept verbatim: it must equal a key of the dataset's data dictionary.
        if not value.strip():
            raise ValueError("Choose a column.")
        return value

    @model_validator(mode="after")
    def mapped_to_classes(self):
        if any(not item.strip() or len(item) > 128 for item in self.classes):
            raise ValueError("Class names must be 1 to 128 characters.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Class names must be unique.")
        if len(set(self.datasetIds)) != len(self.datasetIds):
            raise ValueError("Choose each dataset once.")
        if any(len(raw) > 4096 for raw in self.labels):
            raise ValueError("Source values must be at most 4,096 characters.")
        if set(self.labels.values()) - set(self.classes):
            raise ValueError("Map source values only to the reference's classes.")
        return self


class SaveReference(ReferenceSelection):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)


class ReferenceScoreQuery(RequestModel):
    """Score a run against one reference; ``referenceId`` None means the cohort's labels."""

    referenceId: ConfigurationId | None = None


class AgreementQuery(RequestModel):
    unit: Unit = "selected"


class RecalibrationQuery(RequestModel):
    """A run's calibration against its cohort's labels, or a reference standard's."""

    unit: Unit = "selected"
    referenceId: ConfigurationId | None = None
