"""Dataset targets and an explicit slide- or patient-level train/test partition."""

from typing import Any, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from histopilot.schemas.protocols import Conditions, TargetSpec, require_training_labels
from histopilot.schemas.workspace import RequestModel, Seed


class TargetSplitSettings(RequestModel):
    method: Literal["random", "rules", "imported"] = "random"
    testFraction: float = Field(default=0.2, ge=0, lt=1, allow_inf_nan=False)
    seed: Seed = 42
    stratify: bool = False
    stratifyField: str | None = Field(default=None, min_length=1, max_length=128)
    trainRules: Conditions = Field(default_factory=list)
    testRules: Conditions = Field(default_factory=list)
    # Testing may take every eligible group the training conditions leave out,
    # mirroring the training remainder used when no training conditions are set.
    testRemaining: bool = False
    partitionField: str | None = Field(default=None, min_length=1, max_length=128)
    trainValues: list[str] = Field(default_factory=list, max_length=100)
    testValues: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("testFraction", mode="before")
    @classmethod
    def numeric_fraction(cls, value):
        if type(value) not in (int, float):
            raise ValueError("The testing percentage must be a number.")
        return value

    @model_validator(mode="after")
    def partition_settings(self):
        if self.method == "imported":
            if not self.partitionField or not self.trainValues:
                raise ValueError("Select a partition column and at least one training value.")
            if set(self.trainValues) & set(self.testValues):
                raise ValueError("Training and testing values must be disjoint.")
        elif self.partitionField or self.trainValues or self.testValues:
            raise ValueError("Predefined values apply only to the imported partition method.")
        if self.method != "rules" and (self.trainRules or self.testRules or self.testRemaining):
            raise ValueError("Training/testing conditions apply only to the rules method.")
        if self.testRemaining and self.testRules:
            raise ValueError(
                "Testing uses either its own conditions or every slide outside training, not both."
            )
        for values in (self.trainValues, self.testValues):
            if len(values) != len(set(values)) or any(
                not value or len(value) > 4096 for value in values
            ):
                raise ValueError("Partition values must be distinct nonempty values.")
        return self

    @model_serializer(mode="wrap")
    def omit_default_remainder(self, handler):
        # Historical specifications never carried this field; keep their hashes unchanged.
        serialized = handler(self)
        if not self.testRemaining:
            serialized.pop("testRemaining", None)
        return serialized


class TargetPartitionSpec(RequestModel):
    datasetId: str = Field(pattern=r"^dataset-[a-f0-9]{64}$")
    splitUnit: Literal["slide", "patient"] = "patient"
    eligibility: Conditions = Field(default_factory=list)
    split: TargetSplitSettings = Field(default_factory=TargetSplitSettings)


class TargetFieldSelection(RequestModel):
    train: str | None = Field(default=None, min_length=1, max_length=128)
    test: str | None = Field(default=None, min_length=1, max_length=128)


class TargetSplitPartitionPreviewRequest(TargetPartitionSpec):
    targetFields: TargetFieldSelection = Field(default_factory=TargetFieldSelection)
    # An unfinished target must not hide the already selected partition counts.
    # The service validates these separately and returns target findings in place.
    target: dict[str, Any] | None = None
    testTarget: dict[str, Any] | None = None


class TargetSplitSpec(TargetPartitionSpec):
    target: TargetSpec
    testTarget: TargetSpec | None = None
    # Deprecated and always empty: Targets & Splits never declares tabular predictors. The key
    # stays so stored drafts and frozen specs (all "predictors": []) keep validating and hashing
    # exactly as before.
    predictors: list[str] = Field(default_factory=list, max_length=100)

    @model_serializer(mode="wrap")
    def preserve_inherited_testing_target(self, handler):
        serialized = handler(self)
        if "testTarget" not in self.model_fields_set:
            serialized.pop("testTarget", None)
        if "splitUnit" not in self.model_fields_set:
            serialized.pop("splitUnit", None)
        return serialized

    @field_validator("predictors")
    @classmethod
    def no_predictors(cls, values):
        if values:
            raise ValueError("Targets & Splits does not declare tabular predictors.")
        return values

    @model_validator(mode="after")
    def testing_target_contract(self):
        if "splitUnit" in self.model_fields_set and self.target.unit != self.splitUnit:
            raise ValueError("The target prediction unit must match the selected split unit.")
        require_training_labels(self.target)
        if self.testTarget:
            for key in ("task", "unit", "classes", "positiveClass"):
                if getattr(self.testTarget, key) != getattr(self.target, key):
                    raise ValueError(
                        "Testing targets must preserve training task, prediction unit, class order, "
                        "and positive class."
                    )
            if self.testTarget.field == self.target.field and any(
                self.testTarget.labels[raw] != self.target.labels[raw]
                for raw in self.testTarget.labels.keys() & self.target.labels.keys()
            ):
                raise ValueError("The same target field must preserve training label mappings.")
        return self
