"""Explicit target and patient-split intents; no executable commands or inferred labels."""

import math
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt, field_validator, model_validator

from histopilot.schemas.version_labels import FreezeVersionLabel
from histopilot.schemas.workspace import RequestModel, Seed

Partition = Literal["train", "val", "test"]


class Condition(RequestModel):
    field: str = Field(min_length=1, max_length=128)
    op: Literal["eq", "ne", "in", "not_in", "regex", "lt", "lte", "gt", "gte", "exists"]
    value: JsonValue

    @model_validator(mode="after")
    def typed_value(self):
        def scalar(value):
            return value is None or type(value) in (str, int, float, bool)

        if self.op in {"in", "not_in"}:
            if not isinstance(self.value, list) or not 1 <= len(self.value) <= 100:
                raise ValueError("Membership filters require 1–100 scalar values.")
            if not all(scalar(value) for value in self.value):
                raise ValueError("Membership filter values must be scalar.")
        elif self.op == "exists":
            if type(self.value) is not bool:
                raise ValueError("exists requires a boolean value.")
        elif self.op == "regex":
            if not isinstance(self.value, str) or not 1 <= len(self.value) <= 512:
                raise ValueError("A regular expression must contain 1–512 characters.")
        elif self.op in {"lt", "lte", "gt", "gte"}:
            if (
                type(self.value) not in (int, float)
                or type(self.value) is float
                and not math.isfinite(self.value)
            ):
                raise ValueError("Numeric comparisons require a finite number.")
        elif not scalar(self.value):
            raise ValueError("Equality filters require a scalar value.")
        values = self.value if isinstance(self.value, list) else [self.value]
        if any(type(value) is float and not math.isfinite(value) for value in values):
            raise ValueError("Filter values must be finite.")
        if any(isinstance(value, str) and len(value) > 4096 for value in values):
            raise ValueError("Filter values are too long.")
        return self


Conditions = Annotated[list[Condition], Field(max_length=30)]


class TargetSpec(RequestModel):
    field: str = Field(min_length=1, max_length=128)
    task: Literal["binary_classification", "multiclass_classification"]
    unit: Literal["patient", "slide"] = "patient"
    classes: list[str] = Field(min_length=2, max_length=50)
    labels: dict[str, str] = Field(min_length=1, max_length=200)
    positiveClass: str | None = None
    missing: Literal["block", "exclude"] = "block"
    unmapped: Literal["block", "exclude"] = "block"

    @model_validator(mode="after")
    def label_contract(self):
        if len(set(self.classes)) != len(self.classes) or any(
            not name.strip() or len(name) > 128 for name in self.classes
        ):
            raise ValueError("Classes must be distinct nonempty names of at most 128 characters.")
        if any(
            len(raw) > 4096 or mapped not in self.classes for raw, mapped in self.labels.items()
        ):
            raise ValueError("Every raw label must map to a declared class.")
        if set(self.labels.values()) != set(self.classes):
            raise ValueError("Every declared class needs an explicit raw-label mapping.")
        if self.task == "binary_classification":
            if len(self.classes) != 2 or self.positiveClass not in self.classes:
                raise ValueError(
                    "Binary targets require two classes and an explicit positiveClass."
                )
        elif self.positiveClass is not None:
            raise ValueError("Multiclass targets do not use positiveClass.")
        return self


class Ratios(RequestModel):
    train: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    val: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    test: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)

    @field_validator("train", "val", "test", mode="before")
    @classmethod
    def no_boolean_ratios(cls, value):
        if type(value) not in (int, float):
            raise ValueError("Ratios must be numbers.")
        return value

    @model_validator(mode="after")
    def sums_to_one(self):
        if not math.isclose(self.train + self.val + self.test, 1, abs_tol=1e-9):
            raise ValueError("Holdout ratios must sum to one.")
        return self


class FixedRules(RequestModel):
    train: Conditions = Field(default_factory=list)
    val: Conditions = Field(default_factory=list)
    test: Conditions = Field(default_factory=list)


class ImportedSplit(RequestModel):
    partitionField: str | None = Field(default=None, min_length=1, max_length=128)
    foldField: str | None = Field(default=None, min_length=1, max_length=128)
    partitionLabels: dict[str, Literal["train", "val", "test", "trainval"]] = Field(
        default_factory=dict, max_length=100
    )
    foldLabels: dict[str, Annotated[StrictInt, Field(ge=0, le=9)]] = Field(
        default_factory=dict, max_length=100
    )
    testFoldLabels: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def explicit_mapping(self):
        if not self.partitionField and not self.foldField:
            raise ValueError("Select an imported partition or fold field.")
        if self.partitionField and not self.partitionLabels:
            raise ValueError("Imported partitions require explicit label mappings.")
        if self.foldField and not self.foldLabels:
            raise ValueError("Imported folds require explicit label mappings.")
        if not self.foldField and (self.foldLabels or self.testFoldLabels):
            raise ValueError("Fold mappings require a fold field.")
        if not self.partitionField and self.partitionLabels:
            raise ValueError("Partition mappings require a partition field.")
        if set(self.testFoldLabels) & set(self.foldLabels):
            raise ValueError("A fold label cannot represent both validation and held-out test.")
        if any(
            len(value) > 4096
            for value in (*self.partitionLabels, *self.foldLabels, *self.testFoldLabels)
        ):
            raise ValueError("Imported labels are too long.")
        return self


class PoolSpec(RequestModel):
    source: Literal["rules", "imported"] = "rules"
    trainSelection: Literal["rules", "remaining"] = "rules"
    validationSource: Literal["training_fraction", "fixed"] = "training_fraction"
    rules: FixedRules = Field(default_factory=FixedRules)
    imported: ImportedSplit | None = None

    @model_validator(mode="after")
    def coherent_source(self):
        if (self.source == "imported") != (self.imported is not None):
            raise ValueError("Predefined pool selection requires an imported partition mapping.")
        if self.imported and (not self.imported.partitionField or self.imported.foldField):
            raise ValueError("Pool selection requires a partition field, without fold assignments.")
        if self.source == "imported" and any(
            getattr(self.rules, role) for role in ("train", "val", "test")
        ):
            raise ValueError("Use either pool rules or predefined partition values.")
        if self.source == "rules" and self.trainSelection == "remaining" and self.rules.train:
            raise ValueError("Remaining training selection cannot also include training rules.")
        if self.validationSource == "training_fraction" and self.rules.val:
            raise ValueError("Choose fixed validation to use validation pool rules.")
        return self


class SplitSpec(RequestModel):
    # Versions 1–3 retain their original frozen meanings. Version 4 contains
    # development assignments only; independent inference cohorts live elsewhere.
    version: Literal[1, 2, 3, 4] = 1
    mode: Literal[
        "rules",
        "kfold",
        "holdout",
        "imported",
        "monte_carlo",
        "leave_one_domain_out",
        "nested_kfold",
        "held_out",
    ] = "kfold"
    folds: Annotated[StrictInt, Field(ge=2, le=10)] = 5
    seeds: list[Seed] = Field(default_factory=lambda: [42], min_length=1, max_length=10)
    ratios: Ratios = Field(default_factory=Ratios)
    rules: FixedRules = Field(default_factory=FixedRules)
    imported: ImportedSplit | None = None
    validationFraction: float = Field(default=0.15, gt=0, lt=1, allow_inf_nan=False)
    testFraction: float = Field(default=0.2, gt=0, lt=1, allow_inf_nan=False)
    repeats: Annotated[StrictInt, Field(ge=1, le=100)] = 5
    outerFolds: Annotated[StrictInt, Field(ge=2, le=10)] = 5
    innerFolds: Annotated[StrictInt, Field(ge=2, le=10)] = 3
    stratify: bool = True
    domainField: str | None = Field(default=None, min_length=1, max_length=128)
    domainPolicy: Literal["all", "selected"] = "all"
    heldOutDomains: list[str] = Field(default_factory=list, max_length=100)
    heldOutSource: Literal["fractions", "rules", "imported"] = "fractions"
    pools: PoolSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_validation_default(cls, value):
        # Existing draft specifications may omit this field. Their interpretation
        # and preview hashes retain the original 20% default.
        if (
            isinstance(value, dict)
            and "validationFraction" not in value
            and value.get("version", 1) in (1, 2)
        ):
            return {**value, "validationFraction": 0.2}
        return value

    @field_validator("validationFraction", "testFraction", mode="before")
    @classmethod
    def no_boolean_fractions(cls, value):
        if type(value) not in (int, float):
            raise ValueError("Split fractions must be numbers.")
        return value

    @model_validator(mode="after")
    def coherent_mode(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        self.seeds = sorted(self.seeds)
        if self.version == 1 and self.mode not in {"rules", "kfold", "holdout", "imported"}:
            raise ValueError("The selected strategy requires split version 2.")
        if self.version >= 2 and self.mode not in {
            "kfold",
            "monte_carlo",
            "leave_one_domain_out",
            "nested_kfold",
            "held_out",
        }:
            raise ValueError("Choose one of the five evaluation strategies.")
        if (self.version >= 3) != (self.pools is not None):
            raise ValueError("Versions 3 and 4 require explicit source selection settings.")
        if self.version >= 3 and any(
            getattr(self.rules, role) for role in ("train", "val", "test")
        ):
            raise ValueError("Partition rules belong in the source selection settings.")
        if self.version == 4:
            if self.pools.rules.test or (
                self.pools.imported and "test" in self.pools.imported.partitionLabels.values()
            ):
                raise ValueError(
                    "Development protocols cannot reserve test data. Define inference cohorts in Model evaluation."
                )
            if self.heldOutSource != "fractions" or self.ratios.test:
                raise ValueError(
                    "Development assessment uses its strategy fraction; external holdout settings are not allowed."
                )
        uses_imported = self.mode == "imported" or (
            self.version == 2 and self.mode == "held_out" and self.heldOutSource == "imported"
        )
        if uses_imported != (self.imported is not None):
            raise ValueError("Imported mappings are required only for imported split mode.")
        if self.imported and any(
            value >= self.folds for value in self.imported.foldLabels.values()
        ):
            raise ValueError("Imported fold mappings must lie in the declared fold range.")
        if self.version >= 2:
            if self.imported and (not self.imported.partitionField or self.imported.foldField):
                raise ValueError(
                    "Held-out imported splits require a partition field, without a fold field."
                )
            if self.mode != "held_out" or self.heldOutSource != "rules":
                if any(getattr(self.rules, role) for role in ("train", "val", "test")):
                    raise ValueError("Fixed rules apply only to the held-out rules strategy.")
            if self.mode == "leave_one_domain_out":
                if not self.domainField:
                    raise ValueError("Choose the site or cohort column.")
                if self.domainPolicy == "selected" and not self.heldOutDomains:
                    raise ValueError("Select at least one held-out site or cohort.")
                if self.domainPolicy == "all" and self.heldOutDomains:
                    raise ValueError("Selected domain values require the selected-domain policy.")
            elif self.domainField or self.heldOutDomains:
                raise ValueError("Domain settings apply only to leave-one-site/cohort-out splits.")
            if len(set(self.heldOutDomains)) != len(self.heldOutDomains) or any(
                not value.strip() or len(value) > 4096 for value in self.heldOutDomains
            ):
                raise ValueError("Held-out domains must be distinct nonempty values.")
        return self


class Constraints(RequestModel):
    minPatientsPerClass: Annotated[StrictInt, Field(ge=1, le=100000)] = 1
    minPatientsPerPartition: Annotated[StrictInt, Field(ge=1, le=100000)] = 1


class ProtocolSpec(RequestModel):
    datasetId: str = Field(pattern=r"^dataset-[a-f0-9]{64}$")
    target: TargetSpec
    predictors: list[str] = Field(default_factory=list, max_length=100)
    eligibility: Conditions = Field(default_factory=list)
    split: SplitSpec = Field(default_factory=SplitSpec)
    constraints: Constraints = Field(default_factory=Constraints)
    featureSetId: str | None = Field(default=None, max_length=128)
    featurePackId: str | None = Field(default=None, pattern=r"^pack-[a-f0-9]{64}$")

    @model_validator(mode="after")
    def pack_requires_features(self):
        if self.featurePackId and not self.featureSetId:
            raise ValueError("Select a feature version before selecting its pack.")
        return self

    @field_validator("predictors")
    @classmethod
    def distinct_predictors(cls, values):
        if len(set(values)) != len(values) or any(
            not value or len(value) > 128 for value in values
        ):
            raise ValueError("Predictor fields must be distinct nonempty keys.")
        return values


class ProtocolExploreRequest(RequestModel):
    """Read-only cohort feedback, independent of completed target/feature configuration."""

    datasetId: str = Field(pattern=r"^dataset-[a-f0-9]{64}$")
    targetField: str | None = Field(default=None, min_length=1, max_length=128)
    eligibility: Conditions = Field(default_factory=list)
    rules: FixedRules = Field(default_factory=FixedRules)
    splitMode: Literal[
        "rules",
        "kfold",
        "holdout",
        "imported",
        "monte_carlo",
        "leave_one_domain_out",
        "nested_kfold",
        "held_out",
    ] = "rules"
    # Live cohort counts remain available while strategy controls are incomplete.
    # Authoritative preview validates the full SplitSpec before assigning rows.
    split: dict[str, JsonValue] | None = Field(default=None, max_length=30)


class ProtocolPreviewRequest(RequestModel):
    expectedRevision: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]


class ProtocolFreezeRequest(ProtocolPreviewRequest):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
    versionLabel: FreezeVersionLabel
