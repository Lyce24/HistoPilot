"""Reviewed table, identity, and scalar-attribute import intent."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from histopilot.schemas.workspace import RequestModel


class TableSource(RequestModel):
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    contentBase64: str | None = Field(default=None, min_length=1, max_length=350000)
    filename: str | None = Field(default=None, min_length=1, max_length=255)
    sheet: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def source_choice(self):
        if (self.path is None) == (self.contentBase64 is None):
            raise ValueError("Supply exactly one server path or uploaded file content.")
        if self.contentBase64 is not None and self.filename is None:
            raise ValueError("An uploaded table needs its original filename.")
        return self


class AttributeMapping(RequestModel):
    key: str = Field(min_length=1, max_length=128)
    sourceColumn: str = Field(min_length=1, max_length=256)
    owner: Literal["slide", "patient"] = "slide"
    type: Literal[
        "text", "categorical", "ordered_categorical", "integer", "decimal", "boolean", "date"
    ] = "text"
    missingValues: list[str] | None = Field(default=None, max_length=32)
    categories: list[str] | None = Field(default=None, max_length=100)


class ImportSpec(RequestModel):
    source: TableSource
    parentId: str | None = Field(default=None, pattern=r"^dataset-[a-f0-9]{64}$")
    slideIdColumn: str = Field(min_length=1, max_length=256)
    patientIdColumn: str | None = Field(default=None, min_length=1, max_length=256)
    patientIdFallback: Literal["unresolved", "slide_id"] = "unresolved"
    slideRoot: str | None = Field(default=None, min_length=1, max_length=4096)
    recursive: bool = True
    includeMissingSlides: bool = False
    missingValues: list[str] = Field(default_factory=lambda: [""], max_length=32)
    attributes: list[AttributeMapping] | None = Field(default=None, max_length=128)
    patientSource: TableSource | None = None
    patientSourceKind: Literal["crosswalk", "patients"] = "crosswalk"
    patientSourceSlideIdColumn: str = Field(default="Slide_ID", min_length=1, max_length=256)
    patientSourcePatientIdColumn: str = Field(default="Patient_ID", min_length=1, max_length=256)
    patientAttributes: list[AttributeMapping] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def patient_attributes_need_source(self):
        if self.patientAttributes and self.patientSource is None:
            raise ValueError("Patient-table attributes require a patient source.")
        if any(attribute.owner != "patient" for attribute in self.patientAttributes):
            raise ValueError("Attributes from the patient table must have patient ownership.")
        return self


class InspectRequest(RequestModel):
    source: TableSource


class PreviewRequest(RequestModel):
    expectedRevision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]


class FreezeRequest(PreviewRequest):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
