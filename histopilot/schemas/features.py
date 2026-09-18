"""Read-only attachment of existing, per-slide HDF5 embedding files."""

from typing import Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from histopilot.schemas.slide_lists import SlideListSpec
from histopilot.schemas.version_labels import FreezeVersionLabel


class FeatureSpec(SlideListSpec):
    # The same two-step selection as extraction: the folder's own files are the initial
    # list unless a slide list names one, and a dataset then narrows whatever that produced.
    slideListPath: str | None = Field(default=None, min_length=1, max_length=4096)
    datasetId: str | None = Field(default=None, min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=4096)
    encoderId: str | None = Field(default=None, min_length=1, max_length=128)
    fileSuffix: Literal[".h5", ".hdf5"] = ".h5"
    idSuffix: str = Field(default="", max_length=128, pattern=r"^[^/\\\x00]*$")
    recursive: bool = False
    layout: Literal["auto", "flat", "trident"] = "auto"
    coordinatesPath: str | None = Field(default=None, min_length=1, max_length=4096)
    # Patch bags need coordinates; one embedding per slide from a slide encoder
    # has no patch grid and is validated against a different contract.
    featureKind: Literal["patch", "slide"] = "patch"
    sourceExtractionJobId: str | None = Field(default=None, pattern=r"^extraction-[a-f0-9]{32}$")

    @model_validator(mode="after")
    def one_slide_list(self):
        if self.slideList is not None and self.slideListPath is not None:
            raise ValueError("Choose one slide list.")
        return self

    @model_serializer(mode="wrap")
    def preserve_existing_selection(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Keep the serialized shape a record was frozen with.

        Feature identity is a hash of this document, so an input a record never
        chose must not appear in it. This replaces ``SlideListSpec``'s serializer
        rather than extending it, so it repeats that field's rule.
        """
        serialized = handler(self)
        if self.slideList is None:
            serialized.pop("slideList", None)
        if self.featureKind == "patch":
            serialized.pop("featureKind", None)
        return serialized


class FreezeFeatureRequest(FeatureSpec):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
    versionLabel: FreezeVersionLabel
