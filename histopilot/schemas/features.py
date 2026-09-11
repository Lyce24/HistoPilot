"""Read-only attachment of existing, per-slide HDF5 embedding files."""

from typing import Literal

from pydantic import Field

from histopilot.schemas.version_labels import FreezeVersionLabel
from histopilot.schemas.workspace import RequestModel


class FeatureSpec(RequestModel):
    datasetId: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=4096)
    encoderId: str | None = Field(default=None, min_length=1, max_length=128)
    fileSuffix: Literal[".h5", ".hdf5"] = ".h5"
    idSuffix: str = Field(default="", max_length=128, pattern=r"^[^/\\\x00]*$")
    recursive: bool = False
    layout: Literal["auto", "flat", "trident"] = "auto"
    coordinatesPath: str | None = Field(default=None, min_length=1, max_length=4096)
    sourceExtractionJobId: str | None = Field(default=None, pattern=r"^extraction-[a-f0-9]{32}$")


class FreezeFeatureRequest(FeatureSpec):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
    versionLabel: FreezeVersionLabel
