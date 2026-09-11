"""MIL planning owns how immutable feature bundles will be read."""

from typing import Literal

from pydantic import Field, model_validator

from histopilot.schemas.workspace import RequestModel


class MILInputSpec(RequestModel):
    protocolId: str = Field(min_length=1, max_length=128)
    featureBundleId: str = Field(min_length=1, max_length=128)
    loadingPolicy: Literal["auto", "native", "mmap"] = "auto"
    packArtifactId: str | None = Field(default=None, pattern=r"^pack-[a-f0-9]{64}$")

    @model_validator(mode="after")
    def native_has_no_pack(self):
        if self.loadingPolicy == "native" and self.packArtifactId:
            raise ValueError("Original-file loading cannot also select a pack.")
        return self
