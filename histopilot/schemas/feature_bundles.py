"""Immutable, verified feature collections; loading choices belong to experiments."""

from typing import Annotated

from pydantic import Field, field_validator

from histopilot.schemas.version_labels import FreezeVersionLabel
from histopilot.schemas.workspace import RequestModel


class FeatureBundleSpec(RequestModel):
    featureSetId: str = Field(min_length=1, max_length=128)
    packArtifactIds: list[Annotated[str, Field(pattern=r"^pack-[a-f0-9]{64}$")]] = Field(
        default_factory=list, max_length=100
    )

    @field_validator("packArtifactIds")
    @classmethod
    def canonical_pack_ids(cls, value: list[str]) -> list[str]:
        return sorted(set(value))


class FreezeFeatureBundleRequest(FeatureBundleSpec):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
    versionLabel: FreezeVersionLabel
