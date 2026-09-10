"""Storage contracts keep binary formats outside the scientific domain."""

from typing import Protocol

from histopilot.domain.features import FeatureSet


class FeatureStore(Protocol):
    """HDF5 first; implementations must preserve level-0 coordinates and identity."""

    def inspect(self, feature_set: FeatureSet) -> FeatureSet: ...
    def publish(self, feature_set: FeatureSet) -> FeatureSet:
        """Validate complete outputs before making the manifest available for runs."""
        ...
