"""PFM extraction contract shared by TRIDENT, STAMP, and custom backends."""

from typing import Protocol

from histopilot.domain import FeatureSet, Slide


class PFMPort(Protocol):
    def extract(self, *, slides: tuple[Slide, ...], feature_set: FeatureSet) -> FeatureSet:
        """Write features/coordinates to the requested artifact locations.

        The returned manifest must describe actual completed extraction; this
        protocol neither downloads a checkpoint nor implements extraction.
        """
        ...
