"""MIL adapters exchange artifact references rather than backend tensor types."""

from typing import Protocol

from histopilot.domain import Experiment, FeatureSet, Run, Split


class MILPort(Protocol):
    def train(self, *, experiment: Experiment, run: Run, features: FeatureSet, split: Split) -> Run:
        """Train one run and return its recorded checkpoint and terminal status."""
        ...

    def predict(self, *, run: Run, features: FeatureSet, output_uri: str) -> str:
        """Return a written prediction artifact URI keyed by cohort member ID."""
        ...

    def attention(self, *, run: Run, slide_id: str, features: FeatureSet, output_uri: str) -> str:
        """Return attention aligned to the feature set's patch coordinates."""
        ...
