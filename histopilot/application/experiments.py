"""Experiment planning and result collection are intentionally unimplemented."""

from histopilot.domain import Experiment, Result, Run


class ExperimentService:
    def plan(
        self, *, experiment: Experiment, seeds: tuple[int, ...], folds: tuple[int, ...]
    ) -> tuple[Run, ...]:
        raise NotImplementedError("Experiment validation and run planning are not implemented.")

    def results(self, experiment_id: str) -> tuple[Result, ...]:
        raise NotImplementedError(
            "Result persistence and provenance resolution are not implemented."
        )
