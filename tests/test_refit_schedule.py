"""A refit trains for its epoch budget along the fold models' own learning-rate schedule."""

import pytest
from support.predictors import managed as managed
from support.predictors import registry as registry

from histopilot.schemas.development import TrainingRecipe

torch = pytest.importorskip("torch")

from histopilot.training import refit as refit_module  # noqa: E402

__all__ = ["managed", "registry"]


def test_refit_budget_records_the_fold_schedule(managed):
    service, identity, _jobs, _selections = managed
    service.launch(identity, "start")
    items = [row for row in service.advance(identity)["items"] if row["method"] == "refit"]
    assert items
    for item in items:
        refit = service.refits.get(item["recordId"])["manifest"]
        source = refit["sourceRecipe"]
        budget = refit["epochBudget"]
        assert budget["epochs"] < source["maxEpochs"]
        assert budget["scheduleHorizonEpochs"] == source["maxEpochs"]
        assert budget["scheduleWarmupEpochs"] == source.get("warmupEpochs", 0)
        # The trainer still stops at the budget.
        assert refit["recipe"]["maxEpochs"] == budget["epochs"]


class Captured(Exception):
    pass


@pytest.mark.parametrize(
    "budget, expected",
    [
        ({"epochs": 4, "scheduleHorizonEpochs": 40, "scheduleWarmupEpochs": 5}, (40, 5)),
        # Refits planned before the schedule was recorded keep their trajectory.
        ({"epochs": 4}, (4, 3)),
    ],
)
def test_refit_model_follows_the_recorded_schedule(monkeypatch, tmp_path, budget, expected):
    recipe = TrainingRecipe.model_validate(
        {
            "maxEpochs": 4,
            "minEpochs": 4,
            "earlyStopping": False,
            "warmupEpochs": 3,
            "lrScheduler": "cosine",
        }
    ).model_dump()

    class DataModule:
        clinical_preprocessor = None

        def __init__(self, _plan):
            pass

        def training_class_weights(self):
            return None

        def training_class_weight_unit(self):
            return "slide"

    def model(_dimensions, _target, model_recipe, **_options):
        raise Captured((model_recipe["maxEpochs"], model_recipe["warmupEpochs"]))

    monkeypatch.setattr(refit_module, "RefitDataModule", DataModule)
    monkeypatch.setattr(refit_module, "MILTrainModule", model)
    plan = {
        "recipe": recipe,
        "epochBudget": budget,
        "device": "cpu",
        "resources": {"cpuThreadsPerRun": 1},
        "trainingSeed": 42,
        "target": {"task": "binary_classification", "classes": ["a", "b"]},
        "data": {"featureDim": 8},
    }
    with pytest.raises(Captured) as captured:
        refit_module.train_refit(plan, tmp_path)
    assert captured.value.args[0] == expected
