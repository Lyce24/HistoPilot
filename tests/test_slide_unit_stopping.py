"""The small-validation fallback counts the design's own unit wherever it is replayed."""

import ast
from pathlib import Path

from histopilot.application.development import plan_metadata
from histopilot.application.training_patience import stopping_recipe
from histopilot.storage.io import content_hash

PACKAGE = Path(__file__).resolve().parents[1] / "histopilot"
TARGET = {
    "field": "grade",
    "task": "binary_classification",
    "unit": "slide",
    "classes": ["low", "high"],
    "positiveClass": "high",
}


def membership(split_unit):
    # Four positive validation slides from two patients.
    rows = [
        {"slideId": f"s{index}", "patientId": f"p{index // 2}", "label": "high"}
        for index in range(4)
    ] + [{"slideId": f"n{index}", "patientId": f"q{index}", "label": "low"} for index in range(4)]
    rows = [
        {**row, "partition": "val", "fold": 0, "seed": 42, "phase": "cv", "pool": "development"}
        for row in rows
    ]
    return {
        "kind": "protocol",
        "spec": {"target": TARGET, "splitUnit": split_unit},
        "memberships": rows,
    }


class Store:
    def __init__(self, protocol):
        self.protocol = protocol

    def get_configuration(self, _identity):
        return {"manifest": self.protocol}


def test_fallback_replay_counts_slides_for_slide_level_designs():
    recipe = {
        "minValidationPositives": 3,
        "fixedEpochBudget": 10,
        "maxEpochs": 40,
        "checkpointMetric": "validation_auroc",
    }
    manifest = {"spec": {"inputs": {"protocolId": "protocol"}}}
    for split_unit, fallback in (("slide", False), ("patient", True)):
        protocol = membership(split_unit)
        run = {"splitPlanId": content_hash(plan_metadata(protocol["memberships"][0]))}
        effective = stopping_recipe(Store(protocol), manifest, run, recipe)
        # Four positive slides meet a minimum of three; two positive patients do not.
        assert (effective["maxEpochs"] == 10) is fallback


def test_every_stopping_replay_passes_the_split_unit():
    missing = []
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "resolve_stopping"
                and not any(keyword.arg == "split_unit" for keyword in node.keywords)
            ):
                missing.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert missing == []
