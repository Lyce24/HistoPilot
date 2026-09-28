"""Model interpretation shows each slide's frozen decision without changing saved results."""

import copy

from histopilot.application.interpretation import InterpretationService, slide_predictions
from histopilot.storage.project_lock import StorageError

MULTICLASS = {"classes": ["HG", "IND", "LG", "ND"], "task": "multiclass_classification"}
BINARY = {"classes": ["low", "high"], "positiveClass": "high", "task": "binary_classification"}


def completed(slides):
    return {
        "status": "completed",
        "result": {"slides": slides, "classOrder": ["HG", "IND", "LG", "ND"]},
    }


def test_multiclass_prediction_uses_the_top_class_and_member_agreement():
    execution = completed(
        [
            {
                "slideId": "a",
                "probabilities": [0.1, 0.05, 0.15, 0.7],
                "members": [
                    {"index": 0, "probabilities": [0.1, 0.0, 0.1, 0.8]},
                    {"index": 1, "probabilities": [0.5, 0.0, 0.1, 0.4]},
                ],
            },
            {"slideId": "b", "probabilities": [0.6, 0.1, 0.2, 0.1], "members": []},
        ]
    )
    original = copy.deepcopy(execution)
    shown = slide_predictions(execution, MULTICLASS, 0.5)
    first, second = shown["result"]["slides"]
    assert first["prediction"]["predictedLabel"] == "ND"
    assert first["prediction"]["confidence"] == 0.7
    assert first["prediction"]["margin"] == 0.7 - 0.15
    assert first["prediction"]["memberAgreement"]["agree"] == 1
    assert first["prediction"]["memberAgreement"]["total"] == 2
    assert second["prediction"] == {
        "predictedIndex": 0,
        "predictedLabel": "HG",
        "confidence": 0.6,
        "margin": 0.6 - 0.2,
    }
    assert execution == original, "saved results must not be mutated"


def test_binary_prediction_uses_the_frozen_threshold_not_the_top_class():
    execution = completed([{"slideId": "a", "probabilities": [0.65, 0.35], "members": []}])
    assert (
        slide_predictions(execution, BINARY, 0.3)["result"]["slides"][0]["prediction"][
            "predictedLabel"
        ]
        == "high"
    )
    assert (
        slide_predictions(execution, BINARY, 0.5)["result"]["slides"][0]["prediction"][
            "predictedLabel"
        ]
        == "low"
    )


def test_incomplete_or_malformed_results_are_left_unlabelled():
    running = {"status": "running", "result": None}
    assert slide_predictions(running, MULTICLASS, 0.5) is running
    mismatched = completed([{"slideId": "a", "probabilities": [0.5, 0.5], "members": []}])
    assert "prediction" not in slide_predictions(mismatched, MULTICLASS, 0.5)["result"]["slides"][0]


class _Predictors:
    def __init__(self, recipe=None, missing=False):
        self.recipe, self.missing, self.calls = recipe, missing, 0

    def get(self, _identity):
        self.calls += 1
        if self.missing:
            raise StorageError("Predictor not found.", "PREDICTOR_NOT_FOUND", 404)
        return {"manifest": {"recipe": self.recipe}}


def _service(predictors):
    service = InterpretationService.__new__(InterpretationService)
    service.predictors = predictors
    return service


def test_presented_documents_use_the_predictor_threshold_once_per_listing():
    predictors = _Predictors({"decisionThreshold": 0.3})
    service = _service(predictors)
    document = {
        "manifest": {"predictorId": "p", "target": BINARY},
        "execution": completed([{"slideId": "a", "probabilities": [0.65, 0.35], "members": []}]),
    }
    thresholds = {}
    for _ in range(3):
        shown = service.presented(document, thresholds)
    assert shown["execution"]["result"]["slides"][0]["prediction"]["predictedLabel"] == "high"
    assert predictors.calls == 1
    assert "prediction" not in document["execution"]["result"]["slides"][0]


def test_presented_documents_fall_back_to_half_when_the_predictor_is_unavailable():
    service = _service(_Predictors(missing=True))
    document = {
        "manifest": {"predictorId": "gone", "target": BINARY},
        "execution": completed([{"slideId": "a", "probabilities": [0.45, 0.55], "members": []}]),
    }
    prediction = service.presented(document)["execution"]["result"]["slides"][0]["prediction"]
    assert prediction["predictedLabel"] == "high"
    legacy = {"manifest": {"predictorId": "p"}, "execution": {"status": "completed"}}
    assert service.presented(legacy) is legacy
