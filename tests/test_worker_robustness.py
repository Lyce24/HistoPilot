"""Optional telemetry is bounded and typed before it reaches job controls."""

import pytest

from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_progress


@pytest.mark.parametrize(
    "progress",
    [
        {"epoch": {}},
        {"maxEpochs": "3"},
        {"completed": True},
        {"total": -1},
        {"learningRate": None},
        {"trainingLoss": []},
        {"trainingLoss": 10**400},
        {"percent": "50"},
        {"currentSlide": {"id": "slide-a"}},
        {"unit": []},
        {"validation": []},
        {"validation": {"loss": {"value": 1}}},
        {"validation": {"count": True}},
        {"validation": {"available": "yes"}},
        {"validation": {"reason": []}},
        {"validation": {"missingClasses": [1]}},
    ],
)
def test_valid_json_with_invalid_telemetry_types_returns_a_warning(tmp_path, progress):
    path = tmp_path / "progress.json"
    write_json(path, progress)
    value, warning = read_progress(path)
    assert value is None
    assert warning and "Job status and cancellation remain available" in warning


@pytest.mark.parametrize(
    "progress",
    [
        {"completedModels": 1, "totalModels": 3, "slideCount": 4},
        {
            "epoch": 1,
            "maxEpochs": 3,
            "learningRate": 0.001,
            "validation": {"loss": 1.2, "auroc": None, "available": True, "count": 2},
        },
        {"epoch": 1, "maxEpochs": 3, "validation": None},
        {
            "stage": "validating",
            "unit": "patches",
            "currentSlide": None,
            "completed": 0,
            "total": 10,
            "percent": 0,
        },
        {"futureTelemetry": {"workerVersion": 2}},
    ],
)
def test_supported_progress_shapes_and_unknown_optional_fields_are_preserved(tmp_path, progress):
    path = tmp_path / "progress.json"
    write_json(path, progress)
    assert read_progress(path) == (progress, None)


def test_empty_progress_snapshot_is_treated_as_no_telemetry(tmp_path):
    path = tmp_path / "progress.json"
    write_json(path, {})
    assert read_progress(path) == (None, None)


@pytest.mark.parametrize("depth", [65, 1500, 20000])
def test_deeply_nested_json_does_not_escape_as_a_recursion_error(tmp_path, depth):
    from histopilot.storage.project_lock import StorageError
    from histopilot.workers.training_process import read_json

    path = tmp_path / "progress.json"
    path.write_text('{"extra":' + "[" * depth + "0" + "]" * depth + "}")
    with pytest.raises(StorageError) as error:
        read_json(path)
    assert error.value.code == "TRAINING_STATE_INVALID"
    value, warning = read_progress(path)
    assert value is None and warning
