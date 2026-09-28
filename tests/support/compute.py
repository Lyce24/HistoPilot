"""Compute jobs (evaluations, inference, refits, attention) queued in this test's Task Center.

Services built here queue their work exactly as production does. Tests stand in for the
worker: they write its artifacts and receipts, or run it (``support.workers``), and then
conclude the task the way the runner records an exited worker.
"""

import h5py
import numpy as np
import pytest

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.interpretation import InterpretationService
from histopilot.application.predictors import checkpoint_snapshot
from histopilot.schemas.interpretation import InterpretationSelection, SaveInterpretation
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter.model import LIVE
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json
from support.training import runtime


def managed_jobs(store, center):
    """Compute jobs that queue in ``center``, as production launches them."""
    return ComputeJobService(store, runtime=runtime, task_center=center.client)


def compute_tasks(center, **filters):
    """Queued compute work only; fixtures also leave their feature packing tasks behind."""
    return center.tasks(kind="compute-job", **filters)


def compute_task(service, identity, center):
    """The Task Center task of one interpretation's compute job."""
    return center.task(read_json(service.jobs.folder(identity) / "state.json")["taskId"])


def launches(center):
    """How many compute workers were queued: new tasks plus requeued attempts of old ones.

    A resumed job requeues its task instead of starting another one.
    """
    return sum(task["attempt"] for task in compute_tasks(center))


def complete(service, identity, result, center):
    """Record a finished worker: its receipts, then its task concluded as succeeded.

    Rewriting the receipts of an already concluded job leaves the task as it is.
    """
    folder = service.jobs.folder(identity)
    state = read_json(folder / "state.json")
    write_json(folder / "state.json", {**state, "status": "completed", "result": result})
    write_json(folder / "result.json", result)
    if center.state(state["taskId"]) in LIVE:
        center.finish(state["taskId"], "succeeded")


# -- an attention study --------------------------------------------------------------------


@pytest.fixture
def managed_study(tmp_path, task_center):
    """One reviewed slide and a frozen predictor whose launches queue in ``task_center``.

    Returns the ``InterpretationService``, the selection to save and the Task Center.
    """
    from PIL import Image

    (tmp_path / "project").mkdir()
    store = ScientificStore(tmp_path / "project", "interpret-project")
    draft = store.create_draft("import", "Development", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"name": "Development"},
        artifacts={"slides.json": b'[{"slideId":"development-only"}]'},
        operation_id="data",
    )
    batch_id = "configuration-" + "1" * 64
    checkpoint_folder = store.folder / "training" / batch_id / "runs" / "run-one"
    checkpoint_folder.mkdir(parents=True)
    checkpoint_path = checkpoint_folder / "best.ckpt"
    checkpoint_path.write_bytes(b"test evidence, never deserialized by control API")
    predictor = store.publish_configuration(
        manifest={
            "kind": "frozen-predictor",
            "datasetId": dataset["id"],
            "name": "Model",
            "batchId": batch_id,
            "experimentId": "experiment-one",
            "method": "ensemble",
            "target": {
                "task": "binary_classification",
                "unit": "slide",
                "classes": ["yes", "no"],
                "positiveClass": "yes",
            },
            "recipe": {"model": "abmil", "embedDim": 8, "attentionDim": 4},
            "inputs": {
                "features": {"encoderId": "test-encoder", "dimensions": 4, "dtype": "float32"}
            },
            "checkpoints": [
                {**checkpoint_snapshot(checkpoint_path, checkpoint_folder), "runId": "run-one"}
            ],
        },
        operation_id="predictor",
    )
    image_path = tmp_path / "independent-slide.png"
    Image.new("RGB", (300, 200), color="pink").save(image_path)
    features_path = tmp_path / "features.h5"
    with h5py.File(features_path, "w") as handle:
        handle.create_dataset("features", data=np.arange(12, dtype=np.float32).reshape(3, 4))
        coords = handle.create_dataset(
            "coords", data=np.array([[0, 0], [100, 0], [100, 100]], dtype=np.int64)
        )
        coords.attrs["patch_size_level0"] = 100
        handle["features"].attrs["encoder_id"] = "test-encoder"
    selection = InterpretationSelection(
        name="Attention",
        predictorId=predictor["id"],
        encoderId="test-encoder",
        slides=[
            {
                "slideId": "independent",
                "slidePath": str(image_path),
                "featurePath": str(features_path),
                "confirmRowAlignment": True,
            }
        ],
    )
    service = InterpretationService(store, LocalFilesystem((tmp_path,)))
    service.jobs = managed_jobs(store, task_center)
    return service, selection, task_center


def save_study(study):
    """Save the study's interpretation; returns the saved record and its request."""
    service, selection, _ = study
    preview = service.preview(selection)
    assert preview["canSave"], preview
    request = SaveInterpretation(
        **selection.model_dump(), previewHash=preview["previewHash"], operationId="save-attention"
    )
    return service.save(request), request
