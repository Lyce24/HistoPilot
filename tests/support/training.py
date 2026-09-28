"""Training runtimes, development batches and the Task Center tasks of a training batch.

Batches launch as fold tasks and a final results collection, exactly as in production.
Nothing here runs a real training worker: a fold "finishes" when the receipts its worker
would write are in place and the runner's adapter has concluded the task from them.
"""

import copy
import sys
from pathlib import Path

import pytest

from histopilot.application.development import DevelopmentService
from histopilot.application.protocols import ProtocolService
from histopilot.application.training import TrainingService
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.io import read_json_bounded, utc_now, write_json_atomic
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter.adapters.mil import MilCollectAdapter, MilFoldAdapter
from histopilot.taskcenter.model import LIVE
from histopilot.workers.managed_collect import final_status
from support.projects import TARGET, bundle, dataset
from support.task_center import begin, conclude

HOST = {
    "cpuCount": 8,
    "totalRamGb": 16.0,
    "availableRamGb": 16.0,
    "bootId": "test",
    "kernel": "test",
}


def runtime(**_options):
    """A CPU-only training runtime, shaped as ``training_runtime`` reports one.

    Its fixed host keeps launch checks from inheriting the CI runner's resource limits.
    """
    return {
        "available": True,
        "python": sys.executable,
        "versions": {},
        "cudaAvailable": False,
        "gpuCount": 0,
        "findings": [],
        "host": {"cpuCount": 8, "totalRamGb": 16},
    }


class Runtime:
    """``runtime`` as a probe that counts its calls.

    ``unavailable_after`` makes every later probe report a missing runtime; ``drift_after``
    changes the reported package versions from then on, as an environment edit would.
    """

    def __init__(self):
        self.calls = 0
        self.unavailable_after = None
        self.drift_after = None

    def __call__(self, **_options):
        self.calls += 1
        if self.unavailable_after is not None and self.calls > self.unavailable_after:
            return {
                "available": False,
                "findings": [
                    {
                        "severity": "error",
                        "code": "TRAINING_RUNTIME_UNAVAILABLE",
                        "message": "Synthetic runtime failure",
                    }
                ],
            }
        drifted = self.drift_after is not None and self.calls > self.drift_after
        return {
            **runtime(),
            "versions": {"torch": f"drifted-{self.calls}" if drifted else "fixture"},
        }


def quiet_host(monkeypatch):
    """Launch checks see a fixed CPU host and never call nvidia-smi."""
    monkeypatch.setattr("histopilot.application.training.host_snapshot", lambda: dict(HOST))
    for module in (
        "histopilot.application.training",
        "histopilot.workers.training_process",
        "histopilot.taskcenter.adapters.mil",
    ):
        monkeypatch.setattr(f"{module}.gpu_snapshot", lambda: {"gpus": []})


def training_service(store, filesystem, center, monkeypatch, runtime=None):
    """A Task Center TrainingService whose ``prepared`` lists the batches it checked.

    Submission preflight checks a synthetic ``submission-preflight`` batch; a launch
    checks the batch it launches.
    """
    quiet_host(monkeypatch)
    service = TrainingService(
        store,
        filesystem,
        runtime=runtime or Runtime(),
        task_center=center.client,
    )
    service.prepared = []
    prepare = service._prepare

    def recorded(batch):
        service.prepared.append(batch["id"])
        return prepare(batch)

    monkeypatch.setattr(service, "_prepare", recorded)
    return service


def preflights(training):
    return training.prepared.count("submission-preflight")


# -- development batches -------------------------------------------------------------------


def development_batch(tmp_path):
    """A 30-slide project with packed features, a 5-fold protocol and a grid batch spec.

    Returns the ``DevelopmentService``, the unfrozen spec and the feature source folder.
    """
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-batches")
    rows = [
        {
            "slideId": f"s{i:02}",
            "patientId": f"p{i:02}",
            "attributes": {"label": str(i % 2), "cohort": "development"},
        }
        for i in range(30)
    ]
    data, rows = dataset(store, rows=rows)
    features, _pack_id, source = bundle(
        store, tmp_path, data, [row["slideId"] for row in rows], pack=True
    )
    protocol_spec = {
        "datasetId": data["id"],
        "target": TARGET,
        "split": {
            "version": 4,
            "mode": "kfold",
            "folds": 5,
            "seeds": [42],
            "pools": {"trainSelection": "remaining"},
        },
    }
    draft = store.create_draft(
        "experiment", "Development protocol", {"type": "analysis-protocol", "spec": protocol_spec}
    )
    protocols = ProtocolService(store, LocalFilesystem((tmp_path,)))
    preview = protocols.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    protocol = protocols.freeze(draft["id"], 1, preview["previewHash"], "protocol")
    spec = DevelopmentBatchSpec(
        experimentName="Optimizer tuning",
        batchName="Sweep v1",
        inputs={"protocolId": protocol["id"], "featureBundleId": features["id"]},
        mode="grid",
        grid={
            "learningRates": [0.0001, 0.0003, 0.001],
            "weightDecays": [0, 0.0001],
            "maxEpochs": [50, 100],
        },
        trainingSeeds=[10, 20, 30],
    )
    return DevelopmentService(store, LocalFilesystem((tmp_path,))), spec, source


@pytest.fixture
def tc_execution(tmp_path, monkeypatch, task_center):
    """The same batch run as production runs it: one Task Center task per fold."""
    monkeypatch.setattr("histopilot.application.training.gpu_snapshot", lambda: {"gpus": []})
    monkeypatch.setattr("histopilot.workers.training_process.gpu_snapshot", lambda: {"gpus": []})
    development, spec, source = development_batch(tmp_path)
    values = spec.model_dump()
    values.update(mode="single", trainingSeeds=[11])
    values["recipe"].update(maxEpochs=1, bagSize=2, batchSize=2)
    spec = DevelopmentBatchSpec.model_validate(values)
    preview = development.preview(spec)
    frozen = development.freeze(
        spec, preview["previewHash"], "execution-batch", {"tag": "Executable batch"}
    )
    # New specs omit resources; the Task Center's per-run defaults stay tiny CPU settings.
    task_center.store.update_settings({"defaults": {"cpuThreadsPerRun": 1, "dataLoaderWorkers": 0}})
    service = TrainingService(
        development.store,
        development.filesystem,
        runtime=runtime,
        task_center=task_center.client,
    )
    return service, frozen, source


def rewrite_batch(service, original, change):
    """Publish a copy of a frozen batch whose manifest ``change`` edited in place."""
    manifest = copy.deepcopy(original["manifest"])
    change(manifest)
    return service.store.publish_configuration(manifest=manifest, operation_id="modified-batch")


def synthetic_results(service, frozen):
    """The prepared batch and a completed state whose runs predicted every test slide."""
    batch, _guard = service._prepare(frozen)
    rows = []
    for run in batch["runs"]:
        folder = service._folder(frozen["id"]) / "runs" / run["id"]
        folder.mkdir(parents=True)
        records = []
        for membership in batch["memberships"][run["splitPlanId"]]:
            if membership["partition"] == "test":
                index = batch["target"]["classes"].index(membership["label"])
                records.append(
                    {
                        "slideId": membership["slideId"],
                        "patientId": membership["patientId"],
                        "label": membership["label"],
                        "labelIndex": index,
                        "probabilities": [0.9, 0.1] if index == 0 else [0.1, 0.9],
                    }
                )
        predictions = folder / "assessment-predictions.json"
        write_json_atomic(
            predictions, {"records": records, "classOrder": batch["target"]["classes"]}
        )
        result = {
            "runId": run["id"],
            "state": "succeeded",
            "predictions": {"assessment": str(predictions)},
        }
        write_json_atomic(folder / "result.json", result)
        rows.append({**run, "status": "completed", "result": result, "outputPath": str(folder)})
    return batch, {"status": "completed", "runs": rows}


# -- the tasks of a training batch ---------------------------------------------------------


def launched(center):
    """Ids of the batches whose tasks were enqueued (one final collection each)."""
    return [
        task["group"]["id"]
        for task in center.tasks(kind="mil-collect")
        if task["adapterData"].get("final")
    ]


def batch_tasks(center, identity, kind=None):
    """The tasks queued for one training batch: one fold per run, then the final collection."""
    return center.tasks(group=identity, kind=kind)


def attempts(center, identity):
    """Each task of the batch with its attempt, to tell a replay from another enqueue."""
    return sorted((task["id"], task["attempt"]) for task in batch_tasks(center, identity))


def lose_batch(center, identity, *, completed=(), state="interrupted"):
    """End every live task of the batch as ``state``, by default as a lost runner leaves it.

    Folds of the runs in ``completed`` finished first and end succeeded. Afterwards no task
    can move the batch any more, so it reads as its saved runs say.
    """
    for task in batch_tasks(center, identity):
        if task["state"] not in LIVE:
            continue
        run_id = (task.get("adapterData") or {}).get("runId")
        outcome = "succeeded" if run_id is not None and run_id in completed else state
        if outcome == "interrupted":
            center.finish(task["id"], outcome, returncode=None, reason="lost")
        else:
            center.finish(task["id"], outcome, returncode=0 if outcome == "succeeded" else 1)


def finish_batch(center, batch_id, *, fit="succeeded"):
    """Run every live fold of a batch to ``fit`` and apply its final results collection.

    Each fold leaves the receipt its worker writes (``result.json``, or ``failure.json``
    with exit code 1); the fold adapter records it in ``state.json``. A partial collection
    left behind is skipped once no fold remains, as the runner does; the final collection
    then writes its receipt and the collect adapter applies the batch's final status.
    """
    folds = MilFoldAdapter()
    for task in batch_tasks(center, batch_id, "mil-fold"):
        if task["state"] not in LIVE:
            continue
        center.start(task["id"])
        run_id = task["adapterData"]["runId"]
        output = Path(task["adapterData"]["batchFolder"]) / "runs" / run_id
        output.mkdir(parents=True, exist_ok=True)
        if fit == "succeeded":
            checkpoint = output / "best.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            write_json_atomic(
                output / "result.json",
                {
                    "runId": run_id,
                    "state": "succeeded",
                    "metrics": {"validation": {}},
                    "bestCheckpointPath": str(checkpoint),
                    "epochsCompleted": 1,
                },
            )
            returncode = 0
        else:
            write_json_atomic(
                output / "failure.json",
                {
                    "error": "Synthetic training failure",
                    "type": "RuntimeError",
                    "category": "error",
                    "at": utc_now(),
                },
            )
            returncode = 1
        conclude(center.store, task["id"], folds, returncode)
    collect = MilCollectAdapter()
    center.store.promote_ready()  # as the runner's tick does before it admits a task
    for task in batch_tasks(center, batch_id, "mil-collect"):
        if task["state"] not in LIVE or not begin(center.store, task["id"], collect):
            continue
        assert task["adapterData"].get("final"), f"{task['id']} is a partial collection"
        folder = Path(task["adapterData"]["batchFolder"])
        state = read_json_bounded(folder / "state.json")
        status = final_status(state, cancel_requested=(folder / "cancel.json").exists())
        write_json_atomic(
            folder / "collect-result.json",
            {"status": status, "final": True, "at": utc_now(), "error": None},
        )
        conclude(center.store, task["id"], collect)
