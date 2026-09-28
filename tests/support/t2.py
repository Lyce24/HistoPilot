"""Task Center fixtures for the batch, experiment and setup tests.

Features are validated by a packing task and batches launch as fold tasks, exactly as in
production. Nothing runs a real worker process: a pack job runs its worker in this process
under the task's identity, and a fold "finishes" when the receipts its worker would write
are in place and the runner's adapter has concluded the task from them.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.application.training import TrainingService
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.taskcenter.adapters.mil import MilCollectAdapter, MilFoldAdapter
from histopilot.taskcenter.adapters.packing import PackingAdapter
from histopilot.taskcenter.model import ACTIVE, LIVE, utc_now_iso
from histopilot.workers.managed_collect import final_status
from histopilot.workers.pack_features import run_job
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json

HOST = {
    "cpuCount": 8,
    "totalRamGb": 16.0,
    "availableRamGb": 16.0,
    "bootId": "test",
    "kernel": "test",
}


class Runtime:
    """A CPU-only training runtime probe that counts its calls.

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
            "available": True,
            "python": sys.executable,
            "versions": {"torch": f"drifted-{self.calls}" if drifted else "fixture"},
            "cudaAvailable": False,
            "gpuCount": 0,
            "findings": [],
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
        execution_mode="task-center",
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


# -- features ------------------------------------------------------------------------------


@contextmanager
def task_environment(task):
    """The identity the runner exports to a worker it spawns for ``task``."""
    values = {
        "HISTOPILOT_TASK_MANAGED": "1",
        "HISTOPILOT_TASK_ID": task["id"],
        "HISTOPILOT_TASK_ATTEMPT": str(task["attempt"]),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def exit_record(returncode=0, *, lost=False, stop=None):
    return {
        "returncode": returncode,
        "lost": lost,
        "signalled": False,
        "killed": False,
        "stopReason": stop,
    }


def conclude(center, task_id, adapter, exit=None):
    """Let ``adapter`` read the worker's receipts and conclude the task with its decision."""
    if center.state(task_id) not in ACTIVE:
        center.start(task_id)
    exit = exit or exit_record()
    decision = adapter.on_exit(center.task(task_id), exit, center.context())
    assert decision["state"] != "requeue", decision
    center.finish(
        task_id,
        decision["state"],
        returncode=exit["returncode"],
        reason=decision["exitReason"],
        error=decision["error"],
    )
    return decision


def run_pack(center, packs, job):
    """Run a queued feature job's worker here, under its task, and conclude the task."""
    task = center.task(job["taskId"])
    center.start(task["id"])
    with task_environment(center.task(task["id"])):
        result = run_job(packs.folder / job["id"] / "plan.json")
    assert conclude(center, task["id"], PackingAdapter())["state"] == result["state"]
    return result


def bundle(
    center,
    store,
    root,
    data,
    ids,
    name="features",
    dimension=4,
    encoder="uni_v1",
    pack=False,
    dtype="float32",
    feature_kind="patch",
):
    """``test_evaluations.bundle`` with the feature job run as a Task Center packing task."""
    source = root / name
    source.mkdir()
    for identity in ids:
        with h5py.File(source / f"{identity}.h5", "w") as handle:
            handle.create_dataset(
                "features",
                data=np.ones((1 if feature_kind == "slide" else 3, dimension), dtype=dtype),
            )
            if feature_kind == "patch":
                handle.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))
    filesystem = LocalFilesystem((root,))
    features = FeatureService(store, filesystem)
    spec = FeatureSpec(
        datasetId=data["id"], path=str(source), encoderId=encoder, featureKind=feature_kind
    )
    frozen = features.freeze(spec, features.preview(spec)["previewHash"], name)
    packs = FeaturePackService(
        store, filesystem, execution_mode="task-center", task_center=center.client
    )
    packing = FeaturePackSpec(featureSetId=frozen["id"], action="pack" if pack else "validate")
    job = packs.submit(packing, packs.preview(packing)["previewHash"], name + "-validation")
    assert center.state(job["taskId"]) == "queued"
    result = run_pack(center, packs, job)
    assert result["state"] == "succeeded", result
    assert packs.get(job["id"])["state"] == "succeeded"
    bundle_service = FeatureBundleService(store, filesystem)
    pack_id = result["artifact"]["id"] if pack else None
    spec = FeatureBundleSpec(
        featureSetId=frozen["id"], packArtifactIds=[pack_id] if pack_id else []
    )
    preview = bundle_service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    return bundle_service.freeze(spec, preview["previewHash"], name + "-bundle"), pack_id, source


# -- training batches ----------------------------------------------------------------------


def launched(center):
    """Ids of the batches whose tasks were enqueued (one final collection each)."""
    return [
        task["group"]["id"]
        for task in center.tasks(kind="mil-collect")
        if task["adapterData"].get("final")
    ]


def fold_tasks(center, batch_id):
    return center.tasks(kind="mil-fold", group=batch_id)


def finish_batch(center, batch_id, *, fit="succeeded"):
    """Run every live fold of a batch to ``fit`` and apply its final results collection.

    Each fold leaves the receipt its worker writes (``result.json``, or ``failure.json``
    with exit code 1); the fold adapter records it in ``state.json``. A partial collection
    left behind is skipped once no fold remains, as the runner does; the final collection
    then writes its receipt and the collect adapter applies the batch's final status.
    """
    folds = MilFoldAdapter()
    for task in fold_tasks(center, batch_id):
        if task["state"] not in LIVE:
            continue
        center.start(task["id"])
        run_id = task["adapterData"]["runId"]
        output = Path(task["adapterData"]["batchFolder"]) / "runs" / run_id
        output.mkdir(parents=True, exist_ok=True)
        if fit == "succeeded":
            checkpoint = output / "best.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            write_json(
                output / "result.json",
                {
                    "runId": run_id,
                    "state": "succeeded",
                    "metrics": {"validation": {}},
                    "bestCheckpointPath": str(checkpoint),
                    "epochsCompleted": 1,
                },
            )
            exit = exit_record(0)
        else:
            write_json(
                output / "failure.json",
                {
                    "error": "Synthetic training failure",
                    "type": "RuntimeError",
                    "category": "error",
                    "at": utc_now_iso(),
                },
            )
            exit = exit_record(1)
        conclude(center, task["id"], folds, exit)
    collect = MilCollectAdapter()
    for task in center.tasks(kind="mil-collect", group=batch_id):
        if task["state"] not in LIVE:
            continue
        ctx = center.context()
        if not task["adapterData"].get("final"):
            skipped = collect.prepare(task, ctx)["skip"]
            center.finish(task["id"], skipped["state"], reason=skipped["exitReason"])
            continue
        assert collect.prepare(task, ctx) is None
        folder = Path(task["adapterData"]["batchFolder"])
        state = read_json(folder / "state.json")
        status = final_status(state, cancel_requested=(folder / "cancel.json").exists())
        center.start(task["id"])
        write_json(
            folder / "collect-result.json",
            {"status": status, "final": True, "at": utc_now_iso(), "error": None},
        )
        conclude(center, task["id"], collect)


def interrupt_batch(center, batch_id):
    """Every live task of the batch is lost, as in a host restart with no runner."""
    for task in center.tasks(group=batch_id):
        if task["state"] in LIVE:
            center.finish(task["id"], "interrupted", returncode=None, reason="lost")
