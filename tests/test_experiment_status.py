"""One experiment execution status: needs attention, not failed (hardening 1.4)."""

import sys
import threading
import time
from types import SimpleNamespace

import pytest

from histopilot.application.model_experiments import (
    ModelExperimentService,
    current_batch_ids,
    execution_status,
    task_execution,
)
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.taskcenter import TaskCenterClient, TaskStore, ids


def task(state, **extra):
    return {"state": state, "waitingReason": None, "bookkeeping": None, "error": None, **extra}


def test_task_states_map_to_one_experiment_status():
    assert task_execution([]) is None
    assert task_execution([task("succeeded"), task("running")]) == ("running", None)
    assert task_execution(
        [task("succeeded"), task("queued", waitingReason="Waiting for a GPU.")]
    ) == (
        "queued",
        "Waiting for a GPU.",
    )
    assert task_execution([task("queued")], held=True)[0] == "held"
    assert "runner is stopped" in task_execution([task("queued")], runner_alive=False)[1]
    assert task_execution([task("succeeded"), task("blocked")]) == (
        "waiting",
        "Waiting for earlier tasks to finish.",
    )
    # A coordinator that met a busy project and will be requeued is waiting, not failed.
    busy = task("failed", bookkeeping={"hook": "requeue_intent"})
    assert task_execution([task("succeeded"), busy])[0] == "waiting"
    assert task_execution([task("succeeded"), task("failed", error="Out of memory")]) == (
        "needs-attention",
        "Out of memory",
    )
    assert task_execution([task("succeeded"), task("interrupted")])[0] == "needs-attention"
    assert task_execution([task("succeeded"), task("cancelled")]) == ("cancelled", None)
    assert task_execution([task("succeeded")] * 3) == ("completed", None)


def test_only_the_experiments_current_batches_and_final_collections_count():
    """A failed fold of a trashed batch, or a progress collection its final collection
    superseded, must not pin a finished experiment to needs-attention forever."""
    old = task("failed", group={"kind": "mil-batch", "id": "old"}, error="Out of memory")
    new = task("succeeded", group={"kind": "mil-batch", "id": "new"})
    progress = task(
        "failed",
        kind="mil-collect",
        group={"kind": "mil-batch", "id": "new"},
        adapterData={"final": False},
    )
    final = task(
        "succeeded",
        kind="mil-collect",
        group={"kind": "mil-batch", "id": "new"},
        adapterData={"final": True},
    )
    coordinator = task("succeeded", group={"kind": "predictor-coordinator", "id": "draft-1"})
    tasks = [old, new, progress, final, coordinator]
    assert task_execution(tasks)[0] == "needs-attention"  # unscoped: every batch counts
    assert task_execution(tasks, batch_ids={"new"}) == ("completed", None)
    cancelled = task("cancelled", group={"kind": "mil-batch", "id": "old"})
    assert task_execution([new, cancelled], batch_ids={"new"}) == ("completed", None)
    # Live work of any batch still holds the experiment; tasks without a batch count.
    assert (
        task_execution(
            [new, task("running", group={"kind": "mil-batch", "id": "old"})], batch_ids={"new"}
        )[0]
        == "running"
    )
    failed_coordinator = task("failed", group={"kind": "predictor-coordinator", "id": "d"})
    assert task_execution([new, failed_coordinator], batch_ids={"new"})[0] == "needs-attention"
    assert task_execution([old], batch_ids={"new"}) is None


def test_live_work_outranks_attention_and_completion_needs_every_part():
    assert execution_status(["completed", "attention"]) == "needs-attention"
    assert execution_status(["completed", "failed", "interrupted"]) == "needs-attention"
    assert execution_status(["completed", "waiting", "attention"]) == "waiting"
    assert execution_status(["completed", "cancelling"]) == "running"
    assert execution_status(["completed", "cancelled"]) == "cancelled"
    assert execution_status(["completed", "completed"]) == "completed"
    assert execution_status(["planned"]) is None


@pytest.fixture
def service(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    return ModelExperimentService(
        SimpleNamespace(folder=folder, project_id="project-1"),
        None,
        training=SimpleNamespace(),
    )


def status(service, *, batches=("completed", "completed"), predictors=None, submission=None):
    return service._execution_status(
        "draft-1",
        {"payload": {"frozenSetupId": "setup"}},
        "running",
        [
            {"id": f"batch-{index}", "state": "active", "status": value}
            for index, value in enumerate(batches)
        ],
        [],
        {"protocolId": "p"},
        {"status": "submitted", "executionMode": "task-center", **(submission or {})},
        predictors,
    )


def test_completed_folds_with_a_coordinator_needing_attention_are_not_failed(service, monkeypatch):
    """Two experiments: 30/30 folds completed, the coordinator died of PROJECT_BUSY."""
    busy = {
        "status": "attention",
        "error": {
            "code": "PROJECT_BUSY",
            "message": "Another operation is changing this workspace.",
        },
    }
    monkeypatch.setattr(
        service, "_task_execution", lambda _identity, _batches=None: ("needs-attention", None)
    )
    assert status(service, predictors=busy) == (
        "needs-attention",
        "Another operation is changing this workspace.",
    )
    # Once the Task Center requeues the coordinator, its saved attention no longer wins.
    monkeypatch.setattr(
        service,
        "_task_execution",
        lambda _identity, _batches=None: ("waiting", "Retrying automatically."),
    )
    assert status(service, predictors=busy) == ("waiting", "Retrying automatically.")
    monkeypatch.setattr(
        service, "_task_execution", lambda _identity, _batches=None: ("held", "Held.")
    )
    assert status(service, predictors=busy)[0] == "held"
    # With no live task, saved "running" evidence cannot claim work is moving.
    monkeypatch.setattr(
        service, "_task_execution", lambda _identity, _batches=None: ("completed", None)
    )
    assert status(
        service, batches=("completed", "running"), predictors={"status": "completed"}
    ) == (
        "completed",
        None,
    )


def test_tasks_of_trashed_batches_do_not_pin_the_experiment_status(service, monkeypatch):
    seen = []

    def execution(_identity, batch_ids=None):
        seen.append(batch_ids)
        return ("completed", None) if batch_ids == {"batch-1"} else None

    monkeypatch.setattr(service, "_task_execution", execution)
    result = service._execution_status(
        "draft-1",
        {"payload": {"frozenSetupId": "setup"}},
        "running",
        [
            {"id": "batch-0", "state": "trashed", "status": "failed"},
            {"id": "batch-1", "state": "active", "status": "completed"},
            {"id": "batch-2", "state": "active", "status": "planned"},  # never submitted
        ],
        [],
        {"protocolId": "p"},
        # A trashed batch stays out even when the submission still names it.
        {
            "status": "submitted",
            "executionMode": "task-center",
            "batchIds": ["batch-0", "batch-1"],
        },
        None,
    )
    assert seen == [{"batch-1"}]
    assert result == ("completed", None)
    rows = [{"id": "a", "state": "active"}, {"id": "b", "state": "trashed"}]
    assert current_batch_ids(rows, None) == {"a"}  # before a submission: its own batches


def test_legacy_experiments_derive_the_same_vocabulary(service):
    legacy = {"executionMode": None}
    assert status(service, submission=legacy)[0] == "completed"
    assert status(service, batches=("completed", "failed"), submission=legacy)[0] == (
        "needs-attention"
    )
    assert status(service, predictors={"status": "waiting"}, submission=legacy) == (
        "waiting",
        "Waiting for the experiment's fold batches to finish.",
    )
    unresolved = status(
        service,
        submission={"executionMode": None, "status": "attention", "error": {"message": "Lost."}},
    )
    assert unresolved == ("needs-attention", "Lost.")
    assert status(service, submission={"executionMode": None, "status": "launching"})[0] == "queued"


def test_task_center_status_is_read_from_the_owner_without_the_project_lock(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = TaskStore(tmp_path / "state" / "task-center.sqlite")
    client = TaskCenterClient(store)
    service = ModelExperimentService(
        SimpleNamespace(folder=folder, project_id="project-1"),
        None,
        training=SimpleNamespace(task_center=client),
    )
    assert service._task_execution("draft-1") is None
    owner = {
        "kind": "experiment",
        "id": "draft-1",
        "projectId": "project-1",
        "projectFolder": str(folder),
        "title": "study3",
    }
    store.enqueue(
        owner,
        [
            {
                "id": "fold-1",
                "kind": "mil-fold",
                "adapter": "generic",
                "title": "Fold 1",
                "group": {"kind": "mil-batch", "id": "batch-1"},
                "request": {"lane": "gpu", "vramGb": 1.0},
                "command": {
                    "argv": [sys.executable, "-c", "pass"],
                    "cwd": str(tmp_path),
                    "log": str(tmp_path / "fold-1.log"),
                },
            }
        ],
    )
    key = ids.owner_key("experiment", "draft-1", str(folder))
    held, release = threading.Event(), threading.Event()

    def hold():
        with lifecycle_guard(folder):
            held.set()
            release.wait(30)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(10)
    try:
        start = time.monotonic()
        # No runner in this test: queued work waits for it.
        assert service._task_execution("draft-1")[0] == "waiting"
        store.hold_owner(key, True)
        assert service._task_execution("draft-1")[0] == "held"
        store.hold_owner(key, False)
        assert store.transition("fold-1", from_states={"queued"}, to_state="running")
        assert service._task_execution("draft-1") == ("running", None)
        assert store.transition(
            "fold-1", from_states={"running"}, to_state="failed", error="CUDA out of memory"
        )
        assert service._task_execution("draft-1") == ("needs-attention", "CUDA out of memory")
        assert time.monotonic() - start < 3
        # Once that batch is trashed, its failed fold no longer speaks for the experiment.
        assert service._task_execution("draft-1", {"batch-2"}) is None
    finally:
        release.set()
        thread.join(10)
