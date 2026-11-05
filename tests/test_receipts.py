"""Saved receipts still name their outputs after a project moves to another folder."""

from pathlib import Path

import pytest

from histopilot.storage.receipts import moved_into, receipt_names

HERE = Path("/data/studies/lung/compute-jobs/configuration-abc/predictions.json")


@pytest.mark.parametrize(
    "recorded",
    [
        str(HERE),
        # Written before the project was restored from an archive into a new folder.
        "/home/someone/old-place/lung/compute-jobs/configuration-abc/predictions.json",
        "/mnt/backup/compute-jobs/configuration-abc/predictions.json",
    ],
)
def test_a_moved_project_keeps_naming_the_same_output(recorded):
    assert receipt_names(recorded, HERE)


@pytest.mark.parametrize(
    "recorded",
    [
        None,
        "",
        42,
        "/old/lung/compute-jobs/configuration-abc/metrics.json",  # another file
        "/old/lung/compute-jobs/configuration-def/predictions.json",  # another job
        "configuration-abc/predictions.json",  # not a recorded absolute path
        "/predictions.json",
    ],
)
def test_anything_else_is_not_the_same_output(recorded):
    assert not receipt_names(recorded, HERE)


def test_a_recorded_file_of_a_moved_run_folder_maps_into_the_current_one():
    run = Path("/data/studies/lung/training/batch/runs/run-1")
    recorded = "/old/lung/training/batch/runs/run-1/assessment-predictions.json"
    assert moved_into(recorded, run) == run / "assessment-predictions.json"
    assert moved_into(str(run / "plan.json"), run) == run / "plan.json"
    # Another run's file is not this run's: it comes back as recorded, for the caller to refuse.
    other = "/old/lung/training/batch/runs/run-2/assessment-predictions.json"
    assert moved_into(other, run) == Path(other)
