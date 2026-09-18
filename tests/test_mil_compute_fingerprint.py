"""Training score and stopping changes must invalidate frozen execution identity."""

import pytest

from histopilot.storage.project_lock import StorageError
from histopilot.workers import training_process
from histopilot.workers.compute_archive import prepare_compute_archive


@pytest.mark.parametrize("name", ["scoring.py", "schemas/training_controls.py"])
def test_compute_fingerprint_tracks_metric_and_training_policy_changes(tmp_path, monkeypatch, name):
    expected = training_process.compute_snapshot()
    assert name in expected["files"]
    archive = prepare_compute_archive(tmp_path / "original", expected)
    package = archive / "histopilot"
    monkeypatch.setattr(training_process, "__file__", str(package / "workers/training_process.py"))
    assert training_process.compute_snapshot() == expected
    source = package / name
    source.write_text(source.read_text() + "\n# Changed scoring or stopping implementation.\n")
    assert training_process.compute_snapshot()["sha256"] != expected["sha256"]
    with pytest.raises(StorageError, match="does not match the frozen execution"):
        prepare_compute_archive(tmp_path / "new", expected, source_root=package)
