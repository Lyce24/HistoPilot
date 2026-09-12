import hashlib
import json

import pytest

from histopilot.storage.project_lock import StorageError
from histopilot.workers.compute_archive import prepare_compute_archive


def source(tmp_path):
    package = tmp_path / "source"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "train.py").write_text("VALUE = 1\n")
    (package / "helper.py").write_text("HELPER = 1\n")
    files = {"train.py": hashlib.sha256((package / "train.py").read_bytes()).hexdigest()}
    code = {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }
    return package, code


def test_original_archive_survives_changes_to_current_app(tmp_path):
    package, code = source(tmp_path)
    archive = prepare_compute_archive(tmp_path / "run", code, source_root=package)
    (package / "train.py").write_text("VALUE = 2\n")
    assert prepare_compute_archive(tmp_path / "run", code, source_root=package) == archive
    assert (archive / "histopilot/train.py").read_text() == "VALUE = 1\n"
    with pytest.raises(StorageError, match="does not match"):
        prepare_compute_archive(tmp_path / "other-run", code, source_root=package)


@pytest.mark.parametrize("mutation", ["change", "add", "remove", "symlink"])
def test_archive_rejects_tampering_even_outside_compute_file_subset(tmp_path, mutation):
    package, code = source(tmp_path)
    archive = prepare_compute_archive(tmp_path / "run", code, source_root=package)
    helper = archive / "histopilot/helper.py"
    if mutation == "change":
        helper.write_text("HELPER = 2\n")
    elif mutation == "add":
        (archive / "histopilot/extra.py").write_text("EXTRA = 1\n")
    else:
        helper.unlink()
        if mutation == "symlink":
            helper.symlink_to(package / "helper.py")
    with pytest.raises(StorageError):
        prepare_compute_archive(tmp_path / "run", code, source_root=package)


def test_failed_snapshot_does_not_create_an_archive(tmp_path):
    package, code = source(tmp_path)
    code["sha256"] = "invalid"
    with pytest.raises(StorageError, match="fingerprint is invalid"):
        prepare_compute_archive(tmp_path / "run", code, source_root=package)
    assert not (tmp_path / "run/compute").exists()


def test_archive_rejects_another_frozen_fingerprint(tmp_path):
    package, code = source(tmp_path)
    prepare_compute_archive(tmp_path / "run", code, source_root=package)
    changed = {"files": {"train.py": "a" * 64}}
    changed["sha256"] = hashlib.sha256(
        json.dumps(changed["files"], sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(StorageError, match="different compute code"):
        prepare_compute_archive(tmp_path / "run", changed, source_root=package)
