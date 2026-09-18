"""Archive control metadata stays bounded before eager descriptor decoding."""

import hashlib
import json
import zipfile

import pytest

from histopilot.application.operations import MANIFEST, _archive_manifest, verify_archive
from histopilot.application.project_workspace import DESCRIPTOR
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import DATABASE_FILE


def archive(tmp_path, descriptor, *, extra_files=0):
    entries = {DESCRIPTOR: descriptor, DATABASE_FILE: b"database-placeholder"}
    entries.update({f"evidence/{index}.json": b"{}" for index in range(extra_files)})
    manifest = {
        "format": "histopilot-study-archive",
        "schemaVersion": 1,
        "projectId": "project-" + "a" * 32,
        "files": [
            {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in entries.items()
        ],
    }
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr(MANIFEST, json.dumps(manifest))
        for name, content in entries.items():
            zipped.writestr(f"project/{name}", content)
    return path


def test_oversized_descriptor_rejected_before_eager_read(tmp_path, monkeypatch):
    monkeypatch.setattr("histopilot.application.operations.DESCRIPTOR_LIMIT", 32)
    path = archive(tmp_path, b"x" * 33)
    eager_reads = []
    original = zipfile.ZipFile.read

    def observed_read(self, name, *args, **kwargs):
        eager_reads.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", observed_read)
    with pytest.raises(StorageError, match="descriptor exceeds") as error:
        verify_archive(path)
    assert error.value.status_code == 413
    assert f"project/{DESCRIPTOR}" not in eager_reads


def test_descriptor_limit_is_inclusive_for_large_file_inventories(tmp_path, monkeypatch):
    monkeypatch.setattr("histopilot.application.operations.DESCRIPTOR_LIMIT", 32)
    path = archive(tmp_path, b"x" * 32, extra_files=5000)
    with zipfile.ZipFile(path) as zipped:
        manifest = _archive_manifest(zipped)
    assert len(manifest["files"]) == 5002
