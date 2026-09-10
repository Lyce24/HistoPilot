"""Process and filesystem behavior required by folder-owned persistence."""

import os
import subprocess
import sys

import pytest

from histopilot.storage.project_lock import (
    LOCK_FILE,
    StorageError,
    ensure_managed_directory,
    fsync_directory,
    writer_lock,
)


def test_project_lock_coordinates_processes_and_releases_after_process_death(tmp_path):
    code = """
import sys
from pathlib import Path
from histopilot.storage.project_lock import writer_lock
with writer_lock(Path(sys.argv[1])):
    print('locked', flush=True)
    sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        inode = (tmp_path / LOCK_FILE).stat().st_ino
        with pytest.raises(StorageError) as error, writer_lock(tmp_path):
            pytest.fail("A second process must not acquire an active project writer lock.")
        assert error.value.code == "PROJECT_BUSY"
        process.kill()
        process.wait(timeout=10)
        with writer_lock(tmp_path):
            assert (tmp_path / LOCK_FILE).stat().st_ino == inode
        assert (tmp_path / LOCK_FILE).is_file()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory"])
def test_writer_lock_rejects_nonregular_or_aliased_inode(tmp_path, kind):
    folder = tmp_path / "project"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"unchanged")
    lock = folder / LOCK_FILE
    if kind == "symlink":
        lock.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, lock)
    elif kind == "fifo":
        os.mkfifo(lock)
    else:
        lock.mkdir()
    with pytest.raises(StorageError) as error, writer_lock(folder):
        pytest.fail("Unsafe lock path must not be acquired.")
    assert error.value.status_code == 403
    assert outside.read_bytes() == b"unchanged"


def test_managed_directories_reject_symlink_ancestors(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StorageError):
        ensure_managed_directory(tmp_path / "alias" / "new")
    assert not (outside / "new").exists()
    real = tmp_path / "project" / "datasets"
    ensure_managed_directory(real)
    assert real.is_dir()


def test_unsupported_directory_sync_fails_explicitly(tmp_path, monkeypatch):
    def fail(_descriptor):
        raise OSError("Injected directory synchronization failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(StorageError) as error:
        fsync_directory(tmp_path)
    assert error.value.code == "STORAGE_SYNC_FAILED"
