"""A saved file stamp still names its file after a restart renumbers the disk it is on."""

import os

import pytest

from histopilot.storage.packed import (
    PackedStoreError,
    open_source,
    require_same_stamp,
    same_stamp,
    stat_stamp,
)


def test_a_new_device_number_alone_is_the_same_file(tmp_path):
    path = tmp_path / "features.h5"
    path.write_bytes(b"features")
    saved = {"path": str(path), **stat_stamp(path.stat())}
    # WSL remounted the virtual disk as another block device: sdd (8,48) became sdc (8,32).
    renumbered = {**saved, "deviceId": saved["deviceId"] + 1}
    assert same_stamp(renumbered, saved) and same_stamp(saved, renumbered)
    require_same_stamp(renumbered, saved, path)
    with open_source(path, renumbered) as (stream, _):
        assert stream.read() == b"features"


@pytest.mark.parametrize("key", ["sizeBytes", "mtimeNs", "ctimeNs", "inode"])
def test_any_other_change_is_another_file(tmp_path, key):
    path = tmp_path / "features.h5"
    path.write_bytes(b"features")
    saved = {"path": str(path), **stat_stamp(path.stat())}
    changed = {**saved, key: saved[key] + 1}
    assert not same_stamp(changed, saved)
    with pytest.raises(PackedStoreError, match="changed since it was saved"):
        require_same_stamp(changed, saved, path)


def test_a_rewritten_file_and_another_path_are_refused(tmp_path):
    path = tmp_path / "features.h5"
    path.write_bytes(b"features")
    saved = {"path": str(path), **stat_stamp(path.stat())}
    assert not same_stamp({**saved, "path": str(tmp_path / "other.h5")}, saved)
    path.write_bytes(b"different features")
    os.utime(path, ns=(saved["mtimeNs"], saved["mtimeNs"]))
    with pytest.raises(PackedStoreError):
        with open_source(path, saved):
            pass
