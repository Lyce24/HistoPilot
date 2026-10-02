"""Path checks read once per folder behave exactly as the per-component pathlib checks did."""

import os
from pathlib import Path

import pytest

from histopilot.storage.filesystem import within
from histopilot.storage.project_lock import (
    StorageError,
    SymlinkCheck,
    reject_symlink_components,
)


def pathlib_rejects(path: Path) -> bool:
    """The check as it was: the path, or a parent, is a symbolic link."""
    return any(component.is_symlink() for component in (path, *path.parents))


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "slides" / "site").mkdir(parents=True)
    (tmp_path / "slides" / "site" / "a.svs").write_bytes(b"a")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "b.svs").write_bytes(b"b")
    (tmp_path / "slides" / "site" / "linked.svs").symlink_to(tmp_path / "elsewhere" / "b.svs")
    (tmp_path / "slides" / "via").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    return tmp_path


def test_single_and_batch_checks_agree_with_pathlib(tree):
    candidates = [
        tree / "slides" / "site" / "a.svs",
        tree / "slides" / "site" / "linked.svs",
        tree / "slides" / "via" / "b.svs",
        tree / "slides" / "site" / "missing.svs",
        tree / "slides" / "site" / ".." / "site" / "a.svs",
        tree / "slides" / "via",
        Path("relative") / "name.svs",
        Path("/"),
    ]
    batch = SymlinkCheck()
    for path in candidates + candidates:  # twice: the second pass reads memoized folders
        expected = pathlib_rejects(path)
        for check in (reject_symlink_components, batch):
            if expected:
                with pytest.raises(StorageError) as error:
                    check(path)
                assert error.value.code == "STORAGE_UNSAFE_PATH"
            else:
                check(path)


def test_an_unlistable_folder_falls_back_to_one_lstat(tree):
    folder = tree / "slides" / "site"
    os.chmod(folder, 0o300)  # traversable, not listable
    try:
        check = SymlinkCheck()
        check(folder / "a.svs")
        with pytest.raises(StorageError):
            check(folder / "linked.svs")
    finally:
        os.chmod(folder, 0o755)


@pytest.mark.parametrize(
    ("path", "root"),
    [
        ("/data/slides/a.svs", "/data"),
        ("/data", "/data"),
        ("/data2/a.svs", "/data"),
        ("/data/../etc/passwd", "/data"),
        ("/anything", "/"),
        ("relative/a.svs", "/data"),
        ("/data/slides", "/data/slides/a"),
    ],
)
def test_within_is_the_lexical_is_relative_to(path, root):
    assert within(Path(path), Path(root)) == Path(path).is_relative_to(Path(root))
