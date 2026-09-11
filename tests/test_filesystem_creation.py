"""Explicit folder creation keeps every picker inside its configured filesystem roots."""

import errno
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.config import Settings
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem

ENDPOINT = "/api/v1/filesystem/directories"


@pytest.fixture
def folders(tmp_path):
    roots = (tmp_path / "drive-d", tmp_path / "oceanpath-hot")
    for root in roots:
        root.mkdir()
    return LocalFilesystem(roots), roots


def test_create_empty_child_in_each_root_and_preserve_existing_data(folders):
    filesystem, roots = folders
    for root in roots:
        parent = root / "features"
        parent.mkdir()
        original = parent / "sentinel.txt"
        original.write_text("original source")
        result = filesystem.create_directory(str(parent), "blca 研究")
        assert result == {
            "path": str(parent / "blca 研究"),
            "name": "blca 研究",
            "parent": str(parent),
        }
        assert list((parent / "blca 研究").iterdir()) == []
        assert original.read_text() == "original source"
        assert any(
            entry["path"] == result["path"]
            for entry in filesystem.list_directory(str(parent))["entries"]
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        ".",
        "..",
        "../escape",
        "a/b",
        "/absolute",
        "a\\b",
        "bad\x00name",
        "bad\nname",
        "trailing\n",
        "bad\x7fname",
        "\ud800",
        "研" * 100,
    ],
)
def test_names_are_single_safe_components(folders, name):
    filesystem, roots = folders
    with pytest.raises(FilesystemError) as error:
        filesystem.create_directory(str(roots[0]), name)
    assert error.value.status_code == 422
    assert not list(roots[0].iterdir())


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "broken-link"])
def test_existing_entries_are_never_replaced(folders, kind):
    filesystem, roots = folders
    path = roots[0] / "blca"
    if kind == "directory":
        path.mkdir()
    elif kind == "file":
        path.write_text("keep this")
    else:
        path.symlink_to(roots[1] if kind == "symlink" else roots[1] / "missing")
    before = path.lstat()
    with pytest.raises(FilesystemError) as error:
        filesystem.create_directory(str(roots[0]), "blca")
    assert error.value.status_code == 409
    assert path.lstat().st_ino == before.st_ino


def test_parent_must_exist_and_stay_inside_roots_without_links(folders, tmp_path):
    filesystem, roots = folders
    outside = tmp_path / "outside"
    outside.mkdir()
    (roots[0] / "external-link").symlink_to(outside)
    (roots[0] / "internal-link").symlink_to(roots[1])
    for parent in (outside, roots[0] / "external-link", roots[0] / "internal-link"):
        with pytest.raises(FilesystemError) as error:
            filesystem.create_directory(str(parent), "blca")
        assert error.value.status_code == 403
    for parent, status in (
        (str(roots[0] / "missing"), 404),
        (".", 400),
        (str(roots[0] / ".." / "outside"), 422),
    ):
        with pytest.raises(FilesystemError) as error:
            filesystem.create_directory(parent, "blca")
        assert error.value.status_code == status
    assert not list(outside.iterdir())


def test_link_swap_after_parent_resolution_cannot_redirect_creation(folders, tmp_path, monkeypatch):
    filesystem, roots = folders
    parent = roots[0] / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = os.open

    def replace_parent(path, flags, *args, **kwargs):
        if path == "parent":
            parent.rename(roots[0] / "original-parent")
            parent.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_parent)
    with pytest.raises(FilesystemError) as error:
        filesystem.create_directory(str(parent), "blca")
    assert error.value.status_code == 403
    assert not list(outside.iterdir())


def test_parent_moved_during_creation_does_not_leave_a_folder_outside_roots(
    folders, tmp_path, monkeypatch
):
    filesystem, roots = folders
    parent = roots[0] / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = os.mkdir

    def move_parent(path, *args, **kwargs):
        parent.rename(outside / "moved")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", move_parent)
    with pytest.raises(FilesystemError) as error:
        filesystem.create_directory(str(parent), "blca")
    assert error.value.status_code == 409
    assert list((outside / "moved").iterdir()) == []


@pytest.mark.parametrize(
    "failure,status", [(errno.EACCES, 403), (errno.EROFS, 403), (errno.ENOSPC, 507)]
)
def test_permission_and_storage_errors_are_actionable(folders, monkeypatch, failure, status):
    filesystem, roots = folders

    def fail(*args, **kwargs):
        raise OSError(failure, "simulated filesystem error")

    monkeypatch.setattr(os, "mkdir", fail)
    with pytest.raises(FilesystemError) as error:
        filesystem.create_directory(str(roots[0]), "blca")
    assert error.value.status_code == status


def test_concurrent_creates_keep_one_directory(folders):
    filesystem, roots = folders

    def create():
        try:
            filesystem.create_directory(str(roots[0]), "blca")
            return 201
        except FilesystemError as error:
            return error.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: create(), range(2))) == [201, 409]


def test_api_authentication_purposes_and_create_then_select(tmp_path, folders):
    _filesystem, roots = folders
    settings = Settings(workspace=tmp_path / "workspace", data_roots=roots)
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        payload = {"parentPath": str(roots[1]), "name": "blca"}
        assert client.post(ENDPOINT, json=payload).status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        assert (
            client.post(
                ENDPOINT, json=payload, headers={"Origin": "https://outside.invalid"}
            ).status_code
            == 403
        )
        assert client.post(ENDPOINT, json={**payload, "purpose": "arbitrary"}).status_code == 422
        assert client.post(ENDPOINT, json={**payload, "command": "ignored"}).status_code == 422
        created = client.post(ENDPOINT, json=payload)
        assert created.status_code == 201, created.text
        child = created.json()["path"]
        listing = client.get("/api/v1/filesystem/list", params={"path": child})
        assert listing.status_code == 200
        assert listing.json()["entries"] == []
        assert client.post(ENDPOINT, json=payload).status_code == 409
        workspace_child = {"parentPath": str(settings.workspace), "name": "experiment"}
        assert client.post(ENDPOINT, json=workspace_child).status_code == 403
        created = client.post(ENDPOINT, json={**workspace_child, "purpose": "storage"})
        assert created.status_code == 201, created.text
        assert (settings.workspace / "experiment").is_dir()


def test_api_without_source_roots_only_allows_workspace_creation(tmp_path):
    settings = Settings(workspace=tmp_path / "workspace")
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        payload = {"parentPath": str(settings.workspace), "name": "new"}
        assert client.post(ENDPOINT, json=payload).status_code == 403
        assert client.post(ENDPOINT, json={**payload, "purpose": "storage"}).status_code == 201
