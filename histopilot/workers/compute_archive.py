"""Pin the Python worker package so unfinished runs can resume after app updates."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from histopilot.storage.project_lock import StorageError, _reject_symlink_components


def _digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _inventory(package: Path) -> dict[str, str]:
    _reject_symlink_components(package)
    if not package.is_dir():
        raise ValueError("The archived worker package is missing.")
    files = {}
    size = 0
    for path in sorted(package.rglob("*.py")):
        _reject_symlink_components(path)
        size += path.stat().st_size
        if size > 64 * 1024 * 1024 or len(files) >= 10000:
            raise ValueError("Worker source archive exceeds its size limit.")
        files[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "__init__.py" not in files:
        raise ValueError("The worker package is incomplete.")
    return files


def _check_expected(files: dict[str, str], expected: dict) -> None:
    selected = expected.get("files")
    if (
        not isinstance(selected, dict)
        or not selected
        or _digest(selected) != expected.get("sha256")
    ):
        raise ValueError("The frozen compute fingerprint is invalid.")
    for name, digest in selected.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("The frozen compute fingerprint contains an invalid source path.")
        if files.get(name) != digest:
            raise ValueError(f"Worker source does not match the frozen execution: {name}.")


def prepare_compute_archive(
    folder: Path, expected: dict, *, source_root: Path | None = None
) -> Path:
    """Return a verified import root at ``folder/compute``.

    Existing archives must match both their complete source inventory and the
    frozen compute fingerprint. A new archive is made only from matching source;
    source_root, when supplied, is the directory containing histopilot's Python
    modules (the histopilot directory itself). Never alter a mismatching archive.
    Python dependency compatibility remains the launch service's responsibility.
    """
    archive = Path(folder) / "compute"
    try:
        _reject_symlink_components(archive)
        if archive.exists():
            metadata = archive / "snapshot.json"
            _reject_symlink_components(metadata)
            if metadata.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Worker archive manifest exceeds its size limit.")
            manifest = json.loads(metadata.read_text())
            files = _inventory(archive / "histopilot")
            if manifest.get("version") != 1 or manifest.get("files") != files:
                raise ValueError("The archived worker source has changed.")
            if (
                manifest.get("packageSha256") != _digest(files)
                or manifest.get("compute") != expected
            ):
                raise ValueError("The worker archive belongs to different compute code.")
            _check_expected(files, expected)
            return archive

        source = source_root or Path(__file__).resolve().parents[1]
        files = _inventory(source)
        _check_expected(files, expected)
        Path(folder).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".compute-", dir=folder) as temporary:
            staging = Path(temporary) / "archive"
            package = staging / "histopilot"
            for name in files:
                target = package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            copied = _inventory(package)
            if copied != files:
                raise ValueError("Worker source changed while creating its archive.")
            manifest = {
                "version": 1,
                "compute": expected,
                "files": files,
                "packageSha256": _digest(files),
            }
            (staging / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
            staging.rename(archive)
        return archive
    except (OSError, ValueError, TypeError) as error:
        raise StorageError(
            f"Cannot use the frozen training code: {error} Clone a new batch if the original code is unavailable.",
            "TRAINING_RUNTIME_CHANGED",
        ) from error
