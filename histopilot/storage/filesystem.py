"""Read-only directory discovery, confined to explicitly configured roots."""

import os
from pathlib import Path


class FilesystemError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class LocalFilesystem:
    LIST_LIMIT = 200

    def __init__(self, roots: tuple[Path, ...]):
        self.roots = tuple(dict.fromkeys(root.resolve() for root in roots))

    def root_listing(self) -> dict:
        return {
            "roots": [{"path": str(root), "name": root.name or str(root)} for root in self.roots]
        }

    def directory(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute() or "\x00" in value:
            raise FilesystemError("Select an absolute path within a configured data root.")
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            raise FilesystemError("The selected directory does not exist.", 404) from None
        except (OSError, RuntimeError):
            raise FilesystemError("The selected directory cannot be resolved.", 403) from None
        if not self._contains(resolved):
            raise FilesystemError("The selected path is outside configured data roots.", 403)
        if not resolved.is_dir():
            raise FilesystemError("Select a directory; file contents are not served.")
        return resolved

    def _contains(self, path: Path) -> bool:
        return any(path.is_relative_to(root) for root in self.roots)

    def list_directory(self, value: str) -> dict:
        path = self.directory(value)
        entries = []
        truncated = False
        try:
            with os.scandir(path) as listing:
                for index, item in enumerate(listing):
                    if index >= self.LIST_LIMIT:
                        truncated = True
                        break
                    try:
                        resolved = Path(item.path).resolve(strict=True)
                        if not self._contains(resolved):
                            continue
                        if item.is_dir():
                            kind = "directory"
                        elif item.is_file():
                            kind = "file"
                        else:
                            continue
                    except (OSError, RuntimeError):
                        continue
                    entries.append({"name": item.name, "path": str(resolved), "kind": kind})
        except OSError:
            raise FilesystemError("The selected directory cannot be listed.", 403) from None
        entries.sort(key=lambda entry: (entry["kind"] != "directory", entry["name"].casefold()))
        parent = (
            str(path.parent) if path not in self.roots and self._contains(path.parent) else None
        )
        return {"path": str(path), "parent": parent, "entries": entries, "truncated": truncated}
