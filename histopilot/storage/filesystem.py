"""Root-confined directory discovery and explicit creation of empty child folders."""

import errno
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

    def create_directory(self, parent_value: str, name: str) -> dict:
        """Create exactly one new child; never follow links or replace an existing entry."""
        original_name = name
        name = name.strip()
        try:
            encoded_name = name.encode("utf-8")
            parent_value.encode("utf-8")
        except UnicodeEncodeError:
            raise FilesystemError(
                "Folder names and paths must contain valid Unicode text.", 422
            ) from None
        if (
            not name
            or name in {".", ".."}
            or any(character in name for character in ("/", "\\"))
            or any(ord(character) < 32 or ord(character) == 127 for character in original_name)
            or len(encoded_name) > 255
        ):
            raise FilesystemError(
                "Enter one folder name without slashes, traversal, or control characters.", 422
            )
        requested = Path(parent_value)
        if ".." in requested.parts:
            raise FilesystemError("Choose a parent folder without '..' path components.", 422)
        parent = self.directory(parent_value)
        if requested != parent:
            raise FilesystemError(
                "Create folders in the actual parent location, not through symbolic links.", 403
            )
        if os.name != "posix":
            raise FilesystemError("Folder creation requires POSIX filesystem support.", 501)
        descriptor = None
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            descriptor = os.open(parent.anchor, flags)
            # Open each parent by descriptor so a link substituted after resolution
            # cannot redirect the mutation outside the configured roots.
            for component in parent.parts[1:]:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            current = parent.stat(follow_symlinks=False)
            opened = os.fstat(descriptor)
            if parent.resolve(strict=True) != parent or (current.st_dev, current.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise FilesystemError(
                    "The parent folder changed. Browse it again before creating a folder.", 409
                )
            os.mkdir(name, mode=0o755, dir_fd=descriptor)
            created = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            try:
                after = parent.stat(follow_symlinks=False)
                unchanged = parent.resolve(strict=True) == parent and (
                    after.st_dev,
                    after.st_ino,
                ) == (opened.st_dev, opened.st_ino)
            except (OSError, RuntimeError):
                unchanged = False
            if not unchanged:
                # A directory can also be renamed after opening it. Undo only our
                # newly empty child through the pinned parent, never another entry
                # or a directory to which files have already been added.
                try:
                    child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if (child.st_dev, child.st_ino) == (created.st_dev, created.st_ino):
                        os.rmdir(name, dir_fd=descriptor)
                except OSError:
                    pass
                raise FilesystemError(
                    "The parent folder moved during creation. Browse it again.", 409
                )
        except FileExistsError:
            raise FilesystemError(
                "A file or folder with this name already exists. Choose another name.", 409
            ) from None
        except FileNotFoundError:
            raise FilesystemError(
                "The parent folder no longer exists. Browse to an existing folder.", 404
            ) from None
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                message, status = (
                    "The parent folder changed or contains a symbolic link. Browse it again.",
                    403,
                )
            elif error.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
                message, status = (
                    "HistoPilot does not have permission to create a folder here, or the filesystem is read-only.",
                    403,
                )
            elif error.errno == errno.ENOSPC:
                message, status = "There is not enough space to create a folder here.", 507
            else:
                message, status = "The folder could not be created in this location.", 400
            raise FilesystemError(message, status) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return {"path": str(parent / name), "name": name, "parent": str(parent)}

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
