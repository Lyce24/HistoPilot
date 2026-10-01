"""Root-confined directory discovery and explicit creation of empty child folders."""

import errno
import os
from contextvars import ContextVar
from pathlib import Path

# Raise sites that name no code take one from their status, so each code keeps one kind.
STATUS_CODES = {400: "PATH_INVALID", 403: "PATH_REFUSED", 404: "PATH_NOT_FOUND"}


class FilesystemError(ValueError):
    def __init__(self, message: str, status_code: int = 400, *, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code or STATUS_CODES.get(status_code, "PATH_INVALID")


# A request made with an agent's scoped token reaches only its project's folder and
# registered sources, although the data roots are shared by every project.
SCOPED_ROOTS: ContextVar[tuple[Path, ...] | None] = ContextVar("scoped_roots", default=None)


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
            raise FilesystemError(
                "Select an absolute path within a configured data root.", code="PATH_NOT_ABSOLUTE"
            )
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            raise FilesystemError(
                "The selected directory does not exist.", 404, code="DIRECTORY_NOT_FOUND"
            ) from None
        except (OSError, RuntimeError):
            raise FilesystemError(
                "The selected directory cannot be resolved.", 403, code="PATH_UNRESOLVABLE"
            ) from None
        if not self._contains(resolved):
            raise FilesystemError(
                "The selected path is outside configured data roots.",
                403,
                code="PATH_OUTSIDE_ROOTS",
            )
        if not resolved.is_dir():
            raise FilesystemError(
                "Select a directory; file contents are not served.", code="PATH_NOT_DIRECTORY"
            )
        return resolved

    def _contains(self, path: Path) -> bool:
        if not any(path.is_relative_to(root) for root in self.roots):
            return False
        scoped = SCOPED_ROOTS.get()
        return scoped is None or any(path.is_relative_to(root) for root in scoped)

    def create_directory(self, parent_value: str, name: str) -> dict:
        """Create exactly one new child; never follow links or replace an existing entry."""
        original_name = name
        name = name.strip()
        try:
            encoded_name = name.encode("utf-8")
            parent_value.encode("utf-8")
        except UnicodeEncodeError:
            raise FilesystemError(
                "Folder names and paths must contain valid Unicode text.",
                422,
                code="FOLDER_NAME_INVALID",
            ) from None
        if (
            not name
            or name in {".", ".."}
            or any(character in name for character in ("/", "\\"))
            or any(ord(character) < 32 or ord(character) == 127 for character in original_name)
            or len(encoded_name) > 255
        ):
            raise FilesystemError(
                "Enter one folder name without slashes, traversal, or control characters.",
                422,
                code="FOLDER_NAME_INVALID",
            )
        requested = Path(parent_value)
        if ".." in requested.parts:
            raise FilesystemError(
                "Choose a parent folder without '..' path components.",
                422,
                code="FOLDER_PARENT_INVALID",
            )
        parent = self.directory(parent_value)
        if requested != parent:
            raise FilesystemError(
                "Create folders in the actual parent location, not through symbolic links.",
                403,
                code="FOLDER_PARENT_SYMLINK",
            )
        if os.name != "posix":
            raise FilesystemError(
                "Folder creation requires POSIX filesystem support.",
                501,
                code="FOLDER_CREATION_UNSUPPORTED",
            )
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
                    "The parent folder changed. Browse it again before creating a folder.",
                    409,
                    code="FOLDER_PARENT_CHANGED",
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
                    "The parent folder moved during creation. Browse it again.",
                    409,
                    code="FOLDER_PARENT_CHANGED",
                )
        except FileExistsError:
            raise FilesystemError(
                "A file or folder with this name already exists. Choose another name.",
                409,
                code="FOLDER_EXISTS",
            ) from None
        except FileNotFoundError:
            raise FilesystemError(
                "The parent folder no longer exists. Browse to an existing folder.",
                404,
                code="FOLDER_PARENT_NOT_FOUND",
            ) from None
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise FilesystemError(
                    "The parent folder changed or contains a symbolic link. Browse it again.",
                    403,
                    code="FOLDER_PARENT_SYMLINK",
                ) from error
            if error.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
                raise FilesystemError(
                    "HistoPilot does not have permission to create a folder here, or the filesystem is read-only.",
                    403,
                    code="FOLDER_PERMISSION_DENIED",
                ) from error
            if error.errno == errno.ENOSPC:
                raise FilesystemError(
                    "There is not enough space to create a folder here.",
                    507,
                    code="FOLDER_NO_SPACE",
                ) from error
            raise FilesystemError(
                "The folder could not be created in this location.",
                400,
                code="FOLDER_CREATE_FAILED",
            ) from error
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
            raise FilesystemError(
                "The selected directory cannot be listed.", 403, code="DIRECTORY_UNREADABLE"
            ) from None
        entries.sort(key=lambda entry: (entry["kind"] != "directory", entry["name"].casefold()))
        parent = (
            str(path.parent) if path not in self.roots and self._contains(path.parent) else None
        )
        return {"path": str(path), "parent": parent, "entries": entries, "truncated": truncated}
