"""Bind a saved output receipt to its file, wherever the project folder is now.

Workers record each output's absolute path, size and SHA-256 in a receipt. A project copied
or restored into another folder (Study backups & sources) keeps its receipts, so every
recorded path still names the old folder. The output itself moved with the project: its own
folder, a job or run folder named by its identity, and its file name are unchanged. Readers
therefore accept a recorded path that names the same file below that folder, and go on to
check size and SHA-256, which is what proves the content.

Only readers use this. Anything that resumes or writes work keeps comparing full paths,
because a plan written in another folder would otherwise write there.
"""

from pathlib import Path, PurePath


def receipt_names(recorded: object, path: Path) -> bool:
    """Whether a receipt's recorded path names `path`, in this folder or a moved copy."""
    if not isinstance(recorded, str) or not recorded:
        return False
    if recorded == str(path):
        return True
    saved = PurePath(recorded)
    return saved.is_absolute() and len(saved.parts) > 2 and saved.parts[-2:] == path.parts[-2:]


def moved_into(recorded: object, folder: Path) -> Path:
    """Where a file a receipt recorded in `folder` is now, after the project moved.

    A recorded path that names a file of `folder`, here or in a moved copy, maps to that
    file in `folder`; anything else comes back unchanged, for the caller's own checks to
    refuse.
    """
    if isinstance(recorded, str) and recorded:
        here = folder / PurePath(recorded).name
        if receipt_names(recorded, here):
            return here
    return Path(str(recorded))
