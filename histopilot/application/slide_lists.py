"""One way to choose slides: a source, then an optional cohort filter.

The source is an explicit ``wsi[,mpp]`` list when one is supplied and the slide folder's own
contents otherwise. A frozen dataset, when given, narrows that selection to the slides the
cohort claims; without one the initial selection stands.

Two properties of TRIDENT's contract decide the rest. It names every output after the slide's
basename, so two selected slides may never share a stem however deep their folders differ. And
the pinned release drops empty ``mpp`` cells before pairing the remainder with slides
positionally, so a partially filled column silently gives one slide another slide's pixel
size; a partial column is refused outright.
"""

import base64
import binascii
import csv
import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from histopilot.schemas.slide_lists import (
    MAX_SLIDE_LIST_BYTES,
    MAX_SLIDE_UPLOAD_BYTES,
    SlideListSource,
)
from histopilot.storage.scientific import ScientificStore

MAX_ROWS = 100_000
MAX_FOLDER_FILES = 100_000
SLIDE_EXTENSIONS = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
    ".sdpc",
)


class SlideListError(ValueError):
    """Raised when a slide list cannot be applied to a frozen dataset version."""


def read_slide_list_source(source: SlideListSource, resolve_path) -> tuple[bytes, str]:
    """Read either input through the same CSV selection and size limits."""
    if source.path is not None:
        path = resolve_path(source.path)
        return ScientificStore._read_file(path, MAX_SLIDE_LIST_BYTES), str(path)
    try:
        content = base64.b64decode(source.contentBase64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SlideListError("The uploaded slide list is not valid base64 content.") from error
    if not content or len(content) > MAX_SLIDE_UPLOAD_BYTES:
        raise SlideListError(
            "Uploaded slide lists must contain between 1 byte and 1 MB of CSV text."
        )
    return content, source.filename or "Uploaded slide list"


def _parse(content: bytes, *, context: str) -> tuple[list[tuple[int, str, float | None]], bool]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeError as error:
        raise SlideListError(f"{context} must be UTF-8 text.") from error
    reader = csv.DictReader(io.StringIO(text))
    fields = [(name or "").strip() for name in reader.fieldnames or []]
    if not fields or any(not name for name in fields):
        raise SlideListError(f"{context} needs a header row without blank column names.")
    if len(set(fields)) != len(fields):
        raise SlideListError(f"{context} repeats a column name.")
    if "wsi" not in fields:
        raise SlideListError(
            f"{context} needs a wsi column, and may add an mpp column. Other columns are ignored."
        )
    reader.fieldnames = fields
    has_mpp = "mpp" in fields
    rows: list[tuple[int, str, float | None]] = []
    for number, row in enumerate(reader, start=2):
        if None in row:
            raise SlideListError(f"{context} row {number} has more values than its header.")
        value = (row.get("wsi") or "").strip()
        if not value:
            raise SlideListError(f"{context} row {number} has no wsi value.")
        mpp = None
        if has_mpp:
            mpp = _mpp((row.get("mpp") or "").strip(), number, context=context)
        rows.append((number, value, mpp))
        if len(rows) > MAX_ROWS:
            raise SlideListError(f"{context} exceeds {MAX_ROWS:,} rows.")
    if not rows:
        raise SlideListError(f"{context} contains no rows.")
    return rows, has_mpp


def _mpp(raw: str, number: int, *, context: str) -> float:
    # TRIDENT compacts an mpp column before pairing it with slides, so a half-filled column
    # assigns one slide's microns per pixel to another. Refuse it instead of guessing.
    if not raw:
        raise SlideListError(
            f"{context} row {number} has no mpp value. Give every row an mpp or remove the "
            "column; a partly filled column silently mis-assigns pixel sizes."
        )
    try:
        value = float(raw)
    except ValueError as error:
        raise SlideListError(f"{context} row {number} has a non-numeric mpp {raw!r}.") from error
    if not math.isfinite(value) or value <= 0:
        raise SlideListError(f"{context} row {number} needs a positive finite mpp, got {raw!r}.")
    return value


def _relative(value: str, number: int, *, context: str) -> str:
    while value.startswith("./"):
        value = value[2:]
    relative = PurePosixPath(value)
    if not value or "\x00" in value or ".." in relative.parts:
        raise SlideListError(
            f"{context} row {number}: {value!r} must be a slide path inside the slide folder."
        )
    return relative.as_posix()


def _extensions(wsi_ext: list[str] | None) -> tuple[str, ...]:
    """TRIDENT's own extension list, when a run narrows it, decides what counts as a slide."""
    if not wsi_ext:
        return SLIDE_EXTENSIONS
    values = tuple(str(value).strip().lower() for value in wsi_ext)
    if any(not value.startswith(".") or "/" in value for value in values):
        raise SlideListError("Slide extensions must begin with a dot, such as .svs.")
    return values


@dataclass(frozen=True, slots=True)
class SlideEntry:
    """One slide in a selection: where it is, what TRIDENT will name it, and its source MPP."""

    wsi: str
    path: Path
    slideId: str
    mpp: float | None


def scan_slide_folder(
    root: Path, *, extensions=SLIDE_EXTENSIONS, recursive=True, validate_names=True
) -> list[SlideEntry]:
    """The default selection: every readable slide under a folder, in a stable order."""
    root = Path(root)
    if not root.is_dir():
        raise SlideListError(f"{root} is not a slide folder.")
    entries, pending, seen = [], [root], 0
    allowed = tuple(value.lower() for value in extensions)
    while pending:
        try:
            with os.scandir(pending.pop()) as listing:
                for item in listing:
                    seen += 1
                    if seen > MAX_FOLDER_FILES:
                        raise SlideListError(
                            f"The slide folder exceeds {MAX_FOLDER_FILES:,} entries; "
                            "select a narrower folder or supply a slide list."
                        )
                    path = Path(item.path)
                    if item.is_dir(follow_symlinks=False):
                        if recursive:
                            pending.append(path)
                    elif item.is_file() and path.name.lower().endswith(allowed):
                        entries.append(
                            SlideEntry(
                                wsi=path.relative_to(root).as_posix(),
                                path=path,
                                slideId=path.stem,
                                mpp=None,
                            )
                        )
        except OSError as error:
            raise SlideListError(f"A slide folder cannot be read: {error}") from error
    if not entries:
        raise SlideListError(f"No slide files were found under {root}.")
    return _ordered(entries) if validate_names else entries


def read_slide_list(
    content: bytes,
    root: Path,
    *,
    extensions=SLIDE_EXTENSIONS,
    context: str = "The slide list",
    validate_names: bool = True,
) -> list[SlideEntry]:
    """An explicit selection: every listed slide must exist under the folder."""
    rows, _ = _parse(content, context=context)
    root = Path(root)
    entries = []
    allowed = tuple(value.lower() for value in extensions)
    for number, value, mpp in rows:
        normalized = value.replace("\\", "/")
        if PurePosixPath(normalized).is_absolute() or PureWindowsPath(value).is_absolute():
            raise SlideListError(
                f"{context} row {number}: write {value!r} relative to the slide folder."
            )
        relative = _relative(normalized, number, context=context)
        if not relative.lower().endswith(allowed):
            raise SlideListError(
                f"{context} row {number}: {relative!r} is not a supported whole-slide image."
            )
        path = root / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise SlideListError(f"{context} row {number}: {relative!r} is not under {root}.")
        entries.append(
            SlideEntry(wsi=relative, path=path, slideId=PurePosixPath(relative).stem, mpp=mpp)
        )
    return _ordered(entries) if validate_names else entries


def list_slide_ids(
    content: bytes, *, context: str = "The slide list"
) -> tuple[set[str], bool, str]:
    """Slide identities named by a list, without touching slide storage.

    Registering features needs only the identities: the features are the artifact, and the
    slides they came from may live on another machine or be offline entirely.
    """
    rows, has_mpp = _parse(content, context=context)
    identities: dict[str, int] = {}
    for number, value, _mpp in rows:
        relative = _relative(value.replace("\\", "/"), number, context=context)
        identity = PurePosixPath(relative).stem
        if not identity:
            raise SlideListError(f"{context} row {number}: {relative!r} has no slide name.")
        if identity in identities:
            raise SlideListError(
                f"{context} rows {identities[identity]} and {number} both name slide {identity!r}."
            )
        identities[identity] = number
    return set(identities), has_mpp, hashlib.sha256(content).hexdigest()


def _ordered(entries: list[SlideEntry]) -> list[SlideEntry]:
    """Reject the two collisions that silently corrupt a run, then fix a stable order."""
    ordered = sorted(entries, key=lambda entry: (entry.wsi.casefold(), entry.wsi))
    paths: dict[str, str] = {}
    names: dict[str, str] = {}
    duplicates, collisions = [], []
    for entry in ordered:
        if entry.wsi.casefold() in paths:
            duplicates.append(entry.wsi)
        paths[entry.wsi.casefold()] = entry.wsi
        # TRIDENT names every output from the file stem, so two folders cannot share one.
        first = names.get(entry.slideId.casefold())
        if first is not None:
            collisions.append(f"{first} / {entry.wsi}")
        names.setdefault(entry.slideId.casefold(), entry.wsi)
    if duplicates:
        raise SlideListError(
            f"{len(duplicates)} slides are selected twice, including {duplicates[0]!r}."
        )
    if collisions:
        # Report the whole set: a tree that shadows originals produces hundreds of these,
        # and fixing them one error at a time is not a workflow.
        raise SlideListError(
            f"{len(collisions)} pairs of selected slides share a file name and would "
            f"overwrite each other's output: {'; '.join(collisions[:3])}"
            + (f"; and {len(collisions) - 3} more" if len(collisions) > 3 else "")
            + ". Narrow the folder, or supply a slide list that names one of each pair."
        )
    return ordered


def resolve_slide_selection(
    root: Path,
    *,
    list_content: bytes | None = None,
    list_path: str | None = None,
    records: list[dict] | None = None,
    wsi_ext: list[str] | None = None,
    recursive: bool = True,
    context: str = "The slide list",
) -> dict:
    """Build a selection the way a user describes one: a source, then an optional cohort filter.

    The initial list is the slide list when one is supplied and the folder's own contents
    otherwise. A dataset then narrows it to the slides that cohort claims; without one the
    initial list stands.
    """
    extensions = _extensions(wsi_ext)
    entries = (
        read_slide_list(
            list_content, root, extensions=extensions, context=context, validate_names=False
        )
        if list_content is not None
        else scan_slide_folder(
            root, extensions=extensions, recursive=recursive, validate_names=False
        )
    )
    entries.sort(key=lambda entry: (entry.wsi.casefold(), entry.wsi))
    selection = {
        "source": "list" if list_content is not None else "folder",
        "listPath": list_path,
        "sha256": hashlib.sha256(list_content).hexdigest() if list_content is not None else None,
        "root": str(root),
        "initialCount": len(entries),
        "declaresMpp": bool(entries) and all(entry.mpp is not None for entry in entries),
        "datasetFiltered": records is not None,
        "outside": [],
        "unlisted": [],
    }
    if records is not None:
        claimed = {row["slideId"] for row in records}
        inside = [entry for entry in entries if entry.slideId in claimed]
        selection["outside"] = [entry.wsi for entry in entries if entry.slideId not in claimed]
        selection["unlisted"] = sorted(claimed - {entry.slideId for entry in entries})
        entries = inside
        if not entries:
            raise SlideListError(
                "No selected slide belongs to this dataset version; "
                f"{len(selection['outside'])} selected slides belong to other cohorts."
            )
    # Only selected slides can collide in TRIDENT outputs. Duplicates in unrelated cohorts
    # must not prevent a dataset restriction from selecting a valid set.
    selection["slides"] = _ordered(entries)
    selection["selectedCount"] = len(entries)
    return selection
