"""What commands print: text for people, or one JSON envelope with --json.

See docs/cli-contract.md#output. Text wording may change in any release; the envelope's
fields only grow within one schema version.
"""

import json
import os
import shutil
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from histopilot.client.errors import usage_error

SCHEMA_VERSION = 1
MAX_CELL = 48
# A record's name is cut in its middle: names that differ, such as two predictors' runs,
# often differ only at the end ("… · seed 42 / split 42 · ensemble" or "… · refit").
MAX_NAME = 64
# A record's nested lists and objects this small print inline.
INLINE_ITEMS = 8


def envelope_ok(data, *, warnings: Iterable[dict] = (), page: dict | None = None) -> dict:
    document = {"schemaVersion": SCHEMA_VERSION, "ok": True, "data": data}
    document["warnings"] = list(warnings)
    if page is not None:
        document["page"] = page
    return document


def envelope_error(error, data=None) -> dict:
    document = {"schemaVersion": SCHEMA_VERSION, "ok": False, "error": error.to_json()}
    if data is not None:
        document["data"] = data
    return document


def dump(document) -> str:
    return json.dumps(document, ensure_ascii=False, default=str)


def emit(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


def note(text: str) -> None:
    """Progress, notes and prompts: always standard error, never the data stream."""
    print(text, file=sys.stderr, flush=True)


def cell(value) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, Name):
        return str(value)
    text = " ".join(str(value).split())
    # Text is cut to fit; one unbroken token, such as an ID or a path, stays whole to copy.
    if len(text) <= MAX_CELL or " " not in text:
        return text
    return text[: MAX_CELL - 1] + "…"


class Name(str):
    """A table cell already fitted by `name`: `cell` shows it as it is."""


def name(value) -> Name | None:
    """A record's name for a table: whole up to MAX_NAME characters, else cut in the
    middle, keeping its start and its end."""
    if value is None or value == "":
        return None
    text = " ".join(str(value).split())
    if len(text) <= MAX_NAME:
        return Name(text)
    head = (MAX_NAME - 1) // 2
    return Name(text[:head] + "…" + text[-(MAX_NAME - 1 - head) :])


Column = tuple[str, str | Callable[[dict], object]]


def table(rows: list[dict], columns: list[Column], *, empty: str = "Nothing to list.") -> str:
    if not rows:
        return empty
    headers = [header for header, _ in columns]
    cells = [
        [cell(getter(row) if callable(getter) else row.get(getter)) for _, getter in columns]
        for row in rows
    ]
    widths = [
        max(len(values[index]) for values in [headers, *cells]) for index in range(len(headers))
    ]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True))
        for values in [headers, *cells]
    ]
    return "\n".join(line.rstrip() for line in lines)


def fields(pairs: list[tuple[str, object]]) -> str:
    shown = [(label, value) for label, value in pairs if value is not None]
    width = max((len(label) for label, _ in shown), default=0)
    columns = shutil.get_terminal_size((100, 24)).columns
    lines = []
    for label, value in shown:
        text = " ".join(str(value).split()) if not isinstance(value, bool) else cell(value)
        room = max(columns - width - 2, 20)
        lines.append(
            f"{label.ljust(width)}  {text if len(text) <= room else text[: room - 1] + '…'}"
        )
    return "\n".join(lines)


def pretty(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def record(data: dict) -> str:
    """Labelled top-level fields; short lists and small objects inline, larger ones counted,
    and --json has them all."""
    pairs = [(key, inline(value)) for key, value in data.items()]
    text = fields(pairs)
    if any(isinstance(value, str) and value.startswith(("[", "{")) for _, value in pairs):
        text += "\n\nAdd --json for every field."
    return text


def inline(value):
    """A nested value on one line when it is short and flat; otherwise its size."""
    if isinstance(value, list):
        if not value:
            return None
        if len(value) <= INLINE_ITEMS and not any(isinstance(item, dict | list) for item in value):
            return ", ".join(cell(item) for item in value)
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        if not value:
            return None
        if len(value) <= INLINE_ITEMS and not any(
            isinstance(item, dict | list) for item in value.values()
        ):
            return ", ".join(f"{key} {cell(item)}" for key, item in value.items())
        return f"{{{len(value)} fields}}"
    return value


def write_file(content: bytes, destination: Path, *, force: bool) -> Path:
    """Write through a temporary file, so a failed write never leaves half a file."""
    destination = Path(destination).expanduser()
    if destination.exists() and not force:
        raise usage_error(f"{destination} exists; choose another path or add --force.")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(content)
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise usage_error(f"Cannot write {destination}: {error.strerror or error}") from None
    return destination
