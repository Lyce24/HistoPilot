"""Print one version's section of CHANGELOG.md: the notes of its GitHub release.

python scripts/release_notes.py           the package's own version
python scripts/release_notes.py v0.1.0    another version; a leading "v" is ignored
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Reads only the standard library and the version; run it without an installed package too.
sys.path.insert(0, str(ROOT))

from histopilot import __version__  # noqa: E402

HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")
LINK = re.compile(r"^\[[^\]]+\]: \S")


def section(text: str, version: str) -> str:
    """The body of ``## [version]``, up to the next version or the list of links."""
    body, inside = [], False
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            if inside:
                break
            inside = heading["version"] == version
        elif inside:
            if LINK.match(line):
                break
            body.append(line)
    notes = "\n".join(body).strip()
    if not notes:
        raise ValueError(f"CHANGELOG.md has no notes for version {version}.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", nargs="?", default=__version__, help="default: the package's")
    version = parser.parse_args().version.removeprefix("v")
    try:
        print(section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version))
    except ValueError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
