"""The last step of every file generator here: write the file, or with --check exit 1 when
the file on disk differs from what the code generates."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(description: str, target: Path, render) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is out of date")
    check = parser.parse_args().check
    content, shown = render(), target.relative_to(ROOT)
    if check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != content:
            print(f"{shown} is out of date; run this script without --check.")
            return 1
        return 0
    target.write_text(content, encoding="utf-8")
    print(f"Wrote {shown}.")
    return 0
