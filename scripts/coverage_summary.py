"""Report test line coverage of each histopilot package, as a Markdown table.

    uv run pytest --cov --cov-report=
    uv run python scripts/coverage_summary.py ["with Torch, on CPU"]

Reads the ``.coverage`` data that a ``pytest --cov`` run leaves behind (``[tool.coverage]`` in
pyproject.toml). Modules no test imports count as uncovered. Work done in subprocesses, such
as training workers and the Task Center runner, is not measured, so those packages read lower
than their tests exercise them.

CI appends the table to the job summary, once per test leg, with the leg's label in the heading.
It reports and never fails on the numbers.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import coverage

ROOT = Path(__file__).resolve().parents[1]


def package_totals() -> dict[str, list[int]]:
    cov = coverage.Coverage()
    cov.load()
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "coverage.json"
        cov.json_report(outfile=str(out))
        files = json.loads(out.read_text())["files"]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, entry in files.items():
        parts = Path(name).resolve().relative_to(ROOT).parts
        package = parts[1] if len(parts) > 2 else "(top level)"
        totals[package][0] += entry["summary"]["num_statements"]
        totals[package][1] += entry["summary"]["covered_lines"]
    return dict(sorted(totals.items()))


def report(label: str | None = None) -> str:
    totals = package_totals()
    lines = ["| Package | Statements | Covered |", "| --- | ---: | ---: |"]
    for package, (statements, covered) in totals.items():
        lines.append(f"| `{package}` | {statements:,} | {_share(covered, statements)} |")
    statements = sum(t[0] for t in totals.values())
    covered = sum(t[1] for t in totals.values())
    lines.append(f"| **All** | **{statements:,}** | **{_share(covered, statements)}** |")
    heading = f"### Test coverage, {label}" if label else "### Test coverage"
    return heading + "\n\n" + "\n".join(lines) + "\n"


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "–"


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else None))
