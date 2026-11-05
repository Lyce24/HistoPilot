"""Summarize a pytest JUnit report for the CI job summary, and hold a test leg to its promise.

    uv run pytest -n auto --junitxml=pytest.xml
    uv run python scripts/ci_test_report.py pytest.xml --leg training --label "with Torch, on CPU"

Prints a Markdown section: tests passed, failed and skipped, why tests were skipped, and how many
reference-parity tests passed (tests/test_mil_parity.py against the authors' implementations,
tests/test_oceanpath_parity.py against OceanPath's fused kernels).

In the ``training`` leg it exits 1 when any test was skipped for want of the training extra
(Torch, Lightning, scikit-learn), so the models, their parity tests and the training workers
cannot quietly drop out of CI again. Tests that need a CUDA GPU are expected to skip on CPU
runners. The ``lean`` leg runs without the extra on purpose, so its skips are only reported.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass

PARITY = {
    "tests.test_mil_parity": "MIL models against the authors' implementations",
    "tests.test_oceanpath_parity": "Mamba-2 MIL against OceanPath's fused kernels",
}
# Why a test skips when the training extra is missing: importorskip of these modules, or the
# Task Center training tests' own marker. A skip for want of CUDA mentions CUDA instead.
WANTS_TRAINING = re.compile(
    r"\b(torch|lightning|sklearn|histopilot\.training)\b|training extra", re.I
)


@dataclass(frozen=True)
class Case:
    module: str
    name: str
    outcome: str  # passed, failed or skipped
    reason: str = ""


def cases(path: str) -> list[Case]:
    found = []
    for case in ET.parse(path).getroot().iter("testcase"):
        module = case.get("classname") or ""
        skipped = case.find("skipped")
        if case.find("failure") is not None or case.find("error") is not None:
            found.append(Case(module, case.get("name") or "", "failed"))
        elif skipped is not None:
            reason = skipped.get("message") or skipped.text or ""
            found.append(Case(module, case.get("name") or "", "skipped", reason.strip()))
        else:
            found.append(Case(module, case.get("name") or "", "passed"))
    return found


def wants_training(case: Case) -> bool:
    return (
        case.outcome == "skipped"
        and bool(WANTS_TRAINING.search(case.reason))
        and "cuda" not in case.reason.lower()
    )


def report(found: list[Case], leg: str, label: str | None = None) -> tuple[str, list[Case]]:
    outcomes = Counter(case.outcome for case in found)
    heading = f"### Tests, {label}" if label else "### Tests"
    lines = [
        heading,
        "",
        f"{outcomes['passed']:,} passed, {outcomes['failed']:,} failed, {outcomes['skipped']:,} skipped.",
    ]
    reasons = Counter(case.reason for case in found if case.outcome == "skipped")
    if reasons:
        lines += ["", "| Skipped because | Tests |", "| --- | ---: |"]
        lines += [
            f"| {reason.replace('|', '/') or '(no reason given)'} | {count} |"
            for reason, count in reasons.most_common()
        ]
    parity = [
        f"- {sum(case.outcome == 'passed' for case in found if case.module == module)} passed: {what}"
        for module, what in PARITY.items()
    ]
    lines += ["", "Reference parity:", *parity]
    missing = [case for case in found if wants_training(case)] if leg == "training" else []
    if missing:
        lines += [
            "",
            f"**{len(missing)} test(s) skipped for want of the training extra, which this leg installs:**",
            *(f"- `{case.module}::{case.name}`: {case.reason}" for case in missing[:20]),
        ]
    return "\n".join(lines) + "\n", missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("junit", help="the report pytest --junitxml wrote")
    parser.add_argument("--leg", choices=("lean", "training"), required=True)
    parser.add_argument("--label", help="how the section heading names this leg")
    args = parser.parse_args(argv)
    text, missing = report(cases(args.junit), args.leg, args.label)
    print(text)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
