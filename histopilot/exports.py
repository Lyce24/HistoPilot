"""Exports the browser builds itself, ported so the CLI writes the same bytes.

`result_rows` and `csv_text` port `resultRows` and `csv` (web/src/lib/experimentResults.ts):
the "Folds and seeds (CSV)" download of an experiment's results. Values print as
JavaScript's `String()` prints them, so both files compare equal. The shared cases in
`web/src/lib/resolverCases.json` hold both sides to it.
"""

import re
from decimal import Decimal

from histopilot.resolvers import given

RESULT_METRICS = ("auroc", "auprc", "balancedAccuracy", "macroF1", "accuracy", "loss")
_QUOTED = re.compile(r'[",\n]')


def js_number(value: float) -> str:
    """JavaScript's `String(number)`: shortest round-trip digits, exponent past 1e21 or
    below 1e-6, and no trailing `.0`."""
    if value != value:
        return "NaN"
    if value in (float("inf"), float("-inf")):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    parts = Decimal(repr(abs(float(value)))).normalize().as_tuple()
    digits = "".join(str(digit) for digit in parts.digits)
    k, n = len(digits), len(parts.digits) + parts.exponent
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * -n + digits
    exponent = n - 1
    mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{sign}{mantissa}e{'+' if exponent > 0 else '-'}{abs(exponent)}"


def js_string(value) -> str:
    """JavaScript's `String(value)` for the JSON values results hold."""
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return js_number(value)
    return str(value)


def result_rows(results: dict) -> list[list]:
    """Every seed's out-of-fold result and every completed test fold, one row each."""
    header = [
        "batch",
        "configuration",
        "selected",
        "split_seed",
        "training_seed",
        "level",
        "fold",
        "count",
        *RESULT_METRICS,
        "best_epoch",
        "epochs_completed",
    ]
    rows: list[list] = [header]
    for batch in results.get("batches") or []:
        for configuration in batch.get("configurations") or []:
            selected = js_string(configuration.get("selected"))
            for split in configuration.get("splitSeeds") or []:
                for seed in split.get("seeds") or []:
                    lead = [
                        batch.get("name"),
                        configuration.get("number"),
                        selected,
                        split.get("splitSeed"),
                        seed.get("trainingSeed"),
                    ]
                    oof = seed.get("oof")
                    if oof:
                        rows.append(
                            [
                                *lead,
                                "oof",
                                "",
                                given(oof.get("count"), ""),
                                *(given(oof.get(metric), "") for metric in RESULT_METRICS),
                                "",
                                "",
                            ]
                        )
                    for fold in seed.get("folds") or []:
                        metrics = fold.get("metrics")
                        if metrics:
                            rows.append(
                                [
                                    *lead,
                                    "test_fold",
                                    fold["fold"] + 1,
                                    given(metrics.get("count"), ""),
                                    *(given(metrics.get(metric), "") for metric in RESULT_METRICS),
                                    given(fold.get("bestEpoch"), ""),
                                    given(fold.get("epochsCompleted"), ""),
                                ]
                            )
    return rows


def csv_text(rows: list[list]) -> str:
    """`csv`: comma-separated, quoted where a cell holds a quote, comma or newline."""

    def cell(value) -> str:
        text = "" if value is None else js_string(value)
        return '"' + text.replace('"', '""') + '"' if _QUOTED.search(text) else text

    return "\n".join(",".join(cell(value) for value in row) for row in rows) + "\n"
