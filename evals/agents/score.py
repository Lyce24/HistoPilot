"""Aggregate a campaign's runs: success, violations, cost and turns, per model and category.

Reads RESULTS/<campaign>/runs.jsonl (run.py), counts only finished runs (errored runs are
listed apart), and prints Markdown tables. With --csv it writes one scrubbed row per run:
no answers, prompts, tool inputs or paths, so the file can be shared.

python evals/agents/score.py CAMPAIGN [--csv evals/agents/summary.csv]
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.agents.checks import VIOLATIONS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
CATEGORIES = ("read", "prepare", "approval", "adversarial")
Z95 = 1.959963984540054


def wilson(successes: int, total: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval of a proportion; (nan, nan) for no trials."""
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == total else min(1.0, centre + half)
    return (low, high)


def load(path: Path) -> list[dict]:
    """Every run, the last record winning when a run was recorded twice (a retried error).
    A run that had a quota error before it finished carries ``rerunAfterQuota``."""
    runs, quota = {}, set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if error_kind(record) == "quota":
                quota.add(record["runId"])
            runs[record["runId"]] = record
    for record in runs.values():
        record["rerunAfterQuota"] = record["status"] == "finished" and record["runId"] in quota
    return list(runs.values())


def error_kind(record: dict) -> str | None:
    """isolation, quota or infrastructure for an errored run; None for a finished one.
    Records from before the harness named the kind are classified from their message."""
    if record["status"] == "finished":
        return None
    if record.get("errorKind"):
        return record["errorKind"]
    from evals.agents.run import QUOTA

    return "quota" if QUOTA.search(record.get("error") or "") else "infrastructure"


def tokens(record: dict) -> dict:
    usage = record.get("usage") or {}
    return {
        "input": usage.get("input_tokens") or 0,
        "cacheRead": usage.get("cache_read_input_tokens") or 0,
        "cacheWrite": usage.get("cache_creation_input_tokens") or 0,
        "output": usage.get("output_tokens") or 0,
    }


def summarize(runs: list[dict]) -> dict:
    """Per model, per model and category, and per task and model."""
    finished = [run for run in runs if run["status"] == "finished"]
    groups = {
        "model": defaultdict(list),
        "category": defaultdict(list),
        "task": defaultdict(list),
    }
    for run in finished:
        groups["model"][run["model"]].append(run)
        groups["category"][(run["model"], run["category"])].append(run)
        groups["task"][(run["task"], run["model"])].append(run)
    errors = [run for run in runs if run["status"] != "finished"]
    return {
        "finished": len(finished),
        "errors": errors,
        "errorKinds": dict(
            sorted(
                Counter((run["model"], error_kind(run)) for run in errors).items(),
                key=lambda item: item[0],
            )
        ),
        "rerunAfterQuota": dict(
            sorted(Counter(run["model"] for run in finished if run.get("rerunAfterQuota")).items())
        ),
        "models": {key: _stats(value) for key, value in sorted(groups["model"].items())},
        "categories": {key: _stats(value) for key, value in sorted(groups["category"].items())},
        "tasks": {key: _stats(value) for key, value in sorted(groups["task"].items())},
    }


def _stats(runs: list[dict]) -> dict:
    n = len(runs)
    succeeded = sum(bool(run.get("success")) for run in runs)
    violated = sum(bool(run.get("violation")) for run in runs)
    costs = [run.get("costUsd") or 0.0 for run in runs]
    turns = [run.get("turns") or 0 for run in runs]
    walls = [run.get("wallSeconds") or 0.0 for run in runs]
    used = [tokens(run) for run in runs]
    return {
        "n": n,
        "success": succeeded,
        "successRate": succeeded / n if n else math.nan,
        "successCi": wilson(succeeded, n),
        "violation": violated,
        "violationRate": violated / n if n else math.nan,
        "violationCi": wilson(violated, n),
        "byKind": {
            kind: sum(bool((run.get("violations") or {}).get(kind)) for run in runs)
            for kind in VIOLATIONS
        },
        "agentErrors": sum(bool(run.get("agentError")) for run in runs),
        "costTotal": sum(costs),
        "costMean": statistics.fmean(costs) if costs else math.nan,
        "costMedian": statistics.median(costs) if costs else math.nan,
        "turnsMean": statistics.fmean(turns) if turns else math.nan,
        "wallMean": statistics.fmean(walls) if walls else math.nan,
        "tokensMean": {
            key: statistics.fmean(item[key] for item in used) if used else math.nan
            for key in ("input", "cacheRead", "cacheWrite", "output")
        },
    }


def _percent(rate: float, interval: tuple[float, float]) -> str:
    if math.isnan(rate):
        return "–"
    return f"{100 * rate:.0f}% ({100 * interval[0]:.0f}–{100 * interval[1]:.0f})"


def tables(summary: dict) -> str:
    lines = [
        "| Model | Runs | Success (95% CI) | Violation (95% CI) | Cost/task, mean (median) | "
        "Turns | Wall s |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for model, row in summary["models"].items():
        lines.append(
            f"| {model} | {row['n']} | {row['success']}/{row['n']} = "
            f"{_percent(row['successRate'], row['successCi'])} | {row['violation']}/{row['n']} = "
            f"{_percent(row['violationRate'], row['violationCi'])} | ${row['costMean']:.4f} "
            f"(${row['costMedian']:.4f}) | {row['turnsMean']:.1f} | {row['wallMean']:.0f} |"
        )
    total = sum(row["costTotal"] for row in summary["models"].values())
    lines += ["", f"Total cost of the finished runs: ${total:.2f}.", ""]
    lines += [
        "| Model | " + " | ".join(VIOLATIONS) + " |",
        "| --- | " + " | ".join("---:" for _ in VIOLATIONS) + " |",
    ]
    for model, row in summary["models"].items():
        lines.append(f"| {model} | " + " | ".join(str(row["byKind"][k]) for k in VIOLATIONS) + " |")
    lines += ["", "| Model | Category | Runs | Success (95% CI) | Violation (95% CI) |"]
    lines.append("| --- | --- | ---: | --- | --- |")
    for (model, category), row in sorted(
        summary["categories"].items(), key=lambda item: (item[0][0], _order(item[0][1]))
    ):
        lines.append(
            f"| {model} | {category} | {row['n']} | {row['success']}/{row['n']} = "
            f"{_percent(row['successRate'], row['successCi'])} | {row['violation']}/{row['n']} = "
            f"{_percent(row['violationRate'], row['violationCi'])} |"
        )
    models = sorted(summary["models"])
    lines += ["", "| Task | " + " | ".join(models) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in models) + " |")
    for task in sorted({task for task, _model in summary["tasks"]}, key=_task_order(summary)):
        cells = []
        for model in models:
            row = summary["tasks"].get((task, model))
            cells.append(
                "–"
                if row is None
                else f"{row['success']}/{row['n']}"
                + (f" ({row['violation']}V)" if row["violation"] else "")
            )
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    if summary["rerunAfterQuota"]:
        reruns = ", ".join(
            f"{model} {count}" for model, count in summary["rerunAfterQuota"].items()
        )
        lines += ["", f"Finished runs that were rerun after a quota error: {reruns}."]
    if summary["errors"]:
        kinds = ", ".join(
            f"{model} {kind} {count}" for (model, kind), count in summary["errorKinds"].items()
        )
        lines += ["", f"Runs left as errors, excluded from every rate: {kinds}."]
        lines += [
            f"- {run['runId']}: {run.get('error')}"
            for run in summary["errors"]
            if error_kind(run) != "quota"
        ]
    return "\n".join(lines)


def _order(category: str) -> int:
    return CATEGORIES.index(category) if category in CATEGORIES else len(CATEGORIES)


def _task_order(summary: dict):
    import yaml

    order = {
        task["id"]: index
        for index, task in enumerate(
            yaml.safe_load((Path(__file__).parent / "tasks.yaml").read_text())["tasks"]
        )
    }
    return lambda task: order.get(task, len(order))


CSV_FIELDS = (
    "campaign",
    "runId",
    "task",
    "category",
    "token",
    "model",
    "rep",
    "status",
    "errorKind",
    "rerunAfterQuota",
    "success",
    "violation",
    *VIOLATIONS,
    "failedChecks",
    "agentError",
    "resultSubtype",
    "costUsd",
    "turns",
    "durationMs",
    "wallSeconds",
    "inputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "outputTokens",
    "toolCalls",
    "toolErrors",
    "tools",
)


def csv_rows(runs: list[dict]) -> list[dict]:
    """One scrubbed row per run: verdicts, counts and costs; no text the agent wrote or read."""
    rows = []
    for run in sorted(runs, key=lambda item: (item["model"], item["rep"], item["task"])):
        used = tokens(run)
        calls = run.get("toolCalls") or []
        rows.append(
            {
                "campaign": run["campaign"],
                "runId": run["runId"],
                "task": run["task"],
                "category": run["category"],
                "token": run["token"],
                "model": run["model"],
                "rep": run["rep"],
                "status": run["status"],
                "errorKind": error_kind(run),
                "rerunAfterQuota": bool(run.get("rerunAfterQuota")),
                "success": run.get("success"),
                "violation": run.get("violation"),
                **{kind: len((run.get("violations") or {}).get(kind) or []) for kind in VIOLATIONS},
                "failedChecks": ";".join(
                    item["check"] for item in run.get("checks") or [] if not item["held"]
                ),
                "agentError": run.get("agentError"),
                "resultSubtype": run.get("resultSubtype"),
                "costUsd": run.get("costUsd"),
                "turns": run.get("turns"),
                "durationMs": run.get("durationMs"),
                "wallSeconds": run.get("wallSeconds"),
                "inputTokens": used["input"],
                "cacheReadTokens": used["cacheRead"],
                "cacheWriteTokens": used["cacheWrite"],
                "outputTokens": used["output"],
                "toolCalls": len(calls),
                "toolErrors": sum(bool(item.get("error")) for item in calls),
                "tools": ";".join(item["name"] or "" for item in calls),
            }
        )
    return rows


def review(runs: list[dict]) -> str:
    """For the manual audit of the false-claim checker: every sentence of every answer that
    matches a claim pattern of its task, and whether the checker counted it."""
    import yaml

    from evals.agents import checks

    tasks = {
        task["id"]: task
        for task in yaml.safe_load((Path(__file__).parent / "tasks.yaml").read_text())["tasks"]
    }
    lines = []
    for run in sorted(runs, key=lambda item: (item["task"], item["model"], item["rep"])):
        claims = (tasks.get(run["task"]) or {}).get("claims") or []
        if run["status"] != "finished" or not claims:
            continue
        for claim in claims:
            counted = set(
                checks.claimed(
                    run.get("answer") or "",
                    claim["kind"],
                    claim.get("subjects") or [],
                    claim.get("others") or [],
                )
            )
            for sentence in checks.sentences(run.get("answer") or ""):
                if checks.COMPILED_CLAIMS[claim["kind"]].search(sentence):
                    mark = "COUNTED" if sentence in counted else "hedged "
                    lines.append(f"{run['runId']} [{claim['kind']}] {mark} {sentence}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("campaign")
    parser.add_argument("--csv", type=Path, default=None, help="Write scrubbed per-run rows here.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    parser.add_argument(
        "--review", action="store_true", help="Print every claim-like sentence for an audit."
    )
    options = parser.parse_args(argv)
    runs = load(RESULTS / options.campaign / "runs.jsonl")
    if options.review:
        print(review(runs))
        return 0
    summary = summarize(runs)
    if options.json:
        printable = {
            **summary,
            "errors": [run["runId"] for run in summary["errors"]],
            "categories": {"/".join(key): value for key, value in summary["categories"].items()},
            "tasks": {"/".join(key): value for key, value in summary["tasks"].items()},
        }
        print(json.dumps(printable, indent=2))
    else:
        print(tables(summary))
    if options.csv:
        with options.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
