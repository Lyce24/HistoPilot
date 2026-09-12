"""Verify experiment-owned evaluation choices using a standalone offline browser fixture."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="agent-browser")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/histopilot-evaluation-review/browser")
    )
    args = parser.parse_args()
    output = args.output.resolve()
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "XDG_RUNTIME_DIR": str(runtime)}
    command = [args.browser, "--session", "histopilot-evaluation-review", "--allow-file-access"]
    results = []

    def browser(*parts):
        process = subprocess.run(
            [*command, *parts, "--json"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        value = json.loads(process.stdout)
        if not value["success"]:
            raise AssertionError(value)
        return value["data"]

    def evaluate(code):
        return browser("eval", code)["result"]

    def check(name, expression):
        deadline = time.monotonic() + 8
        while not evaluate(expression):
            if time.monotonic() >= deadline:
                raise AssertionError(f"{name}: {expression}")
            time.sleep(0.15)
        results.append({"name": name, "passed": True})
        print(f"PASS {name}", flush=True)

    def click(name):
        evaluate(
            "(() => { const el = [...document.querySelectorAll('button')].find(el => el.textContent.trim() === "
            + json.dumps(name)
            + "); if (!el || el.disabled) throw Error('Button unavailable'); el.click(); return true; })()"
        )
        browser("snapshot", "-i")

    def select(label, value):
        evaluate(
            "(() => { const el = [...document.querySelectorAll('label')].find(el => el.textContent.trim().startsWith("
            + json.dumps(label)
            + ")).querySelector('select'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, "
            + json.dumps(value)
            + "); el.dispatchEvent(new Event('change', {bubbles:true})); return true; })()"
        )

    preview = "window.__evaluationReview.traffic.filter(item => item.path.endsWith('/preview')).at(-1)?.body"
    try:
        browser("open", (output / "index.html").as_uri())
        browser("set", "viewport", "1440", "1100")
        check(
            "Experiment link shows 90 predictors, not 450 fold models",
            "document.body.innerText.includes('All shown predictors (90)') && document.querySelectorAll('.evaluation-predictor-table tbody tr:not(.run-group)').length === 90",
        )
        check(
            "The linked experiment retains every configuration, seed and method",
            "document.querySelector('.evaluation-predictor-table').innerText.includes('Configuration 15') && document.querySelector('.evaluation-predictor-table').innerText.includes('Fold ensemble') && document.querySelector('.evaluation-predictor-table').innerText.includes('Refit')",
        )
        check(
            "Downstream evidence navigation has no separate predictor step",
            "!document.querySelector('.chain-banner a[href^=\"#post-development\"]') && document.querySelectorAll('.chain-banner .evidence-chain-step').length === 4",
        )
        check(
            "Desktop has no horizontal page overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        browser("screenshot", str(output / "evaluation-experiment-predictors.png"))
        select("Test cohort for all predictors", "cohort")
        select("Predictor method filter", "refit")
        check(
            "Method filtering selects exactly 45 refit predictors",
            "document.body.innerText.includes('All shown predictors (45)') && document.querySelectorAll('.evaluation-predictor-table tbody tr:not(.run-group)').length === 45",
        )
        click("Review all shown predictors")
        check(
            "Filtered review pins only the shown experiment and method IDs",
            f"{preview}?.scope === 'selected' && {preview}.predictorIds.length === 45 && {preview}.predictorIds.every(id => id.startsWith('study-') && id.endsWith('-refit'))",
        )
        check(
            "Review preserves the exact predictor count",
            "document.querySelector('.run-bulk-review').innerText.includes('45 compatible predictors will run')",
        )
        browser("screenshot", str(output / "evaluation-reviewed-selection.png"))
        evaluate(
            "[...document.querySelectorAll('input[type=checkbox]')].find(el => el.parentElement.textContent.includes('I reviewed')).click(); true"
        )
        click("Run selected compatible predictors")
        check(
            "Mocked submission contains exactly the reviewed 45 predictors",
            "window.__evaluationReview.records.length === 1 && window.__evaluationReview.records[0].items.length === 45 && window.__evaluationReview.records[0].items.every(item => item.predictorId.startsWith('study-') && item.method === 'refit')",
        )
        click("Skipped experiment")
        check(
            "A skipped experiment never falls back to other experiments' predictors",
            "document.body.innerText.includes('All shown predictors (0)') && document.body.innerText.includes('Experiments submitted with Skip produce no predictors') && !document.querySelector('.evaluation-predictor-table')",
        )
        check(
            "No predictor selection blocks evaluation review",
            "[...document.querySelectorAll('button')].find(el => el.textContent === 'Review all shown predictors').disabled",
        )
        click("All experiments")
        check(
            "Global selection groups source experiments without merging weights",
            "document.body.innerText.includes('All shown predictors (92)') && document.querySelectorAll('.evaluation-predictor-table .run-group').length === 2",
        )
        select("Source experiment", "other")
        check(
            "Changing the source updates the whole evaluation selection",
            "document.body.innerText.includes('All shown predictors (2)') && document.querySelectorAll('.evaluation-predictor-table tbody tr:not(.run-group)').length === 2",
        )
        click("Single predictor")
        check(
            "Single predictor selection retains the selected source experiment",
            "document.querySelector('select option[value=other]').selected && document.querySelectorAll('optgroup').length === 1 && document.querySelector('optgroup').label === 'Independent study'",
        )
        select("Predictor", "other-1-11-refit")
        check(
            "Single source link opens that predictor within its experiment",
            "!!document.querySelector('a[href=\"#experiments?experiment=other&tab=predictors&predictor=other-1-11-refit\"]')",
        )
        browser("set", "viewport", "390", "844")
        check(
            "Mobile evaluation has no horizontal page overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        browser("screenshot", str(output / "evaluation-mobile.png"))
        check(
            "All requests used the offline fixture",
            "window.__evaluationReview.traffic.every(item => item.matched)",
        )
        errors = browser("errors")
        if errors.get("errors"):
            raise AssertionError(errors)
        results.append({"name": "No browser page errors", "passed": True})
        (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{len(results)} offline evaluation checks passed", flush=True)
    finally:
        browser("close")


if __name__ == "__main__":
    main()
