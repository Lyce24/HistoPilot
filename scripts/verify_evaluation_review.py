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

    def toggle_experiment(identifier):
        evaluate(
            "(() => { const el = document.querySelector('input[aria-label$=\"(' + "
            + json.dumps(identifier)
            + " + ')\"]'); if (!el || el.disabled) throw Error('Experiment unavailable'); el.click(); return true; })()"
        )
        browser("snapshot", "-i")

    def method(value):
        evaluate(
            "document.querySelector('input[name=evaluation-method][value=' + "
            + json.dumps(value)
            + " + ']').click(); true"
        )
        browser("snapshot", "-i")

    preview = "window.__evaluationReview.traffic.filter(item => item.path.endsWith('/preview')).at(-1)?.body"
    try:
        browser("open", (output / "index.html").as_uri())
        browser("set", "viewport", "1440", "1100")
        check(
            "Linked experiment starts selected with 90 ready predictors",
            "document.body.innerText.includes('90 predictors to evaluate from 1 selected experiment')",
        )
        check(
            "Planning and Skip sources stay visible with truthful readiness",
            "document.querySelector('.evaluation-experiment-list').innerText.includes('Running · No ready predictors') && document.querySelector('.evaluation-experiment-list').innerText.includes('Skip predictor study')",
        )
        check(
            "Desktop has no horizontal page overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        browser("screenshot", str(output / "evaluation-experiment-predictors.png"))
        click("Choose experiments")
        check(
            "An unscoped evaluation selects no experiments and cannot run",
            "document.body.innerText.includes('0 predictors to evaluate from 0 selected experiments') && [...document.querySelectorAll('button')].find(el => el.textContent === 'Review experiment evaluation').disabled",
        )
        toggle_experiment("study")
        toggle_experiment("other")
        check(
            "Multiple experiment selection includes exactly their 92 ready predictors",
            "document.body.innerText.includes('92 predictors to evaluate from 2 selected experiments')",
        )
        method("refit")
        check(
            "Refit choice uses the 46 ready refits from selected experiments",
            "document.body.innerText.includes('46 predictors to evaluate from 2 selected experiments')",
        )
        click("Advanced: single predictor plan")
        click("Evaluate experiments")
        check(
            "Switching setup views preserves experiment and method selection",
            "document.body.innerText.includes('46 predictors to evaluate from 2 selected experiments') && document.querySelector('input[name=evaluation-method][value=refit]').checked",
        )
        select("Test cohort for selected experiments", "cohort")
        click("Review experiment evaluation")
        check(
            "Review pins exact predictor IDs from all selected experiments",
            f"{preview}?.scope === 'selected' && {preview}.predictorIds.length === 46 && {preview}.predictorIds.every(id => id.endsWith('-refit')) && {preview}.predictorIds.some(id => id.startsWith('study-')) && {preview}.predictorIds.some(id => id.startsWith('other-'))",
        )
        check(
            "Review locks source, methods and setup switching",
            "[...document.querySelectorAll('.evaluation-experiment-list input')].every(el => el.disabled) && document.querySelector('input[name=evaluation-method]').matches(':disabled') && [...document.querySelectorAll('button')].find(el => el.textContent === 'Advanced: single predictor plan').disabled",
        )
        evaluate("window.__evaluationReview.addPredictor()")
        check(
            "Predictors arriving after review stay excluded",
            "document.body.innerText.includes('46 predictors fixed for review') && document.body.innerText.includes('1 additional ready predictor is excluded from this review')",
        )
        evaluate("document.querySelector('.run-bulk-review').scrollIntoView({block:'start'}); true")
        browser("screenshot", str(output / "evaluation-reviewed-selection.png"))
        evaluate(
            "window.__evaluationReview.loseNextAck(); [...document.querySelectorAll('input[type=checkbox]')].find(el => el.parentElement.textContent.includes('I reviewed')).click(); true"
        )
        click("Run reviewed predictors")
        check(
            "Lost acknowledgement retains the original review and retry controls",
            "document.body.innerText.includes('Offline fixture lost acknowledgement') && [...document.querySelectorAll('button')].some(el => el.textContent === 'Retry unfinished submissions') && window.__evaluationReview.records.length === 1",
        )
        click("Retry unfinished submissions")
        check(
            "Retry uses one operation and the same 46 IDs, excluding the late arrival",
            "(() => { const posts = window.__evaluationReview.traffic.filter(item => item.method === 'POST' && item.path.endsWith('/bulk')); return posts.length === 2 && posts[0].body.operationId === posts[1].body.operationId && JSON.stringify(posts[0].body.predictorIds) === JSON.stringify(posts[1].body.predictorIds) && posts[1].body.predictorIds.length === 46 && !posts[1].body.predictorIds.includes('study-16-11-refit') && window.__evaluationReview.records.length === 1; })()",
        )
        check(
            "Paired means exclude unmatched high-scoring sources",
            "(() => { const group = document.querySelector('.evaluation-comparison-context[data-cohort=cohort][data-unit=patient]'); return group.innerText.includes('2 matched source pairs') && group.innerText.includes('1 ensemble-only') && group.innerText.includes('0.650') && group.innerText.includes('0.725') && !group.querySelector('.evaluation-comparison-means').innerText.includes('0.990'); })()",
        )
        check(
            "Results separate test cohorts and patient versus slide scoring",
            "document.querySelectorAll('.evaluation-comparison-context').length === 3 && !!document.querySelector('.evaluation-comparison-context[data-cohort=second-cohort][data-unit=patient]') && !!document.querySelector('.evaluation-comparison-context[data-cohort=cohort][data-unit=slide]')",
        )
        select("Evaluation method", "ensemble")
        check(
            "Filtering detail rows does not alter paired method means",
            "document.querySelector('.evaluation-comparison-context[data-cohort=cohort][data-unit=patient]').innerText.includes('2 matched source pairs')",
        )
        evaluate(
            "document.querySelector('.evaluation-comparison').scrollIntoView({block:'start'}); true"
        )
        browser("screenshot", str(output / "evaluation-method-comparison.png"))
        click("Skipped experiment")
        check(
            "Skip selection never falls back to other ready predictors",
            "document.body.innerText.includes('0 predictors to evaluate from 1 selected experiment') && document.body.innerText.includes('Batches using Skip produce no predictors') && [...document.querySelectorAll('button')].find(el => el.textContent === 'Review experiment evaluation').disabled",
        )
        click("Choose experiments")
        toggle_experiment("other")
        evaluate(
            "document.querySelector('.evaluation-advanced').open = true; document.querySelector('.evaluation-advanced input[type=checkbox]').click(); true"
        )
        check(
            "Individual selection starts from only the selected experiment",
            "document.querySelectorAll('.evaluation-predictor-table tbody input[type=checkbox]').length === 2",
        )
        evaluate(
            "document.querySelector('.evaluation-predictor-table tbody input[type=checkbox]').click(); true"
        )
        check(
            "Advanced individual selection narrows the experiment output list",
            "document.body.innerText.includes('1 predictor to evaluate from 1 selected experiment')",
        )
        select("Test cohort for selected experiments", "cohort")
        click("Review experiment evaluation")
        check(
            "Individual review also uses one explicit predictor ID",
            f"{preview}.predictorIds.length === 1 && {preview}.predictorIds[0].startsWith('other-')",
        )
        click("Change selection and review again")
        browser("set", "viewport", "390", "844")
        evaluate("scrollTo(0,0); true")
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
