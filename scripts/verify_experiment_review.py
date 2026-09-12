"""Check the real experiment UI against invented file:// fixtures; never start a server."""

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
        "--output", type=Path, default=Path("/tmp/histopilot-experiment-review/browser")
    )
    args = parser.parse_args()
    output = args.output.resolve()
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "XDG_RUNTIME_DIR": str(runtime)}
    command = [args.browser, "--session", "histopilot-experiment-review", "--allow-file-access"]
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
        result = json.loads(process.stdout)
        if not result["success"]:
            raise AssertionError(result)
        return result["data"]

    def evaluate(code):
        return browser("eval", code)["result"]

    def check(name, expression):
        deadline = time.monotonic() + 8
        while not evaluate(expression):
            if time.monotonic() >= deadline:
                print(
                    evaluate(
                        "JSON.stringify({text:document.body.innerText, traffic:window.__experimentReview.traffic.slice(-8), fields:[...document.querySelectorAll('fieldset')].map(el=>({class:el.className, disabled:el.disabled}))})"
                    ),
                    flush=True,
                )
                raise AssertionError(f"{name}: {expression}")
            time.sleep(0.15)
        results.append({"name": name, "passed": True})
        print(f"PASS {name}", flush=True)

    def button(name):
        return (
            "[...document.querySelectorAll('button')].find(el => el.textContent.trim() === "
            + json.dumps(name)
            + ")"
        )

    def click(name):
        evaluate(
            f"(() => {{ const el = {button(name)}; if (!el || el.disabled) throw Error('Button unavailable: ' + {json.dumps(name)}); el.click(); return true; }})()"
        )
        browser("snapshot", "-i")

    def field(label, kind="input"):
        return (
            "[...document.querySelectorAll('label')].find(el => el.textContent.trim().startsWith("
            + json.dumps(label)
            + ")).querySelector("
            + json.dumps(kind)
            + ")"
        )

    def fill(label, value, kind="input"):
        prototype = "HTMLSelectElement" if kind == "select" else "HTMLInputElement"
        event = "change" if kind == "select" else "input"
        evaluate(
            f"(() => {{ const el = {field(label, kind)}; Object.getOwnPropertyDescriptor({prototype}.prototype, 'value').set.call(el, {json.dumps(value)}); el.dispatchEvent(new Event('{event}', {{ bubbles: true }})); return true; }})()"
        )

    current = "window.__experimentReview.records[new URLSearchParams(location.hash.split('?')[1]).get('experiment')]"
    try:
        browser("open", (output / "index.html").as_uri())
        browser("set", "viewport", "1440", "1100")
        check(
            "Registry renders three stages without a downstream pipeline",
            "document.querySelectorAll('.experiment-record-table tbody tr').length === 3 && !document.querySelector('a[href^=\"#post-development\"]') && !document.querySelector('a[href^=\"#evaluation\"]')",
        )
        check(
            "Registry counts frozen batches once",
            "[...document.querySelectorAll('.experiment-record-table tbody tr')].every(el => el.textContent.includes('2 batches'))",
        )
        check(
            "Desktop layout has no horizontal page overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        browser("screenshot", str(output / "experiments-list.png"))
        click("Create experiment")
        fill("Start from template", "finished", "select")
        check(
            "Choosing an experiment template fills its metadata",
            f"{field('Experiment name')}.value === 'ABMIL finished template copy'",
        )
        fill("Experiment name", "Copied comparison")
        click("Create & open inputs")
        check(
            "Creation copies both recipes and opens editable Inputs",
            f"{current}?.name === 'Copied comparison' && {current}.batchPlans.length === 2 && {current}.stage === 'planning' && document.getElementById('development-tab-setup')?.getAttribute('aria-selected') === 'true'",
        )
        check(
            "Planning locks Runs and Results",
            "document.getElementById('development-tab-runs').disabled && document.getElementById('development-tab-results').disabled",
        )
        check(
            "Details retain management without the removed copy action",
            "document.body.innerText.includes('Manage experiment') && !document.body.innerText.includes('Create from these inputs')",
        )
        click("Batches")
        check(
            "Template copies predictor choices and counts groups across folds",
            f"{current}.predictorPolicy.method === 'both' && {current}.predictorPolicy.refitPercentile === 75 && document.body.innerText.includes('4 predictors planned') && document.body.innerText.includes('2 k-fold groups · 10 fold runs')",
        )
        evaluate(
            'document.querySelector(\'input[name^="predictor-policy-"][value="skip"]\').click(); true'
        )
        check(
            "Skip removes refit controls and prevents unsaved submission",
            f"!document.body.innerText.includes('Refit epoch budget') && document.body.innerText.includes('0 predictors planned') && {button('Review & submit')}.disabled",
        )
        click("Save predictor choices")
        check(
            "Skip is saved explicitly",
            f"{current}.predictorPolicy.method === 'skip' && {current}.predictorPolicy.refitPercentile === null && !{button('Review & submit')}.disabled",
        )
        evaluate(
            'document.querySelector(\'input[name^="predictor-policy-"][value="refit"]\').click(); true'
        )
        check(
            "Refit counts one predictor per complete fold group",
            "document.body.innerText.includes('2 predictors planned') && document.body.innerText.includes('0 ensembles + 2 refits')",
        )
        fill("Refit epoch budget", "custom", "select")
        fill("Custom percentile", "")
        check(
            "Blank percentile blocks saving and submission",
            f"{button('Save predictor choices')}.disabled && {button('Review & submit')}.disabled",
        )
        fill("Custom percentile", "101")
        check(
            "Out-of-range percentile is rejected",
            f"{button('Save predictor choices')}.disabled && document.body.innerText.includes('percentile from 1 to 100')",
        )
        fill("Refit epoch budget", "75", "select")
        evaluate(
            'document.querySelector(\'input[name^="predictor-policy-"][value="both"]\').click(); true'
        )
        click("Save predictor choices")
        check(
            "Both and P75 save before submission",
            f"{current}.predictorPolicy.method === 'both' && {current}.predictorPolicy.refitPercentile === 75 && !{button('Review & submit')}.disabled",
        )
        click("Edit batch")
        evaluate(
            f"(() => {{ const record = {current}; record.batchPlans[0].spec.recipe.learningRate = 0.0006; record.revision += 1; window.__experimentReview.refresh(); return true; }})()"
        )
        check(
            "A remote plan update locks the clean editor and submission",
            f"document.querySelector('.development-editor').disabled && {button('Review & submit')}.disabled && document.body.innerText.includes('The saved experiment changed')",
        )
        click("Reload the saved plan")
        check(
            "Reloading the saved plan restores the new value for editing",
            f"{field('Learning rate')}.value === '0.0006' && !document.querySelector('.development-editor').disabled",
        )
        fill("Learning rate", "0.0007")
        check(
            "Unsaved recipe changes prevent submission",
            f"{button('Review & submit')}.disabled && document.body.innerText.includes('Unsaved batch edits')",
        )
        click("Save batch changes")
        check(
            "Saved recipe changes retain both batches and allow submission",
            f"{current}.batchPlans.length === 2 && {current}.batchPlans[0].spec.recipe.learningRate === 0.0007 && !{button('Review & submit')}.disabled",
        )
        click("New batch")
        fill("Start from a template", "quick", "select")
        check(
            "Quick preset opens a third editable batch",
            f"{field('Batch name')}.value === 'Quick check' && {button('Review & submit')}.disabled",
        )
        click("Add batch to plan")
        check(
            "Multiple batches save into the same experiment",
            f"{current}.batchPlans.length === 3 && !{button('Review & submit')}.disabled",
        )
        browser("screenshot", str(output / "experiments-planning.png"))
        click("Lose next submission response")
        click("Review & submit")
        check(
            "Submission explains the permanent configuration lock",
            "document.body.innerText.includes('Inputs, batch settings and predictor choices become permanently read-only') && document.body.innerText.includes('refit epoch budget P75')",
        )
        evaluate(
            f"(() => {{ const el = {button('Freeze & submit experiment')}; el.click(); el.click(); return true; }})()"
        )
        check(
            "An uncertain response preserves a single submission identity",
            "window.__experimentReview.submissions.length === 1 && document.body.innerText.includes('Retry submission')",
        )
        click("Retry submission")
        check(
            "Retry uses the original operation and locks the submitted plan",
            f"window.__experimentReview.submissions.length === 2 && window.__experimentReview.submissions[0].operationId === window.__experimentReview.submissions[1].operationId && {current}.stage === 'running' && {current}.configurationLocked && document.getElementById('development-tab-runs').getAttribute('aria-selected') === 'true'",
        )
        check(
            "Running keeps results locked",
            "document.getElementById('development-tab-results').disabled",
        )
        fill("Batch", evaluate(f"{current}.batches[0].id"), "select")
        check(
            "Runs expose current training progress and GPU measurements",
            "document.body.innerText.includes('74%') && document.body.innerText.includes('Cancel batch')",
        )
        browser("screenshot", str(output / "experiments-running.png"))
        evaluate(
            "document.querySelector('.development-execution').scrollIntoView({block: 'start'}); true"
        )
        browser("screenshot", str(output / "experiments-tracking.png"))
        check(
            "Selected run displays recorded loss history",
            "document.body.innerText.includes('Loss history') && document.querySelector('.experiment-history-charts svg') !== null",
        )
        evaluate(
            "document.querySelector('.experiment-history-charts').scrollIntoView({block: 'start'}); true"
        )
        browser("screenshot", str(output / "experiments-loss-history.png"))
        evaluate("scrollTo(0, 0); true")
        click("Inputs")
        check(
            "Submitted inputs remain readable but disabled",
            "document.querySelector('.mil-plan-fields').disabled && !document.body.innerText.includes('Check inputs & continue')",
        )
        click("Batches")
        check(
            "Submitted batches have no edit or remove actions",
            "!document.body.innerText.includes('Edit batch') && !document.body.innerText.includes('Add batch to plan') && !document.body.innerText.includes('Clone batch')",
        )
        check(
            "Submitted predictor choices are immutable",
            "document.querySelector('.experiment-predictor-fields').disabled && !document.body.innerText.includes('Save predictor choices')",
        )
        click("Finish folds only")
        click("Runs")
        check(
            "Completed folds keep Results locked until refits finish",
            f"{current}.batches.every(batch => batch.status === 'completed') && {current}.stage === 'running' && document.getElementById('development-tab-results').disabled && document.body.innerText.includes('Epoch 8 / 18') && document.body.innerText.includes('Loss 0.2710')",
        )
        check(
            "Completed ensembles become available for scoped evaluation",
            f"document.body.innerText.includes('3 ready to evaluate') && [...document.querySelectorAll('a')].some(el => el.textContent.includes('Evaluate predictors') && el.hash.includes('experiment=' + {current}.id))",
        )
        browser("screenshot", str(output / "experiments-refit-tracking.png"))
        evaluate(
            f"(() => {{ const record = {current}; record.predictorExecution.status = 'interrupted'; record.predictorExecution.retryable = true; record.predictorExecution.cancellable = false; window.__experimentReview.refresh(); window.__experimentReview.losePredictorAction(); return true; }})()"
        )
        check(
            "Interrupted predictor creation offers recovery",
            f"{button('Resume predictor creation')} !== undefined",
        )
        click("Resume predictor creation")
        check(
            "Lost predictor resume response retains recovery action",
            f"{button('Retry resume request')} !== undefined",
        )
        click("Retry resume request")
        check(
            "Predictor resume retries use the same operation",
            "window.__experimentReview.predictorActions.length === 2 && window.__experimentReview.predictorActions[0].operationId === window.__experimentReview.predictorActions[1].operationId",
        )
        click("Finish selected fixture")
        check(
            "Finishing unlocks results while preserving configuration lock",
            f"{current}.stage === 'finished' && !document.getElementById('development-tab-results').disabled",
        )
        check(
            "Finished predictors expose epoch provenance and evaluation links",
            "document.body.innerText.includes('6 ready to evaluate') && document.body.innerText.includes('18 epochs · P75') && !document.body.innerText.includes('Cancel remaining predictors')",
        )
        fill("Predictor method", "refit", "select")
        check(
            "Predictor library filters refits independently of job history",
            "[...document.querySelectorAll('.experiment-predictor-table')].at(-1).querySelectorAll('tbody tr').length === 3",
        )
        click("Results")
        check(
            "Finished results are concise and show completed metrics",
            "document.body.innerText.includes('AUROC') && document.body.innerText.includes('0.891') && !document.body.innerText.includes('Cancel batch') && !document.body.innerText.includes('Build predictor')",
        )
        browser("screenshot", str(output / "experiments-results.png"))
        browser("set", "viewport", "390", "844")
        check(
            "Mobile detail has no horizontal page overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        evaluate("scrollTo(0, 0); true")
        browser("screenshot", str(output / "experiments-mobile.png"))
        check(
            "Every request was handled by the offline fixture",
            "window.__experimentReview.traffic.every(item => item.matched)",
        )
        console_errors = browser("errors")
        if console_errors.get("errors"):
            raise AssertionError(console_errors)
        results.append({"name": "No browser page errors", "passed": True})
        (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{len(results)} offline experiment checks passed", flush=True)
    finally:
        browser("close")


if __name__ == "__main__":
    main()
