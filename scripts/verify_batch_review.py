"""Check the batch editor against invented file:// fixtures; never start a server."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="agent-browser")
    parser.add_argument("--output", type=Path, default=Path("/tmp/histopilot-batch-review/browser"))
    args = parser.parse_args()
    output = args.output.resolve()
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "XDG_RUNTIME_DIR": str(runtime)}
    command = [args.browser, "--session", "histopilot-batch-review", "--allow-file-access"]
    results = []
    opened = False

    def browser(*parts):
        process = subprocess.run(
            [*command, *parts, "--json"],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if process.returncode:
            raise AssertionError(process.stdout + process.stderr)
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

    current = "window.__experimentReview.records.planning"
    patch_count = "window.__experimentReview.traffic.filter(item => item.method === 'PATCH').length"

    def mode(value):
        evaluate(f"document.querySelector('.batch-mode-option input[value={value}]').click(); true")
        browser("snapshot", "-i")

    try:
        browser("open", (output / "index.html").as_uri())
        opened = True
        browser("set", "viewport", "1440", "1100")
        click("Planning example")
        click("2. Batches")
        check(
            "Search modes, training and compute are grouped",
            "document.querySelectorAll('.batch-mode-option').length === 3 && document.body.innerText.includes('Training settings') && document.body.innerText.includes('Compute & parallelism')",
        )
        check(
            "New batches start blank with forty epochs and patience eight",
            f"{field('Start from a template', 'select')}.value === 'blank' && {field('Batch name')}.value === '' && {field('Maximum epochs')}.value === '40' && {field('Early-stopping patience')}.value === '8'",
        )
        check(
            "Batch flow ends with predictor choices and one batch save",
            "(() => { const sections = [...document.querySelector('.development-editor').children].filter(el => el.tagName === 'SECTION').map(el=>el.getAttribute('aria-label')); return JSON.stringify(sections) === JSON.stringify(['Parameter search and repeats', 'Settings', 'Parallel training', 'Configure predictors']) && !document.body.innerText.includes('Save predictor choices'); })()",
        )
        check(
            "Advanced settings begin collapsed",
            "[...document.querySelectorAll('.batch-settings-details')].every(item => !item.open)",
        )
        check(
            "Desktop batch editor has no horizontal page overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        evaluate(
            "document.querySelector('.batch-template-picker').scrollIntoView({block: 'start'}); true"
        )
        browser("screenshot", str(output / "batch-editor-desktop.png"))
        mode("grid")
        fill("Batch name", "Fifteen configurations")
        fill("Learning rates", "0.0001, 0.0002, 0.0003, 0.0004, 0.0005")
        fill("Weight decays", "0, 0.0001, 0.001")
        fill("Maximum epochs", "20")
        fill("Training seeds", "42, 43, 44")
        check(
            "Fifteen configurations and three seeds produce forty-five groups",
            "document.querySelector('.batch-size-summary').innerText.includes('15 configurations × 3 training seeds = 45 training groups')",
        )
        evaluate(
            "document.querySelector('.batch-predictor-settings input[value=both]').click(); true"
        )
        fill("Refit epoch budget", "75", "select")
        check(
            "This batch plans ninety predictors from its own both choice",
            "document.querySelector('.batch-predictor-settings').innerText.includes('90 predictors planned for this batch') && document.querySelector('.batch-predictor-settings').innerText.includes('225 fold runs')",
        )
        fill("Refit epoch budget", "custom", "select")
        fill("Custom percentile", "7e")
        before = evaluate(patch_count)
        click("Add batch to plan")
        check(
            "A partial custom percentile blocks the entire batch save",
            f"{patch_count} === {before} && {field('Custom percentile')}.getAttribute('aria-invalid') === 'true' && {current}.batchPlans.length === 2",
        )
        fill("Refit epoch budget", "75", "select")
        click("Add batch to plan")
        check(
            "Grid and training seeds save together without changing sibling batches",
            f"{current}.batchPlans.length === 3 && {current}.batchPlans[2].spec.grid.learningRates.length === 5 && {current}.batchPlans[2].spec.trainingSeeds.length === 3 && {current}.batchPlans[0].spec.recipe.learningRate === 0.0003",
        )
        check(
            "Predictor method and percentile are saved in the same batch request",
            f"{current}.batchPlans[2].spec.predictorPolicy.method === 'both' && {current}.batchPlans[2].spec.predictorPolicy.refitPercentile === 75",
        )
        evaluate(
            "document.querySelector('.batch-predictor-settings').scrollIntoView({block:'start'}); true"
        )
        browser("screenshot", str(output / "batch-predictor-settings.png"))
        before = evaluate(patch_count)
        fill("Training seeds", "42, 42")
        click("Save batch changes")
        check(
            "Duplicate seeds block saving instead of changing the saved plan",
            f"{patch_count} === {before} && document.querySelector('.batch-seeds-field input').getAttribute('aria-invalid') === 'true' && {current}.batchPlans[2].spec.trainingSeeds.length === 3",
        )
        fill("Training seeds", "42, 43, 44")
        mode("explicit")
        check(
            "First custom configuration starts from the selected grid values",
            f"{field('Learning rate')}.value === '0.0001' && {field('Maximum epochs')}.value === '20'",
        )
        click("Add configuration")
        check(
            "An identical copied row does not inflate the training count",
            "document.querySelectorAll('.batch-configuration-card').length === 2 && document.querySelector('.batch-size-summary').innerText.includes('1 configuration × 3 training seeds = 3 training groups')",
        )
        evaluate("document.querySelectorAll('.batch-configuration-card')[1].open = true; true")
        evaluate(
            "(() => { const el = [...document.querySelectorAll('.batch-configuration-card')[1].querySelectorAll('label')].find(el=>el.textContent.startsWith('Learning rate')).querySelector('input'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '0.002'); el.dispatchEvent(new Event('input', {bubbles:true})); return true; })()"
        )
        check(
            "Distinct custom settings update the count immediately",
            "document.querySelector('.batch-size-summary').innerText.includes('2 configurations × 3 training seeds = 6 training groups')",
        )
        evaluate(
            "(() => { const card = document.querySelectorAll('.batch-configuration-card')[1]; const input = [...card.querySelectorAll('label')].find(el=>el.textContent.startsWith('Dropout')).querySelector('input'); input.closest('details').open = true; Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, '1'); input.dispatchEvent(new Event('input', {bubbles:true})); input.closest('details').open = false; card.open = false; return true; })()"
        )
        before = evaluate(patch_count)
        click("Save batch changes")
        check(
            "Invalid values inside collapsed configurations reopen their groups and block save",
            f"{patch_count} === {before} && document.querySelectorAll('.batch-configuration-card')[1].open && document.querySelectorAll('.batch-configuration-card')[1].querySelector('input[aria-invalid=true]')?.value === '1'",
        )
        evaluate(
            "(() => { const input = document.querySelectorAll('.batch-configuration-card')[1].querySelector('input[aria-invalid=true]'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, '0'); input.dispatchEvent(new Event('input', {bubbles:true})); return true; })()"
        )
        click("Save batch changes")
        check(
            "Explicit row edits preserve exact zero dropout and sibling settings",
            f"{current}.batchPlans[2].spec.mode === 'explicit' && {current}.batchPlans[2].spec.configurations[1].dropout === 0 && {current}.batchPlans[2].spec.configurations[0].learningRate === 0.0001",
        )
        mode("single")
        mode("explicit")
        check(
            "Switching search modes retains custom row edits",
            "document.querySelectorAll('.batch-configuration-card').length === 2 && document.querySelectorAll('.batch-configuration-card')[1].querySelector('summary').innerText.includes('LR 0.002')",
        )
        browser("set", "viewport", "390", "844")
        check(
            "Mobile custom configuration editor has no horizontal overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        evaluate(
            "document.querySelector('.batch-custom-configurations').scrollIntoView({block:'start'}); true"
        )
        browser("screenshot", str(output / "batch-editor-mobile.png"))
        evaluate(
            "document.querySelector('.batch-predictor-settings').scrollIntoView({block:'start'}); true"
        )
        browser("screenshot", str(output / "batch-predictors-mobile.png"))
        click("Save batch changes")
        click("New batch")
        check(
            "Starting another batch resets its predictor choice and training defaults",
            f"document.querySelector('.batch-predictor-settings input[value=ensemble]').checked && {field('Maximum epochs')}.value === '40' && !document.querySelector('.experiment-refit-budget')",
        )
        fill("Batch name", "Cross-validation only")
        evaluate(
            "document.querySelector('.batch-predictor-settings input[value=skip]').click(); true"
        )
        click("Add batch to plan")
        check(
            "Different batch predictor choices remain independent",
            f"{current}.batchPlans.length === 4 && {current}.batchPlans[3].spec.predictorPolicy.method === 'skip' && {current}.batchPlans[2].spec.predictorPolicy.method === 'both' && {current}.batchPlans[2].spec.predictorPolicy.refitPercentile === 75",
        )
        check(
            "Batch cards compare predictor choices without opening settings",
            "[...document.querySelectorAll('.batch-plan-card')].some(el=>el.querySelector('.batch-predictor-summary')?.innerText.includes('Predictors: Both · P75')) && [...document.querySelectorAll('.batch-plan-card')].some(el=>el.querySelector('.batch-predictor-summary')?.innerText.includes('Predictors: Skip'))",
        )
        browser("set", "viewport", "1440", "1100")
        evaluate("document.querySelector('.batch-plan-list').scrollIntoView({block:'start'}); true")
        browser("screenshot", str(output / "batch-plan-predictor-choices.png"))
        evaluate(
            "document.querySelectorAll('.batch-plan-card')[2].querySelector('button').click(); true"
        )
        browser("snapshot", "-i")
        check(
            "Editing a saved batch restores its own predictor percentile",
            f"document.querySelector('.batch-predictor-settings input[value=both]').checked && {field('Refit epoch budget', 'select')}.value === '75'",
        )
        evaluate(
            f"(() => {{ const record = {current}; const spec = {{...record.batchPlans[0].spec, batchName:'Earlier recipe', inputs:{{...record.inputs, featureBundleId:'older-bundle', loadingPolicy:'auto'}}, predictorPolicy:{{method:'refit',refitPercentile:90}}}}; record.drafts = [{{id:'older-draft',name:'Earlier recipe',payload:{{type:'development-batch',spec}}}}]; window.__experimentReview.refresh(); return true; }})()"
        )
        check(
            "Earlier recipes remain available to copy into the current plan",
            f"Boolean({button('Use draft settings')})",
        )
        click("Use draft settings")
        check(
            "A recipe with different inputs keeps verified inputs and remains editable",
            f"document.querySelector('[role=tab][aria-selected=true]').textContent === '2. Batches' && !document.querySelector('.development-editor').disabled && {current}.inputs.featureBundleId === 'bundle-review' && {current}.inputs.loadingPolicy === 'native' && {field('Batch name')}.value === 'Earlier recipe' && document.querySelector('.batch-predictor-settings input[value=refit]').checked && {field('Refit epoch budget', 'select')}.value === '90' && document.body.innerText.includes('This batch will use this experiment’s verified inputs.')",
        )
        click("Check batch")
        check(
            "Copied settings are checked against the current experiment inputs",
            "(() => { const request = window.__experimentReview.traffic.filter(item=>item.path.endsWith('/mil-experiments/batches/preview')).at(-1); return request?.body.inputs.featureBundleId === 'bundle-review' && request.body.inputs.loadingPolicy === 'native' && request.body.predictorPolicy.refitPercentile === 90; })()",
        )
        click("Add batch to plan")
        check(
            "The copied recipe saves with shared verified inputs and its own predictor policy",
            f"{current}.batchPlans.length === 5 && {current}.batchPlans[4].spec.inputs.featureBundleId === 'bundle-review' && {current}.batchPlans[4].spec.inputs.loadingPolicy === 'native' && {current}.batchPlans[4].spec.predictorPolicy.method === 'refit' && {current}.batchPlans[4].spec.predictorPolicy.refitPercentile === 90",
        )
        click("Running example")
        click("Batches")
        check(
            "Submitted batches expose readable settings without editor controls",
            "document.querySelector('.batch-plan-settings') && !document.querySelector('.development-editor') && !document.body.innerText.includes('Edit batch')",
        )
        check(
            "All requests remain inside the fixture",
            "window.__experimentReview.traffic.every(item => item.matched)",
        )
        errors = browser("errors")
        if errors.get("errors"):
            raise AssertionError(errors)
        results.append({"name": "No browser page errors", "passed": True})
        (output / "batch-checks.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{len(results)} offline batch checks passed", flush=True)
    finally:
        if opened:
            browser("close")


if __name__ == "__main__":
    main()
