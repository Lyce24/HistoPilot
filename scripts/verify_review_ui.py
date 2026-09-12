"""Exercise the standalone UI fixture with agent-browser; never start a service."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="agent-browser")
    parser.add_argument("--output", type=Path, default=Path("/tmp/histopilot-dev-review/browser"))
    args = parser.parse_args()
    output = args.output.resolve()
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "XDG_RUNTIME_DIR": str(runtime)}
    command = [args.browser, "--session", "histopilot-dev-review", "--allow-file-access"]
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
                raise AssertionError(f"{name}: {expression}")
            time.sleep(0.15)
        results.append({"name": name, "passed": True})
        print(f"PASS {name}", flush=True)

    def click(name):
        evaluate(
            "(() => { const button = [...document.querySelectorAll('button')]"
            f".find(el => el.textContent.trim() === {json.dumps(name)});"
            "if (!button || button.disabled) throw Error('Button unavailable');"
            "button.click(); return true; })()"
        )

    def field(label):
        return (
            "[...document.querySelectorAll('label')]"
            f".find(el => el.textContent.startsWith({json.dumps(label)})).querySelector('input')"
        )

    def fill(label, value):
        evaluate(
            f"(() => {{ const input = {field(label)};"
            "Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input,"
            f"{json.dumps(value)}); input.dispatchEvent(new Event('input', {{bubbles:true}}));"
            "return true; })()"
        )

    count = "document.querySelector('[data-testid=inference-save-count]').textContent"
    try:
        browser("open", (output / "index.html").as_uri())
        browser("set", "viewport", "1440", "1100")
        check(
            "Project roadmap renders nine modules",
            "document.querySelectorAll('[data-module]').length === 9",
        )
        check(
            "Fresh project begins with datasets",
            "document.querySelector('.roadmap-next h2').textContent === 'Datasets'",
        )
        check(
            "Fixture mocks all API traffic",
            "window.__reviewTraffic.length > 10 && window.__reviewTraffic.every(item => item.matched)",
        )
        check(
            "Desktop has no horizontal overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        browser("snapshot", "-i")
        browser("screenshot", str(output / "roadmap-desktop.png"))
        browser("set", "viewport", "390", "844")
        check(
            "Mobile has no horizontal overflow",
            "document.documentElement.scrollWidth <= innerWidth",
        )
        evaluate("document.querySelector('[aria-label=\"Toggle navigation\"]').click()")
        check(
            "Mobile navigation opens",
            "document.querySelector('[aria-label=\"Toggle navigation\"]').getAttribute('aria-expanded') === 'true'",
        )
        browser("press", "Escape")
        check(
            "Escape closes navigation and restores focus",
            "document.querySelector('[aria-label=\"Toggle navigation\"]').getAttribute('aria-expanded') === 'false' && document.activeElement.getAttribute('aria-label') === 'Toggle navigation'",
        )
        browser("snapshot", "-i")
        browser("screenshot", str(output / "roadmap-mobile.png"))

        click("Inference controls")
        browser("set", "viewport", "1200", "950")
        browser("snapshot", "-i")
        evaluate("document.querySelector('details').open = true")
        fill("Data-loading workers", "")
        click("Save inference settings")
        check(
            "Blank workers stay blank and block save",
            f"{field('Data-loading workers')}.value === '' && {count}.includes('0')",
        )
        fill("Data-loading workers", "0")
        click("Save inference settings")
        check("Explicit zero workers can be saved", f"{count}.includes('1')")
        fill("Decision threshold", "1e-")
        click("Save inference settings")
        check(
            "Partial exponent blocks save",
            f"{field('Decision threshold')}.value === '1e-' && {count}.includes('1')",
        )
        fill("Decision threshold", ".5")
        click("Save inference settings")
        check("Valid decimal threshold saves", f"{count}.includes('2')")
        fill("Batch size", "1025")
        evaluate("document.querySelector('details').open = false")
        click("Save inference settings")
        check(
            "Invalid hidden control opens its details and blocks save",
            f"document.querySelector('details').open && {count}.includes('2')",
        )
        fill("Batch size", "32")
        click("Save inference settings")
        check("Corrected batch size saves", f"{count}.includes('3')")
        click("Load legacy maximum aggregation")
        evaluate("document.querySelector('details').open = true")
        check(
            "Unsupported legacy aggregation is visible",
            "[...document.querySelectorAll('select')].some(el => el.value === 'max' && el.selectedOptions[0].disabled)",
        )
        browser("screenshot", str(output / "inference-controls.png"))

        click("Compute retry")
        check(
            "Real compute controls load a ready job",
            "document.body.innerText.includes('Train refit model')",
        )
        evaluate(
            "(() => {const button = [...document.querySelectorAll('button')].find(el => el.textContent === 'Train refit model'); button.click(); button.click(); return true;})()"
        )
        check(
            "Lost response retains explicit retry without duplicate submission",
            "window.__reviewOperations.length === 1 && document.body.innerText.includes('Retry launch request')",
        )
        click("Retry launch request")
        check(
            "Explicit retry reuses the original operation",
            "window.__reviewOperations.length === 2 && window.__reviewOperations[0] === window.__reviewOperations[1] && document.body.innerText.includes('Queued') && !document.body.innerText.includes('Retry launch request')",
        )
        browser("screenshot", str(output / "compute-retry.png"))

        click("Rendering recovery")
        check(
            "Render failure displays recovery",
            "document.body.innerText.includes('This view could not be displayed')",
        )
        check(
            "Navigation remains available after rendering failure",
            "!!document.querySelector('nav[aria-label=\"Offline verification scenarios\"] button')",
        )
        browser("snapshot", "-i")
        browser("screenshot", str(output / "render-recovery.png"))
        click("Resolve fixture fault")
        click("Try this view again")
        check(
            "Retry remounts the recovered view",
            "!!document.querySelector('#recovered-view') && !document.querySelector('.workspace-recovery')",
        )
        click("Project workspace")
        check(
            "Project navigation recovers after another view failed",
            "document.querySelectorAll('[data-module]').length === 9",
        )
    finally:
        (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
        browser("close")
    print(f"{len(results)} offline browser checks passed; evidence: {output}")


if __name__ == "__main__":
    main()
