"""Checks that the skeleton stays usable before optional backends are installed."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_core_imports_without_site_packages(self):
        script = """
import importlib
import pkgutil
import histopilot.domain
import histopilot.ports
for package in (histopilot.domain, histopilot.ports):
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        importlib.import_module(module.name)
"""
        completed = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_domain_does_not_depend_on_outer_layers(self):
        for path in (ROOT / "histopilot" / "domain").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                for module in modules:
                    self.assertFalse(
                        module.startswith(
                            ("histopilot.application", "histopilot.ports", "histopilot.adapters")
                        ),
                        f"{path.name} imports outer layer {module}",
                    )

    def test_unimplemented_job_submission_cannot_report_success(self):
        from histopilot.application.jobs import JobService

        with self.assertRaises(NotImplementedError):
            JobService().submit("example-run")


if __name__ == "__main__":
    unittest.main()
