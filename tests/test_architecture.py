"""Checks the layering rules the running application actually depends on."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ServiceRuntimeTests(unittest.TestCase):
    """ARCHITECTURE.md: the service does not initialize Torch/CUDA models."""

    def test_building_the_api_does_not_import_torch(self):
        script = """
import sys
import histopilot.api.app  # noqa: F401
heavy = sorted(name for name in ("torch", "lightning", "h5py") if name in sys.modules)
if heavy:
    raise SystemExit("service imported compute modules: " + ", ".join(heavy))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class ModelCatalogTests(unittest.TestCase):
    """One description of the architectures; no layer re-lists their names."""

    def test_every_catalogued_model_can_be_constructed(self):
        from histopilot.models import catalog, registry

        self.assertEqual(set(registry.BUILDERS), set(catalog.NAMES))

    def test_the_catalog_imports_without_compute_dependencies(self):
        script = """
import sys
from histopilot.models import catalog
assert catalog.NAMES, "the catalog is empty"
heavy = sorted(name for name in ("torch", "lightning", "h5py", "pydantic") if name in sys.modules)
if heavy:
    raise SystemExit("catalog imported optional dependencies: " + ", ".join(heavy))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_capabilities_match_the_published_description(self):
        from histopilot.models import catalog

        described = {item["name"]: item for item in catalog.describe()}
        self.assertEqual(set(described), set(catalog.NAMES))
        for name, item in described.items():
            self.assertEqual(
                item["supportsAttention"], catalog.supports_attention(name, "image")
            )
            self.assertFalse(catalog.supports_attention(name, "clinical"))
            self.assertEqual(item["featureKind"], catalog.feature_kind(name))

    def test_unknown_architectures_are_rejected_rather_than_defaulted(self):
        from histopilot.models import catalog, registry

        self.assertFalse(catalog.is_supported("not-a-model"))
        self.assertIsNone(catalog.spec("not-a-model"))
        with self.assertRaises(ValueError):
            registry.build("not-a-model", 8, 2, {})


if __name__ == "__main__":
    unittest.main()
