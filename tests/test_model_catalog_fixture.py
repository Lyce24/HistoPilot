"""The browser's copy of the model catalog must match the service's own.

The interface decides which architectures to offer, which read patch bags and
which can show attention. It reads a generated fixture rather than repeating
those facts, so this check is what keeps the two from drifting apart.
"""

import json
import unittest
from pathlib import Path

from histopilot.models import catalog

FIXTURE = Path(__file__).resolve().parents[1] / "web" / "src" / "lib" / "modelCatalog.json"


class ModelCatalogFixtureTests(unittest.TestCase):
    def test_the_published_fixture_matches_the_catalog(self):
        self.assertEqual(
            json.loads(FIXTURE.read_text(encoding="utf-8")),
            catalog.describe(),
            "Regenerate web/src/lib/modelCatalog.json from histopilot.models.catalog.describe().",
        )


if __name__ == "__main__":
    unittest.main()
