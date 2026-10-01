"""The `histopilot` skill's reference is generated from the code and must stay current."""

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "skill_reference.py"


def test_the_skill_reference_matches_the_code():
    module = runpy.run_path(str(SCRIPT))
    assert module["TARGET"].read_text(encoding="utf-8") == module["render"](), (
        "Regenerate it with: python scripts/skill_reference.py"
    )
