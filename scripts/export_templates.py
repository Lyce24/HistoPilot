"""Write the browser's copy of the server-owned templates: web/src/lib/templates.json.

python scripts/export_templates.py          # write the file
python scripts/export_templates.py --check  # exit 1 if it is out of date
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histopilot.templates import describe  # noqa: E402

TARGET = ROOT / "web" / "src" / "lib" / "templates.json"


def render() -> str:
    return json.dumps(describe(), indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    from scripts.generated import main

    raise SystemExit(main(__doc__.splitlines()[0], TARGET, render))
