"""Copy the Vite build into histopilot/static: the UI a checkout serves and a wheel ships.

python scripts/bundle_web.py           copy web/dist as built (release wheels, CI)
python scripts/bundle_web.py --build   build web/ first and record the sources it used
python scripts/bundle_web.py --check   exit 1 when the bundle is missing or older than web/
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The bundler needs only the standard library; run it without an installed package too.
sys.path.insert(0, str(ROOT))

from histopilot import web_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="run npm run build in web/ first")
    mode.add_argument("--check", action="store_true", help="report whether the bundle is current")
    args = parser.parse_args()
    if args.check:
        current, reason = web_bundle.bundle_state(ROOT)
        print(f"Frontend   {'current' if current else 'out of date'}: {reason}")
        raise SystemExit(0 if current else 1)
    try:
        if args.build:
            print("Frontend   building web/ ...", flush=True)
            web_bundle.build(ROOT)
        count = web_bundle.bundle(ROOT)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Frontend build failed (exit {error.returncode}); the bundled UI was not changed."
        ) from None
    print(f"Bundled {count} frontend files into {ROOT / 'histopilot' / 'static'}")


if __name__ == "__main__":
    main()
