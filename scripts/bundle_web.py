"""Copy the Vite build into Python package data before building a release wheel."""

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source, target = root / "web" / "dist", root / "histopilot" / "static"
    if not (source / "index.html").is_file():
        raise SystemExit("Missing web/dist/index.html. Run npm ci && npm run build in web/ first.")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    # Setuptools can otherwise retain obsolete hashed assets between wheel builds.
    staging = root / "build" / "lib" / "histopilot" / "static"
    if staging.exists():
        shutil.rmtree(staging)
    print(
        f"Bundled {sum(path.is_file() for path in target.rglob('*'))} frontend files into {target}"
    )


if __name__ == "__main__":
    main()
