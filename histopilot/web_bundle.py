"""The browser bundle a source checkout serves, and whether it still matches ``web/``.

``histopilot/static`` is a copy of the Vite build in ``web/dist``. A build made through
``build()`` records a fingerprint of the ``web/`` sources it started from, so ``serve.sh``
rebuilds exactly when the frontend changed and ``histopilot serve`` can warn before it
serves an older bundle. An installed package has no ``web/`` folder and is always current.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

STAMP = ".web-source.sha256"
# Everything `npm run build` reads. Its type check covers the tests too, so they count.
INPUTS = (
    "index.html",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "vite.config.ts",
    "public",
    "src",
)


def checkout_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_fingerprint(root: Path) -> str | None:
    """SHA-256 over the relative path and content of every frontend input; None without web/."""
    web = root / "web"
    if not (web / "package.json").is_file():
        return None
    digest = hashlib.sha256()
    for name in INPUTS:
        path = web / name
        if path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            files = [path] if path.is_file() else []
        for item in files:
            digest.update(item.relative_to(web).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def bundle_state(root: Path, static: Path | None = None) -> tuple[bool, str]:
    """Whether the served bundle was built from the current ``web/`` sources, and why not."""
    static = root / "histopilot" / "static" if static is None else static
    source = source_fingerprint(root)
    if source is None:
        return True, "installed package without web/ sources"
    if not (static / "index.html").is_file():
        return False, "no frontend bundle yet"
    try:
        recorded = (static / STAMP).read_text().strip()
    except OSError:
        return False, "the bundle does not record which web/ sources built it"
    if recorded != source:
        return False, "web/ changed after the bundle was built"
    return True, "the bundle matches web/"


def build(root: Path, npm: str = "npm") -> None:
    """Run ``npm run build`` and record the sources as they were when it started.

    A file edited during the build leaves a fingerprint that no longer matches, so the next
    start rebuilds rather than trusting a bundle of uncertain content.
    """
    source = source_fingerprint(root)
    if source is None:
        raise FileNotFoundError(f"No web/ sources in {root}.")
    executable = shutil.which(npm) or npm
    subprocess.run([executable, "run", "build"], cwd=root / "web", check=True)
    (root / "web" / "dist" / STAMP).write_text(source + "\n")


def bundle(root: Path) -> int:
    """Copy ``web/dist`` into ``histopilot/static``, swapping the folder in by rename.

    A service reading the bundle sees either the old or the new folder, never a partial
    copy. Returns the number of bundled files.
    """
    source, target = root / "web" / "dist", root / "histopilot" / "static"
    if not (source / "index.html").is_file():
        raise FileNotFoundError(
            "Missing web/dist/index.html. Run npm ci && npm run build in web/ first."
        )
    incoming, previous = target.with_name("static.incoming"), target.with_name("static.previous")
    for leftover in (incoming, previous):
        if leftover.exists():
            shutil.rmtree(leftover)
    shutil.copytree(source, incoming)
    if target.exists():
        target.rename(previous)
    incoming.rename(target)
    if previous.exists():
        shutil.rmtree(previous)
    # Setuptools can otherwise retain obsolete hashed assets between wheel builds.
    staging = root / "build" / "lib" / "histopilot" / "static"
    if staging.exists():
        shutil.rmtree(staging)
    return sum(path.is_file() for path in target.rglob("*"))
