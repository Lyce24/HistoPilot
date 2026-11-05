"""Package HistoPilot project folders as the public demo's studies.

    python package_demo.py OUT.tar.gz PROJECT_FOLDER [PROJECT_FOLDER ...]

Writes demo-projects/<name>/ for each project into one gzipped tar, leaving out what a
read-only visitor never needs and a public page must not carry: model checkpoints, feature
packs and per-slide feature embeddings (derived from features whose license may forbid
redistribution), slide images, and SQLite lock sidecars. Stop the projects' service first,
so its database is not copied mid-write. `histopilot serve --demo --demo-projects` reads the
extracted folder.
"""

import argparse
import sys
import tarfile
from pathlib import Path

DESCRIPTOR = "histopilot-project.json"
LEFT_OUT_SUFFIXES = {
    ".ckpt",  # model weights
    ".pt",
    ".pth",
    ".bin",  # feature packs
    ".svs",  # slide images
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".sdpc",
}
LEFT_OUT_NAMES = {"slide-embeddings.npz", "service.lock"}


def left_out(path: Path) -> bool:
    return path.suffix.lower() in LEFT_OUT_SUFFIXES or path.name in LEFT_OUT_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("projects", type=Path, nargs="+")
    args = parser.parse_args()
    kept = skipped = 0
    with tarfile.open(args.out, "w:gz") as archive:
        for project in args.projects:
            if not (project / DESCRIPTOR).is_file():
                sys.exit(f"{project} is not a HistoPilot project folder (no {DESCRIPTOR}).")
            for path in sorted(project.rglob("*")):
                if path.is_symlink():
                    sys.exit(f"{path}: symbolic links are not packaged.")
                if not path.is_file():
                    continue
                if left_out(path):
                    skipped += 1
                    continue
                archive.add(
                    path, arcname=f"demo-projects/{project.name}/{path.relative_to(project)}"
                )
                kept += 1
    size = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out}: {kept} files, {skipped} left out, {size:.1f} MB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
