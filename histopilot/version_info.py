"""Which code a process runs: the package version and, in a source checkout, its revision.

The CLI compares its own with the service's, since a checkout's CLI can talk to a service
started from another checkout. Git is read directly, never spawned.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What this service offers clients beyond the plain API; see docs/cli-contract.md.
FEATURES = (
    "route-classes",
    "error-codes",
    "templates",
    "log-offsets",
    "scoped-tokens",
    "ai-exposure",
    "agent-requests",
    "login",
)


def _git_dir(root: Path) -> Path | None:
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        # A worktree: `.git` names the real git directory.
        text = marker.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            target = Path(text.split(":", 1)[1].strip())
            return target if target.is_absolute() else (root / target).resolve()
    return None


def _common_dir(git_dir: Path) -> Path:
    common = git_dir / "commondir"
    if common.is_file():
        return (git_dir / common.read_text(encoding="utf-8").strip()).resolve()
    return git_dir


def git_revision() -> str | None:
    """The commit this checkout has out, or None outside a git checkout. Read it once per
    process: the checkout can move on while the process runs."""
    return revision_of(ROOT)


def revision_of(root: Path) -> str | None:
    """The commit checked out at ``root``: a plain checkout or a worktree, loose or packed."""
    try:
        git_dir = _git_dir(root)
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        for folder in (git_dir, _common_dir(git_dir)):
            loose = folder / ref
            if loose.is_file():
                return loose.read_text(encoding="utf-8").strip() or None
            packed = folder / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
    except OSError:
        return None
    return None
