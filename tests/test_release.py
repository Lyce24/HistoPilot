"""A release agrees with itself: one version everywhere, its changelog notes, its license."""

import runpy
import tomllib
from pathlib import Path

import pytest

import histopilot

ROOT = Path(__file__).resolve().parents[1]
section = runpy.run_path(str(ROOT / "scripts" / "release_notes.py"))["section"]


def test_the_package_its_metadata_and_the_lock_name_one_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == histopilot.__version__
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    (locked,) = [package for package in lock["package"] if package["name"] == "histopilot"]
    assert locked["version"] == histopilot.__version__, "Run uv lock after changing the version."


def test_a_release_has_notes_and_a_license():
    version = histopilot.__version__
    if ".dev" in version:
        pytest.skip("development versions are released later")
    notes = section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    assert notes.splitlines()[0] and "## [" not in notes and "]: https://" not in notes
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "MIT" and project["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_notes_end_at_the_next_version_or_the_links():
    changelog = (
        "# Changelog\n\n## [Unreleased]\n\n- Next.\n\n## [0.2.0] - 2027-01-01\n\n### Added\n\n"
        "- Two.\n\n## [0.1.0] - 2026-10-29\n\n- One.\n\n"
        "[Unreleased]: https://example.org/compare/v0.2.0...HEAD\n"
        "[0.2.0]: https://example.org/v0.2.0\n"
    )
    assert section(changelog, "0.2.0") == "### Added\n\n- Two."
    assert section(changelog, "0.1.0") == "- One."
    assert section(changelog, "Unreleased") == "- Next."
    for missing in ("0.3.0", "0.1"):
        with pytest.raises(ValueError, match=missing):
            section(changelog, missing)
    with pytest.raises(ValueError, match=r"0\.4\.0"):
        section("## [0.4.0] - 2027-02-01\n\n[0.4.0]: https://example.org\n", "0.4.0")
