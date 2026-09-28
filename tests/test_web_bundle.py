"""The served bundle records the web/ sources it was built from; stale bundles are detected."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from histopilot import web_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    web = tmp_path / "web"
    (web / "src" / "components").mkdir(parents=True)
    (web / "package.json").write_text('{"name": "fixture"}')
    (web / "index.html").write_text("<div id=app></div>")
    (web / "src" / "main.tsx").write_text("export const a = 1;")
    (web / "src" / "components" / "Chart.tsx").write_text("export const b = 2;")
    # A fake npm "build": dist gets an index.html naming the source it saw.
    npm = tmp_path / "npm"
    npm.write_text(
        f"#!{shutil.which('bash')}\n"
        'test "${FAIL_BUILD:-0}" = 0 || exit 3\n'
        "mkdir -p dist/assets && cp src/main.tsx dist/assets/main.js && echo built > dist/index.html\n"
    )
    npm.chmod(0o755)
    return tmp_path, str(npm)


def test_fingerprint_covers_frontend_inputs_only(checkout):
    root, _ = checkout
    first = web_bundle.source_fingerprint(root)
    assert first == web_bundle.source_fingerprint(root)
    for ignored in ("node_modules/pkg/index.js", "dist/index.html", "scripts/verify.mjs"):
        path = root / "web" / ignored
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored")
    assert web_bundle.source_fingerprint(root) == first
    (root / "web" / "src" / "components" / "Chart.tsx").write_text("export const b = 3;")
    assert web_bundle.source_fingerprint(root) != first
    renamed = root / "web" / "src" / "components" / "Renamed.tsx"
    (root / "web" / "src" / "components" / "Chart.tsx").rename(renamed)
    assert web_bundle.source_fingerprint(root) != first
    assert web_bundle.source_fingerprint(root / "installed-package") is None


def test_build_and_bundle_record_sources_and_detect_changes(checkout, monkeypatch):
    root, npm = checkout
    assert web_bundle.bundle_state(root) == (False, "no frontend bundle yet")
    web_bundle.build(root, npm=npm)
    assert web_bundle.bundle(root) == 3
    static = root / "histopilot" / "static"
    assert (static / "index.html").read_text() == "built\n"
    assert (static / web_bundle.STAMP).read_text().strip() == web_bundle.source_fingerprint(root)
    assert web_bundle.bundle_state(root) == (True, "the bundle matches web/")
    (root / "web" / "src" / "main.tsx").write_text("export const a = 2;")
    assert web_bundle.bundle_state(root) == (False, "web/ changed after the bundle was built")
    # A bundle copied from a plain `npm run build` does not say what built it.
    (static / web_bundle.STAMP).unlink()
    assert web_bundle.bundle_state(root)[0] is False
    assert web_bundle.bundle_state(root / "installed")[0] is True


def test_rebundling_swaps_the_folder_and_leaves_no_staging(checkout):
    root, npm = checkout
    web_bundle.build(root, npm=npm)
    web_bundle.bundle(root)
    (root / "web" / "src" / "main.tsx").write_text("export const a = 'new';")
    old = root / "histopilot" / "static" / "stale-asset.js"
    old.write_text("from an older build")
    web_bundle.build(root, npm=npm)
    web_bundle.bundle(root)
    static = root / "histopilot" / "static"
    assert not old.exists()  # obsolete hashed assets do not survive a rebundle
    assert (static / "assets" / "main.js").read_text() == "export const a = 'new';"
    assert sorted(path.name for path in (root / "histopilot").iterdir()) == ["static"]


def test_a_failed_build_leaves_the_bundle_untouched(checkout, monkeypatch):
    root, npm = checkout
    web_bundle.build(root, npm=npm)
    web_bundle.bundle(root)
    before = (root / "histopilot" / "static" / web_bundle.STAMP).read_text()
    (root / "web" / "src" / "main.tsx").write_text("broken")
    monkeypatch.setenv("FAIL_BUILD", "1")
    with pytest.raises(subprocess.CalledProcessError):
        web_bundle.build(root, npm=npm)
    assert (root / "histopilot" / "static" / web_bundle.STAMP).read_text() == before
    assert web_bundle.bundle_state(root)[0] is False


def test_bundle_script_check_exits_by_state_without_changing_anything(tmp_path):
    # The real checkout: --check only reads, and reports in the service's label column.
    static = ROOT / "histopilot" / "static"
    before = (
        sorted((path.name, path.stat().st_mtime_ns) for path in static.iterdir())
        if static.exists()
        else None
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bundle_web.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in {0, 1}
    assert result.stdout.startswith("Frontend   ")
    after = (
        sorted((path.name, path.stat().st_mtime_ns) for path in static.iterdir())
        if static.exists()
        else None
    )
    assert before == after
