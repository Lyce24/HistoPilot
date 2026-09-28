"""Exercise the manual launcher with stubs; never start a HistoPilot service or a build."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "serve.sh"
BASH = shutil.which("bash")


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    shutil.copyfile(LAUNCHER, root / "serve.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("dirname", "cat", "grep"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    events = tmp_path / "events"

    def stub(path, body):
        path.write_text(f"#!{BASH}\n{body}")
        path.chmod(0o755)

    stub(
        bin_dir / "uv",
        'printf \'%s\\0\' "$PWD" "$@" > "$CAPTURE"\n'
        'echo launch >> "$EVENTS"\n'
        'exit "${STUB_EXIT:-0}"\n',
    )
    stub(bin_dir / "ps", 'printf "%s\\n" "${PS_OUTPUT:-/usr/bin/bash}"\n')
    stub(bin_dir / "npm", "exit 0\n")
    runtime = root / ".venv" / "bin"
    runtime.mkdir(parents=True)
    # The import probe, the bundle check and the build all go through this interpreter.
    stub(
        runtime / "python",
        'case "$*" in\n'
        '  *"bundle_web.py --check"*) echo check >> "$EVENTS"; exit "${CHECK_EXIT:-0}" ;;\n'
        '  *"bundle_web.py --build"*) echo build >> "$EVENTS"; exit "${BUILD_EXIT:-0}" ;;\n'
        "esac\n"
        'exit "${IMPORT_EXIT:-0}"\n',
    )
    stub(runtime / "histopilot", "exit 0\n")
    static = root / "histopilot" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("test bundle")
    (root / "web" / "node_modules").mkdir(parents=True)
    capture = tmp_path / "uv-arguments"
    env = {**os.environ, "PATH": str(bin_dir), "CAPTURE": str(capture), "EVENTS": str(events)}
    env.pop("UV_PROJECT_ENVIRONMENT", None)

    def run(*args, **overrides):
        return subprocess.run(
            [BASH, str(root / "serve.sh"), *args],
            cwd=tmp_path,
            env={**env, **overrides},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def happened():
        return events.read_text().split() if events.exists() else []

    return root, bin_dir, capture, run, happened


def test_foreground_arguments_cwd_and_exit_status(launcher):
    root, _, capture, run, happened = launcher
    forwarded = [
        "--workspace",
        "/a workspace",
        "--data-root",
        "/first data",
        "--data-root",
        "/second",
        "--port",
        "8788",
        "--browser",
    ]
    result = run(*forwarded, STUB_EXIT="23")
    assert result.returncode == 23
    assert capture.read_bytes().decode().split("\0") == [
        str(root),
        "run",
        "--locked",
        "--no-sync",
        "--offline",
        "--no-python-downloads",
        "histopilot",
        "serve",
        "--no-browser",
        *forwarded,
        "",
    ]
    assert happened() == ["check", "launch"]  # a current bundle is not rebuilt


def test_help_works_without_uv_or_environment(launcher):
    root, bin_dir, capture, run, happened = launcher
    (bin_dir / "uv").unlink()
    shutil.rmtree(root / ".venv")
    result = run("--help")
    assert result.returncode == 0
    assert "Usage: bash serve.sh" in result.stdout
    assert "--no-build" in result.stdout
    assert not capture.exists() and happened() == []


@pytest.mark.parametrize("missing", ["uv", "environment", "imports"])
def test_missing_setup_is_actionable_and_does_not_launch(launcher, missing):
    root, bin_dir, capture, run, happened = launcher
    overrides = {}
    if missing == "uv":
        (bin_dir / "uv").unlink()
    elif missing == "environment":
        shutil.rmtree(root / ".venv")
    else:
        overrides["IMPORT_EXIT"] = "1"
    result = run(**overrides)
    assert result.returncode == 1
    assert "Cannot start HistoPilot" in result.stderr
    assert not capture.exists() and "build" not in happened()


def test_a_changed_frontend_is_rebuilt_before_the_service_starts(launcher):
    _, _, capture, run, happened = launcher
    result = run("--port", "8788", CHECK_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert happened() == ["check", "build", "launch"]
    assert capture.read_bytes().endswith(b"--no-browser\0--port\x008788\0")


def test_a_failed_build_never_starts_the_service(launcher):
    _, _, capture, run, happened = launcher
    result = run(CHECK_EXIT="1", BUILD_EXIT="2")
    assert result.returncode == 1
    assert "frontend build failed" in result.stderr
    assert "--no-build" in result.stderr
    assert happened() == ["check", "build"] and not capture.exists()


def test_the_bundle_is_never_rebuilt_under_a_running_service_from_this_checkout(launcher):
    root, _, capture, run, happened = launcher
    running = f"{root}/.venv/bin/python3 {root}/.venv/bin/histopilot serve --port 8788"
    result = run(CHECK_EXIT="1", PS_OUTPUT=running)
    assert result.returncode == 1
    assert "a service from this checkout is running" in result.stderr
    assert happened() == ["check"] and not capture.exists()
    # A service of another checkout does not serve this bundle.
    other = "/elsewhere/.venv/bin/python3 /elsewhere/.venv/bin/histopilot serve"
    assert run(CHECK_EXIT="1", PS_OUTPUT=other).returncode == 0
    assert happened()[-2:] == ["build", "launch"]


def test_no_build_serves_the_existing_bundle_and_is_not_forwarded(launcher):
    root, _, capture, run, happened = launcher
    running = f"{root}/.venv/bin/histopilot serve"
    result = run("--no-build", "--port", "8799", CHECK_EXIT="1", PS_OUTPUT=running)
    assert result.returncode == 0, result.stderr
    assert happened() == ["launch"]
    assert capture.read_bytes().endswith(b"--no-browser\0--port\x008799\0")
    (root / "histopilot" / "static" / "index.html").unlink()
    missing = run("--no-build")
    assert missing.returncode == 1 and "frontend bundle is missing" in missing.stderr


@pytest.mark.parametrize("missing", ["npm", "node_modules"])
def test_a_rebuild_without_node_tooling_is_actionable(launcher, missing):
    root, bin_dir, capture, run, happened = launcher
    if missing == "npm":
        (bin_dir / "npm").unlink()
    else:
        (root / "web" / "node_modules").rmdir()
    result = run(CHECK_EXIT="1")
    assert result.returncode == 1
    assert ("npm is not on PATH" if missing == "npm" else "npm --prefix web ci") in result.stderr
    assert happened() == ["check"] and not capture.exists()


def test_dev_mode_accepts_missing_bundle(launcher):
    root, _, capture, run, happened = launcher
    (root / "histopilot" / "static" / "index.html").unlink()
    result = run("--dev", CHECK_EXIT="1")
    assert result.returncode == 0
    assert capture.read_bytes().endswith(b"--no-browser\0--dev\0")
    assert happened() == ["launch"]  # Vite serves the UI; nothing is built
