"""Exercise the manual launcher with stubs; never start a HistoPilot service."""

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
    (bin_dir / "dirname").symlink_to(shutil.which("dirname"))
    (bin_dir / "cat").symlink_to(shutil.which("cat"))
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{BASH}\n"
        "printf '%s\\0' \"$PWD\" \"$@\" > \"$CAPTURE\"\n"
        'exit "${STUB_EXIT:-0}"\n'
    )
    uv.chmod(0o755)
    runtime = root / ".venv" / "bin"
    runtime.mkdir(parents=True)
    for executable in ("python", "histopilot"):
        path = runtime / executable
        path.write_text(f"#!{BASH}\n" 'exit "${IMPORT_EXIT:-0}"\n')
        path.chmod(0o755)
    static = root / "histopilot" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("test bundle")
    capture = tmp_path / "uv-arguments"
    env = {**os.environ, "PATH": str(bin_dir), "CAPTURE": str(capture)}
    env.pop("UV_PROJECT_ENVIRONMENT", None)

    def run(*args, **overrides):
        return subprocess.run(
            [BASH, str(root / "serve.sh"), *args], cwd=tmp_path,
            env={**env, **overrides}, capture_output=True, text=True, timeout=10,
        )

    return root, bin_dir, capture, run


def test_foreground_arguments_cwd_and_exit_status(launcher):
    root, _, capture, run = launcher
    forwarded = [
        "--workspace", "/a workspace", "--data-root", "/first data",
        "--data-root", "/second", "--port", "8788", "--browser",
    ]
    result = run(*forwarded, STUB_EXIT="23")
    assert result.returncode == 23
    assert capture.read_bytes().decode().split("\0") == [
        str(root), "run", "--locked", "--no-sync", "--offline", "--no-python-downloads",
        "histopilot", "serve", "--no-browser", *forwarded, "",
    ]


def test_help_works_without_uv_or_environment(launcher):
    root, bin_dir, capture, run = launcher
    (bin_dir / "uv").unlink()
    shutil.rmtree(root / ".venv")
    result = run("--help")
    assert result.returncode == 0
    assert "Usage: bash serve.sh" in result.stdout
    assert not capture.exists()


@pytest.mark.parametrize("missing", ["uv", "environment", "imports", "bundle"])
def test_missing_setup_is_actionable_and_does_not_launch(launcher, missing):
    root, bin_dir, capture, run = launcher
    overrides = {}
    if missing == "uv":
        (bin_dir / "uv").unlink()
    elif missing == "environment":
        shutil.rmtree(root / ".venv")
    elif missing == "imports":
        overrides["IMPORT_EXIT"] = "1"
    else:
        (root / "histopilot" / "static" / "index.html").unlink()
    result = run(**overrides)
    assert result.returncode == 1
    assert "Cannot start HistoPilot" in result.stderr
    assert not capture.exists()


def test_dev_mode_accepts_missing_bundle(launcher):
    root, _, capture, run = launcher
    (root / "histopilot" / "static" / "index.html").unlink()
    result = run("--dev")
    assert result.returncode == 0
    assert capture.read_bytes().endswith(b"--no-browser\0--dev\0")
