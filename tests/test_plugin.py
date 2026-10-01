"""The Claude Code plugin: its manifests agree, and its launcher starts the agent server."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "histopilot"
LAUNCHER = PLUGIN / "bin" / "histopilot-mcp"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_marketplace_lists_the_plugin_it_ships():
    marketplace = load(ROOT / ".claude-plugin" / "marketplace.json")
    manifest = load(PLUGIN / ".claude-plugin" / "plugin.json")
    (entry,) = marketplace["plugins"]
    assert entry["name"] == manifest["name"] == "histopilot"
    assert (ROOT / entry["source"]).resolve() == PLUGIN
    assert entry["version"] == manifest["version"]
    assert manifest["userConfig"]["token"]["sensitive"] is True


def test_the_mcp_server_runs_the_shipped_launcher_with_the_plugin_settings():
    (server,) = load(PLUGIN / ".mcp.json")["mcpServers"].values()
    assert server["command"] == "${CLAUDE_PLUGIN_ROOT}/bin/histopilot-mcp"
    assert LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK)
    settings = load(PLUGIN / ".claude-plugin" / "plugin.json")["userConfig"]
    used = {
        value[len("${user_config.") : -1]
        for value in server["env"].values()
        if "user_config" in value
    }
    assert used == set(settings)


def test_the_skill_ships_in_the_plugin():
    text = (PLUGIN / "skills" / "histopilot" / "SKILL.md").read_text(encoding="utf-8")
    front = text.split("---")[1]
    assert "name: histopilot" in front and "description:" in front
    assert (PLUGIN / "skills" / "histopilot" / "reference.md").is_file()


def launch(tmp_path, env: dict, uvx: bool = False) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if uvx:
        (bin_dir / "uvx").write_text("#!/bin/sh\n")
        (bin_dir / "uvx").chmod(0o755)
    base = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HISTOPILOT_MCP_DRY_RUN": "1",
    }
    return subprocess.run(
        [str(LAUNCHER)], env=base | env, capture_output=True, text=True, timeout=30, check=False
    )


@pytest.mark.parametrize("token", [None, "", "${user_config.token}"])
def test_the_launcher_explains_a_missing_token(tmp_path, token):
    env = {} if token is None else {"HISTOPILOT_PLUGIN_TOKEN": token}
    result = launch(tmp_path, env, uvx=True)
    assert result.returncode == 2 and "histopilot token create" in result.stderr


def test_the_launcher_prefers_the_plugin_settings_and_never_prints_the_token(tmp_path):
    result = launch(
        tmp_path,
        {
            "HISTOPILOT_PLUGIN_TOKEN": "hpt_secret",
            "HISTOPILOT_PLUGIN_COMMAND": "~/HistoPilot/.venv-agent/bin/histopilot",
            "HISTOPILOT_PLUGIN_URL": "${user_config.service_url}",
            "HISTOPILOT_URL": "http://127.0.0.1:8788",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        f"{tmp_path}/HistoPilot/.venv-agent/bin/histopilot",
        "agent",
        "serve",
        "--url",
        "http://127.0.0.1:8788",
    ]
    assert "hpt_secret" not in result.stdout + result.stderr


def test_the_launcher_runs_histopilot_from_github_with_uvx(tmp_path):
    result = launch(tmp_path, {"HISTOPILOT_TOKEN": "hpt_secret"}, uvx=True)
    assert result.returncode == 0, result.stderr
    assert "histopilot\\[agent\\]\\ @\\ git+https://github.com/Lyce24/HistoPilot" in result.stdout
    assert result.stdout.split()[-3:] == ["serve", "--url", "http://127.0.0.1:8787"]
    # uvx picks an interpreter before it reads HistoPilot's requires-python.
    assert "--python \\>=3.11" in result.stdout
    if shutil.which("uvx", path="/usr/bin:/bin") is None:
        missing = launch(tmp_path / "none", {"HISTOPILOT_TOKEN": "hpt_secret"})
        assert missing.returncode == 2 and "install uv" in missing.stderr


def test_only_the_mcp_resource_pages_of_docs_are_packaged():
    # resources/docs links to docs/, which also holds private, gitignored files: name each
    # page to package, and never use a pattern that walks into the link.
    import tomllib

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    walking = ("recursive-include histopilot/resources", "graft histopilot/resources")
    assert not [line for line in manifest if line.startswith(walking)]
    included = {
        line.split()[1]
        for line in manifest
        if line.startswith("include histopilot/resources/docs/")
    }
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    # Nor may package discovery walk into it and pick up its *.py files.
    excluded = project["tool"]["setuptools"]["packages"]["find"]["exclude"]
    assert {"histopilot.resources.docs", "histopilot.resources.docs.*"} <= set(excluded)
    package_data = project["tool"]["setuptools"]["package-data"]["histopilot"]
    listed = {
        f"histopilot/{entry}" for entry in package_data if entry.startswith("resources/docs/")
    }
    assert included == listed and listed and all("*" not in entry for entry in listed)
    server = (ROOT / "histopilot" / "agent" / "server.py").read_text(encoding="utf-8")
    assert all(f'"{Path(entry).name}"' in server for entry in listed)
