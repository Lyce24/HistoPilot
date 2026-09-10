"""The launcher must not accidentally expose paths or pretend to execute work."""

import json
import os

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from histopilot.cli import app
from histopilot.config import Settings, load_settings
from histopilot.contracts.experiment import ExperimentSpec
from histopilot.service_lock import service_lock


def test_configuration_paths_are_relative_to_config_file(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[storage]\nworkspace = "state"\ndata_roots = ["slides"]\n')
    settings = load_settings(config)
    assert settings.workspace == tmp_path / "state"
    assert settings.data_roots == (tmp_path / "slides",)
    assert not settings.workspace.exists()
    assert load_settings(config, data_roots=()).data_roots == ()


@pytest.mark.parametrize("root_source", ["none", "cli", "config"])
def test_serve_reports_picker_roots_without_changing_access(tmp_path, monkeypatch, root_source):
    roots = [tmp_path / "slides", tmp_path / "features"] if root_source != "none" else []
    for root in roots:
        root.mkdir()
    config = tmp_path / "config.toml"
    configured_roots = (
        json.dumps([root.name for root in roots]) if root_source == "config" else "[]"
    )
    config.write_text(f'[storage]\nworkspace = "state"\ndata_roots = {configured_roots}\n')
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setenv("HISTOPILOT_SERVICE_SETTINGS", "{}")
    arguments = ["serve", "--config", str(config), "--dev", "--no-browser"]
    if root_source == "cli":
        for root in roots:
            arguments.extend(["--data-root", str(root)])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][0] == ("histopilot.cli:create_dev_app",)
    settings = json.loads(os.environ["HISTOPILOT_SERVICE_SETTINGS"])
    assert settings["data_roots"] == [str(root) for root in roots]
    if roots:
        assert "Read-only folders available in the data/slide picker" in result.output
        for root in roots:
            assert str(root) in result.output
        assert "No source folders configured" not in result.output
    else:
        assert "No source folders configured; the data/slide picker is empty" in result.output
        assert "--data-root /path/to/data (repeat for more folders)" in result.output
        assert "[storage].data_roots" in result.output
        assert "Experiment storage is available in the workspace above" in result.output


def test_loopback_and_config_errors_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        Settings(host="0.0.0.0")
    with pytest.raises(ValueError, match="Configuration file"):
        load_settings(tmp_path / "missing.toml")
    config = tmp_path / "config.toml"
    config.write_text('[storage]\ndata_roots = "/all/files"\n')
    with pytest.raises(ValueError, match="array"):
        load_settings(config)


def test_workspace_lock_rejects_duplicate_and_releases(tmp_path):
    with service_lock(tmp_path), pytest.raises(RuntimeError, match="already holds"):
        with service_lock(tmp_path):
            pass
    with service_lock(tmp_path):
        pass


def test_cli_and_api_share_manifest_shape_without_execution(tmp_path):
    spec = ExperimentSpec(
        dataset_id="crc-demo-v1",
        cohort_id="cohort-demo-all",
        split_id="split-demo-42",
        feature_set_id="feature-uni2-demo",
        encoder_id="uni2",
        mil_model="abmil",
        seeds=(84, 42, 42),
    )
    assert spec.seeds == (42, 84)
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({**spec.model_dump(), "shell_command": "anything"})
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({**spec.model_dump(), "seeds": [True]})
    path = tmp_path / "manifest.json"
    path.write_text(spec.model_dump_json())
    runner = CliRunner()
    checked = runner.invoke(app, ["run", str(path), "--validate-only"])
    assert checked.exit_code == 0
    executed = runner.invoke(app, ["run", str(path)])
    assert executed.exit_code == 2
    assert "no job was submitted" in executed.output


def test_doctor_does_not_claim_compute_readiness():
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["compute"]["enabled"] is False
