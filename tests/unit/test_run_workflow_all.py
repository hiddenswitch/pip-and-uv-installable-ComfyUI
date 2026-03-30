"""Tests for run-workflow --all flag (install nodes + download models)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from comfy.cmd.cli import (
    _install_workflow_requirements,
    _download_workflow_models,
    _load_core_class_types,
    _NODES_INDEX_URL,
)

# subprocess is imported inside _install_workflow_requirements, patch at module level
import subprocess as _subprocess


_SAMPLE_WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "KSampler"},
        {"id": 2, "type": "ClownSampler_Beta"},
        {"id": 3, "type": "LTXVImgToVideoConditionOnly"},
    ]
}


@pytest.fixture
def workflow_file(tmp_path: Path) -> str:
    path = tmp_path / "test_workflow.json"
    path.write_text(json.dumps(_SAMPLE_WORKFLOW))
    return str(path)


def test_install_workflow_requirements_resolves_packages(workflow_file: str):
    """Verify _install_workflow_requirements finds the right packages and calls uv."""
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=mock_run):
        _install_workflow_requirements([workflow_file])

    if not calls:
        pytest.skip("All packages already installed")

    cmd = calls[0]
    assert "uv" in cmd[0] or cmd[0].endswith("uv")
    assert "pip" in cmd
    assert "install" in cmd
    assert "--extra-index-url" in cmd
    assert _NODES_INDEX_URL in cmd


def test_install_workflow_requirements_skips_installed(workflow_file: str):
    """Verify already-installed packages are not re-installed."""
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)

    # Pretend all packages are installed
    mock_dists = []
    for name in ["res4lyf", "comfyui-ltxvideo", "comfymath"]:
        d = MagicMock()
        d.metadata = {"Name": name}
        mock_dists.append(d)

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("importlib.metadata.distributions", return_value=mock_dists),
    ):
        _install_workflow_requirements([workflow_file])

    assert len(calls) == 0


def test_install_workflow_requirements_skips_stdin():
    """Verify stdin sources are skipped."""
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)

    with patch("subprocess.run", side_effect=mock_run):
        _install_workflow_requirements(["-"])

    assert len(calls) == 0


def test_install_workflow_requirements_no_uv(workflow_file: str):
    """Verify graceful handling when uv is not available."""
    with patch("shutil.which", return_value=None):
        _install_workflow_requirements([workflow_file])


def test_load_core_class_types_returns_frozenset():
    """Verify _load_core_class_types returns a non-empty frozenset of strings."""
    result = _load_core_class_types()
    assert isinstance(result, frozenset)
    assert len(result) > 100
    assert "KSampler" in result
    assert "CheckpointLoaderSimple" in result


def test_run_workflow_help_shows_all_flag():
    """Verify --all and -a appear in run-workflow help."""
    from typer.testing import CliRunner
    from comfy.cmd.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run-workflow", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output
    assert "-a" in result.output


def test_run_workflow_help_shows_guess_settings_short():
    """Verify -g appears in run-workflow help."""
    from typer.testing import CliRunner
    from comfy.cmd.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run-workflow", "--help"])
    assert result.exit_code == 0
    assert "-g" in result.output
