"""Tests for the Typer CLI app (comfy.cmd.cli)."""
import re

from typer.testing import CliRunner
from comfy.cmd.cli import app, _register_sub_apps

_register_sub_apps()

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "comfyui" in out.lower() or "ComfyUI" in out


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--listen" in out
    assert "--port" in out
    assert "--daemon" in out or "-d" in out


def test_worker_help():
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "distributed-queue" in _plain(result.output)


def test_post_workflow_help():
    result = runner.invoke(app, ["post-workflow", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--prompt" in out
    assert "--seed" in out


def test_create_directories_help():
    result = runner.invoke(app, ["create-directories", "--help"])
    assert result.exit_code == 0
    assert "--base-directory" in _plain(result.output)


def test_list_workflow_templates_help():
    result = runner.invoke(app, ["list-workflow-templates", "--help"])
    assert result.exit_code == 0
    assert "--format" in _plain(result.output)


def test_stop_help():
    result = runner.invoke(app, ["stop", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--server" in out or "--pid-file" in out


def test_logs_help():
    result = runner.invoke(app, ["logs", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--follow" in out or "-f" in out


def test_models_help():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "ls" in out or "available" in out or "download" in out or "paths" in out


def test_workflows_help():
    result = runner.invoke(app, ["workflows", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "run" in out or "submit" in out or "convert" in out or "show" in out


def test_nodes_help():
    result = runner.invoke(app, ["nodes", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "ls" in out or "packages" in out or "info" in out


def test_jobs_help():
    result = runner.invoke(app, ["jobs", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "cancel" in out


def test_env_help():
    result = runner.invoke(app, ["env", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "info" in out or "check" in out or "packages" in out or "paths" in out


def test_workflows_run_help():
    result = runner.invoke(app, ["workflows", "run", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--prompt" in out
    assert "--seed" in out
    assert "--cfg" in out
    assert "--sampler" in out
    assert "--width" in out
    assert "--set" in out


def test_workflows_convert_help():
    result = runner.invoke(app, ["workflows", "convert", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--output" in out or "-o" in out


def test_serve_has_new_override_opts():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--cfg" in out
    assert "--sampler" in out
    assert "--scheduler" in out
    assert "--width" in out
    assert "--height" in out
    assert "--batch-size" in out
    assert "--checkpoint" in out
    assert "--set" in out
