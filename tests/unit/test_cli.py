"""Tests for the Typer CLI app (comfy.cmd.cli)."""
import re

from typer.testing import CliRunner
from comfy.cmd.cli import app

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
