"""Tests for the Typer CLI app (comfy.cmd.cli)."""
import re

from typer.testing import CliRunner
from comfy.cmd.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(output: str) -> str:
    return _ANSI_RE.sub("", output)


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    clean = _clean(result.output)
    assert "comfyui" in clean.lower() or "ComfyUI" in clean


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.output)
    assert "--listen" in clean
    assert "--port" in clean


def test_worker_help():
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.output)
    assert "distributed-queue" in clean


def test_post_workflow_help():
    result = runner.invoke(app, ["post-workflow", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.output)
    assert "--prompt" in clean
    assert "--seed" in clean


def test_create_directories_help():
    result = runner.invoke(app, ["create-directories", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.output)
    assert "--base-directory" in clean


def test_list_workflow_templates_help():
    result = runner.invoke(app, ["list-workflow-templates", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.output)
    assert "--format" in clean
