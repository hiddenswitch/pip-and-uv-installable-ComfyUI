"""Tests for the Typer CLI app (comfy.cmd.cli)."""
from typer.testing import CliRunner
from comfy.cmd.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "comfyui" in result.output.lower() or "ComfyUI" in result.output


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--listen" in result.output
    assert "--port" in result.output


def test_worker_help():
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "distributed-queue" in result.output


def test_post_workflow_help():
    result = runner.invoke(app, ["post-workflow", "--help"])
    assert result.exit_code == 0
    assert "--prompt" in result.output
    assert "--seed" in result.output


def test_create_directories_help():
    result = runner.invoke(app, ["create-directories", "--help"])
    assert result.exit_code == 0
    assert "--base-directory" in result.output


def test_list_workflow_templates_help():
    result = runner.invoke(app, ["list-workflow-templates", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
