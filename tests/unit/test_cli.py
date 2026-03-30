"""Tests for the Typer CLI app (comfy.cmd.cli)."""
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from typer.testing import CliRunner
import comfy.cmd.cli as cli_module
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


def test_run_workflow_help():
    result = runner.invoke(app, ["run-workflow", "--help"])
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


def test_serve_pip_help():
    result = runner.invoke(app, ["serve-pip", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--pip-facade-registry" in out or "registry API" in out
    assert "Read facade registry" in out or "snapshot URI" in out
    assert "--pip-facade-cache" in out or "cached" in out
    assert "--pip-facade-only-kno" in out or "Only expose nodes" in out


def test_snapshot_pip_registry_help():
    result = runner.invoke(app, ["snapshot-pip-registry", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "pip-facade-registr" in out
    assert "pip-facade-snapshot" in out


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


def test_entrypoint_defaults_to_serve_with_no_subcommand(monkeypatch):
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui"])

    cli_module.entrypoint()

    assert called == [["comfyui", "serve"]]


def test_entrypoint_defaults_to_serve_when_first_arg_is_option(monkeypatch):
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui", "--listen", "0.0.0.0"])

    cli_module.entrypoint()

    assert called == [["comfyui", "serve", "--listen", "0.0.0.0"]]


def test_entrypoint_does_not_rewrite_unknown_verbs(monkeypatch):
    called = []
    monkeypatch.setattr(cli_module, "_register_sub_apps", lambda: None)
    monkeypatch.setattr(cli_module, "app", lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["comfyui", "definitely-not-a-command"])

    cli_module.entrypoint()

    assert called == [["comfyui", "definitely-not-a-command"]]


class TestParseListenAddress:
    def test_bare_ipv4(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("0.0.0.0") == ("0.0.0.0", None)

    def test_ipv4_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("0.0.0.0:8189") == ("0.0.0.0", 8189)

    def test_bare_ipv6(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("::") == ("::", None)

    def test_ipv6_loopback(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("::1") == ("::1", None)

    def test_bracketed_ipv6_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("[::]:8189") == ("::", 8189)

    def test_bracketed_ipv6_loopback_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("[::1]:8189") == ("::1", 8189)

    def test_hostname_with_port(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("localhost:9000") == ("localhost", 9000)

    def test_hostname_bare(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("localhost") == ("localhost", None)

    def test_comma_separated_not_split(self):
        from comfy.cmd.cli import _parse_listen_address
        assert _parse_listen_address("0.0.0.0,::") == ("0.0.0.0,::", None)


_SRC_ROOT = Path(__file__).resolve().parents[2]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.slow
def test_serve_starts_and_reaches_ready():
    """Verify that ``comfyui serve`` gets past setup_pre_torch and binds its port.

    This catches import errors (e.g. missing names in cli_args) that only
    surface when the server actually starts, not just when --help is invoked.
    """
    port = _find_free_port()
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_ROOT) if not existing else f"{_SRC_ROOT}{os.pathsep}{existing}"

    process = subprocess.Popen(
        [
            sys.executable, "-m", "comfy.cmd.main",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--cpu",
            "--dont-print-server",
        ],
        cwd=_SRC_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if process.poll() is not None:
                output = process.stdout.read()
                pytest.fail(f"comfyui serve exited early with code {process.returncode}:\n{output}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as resp:
                    if resp.status == 200:
                        return  # success
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                time.sleep(0.5)
        output = process.stdout.read()
        pytest.fail(f"comfyui serve did not become ready within 60s:\n{output}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
