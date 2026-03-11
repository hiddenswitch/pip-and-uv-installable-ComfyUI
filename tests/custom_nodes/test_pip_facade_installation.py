from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, process: subprocess.Popen[str], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"serve-pip exited early with code {process.returncode}:\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(0.5)
    output = process.stdout.read() if process.stdout is not None else ""
    raise AssertionError(f"Timed out waiting for {url}:\n{output}")


def _custom_scripts_expected_nodes() -> set[str]:
    import comfyui_manager

    mapping_path = Path(comfyui_manager.__file__).resolve().parent / "extension-node-map.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    repo_url = "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    class_types = mapping[repo_url][0]
    return {str(item) for item in class_types}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_and_load_custom_node(tmp_path: Path):
    src_root = Path(__file__).resolve().parents[2]
    site_dir = tmp_path / "site"
    site_dir.mkdir()

    port = _find_free_port()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_root) if not existing_pythonpath else f"{src_root}{os.pathsep}{existing_pythonpath}"

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "comfy.cmd.main",
            "serve-pip",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--pip-facade-cache-prefix",
            str(tmp_path / "wheel-cache"),
            "--pip-facade-only-known-nodes",
            "--logging-level",
            "INFO",
        ],
        cwd=src_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_http(f"http://127.0.0.1:{port}/healthz", server)

        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(site_dir),
                "--extra-index-url",
                f"http://127.0.0.1:{port}/simple/",
                "comfyui-custom-scripts==1.2.5",
            ],
            cwd=src_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        expected_nodes = sorted(_custom_scripts_expected_nodes())
        probe_script = f"""
from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

from comfy.nodes.package import _extract_vanilla_custom_node_roots
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy_compatibility.vanilla import prepare_vanilla_environment

prepare_vanilla_environment()

entry_point = next(
    ep for ep in entry_points().select(group='comfyui.custom_nodes')
    if ep.name == 'comfyui-custom-scripts'
)
module = entry_point.load()
roots = _extract_vanilla_custom_node_roots(module)
assert roots, 'facade entry point did not expose any vanilla custom node roots'
repo_path = Path(roots[0]) / 'ComfyUI-Custom-Scripts'
assert repo_path.exists(), repo_path
exported = _vanilla_load_custom_nodes_1(str(repo_path))
keys = set(exported.NODE_CLASS_MAPPINGS.keys())
expected = set({expected_nodes!r})
matched = sorted(keys & expected)
assert matched, f'expected one of {{sorted(expected)}} in loaded keys'
print('MATCHED', matched[0])
"""

        probe_env = env.copy()
        probe_env["PYTHONPATH"] = (
            f"{site_dir}{os.pathsep}{src_root}"
            if not existing_pythonpath
            else f"{site_dir}{os.pathsep}{src_root}{os.pathsep}{existing_pythonpath}"
        )
        probe = subprocess.run(
            [sys.executable, "-c", probe_script],
            cwd=src_root,
            env=probe_env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "MATCHED" in probe.stdout
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
