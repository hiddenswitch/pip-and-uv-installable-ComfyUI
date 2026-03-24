from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_CUSTOM_SCRIPTS_REPO_URL = "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
_CUSTOM_SCRIPTS_ARCHIVE_URL = "https://github.com/pythongosssss/ComfyUI-Custom-Scripts/archive/refs/heads/main.zip"


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
    class_types = mapping[_CUSTOM_SCRIPTS_REPO_URL][0]
    return {str(item) for item in class_types}


@pytest.fixture(scope="module")
def mock_registry_base_url() -> str:
    class _MockRegistryHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/nodes/comfyui-custom-scripts/versions"):
                payload = [
                    {
                        "version": "1.2.5",
                        "downloadUrl": _CUSTOM_SCRIPTS_ARCHIVE_URL,
                        "dependencies": [],
                        "deprecated": False,
                    }
                ]
                self._write_json(payload)
                return
            if self.path.startswith("/nodes"):
                payload = {
                    "totalPages": 1,
                    "nodes": [
                        {
                            "id": "comfyui-custom-scripts",
                            "name": "ComfyUI-Custom-Scripts",
                            "repository": _CUSTOM_SCRIPTS_REPO_URL,
                            "description": "Scripts",
                            "latest_version": {"version": "1.2.5"},
                        }
                    ],
                }
                self._write_json(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            del format, args

        def _write_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockRegistryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def facade_snapshot_path(tmp_path_factory: pytest.TempPathFactory, mock_registry_base_url: str) -> Path:
    src_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path_factory.mktemp("facade_snapshot")
    snapshot_path = output_dir / "registry.sqlite.xz"
    from comfy.component_model.configuration import Configuration
    from comfy.custom_node_facade.snapshot import snapshot_facade_registry

    configuration = Configuration()
    configuration.pip_facade_registry_base_url = mock_registry_base_url
    configuration.pip_facade_snapshot_output = str(snapshot_path)
    configuration.pip_facade_snapshot_compression = "auto"
    configuration.pip_facade_snapshot_overwrite = True
    configuration.pip_facade_only_known_nodes = True

    written = asyncio.run(snapshot_facade_registry(configuration))
    assert written == snapshot_path
    return snapshot_path


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
        _wait_for_http(f"http://127.0.0.1:{port}/readyz", server)

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

from comfy.cmd.node_info import node_info
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
info = node_info(matched[0], exported.NODE_CLASS_MAPPINGS, exported.NODE_DISPLAY_NAME_MAPPINGS)
assert isinstance(info["python_module"], str) and info["python_module"]
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


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_comfyui_layerstyle(tmp_path: Path):
    """Verify that comfyui-layerstyle can be downloaded and installed.

    This catches regressions like missing system tools (zip, sha256sum) in the
    container image that cause 500 errors during on-demand wheel builds.
    """
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
        _wait_for_http(f"http://127.0.0.1:{port}/readyz", server)

        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(site_dir),
                "--no-deps",
                "--extra-index-url",
                f"http://127.0.0.1:{port}/simple/",
                "comfyui-layerstyle",
            ],
            cwd=src_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "comfyui-layerstyle" in result.stdout.lower() or "comfyui_layerstyle" in result.stdout.lower() or result.returncode == 0

        # Verify the wheel was actually installed with content
        installed = list(site_dir.glob("_appmana_facade_*"))
        assert installed, f"Expected facade package in {site_dir}, found: {list(site_dir.iterdir())}"
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_and_load_custom_node_from_snapshot(
    tmp_path: Path,
    facade_snapshot_path: Path,
):
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
            "--pip-facade-snapshot-uri",
            facade_snapshot_path.as_uri(),
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
        _wait_for_http(f"http://127.0.0.1:{port}/readyz", server)

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

from comfy.cmd.node_info import node_info
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
info = node_info(matched[0], exported.NODE_CLASS_MAPPINGS, exported.NODE_DISPLAY_NAME_MAPPINGS)
assert isinstance(info["python_module"], str) and info["python_module"]
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
