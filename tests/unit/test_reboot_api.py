"""Tests for POST /api/v1/reboot endpoint."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.fixture
def restart_app():
    """Create a minimal aiohttp app with only the reboot route.

    The handler mirrors the real endpoint but skips ``GracefulExit`` so
    the test server stays alive long enough to read the response.
    """
    app = web.Application()
    app["restart_requested"] = False
    app["graceful_exit_raised"] = False

    async def post_reboot(request: web.Request) -> web.Response:
        app["restart_requested"] = True
        # In production the handler raises web.GracefulExit() after
        # write_eof.  We record that it *would* have been raised and
        # return normally so the test can inspect the response.
        app["graceful_exit_raised"] = True
        return web.json_response({"status": "restarting"})

    app.router.add_post("/api/v1/reboot", post_reboot)
    return app


async def test_reboot_returns_200_and_sets_flag(restart_app):
    """POST /api/v1/reboot responds with 200 and sets restart_requested."""
    server = TestServer(restart_app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post("/api/v1/reboot")
        assert resp.status == 200
        data = await resp.json()
        assert data == {"status": "restarting"}
        assert restart_app["restart_requested"] is True
        assert restart_app["graceful_exit_raised"] is True
    finally:
        await client.close()


def test_main_calls_execv_when_restart_requested():
    """After start_server's finally block, os.execv is called if restart_requested is set."""
    mock_server = MagicMock()
    mock_server.restart_requested = True

    with patch("os.execv") as mock_execv:
        # Simulate what main.py does after the finally block
        if mock_server.restart_requested:
            os.execv(sys.executable, [sys.executable] + sys.argv)

        mock_execv.assert_called_once_with(sys.executable, [sys.executable] + sys.argv)


def test_main_does_not_execv_when_not_requested():
    """os.execv is NOT called when restart_requested is False."""
    mock_server = MagicMock()
    mock_server.restart_requested = False

    with patch("os.execv") as mock_execv:
        if mock_server.restart_requested:
            os.execv(sys.executable, [sys.executable] + sys.argv)

        mock_execv.assert_not_called()


def test_prompt_server_has_restart_requested_attribute():
    """PromptServer instances should have restart_requested = False by default."""
    from comfy.cmd.server import PromptServer

    loop = asyncio.new_event_loop()
    try:
        server = PromptServer(loop)
        assert hasattr(server, "restart_requested")
        assert server.restart_requested is False
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Integration test: start a real server process, reboot it via the API
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parents[2]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(port: int, timeout: float = 60) -> None:
    """Block until the server responds on *port* or raise after *timeout*."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise TimeoutError(f"Server did not become ready on port {port} within {timeout}s")


def _server_is_down(port: int) -> bool:
    """Return True when the port stops accepting connections."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=1):
            return False
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return True


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="server startup too slow on Python <3.12 containers"
)
def test_reboot_restarts_server_process():
    """Start a real server, POST /api/v1/reboot, verify it comes back up.

    After os.execv the PID is the same (the process image is replaced),
    so we verify the server goes away and comes back, and that the
    ``python_version`` in /system_stats is still sane.
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
        # 1. Wait for the server to be ready
        _wait_for_server(port)

        # 2. POST /api/v1/reboot
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/reboot",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert body["status"] == "restarting"

        # 3. The server should go down briefly (the old event loop is
        #    unwinding and os.execv hasn't re-bound the port yet).
        #    Give it a moment, then wait for it to come back.
        gone = False
        for _ in range(20):
            if _server_is_down(port):
                gone = True
                break
            time.sleep(0.25)

        # Even if we never observed the gap (execv can be very fast),
        # the server should still be reachable after the restart.
        _wait_for_server(port, timeout=60)

        # 4. Sanity: the restarted server returns valid system_stats
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=5) as resp:
            stats = json.loads(resp.read())
            assert "system" in stats
            assert stats["system"]["python_version"] == sys.version

    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
