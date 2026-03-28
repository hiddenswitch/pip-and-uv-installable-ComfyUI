"""WebSocket-to-stdio bridge for basedpyright-langserver.

Starts a single basedpyright-langserver subprocess on the first WebSocket
connection and bridges JSON-RPC messages between all connected editors
and the language server's stdio.

Editor code is transparently wrapped in a function stub so that pyright
sees valid Python (with correct parameter names and types) instead of
bare return/yield statements.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

# The function wrapper prepended to every editor document.
# This gives pyright the correct context: value0..valueN are parameters,
# logger and print are available, and return/yield are valid.
_WRAPPER_PREFIX = """\
# pyright: reportUnusedVariable=false, reportUnusedParameter=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnusedFunction=false, reportUnusedImport=false
def _eval_func(value0=None, value1=None, value2=None, value3=None, value4=None):
"""
_WRAPPER_LINES = _WRAPPER_PREFIX.count("\n")
_INDENT = "    "


def _wrap_content(text: str) -> str:
    """Wrap editor content in the function stub."""
    indented = "\n".join(_INDENT + line if line.strip() else line for line in text.split("\n"))
    return _WRAPPER_PREFIX + indented


def _adjust_position_to_editor(pos: dict) -> dict:
    """Shift a position from wrapped coordinates back to editor coordinates."""
    line = pos.get("line", 0) - _WRAPPER_LINES
    return {**pos, "line": max(line, 0)}


def _adjust_position_to_wrapped(pos: dict) -> dict:
    """Shift a position from editor coordinates to wrapped coordinates."""
    return {**pos, "line": pos.get("line", 0) + _WRAPPER_LINES}


class _JsonRpc:
    """LSP JSON-RPC framing over asyncio streams."""

    def __init__(self, reader: asyncio.StreamReader):
        self._reader = reader

    async def read_message(self) -> dict | None:
        content_length = None
        while True:
            line = await self._reader.readline()
            if not line:
                return None
            decoded = line.decode("ascii", errors="replace").strip()
            if not decoded:
                break
            if decoded.lower().startswith("content-length:"):
                content_length = int(decoded.split(":", 1)[1].strip())
        if content_length is None:
            return None
        body = await self._reader.readexactly(content_length)
        return json.loads(body)

    @staticmethod
    def encode(obj: dict) -> bytes:
        body = json.dumps(obj).encode("utf-8")
        return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


class LspManager:
    """Manages one basedpyright-langserver subprocess shared by all editors."""

    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None
        self._rpc: _JsonRpc | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._lock = asyncio.Lock()
        self._reader_task: asyncio.Task | None = None
        self._workspace: str | None = None

    async def _ensure_running(self):
        async with self._lock:
            if self._process is not None and self._process.returncode is None:
                return

            langserver = self._find_langserver()
            if langserver is None:
                raise FileNotFoundError(
                    "basedpyright-langserver not found. Install with: uv pip install basedpyright"
                )

            self._workspace = tempfile.mkdtemp(prefix="comfyui-lsp-")
            self._write_config(self._workspace)

            self._process = await asyncio.create_subprocess_exec(
                langserver, "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workspace,
            )
            self._rpc = _JsonRpc(self._process.stdout)
            self._reader_task = asyncio.create_task(self._read_loop())
            asyncio.create_task(self._stderr_loop())
            logger.info("Started basedpyright-langserver (pid %d)", self._process.pid)

    async def _read_loop(self):
        try:
            while True:
                msg = await self._rpc.read_message()
                if msg is None:
                    break
                # Adjust positions in responses/notifications back to editor coordinates
                msg = self._adjust_outgoing(msg)
                text = json.dumps(msg)
                dead = set()
                for ws in self._clients:
                    try:
                        await ws.send_str(text)
                    except Exception:
                        dead.add(ws)
                self._clients -= dead
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("LSP read loop ended", exc_info=True)
        finally:
            self._process = None

    async def _stderr_loop(self):
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug("pyright: %s", line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        try:
            await self._ensure_running()
        except FileNotFoundError as e:
            await ws.close(message=str(e).encode())
            return ws

        self._clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    data = self._adjust_incoming(data)
                    if self._process and self._process.stdin:
                        self._process.stdin.write(_JsonRpc.encode(data))
                        await self._process.stdin.drain()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    @staticmethod
    def _adjust_incoming(msg: dict) -> dict:
        """Wrap document content and adjust positions in requests TO pyright."""
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method in ("textDocument/didOpen",):
            td = params.get("textDocument", {})
            text = td.get("text", "")
            td["text"] = _wrap_content(text)

        elif method in ("textDocument/didChange",):
            # Full sync: the first contentChange has the full text
            changes = params.get("contentChanges", [])
            for change in changes:
                if "text" in change and "range" not in change:
                    change["text"] = _wrap_content(change["text"])

        elif method in ("textDocument/completion", "textDocument/hover",
                        "textDocument/signatureHelp", "textDocument/definition",
                        "textDocument/references", "textDocument/documentHighlight"):
            pos = params.get("position")
            if pos:
                params["position"] = _adjust_position_to_wrapped(pos)

        return msg

    @staticmethod
    def _adjust_outgoing(msg: dict) -> dict:
        """Adjust positions in responses/notifications FROM pyright back to editor coords."""
        method = msg.get("method", "")
        params = msg.get("params", {})
        result = msg.get("result")

        # Diagnostics notification
        if method == "textDocument/publishDiagnostics":
            diagnostics = params.get("diagnostics", [])
            adjusted = []
            for diag in diagnostics:
                if "range" not in diag:
                    continue
                start_line = diag["range"]["start"].get("line", 0)
                # Drop diagnostics that originate in the wrapper prefix
                if start_line < _WRAPPER_LINES:
                    continue
                diag["range"]["start"] = _adjust_position_to_editor(diag["range"]["start"])
                diag["range"]["end"] = _adjust_position_to_editor(diag["range"]["end"])
                adjusted.append(diag)
            params["diagnostics"] = adjusted

        # Completion response
        if isinstance(result, dict) and "items" in result:
            for item in result.get("items", []):
                te = item.get("textEdit", {})
                if "range" in te:
                    te["range"]["start"] = _adjust_position_to_editor(te["range"]["start"])
                    te["range"]["end"] = _adjust_position_to_editor(te["range"]["end"])
                for ae in item.get("additionalTextEdits", []):
                    if "range" in ae:
                        ae["range"]["start"] = _adjust_position_to_editor(ae["range"]["start"])
                        ae["range"]["end"] = _adjust_position_to_editor(ae["range"]["end"])

        # Hover response
        if isinstance(result, dict) and "range" in result:
            result["range"]["start"] = _adjust_position_to_editor(result["range"]["start"])
            result["range"]["end"] = _adjust_position_to_editor(result["range"]["end"])

        return msg

    @staticmethod
    def _find_langserver() -> str | None:
        venv_bin = str(Path(sys.executable).parent)
        candidate = os.path.join(venv_bin, "basedpyright-langserver")
        if os.path.isfile(candidate):
            return candidate
        return shutil.which("basedpyright-langserver")

    @staticmethod
    def _write_config(workspace: str):
        config = {
            "pythonPath": sys.executable,
            "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
            "typeCheckingMode": "off",
        }
        with open(os.path.join(workspace, "pyrightconfig.json"), "w") as f:
            json.dump(config, f)
