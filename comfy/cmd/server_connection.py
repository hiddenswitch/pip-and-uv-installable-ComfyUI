"""Shared --server option and async HTTP helpers for client commands (no torch)."""
from __future__ import annotations

import os
from typing import Optional, Any

import aiohttp
from aiohttp import ClientTimeout


def server_url(server: Optional[str] = None) -> str:
    if server:
        return server.rstrip("/")
    return os.environ.get("COMFYUI_SERVER", "http://localhost:8188").rstrip("/")


async def fetch_json(
    server: Optional[str],
    path: str,
    method: str = "GET",
    params: Optional[dict[str, str]] = None,
) -> Any:
    url = f"{server_url(server)}{path}"
    timeout = ClientTimeout(total=30.0, connect=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, url, params=params) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            return await resp.json()


async def post_json(
    server: Optional[str],
    path: str,
    body: Optional[dict] = None,
) -> Any:
    url = f"{server_url(server)}{path}"
    timeout = ClientTimeout(total=60.0, connect=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=body or {}) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            if resp.content_type == "application/json":
                return await resp.json()
            return {"status": "ok"}


async def fetch_text(
    server: Optional[str],
    path: str,
    params: Optional[dict[str, str]] = None,
) -> str:
    url = f"{server_url(server)}{path}"
    timeout = ClientTimeout(total=30.0, connect=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request("GET", url, params=params) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            return await resp.text()
