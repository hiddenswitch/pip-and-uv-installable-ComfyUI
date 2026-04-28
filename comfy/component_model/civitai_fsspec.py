"""fsspec backend for ``civitai://`` and ``civitai-red://`` URIs.

URI shapes (read-only):

* ``civitai://m/<model_id>`` — primary file of the latest model version
* ``civitai://v/<model_version_id>`` — primary file of a specific version
* ``civitai://download/<model_version_id>`` — equivalent to ``v/...``; matches
  Civitai's REST download path for clarity
* ``civitai-red://...`` — same shapes but resolves against ``civitai.red``

Auth: when ``CIVITAI_API_TOKEN`` (or ``CIVITAI_API_KEY``) is set, requests
include ``Authorization: Bearer <token>`` so model file downloads (which
require auth) succeed.
"""
from __future__ import annotations

import os
import urllib.parse
from typing import Optional

from fsspec import AbstractFileSystem, register_implementation


def _api_hostname(scheme: str) -> str:
    return "civitai.red" if scheme == "civitai-red" else "civitai.com"


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_API_KEY")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _resolve_to_download_url(scheme: str, path: str) -> str:
    """Return an https://<hostname>/api/download/models/<version_id> URL."""
    import requests
    hostname = _api_hostname(scheme)
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        raise ValueError(f"empty {scheme}:// path")

    if parts[0] in ("v", "version", "download") and len(parts) >= 2:
        version_id = parts[1]
        return f"https://{hostname}/api/download/models/{version_id}"

    if parts[0] in ("m", "model", "models") and len(parts) >= 2:
        model_id = parts[1]
        r = requests.get(
            f"https://{hostname}/api/v1/models/{model_id}",
            headers=_auth_headers(), timeout=30,
        )
        r.raise_for_status()
        versions = r.json().get("modelVersions") or []
        if not versions:
            raise FileNotFoundError(f"{scheme}://m/{model_id}: no model versions")
        version_id = versions[0].get("id")
        return f"https://{hostname}/api/download/models/{version_id}"

    raise ValueError(f"unrecognized {scheme}:// path: {path!r}")


class CivitaiFileSystem(AbstractFileSystem):
    """Read-only fsspec backend that resolves civitai://{m,v}/<id> to HTTPS downloads."""

    protocol = ("civitai", "civitai-red")
    root_marker = "/"

    def __init__(self, scheme: str = "civitai", **kwargs):
        super().__init__(**kwargs)
        self._scheme = scheme

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        if mode != "rb":
            raise NotImplementedError("civitai filesystem is read-only")
        from fsspec.implementations.http import HTTPFileSystem
        url = _resolve_to_download_url(self._scheme, path)
        http = HTTPFileSystem(headers=_auth_headers())
        return http._open(url, mode=mode, block_size=block_size, **kwargs)

    def _info(self, path, **kwargs):
        return {"name": path, "size": None, "type": "file"}


def register() -> None:
    register_implementation("civitai", CivitaiFileSystem, clobber=True)
    register_implementation(
        "civitai-red",
        lambda **kw: CivitaiFileSystem(scheme="civitai-red", **kw),  # type: ignore[arg-type]
        clobber=True,
    )


# Side-effect: register on import.
register()
