"""Resolve a canonical pip-package name to its source repository URL.

Used as a fallback when a package isn't on ``nodes.appmana.com/simple/``
(the pip facade index hosts pre-built wheels for ~44 popular packages, but
the snapshot's ``class_types`` table maps to ~3956 packages — the long
tail has no wheel and must be installed from source).

Resolution order:
1. comfy.org's node registry (``api.comfy.org/nodes/<name>``) — the most
   authoritative source. Has the canonical repo URL even when no wheel
   has been built yet.
2. comfyui-manager's ``extension-node-map.json`` — the keys are GitHub URLs
   that we can canonicalize to package names and look up in reverse.

Results are cached in-process; misses are also cached so we don't repeat
slow API calls.
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_COMFY_API_BASE = "https://api.comfy.org"
_MANAGER_EXTENSION_MAP_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
)

# Canonical repo URLs for packages whose comfy.org / manager entries are
# missing or outdated. Keep this short — the runtime resolvers should cover
# almost everything.
_HARDCODED: dict[str, str] = {
    "comfyui-swwan": "https://github.com/swwan/ComfyUI_swwan",
}


def canonicalize_project_name(name: str) -> str:
    """Match ``packaging.utils.canonicalize_name`` (lowercase, ``[-_.]+`` → ``-``)."""
    return re.sub(r"[-_.]+", "-", name.lower()).strip("-")


_repo_cache: dict[str, Optional[str]] = {}
_manager_map: Optional[dict[str, str]] = None


def resolve_package_repo_url(canonical_name: str) -> Optional[str]:
    """Return the repository URL for *canonical_name*, or None if unknown.

    The URL is suitable for ``pip install git+<url>`` (i.e. an HTTPS URL
    pointing at a git repository).
    """
    canonical = canonicalize_project_name(canonical_name)
    if canonical in _repo_cache:
        return _repo_cache[canonical]
    if canonical in _HARDCODED:
        _repo_cache[canonical] = _HARDCODED[canonical]
        return _HARDCODED[canonical]
    url = _resolve_via_comfy_org(canonical) or _resolve_via_manager_map(canonical)
    _repo_cache[canonical] = url
    return url


def _resolve_via_comfy_org(canonical: str) -> Optional[str]:
    try:
        r = requests.get(f"{_COMFY_API_BASE}/nodes/{canonical}", timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        repo = r.json().get("repository") or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("comfy.org lookup for %s failed: %s", canonical, exc)
        return None
    return _normalize_repo_url(repo)


def _resolve_via_manager_map(canonical: str) -> Optional[str]:
    global _manager_map
    if _manager_map is None:
        _manager_map = _build_manager_map()
    return _manager_map.get(canonical)


def _build_manager_map() -> dict[str, str]:
    """Build canonical_name → repo_url from comfyui-manager's extension-node-map."""
    out: dict[str, str] = {}
    try:
        r = requests.get(_MANAGER_EXTENSION_MAP_URL, timeout=30)
        r.raise_for_status()
        data: dict = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not fetch extension-node-map: %s", exc)
        return out
    for url, payload in data.items():
        if "github.com/" not in url:
            continue
        m = re.search(r"github\.com/[^/]+/([^/?]+?)(?:\.git)?(?:/|$)", url)
        if not m:
            continue
        repo = m.group(1)
        canonical = canonicalize_project_name(repo)
        # Trim path off the URL — keep only the bare repo root.
        clean = _normalize_repo_url(url)
        if clean:
            # First-seen wins; many entries reference the same repo.
            out.setdefault(canonical, clean)
    return out


def _normalize_repo_url(url: str) -> Optional[str]:
    """Trim a repo URL to its bare root and ensure it ends with ``.git``-friendly form.

    Examples:
        ``https://github.com/foo/bar/blob/main/x.py`` → ``https://github.com/foo/bar``
        ``https://gitlab.com/x/y.git`` → ``https://gitlab.com/x/y.git``
        ``https://gist.githubusercontent.com/.../`` → None (gists aren't pip-installable)
    """
    if not url:
        return None
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        return None
    host = p.hostname or ""
    if "gist.githubusercontent.com" in host or "gist.github.com" in host:
        return None
    parts = [seg for seg in p.path.split("/") if seg]
    if host in {"github.com", "gitlab.com", "codeberg.org"}:
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return f"{p.scheme}://{host}/{owner}/{repo}"
    return f"{p.scheme}://{host}{p.path}"


def clear_cache() -> None:
    """Drop in-process caches (for tests)."""
    global _manager_map
    _repo_cache.clear()
    _manager_map = None
