"""On-demand local facade wheel build for packages not on the pip facade index.

When ``nodes.appmana.com/simple/<pkg>/`` returns 404 — typical for packages
the upstream snapshot doesn't list yet — build the same kind of facade
wheel in-process and ``pip install`` the local file. The wheel still
renames the package to ``_appmana_facade_<name>`` and provides the stub
shim, so ComfyUI's node loader picks it up exactly the same way as a
remotely-built wheel.

Usage::

    wheel_path = build_local_facade_wheel("comfyui-umeairt-toolkit")
    # Then: subprocess.run(["uv", "pip", "install", wheel_path])
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from typing import Optional

from .registry import (
    FacadeProject,
    FacadeRegistry,
    FacadeVersion,
    canonicalize_project_name,
)
from .repo_lookup import resolve_package_repo_url

logger = logging.getLogger(__name__)


_FACADE_BUILD_CACHE_DIR = os.path.expanduser("~/.cache/comfyui/facade_build")


def build_local_facade_wheel(canonical_name: str) -> Optional[str]:
    """Build a facade wheel for *canonical_name* locally, return the wheel path.

    Returns None if no repo URL can be resolved or the build fails.
    """
    return asyncio.run(_build_local_facade_wheel_async(canonical_name))


async def _build_local_facade_wheel_async(canonical_name: str) -> Optional[str]:
    import aiohttp
    from .builder import FacadeWheelBuilder

    canonical = canonicalize_project_name(canonical_name)
    repo_url = resolve_package_repo_url(canonical)
    if not repo_url:
        logger.debug("local facade: no repo URL for %s", canonical)
        return None

    archive_url = FacadeRegistry._github_archive_url(repo_url)
    if archive_url == repo_url:
        # Unknown host (not github / gitlab); facade can't extract.
        logger.debug("local facade: %s host not supported (%s)", canonical, repo_url)
        return None

    project = _synthesize_project(canonical, repo_url)
    version = FacadeVersion(
        version="0.0.1",
        download_url=archive_url,
        dependencies=(),
        deprecated=False,
    )

    async with aiohttp.ClientSession() as session:
        builder = FacadeWheelBuilder(
            session,
            _StaticRegistry(project, version),
            cache_prefix=_FACADE_BUILD_CACHE_DIR,
        )
        try:
            cached = await builder.build_wheel(project, version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("local facade build failed for %s: %s", canonical, exc)
            return None
    if cached.local_path:
        return str(cached.local_path)
    # CachedWheel without a local_path means the bytes are only in remote
    # storage; materialize to a temp file.
    body = await _read_remote_cached(builder, cached)
    if not body:
        return None
    fd, path = tempfile.mkstemp(prefix=f"facade_{canonical}_", suffix=".whl",
                                dir=_FACADE_BUILD_CACHE_DIR)
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(body)
    return path


async def _read_remote_cached(builder, cached) -> bytes:
    try:
        return await builder.read_cached_wheel(cached)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read remote-cached facade wheel: %s", exc)
        return b""


_REPO_NAME_RE = re.compile(r"github\.com/[^/]+/([^/?]+?)(?:\.git)?(?:/|$)|gitlab\.com/[^/]+/([^/?]+?)(?:\.git)?(?:/|$)")


def _synthesize_project(canonical: str, repo_url: str) -> FacadeProject:
    m = _REPO_NAME_RE.search(repo_url)
    repo_name = (m.group(1) or m.group(2)) if m else canonical
    return FacadeProject(
        canonical_name=canonical,
        display_name=repo_name,
        node_id=canonical,
        repo_url=repo_url,
        repo_name=repo_name,
        description="",
        aliases=(canonical,),
        extra_requirements=(),
        skip_requirements=frozenset(),
        depends_on=(),
        latest_version="0.0.1",
    )


class _StaticRegistry:
    """Tiny in-memory FacadeRegistry-shaped object for one project/version."""

    def __init__(self, project: FacadeProject, version: FacadeVersion):
        self._project = project
        self._version = version

    async def get_project(self, name):
        if isinstance(name, FacadeProject):
            return name
        if canonicalize_project_name(str(name)) == self._project.canonical_name:
            return self._project
        return None

    async def get_version(self, project, version):
        if isinstance(version, FacadeVersion):
            return version
        if str(version) == self._version.version:
            return self._version
        return None

    async def list_versions(self, project):
        return [self._version]

    async def dependency_project_name(self, dependency_id):
        return canonicalize_project_name(str(dependency_id))
