"""Local facade wheel build for packages without pre-built wheels on the
pip facade index.

Network-touching: the build fetches a source archive over HTTPS and
shells out to compile a wheel. Heavy tests are skipped offline.
"""
from __future__ import annotations

import socket

import pytest

from comfy.custom_node_facade import local_build
from comfy.custom_node_facade.local_build import _synthesize_project, _StaticRegistry
from comfy.custom_node_facade.registry import FacadeProject, FacadeRegistry, FacadeVersion


def _network_available() -> bool:
    try:
        socket.create_connection(("api.github.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


def test_synthesize_project_github():
    p = _synthesize_project("daxnodes", "https://github.com/Daxamur/DaxNodes")
    assert isinstance(p, FacadeProject)
    assert p.canonical_name == "daxnodes"
    assert p.repo_name == "DaxNodes"
    assert p.repo_url == "https://github.com/Daxamur/DaxNodes"


def test_synthesize_project_gitlab():
    p = _synthesize_project(
        "comfyui-umeairt-toolkit",
        "https://gitlab.com/UmeAiRT-Studio/ComfyUI-UmeAiRT-Toolkit",
    )
    assert p.canonical_name == "comfyui-umeairt-toolkit"
    assert p.repo_name == "ComfyUI-UmeAiRT-Toolkit"


def test_static_registry_returns_only_known_project():
    project = _synthesize_project("foo", "https://github.com/me/foo")
    version = FacadeVersion(version="0.0.1", download_url="https://example.com/x.zip",
                            dependencies=(), deprecated=False)
    reg = _StaticRegistry(project, version)
    import asyncio
    got = asyncio.run(reg.get_project("foo"))
    assert got is project
    miss = asyncio.run(reg.get_project("other"))
    assert miss is None


def test_github_archive_url_passthrough():
    url = FacadeRegistry._github_archive_url("https://github.com/foo/bar")
    assert url.startswith("https://api.github.com/repos/foo/bar/zipball")


def test_gitlab_archive_url_emitted():
    url = FacadeRegistry._github_archive_url("https://gitlab.com/owner/repo")
    assert "gitlab.com/api/v4/projects" in url
    assert "owner%2Frepo" in url
    assert "/repository/archive.zip" in url


def test_unknown_host_returns_original_url():
    url = FacadeRegistry._github_archive_url("https://example.com/foo/bar")
    assert url == "https://example.com/foo/bar"


def test_strips_dot_git_suffix():
    url = FacadeRegistry._github_archive_url("https://github.com/foo/bar.git")
    assert url == "https://api.github.com/repos/foo/bar/zipball"
    glab = FacadeRegistry._github_archive_url("https://gitlab.com/owner/repo.git")
    assert "owner%2Frepo" in glab


def test_build_returns_none_if_no_repo_url(monkeypatch):
    monkeypatch.setattr(local_build, "resolve_package_repo_url", lambda c: None)
    assert local_build.build_local_facade_wheel("definitely-does-not-exist") is None


def test_build_returns_none_for_unknown_host(monkeypatch):
    monkeypatch.setattr(
        local_build, "resolve_package_repo_url",
        lambda c: "https://bitbucket.org/owner/repo",
    )
    assert local_build.build_local_facade_wheel("unsupported-host-pkg") is None
