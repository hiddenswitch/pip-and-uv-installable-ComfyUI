"""Custom-node package → repo URL resolution.

Network-touching: comfy.org's API and comfyui-manager's extension-node-map.
Skipped if either is unreachable so the suite stays green offline.
"""
from __future__ import annotations

import pytest

from comfy.custom_node_facade import repo_lookup


def setup_function(_):
    repo_lookup.clear_cache()


def _network_available() -> bool:
    import socket
    try:
        socket.create_connection(("api.comfy.org", 443), timeout=3).close()
        return True
    except OSError:
        return False


def test_canonicalize_project_name():
    assert repo_lookup.canonicalize_project_name("ComfyUI-UmeAiRT-Toolkit") == "comfyui-umeairt-toolkit"
    assert repo_lookup.canonicalize_project_name("ComfyUI_swwan") == "comfyui-swwan"
    assert repo_lookup.canonicalize_project_name("DaxNodes") == "daxnodes"


def test_normalize_repo_url_strips_subpath():
    assert (
        repo_lookup._normalize_repo_url("https://github.com/foo/bar/blob/main/x.py")
        == "https://github.com/foo/bar"
    )


def test_normalize_repo_url_keeps_gitlab():
    url = "https://gitlab.com/Owner/Repo.git"
    out = repo_lookup._normalize_repo_url(url)
    assert out == "https://gitlab.com/Owner/Repo"


def test_normalize_repo_url_rejects_gist():
    url = "https://gist.githubusercontent.com/user/abc123/raw/file.py"
    assert repo_lookup._normalize_repo_url(url) is None


def test_hardcoded_swwan_resolves():
    repo = repo_lookup.resolve_package_repo_url("comfyui-swwan")
    assert repo is not None
    assert "swwan" in repo.lower()


def test_unknown_package_returns_none(monkeypatch):
    # Force the manager map to be empty + comfy.org to fail-fast.
    monkeypatch.setattr(repo_lookup, "_resolve_via_comfy_org", lambda c: None)
    monkeypatch.setattr(repo_lookup, "_resolve_via_manager_map", lambda c: None)
    assert repo_lookup.resolve_package_repo_url("definitely-does-not-exist-zzz") is None


def test_cache_is_used(monkeypatch):
    calls = {"comfy_org": 0}

    def stub_comfy_org(canonical):
        calls["comfy_org"] += 1
        return "https://github.com/foo/bar" if canonical == "foo-bar" else None

    monkeypatch.setattr(repo_lookup, "_resolve_via_comfy_org", stub_comfy_org)
    monkeypatch.setattr(repo_lookup, "_resolve_via_manager_map", lambda c: None)
    assert repo_lookup.resolve_package_repo_url("foo-bar") == "https://github.com/foo/bar"
    assert repo_lookup.resolve_package_repo_url("foo-bar") == "https://github.com/foo/bar"
    assert calls["comfy_org"] == 1


@pytest.mark.skipif(not _network_available(), reason="network unavailable")
def test_resolves_umeairt_via_comfy_org():
    repo = repo_lookup.resolve_package_repo_url("comfyui-umeairt-toolkit")
    assert repo is not None
    assert "umeairt" in repo.lower()


@pytest.mark.skipif(not _network_available(), reason="network unavailable")
def test_resolves_daxnodes_via_manager_map():
    repo = repo_lookup.resolve_package_repo_url("daxnodes")
    assert repo is not None
    assert "Daxamur" in repo or "daxnodes" in repo.lower()
