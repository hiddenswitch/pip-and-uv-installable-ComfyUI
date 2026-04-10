"""Tests for _ComfyExtrasRedirectFinder (comfy_extras.<name> -> comfy_extras.nodes.<name>).

This is the compatibility shim that allows vanilla custom nodes (like KJNodes)
to import ``from comfy_extras.nodes_mask import composite`` even though the
module lives at ``comfy_extras.nodes.nodes_mask`` in this fork.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from comfy_compatibility.vanilla import _ComfyExtrasRedirectFinder


@pytest.fixture(autouse=True)
def _install_redirect_finder():
    """Ensure the redirect finder is on sys.meta_path for every test, then clean up."""
    finder = _ComfyExtrasRedirectFinder()
    sys.meta_path.append(finder)
    yield
    sys.meta_path.remove(finder)


def _remove_module(name: str):
    """Remove a module from sys.modules so it can be re-imported through the finder."""
    sys.modules.pop(name, None)


class TestComfyExtrasRedirect:
    def test_nodes_mask_redirect(self):
        """KJNodes does ``from comfy_extras.nodes_mask import composite``."""
        _remove_module("comfy_extras.nodes_mask")
        from comfy_extras.nodes_mask import composite

        assert callable(composite)

    def test_nodes_mask_is_same_module(self):
        """The redirected module should be the same object as the canonical one."""
        _remove_module("comfy_extras.nodes_mask")
        import comfy_extras.nodes_mask as redirected
        import comfy_extras.nodes.nodes_mask as canonical

        assert redirected is canonical

    def test_nodes_custom_sampler_redirect(self):
        """Another commonly imported module."""
        _remove_module("comfy_extras.nodes_custom_sampler")
        mod = importlib.import_module("comfy_extras.nodes_custom_sampler")
        canonical = importlib.import_module("comfy_extras.nodes.nodes_custom_sampler")
        assert mod is canonical

    def test_nonexistent_module_returns_none(self):
        """A module that doesn't exist under comfy_extras.nodes should not be found."""
        finder = _ComfyExtrasRedirectFinder()
        spec = finder.find_spec("comfy_extras.this_does_not_exist_at_all")
        assert spec is None

    def test_nodes_subpackage_not_redirected(self):
        """``comfy_extras.nodes`` itself should not be intercepted."""
        finder = _ComfyExtrasRedirectFinder()
        spec = finder.find_spec("comfy_extras.nodes")
        assert spec is None

    def test_private_modules_not_redirected(self):
        """Modules starting with _ should not be intercepted."""
        finder = _ComfyExtrasRedirectFinder()
        spec = finder.find_spec("comfy_extras._private")
        assert spec is None

    def test_deeper_paths_not_redirected(self):
        """Only two-level names (comfy_extras.X) should be intercepted."""
        finder = _ComfyExtrasRedirectFinder()
        spec = finder.find_spec("comfy_extras.nodes.nodes_mask")
        assert spec is None

    def test_unrelated_package_not_intercepted(self):
        finder = _ComfyExtrasRedirectFinder()
        spec = finder.find_spec("os.path")
        assert spec is None
