"""Tests for --enable/disable-comfy-kitchen-backends and the auto-disable
of triton on Ampere-and-older NVIDIA GPUs.

The ``comfy_kitchen.registry`` is process-global mutable state, so any test
that asserts on the *result* of disabling/enabling a backend (rather than just
on Configuration parsing) must run in its own subprocess. We do that with the
``ProcessPoolExecutor`` pattern used elsewhere in this suite.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import pytest

from comfy.cli_args_types import COMFY_KITCHEN_BACKENDS


def _is_disabled_in_subprocess(disable_list: list[str], target: str) -> bool:
    """Worker run in a fresh interpreter: set args, import quant_ops, return state."""
    from comfy.cli_args import args
    args.disable_comfy_kitchen_backends = list(disable_list)
    import comfy.quant_ops  # noqa: F401  (import side effects)
    import comfy_kitchen as ck
    return bool(ck.list_backends().get(target, {}).get("disabled"))


def _is_triton_disabled_with_capability(major: int, minor: int) -> bool:
    """Worker: pretend the local CUDA device has compute capability (major, minor)."""
    from unittest import mock
    import torch

    fake_props = type("Props", (), {"major": major, "minor": minor})()
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "device_count", return_value=1), \
         mock.patch.object(torch.cuda, "get_device_properties", return_value=fake_props):
        import comfy.quant_ops  # noqa: F401
    import comfy_kitchen as ck
    return bool(ck.list_backends().get("triton", {}).get("disabled"))


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_disable_comfy_kitchen_backend_takes_effect(backend):
    """A backend listed under --disable-comfy-kitchen-backends becomes unavailable.

    Runs in its own process so the registry mutation can't leak into other tests.
    """
    with ProcessPoolExecutor(max_workers=1) as pool:
        is_disabled = pool.submit(_is_disabled_in_subprocess, [backend], backend).result(timeout=60)
    assert is_disabled, f"backend '{backend}' should be disabled after --disable-comfy-kitchen-backends={backend}"


def test_known_backends_constant():
    """COMFY_KITCHEN_BACKENDS exposes the names accepted by the CLI flags."""
    assert "triton" in COMFY_KITCHEN_BACKENDS
    assert "eager" in COMFY_KITCHEN_BACKENDS
    assert "cuda" in COMFY_KITCHEN_BACKENDS


@pytest.mark.parametrize("major,minor,expect_disabled", [
    (8, 0, True),   # A100 — Ampere
    (8, 6, True),   # RTX A5000 / 3090 — Ampere
    (8, 9, False),  # RTX 4090 — Ada (fp8e4nv supported)
    (9, 0, False),  # H100 — Hopper
])
def test_quant_ops_auto_disables_triton_on_pre_ada(major, minor, expect_disabled):
    """quant_ops checks torch.cuda.get_device_properties at import and disables
    triton on sm < 8.9 because comfy_kitchen's fp8e4nv kernel won't compile.
    """
    with ProcessPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_is_triton_disabled_with_capability, major, minor).result(timeout=60)
    assert result is expect_disabled, \
        f"sm {major}.{minor}: expected triton disabled={expect_disabled}, got {result}"


def test_capability_section_runs_without_error():
    """The matrix renderer used by `comfyui env check` must not crash on any host."""
    pytest.importorskip("comfy_kitchen")
    from io import StringIO
    from rich.console import Console
    from comfy.cmd.integrity_check import _section_comfy_kitchen_capabilities

    buf = StringIO()
    _section_comfy_kitchen_capabilities(Console(file=buf, force_terminal=False, width=200))
    output = buf.getvalue()
    assert output, "capabilities section produced no output"
    assert any(b in output for b in COMFY_KITCHEN_BACKENDS), \
        f"expected a backend column header, got: {output[:400]}"
