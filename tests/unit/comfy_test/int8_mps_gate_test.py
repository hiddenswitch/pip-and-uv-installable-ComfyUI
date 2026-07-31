"""supports_int8_compute on Apple silicon follows comfy_kitchen's mps backend."""
import platform

import pytest
import torch

from comfy import model_management


requires_mps = pytest.mark.skipif(
    platform.system() != "Darwin" or not torch.backends.mps.is_available(),
    reason="Apple silicon required",
)


@requires_mps
def test_supports_int8_compute_tracks_comfy_kitchen_mps_backend():
    try:
        import comfy_kitchen
    except ImportError:
        assert model_management.supports_int8_compute(torch.device("mps")) is False
        return

    backend_available = comfy_kitchen.list_backends().get("mps", {}).get("available", False)
    assert model_management.supports_int8_compute(torch.device("mps")) == backend_available
    # device=None resolves via mps_mode() on Apple silicon
    if model_management.mps_mode():
        assert model_management.supports_int8_compute() == backend_available


def test_supports_int8_compute_false_without_gpu_stack():
    # A non-mps, non-nvidia device never claims int8 compute.
    assert model_management.supports_int8_compute(torch.device("cpu")) is False
