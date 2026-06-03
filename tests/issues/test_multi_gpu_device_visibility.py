import pytest
import torch

from comfy import model_management


def _assert_basic_tensor_works(device: torch.device):
    x = torch.ones((2, 2), device=device)
    y = x + 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    assert y.device.type == device.type
    assert y.device.index == device.index
    assert y.detach().cpu().tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_visible_multi_gpu_devices_resolve_and_execute():
    devices = model_management.get_all_torch_devices()
    if len(devices) < 2:
        pytest.skip(f"requires at least two visible torch GPU devices, got {devices}")

    options = model_management.get_gpu_device_options()
    assert options[:2] == ["default", "cpu"]

    for idx, device in enumerate(devices):
        assert f"gpu:{idx}" in options
        assert model_management.resolve_gpu_device_option(f"gpu:{idx}") == device
        _assert_basic_tensor_works(device)

    assert model_management.resolve_gpu_device_option(f"gpu:{len(devices)}") is None
