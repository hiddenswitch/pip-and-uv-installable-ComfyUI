"""MPS should support fp16 and not force manual cast to bf16."""
import torch
from unittest.mock import patch

MPS_PATCHES = {
    "comfy.model_management.is_device_mps": True,
    "comfy.model_management.mps_mode": True,
    "comfy.model_management.bfloat16_support_mps": True,
    "comfy.model_management.is_device_cpu": False,
}


def test_mps_fp16_supported():
    with patch.multiple("comfy.model_management",
                        is_device_mps=lambda *a: True,
                        mps_mode=lambda: True,
                        bfloat16_support_mps=lambda *a: True,
                        is_device_cpu=lambda *a: False):
        from comfy.model_management import should_use_fp16
        assert should_use_fp16(torch.device("cpu"), prioritize_performance=False) is True


def test_mps_no_manual_cast_fp16_weights():
    with patch.multiple("comfy.model_management",
                        is_device_mps=lambda *a: True,
                        mps_mode=lambda: True,
                        bfloat16_support_mps=lambda *a: True,
                        is_device_cpu=lambda *a: False):
        from comfy.model_management import unet_manual_cast
        result = unet_manual_cast(torch.float16, torch.device("cpu"),
                                  supported_dtypes=[torch.float16, torch.bfloat16, torch.float32])
        assert result is None, f"expected no manual cast for fp16 weights on MPS, got {result}"
