"""Regression tests for dynamic VRAM (comfy-aimdo) weight loading.

The dynamic patcher (ModelPatcherDynamic) loads weights with
``load_state_dict(assign=True)`` so the vbar streamer can take ownership of the
tensors. ``assign=True`` *replaces* each param with the incoming tensor instead
of ``copy_``-casting it into the fp8 placeholder. For a mixed-precision
checkpoint (e.g. ``flux2_dev_fp8mixed``, whose ``_quantization_metadata`` only
lists the fp8 layers) this **preserves** the higher-precision (bf16) layers the
author stored for quality — which is the whole point of mixed precision. These
tests pin that: dynamic load must NOT coerce a bf16 weight down to the model's
fp8 storage dtype, and quantized/fp8-stored weights must keep their native
layout across the vbar so their scale survives.
"""
import contextlib
import os
import types
import unittest
from unittest import mock

import torch

import comfy.memory_management as memory_management
import comfy.model_base as model_base
import comfy.model_management as model_management
import comfy.ops as ops


def _fake_basemodel_self(diffusion_model):
    """Minimal stand-in exposing only what BaseModel.load_model_weights touches."""
    cfg = types.SimpleNamespace(process_unet_state_dict=lambda sd: sd)
    return types.SimpleNamespace(
        diffusion_model=diffusion_model,
        model_config=cfg,
        get_dtype=lambda: torch.float8_e4m3fn,
        manual_cast_dtype=torch.bfloat16,
    )


def _tiny_fp8_linear():
    """A Linear whose weight param is an fp8 placeholder, as an fp8 model builds."""
    layer = torch.nn.Linear(10, 20, bias=False)
    layer.weight = torch.nn.Parameter(
        torch.zeros(20, 10, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    return layer


@contextlib.contextmanager
def limit_free_vram(free_bytes):
    """Force the dynamic patcher's resident-budget check to see a small amount of
    free VRAM, so a model that would otherwise stay resident is streamed instead.

    This is the unit-test analogue of the runtime levers for capping reported
    VRAM (NVIDIA MPS ``CUDA_MPS_PINNED_DEVICE_MEM_LIMIT``, or a ballast tensor):
    ModelPatcherDynamic.load() decides resident-vs-stream from
    ``model_management.get_free_memory``, so patching it deterministically
    exercises the vbar streaming path without needing a 24 GB+ model.
    """
    def fake_get_free_memory(dev=None, torch_free_too=False):
        return (free_bytes, free_bytes) if torch_free_too else free_bytes

    with mock.patch.object(model_management, "get_free_memory", side_effect=fake_get_free_memory):
        yield


class TestMixedPrecisionStoragePreserved(unittest.TestCase):
    """A mixed checkpoint deliberately keeps some layers in higher precision
    (bf16) for quality; loading them into the model's fp8 storage placeholder
    must not silently down-convert them. This must hold for BOTH the legacy
    ``copy_`` path (assign=False) and the dynamic VRAM ``assign`` path
    (assign=True). Layers stored at equal-or-lower precision still load normally
    (so an fp8 layer upcasts to bf16 when the model dtype is bf16)."""

    def _load(self, model, src, assign):
        model_base.BaseModel.load_model_weights(
            _fake_basemodel_self(model), {"weight": src.clone()}, assign=assign
        )

    def test_higher_precision_bf16_layer_preserved_both_paths(self):
        src = torch.randn(20, 10, dtype=torch.bfloat16)
        for assign in (True, False):
            with self.subTest(assign=assign):
                model = _tiny_fp8_linear()
                self._load(model, src, assign)
                self.assertEqual(model.weight.dtype, torch.bfloat16)
                torch.testing.assert_close(model.weight, src)

    def test_fp8_stored_layer_stays_fp8_both_paths(self):
        src = torch.randn(20, 10).to(torch.float8_e4m3fn)
        for assign in (True, False):
            with self.subTest(assign=assign):
                model = _tiny_fp8_linear()
                self._load(model, src, assign)
                self.assertEqual(model.weight.dtype, torch.float8_e4m3fn)
                torch.testing.assert_close(model.weight.float(), src.float())

    def test_lower_precision_weight_upcasts_to_param_dtype(self):
        """fp8-stored weight into a bf16 placeholder (e.g. --fp8_storage off):
        not retargeted, so it upcasts to bf16 — never preserve a *lower*
        precision than the model asked for."""
        layer = torch.nn.Linear(10, 20, bias=False).to(torch.bfloat16)
        src = torch.randn(20, 10).to(torch.float8_e4m3fn)
        for assign in (False,):  # copy_ path upcasts; assign would replace as fp8
            with self.subTest(assign=assign):
                self._load(layer, src, assign)
                self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_matching_dtype_untouched_both_paths(self):
        src = torch.randn(20, 10, dtype=torch.bfloat16)
        for assign in (True, False):
            with self.subTest(assign=assign):
                layer = torch.nn.Linear(10, 20, bias=False).to(torch.bfloat16)
                self._load(layer, src, assign)
                self.assertEqual(layer.weight.dtype, torch.bfloat16)
                torch.testing.assert_close(layer.weight, src)


class TestStreamsInNativeDtype(unittest.TestCase):
    """The vbar must stream scale-carrying weights in their native low-precision
    layout, not densely materialize them mid-stream (which drops the per-tensor
    scale and diverges from the resident result)."""

    def test_plain_fp8_weight_streams_native(self):
        self.assertTrue(ops._streams_in_native_dtype(torch.zeros(4, 4, dtype=torch.float8_e4m3fn)))

    def test_bf16_weight_does_not_stream_native(self):
        self.assertFalse(ops._streams_in_native_dtype(torch.zeros(4, 4, dtype=torch.bfloat16)))

    def test_fp32_weight_does_not_stream_native(self):
        self.assertFalse(ops._streams_in_native_dtype(torch.zeros(4, 4, dtype=torch.float32)))

    def test_none_is_not_native(self):
        self.assertFalse(ops._streams_in_native_dtype(None))

    @unittest.skipUnless(
        ops.mixed_precision_quantization_available(),
        "requires comfy_kitchen-backed quantized tensors",
    )
    def test_quantized_tensor_streams_native(self):
        from comfy.quant_ops import QuantizedTensor

        qt = QuantizedTensor.from_float(
            torch.randn(20, 10, dtype=torch.bfloat16),
            "TensorCoreFP8E4M3Layout",
            scale="recalculate",
        )
        self.assertTrue(ops._streams_in_native_dtype(qt))


def _aimdo_runtime_available():
    try:
        import comfy.aimdo_integration  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available() and memory_management.aimdo_enabled()


@unittest.skipUnless(
    os.environ.get("COMFY_TEST_FP8_MODEL") and _aimdo_runtime_available(),
    "set COMFY_TEST_FP8_MODEL to an fp8 diffusion checkpoint and run on a CUDA "
    "box with comfy-aimdo to exercise end-to-end dynamic-VRAM streaming quality",
)
class TestDynamicStreamingQuality(unittest.TestCase):
    """End-to-end quality check: a model forced to stream under dynamic VRAM
    must produce the same forward output as the resident reference. The
    ``limit_free_vram`` helper provides the deterministic low-VRAM condition that
    drives the vbar streaming path. Catches both the load-dtype regression and
    any scale corruption in the streaming geometry."""

    def _run(self, seed=7):
        import comfy.sd as sd

        path = os.environ["COMFY_TEST_FP8_MODEL"]
        patcher = sd.load_diffusion_model(path)
        model_management.load_models_gpu([patcher], force_full_load=True)
        model = patcher.model
        device = model_management.get_torch_device()
        cfg = model.model_config.unet_config
        torch.manual_seed(seed)
        x = torch.randn(1, cfg.get("in_channels", 128), 32, 32, device=device, dtype=torch.bfloat16)
        t = torch.tensor([1.0], device=device)
        ctx = torch.randn(1, 64, cfg.get("context_in_dim", 4096), device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.apply_model(x, t, c_crossattn=ctx)
        return out.float().cpu()

    def test_streamed_matches_resident(self):
        resident = self._run()
        # ~256 MB of free VRAM forces the dynamic patcher to stream every block.
        with limit_free_vram(256 * 1024 * 1024):
            streamed = self._run()
        torch.testing.assert_close(streamed, resident, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
