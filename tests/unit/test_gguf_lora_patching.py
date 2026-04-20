"""Regression tests for the GGUF + LoRA model-patch path.

Exercises ``ModelPatcher.patch_weight_to_device`` with a synthetic
GGMLTensor-backed Linear weight and a stub LoRA adapter. Guards against
the regression where the method's quantized branch fell through to the
non-quantized LoRA-merge path, which (a) called ``torch.empty_like`` on
the GGMLTensor and got the *packed* storage shape (e.g. ``[5120, 2880]``
for a Q4_K ``[5120, 5120]`` row), and (b) wrote that garbage tensor
back as the model weight. Upstream ``city96/ComfyUI-GGUF`` attaches
patches to ``tensor.patches`` and defers the merge to
``GGMLLayer.get_weight``; this test locks that behaviour in.
"""
from __future__ import annotations

import pytest
import torch

from comfy.gguf import GGMLTensor, GGMLOps, is_quantized
from comfy.lora_types import PatchTuple
from comfy.model_patcher import ModelPatcher
from comfy.weight_adapter.lora import LoRAAdapter
from gguf import GGMLQuantizationType


LOGICAL_IN = 16
LOGICAL_OUT = 16
PACKED_BYTES_PER_ROW = 9  # deliberately ≠ LOGICAL_IN to mimic Q4_K packing


def _make_gguf_linear() -> torch.nn.Module:
    """Return a Linear module whose .weight is a GGMLTensor with packed
    storage shape ``[LOGICAL_OUT, PACKED_BYTES_PER_ROW]`` but logical
    shape ``[LOGICAL_OUT, LOGICAL_IN]``."""
    packed = torch.zeros(LOGICAL_OUT, PACKED_BYTES_PER_ROW, dtype=torch.uint8)
    weight = GGMLTensor(
        packed,
        tensor_type=GGMLQuantizationType.Q4_K,
        tensor_shape=torch.Size([LOGICAL_OUT, LOGICAL_IN]),
    )
    layer = GGMLOps.Linear(LOGICAL_IN, LOGICAL_OUT, bias=False)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


def _stub_lora_patch() -> PatchTuple:
    up = torch.zeros(LOGICAL_OUT, 2, dtype=torch.float32)
    down = torch.zeros(2, LOGICAL_IN, dtype=torch.float32)
    adapter = LoRAAdapter(
        loaded_keys={"weight.lora_up.weight", "weight.lora_down.weight"},
        weights=(up, down, None, None, None, None),
    )
    return PatchTuple(
        strength_patch=1.0,
        patch=adapter,
        strength_model=1.0,
        offset=None,
        function=None,
    )


class TestGGUFLoRAPatch:
    def _new_patcher(self) -> tuple[ModelPatcher, str]:
        model = _make_gguf_linear()
        patcher = ModelPatcher(
            model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
        )
        patcher.gguf.loaded_from_gguf = True
        key = "weight"
        patcher.patches[key] = [_stub_lora_patch()]
        return patcher, key

    def test_weight_recognised_as_quantized(self):
        patcher, key = self._new_patcher()
        assert is_quantized(patcher.model.weight)

    def test_patch_preserves_gguf_subclass(self):
        """After patching, the model weight is still a GGMLTensor — not a
        raw Tensor with packed-storage shape written by the fall-through."""
        patcher, key = self._new_patcher()
        patcher.patch_weight_to_device(key, device_to=torch.device("cpu"))
        installed = patcher.model.weight
        assert isinstance(installed, GGMLTensor), (
            f"expected GGMLTensor after patch, got {type(installed).__name__}"
        )

    def test_patch_preserves_logical_shape(self):
        """The weight's reported shape is the logical (dequantized) shape,
        not the packed storage shape. Before the fix, fall-through wrote
        a raw tensor of the packed shape back as the model weight."""
        patcher, key = self._new_patcher()
        patcher.patch_weight_to_device(key, device_to=torch.device("cpu"))
        installed = patcher.model.weight
        assert tuple(installed.shape) == (LOGICAL_OUT, LOGICAL_IN)

    def test_patches_attached_to_tensor(self):
        """LoRA patches are attached to ``.patches`` so GGMLLayer.get_weight
        can merge them at forward time after dequantization."""
        patcher, key = self._new_patcher()
        patcher.patch_weight_to_device(key, device_to=torch.device("cpu"))
        installed = patcher.model.weight
        assert hasattr(installed, "patches")
        assert len(installed.patches) == 1
        attached_patches, attached_key = installed.patches[0]
        assert attached_key == key
        assert len(attached_patches) == 1

    def test_unpatch_clears_tensor_patches(self):
        """unpatch_model clears ``.patches`` on every parameter that has them,
        using per-parameter state (not the global self.patches dict)."""
        patcher, key = self._new_patcher()
        patcher.patch_weight_to_device(key, device_to=torch.device("cpu"))
        assert len(patcher.model.weight.patches) == 1
        patcher.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=True)
        assert patcher.model.weight.patches == []


@pytest.mark.parametrize("inplace_update", [False, True])
def test_patch_weight_to_device_does_not_corrupt_gguf_weight(inplace_update):
    """Parametrised over inplace_update: neither path should ever replace
    the GGMLTensor with a raw [out, packed_bytes] tensor."""
    model = _make_gguf_linear()
    patcher = ModelPatcher(
        model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        weight_inplace_update=inplace_update,
    )
    patcher.gguf.loaded_from_gguf = True
    patcher.patches["weight"] = [_stub_lora_patch()]

    patcher.patch_weight_to_device("weight", device_to=torch.device("cpu"))

    weight = patcher.model.weight
    assert isinstance(weight, GGMLTensor)
    assert tuple(weight.shape) == (LOGICAL_OUT, LOGICAL_IN)
    assert len(getattr(weight, "patches", [])) == 1


def test_quantized_branch_skips_eager_lora_merge(monkeypatch):
    """The quantized branch must NOT call the eager merge path (cast_to_device
    + calculate_weight). Previously the ``if patch_on_device: return`` let
    non-on-device runs fall through to calculate_weight, which corrupted
    the model weight on a cross-device cast."""
    from comfy import lora, model_management

    cast_calls: list[tuple] = []
    merge_calls: list[tuple] = []

    original_cast = model_management.cast_to_device
    original_merge = lora.calculate_weight

    def spy_cast(tensor, device, dtype, copy=False):
        cast_calls.append((type(tensor).__name__, device, dtype, copy))
        return original_cast(tensor, device, dtype, copy)

    def spy_merge(*args, **kwargs):
        merge_calls.append(args[:2])
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(model_management, "cast_to_device", spy_cast)
    monkeypatch.setattr(lora, "calculate_weight", spy_merge)
    # model_patcher imports the symbol directly
    import comfy.model_patcher as mp
    monkeypatch.setattr(mp.lora, "calculate_weight", spy_merge)

    model = _make_gguf_linear()
    patcher = ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    patcher.gguf.loaded_from_gguf = True
    patcher.patches["weight"] = [_stub_lora_patch()]

    patcher.patch_weight_to_device("weight", device_to=torch.device("cpu"))

    assert merge_calls == [], (
        "quantized-weight path must defer LoRA merge to GGMLLayer.get_weight; "
        f"saw eager calculate_weight calls: {merge_calls}"
    )
    assert cast_calls == [], (
        "quantized-weight path must not call cast_to_device (which uses "
        f"torch.empty_like on the packed shape); saw: {cast_calls}"
    )
