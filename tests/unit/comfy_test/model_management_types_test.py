"""Regression: ModelManageableStub must satisfy every method
load_models_gpu (model_management.py) invokes on a loaded model. Previously
absent: model_mmap_residency, pinned_memory_size, partially_unload_ram.
Any class extending ModelManageableStub directly — e.g.
LatentUpscaleModelManageable in comfy_extras/nodes/nodes_latent_upscaler.py
— inherited the gap and raised AttributeError when memory-managed."""

import torch

from comfy.model_management_types import ModelManageable, ModelManageableStub


# Every method load_models_gpu invokes on a managed model. If any of these
# is missing from Protocol or Stub, an LTX-style production failure recurs.
_HOT_PATH_METHODS = (
    "current_loaded_device",
    "detach",
    "is_clone",
    "is_dynamic",
    "loaded_size",
    "lowvram_patch_counter",
    "model_dtype",
    "model_mmap_residency",
    "model_patches_to",
    "model_size",
    "partially_load",
    "partially_unload",
    "partially_unload_ram",
    "pinned_memory_size",
)


class _MinimalManageable(ModelManageableStub):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        return self.model


def test_protocol_declares_every_hot_path_method():
    missing = [m for m in _HOT_PATH_METHODS if not hasattr(ModelManageable, m)]
    assert missing == [], f"ModelManageable Protocol missing: {missing}"


def test_protocol_declares_clone_identity():
    assert "clone_base_uuid" in ModelManageable.__annotations__


def test_stub_provides_every_hot_path_method():
    missing = [m for m in _HOT_PATH_METHODS if not hasattr(ModelManageableStub, m)]
    assert missing == [], f"ModelManageableStub missing: {missing}"


def test_stub_replays_load_models_gpu_call_sequence_without_attribute_error():
    # Mirrors the exact sequence load_models_gpu uses on each loaded model.
    m = _MinimalManageable(torch.nn.Linear(8, 8))
    assert m.is_dynamic() is False
    assert m.loaded_size() >= 0
    resident, total = m.model_mmap_residency()
    assert (resident, total) == (0, sum(t.nbytes for t in m.model.state_dict().values()))
    assert m.pinned_memory_size() == 0
    m.partially_unload_ram(1024)  # no-op default; must not raise
    resident, _ = m.model_mmap_residency(free=True)
    assert resident == 0


def test_stub_default_reports_zero_resident_for_non_mmap_module():
    module = torch.nn.Linear(8, 8)
    expected_total = sum(t.nbytes for t in module.state_dict().values())
    resident, total = _MinimalManageable(module).model_mmap_residency()
    assert resident == 0
    assert total == expected_total
