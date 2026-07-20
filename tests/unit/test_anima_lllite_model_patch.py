import pytest
import torch

from comfy_extras.nodes import nodes_model_patch


def test_model_patch_loader_routes_anima_lllite_checkpoint(monkeypatch):
    state_dict = {"lllite_conditioning1.conv1.weight": object()}
    metadata = {"lllite.version": "2"}
    created = {}

    class FakeAnimaLLLite:
        def __init__(self, loaded_state_dict, loaded_metadata, **kwargs):
            created["init"] = (loaded_state_dict, loaded_metadata, kwargs)

        def load_state_dict(self, loaded_state_dict, assign=False):
            created["load"] = (loaded_state_dict, assign)

    class FakeModelPatcher:
        def __init__(self, model, load_device, offload_device):
            self.model = model
            created["devices"] = (load_device, offload_device)

        @staticmethod
        def is_dynamic():
            return True

    monkeypatch.setattr(nodes_model_patch, "get_full_path_or_raise", lambda *_: "/models/anima.safetensors")
    monkeypatch.setattr(nodes_model_patch.comfy.utils, "load_torch_file", lambda *_, **__: (state_dict, metadata))
    monkeypatch.setattr(nodes_model_patch.comfy.utils, "weight_dtype", lambda _: torch.float16)
    monkeypatch.setattr(nodes_model_patch.comfy.model_management, "unet_offload_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes_model_patch.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes_model_patch.comfy.ldm.anima.lllite, "AnimaLLLite", FakeAnimaLLLite)
    monkeypatch.setattr(nodes_model_patch, "get_model_patcher_class", lambda: FakeModelPatcher)

    (model_patcher,) = nodes_model_patch.ModelPatchLoader().load_model_patch("anima.safetensors")

    assert isinstance(model_patcher.model, FakeAnimaLLLite)
    assert created["init"][:2] == (state_dict, metadata)
    assert created["init"][2]["dtype"] is torch.float16
    assert created["load"] == (state_dict, True)


def test_anima_lllite_rejects_unsupported_checkpoint_version():
    with pytest.raises(ValueError, match="Unsupported Anima LLLite version"):
        nodes_model_patch.comfy.ldm.anima.lllite.AnimaLLLite(
            {}, {"lllite.version": "1"}, operations=nodes_model_patch.comfy.ops.disable_weight_init,
        )
