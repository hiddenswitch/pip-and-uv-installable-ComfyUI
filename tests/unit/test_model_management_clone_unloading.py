import torch

from comfy import model_management
from comfy.model_management_types import ModelManageableStub
from comfy.model_patcher import ModelPatcher


class _StubManageable(ModelManageableStub):
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        if device_to is not None:
            self.model.to(device_to)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        if device_to is not None:
            self.model.to(device_to)
        return self.model


def _model_patcher():
    module = torch.nn.Linear(2, 2)
    return ModelPatcher(
        module,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=sum(t.nbytes for t in module.state_dict().values()),
        _force_core=True,
    )


def test_issue_64_unload_clone_ignores_manageable_without_clone_identity(monkeypatch):
    target = _model_patcher()
    unrelated_model = _StubManageable()
    unrelated = model_management.LoadedModel(unrelated_model)
    model_management.current_loaded_models.append(unrelated)
    keep_loaded = None

    def capture_free_memory(memory_required, device, keep_loaded_arg):
        nonlocal keep_loaded
        keep_loaded = keep_loaded_arg

    monkeypatch.setattr(model_management, "free_memory", capture_free_memory)
    try:
        model_management._unload_model_and_clones(target)
    finally:
        model_management.current_loaded_models.clear()

    assert keep_loaded == [unrelated]


def test_issue_65_unloading_resident_clone_invalidates_caller_residency(monkeypatch):
    resident_patcher = _model_patcher()
    resident = model_management.LoadedModel(resident_patcher)
    resident.model_load()
    caller_clone = resident_patcher.clone()
    assert caller_clone.loaded_size() == caller_clone.model_size()
    model_management.current_loaded_models.append(resident)

    def unload_except_kept(memory_required, device, keep_loaded):
        unloaded = []
        for loaded_model in model_management.current_loaded_models.copy():
            if loaded_model not in keep_loaded:
                loaded_model.model_unload()
                model_management.current_loaded_models.remove(loaded_model)
                unloaded.append(loaded_model)
        return unloaded

    monkeypatch.setattr(model_management, "free_memory", unload_except_kept)
    load_calls = 0
    load = caller_clone.load

    def record_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(caller_clone, "load", record_load)
    try:
        model_management._unload_model_and_clones(caller_clone)
        assert resident_patcher.loaded_size() == 0
        assert caller_clone.loaded_size() == 0

        model_management._load_models_gpu([caller_clone], force_full_load=True)
        assert load_calls == 1
        assert caller_clone.loaded_size() == caller_clone.model_size()
    finally:
        model_management.current_loaded_models.clear()
