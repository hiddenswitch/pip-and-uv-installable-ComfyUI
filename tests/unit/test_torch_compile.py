from __future__ import annotations

import os

import torch

from comfy.component_model.torch_cache import setup_torch_compile_cache_dirs
from comfy.model_management_types import HooksSupportStub
from comfy.model_patcher import ModelPatcherDynamic
from comfy.patcher_extension import CallbacksMP, WrappersMP
from comfy_api.torch_helpers.torch_compile import (
    TORCH_COMPILE_KWARGS,
    TORCH_COMPILE_STRATEGY,
    _CompiledModel,
    set_torch_compile_wrapper,
)
from comfy_extras.nodes.nodes_torch_compile import TorchCompileModel


class _FakePatcher(HooksSupportStub):
    def __init__(self):
        self.model_options = {}
        self.clone_disable_dynamic = None
        self.removed = []
        self.added = []
        self.callbacks = {}
        self.module = torch.nn.Linear(1, 1)

    def clone(self, disable_dynamic=False):
        clone = type(self)()
        clone.clone_disable_dynamic = disable_dynamic
        return clone

    def get_model_object(self, key):
        assert key == "diffusion_model"
        return self.module

    def remove_wrappers_with_key(self, wrapper_type: str, key: str) -> list:
        self.removed.append((wrapper_type, key))
        return []

    def add_wrapper_with_key(self, wrapper_type: str, key: str, wrapper):
        self.added.append((wrapper_type, key, wrapper))

    def remove_callbacks_with_key(self, call_type: str, key: str):
        callbacks = self.callbacks.get(call_type, {})
        callbacks.pop(key, None)

    def add_callback_with_key(self, call_type: str, key: str, callback):
        self.callbacks.setdefault(call_type, {})[key] = [callback]


def test_set_torch_compile_wrapper_omits_mode_when_options_are_present(monkeypatch):
    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()
    sentinel = object()

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        options={"guard_filter_fn": sentinel},
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )

    assert calls[0][1] == {
        "backend": "inductor",
        "options": {"guard_filter_fn": sentinel},
        "fullgraph": False,
        "dynamic": False,
    }
    assert "mode" not in calls[0][1]
    assert "mode" not in patcher.model_options[TORCH_COMPILE_KWARGS]


def test_set_torch_compile_wrapper_keeps_mode_without_options(monkeypatch):
    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        mode="reduce-overhead",
    )

    assert calls[0]["mode"] == "reduce-overhead"
    assert "options" not in calls[0]


def test_set_torch_compile_wrapper_compiles_top_level_model_with_apply_wrapper(monkeypatch):
    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return torch.nn.Identity()

    class RepeatedRegionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
            self.single_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.final_layer = torch.nn.Linear(1, 1)

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()
    patcher.module = RepeatedRegionModel()

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        options={"guard_filter_fn": lambda guards: guards},
        mode="reduce-overhead",
    )

    assert len(calls) == 1
    assert calls[0][0] is patcher.module
    assert calls[0][1]["options"]["guard_filter_fn"] is not None
    assert "mode" not in calls[0][1]
    assert len(patcher.added) == 1
    assert all(isinstance(block, torch.nn.Linear) for block in patcher.module.double_blocks)
    assert all(isinstance(block, torch.nn.Linear) for block in patcher.module.single_blocks)
    assert isinstance(patcher.module.final_layer, torch.nn.Linear)
    assert patcher.model_options[TORCH_COMPILE_STRATEGY] == {"diffusion_model": "module"}


def test_set_torch_compile_wrapper_compiles_nonresident_repeated_model_at_top_level(monkeypatch):
    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return model

    class RepeatedRegionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
            self.single_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.final_layer = torch.nn.Linear(1, 1)

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()
    patcher.module = RepeatedRegionModel()
    patcher.module.double_blocks[0]._v = object()

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        mode="reduce-overhead",
    )

    assert len(calls) == 1
    assert calls[0][0] is patcher.module
    assert callable(calls[0][1]["backend"])
    assert len(patcher.added) == 1
    assert all(isinstance(block, torch.nn.Linear) for block in patcher.module.double_blocks)
    assert all(isinstance(block, torch.nn.Linear) for block in patcher.module.single_blocks)
    assert isinstance(patcher.module.final_layer, torch.nn.Linear)
    assert patcher.model_options[TORCH_COMPILE_STRATEGY] == {"diffusion_model": "module_weight_cast"}


def test_set_torch_compile_wrapper_uses_aimdo_strategy_for_dynamic_vbar_modules(monkeypatch):
    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return model

    class DynamicLinear(torch.nn.Linear):
        pass

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()
    patcher.module = torch.nn.Sequential(DynamicLinear(1, 1))
    patcher.module[0]._v = object()

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        mode="reduce-overhead",
    )

    assert calls[0][0] is patcher.module
    assert callable(calls[0][1]["backend"])
    assert calls[0][1]["options"]["triton.cudagraphs"] is False
    assert calls[0][1]["options"]["triton.cudagraph_trees"] is False
    assert "mode" not in calls[0][1]
    assert len(patcher.added) == 1
    assert patcher.model_options[TORCH_COMPILE_STRATEGY] == {"diffusion_model": "module_weight_cast"}


def test_compiled_model_preserves_inert_transformer_options(monkeypatch):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None,
                    transformer_options=None, **kwargs):
            self.seen.append(transformer_options)
            return x

    monkeypatch.setattr(torch, "compile", lambda *, model, **kwargs: model)
    model = RecordingModel()
    model.dtype = torch.float16
    compiled = _CompiledModel(model, {"backend": "inductor"})
    options = {
        "cond_or_uncond": [0],
        "sigmas": object(),
        "wrappers": {WrappersMP.APPLY_MODEL: {"torch.compile": [object()]}},
    }

    assert compiled.dtype == torch.float16
    compiled(torch.ones(1), torch.ones(1), torch.ones(1), transformer_options=options)

    assert model.seen == [options]


def test_compiled_model_uses_eager_path_for_transformer_features(monkeypatch):
    class RecordingModel(torch.nn.Module):
        def __init__(self, label):
            super().__init__()
            self.label = label
            self.called = False

        def forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None,
                    transformer_options=None, **kwargs):
            self.called = True
            return x + (1 if self.label == "compiled" else 2)

    original = RecordingModel("original")
    compiled_target = RecordingModel("compiled")
    monkeypatch.setattr(torch, "compile", lambda *, model, **kwargs: compiled_target)
    compiled = _CompiledModel(original, {"backend": "inductor"})

    result = compiled(
        torch.zeros(1),
        torch.ones(1),
        torch.ones(1),
        transformer_options={"patches": {"post_input": [object()]}},
    )

    assert result.item() == 2
    assert original.called
    assert not compiled_target.called


def test_compiled_model_uses_eager_path_for_diffusion_wrappers(monkeypatch):
    class RecordingModel(torch.nn.Module):
        def __init__(self, label):
            super().__init__()
            self.label = label
            self.called = False

        def forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None,
                    transformer_options=None, **kwargs):
            self.called = True
            return x + (1 if self.label == "compiled" else 2)

    original = RecordingModel("original")
    compiled_target = RecordingModel("compiled")
    monkeypatch.setattr(torch, "compile", lambda *, model, **kwargs: compiled_target)
    compiled = _CompiledModel(original, {"backend": "inductor"})

    result = compiled(
        torch.zeros(1),
        torch.ones(1),
        torch.ones(1),
        transformer_options={"wrappers": {WrappersMP.DIFFUSION_MODEL: {"extension": [object()]}}},
    )

    assert result.item() == 2
    assert original.called
    assert not compiled_target.called


def test_compiled_model_falls_back_after_unsupported_fp8e4nv(monkeypatch):
    class OriginalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None,
                    transformer_options=None, **kwargs):
            self.calls += 1
            return x + 2

    class FailingCompiledFlux(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise ValueError("type fp8e4nv not supported in this architecture")

    original = OriginalModel()
    monkeypatch.setattr(torch, "compile", lambda *, model, **kwargs: FailingCompiledFlux())
    compiled = _CompiledModel(original, {"backend": "inductor"})

    assert compiled(torch.zeros(1), torch.ones(1), torch.ones(1)).item() == 2
    assert compiled(torch.zeros(1), torch.ones(1), torch.ones(1)).item() == 2
    assert original.calls == 2


def test_torch_compile_model_clones_dynamic_patchers_as_static(monkeypatch):
    compile_calls = []
    monkeypatch.setattr(
        "comfy_extras.nodes.nodes_torch_compile.set_torch_compile_wrapper",
        lambda patcher, **kwargs: compile_calls.append((patcher, kwargs)),
    )

    compiled, = TorchCompileModel().patch(_FakePatcher())

    assert compiled.clone_disable_dynamic is True
    callback = compiled.callbacks[CallbacksMP.ON_LOAD]["torch.compile"][0]
    callback(compiled, torch.device("cuda"), 0, False, False)
    assert compile_calls[0][0] is compiled
    assert compile_calls[0][1]["keys"] == ["diffusion_model"]


def test_torch_compile_model_preserves_real_dynamic_patchers(monkeypatch):
    compile_calls = []
    monkeypatch.setattr(
        "comfy_extras.nodes.nodes_torch_compile.set_torch_compile_wrapper",
        lambda patcher, **kwargs: compile_calls.append((patcher, kwargs)),
    )
    patcher = ModelPatcherDynamic(torch.nn.Linear(1, 1), torch.device("cuda:0"), torch.device("cpu"))

    compiled, = TorchCompileModel().patch(patcher)

    assert compiled.is_dynamic() is True
    callback = compiled.callbacks[CallbacksMP.ON_LOAD]["torch.compile"][0]
    callback(compiled, torch.device("cuda:0"), 0, False, False)
    assert compile_calls[0][0] is compiled


def test_setup_torch_compile_cache_dirs_uses_app_cache_and_preserves_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "comfy.component_model.torch_cache.user_cache_dir",
        lambda appname: str(tmp_path / appname),
    )
    monkeypatch.setenv("TRITON_CACHE_DIR", "/already/set")
    for env_name in ("TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
        monkeypatch.delenv(env_name, raising=False)

    setup_torch_compile_cache_dirs()

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "comfyui" / "torch_compile" / "inductor")
    assert os.environ["TRITON_CACHE_DIR"] == "/already/set"
    assert os.environ["CUDA_CACHE_PATH"] == str(tmp_path / "comfyui" / "torch_compile" / "cuda")
