from __future__ import annotations

import os

import torch

from comfy.component_model.torch_cache import setup_torch_compile_cache_dirs
from comfy.model_management_types import HooksSupportStub
from comfy.model_patcher import ModelPatcherDynamic
from comfy.patcher_extension import WrappersMP
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
        removed = [entry for entry in self.added if entry[0] == wrapper_type and entry[1] == key]
        self.added = [entry for entry in self.added if entry[0] != wrapper_type or entry[1] != key]
        return removed

    def add_wrapper_with_key(self, wrapper_type: str, key: str, wrapper):
        self.added.append((wrapper_type, key, wrapper))

    def get_wrappers(self, wrapper_type: str, key: str):
        return [wrapper for wt, k, wrapper in self.added if wt == wrapper_type and k == key]

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


def test_set_torch_compile_wrapper_uses_weight_cast_strategy_for_dynamic_patcher(monkeypatch):
    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Linear(1, 1)
    patcher = ModelPatcherDynamic(root, torch.device("cuda:0"), torch.device("cpu"))

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        mode="reduce-overhead",
    )

    assert len(calls) == 1
    assert callable(calls[0][1]["backend"])
    assert calls[0][1]["options"]["triton.cudagraphs"] is False
    assert patcher.model_options[TORCH_COMPILE_STRATEGY] == {"diffusion_model": "module_weight_cast"}


def test_set_torch_compile_wrapper_uses_weight_cast_strategy_for_cast_capable_model(monkeypatch):
    from comfy import ops

    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()
    patcher.module = ops.disable_weight_init.Linear(1, 1)

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
    )

    assert len(calls) == 1
    assert callable(calls[0][1]["backend"])
    assert patcher.model_options[TORCH_COMPILE_STRATEGY] == {"diffusion_model": "module_weight_cast"}


def test_set_torch_compile_wrapper_honors_cudagraph_mode_on_resident_model(monkeypatch):
    # A cudagraph compile mode requested on a non-dynamic (resident-weight)
    # model must not be silently stripped: the plain strategy preserves it.
    from comfy import ops

    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()
    patcher.module = ops.disable_weight_init.Linear(1, 1)

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        mode="reduce-overhead",
    )

    assert len(calls) == 1
    assert calls[0][1].get("mode") == "reduce-overhead"
    assert patcher.model_options[TORCH_COMPILE_STRATEGY] == {"diffusion_model": "module"}


def test_dynamic_vbar_compile_forces_all_cast_capable_layers_graph_visible(monkeypatch):
    from comfy import ops

    calls = []

    def fake_compile(*, model, **kwargs):
        calls.append((model, kwargs))
        return model

    class MixedResidencyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offloaded = ops.manual_cast.Linear(1, 1)
            self.resident = ops.disable_weight_init.Linear(1, 1)
            self.offloaded._v = object()
            self.resident.comfy_cast_weights = False

    monkeypatch.setattr(torch, "compile", fake_compile)
    patcher = _FakePatcher()
    patcher.module = MixedResidencyModel()

    set_torch_compile_wrapper(
        patcher,
        keys=["diffusion_model"],
        backend="inductor",
        mode="reduce-overhead",
    )

    assert calls[0][0] is patcher.module
    assert patcher.module.offloaded.comfy_cast_weights is True
    assert patcher.module.resident.comfy_cast_weights is True
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


def test_compiled_model_stabilizes_small_manual_cast_parameters_before_compile(monkeypatch):
    from comfy import ops
    from comfy import weight_cast_ops

    layer = ops.manual_cast.Linear(2, 1)
    seen_devices = []
    seen_keys = []

    def fake_compile(*, model, **kwargs):
        class Compiled(torch.nn.Module):
            def forward(self, x):
                seen_devices.append(model.weight.device)
                seen_keys.append(model._comfy_weight_cast_key)
                return model(x)

        return Compiled()

    monkeypatch.setattr(torch, "compile", fake_compile)
    compiled = _CompiledModel(layer, {"backend": "inductor"})
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    compiled(torch.ones(1, 2, device=device))

    assert seen_devices == [device]
    from comfy_api.torch_helpers.torch_compile import _compile_module_identity

    assert seen_keys == [weight_cast_ops.register_module_with_stable_key(layer, _compile_module_identity("", layer))]


def test_compiled_model_reuses_jit_graph_for_value_only_sampling_changes():
    compile_calls = []

    class SamplingInputsModel(torch.nn.Module):
        def forward(self, x, timestep, context, *, noise, transformer_options=None, **kwargs):
            sigmas = transformer_options["sigmas"]
            return (
                x
                + noise * 0.01
                + context.mean(dim=-1, keepdim=True)
                + timestep.reshape(1, 1).to(dtype=x.dtype)
                + sigmas.mean().to(dtype=x.dtype)
            )

    def counting_backend(gm, example_inputs):
        compile_calls.append((gm, example_inputs))
        return gm.forward

    torch._dynamo.reset()
    try:
        compiled = _CompiledModel(SamplingInputsModel(), {"backend": counting_backend, "dynamic": False})

        first_seed = torch.Generator().manual_seed(101)
        first = compiled(
            torch.zeros(2, 4),
            torch.tensor([999.0]),
            torch.full((2, 4), 0.25),
            noise=torch.randn(2, 4, generator=first_seed),
            transformer_options={"sigmas": torch.linspace(1.0, 0.0, 4)},
        )

        second_seed = torch.Generator().manual_seed(202)
        second = compiled(
            torch.zeros(2, 4),
            torch.tensor([333.0]),
            torch.full((2, 4), 0.75),
            noise=torch.randn(2, 4, generator=second_seed),
            transformer_options={"sigmas": torch.linspace(0.8, 0.2, 4)},
        )
    finally:
        torch._dynamo.reset()

    assert len(compile_calls) == 1
    assert not torch.allclose(first, second)


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
    def fake_set_torch_compile_wrapper(patcher, **kwargs):
        compile_calls.append((patcher, kwargs))
        patcher.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, "torch.compile")
        patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "torch.compile", lambda executor, *args, **kwargs: "compiled")

    monkeypatch.setattr(
        "comfy_extras.nodes.nodes_torch_compile.set_torch_compile_wrapper",
        fake_set_torch_compile_wrapper,
    )

    compiled, = TorchCompileModel().patch(_FakePatcher())

    assert compiled.clone_disable_dynamic is True
    wrapper = compiled.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile")[0]
    executor = lambda *args, **kwargs: None
    wrapper(executor)
    assert compile_calls[0][0] is compiled
    assert compile_calls[0][1]["keys"] == ["diffusion_model"]


def test_torch_compile_model_preserves_real_dynamic_patchers(monkeypatch):
    compile_calls = []
    def fake_set_torch_compile_wrapper(patcher, **kwargs):
        compile_calls.append((patcher, kwargs))
        patcher.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, "torch.compile")
        patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "torch.compile", lambda executor, *args, **kwargs: "compiled")

    monkeypatch.setattr(
        "comfy_extras.nodes.nodes_torch_compile.set_torch_compile_wrapper",
        fake_set_torch_compile_wrapper,
    )
    patcher = ModelPatcherDynamic(torch.nn.Linear(1, 1), torch.device("cuda:0"), torch.device("cpu"))

    compiled, = TorchCompileModel().patch(patcher)

    assert compiled.is_dynamic() is True
    wrapper = compiled.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile")[0]
    executor = lambda *args, **kwargs: None
    wrapper(executor)
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
