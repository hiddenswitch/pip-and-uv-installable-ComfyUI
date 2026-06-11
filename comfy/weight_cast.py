from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch

from . import memory_management
from .cli_args import _args
from .weight_cast_ops import (
    device_type_to_code,
    dtype_to_code,
    module_bias_shape,
    module_weight_shape,
    next_invocation_id,
    register_module,
    register_module_with_stable_key,
)

logger = logging.getLogger(__name__)

LegacyCast = Callable[..., tuple[torch.Tensor, torch.Tensor | None, Any]]
LegacyUncast = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor | None, Any], None]

BACKEND_EAGER = "eager"
BACKEND_AIMDO = "aimdo"
BACKEND_GRAPH_VISIBLE = "graph_visible"
BACKEND_CUDA = "cuda"


def is_torch_compiling() -> bool:
    compiler_is_compiling = getattr(torch.compiler, "is_compiling", None)
    if compiler_is_compiling is not None and compiler_is_compiling():
        return True
    dynamo_is_compiling = getattr(torch._dynamo, "is_compiling", None)
    return bool(dynamo_is_compiling is not None and dynamo_is_compiling())


@dataclass
class WeightCastState:
    backend: str
    weight: torch.Tensor
    bias: torch.Tensor | None
    token: Any = None


@dataclass(frozen=True)
class WeightMaterializationSpec:
    module_key: int
    weight_key: str | None = None
    bias_key: str | None = None
    weight_shape: tuple[int, ...] | None = None
    bias_shape: tuple[int, ...] | None = None
    weight_storage_dtype: torch.dtype | None = None
    bias_storage_dtype: torch.dtype | None = None
    weight_model_dtype: torch.dtype | None = None
    bias_model_dtype: torch.dtype | None = None
    weight_vram_bytes: int = 0
    bias_vram_bytes: int = 0
    has_weight_lowvram_patch: bool = False
    has_bias_lowvram_patch: bool = False
    weight_function_count: int = 0
    bias_function_count: int = 0
    force_loaded: bool = False

    @property
    def vram_bytes(self) -> int:
        return self.weight_vram_bytes + self.bias_vram_bytes

    @property
    def has_python_materialization(self) -> bool:
        return (
            self.has_weight_lowvram_patch
            or self.has_bias_lowvram_patch
            or self.weight_function_count > 0
            or self.bias_function_count > 0
        )


def _empty_materialization_spec(module: torch.nn.Module) -> WeightMaterializationSpec:
    return WeightMaterializationSpec(module_key=register_module(module))


def get_materialization_spec(module: torch.nn.Module) -> WeightMaterializationSpec:
    spec = getattr(module, "_comfy_weight_materialization_spec", None)
    if spec is None:
        spec = _empty_materialization_spec(module)
        try:
            module._comfy_weight_materialization_spec = spec
        except Exception:
            pass
    return spec


def set_materialization_param(
    module: torch.nn.Module,
    param_key: str,
    *,
    key: str | None,
    tensor: torch.Tensor | None,
    model_dtype: torch.dtype | None,
    vram_bytes: int = 0,
    has_lowvram_patch: bool = False,
    function_count: int = 0,
) -> WeightMaterializationSpec:
    spec = get_materialization_spec(module)
    updates = {
        f"{param_key}_key": key,
        f"{param_key}_shape": None if tensor is None else tuple(tensor.shape),
        f"{param_key}_storage_dtype": None if tensor is None else tensor.dtype,
        f"{param_key}_model_dtype": model_dtype,
        f"{param_key}_vram_bytes": int(vram_bytes),
        f"has_{param_key}_lowvram_patch": bool(has_lowvram_patch),
        f"{param_key}_function_count": int(function_count),
    }
    spec = replace(spec, **updates)
    identity = _stable_materialization_identity(spec)
    if identity is not None:
        module_key = register_module_with_stable_key(module, identity)
        spec = replace(spec, module_key=module_key)
    else:
        register_module(module)
    try:
        module._comfy_weight_materialization_spec = spec
    except Exception:
        pass
    return spec


def set_materialization_force_loaded(module: torch.nn.Module, force_loaded: bool) -> WeightMaterializationSpec:
    spec = replace(get_materialization_spec(module), force_loaded=bool(force_loaded))
    try:
        module._comfy_weight_materialization_spec = spec
    except Exception:
        pass
    return spec


def _stable_materialization_identity(spec: WeightMaterializationSpec) -> str | None:
    if spec.weight_key is None and spec.bias_key is None:
        return None
    return "|".join(
        str(part)
        for part in (
            spec.weight_key,
            spec.bias_key,
            spec.weight_shape,
            spec.bias_shape,
            spec.weight_storage_dtype,
            spec.bias_storage_dtype,
            spec.weight_model_dtype,
            spec.bias_model_dtype,
        )
    )


class WeightCastRuntime:
    name = BACKEND_EAGER

    def resolve(
        self,
        module: torch.nn.Module,
        legacy_cast: LegacyCast,
        input: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        bias_dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
        want_requant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        raise NotImplementedError

    def release(
        self,
        module: torch.nn.Module,
        legacy_uncast: LegacyUncast,
        output: torch.Tensor,
        state: WeightCastState,
    ) -> torch.Tensor:
        raise NotImplementedError


class EagerWeightCastRuntime(WeightCastRuntime):
    name = BACKEND_EAGER

    def resolve(
        self,
        module: torch.nn.Module,
        legacy_cast: LegacyCast,
        input: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        bias_dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
        want_requant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, WeightCastState]:
        weight, bias, token = legacy_cast(
            module,
            input,
            dtype=dtype,
            device=device,
            bias_dtype=bias_dtype,
            offloadable=True,
            compute_dtype=compute_dtype,
            want_requant=want_requant,
        )
        return weight, bias, WeightCastState(self.name, weight, bias, token)

    def release(
        self,
        module: torch.nn.Module,
        legacy_uncast: LegacyUncast,
        output: torch.Tensor,
        state: WeightCastState,
    ) -> torch.Tensor:
        legacy_uncast(module, state.weight, state.bias, state.token)
        return output


class GraphVisibleWeightCastRuntime(WeightCastRuntime):
    name = BACKEND_GRAPH_VISIBLE

    def resolve(
        self,
        module: torch.nn.Module,
        legacy_cast: LegacyCast,
        input: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        bias_dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
        want_requant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, WeightCastState]:
        if input is None:
            raise RuntimeError("Graph-visible weight casting requires an input/exemplar tensor")
        module_key = getattr(module, "_comfy_weight_cast_key", None)
        if module_key is None:
            module_key = register_module(module)
        invocation_id = next_invocation_id()
        weight_shape = _materialization_shape(module, "weight")
        if weight_shape is None:
            weight_shape = module_weight_shape(module)
        bias_shape = _materialization_shape(module, "bias")
        if bias_shape is None:
            bias_shape = module_bias_shape(module)
        effective_dtype = dtype or input.dtype
        effective_bias_dtype = bias_dtype or effective_dtype
        device_index = -1 if input.device.index is None else input.device.index
        op_args = (
            input,
            weight_shape,
            module_key,
            invocation_id,
            dtype_to_code(effective_dtype),
            dtype_to_code(effective_bias_dtype),
            dtype_to_code(compute_dtype),
            want_requant,
            device_type_to_code(input.device.type),
            device_index,
        )
        if bias_shape is not None:
            weight, bias = torch.ops.comfy_weight.resolve_weight_bias(input, weight_shape, bias_shape, *op_args[2:])
        else:
            weight = torch.ops.comfy_weight.resolve_weight(*op_args)
            bias = None
        return weight, bias, (module_key, invocation_id)

    def release(
        self,
        module: torch.nn.Module,
        legacy_uncast: LegacyUncast,
        output: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        module_key, invocation_id = state
        torch.ops.comfy_weight.release_(output, module_key, invocation_id)
        return output


_EAGER_RUNTIME = EagerWeightCastRuntime()
_GRAPH_VISIBLE_RUNTIME = GraphVisibleWeightCastRuntime()


def _materialization_shape(module: torch.nn.Module, param_key: str) -> list[int] | None:
    cached_shape = getattr(module, f"_comfy_weight_cast_{param_key}_shape", None)
    if cached_shape is not None:
        return [int(dim) for dim in cached_shape]
    spec = get_materialization_spec(module)
    shape = getattr(spec, f"{param_key}_shape")
    if shape is None:
        return None
    return [int(dim) for dim in shape]


@torch.compiler.assume_constant_result
def _dynamic_vram_disabled() -> bool:
    return bool(getattr(_args(), "disable_dynamic_vram", False))


def aimdo_backend_unavailable_reason() -> str | None:
    if _dynamic_vram_disabled():
        return "dynamic VRAM disabled"
    if memory_management.aimdo_allocator is None:
        return "Aimdo allocator unavailable"
    return None


def aimdo_backend_available() -> bool:
    return aimdo_backend_unavailable_reason() is None


def graph_visible_backend_unavailable_reason() -> str | None:
    if _dynamic_vram_disabled():
        return "dynamic VRAM disabled"
    return None


def _module_needs_graph_visible_weight_cast(module: torch.nn.Module, input: torch.Tensor | None) -> bool:
    if hasattr(module, "comfy_cast_weights"):
        return True
    if getattr(module, "_v", None) is not None:
        return True
    if input is None:
        return False
    if len(getattr(module, "weight_function", ())) > 0 or len(getattr(module, "bias_function", ())) > 0:
        return True
    weight = getattr(module, "weight", None)
    if weight is not None and (weight.device != input.device or weight.dtype != input.dtype):
        return True
    bias = getattr(module, "bias", None)
    if bias is not None and (bias.device != input.device or bias.dtype != input.dtype):
        return True
    return False


def _is_device_cpu(device: torch.device) -> bool:
    return device.type == "cpu"


def get_weight_cast_runtime(module: torch.nn.Module, input: torch.Tensor | None = None) -> WeightCastRuntime:
    if (
        input is not None
        and is_torch_compiling()
        and graph_visible_backend_unavailable_reason() is None
        and not _is_device_cpu(input.device)
        and _module_needs_graph_visible_weight_cast(module, input)
    ):
        return _GRAPH_VISIBLE_RUNTIME
    return _EAGER_RUNTIME


def get_weight_cast_runtime_by_name(name: str) -> WeightCastRuntime:
    if name in (BACKEND_AIMDO, BACKEND_GRAPH_VISIBLE):
        return _GRAPH_VISIBLE_RUNTIME
    return _EAGER_RUNTIME


def list_weight_cast_backends() -> dict[str, dict[str, Any]]:
    reason = aimdo_backend_unavailable_reason()
    return {
        BACKEND_EAGER: {
            "available": True,
            "disabled": False,
            "unavailable_reason": None,
            "capabilities": ["legacy_layerwise_cast"],
        },
        BACKEND_AIMDO: {
            "available": reason is None,
            "disabled": _dynamic_vram_disabled(),
            "unavailable_reason": reason,
            "capabilities": ["dynamic_vbar", "torch_compile_custom_ops"],
        },
        BACKEND_GRAPH_VISIBLE: {
            "available": graph_visible_backend_unavailable_reason() is None,
            "disabled": _dynamic_vram_disabled(),
            "unavailable_reason": graph_visible_backend_unavailable_reason(),
            "capabilities": ["manual_cast", "dynamic_vbar", "torch_compile_custom_ops"],
        },
        BACKEND_CUDA: {
            "available": False,
            "disabled": False,
            "unavailable_reason": "native weight-cast backend not installed",
            "capabilities": [],
        },
    }
