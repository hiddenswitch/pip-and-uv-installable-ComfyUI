from __future__ import annotations

import logging
import itertools
import weakref
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_MODULES: weakref.WeakValueDictionary[int, torch.nn.Module] = weakref.WeakValueDictionary()
_ACTIVE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor | None, object]] = {}
_PREFETCH: Callable[..., object] | None = None
_RESOLVE: Callable[..., tuple[torch.Tensor, torch.Tensor | None, object]] | None = None
_RELEASE: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor | None, object], None] | None = None
_PREFETCHED: dict[tuple[int, int], object] = {}
_INVOCATION_IDS = itertools.count(1)

_DTYPES = {
    0: None,
    1: torch.float32,
    2: torch.float16,
    3: torch.bfloat16,
    4: getattr(torch, "float8_e4m3fn", None),
    5: getattr(torch, "float8_e5m2", None),
}


def dtype_to_code(dtype: torch.dtype | None) -> int:
    for code, candidate in _DTYPES.items():
        if candidate is dtype:
            return code
    return 0


def code_to_dtype(code: int) -> torch.dtype | None:
    return _DTYPES.get(_concrete_int(code))


def _concrete_int(value: int) -> int:
    try:
        return int(value)
    except TypeError:
        node = getattr(value, "node", None)
        hint = getattr(node, "hint", None)
        if hint is not None:
            return int(hint)
        raise


def register_module(module: torch.nn.Module) -> int:
    key = getattr(module, "_comfy_weight_cast_key", None)
    if key is None:
        key = id(module)
        try:
            module._comfy_weight_cast_key = key
        except Exception:
            pass
    _MODULES[key] = module
    return key


def module_key_tensor(module: torch.nn.Module) -> torch.Tensor:
    key_tensor = getattr(module, "_comfy_weight_cast_key_tensor", None)
    if key_tensor is None:
        key = register_module(module)
        key_tensor = torch.tensor(key, dtype=torch.int64)
        try:
            module._comfy_weight_cast_key_tensor = key_tensor
        except Exception:
            pass
    return key_tensor


def module_invocation_tensor(module: torch.nn.Module) -> torch.Tensor:
    invocation_tensor = getattr(module, "_comfy_weight_cast_invocation_tensor", None)
    if invocation_tensor is None:
        invocation_tensor = torch.zeros((), dtype=torch.int64)
        try:
            module._comfy_weight_cast_invocation_tensor = invocation_tensor
        except Exception:
            pass
    return invocation_tensor


def module_weight_shape_tensor(module: torch.nn.Module) -> torch.Tensor:
    shape_tensor = getattr(module, "_comfy_weight_cast_weight_shape_tensor", None)
    if shape_tensor is not None:
        return shape_tensor
    weight = getattr(module, "weight", None)
    if weight is None:
        raise RuntimeError(f"Module {type(module).__name__} has no weight")
    if shape_tensor is None or tuple(shape_tensor.shape) != tuple(weight.shape) or shape_tensor.dtype is not torch.uint8:
        shape_tensor = torch.empty(tuple(weight.shape), dtype=torch.uint8)
        try:
            module._comfy_weight_cast_weight_shape_tensor = shape_tensor
        except Exception:
            pass
    return shape_tensor


def module_bias_shape_tensor(module: torch.nn.Module) -> torch.Tensor | None:
    shape_tensor = getattr(module, "_comfy_weight_cast_bias_shape_tensor", None)
    if shape_tensor is not None:
        return shape_tensor
    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    if shape_tensor is None or tuple(shape_tensor.shape) != tuple(bias.shape) or shape_tensor.dtype is not torch.uint8:
        shape_tensor = torch.empty(tuple(bias.shape), dtype=torch.uint8)
        try:
            module._comfy_weight_cast_bias_shape_tensor = shape_tensor
        except Exception:
            pass
    return shape_tensor


@torch.compiler.assume_constant_result
def next_invocation_id() -> int:
    return next(_INVOCATION_IDS)


def set_callbacks(
    resolve: Callable[..., tuple[torch.Tensor, torch.Tensor | None, object]],
    release: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor | None, object], None],
    prefetch: Callable[..., object] | None = None,
) -> None:
    global _PREFETCH, _RESOLVE, _RELEASE
    _PREFETCH = prefetch
    _RESOLVE = resolve
    _RELEASE = release


def _module(module_key: int) -> torch.nn.Module:
    module = _MODULES.get(_concrete_int(module_key))
    if module is None:
        raise RuntimeError(f"Unknown comfy weight-cast module id {module_key}")
    return module


def get_registered_module(module_key: torch.Tensor | int) -> torch.nn.Module | None:
    return _MODULES.get(_tensor_key(module_key))


def _tensor_key(value: torch.Tensor | int) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError("Expected scalar comfy weight-cast key tensor")
        return int(value.item())
    return _concrete_int(value)


def _fake_weight(weight_shape: torch.Tensor, exemplar: torch.Tensor, dtype_code: int) -> torch.Tensor:
    dtype = code_to_dtype(dtype_code) or exemplar.dtype
    return exemplar.new_empty(tuple(weight_shape.shape), dtype=dtype)


def _fake_bias(bias_shape: torch.Tensor, exemplar: torch.Tensor, dtype_code: int) -> torch.Tensor:
    dtype = code_to_dtype(dtype_code) or exemplar.dtype
    return exemplar.new_empty(tuple(bias_shape.shape), dtype=dtype)


def _custom_op_return(output: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
    # torch.library custom ops are functional by default and runtime alias
    # checking also sees AOT-wrapped tensors. Return an owned tensor so materialized
    # module parameters never alias lifted graph inputs.
    return output.clone()


def _prefetch_token(module_key: int, invocation_id: int) -> torch.Tensor:
    return torch.tensor(invocation_id, dtype=torch.int64)


def _prefetch(
    exemplar: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    if _PREFETCH is None:
        raise RuntimeError("comfy_weight prefetch callback is not installed")
    module_key = _tensor_key(module_key)
    invocation_id = _tensor_key(invocation_id)
    module = _module(module_key)
    _PREFETCHED[(module_key, invocation_id)] = _PREFETCH(
        module,
        exemplar,
        code_to_dtype(dtype_code),
        code_to_dtype(bias_dtype_code),
        code_to_dtype(compute_dtype_code),
        want_requant,
    )
    return _prefetch_token(module_key, invocation_id)


@torch.library.custom_op(
    "comfy_weight::prefetch_weight",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def prefetch_weight(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    return _prefetch(exemplar, module_key, invocation_id, dtype_code, bias_dtype_code, compute_dtype_code, want_requant)


@prefetch_weight.register_fake
def _prefetch_weight_fake(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    return torch.empty((), dtype=torch.int64)


@torch.library.custom_op(
    "comfy_weight::prefetch_weight_bias",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def prefetch_weight_bias(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    bias_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    return _prefetch(exemplar, module_key, invocation_id, dtype_code, bias_dtype_code, compute_dtype_code, want_requant)


@prefetch_weight_bias.register_fake
def _prefetch_weight_bias_fake(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    bias_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    return torch.empty((), dtype=torch.int64)


def _consume_prefetch(module_key: int, invocation_id: int) -> object:
    return _PREFETCHED.pop((module_key, invocation_id), None)


@torch.library.custom_op(
    "comfy_weight::resolve_weight",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def resolve_weight(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    if _RESOLVE is None:
        raise RuntimeError("comfy_weight resolve callback is not installed")
    module_key = _tensor_key(module_key)
    invocation_id = _tensor_key(invocation_id)
    module = _module(module_key)
    weight, bias, state = _RESOLVE(
        module,
        exemplar,
        code_to_dtype(dtype_code),
        code_to_dtype(bias_dtype_code),
        code_to_dtype(compute_dtype_code),
        want_requant,
    )
    weight = _custom_op_return(weight, exemplar, weight_shape, module_key, invocation_id)
    _ACTIVE[(module_key, invocation_id)] = (weight, bias, state)
    return weight


@torch.library.custom_op(
    "comfy_weight::resolve_prefetched_weight",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def resolve_prefetched_weight(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    prefetch_token: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    if _RESOLVE is None:
        raise RuntimeError("comfy_weight resolve callback is not installed")
    module_key = _tensor_key(module_key)
    invocation_id = _tensor_key(invocation_id)
    module = _module(module_key)
    weight, bias, state = _RESOLVE(
        module,
        exemplar,
        code_to_dtype(dtype_code),
        code_to_dtype(bias_dtype_code),
        code_to_dtype(compute_dtype_code),
        want_requant,
        prefetch_state=_consume_prefetch(module_key, invocation_id),
    )
    weight = _custom_op_return(
        weight, exemplar, weight_shape, prefetch_token, module_key, invocation_id
    )
    _ACTIVE[(module_key, invocation_id)] = (weight, bias, state)
    return weight


@resolve_prefetched_weight.register_fake
def _resolve_prefetched_weight_fake(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    prefetch_token: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    return _fake_weight(weight_shape, exemplar, _concrete_int(dtype_code))


@resolve_weight.register_fake
def _resolve_weight_fake(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> torch.Tensor:
    return _fake_weight(weight_shape, exemplar, _concrete_int(dtype_code))


@torch.library.custom_op(
    "comfy_weight::resolve_weight_bias",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def resolve_weight_bias(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    bias_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _RESOLVE is None:
        raise RuntimeError("comfy_weight resolve callback is not installed")
    module_key = _tensor_key(module_key)
    invocation_id = _tensor_key(invocation_id)
    module = _module(module_key)
    weight, bias, state = _RESOLVE(
        module,
        exemplar,
        code_to_dtype(dtype_code),
        code_to_dtype(bias_dtype_code),
        code_to_dtype(compute_dtype_code),
        want_requant,
    )
    if bias is None:
        raise RuntimeError(f"Module {type(module).__name__} has no bias")
    weight = _custom_op_return(weight, exemplar, weight_shape, bias_shape, module_key, invocation_id)
    bias = _custom_op_return(bias, exemplar, weight_shape, bias_shape, module_key, invocation_id)
    _ACTIVE[(module_key, invocation_id)] = (weight, bias, state)
    return weight, bias


@torch.library.custom_op(
    "comfy_weight::resolve_prefetched_weight_bias",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def resolve_prefetched_weight_bias(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    bias_shape: torch.Tensor,
    prefetch_token: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _RESOLVE is None:
        raise RuntimeError("comfy_weight resolve callback is not installed")
    module_key = _tensor_key(module_key)
    invocation_id = _tensor_key(invocation_id)
    module = _module(module_key)
    weight, bias, state = _RESOLVE(
        module,
        exemplar,
        code_to_dtype(dtype_code),
        code_to_dtype(bias_dtype_code),
        code_to_dtype(compute_dtype_code),
        want_requant,
        prefetch_state=_consume_prefetch(module_key, invocation_id),
    )
    if bias is None:
        raise RuntimeError(f"Module {type(module).__name__} has no bias")
    weight = _custom_op_return(
        weight, exemplar, weight_shape, bias_shape, prefetch_token, module_key, invocation_id
    )
    bias = _custom_op_return(
        bias, exemplar, weight_shape, bias_shape, prefetch_token, module_key, invocation_id
    )
    _ACTIVE[(module_key, invocation_id)] = (weight, bias, state)
    return weight, bias


@resolve_prefetched_weight_bias.register_fake
def _resolve_prefetched_weight_bias_fake(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    bias_shape: torch.Tensor,
    prefetch_token: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _fake_weight(weight_shape, exemplar, _concrete_int(dtype_code)),
        _fake_bias(bias_shape, exemplar, _concrete_int(bias_dtype_code or dtype_code)),
    )


@resolve_weight_bias.register_fake
def _resolve_weight_bias_fake(
    exemplar: torch.Tensor,
    weight_shape: torch.Tensor,
    bias_shape: torch.Tensor,
    module_key: int,
    invocation_id: int,
    dtype_code: int,
    bias_dtype_code: int,
    compute_dtype_code: int,
    want_requant: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _fake_weight(weight_shape, exemplar, _concrete_int(dtype_code)),
        _fake_bias(bias_shape, exemplar, _concrete_int(bias_dtype_code or dtype_code)),
    )


_LIB = torch.library.Library("comfy_weight", "FRAGMENT")
_LIB.define(
    "release_(Tensor(a!) output, int module_key, int invocation_id) -> ()",
    tags=(torch.Tag.cudagraph_unsafe, torch.Tag.maybe_aliasing_or_mutating),
)


def release_(output: torch.Tensor, module_key: int, invocation_id: int) -> None:
    if _RELEASE is None:
        raise RuntimeError("comfy_weight release callback is not installed")
    module_key = _tensor_key(module_key)
    invocation_id = _tensor_key(invocation_id)
    module = _module(module_key)
    weight, bias, state = _ACTIVE.pop((module_key, invocation_id), (None, None, None))
    if weight is not None:
        _RELEASE(module, weight, bias, state)
    return None


def _release_fake(output: torch.Tensor, module_key: int, invocation_id: int) -> None:
    return None


_LIB.impl("release_", release_, "CompositeExplicitAutograd")
_LIB.impl("release_", _release_fake, "Meta")
