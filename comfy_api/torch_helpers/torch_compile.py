from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

import comfy.utils
from comfy.weight_cast_schedule import wrap_backend_with_weight_prefetch_scheduler
from comfy.weight_cast import get_materialization_spec
from comfy.weight_cast_ops import (
    module_bias_shape_tensor,
    module_weight_shape_tensor,
    register_module,
    reset_invocation_ids,
)
from comfy.patcher_extension import WrappersMP

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor

COMPILE_KEY = "torch.compile"
logger = logging.getLogger(__name__)
TORCH_COMPILE_KWARGS = "torch_compile_kwargs"
TORCH_COMPILE_STRATEGY = "torch_compile_strategy"
_MODEL_TRANSFORMER_OPTION_KEYS = frozenset((
    "optimized_attention_override",
    "patches",
    "patches_replace",
    "rope_options",
))
_FP8E4NV_UNSUPPORTED = "fp8e4nv"


def apply_torch_compile_factory(compiled_module_dict: dict[str, Callable]) -> Callable:
    '''
    Create a wrapper that will refer to the compiled_diffusion_model.
    '''

    def apply_torch_compile_wrapper(executor: WrapperExecutor, *args, **kwargs):
        try:
            orig_modules = {}
            for key, value in compiled_module_dict.items():
                orig_modules[key] = comfy.utils.get_attr(executor.class_obj, key)
                comfy.utils.set_attr(executor.class_obj, key, value)
            return executor(*args, **kwargs)
        finally:
            for key, value in orig_modules.items():
                comfy.utils.set_attr(executor.class_obj, key, value)

    return apply_torch_compile_wrapper


def _compile_kwargs(
    *,
    backend: str,
    options: Optional[dict[str, Any]] = None,
    mode: Optional[str] = None,
    fullgraph: Optional[bool] = False,
    dynamic: Optional[bool] = None,
) -> dict[str, Any]:
    compile_kwargs: dict[str, Any] = {"backend": backend}
    if options is not None:
        compile_kwargs["options"] = options
    elif mode is not None:
        compile_kwargs["mode"] = mode
    if fullgraph is not None:
        compile_kwargs["fullgraph"] = fullgraph
    if dynamic is not None:
        compile_kwargs["dynamic"] = dynamic
    return compile_kwargs


def _without_cudagraphs(compile_kwargs: dict[str, Any]) -> dict[str, Any]:
    graph_kwargs = dict(compile_kwargs)
    options = dict(graph_kwargs.pop("options", {}) or {})
    graph_kwargs.pop("mode", None)
    options["triton.cudagraphs"] = False
    options["triton.cudagraph_trees"] = False
    graph_kwargs["options"] = options
    if graph_kwargs.get("dynamic") is None:
        graph_kwargs["dynamic"] = True
    return graph_kwargs


def _with_weight_prefetch_scheduler(compile_kwargs: dict[str, Any]) -> dict[str, Any]:
    scheduled_kwargs = dict(compile_kwargs)
    scheduled_kwargs["backend"] = wrap_backend_with_weight_prefetch_scheduler(scheduled_kwargs["backend"])
    return scheduled_kwargs


def _is_unsupported_fp8e4nv_compile_error(exc: BaseException) -> bool:
    return _FP8E4NV_UNSUPPORTED in str(exc)


def _make_module_tensors_contiguous(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if not param.is_contiguous():
                param.data = param.detach().contiguous()
        for buffer in module.buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.detach().contiguous()


def _stabilize_comfy_weight_cast_attrs(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "comfy_cast_weights") and "comfy_cast_weights" not in child.__dict__:
            child.comfy_cast_weights = bool(child.comfy_cast_weights)
        if hasattr(child, "weight_function") and "weight_function" not in child.__dict__:
            child.weight_function = list(child.weight_function)
        if hasattr(child, "bias_function") and "bias_function" not in child.__dict__:
            child.bias_function = list(child.bias_function)
        if hasattr(child, "comfy_cast_weights"):
            get_materialization_spec(child)
            child._comfy_weight_cast_key = register_module(child)
            if hasattr(child, "weight") and child.weight is not None:
                child._comfy_weight_cast_weight_shape_tensor = module_weight_shape_tensor(child)
            child._comfy_weight_cast_bias_shape_tensor = module_bias_shape_tensor(child)


def _mark_cudagraph_step_begin() -> None:
    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark_step_begin is not None:
        mark_step_begin()


def _module_has_dynamic_vbar(module: torch.nn.Module) -> bool:
    return any(getattr(child, "_v", None) is not None for child in module.modules())


def _transformer_options_affect_model(transformer_options: Any) -> bool:
    if not transformer_options:
        return False
    if any(transformer_options.get(key) for key in _MODEL_TRANSFORMER_OPTION_KEYS):
        return True
    wrappers = transformer_options.get("wrappers", {})
    return bool(wrappers.get(WrappersMP.DIFFUSION_MODEL))


class _CompiledModel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compile_kwargs: dict[str, Any]):
        super().__init__()
        _make_module_tensors_contiguous(module)
        _stabilize_comfy_weight_cast_attrs(module)
        self.compiled = torch.compile(model=module, **compile_kwargs)
        object.__setattr__(self, "_original", module)
        self._compile_disabled_reason: str | None = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            original = self.__dict__.get("_original")
            if original is None:
                raise
            return getattr(original, name)

    def forward(self, *args, **kwargs):
        transformer_options = kwargs.get("transformer_options")
        dynamic_reasons = []
        if kwargs.get("control") is not None:
            dynamic_reasons.append("control")
        if kwargs.get("ref_latents") is not None:
            dynamic_reasons.append("ref_latents")
        if kwargs.get("attention_mask") is not None:
            dynamic_reasons.append("attention_mask")
        if _transformer_options_affect_model(transformer_options):
            dynamic_reasons.append("transformer_options")
        has_dynamic_model_options = (
            len(dynamic_reasons) > 0
        )
        if (
            self._compile_disabled_reason is not None
            or has_dynamic_model_options
        ):
            logger.debug(
                "Bypassing torch.compile wrapper: disabled=%s dynamic_reasons=%s transformer_option_keys=%s",
                self._compile_disabled_reason is not None,
                dynamic_reasons,
                sorted(transformer_options.keys()) if isinstance(transformer_options, dict) else None,
            )
            return self._original(*args, **kwargs)
        _mark_cudagraph_step_begin()
        reset_invocation_ids()
        try:
            return self.compiled(*args, **kwargs)
        except Exception as exc:
            if not _is_unsupported_fp8e4nv_compile_error(exc):
                raise
            self._compile_disabled_reason = str(exc)
            return self._original(*args, **kwargs)


def set_torch_compile_wrapper(model: ModelPatcher, backend: str, options: Optional[dict[str, Any]] = None,
                              mode: Optional[str] = None, fullgraph=False, dynamic: Optional[bool] = None,
                              keys: list[str] = None, *args, **kwargs):
    '''
    Perform torch.compile that will be applied at sample time for either the whole model or specific params of the BaseModel instance.

    When keys is None, it will default to using ["diffusion_model"], compiling the whole diffusion_model.
    When a list of keys is provided, it will perform torch.compile on only the selected modules.
    '''
    logger.debug("set_torch_compile_wrapper called for %s backend=%s keys=%s", type(model).__name__, backend, keys)
    # clear out any other torch.compile wrappers
    model.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY)
    # if no keys, default to 'diffusion_model'
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.force_nn_module_property_static_shapes = False
    if not keys:
        keys = ["diffusion_model"]
    # create kwargs dict that can be referenced later
    compile_kwargs = _compile_kwargs(
        backend=backend,
        options=options,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    # get a dict of compiled keys
    compiled_modules = {}
    compiled_strategies = {}
    for key in keys:
        module = model.get_model_object(key)
        _stabilize_comfy_weight_cast_attrs(module)
        if _module_has_dynamic_vbar(module):
            compiled_modules[key] = _CompiledModel(
                module,
                _with_weight_prefetch_scheduler(_without_cudagraphs(compile_kwargs)),
            )
            compiled_strategies[key] = "module_weight_cast"
            continue
        compiled_modules[key] = _CompiledModel(module, compile_kwargs)
        compiled_strategies[key] = "module"
    if compiled_modules:
        # add torch.compile wrapper
        wrapper_func = apply_torch_compile_factory(
            compiled_module_dict=compiled_modules,
        )
        # store wrapper to run on BaseModel's apply_model function
        model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY, wrapper_func)
    # keep compile kwargs for reference
    model.model_options[TORCH_COMPILE_KWARGS] = compile_kwargs
    model.model_options[TORCH_COMPILE_STRATEGY] = compiled_strategies
