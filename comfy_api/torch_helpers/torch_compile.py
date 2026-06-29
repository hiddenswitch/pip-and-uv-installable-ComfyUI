from __future__ import annotations

from comfy.cmd.main_pre import tracer

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

import comfy.utils
import comfy.weight_cast as weight_cast
from comfy.weight_cast_schedule import wrap_backend_with_weight_prefetch_scheduler
from comfy.weight_cast import get_materialization_spec
from comfy.weight_cast_ops import (
    module_bias_shape,
    module_weight_shape,
    register_module,
    register_module_with_stable_key,
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


_CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")


def _set_dynamo_config_if_present(name: str, value: object) -> None:
    try:
        setattr(torch._dynamo.config, name, value)
    except AttributeError:
        logger.debug("Skipping unsupported torch._dynamo.config.%s", name)


def _mode_wants_cudagraphs(mode: Optional[str]) -> bool:
    if not mode:
        return False
    if "no-cudagraphs" in mode:
        return False
    return any(mode.startswith(prefix) for prefix in _CUDAGRAPH_MODES)


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


def _is_graceful_compile_disable_error(exc: BaseException) -> bool:
    return _is_unsupported_fp8e4nv_compile_error(exc)


def _make_module_tensors_contiguous(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if not param.is_contiguous():
                param.data = param.detach().contiguous()
        for buffer in module.buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.detach().contiguous()


def _stabilize_comfy_weight_cast_attrs(module: torch.nn.Module, *, force_graph_visible_cast: bool = False) -> None:
    for child in module.modules():
        if hasattr(child, "comfy_cast_weights") and "comfy_cast_weights" not in child.__dict__:
            child.comfy_cast_weights = bool(child.comfy_cast_weights)
        if force_graph_visible_cast and hasattr(child, "comfy_cast_weights"):
            child.comfy_cast_weights = True
        if hasattr(child, "weight_function") and "weight_function" not in child.__dict__:
            child.weight_function = list(child.weight_function)
        if hasattr(child, "bias_function") and "bias_function" not in child.__dict__:
            child.bias_function = list(child.bias_function)
        if hasattr(child, "comfy_cast_weights"):
            get_materialization_spec(child)
            child._comfy_weight_cast_key = register_module(child)
            if hasattr(child, "weight") and child.weight is not None:
                child._comfy_weight_cast_weight_shape = module_weight_shape(child)
            child._comfy_weight_cast_bias_shape = module_bias_shape(child)


def _mark_cudagraph_step_begin() -> None:
    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark_step_begin is not None:
        mark_step_begin()


def _module_has_dynamic_vbar(module: torch.nn.Module) -> bool:
    return any(getattr(child, "_v", None) is not None for child in module.modules())


def _module_has_cast_capable_weights(module: torch.nn.Module) -> bool:
    return any(hasattr(child, "comfy_cast_weights") for child in module.modules())


def _model_uses_dynamic_vram(model: ModelPatcher, module: torch.nn.Module) -> bool:
    is_dynamic = getattr(model, "is_dynamic", None)
    if callable(is_dynamic):
        try:
            if is_dynamic():
                return True
        except Exception:
            pass
    return _module_has_dynamic_vbar(module)


def _model_needs_graph_visible_weight_cast(model: ModelPatcher, module: torch.nn.Module) -> bool:
    if _model_uses_dynamic_vram(model, module):
        return True
    return (
        weight_cast.graph_visible_backend_unavailable_reason() is None
        and _module_has_cast_capable_weights(module)
    )


def _first_tensor_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.device | None:
    values = list(args) + list(kwargs.values())
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict):
            device = _first_tensor_device((), value)
            if device is not None:
                return device
        if isinstance(value, (list, tuple)):
            device = _first_tensor_device(tuple(value), {})
            if device is not None:
                return device
    return None


_CONDITIONING_DYNAMIC_KEYS = frozenset((
    "context",
    "c_crossattn",
    "conditioning",
    "conditioning_to",
    "conditioning_from",
    "encoder_hidden_states",
    "attention_mask",
))


def _mark_dynamic_dim(tensor: torch.Tensor, dim: int, *, min_value: int = 1) -> None:
    if tensor.ndim <= dim:
        return
    if tensor.shape[dim] <= min_value:
        return
    try:
        maybe_mark_dynamic = getattr(torch._dynamo, "maybe_mark_dynamic", None)
        if maybe_mark_dynamic is not None:
            maybe_mark_dynamic(tensor, dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim, min=min_value)
    except Exception as exc:
        logger.debug(
            "Skipping torch dynamic dim hint for shape=%s dim=%s: %s",
            tuple(tensor.shape),
            dim,
            exc,
        )


def _mark_conditioning_sequence_dims(value: Any, *, key: str | None = None) -> None:
    if isinstance(value, torch.Tensor):
        if key in _CONDITIONING_DYNAMIC_KEYS:
            # Text/context tensors are typically [batch, seq, channels].
            # Attention masks are [batch, seq]. Keep latent/image spatial
            # dimensions specialized; only padded sequence length varies.
            _mark_dynamic_dim(value, 1)
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child_key_str = str(child_key)
            _mark_conditioning_sequence_dims(
                child_value,
                key=child_key_str if child_key_str in _CONDITIONING_DYNAMIC_KEYS else key,
            )
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _mark_conditioning_sequence_dims(child, key=key)


def _mark_compile_dynamic_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    # torch.compile traces lazily on first call, so mark runtime inputs just
    # before invoking the compiled callable. Keep latent spatial dimensions
    # static; only padded text/context sequence lengths should vary here.
    for key, value in kwargs.items():
        _mark_conditioning_sequence_dims(value, key=key)


def _stabilize_compile_parameter_residency(
    module: torch.nn.Module,
    device: torch.device | None,
    *,
    max_resident_bytes: int = 64 * 1024 * 1024,
) -> None:
    # Stable module keys keep compiled-graph caches deterministic on every
    # device; only the small-parameter relocation below needs an accelerator.
    move_params = device is not None and device.type != "cpu"
    with torch.no_grad():
        for module_name, child in module.named_modules():
            if not (
                hasattr(child, "comfy_cast_weights")
                or hasattr(child, "weight_function")
                or hasattr(child, "bias_function")
            ):
                continue
            if getattr(child, "_v", None) is not None:
                continue
            register_module_with_stable_key(
                child,
                _compile_module_identity(module_name, child),
            )
            if not move_params:
                continue
            for name in ("weight", "bias"):
                param = getattr(child, name, None)
                if param is None or not isinstance(param, torch.nn.Parameter):
                    continue
                if param.numel() * param.element_size() > max_resident_bytes:
                    continue
                if param.device == device:
                    continue
                moved = torch.nn.Parameter(param.detach().to(device=device), requires_grad=param.requires_grad)
                setattr(child, name, moved)


def _compile_module_identity(module_name: str, module: torch.nn.Module) -> str:
    parts: list[str] = [module_name, type(module).__module__, type(module).__qualname__]
    for name in ("weight", "bias"):
        param = getattr(module, name, None)
        if isinstance(param, torch.Tensor):
            parts.extend((name, str(tuple(param.shape)), str(param.dtype)))
    return "|".join(parts)


def _transformer_options_affect_model(transformer_options: Any) -> bool:
    if not transformer_options:
        return False
    if any(transformer_options.get(key) for key in _MODEL_TRANSFORMER_OPTION_KEYS):
        return True
    wrappers = transformer_options.get("wrappers", {})
    return bool(wrappers.get(WrappersMP.DIFFUSION_MODEL))


class _CompiledModel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compile_kwargs: dict[str, Any], *, force_graph_visible_cast: bool = False):
        super().__init__()
        _make_module_tensors_contiguous(module)
        _stabilize_comfy_weight_cast_attrs(module, force_graph_visible_cast=force_graph_visible_cast)
        self.compiled = torch.compile(model=module, **compile_kwargs)
        object.__setattr__(self, "_original", module)
        self._compile_disabled_reason: str | None = None
        self._logged_bypass_reasons: set = set()
        self._forward_calls = 0

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
            bypass_key = (self._compile_disabled_reason is not None, tuple(dynamic_reasons))
            if bypass_key not in self._logged_bypass_reasons:
                self._logged_bypass_reasons.add(bypass_key)
                logger.info(
                    "Bypassing torch.compile wrapper (running eager): disabled=%s dynamic_reasons=%s",
                    self._compile_disabled_reason is not None,
                    dynamic_reasons,
                )
            logger.debug(
                "Bypassing torch.compile wrapper: disabled=%s dynamic_reasons=%s transformer_option_keys=%s",
                self._compile_disabled_reason is not None,
                dynamic_reasons,
                sorted(transformer_options.keys()) if isinstance(transformer_options, dict) else None,
            )
            return self._original(*args, **kwargs)
        self._forward_calls += 1
        with tracer.start_as_current_span("Torch Compile Forward") as span:
            span.set_attribute("module_class", type(self._original).__name__)
            span.set_attribute("forward_calls", self._forward_calls)
            device = _first_tensor_device(args, kwargs)
            if device is not None:
                span.set_attribute("device", str(device))
            with tracer.start_as_current_span("Torch Compile Stabilize Residency"):
                _stabilize_compile_parameter_residency(self._original, device)
            _mark_cudagraph_step_begin()
            reset_invocation_ids()
            _mark_compile_dynamic_inputs(args, kwargs)
            try:
                with tracer.start_as_current_span("Torch Compile Invoke"):
                    return self.compiled(*args, **kwargs)
            except Exception as exc:
                if not _is_graceful_compile_disable_error(exc):
                    raise
                logger.warning("Disabling torch.compile for this module after kernel-specific failure: %s", exc)
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
    _set_dynamo_config_if_present("allow_unspec_int_on_nn_module", True)
    _set_dynamo_config_if_present("force_parameter_static_shapes", False)
    _set_dynamo_config_if_present("force_nn_module_property_static_shapes", False)
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
        use_graph_visible = _model_needs_graph_visible_weight_cast(model, module)
        if (
            use_graph_visible
            and _mode_wants_cudagraphs(mode)
            and not _model_uses_dynamic_vram(model, module)
        ):
            # The user explicitly asked for a cudagraph compile mode and the
            # model's weights are resident (no dynamic VRAM prefetch to
            # schedule). Honor the requested mode with the plain strategy
            # instead of silently stripping cudagraphs: resident quantized
            # layers (int8 inline path) and cast-at-use layers both run
            # correctly under cudagraph trees.
            logger.info("Using plain compile strategy for %s to honor mode=%s on resident weights", key, mode)
            use_graph_visible = False
        if use_graph_visible:
            compiled_modules[key] = _CompiledModel(
                module,
                _with_weight_prefetch_scheduler(_without_cudagraphs(compile_kwargs)),
                force_graph_visible_cast=True,
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
