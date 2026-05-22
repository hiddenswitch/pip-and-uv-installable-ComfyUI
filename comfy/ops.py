"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import logging
import os
import collections
from typing import Optional

import comfy_aimdo.model_vbar
import comfy_aimdo.torch
import torch
import typing
from torch import Tensor

from . import float as comfy_float
from . import memory_management
from . import model_management, rmsnorm
from . import pinned_memory
from . import weight_cast
from . import weight_cast_ops
from . import utils
from .cli_args import args, PerformanceFeature
from .execution_context import current_execution_context
from .interruption import throw_exception_if_processing_interrupted

logger = logging.getLogger(__name__)

_DYNAMIC_VRAM_FP8_POLICIES = {"auto", "resident", "materialize"}
_CPU_MATERIALIZATION_BUFFERS = collections.OrderedDict()
_CPU_MATERIALIZATION_BUFFER_LIMIT = 4


def dynamic_vram_fp8_policy():
    policy = os.environ.get("COMFY_DYNAMIC_VRAM_FP8_POLICY", "auto").strip().lower()
    if policy not in _DYNAMIC_VRAM_FP8_POLICIES:
        logger.warning("Unknown COMFY_DYNAMIC_VRAM_FP8_POLICY=%r; using auto", policy)
        return "auto"
    return policy


def dynamic_vram_diag_enabled():
    return os.environ.get("COMFY_DYNAMIC_VRAM_DIAG", "").strip().lower() in {"1", "true", "yes", "on"}


def direct_materialize_pinning_enabled():
    return os.environ.get("COMFY_DIRECT_MATERIALIZE_PINNING", "").strip().lower() in {"1", "true", "yes", "on"}


def _tensor_diag(tensor):
    if tensor is None:
        return "None"
    if isinstance(tensor, QuantizedTensor):
        qdata = getattr(tensor, "_qdata", None)
        params = getattr(tensor, "_params", None)
        param_parts = []
        if params is not None:
            for name in ("scale", "block_scale"):
                value = getattr(params, name, None)
                if isinstance(value, torch.Tensor):
                    param_parts.append(f"{name}={tuple(value.shape)}/{value.dtype}/{value.device}")
            orig_dtype = getattr(params, "orig_dtype", None)
            if orig_dtype is not None:
                param_parts.append(f"orig_dtype={orig_dtype}")
        return (
            f"QuantizedTensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"qdata={tuple(qdata.shape) if isinstance(qdata, torch.Tensor) else None}/"
            f"{getattr(qdata, 'dtype', None)}/{getattr(qdata, 'device', None)}, "
            f"layout={getattr(tensor, '_layout_cls', None)}, {', '.join(param_parts)})"
        )
    return f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"


def _vbar_diag(vbar):
    try:
        residency = vbar.get_residency()
        resident_pages = sum(1 for page in residency if page & 1)
        pinned_pages = sum(1 for page in residency if page & 2)
        return (
            f"loaded={vbar.loaded_size()} watermark={vbar.get_watermark()} "
            f"pages={len(residency)} resident_pages={resident_pages} pinned_pages={pinned_pages}"
        )
    except Exception as exc:
        return f"unavailable={exc}"


def _cuda_mem_diag(device):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return "cuda=unavailable"
    try:
        free, total = torch.cuda.mem_get_info(device)
        return (
            f"cuda_free={free} cuda_total={total} "
            f"torch_alloc={torch.cuda.memory_allocated(device)} "
            f"torch_reserved={torch.cuda.memory_reserved(device)}"
        )
    except Exception as exc:
        return f"cuda=unavailable:{exc}"

_RUN_EVERY_OP_ENABLED = model_management.torch_version_numeric >= (2, 5)


def run_every_op():
    global _RUN_EVERY_OP_ENABLED
    # this is not available on torch 2.3 by testing
    if not _RUN_EVERY_OP_ENABLED or torch.compiler.is_compiling():
        return

    throw_exception_if_processing_interrupted()


def _scaled_dot_product_attention(q, k, v, *args, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


scaled_dot_product_attention = _scaled_dot_product_attention

# Track if cuDNN attention should be disabled due to runtime errors
_cudnn_attention_disabled = False


def _check_cudnn_nvrtc_compatibility():
    """Check if cuDNN attention is likely to work based on CUDA version compatibility.

    cuDNN attention uses nvrtc (NVIDIA Runtime Compilation) which requires
    the system nvrtc version to match the CUDA runtime version. A mismatch
    will cause runtime errors like:
    'major version mismatch between cudart and nvrtc'
    """
    try:
        # Get PyTorch's CUDA version
        pytorch_cuda = torch.version.cuda
        if pytorch_cuda is None:
            return False

        pytorch_cuda_major = int(pytorch_cuda.split('.')[0])

        # Try to detect system CUDA/nvrtc version mismatch
        # This is a heuristic - if cuDNN version doesn't align with expectations
        # for the PyTorch CUDA version, there may be issues
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None:
            return False

        # cuDNN 9.x requires CUDA 12.x or 13.x
        # If there's a significant version gap, there may be compatibility issues
        cudnn_major = cudnn_version // 10000

        # cuDNN 9 works with CUDA 12+
        if cudnn_major >= 9 and pytorch_cuda_major >= 12:
            return True

        return False
    except Exception:
        return False


try:
    if torch.cuda.is_available():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import inspect

        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            SDPA_BACKEND_PRIORITY = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

            # Only add cuDNN attention if compatibility check passes
            if _check_cudnn_nvrtc_compatibility():
                SDPA_BACKEND_PRIORITY.insert(0, SDPBackend.CUDNN_ATTENTION)
            else:
                logger.debug("Skipping cuDNN attention backend due to potential version incompatibility")


            def _scaled_dot_product_attention_sdpa2(q, k, v, *args, **kwargs):
                global _cudnn_attention_disabled
                try:
                    if q.nelement() < 1024 * 128:  # arbitrary number, for small inputs cudnn attention seems slower
                        return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                    with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                        return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                except RuntimeError as e:
                    # Check for cuDNN-specific errors and fall back
                    error_msg = str(e)
                    if "cuDNN" in error_msg or "cudnn" in error_msg.lower() or "nvrtc" in error_msg.lower():
                        if not _cudnn_attention_disabled:
                            logger.warning(f"cuDNN attention failed, falling back to other backends: {error_msg}")
                            _cudnn_attention_disabled = True
                        # Try without cuDNN
                        fallback_priority = [b for b in SDPA_BACKEND_PRIORITY if b != SDPBackend.CUDNN_ATTENTION]
                        with sdpa_kernel(fallback_priority, set_priority=True):
                            return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                    raise


            scaled_dot_product_attention = _scaled_dot_product_attention_sdpa2
        else:
            logger.warning("Torch version too old to set sdpa backend priority, even though you are using CUDA")
except Exception as exc_info:
    if torch.cuda.is_available():
        logger.debug("Could not set sdpa backend priority.", exc_info=exc_info)

NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False
try:
    if model_management.is_nvidia():
        cudnn_version = torch.backends.cudnn.version()
        if (cudnn_version >= 91002 and cudnn_version < 91500) and model_management.torch_version_numeric >= (2, 9) and model_management.torch_version_numeric <= (2, 10):
            # TODO: change upper bound version once it's fixed'
            NVIDIA_MEMORY_CONV_BUG_WORKAROUND = True
            logger.debug("working around nvidia conv3d memory bug.")
except:
    pass

cast_to = model_management.cast_to  # TODO: remove once no more references

_DEFERRED_VBAR_UNPINS = []


def cast_to_input(weight, input, non_blocking=False, copy=True):
    return model_management.cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)


def materialize_meta_param(s, param_keys):
    for param_key in param_keys:
        param = getattr(s, param_key, None)
        if param is not None and getattr(param, "is_meta", False):
            setattr(s, param_key, torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype), requires_grad=param.requires_grad))


def _defer_vbar_unpin(alloc, device):
    if device is None or device.type != "cuda":
        comfy_aimdo.model_vbar.vbar_unpin(alloc)
        return
    event = torch.cuda.Event()
    event.record(model_management.current_stream(device))
    _DEFERRED_VBAR_UNPINS.append((event, alloc))


def _drain_deferred_vbar_unpins(block=False):
    if not _DEFERRED_VBAR_UNPINS:
        return
    pending = []
    for event, alloc in _DEFERRED_VBAR_UNPINS:
        if block:
            event.synchronize()
            comfy_aimdo.model_vbar.vbar_unpin(alloc)
        elif event.query():
            comfy_aimdo.model_vbar.vbar_unpin(alloc)
        else:
            pending.append((event, alloc))
    _DEFERRED_VBAR_UNPINS[:] = pending


# FIXME: add n=1 cache hit fast path
def cast_modules_with_vbar(comfy_modules, dtype, device, bias_dtype, non_blocking, want_requant=False, dedicated_buffer=False, prefetch_hint=False):
    offload_stream = None
    cast_buffer = None
    cast_buffer_offset = 0

    def ensure_offload_stream(module, required_size, check_largest):
        nonlocal offload_stream
        nonlocal cast_buffer

        if offload_stream is None:
            offload_stream = model_management.get_offload_stream(device)
        if offload_stream is None or not check_largest or len(comfy_modules) != 1:
            return

        current_size = 0 if cast_buffer is None else cast_buffer.size()
        if current_size < required_size and module is model_management.LARGEST_AIMDO_CASTED_WEIGHT[0]:
            offload_stream = model_management.get_offload_stream(device)
            cast_buffer = None
        if required_size > model_management.LARGEST_AIMDO_CASTED_WEIGHT[1]:
            model_management.LARGEST_AIMDO_CASTED_WEIGHT = (module, required_size)

    def get_cast_buffer(buffer_size, reclaim_vbar=None):
        nonlocal offload_stream
        nonlocal cast_buffer
        nonlocal cast_buffer_offset

        if buffer_size == 0:
            return None

        if dedicated_buffer:
            return torch.empty((buffer_size,), dtype=torch.uint8, device=device)

        if offload_stream is None:
            return torch.empty((buffer_size,), dtype=torch.uint8, device=device)

        cast_buffer = model_management.get_aimdo_cast_buffer(offload_stream, device)
        try:
            alloc = cast_buffer.get(buffer_size, cast_buffer_offset)
        except RuntimeError as exc:
            logger.debug("Falling back to torch cast buffer after aimdo cast buffer allocation failed: %s", exc)
            if reclaim_vbar is not None:
                if dynamic_vram_diag_enabled():
                    logger.warning(
                        "DYNAMIC_VRAM_DIAG cast-buffer fallback before free: requested=%s offset=%s %s %s",
                        buffer_size,
                        cast_buffer_offset,
                        _vbar_diag(reclaim_vbar),
                        _cuda_mem_diag(device),
                    )
                freed = reclaim_vbar.free_memory(1e30)
                if dynamic_vram_diag_enabled():
                    logger.warning(
                        "DYNAMIC_VRAM_DIAG cast-buffer fallback after free: requested=%s freed=%s %s %s",
                        buffer_size,
                        freed,
                        _vbar_diag(reclaim_vbar),
                        _cuda_mem_diag(device),
                    )
                if freed < buffer_size:
                    model_management.free_memory(
                        buffer_size + 512 * 1024 ** 2,
                        device,
                        for_dynamic=True,
                    )
                    if dynamic_vram_diag_enabled():
                        logger.warning(
                            "DYNAMIC_VRAM_DIAG cast-buffer fallback after global free: requested=%s %s %s",
                            buffer_size,
                            _vbar_diag(reclaim_vbar),
                            _cuda_mem_diag(device),
                        )
            try:
                alloc = cast_buffer.get(buffer_size, cast_buffer_offset)
            except RuntimeError as retry_exc:
                logger.debug("Falling back to torch cast buffer after aimdo cast buffer retry failed: %s", retry_exc)
                buffer = torch.empty((buffer_size,), dtype=torch.uint8, device=device)
            else:
                buffer = comfy_aimdo.torch.aimdo_to_tensor(alloc, device)
        else:
            buffer = comfy_aimdo.torch.aimdo_to_tensor(alloc, device)
        cast_buffer_offset += buffer_size
        return buffer

    def target_geometry_for(tensor, target_dtype):
        if tensor is None:
            return None
        if target_dtype is None:
            return tensor
        if isinstance(tensor, QuantizedTensor) and want_requant and should_keep_quantized_vbar(s, tensor):
            return tensor
        return model_management.tensor_materialization_geometry(tensor, dtype=target_dtype)

    for s in comfy_modules:
        _drain_deferred_vbar_unpins(block=False)
        fault_failed = False
        if dynamic_vram_diag_enabled():
            logger.warning(
                "DYNAMIC_VRAM_DIAG before fault module=%s alloc=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s weight=%s bias=%s %s %s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
                _tensor_diag(s.weight),
                _tensor_diag(s.bias),
                _vbar_diag(s._v[0]),
                _cuda_mem_diag(device),
            )
        try:
            signature = comfy_aimdo.model_vbar.vbar_fault(s._v)
        except RuntimeError as exc:
            logger.debug(
                "Dynamic VBAR fault failed for %s; using temporary cast buffer: %s",
                getattr(s, "seed_key", type(s).__name__),
                exc,
            )
            signature = None
            fault_failed = True
        if signature is None and not prefetch_hint:
            _drain_deferred_vbar_unpins(block=True)
            try:
                signature = comfy_aimdo.model_vbar.vbar_fault(s._v)
                fault_failed = False
            except RuntimeError as exc:
                logger.debug(
                    "Dynamic VBAR fault retry failed for %s; using temporary cast buffer: %s",
                    getattr(s, "seed_key", type(s).__name__),
                    exc,
                )
                signature = None
                fault_failed = True
        if signature is None:
            try:
                comfy_aimdo.model_vbar.vbar_unpin(s._v)
            except Exception as exc:
                if fault_failed:
                    logger.debug("Dynamic VBAR unpin after failed fault failed: %s", exc)
        if dynamic_vram_diag_enabled():
            logger.warning(
                "DYNAMIC_VRAM_DIAG after fault module=%s signature=%s fault_failed=%s %s %s",
                getattr(s, "seed_key", type(s).__name__),
                signature is not None,
                fault_failed,
                _vbar_diag(s._v[0]),
                _cuda_mem_diag(device),
            )
        if signature is None and prefetch_hint:
            logger.debug(
                "Dynamic VBAR prefetch hint deferred for %s: allocated=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
            )
            s._prefetch = None
            continue
        resident = comfy_aimdo.model_vbar.vbar_signature_compare(signature, s._v_signature)
        prefetch = {
            "signature": signature,
            "resident": resident,
        }

        if resident:
            s._prefetch = prefetch
            continue

        materialize_meta_param(s, ["weight", "bias"])
        xfer_dest = comfy_aimdo.torch.aimdo_to_tensor(s._v, device) if signature is not None else None
        if signature is None:
            logger.debug(
                "Dynamic VBAR fault returned no signature for %s: allocated=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
            )
        source_geometry = [
            model_management.tensor_materialization_geometry(s.weight),
            model_management.tensor_materialization_geometry(s.bias),
        ]
        cast_geometry = [
            target_geometry_for(s.weight, dtype),
            target_geometry_for(s.bias, bias_dtype),
        ]
        cast_dest = None
        needs_cast = False
        direct_materialize = cast_geometry != source_geometry
        has_patch_functions = len(getattr(s, "weight_function", [])) > 0 or len(getattr(s, "bias_function", [])) > 0

        xfer_source = [s.weight, s.bias]

        use_pin = not direct_materialize or (direct_materialize_pinning_enabled() and not has_patch_functions)
        pin_geometry = cast_geometry if direct_materialize else None
        pin = pinned_memory.get_pin(s, pin_geometry) if use_pin else None
        raw_pin_transfer = False
        if pin is not None and not direct_materialize:
            xfer_source = [pin]

        if not direct_materialize:
            for data, geometry in zip([s.weight, s.bias], cast_geometry):
                if data is None:
                    continue
                if data.dtype != geometry.dtype:
                    needs_cast = True
                    cast_dest = xfer_dest
                    xfer_dest = None
                    break

        dest_geometry = cast_geometry if direct_materialize else xfer_source
        dest_size = memory_management.vram_aligned_size(dest_geometry)
        if dynamic_vram_diag_enabled():
            logger.warning(
                "DYNAMIC_VRAM_DIAG geometry module=%s source_geometry=%r cast_geometry=%r direct_materialize=%s needs_cast=%s dest_size=%s allocated=%s pin=%s weight=%s bias=%s",
                getattr(s, "seed_key", type(s).__name__),
                source_geometry,
                cast_geometry,
                direct_materialize,
                needs_cast,
                dest_size,
                s._v[2],
                pin is not None,
                _tensor_diag(s.weight),
                _tensor_diag(s.bias),
            )
        if xfer_dest is not None and dest_size > s._v[2]:
            logger.debug(
                "Dynamic VBAR allocation too small for %s: allocated=%s requested=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dest_size,
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
            )
            xfer_dest = None
            signature = None
            prefetch["signature"] = None
            if prefetch_hint:
                s._prefetch = None
                continue
        ensure_offload_stream(s, dest_size if xfer_dest is None else 0, True)
        if xfer_dest is None:
            reclaim_vbar = s._v[0] if signature is None else None
            xfer_dest = get_cast_buffer(dest_size, reclaim_vbar=reclaim_vbar)

        if signature is not None or not use_pin:
            pin = None
        elif pin is None:
            pinned_memory.pin_memory(s, pin_geometry)
            pin = pinned_memory.get_pin(s, pin_geometry)

        if pin is not None:
            pinned_memory.wait_pin_ready(s, pin_geometry)
            model_management.cast_to_gathered(
                xfer_source,
                pin,
                target_geometries=cast_geometry if direct_materialize else None,
            )
            if direct_materialize:
                xfer_source = memory_management.interpret_gathered_like(cast_geometry, pin)
                raw_pin_transfer = True
            else:
                xfer_source = [ pin ]
        #send it over
        model_management.cast_to_gathered(
            xfer_source,
            xfer_dest,
            non_blocking=non_blocking and not raw_pin_transfer,
            stream=offload_stream,
            target_geometries=cast_geometry if direct_materialize and not raw_pin_transfer else None,
        )
        if pin is not None:
            pinned_memory.record_pin_use(s, pin_geometry, offload_stream, device)

        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                ensure_offload_stream(s, cast_buffer_offset, False)
                lowvram_fn.prepare(lambda size: get_cast_buffer(size), offload_stream)

        prefetch["xfer_dest"] = xfer_dest
        prefetch["cast_dest"] = cast_dest
        prefetch["cast_geometry"] = cast_geometry
        prefetch["needs_cast"] = needs_cast
        s._prefetch = prefetch

    return offload_stream


def _cpu_materialization_buffer(shape, dtype):
    key = (tuple(int(dim) for dim in shape), dtype)
    buffer = _CPU_MATERIALIZATION_BUFFERS.pop(key, None)
    if buffer is None:
        buffer = torch.empty(key[0], dtype=dtype, device="cpu")
    _CPU_MATERIALIZATION_BUFFERS[key] = buffer
    while len(_CPU_MATERIALIZATION_BUFFERS) > _CPU_MATERIALIZATION_BUFFER_LIMIT:
        _CPU_MATERIALIZATION_BUFFERS.popitem(last=False)
    return buffer


def _copy_fp8_to_cpu_materialization_buffer(tensor, dtype):
    qdata = getattr(tensor, "_qdata", None)
    params = getattr(tensor, "_params", None)
    scale = getattr(params, "scale", None)
    orig_shape = tuple(getattr(params, "orig_shape", tuple(tensor.shape)))
    if (
        not isinstance(qdata, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or scale.numel() != 1
        or qdata.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
    ):
        return None

    qdata_cpu = qdata if qdata.device.type == "cpu" else qdata.cpu()
    materialized = _cpu_materialization_buffer(qdata_cpu.shape, dtype)
    materialized.copy_(qdata_cpu)
    materialized.mul_(scale.cpu().to(dtype=dtype))
    if tuple(materialized.shape) != orig_shape:
        slices = tuple(slice(0, dim) for dim in orig_shape)
        materialized = materialized[slices]
    return materialized


def _materialize_quantized_tensor_on_cpu(tensor, dtype):
    materialized = _copy_fp8_to_cpu_materialization_buffer(tensor, dtype)
    if materialized is not None:
        return materialized
    tensor = tensor.to(device=torch.device("cpu")).to(dtype=dtype)
    return tensor.dequantize()


def _dequantize_quantized_tensor_on_cpu(tensor, dtype, device):
    return _materialize_quantized_tensor_on_cpu(tensor, dtype).to(device=device, non_blocking=True)


def resolve_cast_module_with_vbar(s, dtype, device, bias_dtype, compute_dtype, want_requant):

    prefetch = s._prefetch

    if prefetch["resident"]:
        weight = s._v_weight
        bias = s._v_bias
    else:
        xfer_dest = prefetch["xfer_dest"]
        if prefetch["needs_cast"]:
            cast_dest = prefetch["cast_dest"] if prefetch["cast_dest"] is not None else torch.empty((memory_management.vram_aligned_size(prefetch["cast_geometry"]),), dtype=torch.uint8, device=device)
            for pre_cast, post_cast in zip(memory_management.interpret_gathered_like([s.weight, s.bias ], xfer_dest),
                                           memory_management.interpret_gathered_like(prefetch["cast_geometry"], cast_dest)):
                if post_cast is not None:
                    post_cast.copy_(pre_cast)
            xfer_dest = cast_dest

        params = memory_management.interpret_gathered_like(prefetch["cast_geometry"], xfer_dest)
        weight = params[0]
        bias = params[1]
        if prefetch["signature"] is not None:
            s._v_weight = weight
            s._v_bias = bias
        s._v_signature = prefetch["signature"]

    def post_cast(s, param_key, x, dtype, resident, update_weight):
        lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
        fns = getattr(s, param_key + "_function", [])

        if x is None:
            return None

        orig = x

        def to_dequant(tensor, dtype):
            if isinstance(tensor, QuantizedTensor):
                qdata = getattr(tensor, "_qdata", None)
                if isinstance(qdata, torch.Tensor) and qdata.device.type != "cpu":
                    return _dequantize_quantized_tensor_on_cpu(tensor, dtype, device)
                return tensor.to(dtype=dtype).dequantize()
            return tensor.to(dtype=dtype)

        keep_quantized = want_requant and isinstance(x, QuantizedTensor) and len(fns) == 0
        if (not keep_quantized and orig.dtype != dtype) or len(fns) > 0 or (isinstance(x, QuantizedTensor) and not want_requant):
            x = to_dequant(x, dtype)
        if not resident and lowvram_fn is not None:
            x = to_dequant(x, dtype if compute_dtype is None else compute_dtype)
            x = lowvram_fn(x)
            if (want_requant and len(fns) == 0 or update_weight):
                seed = utils.string_to_seed(s.seed_key)
                if isinstance(orig, QuantizedTensor):
                    y = QuantizedTensor.from_float(x, s.layout_type, scale="recalculate", stochastic_rounding=seed)
                else:
                    y = comfy_float.stochastic_rounding(x, orig.dtype, seed=seed)
            if want_requant and len(fns) == 0:
                x = y
            if update_weight:
                orig.copy_(y)
        for f in fns:
            x = f(x)
        return x

    update_weight = prefetch["signature"] is not None
    weight = post_cast(s, "weight", weight, dtype, prefetch["resident"], update_weight)
    if bias is not None:
        bias = post_cast(s, "bias", bias, bias_dtype, prefetch["resident"], update_weight)

    if prefetch["signature"] is not None:
        prefetch["resident"] = True

    return weight, bias


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False, compute_dtype=None, want_requant=False):
    # NOTE: offloadable=False is a legacy and if you are a custom node author reading this please pass
    # offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias. This
    # will add async-offload support to your cast and improve performance.
    if input is not None:
        if dtype is None:
            if isinstance(input, QuantizedTensor):
                dtype = input.params.orig_dtype
            else:
                dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    def format_return(result, offloadable):
        weight, bias, offload_stream = result
        return (weight, bias, offload_stream) if offloadable else (weight, bias)

    non_blocking = model_management.device_supports_non_blocking(device)

    if s._v is not None:

        #vbar doesn't support CPU weights, but some custom nodes have weird paths
        #that might switch the layer to the CPU and expect it to work. We have to take
        #a clone conservatively as we are mmapped and some SFT files are packed misaligned
        #If you are a custom node author reading this, please move your layer to the GPU
        #or declare your ModelPatcher as CPU in the first place.
        if model_management.is_device_cpu(device):
            materialize_meta_param(s, ["weight", "bias"])
            weight = s.weight.to(dtype=dtype, copy=True)
            if isinstance(weight, QuantizedTensor):
                weight = weight.dequantize()
            bias = s.bias.to(dtype=bias_dtype, copy=True) if s.bias is not None else None
            return format_return((weight, bias, (None, None, None)), offloadable)

        prefetched = s._prefetch is not None
        offload_stream = None
        offload_device = None
        if not prefetched:
            offload_stream = cast_modules_with_vbar([s], dtype, device, bias_dtype, non_blocking, want_requant=want_requant)
            model_management.sync_stream(device, offload_stream)

        weight, bias = resolve_cast_module_with_vbar(s, dtype, device, bias_dtype, compute_dtype, want_requant)

        if not prefetched:
            if s._prefetch["signature"] is not None:
                offload_device = device
            for param_key in ("weight", "bias"):
                lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
                if lowvram_fn is not None:
                    lowvram_fn.clear_prepared()
            s._prefetch = None
        return format_return((weight, bias, (offload_stream, offload_device, None)), offloadable)


    weight_has_function = len(s.weight_function) > 0
    bias_has_function = len(s.bias_function) > 0
    keep_quantized_weight = want_requant and isinstance(s.weight, QuantizedTensor) and not weight_has_function
    materialize_weight_before_transfer = isinstance(s.weight, QuantizedTensor) and (
        weight_has_function or not keep_quantized_weight
    )
    materialize_bias_before_transfer = isinstance(s.bias, QuantizedTensor) if s.bias is not None else False

    source_weight = _materialize_quantized_tensor_on_cpu(s.weight, dtype) if materialize_weight_before_transfer else s.weight
    source_bias = _materialize_quantized_tensor_on_cpu(s.bias, bias_dtype) if materialize_bias_before_transfer else s.bias

    if offloadable and (device != source_weight.device or
                        (source_bias is not None and device != source_bias.device)):
        offload_stream = model_management.get_offload_stream(device)
    else:
        offload_stream = None

    bias = None
    weight = None

    if offload_stream is not None and not args.cuda_malloc:
        cast_buffer_size = memory_management.vram_aligned_size([source_weight, source_bias])
        cast_buffer = model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        # The streams can be uneven in buffer capability and reject us. Retry to get the other stream
        if cast_buffer is None:
            offload_stream = model_management.get_offload_stream(device)
            cast_buffer = model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        if cast_buffer is not None:
            params = memory_management.interpret_gathered_like([source_weight, source_bias], cast_buffer)
            weight = params[0]
            bias = params[1]

    weight = model_management.cast_to(source_weight, None, device, non_blocking=non_blocking, copy=weight_has_function, stream=offload_stream, r=weight)

    if source_bias is not None:
        bias = model_management.cast_to(source_bias, None, device, non_blocking=non_blocking, copy=bias_has_function, stream=offload_stream, r=bias)

    model_management.sync_stream(device, offload_stream)

    bias_a = bias
    weight_a = weight

    if source_bias is not None:
        if isinstance(bias, QuantizedTensor):
            qdata = getattr(bias, "_qdata", None)
            if isinstance(qdata, torch.Tensor) and qdata.device.type != "cpu":
                bias = _dequantize_quantized_tensor_on_cpu(bias, bias_dtype, device)
            else:
                bias = bias.to(dtype=bias_dtype).dequantize()
        else:
            bias = bias.to(dtype=bias_dtype)
        for f in s.bias_function:
            bias = f(bias)

    if weight_has_function or (not keep_quantized_weight and weight.dtype != dtype) or (isinstance(weight, QuantizedTensor) and not want_requant):
        if isinstance(weight, QuantizedTensor):
            qdata = getattr(weight, "_qdata", None)
            if isinstance(qdata, torch.Tensor) and qdata.device.type != "cpu":
                weight = _dequantize_quantized_tensor_on_cpu(weight, dtype, device)
            else:
                weight = weight.to(dtype=dtype).dequantize()
        else:
            weight = weight.to(dtype=dtype)
        for f in s.weight_function:
            weight = f(weight)

    return format_return((weight, bias, (offload_stream, weight_a, bias_a)), offloadable)


def uncast_bias_weight(s, weight, bias, offload_stream):
    if offload_stream is None:
        return
    os, weight_a, bias_a = offload_stream
    device = None
    # FIXME: This is really bad RTTI
    if weight_a is not None and not isinstance(weight_a, torch.Tensor):
        device = weight_a
        _defer_vbar_unpin(s._v, device)
    if os is None:
        return
    if device is None:
        if weight_a is not None:
            device = weight_a.device
        else:
            if bias_a is None:
                return
            device = bias_a.device
    os.wait_stream(model_management.current_stream(device))


def _legacy_weight_cast_prefetch(module, device, dtype, bias_dtype, compute_dtype, want_requant):
    non_blocking = device is not None and model_management.device_supports_non_blocking(device)
    if device is None or module._v is None or model_management.is_device_cpu(device):
        return None
    offload_stream = cast_modules_with_vbar(
        [module],
        dtype,
        device,
        bias_dtype,
        non_blocking,
        want_requant=want_requant,
        dedicated_buffer=True,
        prefetch_hint=False,
    )
    if module._prefetch is None:
        return None
    ready_event = None
    if offload_stream is not None and device is not None and device.type == "cuda":
        ready_event = torch.cuda.Event()
        ready_event.record(offload_stream)
    return offload_stream, device, ready_event


def _legacy_weight_cast_resolve(module, input, dtype, bias_dtype, compute_dtype, want_requant, prefetch_state=None):
    if prefetch_state is not None:
        offload_stream, device, ready_event = prefetch_state
        if ready_event is not None:
            model_management.current_stream(device).wait_event(ready_event)
        else:
            model_management.sync_stream(device, offload_stream)
        weight, bias = resolve_cast_module_with_vbar(module, dtype, device, bias_dtype, compute_dtype, want_requant)
        if module._prefetch["signature"] is not None:
            release_state = (offload_stream, device, None)
        else:
            release_state = (offload_stream, weight, bias)
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(module, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                lowvram_fn.clear_prepared()
        module._prefetch = None
        return weight, bias, release_state
    return cast_bias_weight(
        module,
        input,
        dtype=dtype,
        bias_dtype=bias_dtype,
        offloadable=True,
        compute_dtype=compute_dtype,
        want_requant=want_requant,
    )


def _legacy_weight_cast_release(module, weight, bias, token):
    uncast_bias_weight(module, weight, bias, token)


weight_cast_ops.set_callbacks(_legacy_weight_cast_resolve, _legacy_weight_cast_release, _legacy_weight_cast_prefetch)


def _cast_weight_bias(module, input=None, *, dtype=None, device=None, bias_dtype=None,
                      compute_dtype=None, want_requant=False):
    runtime = weight_cast.get_weight_cast_runtime(module, input)
    return runtime.resolve(
        module,
        cast_bias_weight,
        input,
        dtype=dtype,
        device=device,
        bias_dtype=bias_dtype,
        compute_dtype=compute_dtype,
        want_requant=want_requant,
    )


def _release_weight_bias(module, output, state):
    if isinstance(state, tuple):
        module_key, invocation_id = state
        torch.ops.comfy_weight.release_(output, module_key, invocation_id)
        return output
    runtime = weight_cast.get_weight_cast_runtime_by_name(state.backend)
    return runtime.release(module, uncast_bias_weight, output, state)


class SkipInit:
    def reset_parameters(self):
        return None


class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []
    _v = None
    _v_signature = None
    _v_weight = None
    _v_bias = None
    _prefetch = None


class skip_init:
    class Linear(SkipInit, torch.nn.Linear):
        pass

    class Conv1d(SkipInit, torch.nn.Conv1d):
        pass

    class Conv2d(SkipInit, torch.nn.Conv2d):
        pass

    class Conv3d(SkipInit, torch.nn.Conv3d):
        pass

    class GroupNorm(SkipInit, torch.nn.GroupNorm):
        pass

    class LayerNorm(SkipInit, torch.nn.LayerNorm):
        pass

    class ConvTranspose2d(SkipInit, torch.nn.ConvTranspose2d):
        pass

    class ConvTranspose1d(SkipInit, torch.nn.ConvTranspose1d):
        pass

    class Embedding(SkipInit, torch.nn.Embedding):
        def forward(self, *args, **kwargs) -> Tensor:
            if "out_dtype" in kwargs:
                kwargs.pop("out_dtype")
            return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class disable_weight_init:
    @staticmethod
    def _zero_init_parameter(module, name):
        param = getattr(module, name)
        device = None if getattr(param, "is_meta", False) else param.device
        setattr(module, name, torch.nn.Parameter(torch.zeros(param.shape, device=device, dtype=param.dtype), requires_grad=False))

    @staticmethod
    def _lazy_load_from_state_dict(module, state_dict, prefix, local_metadata,
                                   missing_keys, unexpected_keys, weight_shape,
                                   bias_shape=None):
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            key = k[prefix_len:]
            if key == "weight":
                if not assign_to_params_buffers:
                    v = v.clone()
                module.weight = torch.nn.Parameter(v, requires_grad=False)
            elif bias_shape is not None and key == "bias" and v is not None:
                if not assign_to_params_buffers:
                    v = v.clone()
                module.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        if module.weight is None:
            module.weight = torch.nn.Parameter(torch.zeros(weight_shape), requires_grad=False)
            missing_keys.append(prefix + "weight")

        if bias_shape is not None and module.bias is None and getattr(module, "comfy_need_lazy_init_bias", False):
            module.bias = torch.nn.Parameter(torch.zeros(bias_shape), requires_grad=False)
            missing_keys.append(prefix + "bias")

    class Linear(torch.nn.Linear, CastWeightBiasOp):

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # don't trust subclasses that BYO state dict loader to call us.
            if (not model_management.WINDOWS
                    or not memory_management.aimdo_enabled()
                    or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
                super().__init__(in_features, out_features, bias, device, dtype)
                return

            # Issue is with `torch.empty` still reserving the full memory for the layer.
            # Windows doesn't over-commit memory so without this, We are momentarily commit
            # charged for the weight even though we might zero-copy it when we load the
            # state dict. If the commit charge exceeds the ceiling we can destabilize the
            # system.
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.comfy_need_lazy_init_bias = bias
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys, error_msgs):

            if (not model_management.WINDOWS
                    or not memory_management.aimdo_enabled()
                    or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
                return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            disable_weight_init._lazy_load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                missing_keys,
                unexpected_keys,
                weight_shape=(self.in_features, self.out_features),
                bias_shape=(self.out_features,),
            )

        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.linear(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = self._conv_forward(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = self._conv_forward(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
            if autopad == "causal_zero":
                weight = weight[:, :, -input.shape[2]:, :, :]
            if NVIDIA_MEMORY_CONV_BUG_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
                out = torch.cudnn_convolution(input, weight, self.padding, self.stride, self.dilation, self.groups, benchmark=False, deterministic=False, allow_tf32=True)
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            else:
                return super()._conv_forward(input, weight, bias, *args, **kwargs)

        def forward_comfy_cast_weights(self, input, autopad=None):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = self._conv_forward(input, weight, bias, autopad=autopad)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0 or "autopad" in kwargs:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class BatchNorm2d(torch.nn.BatchNorm2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            running_mean = self.running_mean.to(device=input.device, dtype=weight.dtype) if self.running_mean is not None else None
            running_var = self.running_var.to(device=input.device, dtype=weight.dtype) if self.running_var is not None else None
            x = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, self.training, self.momentum, self.eps)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias, cast_state = _cast_weight_bias(self, input)
            else:
                weight = None
                bias = None
                cast_state = None
            x = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
            return _release_weight_bias(self, x, cast_state) if cast_state is not None else x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class RMSNorm(rmsnorm.RMSNorm, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias, cast_state = _cast_weight_bias(self, input)
            else:
                weight = None
                bias = None
                cast_state = None
            x = rmsnorm.rms_norm(input, weight, self.eps)  # TODO: switch to commented out line when old torch is deprecated
            # x = torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
            return _release_weight_bias(self, x, cast_state) if cast_state is not None else x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                     norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                     _freeze=False, device=None, dtype=None):
            # don't trust subclasses that BYO state dict loader to call us.
            if (not model_management.WINDOWS
                    or not memory_management.aimdo_enabled()
                    or type(self)._load_from_state_dict is not disable_weight_init.Embedding._load_from_state_dict):
                super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                                 norm_type, scale_grad_by_freq, sparse, _weight,
                                 _freeze, device, dtype)
                return

            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            # Keep shape/dtype visible for module introspection without reserving storage.
            embedding_dtype = dtype if dtype is not None else torch.get_default_dtype()
            self.weight = torch.nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device="meta", dtype=embedding_dtype),
                requires_grad=False,
            )
            self.bias = None
            self.weight_comfy_model_dtype = dtype

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys, error_msgs):

            if (not model_management.WINDOWS
                    or not memory_management.aimdo_enabled()
                    or type(self)._load_from_state_dict is not disable_weight_init.Embedding._load_from_state_dict):
                return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            disable_weight_init._lazy_load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                missing_keys,
                unexpected_keys,
                weight_shape=(self.num_embeddings, self.embedding_dim),
            )

        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, bias, cast_state = _cast_weight_bias(self, device=input.device, dtype=out_dtype)
            x = torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class BatchNorm2d(disable_weight_init.BatchNorm2d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class RMSNorm(disable_weight_init.RMSNorm):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


def fp8_linear(self, input):
    """
    Legacy FP8 linear function for backward compatibility.
    Uses QuantizedTensor subclass for dispatch.
    """
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    input_dtype = input.dtype
    input_shape = input.shape
    tensor_3d = input.ndim == 3

    if tensor_3d:
        input = input.reshape(-1, input_shape[2])

    if input.ndim != 2:
        return None
    lora_compute_dtype = model_management.lora_compute_dtype(input.device)
    w, bias, cast_state = _cast_weight_bias(
        self,
        input,
        dtype=dtype,
        bias_dtype=input_dtype,
        compute_dtype=lora_compute_dtype,
        want_requant=True,
    )
    scale_weight = torch.ones((), device=input.device, dtype=torch.float32)

    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
    input = torch.clamp(input, min=-448, max=448, out=input)
    input_fp8 = input.to(dtype).contiguous()
    layout_params_input = TensorCoreFP8Layout.Params(scale=scale_input, orig_dtype=input_dtype, orig_shape=tuple(input_fp8.shape))
    quantized_input = QuantizedTensor(input_fp8, "TensorCoreFP8Layout", layout_params_input)

    # Wrap weight in QuantizedTensor - this enables unified dispatch
    # Call F.linear - __torch_dispatch__ routes to fp8_linear handler in quant_ops.py!
    layout_params_weight = TensorCoreFP8Layout.Params(scale=scale_weight, orig_dtype=input_dtype, orig_shape=tuple(w.shape))
    quantized_weight = QuantizedTensor(w, "TensorCoreFP8Layout", layout_params_weight)
    o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)

    o = _release_weight_bias(self, o, cast_state)
    if tensor_3d:
        o = o.reshape((input_shape[0], input_shape[1], w.shape[0]))

    return o


class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            if len(self.weight_function) == 0 and len(self.bias_function) == 0:
                try:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out
                except Exception as e:
                    logger.info("Exception during fp8 op: {}".format(e))

            # todo: this was also in our code and is probably unnecessary now
            # if input.dtype == torch.float32 and (self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16):
            #     input = input.to(self.weight.dtype)

            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.linear(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)


class scaled_fp8_op_base(manual_cast):
    pass


CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear, cublas_half_matmul

    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:
    class cublas_ops(manual_cast):
        class Linear(CublasLinear, manual_cast.Linear):
            def reset_parameters(self):
                return None

            def forward_comfy_cast_weights(self, input):
                weight, bias, cast_state = _cast_weight_bias(self, input)
                x = cublas_half_matmul(input, weight, bias, self._epilogue_str, self.has_bias)
                return _release_weight_bias(self, x, cast_state)

            def forward(self, *args, **kwargs):
                run_every_op()
                if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                else:
                    return super().forward(*args, **kwargs)
else:
    class cublas_ops(disable_weight_init):
        pass

Operations = typing.Type[typing.Union[manual_cast, fp8_ops, disable_weight_init, skip_init, scaled_fp8_op_base]]

# ==============================================================================
# Mixed Precision Operations
# ==============================================================================
from .quant_ops import (
    QuantizedTensor,
    QUANT_ALGOS,
    TensorCoreFP8Layout,
    get_layout_class,
    mixed_precision_quantization_available,
)


def _quantized_layout_supports_fast_matmul(layout_type):
    if layout_type is None:
        return False
    try:
        layout_cls = get_layout_class(layout_type)
    except Exception:
        return True
    if layout_cls is None:
        return True
    supports_fast_matmul = getattr(layout_cls, "supports_fast_matmul", None)
    if supports_fast_matmul is None:
        return True
    try:
        return supports_fast_matmul()
    except Exception:
        return True


def should_keep_quantized_vbar(module, tensor):
    if not isinstance(tensor, QuantizedTensor):
        return False
    policy = dynamic_vram_fp8_policy()
    if policy == "resident":
        return True
    if policy == "materialize":
        return False
    layout_type = getattr(module, "layout_type", None)
    return (
        layout_type is not None
        and not getattr(module, "_full_precision_mm", False)
    )


class QuantLinearFunc(torch.autograd.Function):
    """Custom autograd function for quantized linear: quantized forward, optionally FP8 backward.

    When training_fp8_bwd is enabled:
      - Forward: quantize input per layout (FP8/NVFP4), use quantized matmul
      - Backward: all matmuls use FP8 tensor cores via torch.mm dispatch
      - Cached input is FP8 (half the memory of bf16)

    When training_fp8_bwd is disabled:
      - Forward: quantize input per layout, use quantized matmul
      - Backward: dequantize weight to compute_dtype, use standard matmul
    """

    @staticmethod
    def forward(ctx, input_float, weight, bias, layout_type, input_scale, compute_dtype):
        input_shape = input_float.shape
        inp = input_float.detach().flatten(0, -2)  # zero-cost view to 2D

        # Quantize input for forward (same layout as weight)
        if layout_type is not None:
            q_input = QuantizedTensor.from_float(inp, layout_type, scale=input_scale)
        else:
            q_input = inp

        w = weight.detach() if weight.requires_grad else weight
        b = bias.detach() if bias is not None and bias.requires_grad else bias

        output = torch.nn.functional.linear(q_input, w, b)

        # Unflatten output to match original input shape
        if len(input_shape) > 2:
            output = output.unflatten(0, input_shape[:-1])

        # Save for backward
        ctx.input_shape = input_shape
        ctx.has_bias = bias is not None
        ctx.compute_dtype = compute_dtype
        ctx.weight_requires_grad = weight.requires_grad
        ctx.fp8_bwd = model_management.training_fp8_bwd

        if ctx.fp8_bwd:
            # Cache FP8 quantized input — half the memory of bf16
            if isinstance(q_input, QuantizedTensor) and layout_type.startswith('TensorCoreFP8'):
                ctx.q_input = q_input  # already FP8, reuse
            else:
                # NVFP4 or other layout — quantize input to FP8 for backward
                ctx.q_input = QuantizedTensor.from_float(inp, "TensorCoreFP8E4M3Layout")
            ctx.save_for_backward(weight)
        else:
            ctx.q_input = None
            ctx.save_for_backward(input_float, weight)

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        compute_dtype = ctx.compute_dtype
        grad_2d = grad_output.flatten(0, -2).to(compute_dtype)

        # Value casting — only difference between fp8 and non-fp8 paths
        if ctx.fp8_bwd:
            weight, = ctx.saved_tensors
            # Wrap as FP8 QuantizedTensors → torch.mm dispatches to _scaled_mm
            grad_mm = QuantizedTensor.from_float(grad_2d, "TensorCoreFP8E5M2Layout")
            if isinstance(weight, QuantizedTensor) and weight._layout_cls.startswith("TensorCoreFP8"):
                weight_mm = weight
            elif isinstance(weight, QuantizedTensor):
                weight_mm = QuantizedTensor.from_float(weight.dequantize().to(compute_dtype), "TensorCoreFP8E4M3Layout")
            else:
                weight_mm = QuantizedTensor.from_float(weight.to(compute_dtype), "TensorCoreFP8E4M3Layout")
            input_mm = ctx.q_input
        else:
            input_float, weight = ctx.saved_tensors
            # Standard tensors → torch.mm does regular matmul
            grad_mm = grad_2d
            if isinstance(weight, QuantizedTensor):
                weight_mm = weight.dequantize().to(compute_dtype)
            else:
                weight_mm = weight.to(compute_dtype)
            input_mm = input_float.flatten(0, -2).to(compute_dtype) if ctx.weight_requires_grad else None

        # Computation — same for both paths, dispatch handles the rest
        grad_input = torch.mm(grad_mm, weight_mm)
        if len(ctx.input_shape) > 2:
            grad_input = grad_input.unflatten(0, ctx.input_shape[:-1])

        grad_weight = None
        if ctx.weight_requires_grad:
            grad_weight = torch.mm(grad_mm.t(), input_mm)

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None


def mixed_precision_ops(quant_config=None, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
    if quant_config is None:
        quant_config = {}

    class MixedPrecisionOps(manual_cast):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled

        class Linear(torch.nn.Module, CastWeightBiasOp):
            def __init__(
                    self,
                    in_features: int,
                    out_features: int,
                    bias: bool = True,
                    device=None,
                    dtype=None,
            ) -> None:
                super().__init__()

                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
                # self.factory_kwargs = {"device": device, "dtype": dtype}

                self.in_features = in_features
                self.out_features = out_features
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                self.tensor_class = None
                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False
                self.weight_function = []
                self.bias_function = []
                self.comfy_cast_weights = False

            def reset_parameters(self):
                return None

            def _load_scale_param(self, state_dict, prefix, param_name, device, manually_loaded_keys, dtype=None):
                key = f"{prefix}{param_name}"
                value = state_dict.pop(key, None)
                if value is not None:
                    value = value.to(device=device)
                    if dtype is not None:
                        value = value.view(dtype=dtype)
                    manually_loaded_keys.append(key)
                return value

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs):

                device = self.factory_kwargs["device"]
                layer_name = prefix.rstrip('.')
                weight_key = f"{prefix}weight"
                weight = state_dict.pop(weight_key, None)
                if weight is None:
                    logger.warning(f"Missing weight for layer {layer_name}")
                    self.weight = None
                    return

                manually_loaded_keys = [weight_key]

                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                if layer_conf is None:
                    self.weight = torch.nn.Parameter(weight.to(device=device, dtype=MixedPrecisionOps._compute_dtype), requires_grad=False)
                else:
                    self.quant_format = layer_conf.get("format", None)
                    self._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
                    if not self._full_precision_mm:
                        self._full_precision_mm = self._full_precision_mm_config

                    if self.quant_format is None:
                        raise ValueError(f"Unknown quantization format for layer {layer_name}")

                    qconfig = QUANT_ALGOS[self.quant_format]
                    if self.quant_format in MixedPrecisionOps._disabled:
                        self._full_precision_mm = True

                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)

                    # Load format-specific parameters
                    if self.quant_format in ["float8_e4m3fn", "float8_e5m2"]:
                        # FP8: single tensor scale
                        scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys)

                        params = layout_cls.Params(
                            scale=scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )

                    elif self.quant_format == "mxfp8":
                        # MXFP8: E8M0 block scales stored as uint8 in safetensors
                        block_scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys,
                                                             dtype=torch.uint8)

                        if block_scale is None:
                            raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")

                        block_scale = block_scale.view(torch.float8_e8m0fnu)

                        params = layout_cls.Params(
                            scale=block_scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )

                    elif self.quant_format == "nvfp4":
                        # NVFP4: tensor_scale (weight_scale_2) + block_scale (weight_scale)
                        tensor_scale = self._load_scale_param(state_dict, prefix, "weight_scale_2", device, manually_loaded_keys)
                        block_scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys,
                                                             dtype=torch.float8_e4m3fn)

                        if tensor_scale is None or block_scale is None:
                            raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")

                        params = layout_cls.Params(
                            scale=tensor_scale,
                            block_scale=block_scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )
                    else:
                        raise ValueError(f"Unsupported quantization format: {self.quant_format}")

                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), self.layout_type, params),
                        requires_grad=False
                    )

                    for param_name in qconfig["parameters"]:
                        if param_name in {"weight_scale", "weight_scale_2"}:
                            continue  # Already handled above

                        param_key = f"{prefix}{param_name}"
                        _v = state_dict.pop(param_key, None)
                        if _v is None:
                            continue
                        self.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                        manually_loaded_keys.append(param_key)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                for key in manually_loaded_keys:
                    if key in missing_keys:
                        missing_keys.remove(key)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                if destination is not None:
                    sd: dict = destination
                else:
                    sd = {}

                if not hasattr(self, 'weight'):
                    logger.warning("Warning: state dict on uninitialized op {}".format(prefix))
                    return sd

                if self.bias is not None:
                    sd["{}bias".format(prefix)] = self.bias

                if self.weight is None:
                    return sd

                if isinstance(self.weight, QuantizedTensor):
                    sd_out = self.weight.state_dict("{}weight".format(prefix))
                    for k in sd_out:
                        sd[k] = sd_out[k]

                    quant_conf = {"format": self.quant_format}
                    if self._full_precision_mm_config:
                        quant_conf["full_precision_matrix_mult"] = True
                    sd["{}comfy_quant".format(prefix)] = torch.tensor(list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8)

                    input_scale = getattr(self, 'input_scale', None)
                    if input_scale is not None:
                        sd["{}input_scale".format(prefix)] = input_scale
                else:
                    sd["{}weight".format(prefix)] = self.weight
                return sd

            def _forward(self, input, weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

            def forward_comfy_cast_weights(self, input, compute_dtype=None, want_requant=False):
                # todo: not sure why this looks the way it does in our fork
                # if input.dtype == torch.float32 and (self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16):
                #     input = input.to(self.weight.dtype)
                # comfyui original:
                weight, bias, cast_state = _cast_weight_bias(
                    self,
                    input,
                    compute_dtype=compute_dtype,
                    want_requant=want_requant,
                )
                x = self._forward(input, weight, bias)
                return _release_weight_bias(self, x, cast_state)

            def forward(self, input, *args, **kwargs):
                run_every_op()

                if (
                    weight_cast.is_torch_compiling()
                    and weight_cast.graph_visible_backend_unavailable_reason() is None
                    and not model_management.is_device_cpu(input.device)
                ):
                    return self.forward_comfy_cast_weights(input, input.dtype, want_requant=False)

                input_shape = input.shape
                reshaped_3d = False
                # If cast needs to apply lora, it should be done in the compute dtype
                compute_dtype = input.dtype

                force_cast_blocks_quantized = getattr(self, 'comfy_force_cast_weights', False) and not isinstance(self.weight, QuantizedTensor)
                _use_quantized = (
                        getattr(self, 'layout_type', None) is not None and
                        not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
                        not force_cast_blocks_quantized and
                        len(self.weight_function) == 0 and len(self.bias_function) == 0
                )

                # Training path: quantized forward with compute_dtype backward via autograd function
                if (input.requires_grad and _use_quantized):

                    weight, bias, cast_state = _cast_weight_bias(
                        self,
                        input,
                        compute_dtype=compute_dtype,
                        want_requant=True
                    )

                    scale = getattr(self, 'input_scale', None)
                    if scale is not None:
                        scale = model_management.cast_to_device(scale, input.device, None)

                    output = QuantLinearFunc.apply(
                        input, weight, bias, self.layout_type, scale, compute_dtype
                    )

                    return _release_weight_bias(self, output, cast_state)

                # Inference path (unchanged)
                if _use_quantized:

                    # Reshape 3D tensors to 2D for quantization (needed for NVFP4 and others)
                    input_reshaped = input.reshape(-1, input_shape[2]) if input.ndim == 3 else input

                    # Fall back to non-quantized for non-2D tensors
                    if input_reshaped.ndim == 2:
                        reshaped_3d = input.ndim == 3
                        # dtype is now implicit in the layout class
                        scale = getattr(self, 'input_scale', None)
                        if scale is not None:
                            scale = model_management.cast_to_device(scale, input.device, None)
                        input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                output = self.forward_comfy_cast_weights(input, compute_dtype, want_requant=isinstance(input, QuantizedTensor))

                # Reshape output back to 3D if input was 3D
                if reshaped_3d:
                    output = output.reshape((input_shape[0], input_shape[1], self.weight.shape[0]))

                return output

            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, 'layout_type', None) is not None:
                    # dtype is now implicit in the layout class
                    weight = QuantizedTensor.from_float(weight, self.layout_type, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):  # This is to get torch.compile + moving weights to another device working
                if recurse:
                    for module in self.children():
                        module._apply(fn)

                for key, param in self._parameters.items():
                    if param is None:
                        continue
                    p = fn(param)
                    if (not torch.is_inference_mode_enabled()) and p.is_inference():
                        p = p.clone()
                    self.register_parameter(key, torch.nn.Parameter(p, requires_grad=False))
                for key, buf in self._buffers.items():
                    if buf is not None:
                        self._buffers[key] = fn(buf)
                return self

        class Embedding(manual_cast.Embedding):
            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                    strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                # Only fp8 makes sense for embeddings (per-row dequant via index select).
                # Block-scaled formats (NVFP4, MXFP8) can't do per-row lookup efficiently.
                quant_format = layer_conf.get("format", None) if layer_conf is not None else None
                if quant_format in ["float8_e4m3fn", "float8_e5m2"] and weight_key in state_dict:
                    self.quant_format = quant_format
                    qconfig = QUANT_ALGOS[quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)
                    weight = state_dict.pop(weight_key)
                    manually_loaded_keys = [weight_key]

                    scale_key = f"{prefix}weight_scale"
                    scale = state_dict.pop(scale_key, None)
                    if scale is not None:
                        scale = scale.float()
                        manually_loaded_keys.append(scale_key)

                    params = layout_cls.Params(
                        scale=scale if scale is not None else torch.ones((), dtype=torch.float32),
                        orig_dtype=MixedPrecisionOps._compute_dtype,
                        orig_shape=(self.num_embeddings, self.embedding_dim),
                    )
                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight.to(dtype=qconfig["storage_t"]), qconfig["comfy_tensor_layout"], params),
                        requires_grad=False)

                    super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                    for k in manually_loaded_keys:
                        if k in missing_keys:
                            missing_keys.remove(k)
                else:
                    if layer_conf is not None:
                        state_dict[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(layer_conf).encode('utf-8')), dtype=torch.uint8)
                    super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                if destination is not None:
                    sd = destination
                else:
                    sd = {}

                if not hasattr(self, 'weight') or self.weight is None:
                    return sd

                if isinstance(self.weight, QuantizedTensor):
                    sd_out = self.weight.state_dict("{}weight".format(prefix))
                    for k in sd_out:
                        sd[k] = sd_out[k]

                    quant_conf = {"format": self.quant_format}
                    sd["{}comfy_quant".format(prefix)] = torch.tensor(list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8)
                else:
                    sd["{}weight".format(prefix)] = self.weight
                return sd

            def forward_comfy_cast_weights(self, input, out_dtype=None):
                weight = self.weight

                # Optimized path: lookup in fp8, dequantize only the selected rows.
                if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
                    qdata, _, cast_state = _cast_weight_bias(self, input, device=input.device, dtype=weight.dtype)
                    if isinstance(qdata, QuantizedTensor):
                        scale = qdata._params.scale
                        qdata = qdata._qdata
                    else:
                        scale = None

                    x = torch.nn.functional.embedding(
                        input, qdata, self.padding_idx, self.max_norm,
                        self.norm_type, self.scale_grad_by_freq, self.sparse)
                    x = _release_weight_bias(self, x, cast_state)
                    target_dtype = out_dtype if out_dtype is not None else weight._params.orig_dtype
                    x = x.to(dtype=target_dtype)
                    if scale is not None and scale != 1.0:
                        x = x * scale.to(dtype=target_dtype)
                    return x

                # Fallback for non-quantized or weight_function (LoRA) case
                return super().forward_comfy_cast_weights(input, out_dtype=out_dtype)

    return MixedPrecisionOps


def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, model_config=None, inference_mode: Optional[bool] = None):
    if inference_mode is None:
        # todo: check a context here, since this isn't being used by any callers yet
        inference_mode = current_execution_context().inference_mode
    fp8_compute = model_management.supports_fp8_compute(load_device)  # TODO: if we support more ops this needs to be more granular
    nvfp4_compute = model_management.supports_nvfp4_compute(load_device)
    mxfp8_compute = model_management.supports_mxfp8_compute(load_device)

    if model_config and hasattr(model_config, 'quant_config') and model_config.quant_config:
        logger.info("Using mixed precision operations")
        disabled = set()
        if not mixed_precision_quantization_available():
            disabled.update({"float8_e4m3fn", "float8_e5m2", "nvfp4"})
        if not nvfp4_compute:
            disabled.add("nvfp4")
        if not mxfp8_compute:
            disabled.add("mxfp8")
        if not fp8_compute:
            disabled.add("float8_e4m3fn")
            disabled.add("float8_e5m2")
        return mixed_precision_ops(model_config.quant_config, compute_dtype, disabled=disabled)

    if (
            fp8_compute and
            (fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast) and
            not disable_fast_fp8
    ):
        return fp8_ops

    if (
            PerformanceFeature.CublasOps in args.fast and
            CUBLAS_IS_AVAILABLE and
            weight_dtype == torch.float16 and
            (compute_dtype == torch.float16 or compute_dtype is None)
    ):
        logger.debug("Using cublas ops")
        return cublas_ops

    if compute_dtype is None or weight_dtype == compute_dtype:
        # disable_weight_init seems to interact poorly with some other optimization code
        return disable_weight_init if inference_mode else skip_init

    return manual_cast
