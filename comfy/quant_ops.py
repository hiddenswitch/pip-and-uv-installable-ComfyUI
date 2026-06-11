from dataclasses import dataclass
import logging
import os
import torch

from .cli_args import args
from .float import stochastic_rounding as stochastic_rounding_fn, stochastic_round_quantize_nvfp4_by_block, stochastic_round_quantize_mxfp8_by_block

logger = logging.getLogger(__name__)

_OUTPUT_DTYPE_CODES = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
}
_OUTPUT_DTYPE_TO_CODE = {dtype: code for code, dtype in _OUTPUT_DTYPE_CODES.items()}
_FP8_MATERIALIZATION_CODES = {
    "auto": 0,
    "torch": 1,
    "comfy_kitchen": 2,
}


def _output_dtype_code(dtype: torch.dtype) -> int:
    try:
        return _OUTPUT_DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported FP8 dequant output dtype: {dtype}") from exc


def _fp8_materialization_code() -> int:
    mode = os.environ.get("COMFYUI_FP8_MATERIALIZATION", None) or getattr(args, "fp8_materialization", "auto")
    try:
        return _FP8_MATERIALIZATION_CODES[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported FP8 materialization mode: {mode}") from exc


@torch.library.custom_op(
    "comfy_quant::dequantize_per_tensor_fp8",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe, torch.Tag.maybe_aliasing_or_mutating),
)
def _safe_dequantize_per_tensor_fp8(qdata: torch.Tensor, scale: torch.Tensor, output_dtype_code: int) -> torch.Tensor:
    output_dtype = _OUTPUT_DTYPE_CODES[output_dtype_code]
    return qdata.to(dtype=output_dtype) * scale.to(dtype=output_dtype)


@_safe_dequantize_per_tensor_fp8.register_fake
def _safe_dequantize_per_tensor_fp8_fake(qdata: torch.Tensor, scale: torch.Tensor, output_dtype_code: int) -> torch.Tensor:
    return qdata.new_empty(tuple(qdata.shape), dtype=_OUTPUT_DTYPE_CODES[output_dtype_code])


@torch.library.custom_op(
    "comfy_quant::materialize_per_tensor_fp8",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe, torch.Tag.maybe_aliasing_or_mutating),
)
def _materialize_per_tensor_fp8(qdata: torch.Tensor, scale: torch.Tensor, output_dtype_code: int, mode_code: int) -> torch.Tensor:
    output_dtype = _OUTPUT_DTYPE_CODES[output_dtype_code]
    if mode_code == _FP8_MATERIALIZATION_CODES["torch"]:
        return qdata.to(dtype=output_dtype) * scale.to(dtype=output_dtype)
    if mode_code == _FP8_MATERIALIZATION_CODES["comfy_kitchen"] and _CK_AVAILABLE:
        return ck.dequantize_per_tensor_fp8(qdata, scale, output_dtype)
    if _CK_AVAILABLE:
        return ck.dequantize_per_tensor_fp8(qdata, scale, output_dtype)
    return qdata.to(dtype=output_dtype) * scale.to(dtype=output_dtype)


@_materialize_per_tensor_fp8.register_fake
def _materialize_per_tensor_fp8_fake(qdata: torch.Tensor, scale: torch.Tensor, output_dtype_code: int, mode_code: int) -> torch.Tensor:
    return qdata.new_empty(tuple(qdata.shape), dtype=_OUTPUT_DTYPE_CODES[output_dtype_code])


@torch.library.custom_op(
    "comfy_quant::materialize_per_tensor_fp8_after",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe, torch.Tag.maybe_aliasing_or_mutating),
)
def _materialize_per_tensor_fp8_after(
    memory_token: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    output_dtype_code: int,
    mode_code: int,
) -> torch.Tensor:
    return _materialize_per_tensor_fp8(qdata, scale, output_dtype_code, mode_code)


@_materialize_per_tensor_fp8_after.register_fake
def _materialize_per_tensor_fp8_after_fake(
    memory_token: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    output_dtype_code: int,
    mode_code: int,
) -> torch.Tensor:
    return qdata.new_empty(tuple(qdata.shape), dtype=_OUTPUT_DTYPE_CODES[output_dtype_code])


@torch.library.custom_op(
    "comfy_quant::release_materialization_",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe, torch.Tag.maybe_aliasing_or_mutating),
)
def _release_materialization_(output: torch.Tensor, materialized: torch.Tensor, memory_token: torch.Tensor) -> torch.Tensor:
    return memory_token.new_empty((), dtype=torch.int64)


@_release_materialization_.register_fake
def _release_materialization_fake(output: torch.Tensor, materialized: torch.Tensor, memory_token: torch.Tensor) -> torch.Tensor:
    return memory_token.new_empty((), dtype=torch.int64)


def materialize_per_tensor_fp8(qdata: torch.Tensor, scale: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return torch.ops.comfy_quant.materialize_per_tensor_fp8(
        qdata,
        scale,
        _output_dtype_code(output_dtype),
        _fp8_materialization_code(),
    )


def _fp8e4m3fn_triton_unsupported(device: torch.device | None) -> bool:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability(device)
    except (RuntimeError, AssertionError):
        return False
    return capability < (8, 9)


@torch.compiler.disable
def _dequantize_per_tensor_fp8_eager(qdata: torch.Tensor, scale: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return qdata.to(dtype=output_dtype) * scale.to(dtype=output_dtype)

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        QuantizedLayout,
        TensorCoreFP8Layout as _CKFp8Layout,
        TensorCoreNVFP4Layout as _CKNvfp4Layout,
        register_layout_op,
        register_layout_class,
        get_layout_class as _ck_get_layout_class,
    )

    _CK_AVAILABLE = True
    if torch.version.cuda is None:
        ck.registry.disable("cuda")
    else:
        cuda_version = tuple(map(int, str(torch.version.cuda).split('.')))
        if cuda_version < (13,):
            ck.registry.disable("cuda")
            logger.debug(f"You need pytorch with cu130 or higher to use optimized CUDA operations, found {torch.version.cuda}")

    try:
        pass
    except:
        logger.debug("Disabling triton support, it was not installed")
        ck.registry.disable("triton")

    # comfy_kitchen's triton fp8 kernels use fp8e4nv, which fails to compile
    # on Ampere (sm < 8.9). Force-disable triton on those GPUs unless the
    # user re-enables it via --enable-comfy-kitchen-backends triton.
    # This disable is fp8-specific: the INT8 w8a8 triton kernels live in
    # comfy/int8_kernels.py outside this registry and stay enabled on Ampere.
    if torch.cuda.is_available():
        try:
            min_cap = min(
                (torch.cuda.get_device_properties(i).major,
                 torch.cuda.get_device_properties(i).minor)
                for i in range(torch.cuda.device_count())
            )
        except (RuntimeError, ValueError, AssertionError):
            min_cap = None
        if min_cap is not None and min_cap < (8, 9):
            ck.registry.disable("triton")
            logger.debug(
                f"Disabling comfy_kitchen 'triton' backend: NVIDIA compute capability "
                f"{min_cap[0]}.{min_cap[1]} < 8.9 (fp8e4nv unsupported)"
            )

    for backend_name in (args.disable_comfy_kitchen_backends or ()):
        ck.registry.disable(backend_name)
        logger.debug(f"Disabling comfy_kitchen backend '{backend_name}' (--disable-comfy-kitchen-backends)")
    for backend_name in (args.enable_comfy_kitchen_backends or ()):
        ck.registry.enable(backend_name)
        logger.debug(f"Enabling comfy_kitchen backend '{backend_name}' (--enable-comfy-kitchen-backends)")

    if args.enable_triton_backend:
        try:
            import triton
            logging.info("Found triton %s. Enabling comfy-kitchen triton backend.", triton.__version__)
        except ImportError as e:
            logging.error(f"Failed to import triton, Error: {e}, the comfy-kitchen triton backend will not be available.")
            ck.registry.disable("triton")
    else:
        ck.registry.disable("triton")
    for k, v in ck.list_backends().items():
        logger.debug(f"Found comfy_kitchen backend {k}: {v}")

    _CK_QUANTIZED_TENSOR_DEQUANTIZE = QuantizedTensor.dequantize

    @torch.compiler.disable
    def _quantized_tensor_dequantize_eager(qtensor: QuantizedTensor) -> torch.Tensor:
        return _CK_QUANTIZED_TENSOR_DEQUANTIZE(qtensor)

    def _quantized_tensor_dequantize_compile_safe(qtensor: QuantizedTensor) -> torch.Tensor:
        qdata = getattr(qtensor, "_qdata", None)
        if (
            torch.compiler.is_compiling()
            and isinstance(qdata, torch.Tensor)
            and qdata.dtype == torch.float8_e4m3fn
            and _fp8e4m3fn_triton_unsupported(qdata.device)
        ):
            return _quantized_tensor_dequantize_eager(qtensor)
        return _CK_QUANTIZED_TENSOR_DEQUANTIZE(qtensor)

    QuantizedTensor.dequantize = _quantized_tensor_dequantize_compile_safe
except Exception as e:
    logger.debug(f"Failed to import comfy_kitchen, Error: {e}, fp8 and fp4 support will not be available.")
    _CK_AVAILABLE = False
    ck = None


    class QuantizedTensor:
        pass

    class QuantizedLayout:
        pass


    class _CKFp8Layout:
        pass


    class _CKNvfp4Layout:
        pass


    def register_layout_class(name, cls):
        pass


    def _ck_get_layout_class(name):
        return None

    def register_layout_op(*args, **kwargs):
        pass

_CK_MXFP8_AVAILABLE = False
if _CK_AVAILABLE:
    try:
        from comfy_kitchen.tensor import TensorCoreMXFP8Layout as _CKMxfp8Layout
        _CK_MXFP8_AVAILABLE = True
    except ImportError:
        logger.debug("comfy_kitchen does not support MXFP8")

if not _CK_MXFP8_AVAILABLE:
    class _CKMxfp8Layout:  # noqa: F811
        pass


# ==============================================================================
# FP8 Layouts with Comfy-Specific Extensions
# ==============================================================================

class _TensorCoreFP8LayoutBase(_CKFp8Layout):
    FP8_DTYPE = None  # Must be overridden in subclass

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if cls.FP8_DTYPE is None:
            raise NotImplementedError(f"{cls.__name__} must define FP8_DTYPE")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:  # Prevent scale from being too small
                tensor_info = torch.finfo(tensor.dtype)
                scale = (1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max))

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = stochastic_rounding_fn(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params

    @classmethod
    def dequantize(cls, qdata, params):
        if not torch.compiler.is_compiling():
            return super(_TensorCoreFP8LayoutBase, cls).dequantize(qdata, params)
        if _fp8_materialization_code() == _FP8_MATERIALIZATION_CODES["torch"]:
            return materialize_per_tensor_fp8(qdata, params.scale, params.orig_dtype)
        if (
            qdata.dtype == torch.float8_e4m3fn
            and _fp8e4m3fn_triton_unsupported(qdata.device)
        ):
            return _dequantize_per_tensor_fp8_eager(qdata, params.scale, params.orig_dtype)
        return materialize_per_tensor_fp8(qdata, params.scale, params.orig_dtype)


class TensorCoreMXFP8Layout(_CKMxfp8Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = stochastic_round_quantize_mxfp8_by_block(tensor, pad_32x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params


class TensorCoreNVFP4Layout(_CKNvfp4Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params


class TensorCoreFP8E4M3Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


# Backward compatibility alias - default to E4M3
TensorCoreFP8Layout = TensorCoreFP8E4M3Layout


if not hasattr(TensorCoreFP8Layout, "Params"):
    @dataclass(frozen=True)
    class _FP8Params:
        scale: torch.Tensor
        orig_dtype: torch.dtype
        orig_shape: tuple[int, ...]

    TensorCoreFP8Layout.Params = _FP8Params
    TensorCoreFP8E4M3Layout.Params = _FP8Params
    TensorCoreFP8E5M2Layout.Params = _FP8Params


if not hasattr(TensorCoreNVFP4Layout, "Params"):
    @dataclass(frozen=True)
    class _NVFP4Params:
        scale: torch.Tensor
        orig_dtype: torch.dtype
        orig_shape: tuple[int, ...]
        block_scale: torch.Tensor | None = None

    TensorCoreNVFP4Layout.Params = _NVFP4Params


if _CK_AVAILABLE:
    try:
        from comfy_kitchen.tensor.fp8 import (
            _handle_fp8_addmm as _ck_handle_fp8_addmm,
            _handle_fp8_linear as _ck_handle_fp8_linear,
            _handle_fp8_mm as _ck_handle_fp8_mm,
        )
    except Exception:
        _ck_handle_fp8_addmm = None
        _ck_handle_fp8_linear = None
        _ck_handle_fp8_mm = None

    def _compile_unsupported_fp8_qtensor(value) -> bool:
        qdata = getattr(value, "_qdata", None)
        return (
            isinstance(value, QuantizedTensor)
            and isinstance(qdata, torch.Tensor)
            and qdata.dtype == torch.float8_e4m3fn
            and _fp8e4m3fn_triton_unsupported(qdata.device)
        )

    def _dequantize_qtensor_arg(value):
        if isinstance(value, QuantizedTensor):
            return value.dequantize()
        return value

    @torch.compiler.disable
    def _fp8_linear_dequant_eager(input_tensor, weight, bias):
        return torch.nn.functional.linear(
            _dequantize_qtensor_arg(input_tensor),
            _dequantize_qtensor_arg(weight),
            bias,
        )

    @torch.compiler.disable
    def _fp8_mm_dequant_eager(a, b):
        return torch.mm(_dequantize_qtensor_arg(a), _dequantize_qtensor_arg(b))

    @torch.compiler.disable
    def _fp8_addmm_dequant_eager(bias, input_tensor, weight):
        return torch.addmm(
            bias,
            _dequantize_qtensor_arg(input_tensor),
            _dequantize_qtensor_arg(weight),
        )

    @register_layout_op(torch.ops.aten.linear.default, TensorCoreFP8E4M3Layout)
    def _handle_fp8_e4m3_linear(qt, args, kwargs):
        input_tensor, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        if _compile_unsupported_fp8_qtensor(input_tensor) or _compile_unsupported_fp8_qtensor(weight):
            return _fp8_linear_dequant_eager(input_tensor, weight, bias)
        if _ck_handle_fp8_linear is not None:
            return _ck_handle_fp8_linear(qt, args, kwargs)
        return torch.nn.functional.linear(
            _dequantize_qtensor_arg(input_tensor),
            _dequantize_qtensor_arg(weight),
            bias,
        )

    @register_layout_op(torch.ops.aten.mm.default, TensorCoreFP8E4M3Layout)
    def _handle_fp8_e4m3_mm(qt, args, kwargs):
        a, b = args[0], args[1]
        if _compile_unsupported_fp8_qtensor(a) or _compile_unsupported_fp8_qtensor(b):
            return _fp8_mm_dequant_eager(a, b)
        if _ck_handle_fp8_mm is not None:
            return _ck_handle_fp8_mm(qt, args, kwargs)
        return torch.mm(_dequantize_qtensor_arg(a), _dequantize_qtensor_arg(b))

    @register_layout_op(torch.ops.aten.addmm.default, TensorCoreFP8E4M3Layout)
    def _handle_fp8_e4m3_addmm(qt, args, kwargs):
        bias, input_tensor, weight = args[0], args[1], args[2]
        if _compile_unsupported_fp8_qtensor(input_tensor) or _compile_unsupported_fp8_qtensor(weight):
            return _fp8_addmm_dequant_eager(bias, input_tensor, weight)
        if _ck_handle_fp8_addmm is not None:
            return _ck_handle_fp8_addmm(qt, args, kwargs)
        return torch.addmm(
            bias,
            _dequantize_qtensor_arg(input_tensor),
            _dequantize_qtensor_arg(weight),
        )

# ==============================================================================
# Registry
# ==============================================================================

register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
# todo: needs merge, how does this change for torch 2.2.0 compatibility?
if _CK_MXFP8_AVAILABLE:
    register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)

_LAYOUT_CLASS_FALLBACKS = {
    "TensorCoreFP8Layout": TensorCoreFP8Layout,
    "TensorCoreFP8E4M3Layout": TensorCoreFP8E4M3Layout,
    "TensorCoreFP8E5M2Layout": TensorCoreFP8E5M2Layout,
    "TensorCoreNVFP4Layout": TensorCoreNVFP4Layout,
}

_INT8_AVAILABLE = False
if _CK_AVAILABLE:
    try:
        from .quant_ops_int8 import Int8ConvRotLayout, Int8RowwiseLayout

        _INT8_AVAILABLE = True
        _LAYOUT_CLASS_FALLBACKS["Int8RowwiseLayout"] = Int8RowwiseLayout
        _LAYOUT_CLASS_FALLBACKS["Int8ConvRotLayout"] = Int8ConvRotLayout
    except Exception as e:
        logger.debug(f"Failed to load int8 quantized layouts, Error: {e}")


def int8_quantization_available() -> bool:
    return _INT8_AVAILABLE


def get_layout_class(name):
    layout_cls = _ck_get_layout_class(name)
    if layout_cls is not None:
        return layout_cls
    return _LAYOUT_CLASS_FALLBACKS.get(name)


def mixed_precision_quantization_available() -> bool:
    return _CK_AVAILABLE

QUANT_ALGOS = {
    "int8": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "Int8RowwiseLayout",
    },
    "int8_convrot": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "Int8ConvRotLayout",
        "group_size": 256,
    },
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
    # SVDQuant W4A4 (nunchaku): packed int4 weights + per-group scales + SVD
    # low-rank correction. Offline-calibrated (DeepCompressor); load-only.
    "svdquant_w4a4": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "weight_proj_down", "weight_proj_up", "weight_smooth_factor"},
        "comfy_tensor_layout": "TensorCoreSVDQuantW4A4Layout",
        "group_size": 64,
    },
    # AWQ W4A16: packed int4 weights + per-group fp scales/zeros, fp
    # activations (nunchaku uses it for modulation linears). Load-only.
    "awq_w4a16": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "weight_zeros"},
        "comfy_tensor_layout": "TensorCoreAWQW4A16Layout",
        "group_size": 64,
    },
}

if _CK_MXFP8_AVAILABLE:
    QUANT_ALGOS["mxfp8"] = {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    }
else:
    # todo: needs merge, stub for torch 2.2.0?
    pass

# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "QuantizedTensor",
    "QuantizedLayout",
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
    "TensorCoreNVFP4Layout",
    "QUANT_ALGOS",
    "register_layout_op",
    "mixed_precision_quantization_available",
    "int8_quantization_available",
]
