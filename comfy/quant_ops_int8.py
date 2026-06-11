"""INT8 w8a8 quantized layouts with optional ConvRot Hadamard rotation.

Two layouts are registered with the comfy_kitchen QuantizedTensor registry:

- ``Int8RowwiseLayout``: symmetric absmax int8, one fp32 scale per row over the
  last dimension. Weights are stored as int8 ``[out, in]`` with ``weight_scale``
  ``[out, 1]`` (a scalar scale is accepted for legacy tensorwise checkpoints).
  Activations are quantized dynamically per token on every forward; static
  ``input_scale`` tensors in checkpoints are ignored.

- ``Int8ConvRotLayout``: the same storage, but weights and activations live in
  a rotated space. Each contiguous group of ``GROUP_SIZE`` input channels is
  right-multiplied by a normalized "regular" Hadamard matrix H before
  quantization. H is symmetric and orthogonal, hence self-inverse, so
  ``x·H @ (W·H)^T == x @ W^T`` exactly in full precision while the rotation
  spreads activation outliers across channels, reducing absmax quantization
  loss. ``dequantize`` always returns original-space tensors, which keeps LoRA
  baking (dequant -> patch -> requant) and every dequantization fallback
  correct without special cases.

The fast matmul path dispatches ``aten.linear`` to the fused triton GEMM in
``comfy.int8_kernels`` (independent of the comfy_kitchen "triton" backend,
which is disabled on Ampere for fp8 reasons that do not apply here).
"""

import logging
from dataclasses import dataclass

import torch

from comfy_kitchen.tensor import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    register_layout_class,
    register_layout_op,
)

from . import int8_kernels

logger = logging.getLogger(__name__)

_SCALE_EPS = 1e-12


# ==============================================================================
# Regular Hadamard rotation
# ==============================================================================

# The "regular" Hadamard base: every row and column sums to 2, unlike the
# Sylvester construction whose all-ones first row/column concentrates rather
# than spreads row-wise outliers in DiT activations. H4 is symmetric and
# H4 @ H4^T = 4·I.
_H4 = [
    [1, 1, 1, -1],
    [1, 1, -1, 1],
    [1, -1, 1, 1],
    [-1, 1, 1, 1],
]

_hadamard_cache = {}


def regular_hadamard(size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix of the given power-of-4 size.

    Grown by Kronecker products of the 4x4 regular Hadamard base and normalized
    by 1/sqrt(size), so the result is symmetric and orthogonal (self-inverse).
    """
    n = size
    while n > 1:
        if n % 4 != 0:
            raise ValueError(f"regular_hadamard size must be a power of 4, got {size}")
        n //= 4

    key = (size, str(device), dtype)
    cached = _hadamard_cache.get(key)
    if cached is not None:
        return cached

    h = torch.tensor(_H4, device=device, dtype=torch.float64)
    while h.shape[0] < size:
        h = torch.kron(h, torch.tensor(_H4, device=device, dtype=torch.float64))
    h = (h / (size ** 0.5)).to(dtype=dtype)
    _hadamard_cache[key] = h
    return h


def rotate_groups(tensor: torch.Tensor, group_size: int, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Rotate each contiguous group of ``group_size`` last-dim channels by H.

    Because H is symmetric, this single function implements both the offline
    weight rotation (W · H^T per group) and the online activation rotation
    (x · H per group), and applying it twice is the identity.

    Weights rotate in fp32 (one-time, quality matters). The per-forward
    activation rotation passes ``compute_dtype=None`` to rotate in the
    tensor's own dtype: a bf16 rotation matmul runs at tensor-core rate and
    its rounding is far below the int8 quantization grain that follows.
    """
    k = tensor.shape[-1]
    if k % group_size != 0:
        raise ValueError(f"last dim {k} not divisible by convrot group size {group_size}")
    mm_dtype = compute_dtype if compute_dtype is not None else tensor.dtype
    h = regular_hadamard(group_size, device=tensor.device, dtype=mm_dtype)
    orig_dtype = tensor.dtype
    grouped = tensor.reshape(*tensor.shape[:-1], k // group_size, group_size).to(mm_dtype)
    rotated = grouped @ h
    return rotated.reshape(tensor.shape).to(orig_dtype)


# ==============================================================================
# Quantization helpers
# ==============================================================================

def _stochastic_round_rowwise_int8(tensor: torch.Tensor, scale: torch.Tensor, seed: int) -> torch.Tensor:
    """Seeded stochastic rounding of tensor/scale into int8."""
    scaled = tensor.to(torch.float32) / scale
    generator = torch.Generator(device=scaled.device)
    generator.manual_seed(seed)
    floor = scaled.floor()
    frac = scaled - floor
    rnd = torch.rand(scaled.shape, generator=generator, device=scaled.device, dtype=torch.float32)
    q = floor + (rnd < frac).to(torch.float32)
    return q.clamp_(-128, 127).to(torch.int8)


# ==============================================================================
# Layouts
# ==============================================================================

class Int8RowwiseLayout(QuantizedLayout):
    """Symmetric absmax int8 with per-row (last-dim) fp32 scales."""

    MIN_SM_VERSION = int8_kernels.INT8_MIN_SM
    STORAGE_DTYPE = torch.int8

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        def _validate_tensor_fields(self):
            if isinstance(self.scale, torch.Tensor) and self.scale.dtype != torch.float32:
                object.__setattr__(self, "scale", self.scale.to(dtype=torch.float32))

    @classmethod
    def _pre_quant_transform(cls, tensor: torch.Tensor, rotation_dtype=torch.float32) -> torch.Tensor:
        return tensor

    @classmethod
    def _post_dequant_transform(cls, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @classmethod
    def quantize_activation(cls, tensor: torch.Tensor) -> QuantizedTensor:
        """Per-token dynamic quantization for the forward hot path.

        Identical math to ``quantize`` except the convrot rotation runs in the
        activation's own dtype (tensor-core rate) instead of fp32; the rounding
        difference is far below the int8 quantization grain.
        """
        qdata, params = cls.quantize(tensor, rotation_dtype=None)
        return QuantizedTensor(qdata, cls.__name__, params)

    @classmethod
    def quantize(cls, tensor: torch.Tensor, scale=None, stochastic_rounding=0, inplace_ops=False,
                 rotation_dtype=torch.float32, **kwargs):
        """Quantize a weight or activation tensor.

        The ``scale`` argument is accepted for interface compatibility and
        ignored: scales are always recalculated per row, which covers dynamic
        per-token activation quantization, ad-hoc weight quantization, and
        LoRA-bake requantization alike. Static ``input_scale`` values from
        checkpoints are deliberately not used.
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        rotated = cls._pre_quant_transform(tensor, rotation_dtype)

        if stochastic_rounding:
            absmax = rotated.abs().amax(dim=-1, keepdim=True).to(torch.float32)
            s = (absmax / 127.0).clamp(min=_SCALE_EPS)
            qdata = _stochastic_round_rowwise_int8(rotated, s, stochastic_rounding)
        elif rotated.is_cuda and rotated.ndim == 2:
            qdata, s = torch.ops.comfy_int8.quantize_rowwise(rotated)
        else:
            qdata, s = int8_kernels.quantize_rowwise_eager(rotated)

        params = cls.Params(scale=s, orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        full = qdata.to(torch.float32) * params.scale
        full = cls._post_dequant_transform(full)
        return full.to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor):
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params):
        return {
            "": qdata,
            "_scale": params.scale,
        }

    @classmethod
    def extra_state_dict_conf(cls) -> dict:
        """Extra keys merged into the per-layer comfy_quant JSON on save."""
        return {}


class Int8ConvRotLayout(Int8RowwiseLayout):
    """Int8 rowwise with block-diagonal regular-Hadamard rotation (ConvRot)."""

    GROUP_SIZE = 256

    @classmethod
    def _pre_quant_transform(cls, tensor: torch.Tensor, rotation_dtype=torch.float32) -> torch.Tensor:
        return rotate_groups(tensor, cls.GROUP_SIZE, compute_dtype=rotation_dtype)

    @classmethod
    def _post_dequant_transform(cls, tensor: torch.Tensor) -> torch.Tensor:
        # H is self-inverse: rotating again returns to original space.
        return rotate_groups(tensor, cls.GROUP_SIZE)

    @classmethod
    def extra_state_dict_conf(cls) -> dict:
        # "convrot"/"per_row" mirror the metadata written by ComfyUI-INT8-Fast
        # so checkpoints round-trip between implementations.
        return {"convrot": True, "convrot_groupsize": cls.GROUP_SIZE, "per_row": True}


# ==============================================================================
# Dispatch: fused int8 matmuls
#
# aten.linear is CompositeImplicitAutograd, so under __torch_dispatch__ it
# usually arrives decomposed as t() + addmm()/mm(). All three entry points are
# registered (the fp8 layout does the same); the t() handler keeps the
# QuantizedTensor wrapper through the transpose so addmm/mm still see two
# quantized operands.
# ==============================================================================

def _int8_gemm_or_none(a, b, bias, out_dtype):
    """Fused gemm for a [M, K] per-row-quantized activation against a weight
    that arrived either as [N, K] (linear) or transposed [K, N] (mm/addmm).
    Returns None when the operands don't fit the fused kernel."""
    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return None
    x_q, x_scale = a._qdata, a._params.scale
    w_q, w_scale = b._qdata, b._params.scale
    if x_q.ndim != 2 or w_q.ndim != 2:
        return None
    if x_scale.numel() not in (1, x_q.shape[0]):
        return None
    if w_q.shape[0] == x_q.shape[1]:  # transposed weight [K, N] from linear's decomposition
        w_q = w_q.t()
        w_scale = w_scale.reshape(-1)
    if w_q.shape[1] != x_q.shape[1]:
        return None
    if w_scale.numel() not in (1, w_q.shape[0]):
        return None
    try:
        return torch.ops.comfy_int8.gemm(x_q, x_scale, w_q, w_scale, bias, out_dtype)
    except (RuntimeError, TypeError) as exc:
        logger.warning("INT8 gemm failed: %s, falling back to dequantization", exc)
        return None


@register_layout_op(torch.ops.aten.linear.default, Int8RowwiseLayout)
def _handle_int8_linear(qt, args, kwargs):
    """INT8 linear: out = x_q @ w_q.T scaled per-token and per-row, fused.

    Mixed or plain operands fall back to dequantization, which for ConvRot
    also de-rotates, so the fallback is exactly the reference small-batch
    behavior.
    """
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    out_dtype = kwargs.get("out_dtype", getattr(input_tensor, "dtype", None))
    out = _int8_gemm_or_none(input_tensor, weight, bias, out_dtype)
    if out is not None:
        return out
    return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))


@register_layout_op(torch.ops.aten.mm.default, Int8RowwiseLayout)
def _handle_int8_mm(qt, args, kwargs):
    a, b = args[0], args[1]
    out_dtype = kwargs.get("out_dtype", getattr(a, "dtype", None))
    out = _int8_gemm_or_none(a, b, None, out_dtype)
    if out is not None:
        return out
    return torch.mm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.addmm.default, Int8RowwiseLayout)
def _handle_int8_addmm(qt, args, kwargs):
    bias, a, b = args[0], args[1], args[2]
    out_dtype = kwargs.get("out_dtype", getattr(a, "dtype", None))
    out = _int8_gemm_or_none(a, b, bias if not isinstance(bias, QuantizedTensor) else bias.dequantize(), out_dtype)
    if out is not None:
        return out
    return torch.addmm(*dequantize_args(args))


def _make_int8_shape_handler(aten_op):
    """Shape ops keep the wrapper: int8 is unpacked (1:1 element mapping), and
    for t() the per-row scale transposes along with the data."""

    def handler(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return aten_op(*args, **kwargs)
        new_qdata = aten_op(input_tensor._qdata, *args[1:], **kwargs)
        scale = input_tensor._params.scale
        if aten_op is torch.ops.aten.t.default and scale.ndim == 2:
            scale = scale.t()
        new_params = type(input_tensor._params)(
            scale=scale,
            orig_dtype=input_tensor._params.orig_dtype,
            orig_shape=tuple(new_qdata.shape),
        )
        return QuantizedTensor(new_qdata, input_tensor._layout_cls, new_params)

    return handler


register_layout_op(torch.ops.aten.t.default, Int8RowwiseLayout)(_make_int8_shape_handler(torch.ops.aten.t.default))

register_layout_class("Int8RowwiseLayout", Int8RowwiseLayout)
register_layout_class("Int8ConvRotLayout", Int8ConvRotLayout)
