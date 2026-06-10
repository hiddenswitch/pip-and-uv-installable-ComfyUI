"""INT8 w8a8 kernels: per-token dynamic quantization and fused int8 GEMM.

This module deliberately manages its own triton availability. It must not be
gated by the comfy_kitchen "triton" backend registry: that backend is disabled
on NVIDIA compute capability < 8.9 because comfy_kitchen's fp8 kernels use
fp8e4nv, which triton cannot compile on Ampere. INT8 tensor-core kernels work
on sm_75 and newer, including all of Ampere.

The two public entry points are torch custom ops so that torch.compile treats
them as opaque nodes with correct meta functions (no graph breaks):

    torch.ops.comfy_int8.quantize_rowwise(x)        -> (int8 [M, K], fp32 [M, 1])
    torch.ops.comfy_int8.gemm(x_q, x_scale, w_q, w_scale, bias, out_dtype)

Quantization is symmetric absmax per row over the last dimension, no zero
points: q = round(x / s).clamp(-128, 127) with s = absmax / 127.
"""

import logging
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)

INT8_MIN_SM = (7, 5)
_SCALE_EPS = 1e-12

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@lru_cache(maxsize=1)
def _min_cuda_capability():
    if not torch.cuda.is_available():
        return None
    try:
        return min(
            (torch.cuda.get_device_properties(i).major,
             torch.cuda.get_device_properties(i).minor)
            for i in range(torch.cuda.device_count())
        )
    except (RuntimeError, ValueError, AssertionError):
        return None


def int8_compute_capable() -> bool:
    """True when the CUDA devices have int8 tensor-core support (sm_75+)."""
    cap = _min_cuda_capability()
    return cap is not None and cap >= INT8_MIN_SM


def int8_fast_available() -> bool:
    """True when the triton fast path can run: triton importable + sm_75+ CUDA."""
    return TRITON_AVAILABLE and int8_compute_capable()


# ==============================================================================
# Triton kernels
# ==============================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _int8_rowwise_quant_kernel(
        x_ptr, q_ptr, scale_ptr,
        K,
        stride_xm, stride_xk,
        stride_qm, stride_qk,
        BLOCK_K: tl.constexpr,
    ):
        # One program per row. K is chunked so arbitrarily wide rows (large ff
        # dims) never exceed one block's shared memory.
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_xm
        q_row = q_ptr + row * stride_qm

        absmax = 0.0
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs * stride_xk, mask=mask, other=0.0).to(tl.float32)
            absmax = tl.maximum(absmax, tl.max(tl.abs(x), axis=0))

        scale = tl.maximum(absmax / 127.0, 1e-12)
        inv_scale = 1.0 / scale
        tl.store(scale_ptr + row, scale)

        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs * stride_xk, mask=mask, other=0.0).to(tl.float32)
            q = x * inv_scale
            # round half up via floor(q + 0.5): portable across triton builds
            # (CUDA and ROCm), unlike libdevice.rint / tl.math intrinsics
            q = tl.floor(q + 0.5)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(q_row + offs * stride_qk, q.to(tl.int8), mask=mask)

    _GEMM_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ]

    @triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _int8_gemm_dequant_kernel(
        x_ptr, w_ptr, out_ptr,
        x_scale_ptr, w_scale_ptr, bias_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # C[m, n] = (sum_k A[m, k] * W[n, k]) * x_scale[m] * w_scale[n] + bias[n]
        # W is [N, K] and read transposed through its strides.
        # Grouped (swizzled) program ordering keeps tiles that share weight
        # columns resident in L2 together.
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Row/col offsets wrap modulo M/N so the inner-loop loads never need
        # m/n bounds masks (only the K mask); duplicated rows are discarded by
        # the store mask.
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + offs_k) < K
            a = tl.load(x_ptrs, mask=k_mask[None, :], other=0)
            b = tl.load(w_ptrs, mask=k_mask[:, None], other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        x_scale = tl.load(x_scale_ptr + offs_m).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + offs_n).to(tl.float32)
        out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n).to(tl.float32)
            out += bias[None, :]

        out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        out_ptrs = out_ptr + out_m[:, None] * stride_om + out_n[None, :] * stride_on
        out_mask = (out_m[:, None] < M) & (out_n[None, :] < N)
        tl.store(out_ptrs, out.to(OUT_DTYPE), mask=out_mask)

    _TL_OUT_DTYPES = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }


# ==============================================================================
# Eager reference implementations (CPU fallback / no-triton fallback)
# ==============================================================================

def quantize_rowwise_eager(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax quantization per row over the last dimension."""
    absmax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (absmax / 127.0).clamp(min=_SCALE_EPS)
    q = (x.to(torch.float32) / scale).round_().clamp_(-128, 127).to(torch.int8)
    return q, scale


def _gemm_dequant_eager(x_q, x_scale, w_q, w_scale, bias, out_dtype):
    out = (x_q.to(torch.float32) * x_scale) @ (w_q.to(torch.float32) * w_scale).t()
    if bias is not None:
        out += bias.to(torch.float32)
    return out.to(out_dtype)


def _int_mm_supported(x_q: torch.Tensor, w_q: torch.Tensor) -> bool:
    # cublas int8 gemm constraints for torch._int_mm
    m, k = x_q.shape
    n = w_q.shape[0]
    return m > 16 and k % 8 == 0 and n % 8 == 0 and k >= 32


def _gemm_dequant_int_mm(x_q, x_scale, w_q, w_scale, bias, out_dtype):
    acc = torch._int_mm(x_q, w_q.t())
    out = acc.to(torch.float32) * x_scale * w_scale.t()
    if bias is not None:
        out += bias.to(torch.float32)
    return out.to(out_dtype)


# ==============================================================================
# Custom ops (compile-visible, opaque to dynamo)
# ==============================================================================

@torch.library.custom_op("comfy_int8::quantize_rowwise", mutates_args=())
def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if TRITON_AVAILABLE and x.is_cuda and x.ndim == 2 and x.numel() > 0:
        m, k = x.shape
        q = torch.empty((m, k), device=x.device, dtype=torch.int8)
        scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
        block_k = min(4096, triton.next_power_of_2(max(k, 16)))
        _int8_rowwise_quant_kernel[(m,)](
            x, q, scale,
            k,
            x.stride(0), x.stride(1),
            q.stride(0), q.stride(1),
            BLOCK_K=block_k,
        )
        return q, scale
    return quantize_rowwise_eager(x)


@quantize_rowwise.register_fake
def _(x):
    scale_shape = list(x.shape[:-1]) + [1]
    return (
        torch.empty(x.shape, device=x.device, dtype=torch.int8),
        torch.empty(scale_shape, device=x.device, dtype=torch.float32),
    )


@torch.library.custom_op("comfy_int8::gemm", mutates_args=())
def gemm(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """out = (x_q @ w_q.T) * x_scale * w_scale + bias, cast to out_dtype.

    x_q: int8 [M, K]; x_scale: fp32 [M] or [M, 1]
    w_q: int8 [N, K]; w_scale: fp32 [N], [N, 1] or scalar (legacy tensorwise)
    """
    m, k = x_q.shape
    n = w_q.shape[0]
    x_scale2d = x_scale.reshape(m, 1).to(torch.float32)
    if w_scale.numel() == 1:
        w_scale2d = w_scale.reshape(1, 1).to(torch.float32).expand(n, 1).contiguous()
    else:
        w_scale2d = w_scale.reshape(n, 1).to(torch.float32)

    if TRITON_AVAILABLE and x_q.is_cuda and m > 0:
        tl_out_dtype = _TL_OUT_DTYPES.get(out_dtype)
        # tl.dot's 16-minimum applies to the constexpr tile sizes, not M/N/K:
        # masked loads make any actual shape work, including M=1.
        if tl_out_dtype is not None:
            out = torch.empty((m, n), device=x_q.device, dtype=out_dtype)
            x_c = x_q.contiguous()
            w_c = w_q.contiguous()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            )
            _int8_gemm_dequant_kernel[grid](
                x_c, w_c, out,
                x_scale2d, w_scale2d,
                bias if bias is not None else x_c,
                m, n, k,
                x_c.stride(0), x_c.stride(1),
                w_c.stride(0), w_c.stride(1),
                out.stride(0), out.stride(1),
                HAS_BIAS=bias is not None,
                OUT_DTYPE=tl_out_dtype,
            )
            return out

    if x_q.is_cuda and _int_mm_supported(x_q, w_q):
        try:
            return _gemm_dequant_int_mm(x_q.contiguous(), x_scale2d, w_q.contiguous(), w_scale2d, bias, out_dtype)
        except RuntimeError as exc:
            logger.debug("torch._int_mm failed, using dequantized matmul: %s", exc)

    return _gemm_dequant_eager(x_q, x_scale2d, w_q, w_scale2d, bias, out_dtype)


@gemm.register_fake
def _(x_q, x_scale, w_q, w_scale, bias, out_dtype):
    return torch.empty((x_q.shape[0], w_q.shape[0]), device=x_q.device, dtype=out_dtype)


# These ops are inference-only. Registering an explicitly raising backward lets
# torch.compile build graphs when an input accidentally carries requires_grad
# (e.g. a freshly constructed norm Parameter); the error only fires if someone
# actually calls backward through them. Training uses QuantLinearFunc instead.
def _no_backward(ctx, *grads):
    raise NotImplementedError("comfy_int8 ops are inference-only and have no gradient")


quantize_rowwise.register_autograd(_no_backward, setup_context=lambda ctx, inputs, output: None)
gemm.register_autograd(_no_backward, setup_context=lambda ctx, inputs, output: None)
