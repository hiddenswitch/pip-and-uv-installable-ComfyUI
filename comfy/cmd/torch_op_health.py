"""Per-op health check for torch on the current GPU backend.

Each op runs in its own fresh ``python -c`` subprocess so a kernel
crash in one op (e.g. UR_RESULT_ERROR_DEVICE_LOST on Intel Arc DG2's
broken triu kernel) cannot poison subsequent tests. The order in which
cases are listed is irrelevant — the only state shared across cases is
the GPU itself, which the kernel driver tears down between processes.

The case list covers the ops comfy actually exercises in the SD/CLIP/
VAE/UNet path: allocs + dtype casts, Linear / matmul / bmm, layer_norm
/ group_norm, GELU / SiLU / softmax, embedding lookup, conv1d/2d/3d
+ conv_transpose2d, the triangular family (triu/tril, known-buggy on
Intel DG2), and scaled_dot_product_attention under several call shapes.

Returned as ``list[tuple[str, bool, str]]`` matching the
``_CheckResult`` shape used by ``integrity_check``.
"""
from __future__ import annotations

import subprocess
import sys


# Subprocess template. {body} is a single-line Python statement (or
# semicolon-joined statements). The subprocess imports torch, runs the
# body on the requested device, calls device sync, and exits 0 (PASS)
# or 1 (FAIL — traceback to stderr).
_TEMPLATE = (
    "import sys, traceback, torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "device = torch.device({device!r})\n"
    "sync = getattr(torch, {sync_module!r}, None)\n"
    "try:\n"
    "    {body}\n"
    "    if sync is not None:\n"
    "        sync.synchronize()\n"
    "    sys.exit(0)\n"
    "except SystemExit:\n"
    "    raise\n"
    "except BaseException:\n"
    "    traceback.print_exc()\n"
    "    sys.exit(1)\n"
)


# (display_name, code_body). Executed verbatim against {device}; multiple
# statements are joined with ';' to keep the wrapper a single .format().
_CASES: list[tuple[str, str]] = [
    # --- elementary allocs (does the device exist + take simple kernels?) ---
    ("alloc fp32", "torch.randn(1024, 1024, dtype=torch.float32, device=device)"),
    ("alloc fp16", "torch.randn(1024, 1024, dtype=torch.float16, device=device)"),
    ("alloc bf16", "torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)"),

    # --- dtype/device transfer ---
    ("Tensor.to(fp32) from fp16",
     "torch.randn(64, 768, dtype=torch.float16, device=device).to(torch.float32)"),
    ("Parameter.to(fp32) from fp16",
     "nn.Parameter(torch.randn(768, dtype=torch.float16, device=device)).to(torch.float32)"),
    ("CPU->GPU transfer fp16",
     "torch.randn(768, 768, dtype=torch.float16).to(device)"),

    # --- matmul / linear ---
    ("matmul fp16",
     "(torch.randn(1024, 1024, dtype=torch.float16, device=device) @ "
     "torch.randn(1024, 1024, dtype=torch.float16, device=device))"),
    ("F.linear fp32",
     "F.linear("
     "torch.randn(1, 77, 768, dtype=torch.float32, device=device),"
     "torch.randn(768, 768, dtype=torch.float32, device=device),"
     "torch.randn(768, dtype=torch.float32, device=device))"),

    # --- norms / activations ---
    ("F.layer_norm fp32",
     "F.layer_norm("
     "torch.randn(1, 77, 768, dtype=torch.float32, device=device), (768,),"
     "torch.randn(768, dtype=torch.float32, device=device),"
     "torch.randn(768, dtype=torch.float32, device=device))"),
    ("F.group_norm fp32",
     "F.group_norm("
     "torch.randn(1, 32, 64, 64, dtype=torch.float32, device=device), 32,"
     "torch.randn(32, dtype=torch.float32, device=device),"
     "torch.randn(32, dtype=torch.float32, device=device))"),
    ("F.gelu fp32",
     "F.gelu(torch.randn(1, 77, 3072, dtype=torch.float32, device=device))"),
    ("F.silu fp32",
     "F.silu(torch.randn(1, 320, 64, 64, dtype=torch.float32, device=device))"),
    ("F.softmax fp32",
     "F.softmax(torch.randn(1, 12, 77, 77, dtype=torch.float32, device=device), dim=-1)"),

    # --- embedding ---
    ("F.embedding fp16 vocab=49408",
     "F.embedding("
     "torch.randint(0, 49408, (1, 77), device=device),"
     "torch.randn(49408, 768, dtype=torch.float16, device=device))"),

    # --- conv (VAE / UNet) ---
    ("F.conv2d fp32 (1,3,64,64)",
     "F.conv2d("
     "torch.randn(1, 3, 64, 64, dtype=torch.float32, device=device),"
     "torch.randn(32, 3, 3, 3, dtype=torch.float32, device=device),"
     "torch.randn(32, dtype=torch.float32, device=device), padding=1)"),
    ("F.conv2d fp16 UNet-shape",
     "F.conv2d("
     "torch.randn(1, 320, 64, 64, dtype=torch.float16, device=device),"
     "torch.randn(320, 320, 3, 3, dtype=torch.float16, device=device),"
     "torch.randn(320, dtype=torch.float16, device=device), padding=1)"),
    ("F.conv1d fp32",
     "F.conv1d("
     "torch.randn(1, 16, 64, dtype=torch.float32, device=device),"
     "torch.randn(32, 16, 3, dtype=torch.float32, device=device),"
     "torch.randn(32, dtype=torch.float32, device=device), padding=1)"),
    ("F.conv3d fp32 (video VAE)",
     "F.conv3d("
     "torch.randn(1, 4, 8, 32, 32, dtype=torch.float32, device=device),"
     "torch.randn(8, 4, 3, 3, 3, dtype=torch.float32, device=device),"
     "torch.randn(8, dtype=torch.float32, device=device), padding=1)"),
    ("F.conv_transpose2d fp32",
     "F.conv_transpose2d("
     "torch.randn(1, 32, 32, 32, dtype=torch.float32, device=device),"
     "torch.randn(32, 16, 4, 4, dtype=torch.float32, device=device),"
     "torch.randn(16, dtype=torch.float32, device=device), stride=2, padding=1)"),

    # --- triangular (causal masks; broken on Intel Arc DG2 + torch+xpu 2.9-2.11) ---
    ("torch.full + .triu_(1)",
     "torch.full((77, 77), -1e9, dtype=torch.float32, device=device).triu_(1)"),
    ("torch.triu",
     "torch.triu(torch.full((77, 77), -1e9, dtype=torch.float32, device=device), diagonal=1)"),
    ("torch.tril",
     "torch.tril(torch.full((77, 77), -1e9, dtype=torch.float32, device=device), diagonal=-1)"),

    # --- scaled-dot-product attention (broken on Intel Arc DG2 too: oneDNN
    # 'could not create a primitive' / 'OpenCL device not found') ---
    ("F.sdpa fp32 no mask",
     "q = torch.randn(1, 12, 77, 64, dtype=torch.float32, device=device);"
     "k = torch.randn(1, 12, 77, 64, dtype=torch.float32, device=device);"
     "v = torch.randn(1, 12, 77, 64, dtype=torch.float32, device=device);"
     "F.scaled_dot_product_attention(q, k, v)"),
    ("F.sdpa fp32 + CPU-built causal mask",
     "q = torch.randn(1, 12, 77, 64, dtype=torch.float32, device=device);"
     "k = torch.randn(1, 12, 77, 64, dtype=torch.float32, device=device);"
     "v = torch.randn(1, 12, 77, 64, dtype=torch.float32, device=device);"
     "mask = torch.full((77, 77), -1e9, dtype=torch.float32).triu_(1).to(device);"
     "F.scaled_dot_product_attention(q, k, v, attn_mask=mask)"),
]


def _detect_device() -> tuple[str | None, str]:
    """Return (device_string, sync_module_name) or (None, '') if no GPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda", "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "mps"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu", "npu"
    return None, ""


def run_torch_ops_check(timeout_s: float = 60.0) -> list[tuple[str, bool | None, str]]:
    """Run every case in a fresh subprocess; return one (name, ok, msg) per case.

    Returns a single ``(_, None, msg)`` entry if no GPU is available.
    """
    device, sync_mod = _detect_device()
    if device is None:
        return [("torch ops health", None, "no GPU device available")]

    results: list[tuple[str, bool | None, str]] = []
    for name, body in _CASES:
        snippet = _TEMPLATE.format(device=device, sync_module=sync_mod, body=body)
        label = f"torch op [{device}]: {name}"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", snippet],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            results.append((label, False, f"timeout after {timeout_s}s"))
            continue
        if proc.returncode == 0:
            results.append((label, True, "OK"))
        else:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            msg = tail[-1] if tail else f"exit code {proc.returncode}"
            results.append((label, False, msg))
    return results
