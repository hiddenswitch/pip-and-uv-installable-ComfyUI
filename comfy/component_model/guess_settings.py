"""Auto-detect best settings for the current machine.

Called from ``main_pre`` when ``--guess-settings`` is passed.  All detection
is done *without* importing ``torch`` so that environment variables can be set
before the CUDA runtime initialises.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cli_args_types import Configuration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# low-level probes (no torch)
# ---------------------------------------------------------------------------

def _total_ram_gb() -> float:
    """Return total physical RAM in GiB."""
    try:
        import psutil  # pylint: disable=import-outside-toplevel
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    # fallback: /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except (OSError, ValueError):
        pass

    return 0.0


def _has_nvidia_gpu() -> bool:
    """True if ``nvidia-smi`` is on PATH (works on Linux and Windows)."""
    import shutil  # pylint: disable=import-outside-toplevel
    return shutil.which("nvidia-smi") is not None


def _has_amd_gpu() -> bool:
    """Heuristic: check for ROCm tooling or ``/dev/kfd``."""
    import shutil  # pylint: disable=import-outside-toplevel
    if shutil.which("rocm-smi") is not None or shutil.which("rocminfo") is not None:
        return True
    if os.path.exists("/dev/kfd"):
        return True
    return False


_BENIGN_PROCESS_RE = re.compile(r"^(python|nv)", re.IGNORECASE)


def _competing_gpu_processes() -> list[str]:
    """Return names of non-Python, non-nv* processes using the NVIDIA GPU.

    Returns an empty list when ``nvidia-smi`` is unavailable or fails.
    """
    try:
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        names: list[str] = []
        for line in result.stdout.strip().splitlines():
            # Handle both Unix and Windows paths (nvidia-smi may return
            # full paths like C:\Program Files\Discord\Discord.exe)
            raw = line.strip().replace("\\", "/")
            proc = os.path.basename(raw)
            if proc and not _BENIGN_PROCESS_RE.match(proc):
                names.append(proc)
        return names
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def apply_guess_settings(configuration: Configuration) -> None:  # pylint: disable=too-many-branches
    """Mutate *configuration* with auto-detected defaults.

    Only touches settings that are still at their parser defaults so that
    explicit CLI flags always win.
    """
    from ..cli_args_types import PerformanceFeature  # pylint: disable=import-outside-toplevel

    is_nvidia = _has_nvidia_gpu()
    is_amd = _has_amd_gpu()
    ram_gb = _total_ram_gb()

    # -- RAM < 32 GB → disable pinned memory ----------------------------
    if ram_gb and ram_gb < 32 and not configuration.disable_pinned_memory:
        logger.info("guess-settings: %.1f GB RAM detected, disabling pinned memory", ram_gb)
        configuration.disable_pinned_memory = True

    # -- NVIDIA: add cublas_ops to --fast --------------------------------
    if is_nvidia:
        fast = set(configuration.fast) if configuration.fast else set()
        if PerformanceFeature.CublasOps not in fast:
            fast.add(PerformanceFeature.CublasOps)
            configuration.fast = list(fast)
            logger.info("guess-settings: NVIDIA GPU detected, enabling cublas_ops")

    # -- NVIDIA: competing GPU processes → novram -----------------------
    if is_nvidia:
        vram_fields = ("gpu_only", "highvram", "normalvram", "lowvram", "novram", "cpu")
        user_set_vram = any(getattr(configuration, f, False) for f in vram_fields)
        if not user_set_vram:
            procs = _competing_gpu_processes()
            if procs:
                logger.info("guess-settings: competing GPU processes detected (%s), enabling novram", ", ".join(procs))
                configuration.novram = True

    # -- AMD: fp32 VAE --------------------------------------------------
    if is_amd:
        vae_fields = ("fp16_vae", "fp32_vae", "bf16_vae")
        if not any(getattr(configuration, f, False) for f in vae_fields):
            logger.info("guess-settings: AMD GPU detected, enabling fp32 VAE")
            configuration.fp32_vae = True

    # -- attention backend ----------------------------------------------
    attn_fields = ("use_split_cross_attention", "use_quad_cross_attention",
                   "use_sage_attention", "use_flash_attention")
    user_set_attn = any(getattr(configuration, f, False) for f in attn_fields)
    if not user_set_attn:
        if _has_package("sageattention"):
            logger.info("guess-settings: sageattention found, enabling sage attention")
            configuration.use_sage_attention = True
        elif _has_package("xformers"):
            logger.info("guess-settings: xformers found, keeping xformers enabled")
            configuration.disable_xformers = False
        else:
            logger.info("guess-settings: using default PyTorch cross attention")
            configuration.use_pytorch_cross_attention = True
