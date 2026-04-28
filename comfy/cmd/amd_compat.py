"""Auto-set HSA_OVERRIDE_GFX_VERSION when local AMD GPU isn't in torch's compiled arch list.

Designed to run *before* torch is imported, so the env var is in place when
torch's HIP runtime initializes. Uses two probes that don't require torch:

* ``_torch_compiled_amd_arches()`` mmap-scans torch's ROCm shared libraries
  (``libtorch_hip.so`` etc.) for ``gfxNNNN`` ASCII strings embedded in the
  HSA bitcode metadata — much faster than spawning a subprocess and avoids
  the side effects of importing torch.
* ``_local_amd_gpu_arch()`` runs ``rocminfo`` (or falls back to scanning
  ``/sys/class/drm/card*/device/uevent`` for PCI device IDs) to determine
  the installed GPU's architecture.

When the local arch isn't compiled in but a same-family fallback is (e.g.
gfx1102 → gfx1100 for RX 7600 on a wheel built for gfx1100/1101 only), we
set ``HSA_OVERRIDE_GFX_VERSION`` to the fallback's version. The user can
short-circuit this entirely by exporting the variable themselves.
"""
from __future__ import annotations

import os
import re
import subprocess
import shutil
from importlib.util import find_spec
from typing import Iterable, Optional

# Same-family fallback: when local GPU is the key, accept any value's
# arch as a substitute kernel. RDNA 3 (11.0.x) and RDNA 2 (10.3.x) only.
_FAMILY_FALLBACK: dict[str, list[str]] = {
    "gfx1102": ["gfx1100"],
    "gfx1103": ["gfx1100"],
    "gfx1031": ["gfx1030"],
    "gfx1032": ["gfx1030"],
    "gfx1034": ["gfx1030"],
    "gfx1035": ["gfx1030"],
    "gfx1036": ["gfx1030"],
}

# PCI device id -> gfx arch. Used as fallback when rocminfo isn't on PATH.
_PCI_DEVICE_ID_TO_ARCH: dict[str, str] = {
    "7480": "gfx1102",   # Navi 33 — RX 7600 / 7600 XT / 7700S
    "7483": "gfx1102",   # Navi 33 mobile
    "73ff": "gfx1031",   # Navi 23 — RX 6600 / 6700 XT
    "73df": "gfx1031",   # Navi 22
    "73a5": "gfx1030",   # Navi 21 — RX 6800 / 6900
    "744c": "gfx1100",   # Navi 31 — RX 7900
    "7470": "gfx1101",   # Navi 32 — RX 7800 / 7700
}

_GFX_ARCH_RE = re.compile(rb"gfx[0-9]{3,4}")


def _torch_compiled_amd_arches() -> Optional[list[str]]:
    """Return torch's compiled-in gfx arches, or None if torch isn't ROCm.

    Scans torch's ROCm shared libraries for ``gfx<N>`` strings. Doesn't
    import torch.
    """
    spec = find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    torch_root = spec.submodule_search_locations[0]
    lib_dir = os.path.join(torch_root, "lib")
    if not os.path.isdir(lib_dir):
        return None
    candidates = [
        "libtorch_hip.so",
        "libtorch_hsa.so",
        "libtorch_cuda.so",  # ROCm wheels reuse this filename
    ]
    arches: set[str] = set()
    found_rocm_so = False
    for name in candidates:
        path = os.path.join(lib_dir, name)
        if not os.path.isfile(path):
            continue
        found_rocm_so = True
        try:
            with open(path, "rb") as fh:
                chunk_size = 4 * 1024 * 1024
                while True:
                    chunk = fh.read(chunk_size)
                    if not chunk:
                        break
                    for m in _GFX_ARCH_RE.finditer(chunk):
                        arches.add(m.group().decode())
        except OSError:
            continue
    if not found_rocm_so:
        return None
    # Filter spurious matches (e.g. gfx80, gfxf hex prefixes) to actual arch ids.
    valid = {a for a in arches if re.fullmatch(r"gfx\d{3,4}", a)}
    return sorted(valid) if valid else []


def _local_amd_gpu_arch() -> Optional[str]:
    """Best-effort detect the installed AMD GPU's gfx arch."""
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    if os.path.isfile(rocminfo) and os.access(rocminfo, os.X_OK):
        try:
            out = subprocess.check_output(
                [rocminfo], stderr=subprocess.DEVNULL, timeout=10,
            ).decode("utf-8", errors="ignore")
            for line in out.splitlines():
                m = re.search(r"\bName:\s+(gfx\d{3,4})\b", line)
                if m:
                    return m.group(1)
        except (OSError, subprocess.SubprocessError):
            pass

    # Fallback: parse /sys/class/drm/card*/device/uevent for PCI device IDs.
    try:
        for entry in sorted(os.listdir("/sys/class/drm")):
            if not entry.startswith("card") or "-" in entry:
                continue
            uevent = f"/sys/class/drm/{entry}/device/uevent"
            if not os.path.isfile(uevent):
                continue
            try:
                content = open(uevent, "r", encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            m = re.search(r"PCI_ID=1002:([0-9A-Fa-f]+)", content)
            if m and m.group(1).lower() in _PCI_DEVICE_ID_TO_ARCH:
                return _PCI_DEVICE_ID_TO_ARCH[m.group(1).lower()]
    except OSError:
        pass

    return None


def _arch_to_hsa_version(arch: str) -> Optional[str]:
    if not arch.startswith("gfx"):
        return None
    suffix = arch[3:]
    if len(suffix) < 3:
        return None
    major, minor, stepping = suffix[:-2], suffix[-2], suffix[-1]
    if not major.isdigit() or not minor.isdigit():
        return None
    try:
        stepping_int = int(stepping, 16)
    except ValueError:
        return None
    return f"{major}.{minor}.{stepping_int}"


def _pick_fallback(local: str, compiled: Iterable[str]) -> Optional[str]:
    compiled_set = set(compiled)
    for candidate in _FAMILY_FALLBACK.get(local, ()):
        if candidate in compiled_set:
            return candidate
    return None


def maybe_set_hsa_override() -> Optional[str]:
    """Set HSA_OVERRIDE_GFX_VERSION when needed, returning the chosen value or None.

    Skipped if the user already exported it, if torch isn't a ROCm build, if
    the local GPU's arch is already in torch's compiled list, or if no
    same-family fallback is available.
    """
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        return None
    compiled = _torch_compiled_amd_arches()
    if compiled is None:
        return None  # not a ROCm torch build
    local = _local_amd_gpu_arch()
    if local is None:
        return None
    if local in compiled:
        return None  # native support already
    fallback = _pick_fallback(local, compiled)
    if fallback is None:
        return None
    version = _arch_to_hsa_version(fallback)
    if version is None:
        return None
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = version
    return version
