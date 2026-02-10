"""Compatibility shim for torchvision InterpolationMode.

Some torchvision builds (e.g. CUDA-specific wheels from the PyTorch index
installed via ``uv --torch-backend=auto``) do not export
``InterpolationMode`` from ``torchvision.transforms``, even though the
upstream source code does.  The symbol may live in a different submodule
or be absent entirely.

This module locates ``InterpolationMode`` by trying every known location.
If it cannot be found at all, we define a compatible enum ourselves.
Either way we monkey-patch ``torchvision.transforms`` (and ``.functional``)
so that third-party packages (transformers, spandrel, diffusers) that
``from torchvision.transforms import InterpolationMode`` will succeed.
"""

from __future__ import annotations

# pylint: disable=broad-exception-caught,invalid-name

InterpolationMode = None  # type: ignore[assignment]

# 1. Try every location where InterpolationMode has lived across versions.
for _modpath in (
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "torchvision.transforms.v2",
    "torchvision.transforms.v2.functional",
):
    try:
        _mod = __import__(_modpath, fromlist=["InterpolationMode"])
        _candidate = getattr(_mod, "InterpolationMode", None)
        if _candidate is not None:
            InterpolationMode = _candidate
            break
    except Exception:
        continue

# 2. If none of the imports produced a result, define our own enum that is
#    value-compatible with the real one (the values are just strings).
if InterpolationMode is None:
    from enum import Enum

    class InterpolationMode(Enum):  # type: ignore[no-redef]  # pylint: disable=function-redefined
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

# 3. Monkey-patch so third-party code finds the symbol in the usual locations.
try:
    import torchvision.transforms as _transforms  # pylint: disable=import-outside-toplevel
    if not hasattr(_transforms, "InterpolationMode"):
        _transforms.InterpolationMode = InterpolationMode  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import torchvision.transforms.functional as _functional  # pylint: disable=import-outside-toplevel
    if not hasattr(_functional, "InterpolationMode"):
        _functional.InterpolationMode = InterpolationMode  # type: ignore[attr-defined]
except Exception:
    pass

__all__ = ["InterpolationMode"]
