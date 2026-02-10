"""Compatibility shim for torchvision API changes.

In torchvision >= 0.20, ``InterpolationMode`` moved from
``torchvision.transforms`` to ``torchvision.transforms.v2``.  Some builds
(e.g. ``--torch-backend=auto`` on Windows/CUDA 12.7) ship a torchvision
where the old location no longer exports the symbol.

This module re-exports a working ``InterpolationMode`` regardless of
torchvision version, and monkey-patches ``torchvision.transforms`` so that
third-party packages (transformers, spandrel, diffusers) that import from
the old location also work.
"""

from __future__ import annotations

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    from torchvision.transforms.v2 import InterpolationMode  # type: ignore[no-redef]

    # Patch the old location so that third-party code importing from
    # torchvision.transforms.InterpolationMode also works.
    import torchvision.transforms as _transforms
    _transforms.InterpolationMode = InterpolationMode  # type: ignore[attr-defined]

    # Some packages import from torchvision.transforms.functional directly
    try:
        import torchvision.transforms.functional as _functional
        if not hasattr(_functional, "InterpolationMode"):
            _functional.InterpolationMode = InterpolationMode  # type: ignore[attr-defined]
    except ImportError:
        pass

__all__ = ["InterpolationMode"]
