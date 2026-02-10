"""Tests for torchvision compatibility shim."""

from comfy.torchvision_compat import InterpolationMode


def test_interpolation_mode_available():
    """InterpolationMode should be importable regardless of torchvision version."""
    assert hasattr(InterpolationMode, "NEAREST")
    assert hasattr(InterpolationMode, "BILINEAR")


def test_transforms_patched():
    """After importing the compat module, torchvision.transforms.InterpolationMode should work."""
    from torchvision import transforms
    assert hasattr(transforms, "InterpolationMode")
    assert transforms.InterpolationMode.NEAREST == InterpolationMode.NEAREST
