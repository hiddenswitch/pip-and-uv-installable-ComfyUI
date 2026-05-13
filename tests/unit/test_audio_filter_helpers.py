import pytest
import torch

from comfy.ldm.lightricks.vocoders import vocoder
from comfy.ldm.mmaudio.vae import alias_free_torch


@pytest.mark.parametrize("module", [vocoder, alias_free_torch])
def test_kaiser_sinc_filter_zero_cutoff_returns_zero_kernel(module):
    kernel = module.kaiser_sinc_filter1d(cutoff=0, half_width=0.6, kernel_size=5)

    assert kernel.shape == (1, 1, 5)
    assert torch.equal(kernel, torch.zeros(1, 1, 5, dtype=kernel.dtype))


@pytest.mark.parametrize("module", [vocoder, alias_free_torch])
def test_kaiser_sinc_filter_nonzero_cutoff_is_normalized(module):
    kernel = module.kaiser_sinc_filter1d(cutoff=0.25, half_width=0.6, kernel_size=9)

    assert kernel.shape == (1, 1, 9)
    assert kernel.sum().item() == pytest.approx(1.0)
