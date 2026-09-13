from types import SimpleNamespace

import pytest
import torch

from comfy.k_diffusion import sampling


@pytest.mark.parametrize("sampler", [sampling.sample_sa_solver, sampling.sample_sa_solver_pece])
@pytest.mark.parametrize("tau", [0.0, 1.0])
def test_sa_solver_completes_predictor_and_corrector_steps(sampler, tau):
    calls = []

    def model(x, sigma, **kwargs):
        calls.append(sigma)
        return torch.full_like(x, 0.25)

    model.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda name: SimpleNamespace()))
    x = torch.ones(1, 2, 4, 4)
    sigmas = torch.tensor([4.0, 2.0, 1.0, 0.5, 0.0])
    result = sampler(model, x, sigmas, disable=True, tau_func=lambda sigma: tau, noise_sampler=lambda *args: torch.ones_like(x))

    assert len(calls) == (7 if sampler is sampling.sample_sa_solver_pece else 4)
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.device == x.device
    torch.testing.assert_close(result, torch.full_like(x, 0.25))
