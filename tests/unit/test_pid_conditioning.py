import pytest
import torch

import comfy.latent_formats
from comfy_extras.nodes.nodes_pid import PiDConditioning


def _conditioning():
    return [[torch.zeros(1), {"existing": "value"}]]


def _lq_latent_for(latent_format: str, samples: torch.Tensor):
    output = PiDConditioning.execute(
        _conditioning(),
        {"samples": samples},
        latent_format=latent_format,
        degrade_sigma=0.25,
    )
    conditioning = output.result[0]
    values = conditioning[0][1]
    return values["lq_latent"], values["degrade_sigma"], values


def test_pid_conditioning_supports_sdxl_latents():
    samples = torch.ones((1, 4, 2, 3))

    lq_latent, degrade_sigma, values = _lq_latent_for("sdxl", samples)

    assert torch.allclose(lq_latent, comfy.latent_formats.SDXL().process_in(samples))
    assert lq_latent.shape == samples.shape
    assert degrade_sigma.tolist() == [0.25]
    assert values["existing"] == "value"


def test_pid_conditioning_supports_qwen_image_latents_and_squeezes_temporal_dim():
    samples = torch.ones((1, 16, 2, 4, 4))
    expected = comfy.latent_formats.Wan21().process_in(samples)[:, :, 0]

    lq_latent, _, _ = _lq_latent_for("qwenimage", samples)

    assert torch.allclose(lq_latent, expected)
    assert lq_latent.shape == (1, 16, 4, 4)


def test_pid_conditioning_keeps_flux_auto_detection_for_flux2_channels():
    samples = torch.ones((1, 128, 2, 3))

    lq_latent, _, _ = _lq_latent_for("flux", samples)

    assert torch.allclose(lq_latent, comfy.latent_formats.Flux2().process_in(samples))


def test_pid_conditioning_rejects_unknown_latent_format():
    with pytest.raises(ValueError, match="Unknown latent_format"):
        PiDConditioning.execute(
            _conditioning(),
            {"samples": torch.ones((1, 4, 2, 3))},
            latent_format="not-a-format",
            degrade_sigma=0.0,
        )
