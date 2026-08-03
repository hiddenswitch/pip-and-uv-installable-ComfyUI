from unittest import mock

import pytest
import torch

from comfy.ldm.minimax.audio_vae import LowPassFilter1d, Snake1d, SnakeBeta, UpSample1d
from comfy.ldm.minimax.vae import ViT3DDecoder


@pytest.mark.parametrize(
    ("module", "convolution"),
    [
        (UpSample1d(), "conv_transpose1d"),
        (LowPassFilter1d(), "conv1d"),
    ],
)
def test_resampling_filter_follows_input_device(module, convolution):
    """VAE resampling must work when its registered filter was not migrated with the model."""
    input_tensor = torch.empty((1, 2, 16), device="meta")

    def assert_same_device(input_, weight, **kwargs):
        assert weight.device == input_.device
        return input_

    with mock.patch(
        f"comfy.ldm.minimax.audio_vae.F.{convolution}",
        side_effect=assert_same_device,
    ):
        module(input_tensor)


@pytest.mark.parametrize("activation", [Snake1d(2), SnakeBeta(2)])
def test_snake_parameters_follow_input_device(activation):
    """NO_VRAM may leave standalone parameters offloaded while activations run on the GPU."""
    output = activation(torch.empty((1, 2, 16), device="meta"))

    assert output.device.type == "meta"


def test_video_vae_register_tokens_follow_input_device():
    decoder = ViT3DDecoder(
        patch_size=1,
        patch_size_t=1,
        in_channels=2,
        out_channels=2,
        num_layers=0,
        heads=1,
        dim_head=2,
        rope_dim_ratio=0.5,
        num_register_tokens=1,
    )
    decoder.x_embedder = torch.nn.Identity()
    decoder.norm_out = torch.nn.Identity()
    decoder.proj_out = torch.nn.Identity()

    output = decoder(torch.empty((1, 2, 1, 1, 1), device="meta"))

    assert output.device.type == "meta"
