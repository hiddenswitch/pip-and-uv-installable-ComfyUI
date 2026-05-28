import torch

from comfy_extras.nodes.nodes_audio_vae import AudioVAEModelManageable
from comfy_extras.nodes.nodes_lt_audio import LTXVAudioVAEDecode, LTXVEmptyLatentAudio


class DummyAudioVAE(torch.nn.Module):
    sample_rate = 16000
    output_sample_rate = 24000
    latent_channels = 8
    latent_frequency_bins = 4

    def __init__(self):
        super().__init__()
        self.autoencoder = type(
            "Autoencoder",
            (),
            {
                "encoder": type("Encoder", (), {"in_channels": 2})(),
                "decoder": type("Decoder", (), {"out_ch": 2})(),
            },
        )()

    def num_of_latents_from_frames(self, frames_number, frame_rate):
        return 3

    def encode(self, waveform, sample_rate=44100):
        assert waveform.shape == (2, 2, 10)
        assert sample_rate == 16000
        return torch.zeros((2, self.latent_channels, 3, self.latent_frequency_bins))

    def decode(self, latents):
        return torch.zeros((latents.shape[0], 2, 10), device=latents.device)


def test_audio_vae_model_manageable_supports_ltx_legacy_access():
    manageable = object.__new__(AudioVAEModelManageable)
    manageable.ckpt_name = "dummy.safetensors"
    manageable.model = DummyAudioVAE()
    manageable.load_device = torch.device("cpu")
    manageable.offload_device = torch.device("cpu")

    assert manageable.first_stage_model is manageable.model
    assert manageable.audio_sample_rate == 16000
    assert manageable.audio_sample_rate_output == 24000
    assert manageable.encode(torch.zeros((2, 10, 2))).shape == (2, 8, 3, 4)

    latent = LTXVEmptyLatentAudio.execute(
        frames_number=10,
        frame_rate=25,
        batch_size=2,
        audio_vae=manageable,
    )[0]

    assert latent["samples"].shape == (2, 8, 3, 4)

    decoded = LTXVAudioVAEDecode.execute({"samples": latent["samples"]}, manageable)[0]

    assert decoded["sample_rate"] == 24000
    assert decoded["waveform"].shape == (2, 2, 10)
