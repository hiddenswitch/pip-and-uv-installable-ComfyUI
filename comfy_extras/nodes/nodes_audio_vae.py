from typing import Any, Optional

import torch

import comfy.model_management
import comfy.utils
from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
from comfy.model_downloader import get_filename_list_with_downloadable, get_full_path_or_raise
from comfy.model_management_types import ModelManageableStub
from comfy_api.latest import ComfyExtension, io


class AudioVAEModelManageable(ModelManageableStub):
    """Model management wrapper for Audio VAE models."""

    def __init__(self, model: AudioVAE, ckpt_name: str):
        self.ckpt_name = ckpt_name
        self.model = model
        self.load_device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.vae_offload_device()

    @property
    def current_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.offload_device

    def is_clone(self, other: Any) -> bool:
        return isinstance(other, AudioVAEModelManageable) and self.model is other.model

    def clone_has_same_weights(self, clone) -> bool:
        return self.is_clone(clone)

    def model_size(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def model_patches_to(self, arg: torch.device | torch.dtype):
        if isinstance(arg, torch.device):
            self.model.to(device=arg)
        else:
            self.model.to(dtype=arg)

    def model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def patch_model(self, device_to: torch.device | None = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> torch.nn.Module:
        if device_to is not None:
            self.model.to(device=device_to)
        return self.model

    def unpatch_model(self, device_to: torch.device | None = None, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        if device_to is not None:
            self.model.to(device=device_to)
        return self.model

    def encode(self, audio: dict) -> torch.Tensor:
        comfy.model_management.load_models_gpu([self])
        waveform = audio["waveform"].to(self.load_device)
        return self.model.encode(waveform, sample_rate=audio["sample_rate"])

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        comfy.model_management.load_models_gpu([self])
        return self.model.decode(latents.to(self.load_device))

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
        return self.model.num_of_latents_from_frames(frames_number, frame_rate)

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape)

    # Delegate AudioVAE properties
    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self.model.output_sample_rate

    @property
    def latent_channels(self) -> int:
        return self.model.latent_channels

    @property
    def latent_frequency_bins(self) -> int:
        return self.model.latent_frequency_bins

    @property
    def latents_per_second(self) -> float:
        return self.model.latents_per_second

    @property
    def mel_hop_length(self) -> int:
        return self.model.mel_hop_length

    @property
    def mel_bins(self) -> int:
        return self.model.mel_bins

    def __str__(self):
        if self.ckpt_name is not None:
            return f"<AudioVAEModelManageable for {self.ckpt_name}>"
        else:
            return f"<AudioVAEModelManageable for {self.model.__class__.__name__}>"


class LTXVAudioVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVAudioVAELoader",
            display_name="LTXV Audio VAE Loader",
            category="audio",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=get_filename_list_with_downloadable("checkpoints"),
                    tooltip="Audio VAE checkpoint to load.",
                )
            ],
            outputs=[io.Vae.Output(display_name="Audio VAE")],
        )

    @classmethod
    def execute(cls, ckpt_name: str) -> io.NodeOutput:
        ckpt_path = get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        sd = comfy.utils.state_dict_prefix_replace(sd, {"audio_vae.": "autoencoder."})
        model = AudioVAE(metadata=metadata)
        model.load_state_dict(sd, strict=False)
        return io.NodeOutput(AudioVAEModelManageable(model, ckpt_name))


class AudioVAEExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXVAudioVAELoader,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return AudioVAEExtension()
