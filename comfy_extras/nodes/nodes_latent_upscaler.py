import json
import math
from typing import Any, Optional

import torch

import comfy.model_management
import comfy.ops
import comfy.utils
from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
from comfy.model_downloader import get_filename_list_with_downloadable, get_full_path_or_raise
from comfy.model_management_types import ModelManageableStub


class LatentUpscaleModelManageable(ModelManageableStub):
    """Model management wrapper for latent upscale models."""

    def __init__(self, model: torch.nn.Module, ckpt_name: str):
        self.ckpt_name = ckpt_name
        self.model = model
        self.load_device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.unet_offload_device()
        self._input_size: tuple[int, ...] = (1, 32, 1, 64, 64)  # batch, channels, frames, height, width

    @property
    def current_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.offload_device

    @property
    def input_size(self) -> tuple[int, ...]:
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int, ...]):
        self._input_size = size

    def set_input_size_from_latent(self, latent: torch.Tensor):
        """Set input size from latent tensor shape."""
        self._input_size = tuple(latent.shape)

    def is_clone(self, other: Any) -> bool:
        return isinstance(other, LatentUpscaleModelManageable) and self.model is other.model

    def clone_has_same_weights(self, clone) -> bool:
        return self.is_clone(clone)

    def model_size(self) -> int:
        model_params_size = comfy.model_management.module_size(self.model)
        # Activation memory estimate from LTXVLatentUpsampler
        activation_mem = int(math.prod(self._input_size) * 3000.0)
        return model_params_size + activation_mem

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

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upscale model."""
        return self.model(latents)

    def __str__(self):
        if self.ckpt_name is not None:
            return f"<LatentUpscaleModelManageable for {self.ckpt_name} ({self.model.__class__.__name__})>"
        else:
            return f"<LatentUpscaleModelManageable for {self.model.__class__.__name__}>"


class LatentUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_filename_list_with_downloadable("latent_upscale_models"),),
            }
        }

    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = get_full_path_or_raise("latent_upscale_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)

        if "blocks.0.block.0.conv.weight" in sd:
            config = {
                "in_channels": sd["in_conv.conv.weight"].shape[1],
                "out_channels": sd["out_conv.conv.weight"].shape[0],
                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
                "global_residual": False,
            }
            model_type = "720p"
            model = HunyuanVideo15SRModel(model_type, config)
            model.load_sd(sd)
        elif "up.0.block.0.conv1.conv.weight" in sd:
            sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
            config = {
                "z_channels": sd["conv_in.conv.weight"].shape[1],
                "out_channels": sd["conv_out.conv.weight"].shape[0],
                "block_out_channels": tuple(sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0] for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))),
            }
            model_type = "1080p"
            model = HunyuanVideo15SRModel(model_type, config)
            model.load_sd(sd)
        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
            config = json.loads(metadata["config"])
            model = LatentUpsampler.from_config(config, operations=comfy.ops.disable_weight_init).to(dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
            comfy.model_management.archive_model_dtypes(model)
            model.load_state_dict(sd, assign=True)
        else:
            raise ValueError(f"Unknown latent upscale model format in {model_name}")

        return (LatentUpscaleModelManageable(model, model_name),)


NODE_CLASS_MAPPINGS = {
    "LatentUpscaleModelLoader": LatentUpscaleModelLoader,
}
