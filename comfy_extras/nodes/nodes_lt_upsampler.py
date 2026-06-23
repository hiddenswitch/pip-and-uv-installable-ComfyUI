from comfy import model_management
from comfy_api.latest import ComfyExtension, IO
from typing_extensions import override
import math

from comfy.model_management import load_models_gpu


class LTXVLatentUpsampler(IO.ComfyNode):
    """
    Upsamples a video latent by a factor of 2.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LTXVLatentUpsampler",
            category="model/latent/ltxv",
            is_experimental=True,
            inputs=[
                IO.Latent.Input("samples"),
                IO.LatentUpscaleModel.Input("upscale_model"),
                IO.Vae.Input("vae"),
            ],
            outputs=[
                IO.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, upscale_model, vae) -> IO.NodeOutput:
        """
        Upsample the input latent using the provided model.

        Args:
            samples (dict): Input latent samples
            upscale_model: Loaded upscale model (either raw model or ModelManageable)
            vae: VAE model for normalization

        Returns:
            tuple: Tuple containing the upsampled latent
        """
        from .nodes_latent_upscaler import LatentUpscaleModelManageable

        latents = samples["samples"]
        input_dtype = latents.dtype

        # Check if upscale_model is a ModelManageable or raw model
        if isinstance(upscale_model, LatentUpscaleModelManageable):
            # New path: use model management
            upscale_model.set_input_size_from_latent(latents)
            load_models_gpu([upscale_model])

            model_dtype = upscale_model.model_dtype()
            latents = latents.to(dtype=model_dtype, device=upscale_model.current_device)

            latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
            upsampled_latents = upscale_model(latents)
        else:
            # Legacy path: manual device management
            device = model_management.get_torch_device()
            memory_required = model_management.module_size(upscale_model)

            model_dtype = next(upscale_model.parameters()).dtype

            memory_required += math.prod(latents.shape) * 3000.0  # TODO: more accurate
            model_management.free_memory(memory_required, device)

            try:
                upscale_model.to(device)

                latents = latents.to(dtype=model_dtype, device=device)

                latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
                upsampled_latents = upscale_model(latents)
            finally:
                upscale_model.cpu()

        upsampled_latents = vae.first_stage_model.per_channel_statistics.normalize(
            upsampled_latents
        )
        upsampled_latents = upsampled_latents.to(dtype=input_dtype, device=model_management.intermediate_device())

        return_dict = samples.copy()
        return_dict["samples"] = upsampled_latents
        return_dict.pop("noise_mask", None)
        return IO.NodeOutput(return_dict)

    upsample_latent = execute  # TODO: remove


class LTXVLatentUpsamplerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [LTXVLatentUpsampler]


async def comfy_entrypoint() -> LTXVLatentUpsamplerExtension:
    return LTXVLatentUpsamplerExtension()
