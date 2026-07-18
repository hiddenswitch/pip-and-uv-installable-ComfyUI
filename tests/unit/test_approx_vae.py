import pytest
import torch
from safetensors.torch import save_file

from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.taesd.taesd import Decoder, Encoder


def build_workflow(vae_encoder_option="", vae_decoder_option=""):
    return {
        "1": {
            "inputs": {
                "vae_name": vae_encoder_option
            },
            "class_type": "VAELoader",
            "_meta": {
                "title": "Load VAE"
            }
        },
        "2": {
            "inputs": {
                "pixels": [
                    "3",
                    0
                ],
                "vae": [
                    "1",
                    0
                ]
            },
            "class_type": "VAEEncode",
            "_meta": {
                "title": "VAE Encode"
            }
        },
        "3": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1,
                "color": 0
            },
            "class_type": "EmptyImage",
            "_meta": {
                "title": "EmptyImage"
            }
        },
        "4": {
            "inputs": {
                "samples": [
                    "2",
                    0
                ],
                "vae": [
                    "5",
                    0
                ]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE Decode"
            }
        },
        "5": {
            "inputs": {
                "vae_name": vae_decoder_option
            },
            "class_type": "VAELoader",
            "_meta": {
                "title": "Load VAE"
            }
        },
        "6": {
            "inputs": {
                "filename_prefix": "test",
                "images": [
                    "4",
                    0
                ]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        }
    }


async def test_approx_vae_found(tmp_path_factory):
    workflow = build_workflow("taesdxl", "taesdxl")
    configuration = Configuration()
    configuration.cpu = True
    configuration.disable_known_models = True
    base_dir = tmp_path_factory.mktemp("test_approx_vae_found")
    configuration.base_directory = str(base_dir)
    async with Comfy(configuration) as comfy:
        from comfy.nodes.base_nodes import VAELoader

        assert "taesdxl" not in VAELoader.vae_list(), "should not be downloadable"
        with pytest.raises(ValueError):
            # should not attempt to download
            await comfy.queue_prompt_api(workflow)

        vae_approx_dir = base_dir / "models" / "vae_approx"
        vae_approx_dir.mkdir(parents=True, exist_ok=True)

        encoder = {name: torch.zeros_like(value) for name, value in Encoder().state_dict().items()}
        decoder = {name: torch.zeros_like(value) for name, value in Decoder().state_dict().items()}
        save_file(encoder, vae_approx_dir / "taesdxl_encoder.safetensors")
        save_file(decoder, vae_approx_dir / "taesdxl_decoder.safetensors")

        # now should work
        await comfy.queue_prompt_api(workflow)
