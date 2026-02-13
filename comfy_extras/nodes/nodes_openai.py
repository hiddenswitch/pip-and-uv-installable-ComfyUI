import os
from io import BytesIO
from typing import Literal, Optional

import numpy as np
import requests
import torch
from PIL import Image
from openai import OpenAI

from comfy.cli_args import args
from comfy.component_model.tensor_types import RGBImageBatch
from comfy.language.language_types import LanguageModel
from comfy.language.remote_model import RemoteLanguageModel
from comfy.nodes.package_typing import CustomNode, InputTypes


class _Client:
    _client: Optional[OpenAI] = None

    @staticmethod
    def instance() -> OpenAI:
        if _Client._client is None:
            open_ai_api_key = args.openai_api_key
            _Client._client = OpenAI(
                api_key=open_ai_api_key,
            )

        return _Client._client


def validate_has_key():
    open_api_key = os.environ.get("OPENAI_API_KEY", args.openai_api_key)
    if open_api_key is None or open_api_key == "":
        return "set OPENAI_API_KEY environment variable"
    return True


class OpenAILanguageModelLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": (["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], {"default": "gpt-3.5-turbo"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("language model",)

    FUNCTION = "execute"
    CATEGORY = "openai"

    def execute(self, model: str) -> tuple[LanguageModel]:
        return RemoteLanguageModel(f"openai:{model}"),

    @classmethod
    def VALIDATE_INPUTS(cls):
        return validate_has_key()


class DallEGenerate(CustomNode):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (["dall-e-2", "dall-e-3"], {"default": "dall-e-3"}),
            "text": ("STRING", {"multiline": True}),
            "size": (["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"], {"default": "1024x1024"}),
            "quality": (["standard", "hd"], {"default": "standard"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "revised prompt")
    FUNCTION = "generate"

    CATEGORY = "openai"

    def generate(self,
                 model: Literal["dall-e-2", "dall-e-3"],
                 text: str,
                 size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
                 quality: Literal["standard", "hd"]) -> tuple[RGBImageBatch, str]:
        response = _Client.instance().images.generate(
            model=model,
            prompt=text,
            size=size,
            quality=quality,
            n=1,
        )

        image_url = response.data[0].url
        image_response = requests.get(image_url)

        img = Image.open(BytesIO(image_response.content))

        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image, response.data[0].revised_prompt

    @classmethod
    def VALIDATE_INPUTS(cls):
        return validate_has_key()


NODE_CLASS_MAPPINGS = {
    "DallEGenerate": DallEGenerate,
    "OpenAILanguageModelLoader": OpenAILanguageModelLoader
}
