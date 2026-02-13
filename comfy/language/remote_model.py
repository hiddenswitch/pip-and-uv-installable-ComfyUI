from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, UserContent
from pydantic_ai.settings import ModelSettings

from .language_types import (
    LanguageModel,
    ProcessorResult,
    GENERATION_KWARGS_TYPE,
    TOKENS_TYPE,
    TransformerStreamedProgress,
    LanguagePrompt,
)
from ..component_model.tensor_types import RGBImageBatch
from ..utils import comfy_progress, ProgressBar

logger = logging.getLogger(__name__)


def _image_tensor_to_jpeg_bytes(image: torch.Tensor) -> bytes:
    pil_image = Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return buf.getvalue()


class RemoteLanguageModel(LanguageModel):
    def __init__(self, model_id: str):
        self._model_id = model_id

    @staticmethod
    def from_pretrained(ckpt_name: str, subfolder: Optional[str] = None) -> "RemoteLanguageModel":
        return RemoteLanguageModel(ckpt_name)

    def tokenize(
        self,
        prompt: str | LanguagePrompt,
        images: RGBImageBatch | None,
        videos: list[torch.Tensor] | None = None,
        chat_template: str | None = None,
    ) -> ProcessorResult:
        return {
            "inputs": prompt if isinstance(prompt, list) else [prompt],
            "attention_mask": torch.ones(1, 1),
            "images": images,
        }

    def generate(
        self,
        tokens: TOKENS_TYPE = None,
        max_new_tokens: int = 512,
        seed: int = 0,
        sampler: Optional[GENERATION_KWARGS_TYPE] = None,
        *args,
        **kwargs,
    ) -> str:
        sampler = sampler or {}
        inputs = tokens.get("inputs", [])
        images = tokens.get("images", None)

        system_prompt = None
        user_parts: list[UserContent] = []

        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
            messages: LanguagePrompt = inputs
            for msg in messages:
                if msg["role"] == "system":
                    content = msg["content"]
                    system_prompt = content if isinstance(content, str) else content.get("text", "")
                elif msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, str):
                        user_parts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if part.get("type") == "text":
                                user_parts.append(part["text"])
        else:
            if isinstance(inputs, list):
                user_parts.append("".join(str(s) for s in inputs))
            else:
                user_parts.append(str(inputs))

        if images is not None:
            image_list = images if isinstance(images, list) else [images[i] for i in range(images.shape[0])] if hasattr(images, 'shape') and len(images.shape) == 4 else [images]
            for img in image_list:
                if img is not None:
                    jpeg_bytes = _image_tensor_to_jpeg_bytes(img)
                    user_parts.append(BinaryContent(data=jpeg_bytes, media_type="image/jpeg"))

        settings: dict = {"max_tokens": max_new_tokens}
        if "temperature" in sampler:
            settings["temperature"] = sampler["temperature"]
        if "top_p" in sampler:
            settings["top_p"] = sampler["top_p"]
        model_settings = ModelSettings(**settings)

        agent = Agent(
            model=self._model_id,
            instructions=system_prompt or "",
        )

        progress_bar: ProgressBar
        with comfy_progress(total=max_new_tokens) as progress_bar:
            token_count = 0
            full_response = ""

            async def _run():
                nonlocal token_count, full_response
                async with agent.run_stream(
                    user_prompt=user_parts,
                    model_settings=model_settings,
                ) as result:
                    async for chunk in result.stream_text(delta=True):
                        token_count += 1
                        full_response += chunk
                        preview = TransformerStreamedProgress(next_token=chunk)
                        progress_bar.update_absolute(
                            token_count,
                            total=max_new_tokens,
                            preview_image_or_output=preview,
                        )

                progress_bar.update_absolute(
                    max_new_tokens,
                    total=max_new_tokens,
                    preview_image_or_output=TransformerStreamedProgress(next_token=""),
                )

            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(lambda: asyncio.run(_run())).result()
            else:
                asyncio.run(_run())

        return full_response

    @property
    def repo_id(self) -> str:
        return self._model_id
