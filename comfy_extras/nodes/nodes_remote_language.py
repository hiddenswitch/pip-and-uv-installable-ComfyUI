from __future__ import annotations

import os
from typing import Optional

from comfy.cli_args import args
from comfy.language.language_types import LanguageModel
from comfy.language.remote_model import RemoteLanguageModel
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes

PROVIDERS = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "cli_attr": "openai_api_key",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3-mini"],
    },
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "cli_attr": "anthropic_api_key",
        "models": ["claude-sonnet-4-5-20250514", "claude-haiku-4-5-20250514", "claude-3-5-haiku-latest"],
    },
    "google-gla": {
        "env_var": "GOOGLE_API_KEY",
        "cli_attr": "google_api_key",
        "models": ["gemini-2.0-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
    "mistral": {
        "env_var": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-small-latest"],
    },
    "xai": {
        "env_var": "XAI_API_KEY",
        "models": ["grok-2", "grok-3"],
    },
    "cohere": {
        "env_var": "COHERE_API_KEY",
        "models": ["command-r-plus", "command-r"],
    },
    "cerebras": {
        "env_var": "CEREBRAS_API_KEY",
        "models": ["llama-3.3-70b"],
    },
}


def _has_api_key(provider: str) -> bool:
    info = PROVIDERS.get(provider, {})
    env_var = info.get("env_var", "")
    if os.environ.get(env_var):
        return True
    cli_attr = info.get("cli_attr")
    if cli_attr and getattr(args, cli_attr, None):
        return True
    return False


def get_available_models() -> list[str]:
    models = []
    for provider, info in PROVIDERS.items():
        for model in info["models"]:
            models.append(f"{provider}:{model}")
    return models


class RemoteLanguageLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": (get_available_models(), {"default": "openai:gpt-4o"}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("language model",)
    FUNCTION = "execute"
    CATEGORY = "language"

    def execute(self, model: str, custom_model: str = "") -> tuple[LanguageModel]:
        model_id = custom_model.strip() if custom_model and custom_model.strip() else model
        return RemoteLanguageModel(model_id),

    @classmethod
    def VALIDATE_INPUTS(cls, model: str = "", custom_model: str = "") -> str | bool:
        model_id = custom_model.strip() if custom_model and custom_model.strip() else model

        if ":" in model_id:
            provider = model_id.split(":")[0]
        else:
            return True

        if not _has_api_key(provider):
            info = PROVIDERS.get(provider, {})
            env_var = info.get("env_var", f"{provider.upper()}_API_KEY")
            return f"No API key found for provider '{provider}'. Set the {env_var} environment variable or use the corresponding CLI flag."

        return True


export_custom_nodes()
