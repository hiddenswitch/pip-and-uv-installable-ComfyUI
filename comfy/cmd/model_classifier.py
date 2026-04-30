"""Classify discovered model files into ComfyUI ``folder_paths`` kinds.

Given an absolute path to a model file, decide which ``folder_paths`` kind
it satisfies (``checkpoints``, ``loras``, ``vae``, etc.) and which directory
should be registered with ``add_model_folder_path`` so ComfyUI can find it.

Three signals, in order of confidence:

  1. **Parent directory name** — strongest signal. Matches the canonical
     ComfyUI kind names plus common aliases used by A1111/Forge/InvokeAI.
  2. **Filename pattern** — fallback for generic dirs (``~/Downloads``, ``~/Models``).
  3. **None** — return ``Classification`` with ``kind=None`` so the caller
     knows we didn't classify this file.

HuggingFace cache files (anywhere under a ``models--<org>--<repo>/snapshots``
tree) are tagged ``is_hf_cache=True`` and not classified into a kind — they
are resolved per-file via ``huggingface_hub.hf_hub_download(local_files_only=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Canonical ComfyUI folder_paths kinds, plus aliases from other tools.
# Keys are matched case-insensitively against parent directory names.
_KIND_BY_DIR_NAME: dict[str, str] = {
    # checkpoints
    "checkpoints": "checkpoints",
    "stable-diffusion": "checkpoints",   # A1111 / Forge
    "stablediffusion": "checkpoints",
    "sdxl": "checkpoints",
    "sd": "checkpoints",
    # loras
    "loras": "loras",
    "lora": "loras",                     # A1111 / Forge
    "lycoris": "loras",
    # vae
    "vae": "vae",
    "vaes": "vae",
    # text encoders / clip
    "text_encoders": "text_encoders",
    "clip": "text_encoders",
    # clip vision
    "clip_vision": "clip_vision",
    "clip-vision": "clip_vision",
    # controlnet
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "t2i_adapter": "controlnet",
    "diff_controlnet": "controlnet",
    # upscalers
    "upscale_models": "upscale_models",
    "esrgan": "upscale_models",
    "realesrgan": "upscale_models",
    "swinir": "upscale_models",
    # embeddings
    "embeddings": "embeddings",
    "textual_inversion": "embeddings",
    # diffusion models / unet
    "diffusion_models": "diffusion_models",
    "unet": "diffusion_models",
    "unet_models": "diffusion_models",
    # other comfy kinds
    "style_models": "style_models",
    "hypernetworks": "hypernetworks",
    "photomaker": "photomaker",
    "audio_encoders": "audio_encoders",
    "model_patches": "model_patches",
    "frame_interpolation": "frame_interpolation",
    "latent_upscale_models": "latent_upscale_models",
    "diffusers": "diffusers",
    "vae_approx": "vae_approx",
    "configs": "configs",
    "gligen": "gligen",
    # detector / face models (Impact Pack et al.)
    "ultralytics_bbox": "ultralytics_bbox",
    "ultralytics_segm": "ultralytics_segm",
    "adetailer": "ultralytics_bbox",
    # insightface
    "insightface": "insightface",
    "antelopev2": "insightface",
    "buffalo_l": "insightface",
    "buffalo_m": "insightface",
    "buffalo_s": "insightface",
}


@dataclass
class Classification:
    path: str
    kind: Optional[str]                  # folder_paths kind, None if unclassified or HF-cache
    register_dir: Optional[str]          # directory to add_model_folder_path, None if N/A
    is_hf_cache: bool = False
    confidence: str = "low"              # "high" / "medium" / "low"


def _is_hf_cache_path(parts: tuple[str, ...]) -> bool:
    return any(seg.startswith("models--") for seg in parts)


def classify(path: str) -> Classification:
    p = Path(path)
    parts = p.parts

    if _is_hf_cache_path(parts):
        return Classification(path=path, kind=None, register_dir=None, is_hf_cache=True)

    # Walk parents from closest (immediate parent) to root, return on first hit
    for parent in p.parents:
        seg = parent.name.lower()
        if seg in _KIND_BY_DIR_NAME:
            # Special case for the segment named like a "bbox"/"segm" subdir
            # that lives under "ultralytics" — keep the whole structure.
            return Classification(
                path=path,
                kind=_KIND_BY_DIR_NAME[seg],
                register_dir=str(parent),
                confidence="high",
            )

    # Filename-pattern fallback
    name = p.name.lower()
    parent_dir = str(p.parent)
    if "_lora" in name or name.startswith("lora_"):
        return Classification(path, "loras", parent_dir, confidence="medium")
    if "vae" in name and ("." in name):
        return Classification(path, "vae", parent_dir, confidence="medium")
    if (("yolov" in name) or ("face" in name) or ("hand" in name) or ("person" in name)) and name.endswith(".pt"):
        return Classification(path, "ultralytics_bbox", parent_dir, confidence="medium")
    if name.endswith(".gguf"):
        return Classification(path, "diffusion_models", parent_dir, confidence="medium")

    return Classification(path=path, kind=None, register_dir=None, confidence="low")


def classify_many(paths: list[str]) -> list[Classification]:
    return [classify(p) for p in paths]
