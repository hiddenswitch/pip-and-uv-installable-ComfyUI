"""Utilities for manipulating workflow prompt dicts."""
from __future__ import annotations

import copy
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_TEXT_ENCODE_FIELDS: dict[str, list[str]] = {
    "CLIPTextEncode": ["text"],
    "CLIPTextEncodeSD3": ["clip_l", "clip_g", "t5xxl"],
    "TextEncodeQwenImageEdit": ["prompt"],
    "OneShotInstructTokenize": ["prompt"],
    "TransformersTranslationTokenize": ["prompt"],
    "TransformersTokenize": ["prompt"],
}


_STEPS_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "BasicScheduler",
    "Flux2Scheduler",
    "LTXVScheduler",
    "AlignYourStepsScheduler",
})

# class_type -> seed field name
_SEED_FIELDS: dict[str, str] = {
    "KSampler": "seed",
    "KSamplerAdvanced": "seed",
    "RandomNoise": "noise_seed",
    "SamplerCustom": "noise_seed",
    "TransformersGenerate": "seed",
}

_IMAGE_LOAD_CLASS_TYPES = frozenset({
    "LoadImage",
    "LoadImageFromURL",
    "ImageRequestParameter",
})

_VIDEO_LOAD_CLASS_TYPES = frozenset({
    "LoadVideo",
    "LoadVideoFromURL",
    "VideoRequestParameter",
})

_AUDIO_LOAD_CLASS_TYPES = frozenset({
    "LoadAudio",
    "LoadAudioFromURL",
    "AudioRequestParameter",
})


def _is_node_ref(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _find_text_encoder_in_predecessors(prompt: dict, start_node_id: str) -> Optional[str]:
    visited: set[str] = set()
    stack = [start_node_id]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        node = prompt.get(nid)
        if node is None:
            continue
        if node.get("class_type", "") in _TEXT_ENCODE_FIELDS:
            return nid
        for val in node.get("inputs", {}).values():
            if _is_node_ref(val):
                ref_id = str(val[0])
                if ref_id not in visited:
                    stack.append(ref_id)
    return None



def _find_positive_text_encoder_via_positive_input(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        positive_ref = node.get("inputs", {}).get("positive")
        if not _is_node_ref(positive_ref):
            continue
        result = _find_text_encoder_in_predecessors(prompt, str(positive_ref[0]))
        if result is not None:
            return result
    return None


def _find_positive_text_encoder_via_guider(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        if node.get("class_type", "") != "BasicGuider":
            continue
        cond_ref = node.get("inputs", {}).get("conditioning")
        if not _is_node_ref(cond_ref):
            continue
        result = _find_text_encoder_in_predecessors(prompt, str(cond_ref[0]))
        if result is not None:
            return result
    return None


def _find_positive_text_encoder_via_title(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        if node.get("class_type", "") not in _TEXT_ENCODE_FIELDS:
            continue
        title = node.get("_meta", {}).get("title", "").lower()
        if "positive" in title or "(prompt)" in title:
            return node_id
    return None


def _find_sole_text_encoder(prompt: dict) -> Optional[str]:
    text_nodes = [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in _TEXT_ENCODE_FIELDS
    ]
    if len(text_nodes) == 1:
        return text_nodes[0]
    return None


def find_positive_text_encoder(prompt: dict) -> Optional[str]:
    return (
        _find_positive_text_encoder_via_positive_input(prompt)
        or _find_positive_text_encoder_via_guider(prompt)
        or _find_positive_text_encoder_via_title(prompt)
        or _find_sole_text_encoder(prompt)
    )


def replace_prompt_text(prompt: dict, text: str) -> dict:
    """Return a copy of *prompt* with the positive text encoding node's text replaced.

    Raises ``ValueError`` if no suitable text encoding node is found.
    """
    node_id = find_positive_text_encoder(prompt)
    if node_id is None:
        raise ValueError("Could not find a positive text encoding node to replace")

    prompt = copy.deepcopy(prompt)
    node = prompt[node_id]
    class_type = node["class_type"]
    fields = _TEXT_ENCODE_FIELDS[class_type]
    for field in fields:
        if field in node["inputs"]:
            node["inputs"][field] = text
    return prompt



def _find_negative_text_encoder_via_negative_input(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        negative_ref = node.get("inputs", {}).get("negative")
        if not _is_node_ref(negative_ref):
            continue
        result = _find_text_encoder_in_predecessors(prompt, str(negative_ref[0]))
        if result is not None:
            return result
    return None


def _find_negative_text_encoder_via_title(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        if node.get("class_type", "") not in _TEXT_ENCODE_FIELDS:
            continue
        title = node.get("_meta", {}).get("title", "").lower()
        if "negative" in title:
            return node_id
    return None


def find_negative_text_encoder(prompt: dict) -> Optional[str]:
    return (
        _find_negative_text_encoder_via_negative_input(prompt)
        or _find_negative_text_encoder_via_title(prompt)
    )


def replace_negative_prompt_text(prompt: dict, text: str) -> dict:
    """Return a copy of *prompt* with the negative text encoding node's text replaced.

    Raises ``ValueError`` if no suitable negative text encoding node is found.
    """
    node_id = find_negative_text_encoder(prompt)
    if node_id is None:
        raise ValueError("Could not find a negative text encoding node to replace")

    prompt = copy.deepcopy(prompt)
    node = prompt[node_id]
    class_type = node["class_type"]
    fields = _TEXT_ENCODE_FIELDS[class_type]
    for field in fields:
        if field in node["inputs"]:
            node["inputs"][field] = text
    return prompt



def find_steps_nodes(prompt: dict) -> list[str]:
    """Return node IDs of all nodes that have a ``steps`` input."""
    return [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in _STEPS_CLASS_TYPES
        and "steps" in node.get("inputs", {})
    ]


def replace_steps(prompt: dict, steps: int) -> dict:
    """Return a copy of *prompt* with all sampler/scheduler step counts replaced."""
    node_ids = find_steps_nodes(prompt)
    if not node_ids:
        return prompt
    prompt = copy.deepcopy(prompt)
    for nid in node_ids:
        prompt[nid]["inputs"]["steps"] = steps
    return prompt



def find_seed_nodes(prompt: dict) -> list[tuple[str, str]]:
    """Return ``(node_id, field_name)`` pairs for all nodes with a seed input."""
    results = []
    for nid, node in prompt.items():
        class_type = node.get("class_type", "")
        field = _SEED_FIELDS.get(class_type)
        if field is not None and field in node.get("inputs", {}):
            results.append((nid, field))
    return results


def replace_seed(prompt: dict, seed: int) -> dict:
    """Return a copy of *prompt* with all seed values replaced."""
    pairs = find_seed_nodes(prompt)
    if not pairs:
        return prompt
    prompt = copy.deepcopy(prompt)
    for nid, field in pairs:
        prompt[nid]["inputs"][field] = seed
    return prompt


# filesystem-loader -> URL-loader class_type
_MEDIA_LOADER_TO_URL: dict[str, str] = {
    "LoadImage": "LoadImageFromURL",
    "LoadVideo": "LoadVideoFromURL",
    "LoadAudio": "LoadAudioFromURL",
}


def _find_media_nodes(prompt: dict, class_types: frozenset) -> list[str]:
    """Return node IDs of media-loading nodes matching *class_types*."""
    return [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in class_types
    ]


def _replace_media(
    prompt: dict,
    values: list[str],
    class_types: frozenset,
) -> dict:
    """Generic replacement for image / video / audio loading nodes.

    Filesystem loaders (``LoadImage``, ``LoadVideo``, ``LoadAudio``) are
    converted to their ``*FromURL`` counterparts.  Nodes that already accept
    a ``value`` input simply have it updated.
    """
    node_ids = _find_media_nodes(prompt, class_types)
    if not node_ids or not values:
        return prompt
    prompt = copy.deepcopy(prompt)
    for i, nid in enumerate(node_ids):
        if i >= len(values):
            break
        node = prompt[nid]
        class_type = node["class_type"]
        url_class = _MEDIA_LOADER_TO_URL.get(class_type)
        if url_class is not None:
            node["class_type"] = url_class
            node["inputs"] = {"value": values[i]}
            node.pop("_meta", None)
        else:
            node["inputs"]["value"] = values[i]
    return prompt


# --image

def find_image_load_nodes(prompt: dict) -> list[str]:
    """Return node IDs of image loading nodes."""
    return _find_media_nodes(prompt, _IMAGE_LOAD_CLASS_TYPES)


def replace_images(prompt: dict, images: list[str]) -> dict:
    """Return a copy of *prompt* with image loading nodes replaced."""
    return _replace_media(prompt, images, _IMAGE_LOAD_CLASS_TYPES)


# --video

def find_video_load_nodes(prompt: dict) -> list[str]:
    """Return node IDs of video loading nodes."""
    return _find_media_nodes(prompt, _VIDEO_LOAD_CLASS_TYPES)


def replace_videos(prompt: dict, videos: list[str]) -> dict:
    """Return a copy of *prompt* with video loading nodes replaced."""
    return _replace_media(prompt, videos, _VIDEO_LOAD_CLASS_TYPES)


# --audio

def find_audio_load_nodes(prompt: dict) -> list[str]:
    """Return node IDs of audio loading nodes."""
    return _find_media_nodes(prompt, _AUDIO_LOAD_CLASS_TYPES)


def replace_audios(prompt: dict, audios: list[str]) -> dict:
    """Return a copy of *prompt* with audio loading nodes replaced."""
    return _replace_media(prompt, audios, _AUDIO_LOAD_CLASS_TYPES)
