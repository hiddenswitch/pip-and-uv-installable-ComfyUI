"""Utilities for manipulating workflow prompt dicts."""
from __future__ import annotations

import copy
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# --prompt: text encoding / tokenize node class_types → text input field names
# ---------------------------------------------------------------------------
_TEXT_ENCODE_FIELDS: dict[str, list[str]] = {
    "CLIPTextEncode": ["text"],
    "CLIPTextEncodeSD3": ["clip_l", "clip_g", "t5xxl"],
    "TextEncodeQwenImageEdit": ["prompt"],
    "OneShotInstructTokenize": ["prompt"],
    "TransformersTranslationTokenize": ["prompt"],
    "TransformersTokenize": ["prompt"],
}

# Sampler nodes that have "positive" / "negative" conditioning inputs
_SAMPLER_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
})

# Nodes that forward conditioning (input → output) so we can trace chains
_CONDITIONING_PASSTHROUGH = frozenset({
    "FluxGuidance",
    "BasicGuider",
    "ConditioningSetTimestepRange",
    "ConditioningZeroOut",
    "ConditioningCombine",
    "ConditioningConcat",
    "ConditioningAverage",
    "StyleModelApply",
    "LTXVConditioning",
})

# ---------------------------------------------------------------------------
# --steps: nodes that accept a ``steps`` input
# ---------------------------------------------------------------------------
_STEPS_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "BasicScheduler",
    "Flux2Scheduler",
    "LTXVScheduler",
    "AlignYourStepsScheduler",
})

# ---------------------------------------------------------------------------
# --image: nodes that load images
# ---------------------------------------------------------------------------
_IMAGE_LOAD_CLASS_TYPES = frozenset({
    "LoadImage",
    "LoadImageFromURL",
    "ImageRequestParameter",
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_node_ref(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _trace_to_text_encoder(prompt: dict, node_id: str, visited: Optional[set] = None) -> Optional[str]:
    """Follow conditioning references back to find a text encoding node."""
    if visited is None:
        visited = set()
    if node_id in visited:
        return None
    visited.add(node_id)

    node = prompt.get(node_id)
    if node is None:
        return None

    class_type = node.get("class_type", "")
    if class_type in _TEXT_ENCODE_FIELDS:
        return node_id

    inputs = node.get("inputs", {})
    for key, val in inputs.items():
        if _is_node_ref(val):
            ref_id = str(val[0])
            ref_node = prompt.get(ref_id)
            if ref_node is None:
                continue
            ref_class = ref_node.get("class_type", "")
            if ref_class in _TEXT_ENCODE_FIELDS:
                return ref_id
            if ref_class in _CONDITIONING_PASSTHROUGH:
                result = _trace_to_text_encoder(prompt, ref_id, visited)
                if result is not None:
                    return result

    return None


# ---------------------------------------------------------------------------
# --prompt helpers
# ---------------------------------------------------------------------------

def _find_positive_text_encoder_via_sampler(prompt: dict) -> Optional[str]:
    for node_id, node in prompt.items():
        if node.get("class_type", "") not in _SAMPLER_CLASS_TYPES:
            continue
        positive_ref = node.get("inputs", {}).get("positive")
        if not _is_node_ref(positive_ref):
            continue
        result = _trace_to_text_encoder(prompt, str(positive_ref[0]))
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
        result = _trace_to_text_encoder(prompt, str(cond_ref[0]))
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
    """Find the node ID of the positive text encoding node in a workflow prompt.

    Tries multiple strategies in order:
    1. Trace from sampler's "positive" input
    2. Trace from BasicGuider's "conditioning" input
    3. Match by _meta.title heuristic
    4. Use the sole text encoding node if there's only one
    """
    return (
        _find_positive_text_encoder_via_sampler(prompt)
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


# ---------------------------------------------------------------------------
# --steps
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# --image
# ---------------------------------------------------------------------------

def find_image_load_nodes(prompt: dict) -> list[str]:
    """Return node IDs of image loading nodes, in the order they appear."""
    return [
        nid for nid, node in prompt.items()
        if node.get("class_type", "") in _IMAGE_LOAD_CLASS_TYPES
    ]


def replace_images(prompt: dict, images: list[str]) -> dict:
    """Return a copy of *prompt* with image loading nodes replaced.

    - ``LoadImage`` nodes are converted to ``LoadImageFromURL`` with the URI.
    - ``LoadImageFromURL`` and ``ImageRequestParameter`` nodes get their ``value`` updated.
    - Images are assigned to nodes in order; extra images or nodes are ignored.
    """
    node_ids = find_image_load_nodes(prompt)
    if not node_ids or not images:
        return prompt
    prompt = copy.deepcopy(prompt)
    for i, nid in enumerate(node_ids):
        if i >= len(images):
            break
        node = prompt[nid]
        class_type = node["class_type"]
        if class_type == "LoadImage":
            # Convert to LoadImageFromURL
            node["class_type"] = "LoadImageFromURL"
            node["inputs"] = {"value": images[i]}
            node.pop("_meta", None)
        else:
            # LoadImageFromURL or ImageRequestParameter — just set value
            node["inputs"]["value"] = images[i]
    return prompt
