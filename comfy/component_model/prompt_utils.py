"""Utilities for manipulating workflow prompt dicts."""
from __future__ import annotations

import copy
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Text encoding node class_types and their text input field names
_TEXT_ENCODE_FIELDS: dict[str, list[str]] = {
    "CLIPTextEncode": ["text"],
    "CLIPTextEncodeSD3": ["clip_l", "clip_g", "t5xxl"],
    "TextEncodeQwenImageEdit": ["prompt"],
}

# Sampler nodes that have "positive" / "negative" conditioning inputs
_SAMPLER_CLASS_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "KSamplerSelect",
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


def _is_node_ref(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _trace_to_text_encoder(prompt: dict, node_id: str, visited: Optional[set] = None) -> Optional[str]:
    """Follow conditioning references back to find a text encoding node.
    Returns the node_id of the text encoding node, or None."""
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

    # For passthrough / conditioning nodes, follow their conditioning-like inputs
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


def _find_positive_text_encoder_via_sampler(prompt: dict) -> Optional[str]:
    """Find the text encoding node connected to a sampler's positive input."""
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        if class_type not in _SAMPLER_CLASS_TYPES:
            continue
        inputs = node.get("inputs", {})
        positive_ref = inputs.get("positive")
        if not _is_node_ref(positive_ref):
            continue
        result = _trace_to_text_encoder(prompt, str(positive_ref[0]))
        if result is not None:
            return result
    return None


def _find_positive_text_encoder_via_guider(prompt: dict) -> Optional[str]:
    """For SamplerCustomAdvanced workflows that use BasicGuider with a conditioning input."""
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        if class_type != "BasicGuider":
            continue
        inputs = node.get("inputs", {})
        cond_ref = inputs.get("conditioning")
        if not _is_node_ref(cond_ref):
            continue
        result = _trace_to_text_encoder(prompt, str(cond_ref[0]))
        if result is not None:
            return result
    return None


def _find_positive_text_encoder_via_title(prompt: dict) -> Optional[str]:
    """Fall back to _meta.title heuristics."""
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        if class_type not in _TEXT_ENCODE_FIELDS:
            continue
        title = node.get("_meta", {}).get("title", "").lower()
        if "positive" in title or "(prompt)" in title:
            return node_id
    return None


def _find_sole_text_encoder(prompt: dict) -> Optional[str]:
    """If there's exactly one text encoding node, return it."""
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
