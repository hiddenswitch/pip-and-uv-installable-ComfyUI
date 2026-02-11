"""
Convert UI (LiteGraph) workflow format to API format.

The UI format stores workflows with ``nodes``, ``links``, and ``widgets_values``.
The API format stores workflows as ``{node_id: {"class_type": ..., "inputs": ...}}``.

Conversion requires node INPUT_TYPES, so the node system must be booted first
(``import_all_nodes_in_workspace``).
"""
from __future__ import annotations

import logging
from typing import Optional

from .litegraph_types import LiteLink

logger = logging.getLogger(__name__)


def convert_ui_to_api(workflow: dict) -> dict:
    """Convert a UI (LiteGraph) workflow dict to API format.

    Raises:
        RuntimeError: If node system is not loaded.
    """
    from ..execution_context import current_execution_context

    ctx = current_execution_context()
    node_mappings = ctx.custom_nodes
    if len(node_mappings) == 0:
        raise RuntimeError(
            "Node system not loaded. Call import_all_nodes_in_workspace() first."
        )

    links: dict[int, LiteLink] = {}
    for raw in workflow.get("links", []):
        link = LiteLink.from_list(raw)
        links[link.link_id] = link

    api_workflow = {}

    for node in workflow.get("nodes", []):
        node_id = node.get("id")
        if node_id is None:
            continue

        if node.get("mode", 0) != 0:
            continue

        class_type = node.get("type")
        if class_type is None:
            continue

        class_def = _get_node_class(node_mappings, class_type)
        if class_def is None:
            continue

        input_types = _get_input_types(class_def)
        if input_types is None:
            continue

        required = input_types.get("required", {})
        optional = input_types.get("optional", {})
        input_order = list(required.keys()) + list(optional.keys())

        linked_inputs = {}
        linked_names = set()
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            inp_name = inp.get("name")
            if link_id is not None and inp_name is not None and link_id in links:
                link = links[link_id]
                linked_inputs[inp_name] = [str(link.src_node), link.src_slot]
                linked_names.add(inp_name)

        widgets_values = node.get("widgets_values", [])
        api_inputs = {}
        non_linked_names = [name for name in input_order if name not in linked_names]

        widget_idx = 0
        for name in non_linked_names:
            if widget_idx < len(widgets_values):
                api_inputs[name] = widgets_values[widget_idx]
                widget_idx += 1

        api_inputs.update(linked_inputs)

        api_workflow[str(node_id)] = {
            "class_type": class_type,
            "inputs": api_inputs,
        }

    return api_workflow


def _get_node_class(node_mappings, class_type: str) -> Optional[type]:
    if hasattr(node_mappings, 'NODE_CLASS_MAPPINGS'):
        return node_mappings.NODE_CLASS_MAPPINGS.get(class_type)
    if isinstance(node_mappings, dict):
        return node_mappings.get(class_type)
    mappings = getattr(node_mappings, 'class_mappings', None)
    if mappings:
        return mappings.get(class_type)
    return None


def _get_input_types(class_def) -> Optional[dict]:
    try:
        if hasattr(class_def, "INPUT_TYPES"):
            result = class_def.INPUT_TYPES()
            if isinstance(result, dict):
                return result
    except Exception as exc:
        logger.debug("Failed to get INPUT_TYPES for %s: %s", class_def, exc)
    return None
