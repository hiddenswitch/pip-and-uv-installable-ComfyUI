"""Shared node_info() logic used by both server.py and nodes sub-command."""
from __future__ import annotations

from comfy_api.internal import _ComfyNodeInternal


def node_info(node_class: str, node_class_mappings: dict, node_display_name_mappings: dict) -> dict:
    obj_class = node_class_mappings[node_class]
    if issubclass(obj_class, _ComfyNodeInternal):
        return obj_class.GET_NODE_INFO_V1()
    info = {}
    info['input'] = obj_class.INPUT_TYPES()
    info['input_order'] = {key: list(value.keys()) for (key, value) in obj_class.INPUT_TYPES().items()}
    info['is_input_list'] = getattr(obj_class, "INPUT_IS_LIST", False)
    _return_types = ["*" if isinstance(rt, list) and rt == [] else rt for rt in obj_class.RETURN_TYPES]
    info['output'] = _return_types
    info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(_return_types)
    info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
    info['name'] = node_class
    info['display_name'] = node_display_name_mappings.get(node_class, node_class)
    info['description'] = obj_class.DESCRIPTION if hasattr(obj_class, 'DESCRIPTION') else ''
    info['python_module'] = getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes")
    info['category'] = 'sd'
    if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE is True:
        info['output_node'] = True
    else:
        info['output_node'] = False

    if hasattr(obj_class, 'CATEGORY'):
        info['category'] = obj_class.CATEGORY

    if hasattr(obj_class, 'OUTPUT_TOOLTIPS'):
        info['output_tooltips'] = obj_class.OUTPUT_TOOLTIPS

    if getattr(obj_class, "DEPRECATED", False):
        info['deprecated'] = True
    if getattr(obj_class, "EXPERIMENTAL", False):
        info['experimental'] = True
    if getattr(obj_class, "DEV_ONLY", False):
        info['dev_only'] = True

    if hasattr(obj_class, 'API_NODE'):
        info['api_node'] = obj_class.API_NODE

    info['search_aliases'] = getattr(obj_class, 'SEARCH_ALIASES', [])
    return info
