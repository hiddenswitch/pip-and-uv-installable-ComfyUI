"""
Convert UI (LiteGraph) workflow format to API format.

The UI format stores workflows with ``nodes``, ``links``, and ``widgets_values``.
The API format stores workflows as ``{node_id: {"class_type": ..., "inputs": ...}}``.

Conversion requires node INPUT_TYPES, so the node system must be booted first
(``import_all_nodes_in_workspace``).

The logic mirrors the frontend ``graphToPrompt`` implementation from
``ComfyUI_frontend/src/utils/executionUtil.ts`` and its helper classes.
"""
from __future__ import annotations

import logging
import re
from types import MappingProxyType
from typing import Final, Optional

from .litegraph_types import LiteLink

logger = logging.getLogger(__name__)

_WIDGET_TYPES: Final[frozenset[str]] = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})

_VIRTUAL_NODE_TYPES: Final[frozenset[str]] = frozenset({
    "Reroute",
    "PrimitiveNode",
    "Note",
    "MarkdownNote",
})

_MODE_ALWAYS: Final[int] = 0
_MODE_ON_EVENT: Final[int] = 1
_MODE_NEVER: Final[int] = 2
_MODE_ON_TRIGGER: Final[int] = 3
_MODE_BYPASS: Final[int] = 4

_SUBGRAPH_INPUT_NODE_ID: Final[int] = -10
_SUBGRAPH_OUTPUT_NODE_ID: Final[int] = -20

_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I,
)

_FRONTEND_INJECTED_WIDGETS: Final[MappingProxyType[str, tuple[tuple[str, object], ...]]] = MappingProxyType({
    "PreviewAny": (("preview", ""), ("previewMode", False)),
    "LoadAudio": (("audioUI", ""),),
    "SaveAudio": (("audioUI", ""),),
    "PreviewAudio": (("audioUI", ""),),
    "SaveAudioMP3": (("audioUI", ""),),
    "SaveAudioOpus": (("audioUI", ""),),
    "Preview3D": (("image", ""),),
    "SaveGLB": (("image", ""),),
    "RecordAudio": (("audio", ""),),
})


def _is_widget_type(type_spec, opts=None) -> bool:
    if isinstance(type_spec, list):
        return True
    if not isinstance(type_spec, str):
        return False
    if type_spec in _WIDGET_TYPES:
        return True
    if type_spec == "COMFY_DYNAMICCOMBO_V3":
        return True
    if opts:
        if opts.get("socketless"):
            return True
        if opts.get("widgetType"):
            return True
    return False


def _input_type_and_opts(entry) -> tuple:
    if isinstance(entry, (list, tuple)):
        type_spec = entry[0]
        opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        return type_spec, opts
    return entry, {}


def _extra_widgets_after(opts: dict) -> list[str | None]:
    extras: list[str | None] = []
    if opts.get("control_after_generate"):
        extras.append(None)
    if opts.get("image_upload") or opts.get("video_upload") or opts.get("audio_upload"):
        extras.append(None)
    return extras


def _wrap_value(val):
    return {"__value__": val} if isinstance(val, list) else val


def _map_widgets(input_types: dict, widgets_values: list) -> tuple[dict[str, object], int]:
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})

    result: dict[str, object] = {}
    idx = 0
    in_optional = False

    for name, entry in list(required.items()) + list(optional.items()):
        if not in_optional and name in optional:
            in_optional = True

        type_spec, opts = _input_type_and_opts(entry)

        if not _is_widget_type(type_spec, opts):
            continue
        if opts.get("forceInput"):
            continue

        if idx < len(widgets_values):
            result[name] = _wrap_value(widgets_values[idx])
            idx += 1
        elif in_optional:
            if "default" in opts:
                result[name] = _wrap_value(opts["default"])
            elif isinstance(type_spec, list) and type_spec:
                result[name] = _wrap_value(type_spec[0])
            elif type_spec == "COMBO" and opts.get("options"):
                result[name] = _wrap_value(opts["options"][0])
            else:
                break
        else:
            break

        for extra_name in _extra_widgets_after(opts):
            if idx < len(widgets_values):
                if extra_name is not None:
                    result[extra_name] = _wrap_value(widgets_values[idx])
                idx += 1

        if type_spec == "COMFY_DYNAMICCOMBO_V3":
            idx = _consume_dynamic_combo_subwidgets(
                name, result.get(name), opts, widgets_values, idx, result,
            )

    return result, idx


def _consume_dynamic_combo_subwidgets(
    parent_name: str,
    selected_key,
    opts: dict,
    widgets_values: list,
    idx: int,
    result: dict,
) -> int:
    options = opts.get("options", [])
    if not options or selected_key is None:
        return idx

    matched_option = None
    for option in options:
        if not isinstance(option, dict):
            continue
        key = option.get("key")
        if key == selected_key or getattr(key, "value", None) == selected_key:
            matched_option = option
            break
    if matched_option is None:
        return idx

    sub_inputs = matched_option.get("inputs", {})
    for section in ("required", "optional"):
        for sub_name, sub_entry in sub_inputs.get(section, {}).items():
            sub_type, sub_opts = _input_type_and_opts(sub_entry)
            if not _is_widget_type(sub_type, sub_opts):
                continue
            if sub_opts.get("forceInput"):
                continue
            dotted = f"{parent_name}.{sub_name}"
            if idx < len(widgets_values):
                result[dotted] = _wrap_value(widgets_values[idx])
                idx += 1
            elif "default" in sub_opts:
                result[dotted] = _wrap_value(sub_opts["default"])

            for extra_name in _extra_widgets_after(sub_opts):
                if idx < len(widgets_values):
                    if extra_name is not None:
                        result[f"{dotted}.{extra_name}"] = _wrap_value(widgets_values[idx])
                    idx += 1

    return idx


def _map_widgets_dict(input_types: dict, widgets_values: dict) -> dict[str, object]:
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})
    all_inputs = {**required, **optional}

    result: dict[str, object] = {}
    for name, entry in all_inputs.items():
        type_spec, opts = _input_type_and_opts(entry)
        if _is_widget_type(type_spec, opts) and not opts.get("forceInput"):
            result[name] = None
    return result


def __unknown_widget_value(val):
    if isinstance(val, (str, int, float, bool)):
        return val
    import json as _json
    return _json.dumps(val)


def _map_unknown_widgets(widgets_values) -> dict[str, object]:
    if isinstance(widgets_values, dict):
        return {k: _wrap_value(v) for k, v in widgets_values.items()}
    if isinstance(widgets_values, list) and widgets_values:
        return {"UNKNOWN": _wrap_value(__unknown_widget_value(widgets_values[-1]))}
    return {}


def _types_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    if a == "*" or b == "*":
        return True
    return False


def _get_bypass_slot_index(
    inputs: list[dict], outputs: list[dict], slot: int, target_type: str | None,
) -> int:
    if not target_type or target_type == "*":
        return slot if slot < len(inputs) else 0

    out_type = outputs[slot].get("type") if slot < len(outputs) else None

    if slot < len(inputs):
        opp_type = inputs[slot].get("type")
        if _types_match(opp_type, out_type) and _types_match(opp_type, target_type):
            return slot

    for i, inp in enumerate(inputs):
        if inp.get("type") == target_type:
            return i

    for i, inp in enumerate(inputs):
        inp_type = inp.get("type")
        if _types_match(inp_type, out_type) and _types_match(inp_type, target_type):
            return i

    return -1


def _resolve_source(
    src_node_id: int,
    src_slot: int,
    nodes_by_id: dict[int, dict],
    links: dict[int, LiteLink],
    visited: set | None = None,
    target_type: str | None = None,
) -> tuple | None:
    if visited is None:
        visited = set()
    key = (src_node_id, src_slot)
    if key in visited:
        return None
    visited.add(key)

    node = nodes_by_id.get(src_node_id)
    if node is None:
        return None

    node_type = node.get("type", "")
    mode = node.get("mode", 0)

    if node_type == "Reroute":
        inputs = node.get("inputs", [])
        if inputs and inputs[0].get("link") is not None:
            link = links.get(inputs[0]["link"])
            if link:
                return _resolve_source(
                    link.src_node, link.src_slot, nodes_by_id, links,
                    visited, target_type,
                )
        return None

    if node_type == "PrimitiveNode":
        wv = node.get("widgets_values", [])
        if wv:
            return ("value", wv[0])
        return None

    if mode == _MODE_NEVER:
        return None

    if mode == _MODE_BYPASS:
        outputs = node.get("outputs", [])
        node_inputs = node.get("inputs", [])
        if src_slot >= len(outputs) or not node_inputs:
            return None
        out_type = outputs[src_slot].get("type")
        bypass_type = target_type or out_type
        match_idx = _get_bypass_slot_index(
            node_inputs, outputs, src_slot, bypass_type,
        )
        if match_idx == -1:
            return None
        inp = node_inputs[match_idx]
        link_id = inp.get("link")
        if link_id is not None:
            link = links.get(link_id)
            if link:
                return _resolve_source(
                    link.src_node, link.src_slot,
                    nodes_by_id, links, visited, bypass_type,
                )
        return None

    return ("link", str(src_node_id), src_slot)


def _collect_subgraph_defs(workflow: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}

    def _search(obj, depth=0):
        if depth > 4 or not isinstance(obj, dict):
            return
        sgs = obj.get("subgraphs")
        if isinstance(sgs, list):
            for sg in sgs:
                if isinstance(sg, dict) and "id" in sg:
                    result[sg["id"]] = sg
        for v in obj.values():
            if isinstance(v, dict):
                _search(v, depth + 1)

    _search(workflow)
    return result


def _parse_link(raw) -> LiteLink:
    if isinstance(raw, dict):
        return LiteLink.from_dict(raw)
    return LiteLink.from_list(raw)


def _build_inner_links(sg_def: dict) -> dict[int, LiteLink]:
    inner_links: dict[int, LiteLink] = {}
    for raw in sg_def.get("links", []):
        link = _parse_link(raw)
        inner_links[link.link_id] = link
    return inner_links


class _NodeDTO:
    __slots__ = (
        'exec_id', 'node', 'graph_links', 'graph_nodes_by_id',
        'subgraph_node_path', 'sg_node_exec_id',
        'sg_def', 'inner_links', 'proxy_overrides',
    )

    def __init__(self, node, subgraph_node_path, graph_links, graph_nodes_by_id,
                 sg_node_exec_id=None, sg_def=None):
        self.node = node
        self.subgraph_node_path = list(subgraph_node_path)
        nid = node['id']
        self.exec_id = ':'.join(str(x) for x in [*subgraph_node_path, nid])
        self.graph_links = graph_links
        self.graph_nodes_by_id = graph_nodes_by_id
        self.sg_node_exec_id = sg_node_exec_id
        self.sg_def = sg_def
        self.inner_links = _build_inner_links(sg_def) if sg_def else {}
        self.proxy_overrides: dict[tuple[int, str], object] = {}


def _compute_proxy_overrides(sg_node, parent_overrides=None):
    proxy_widgets = sg_node.get('properties', {}).get('proxyWidgets', [])
    wv = sg_node.get('widgets_values', [])
    if isinstance(wv, dict):
        wv = []

    overrides: dict[tuple[int, str], object] = {}
    for pw_idx, pw in enumerate(proxy_widgets):
        if not isinstance(pw, (list, tuple)) or len(pw) < 2:
            continue
        pw_target, pw_name = pw[0], pw[1]
        if str(pw_target) == '-1':
            continue
        if pw_idx >= len(wv):
            continue
        try:
            inner_nid = int(pw_target)
        except (ValueError, TypeError):
            continue
        val = wv[pw_idx]
        if parent_overrides:
            nid = sg_node['id']
            parent_val = parent_overrides.get((nid, pw_name))
            if parent_val is not None:
                val = parent_val
        overrides[(inner_nid, pw_name)] = val

    return overrides


def _expand_subgraph(sg_node, sg_def, subgraph_node_path,
                     sg_defs, dto_map, sg_exec_id):
    sg_nid = sg_node['id']
    instance_path = [*subgraph_node_path, sg_nid]

    sg_dto = dto_map[sg_exec_id]
    inner_links = sg_dto.inner_links
    inner_nodes_by_id: dict[int, dict] = {}
    for n in sg_def.get('nodes', []):
        nid = n.get('id')
        if nid is not None:
            inner_nodes_by_id[nid] = n

    for inner_node in sg_def.get('nodes', []):
        inner_nid = inner_node.get('id')
        if inner_nid is None or inner_nid < 0:
            continue

        class_type = inner_node.get('type', '')
        mode = inner_node.get('mode', 0)

        if _is_subgraph_type(class_type, sg_defs):
            child_sg_def = sg_defs[class_type]
            child_dto = _NodeDTO(
                inner_node, instance_path, inner_links, inner_nodes_by_id,
                sg_node_exec_id=sg_exec_id, sg_def=child_sg_def,
            )
            child_dto.proxy_overrides = _compute_proxy_overrides(
                inner_node,
                parent_overrides=sg_dto.proxy_overrides,
            )
            dto_map[child_dto.exec_id] = child_dto

            if mode not in (_MODE_NEVER, _MODE_BYPASS):
                _expand_subgraph(
                    inner_node, child_sg_def, instance_path,
                    sg_defs, dto_map, child_dto.exec_id,
                )
        else:
            dto = _NodeDTO(
                inner_node, instance_path, inner_links, inner_nodes_by_id,
                sg_node_exec_id=sg_exec_id,
            )
            dto_map[dto.exec_id] = dto


def _build_dto_map(workflow, sg_defs):
    links: dict[int, LiteLink] = {}
    for raw in workflow.get('links', []):
        link = LiteLink.from_list(raw)
        links[link.link_id] = link

    nodes_by_id: dict[int, dict] = {}
    for node in workflow.get('nodes', []):
        nid = node.get('id')
        if nid is not None:
            nodes_by_id[nid] = node

    dto_map: dict[str, _NodeDTO] = {}

    for node in workflow.get('nodes', []):
        nid = node.get('id')
        if nid is None:
            continue

        mode = node.get('mode', 0)
        class_type = node.get('type', '')

        if _is_subgraph_type(class_type, sg_defs):
            sg_def = sg_defs[class_type]
            sg_dto = _NodeDTO(node, [], links, nodes_by_id, sg_def=sg_def)
            sg_dto.proxy_overrides = _compute_proxy_overrides(node)
            dto_map[sg_dto.exec_id] = sg_dto

            if mode not in (_MODE_NEVER, _MODE_BYPASS):
                _expand_subgraph(
                    node, sg_def, [], sg_defs, dto_map, sg_dto.exec_id,
                )
        else:
            dto = _NodeDTO(node, [], links, nodes_by_id)
            dto_map[dto.exec_id] = dto

    return dto_map


def _get_sg_widget_by_name(sg_node, inp_name, sg_def=None):
    if inp_name is None:
        return False, None

    proxy_widgets = sg_node.get('properties', {}).get('proxyWidgets', [])
    wv = sg_node.get('widgets_values', [])
    if isinstance(wv, dict):
        wv = []

    for pw_idx, pw in enumerate(proxy_widgets):
        if not isinstance(pw, (list, tuple)) or len(pw) < 2:
            continue
        if str(pw[0]) == '-1' and pw[1] == inp_name:
            if pw_idx < len(wv):
                return True, wv[pw_idx]
            return False, None

    if sg_def and wv:
        return _get_sg_widget_positional(sg_def, inp_name, wv)

    return False, None


def _get_sg_widget_positional(sg_def, boundary_name, wv):
    inner_links = _build_inner_links(sg_def)
    inner_nodes_by_id: dict[int, dict] = {}
    for n in sg_def.get('nodes', []):
        nid = n.get('id')
        if nid is not None:
            inner_nodes_by_id[nid] = n

    sg_def_inputs = sg_def.get('inputs', [])

    slot_is_widget: dict[int, bool] = {}
    for link in inner_links.values():
        if link.src_node != _SUBGRAPH_INPUT_NODE_ID:
            continue
        slot = link.src_slot
        if slot in slot_is_widget:
            continue
        target_node = inner_nodes_by_id.get(link.dst_node)
        if target_node is None:
            slot_is_widget[slot] = False
            continue
        for tinp in target_node.get('inputs', []):
            if tinp.get('link') == link.link_id:
                slot_is_widget[slot] = 'widget' in tinp
                break
        else:
            slot_is_widget[slot] = False

    wv_idx = 0
    for slot_idx, sg_inp in enumerate(sg_def_inputs):
        if not slot_is_widget.get(slot_idx, False):
            continue
        if sg_inp.get('name') == boundary_name:
            if wv_idx < len(wv):
                return True, wv[wv_idx]
            return False, None
        wv_idx += 1

    return False, None


def _resolve_dto_input(dto, slot, dto_map, visited=None, skip_boundary_widgets=False):
    if visited is None:
        visited = set()
    uid = f"{dto.exec_id}[I]{slot}"
    if uid in visited:
        return None
    visited.add(uid)

    inputs = dto.node.get('inputs', [])
    if slot >= len(inputs):
        return None

    inp = inputs[slot]
    link_id = inp.get('link')
    if link_id is None:
        return None

    link = dto.graph_links.get(link_id)
    if link is None:
        return None

    if dto.sg_node_exec_id is not None and link.src_node == _SUBGRAPH_INPUT_NODE_ID:
        sg_dto = dto_map.get(dto.sg_node_exec_id)
        if sg_dto is None:
            return None

        sg_def = sg_dto.sg_def
        sg_def_inputs = sg_def.get('inputs', []) if sg_def else []
        if link.src_slot >= len(sg_def_inputs):
            return None
        boundary_name = sg_def_inputs[link.src_slot].get('name')

        sg_inputs = sg_dto.node.get('inputs', [])
        outer_link_id = None
        sg_inp_idx = None
        for idx, sg_inp in enumerate(sg_inputs):
            if sg_inp.get('name') == boundary_name:
                outer_link_id = sg_inp.get('link')
                sg_inp_idx = idx
                break

        if outer_link_id is None:
            if skip_boundary_widgets:
                return None
            found, val = _get_sg_widget_by_name(sg_dto.node, boundary_name, sg_def)
            return ("value", val) if found else None

        outer_link = sg_dto.graph_links.get(outer_link_id)
        if outer_link is None:
            return None

        return _resolve_dto_input(sg_dto, sg_inp_idx, dto_map, visited)

    src_exec_id = ':'.join(str(x) for x in [*dto.subgraph_node_path, link.src_node])
    src_dto = dto_map.get(src_exec_id)
    if src_dto is None:
        return None

    return _resolve_dto_output(
        src_dto, link.src_slot, inp.get('type'), dto_map, visited,
    )


def _resolve_dto_output(dto, slot, target_type, dto_map, visited):
    uid = f"{dto.exec_id}[O]{slot}"
    if uid in visited:
        return None
    visited.add(uid)

    mode = dto.node.get('mode', 0)

    if mode == _MODE_BYPASS:
        idx = _get_bypass_slot_index(
            dto.node.get('inputs', []), dto.node.get('outputs', []),
            slot, target_type,
        )
        if idx == -1:
            return None
        return _resolve_dto_input(dto, idx, dto_map, visited)

    if mode == _MODE_NEVER:
        return None

    class_type = dto.node.get('type', '')

    if dto.sg_def is not None:
        return _resolve_sg_output(dto, slot, target_type, dto_map, visited)

    if class_type == 'Reroute':
        return _resolve_dto_input(dto, slot, dto_map, visited)

    if class_type == 'PrimitiveNode':
        wv = dto.node.get('widgets_values', [])
        return ("value", wv[0]) if wv else None

    return ("link", dto.exec_id, slot)


def _resolve_sg_output(sg_dto, slot, target_type, dto_map, visited):
    for link in sg_dto.inner_links.values():
        if link.dst_node == _SUBGRAPH_OUTPUT_NODE_ID and link.dst_slot == slot:
            inner_exec_id = ':'.join(
                str(x) for x in [
                    *sg_dto.subgraph_node_path, sg_dto.node['id'], link.src_node,
                ]
            )
            inner_dto = dto_map.get(inner_exec_id)
            if inner_dto is None:
                continue
            result = _resolve_dto_output(
                inner_dto, link.src_slot, target_type, dto_map, visited,
            )
            if result is not None:
                return result
    return None


def _is_subgraph_type(class_type: str, sg_defs: dict[str, dict]) -> bool:
    return class_type in sg_defs


def is_ui_workflow(workflow: dict) -> bool:
    """Return True if *workflow* is a UI/LiteGraph workflow (not API format)."""
    return "nodes" in workflow and "links" in workflow


def convert_ui_to_api(workflow: dict) -> dict:
    """Convert a UI (LiteGraph) workflow dict to API format.

    Uses a DTO-based approach mirroring the frontend's ``graphToPrompt``
    and ``ExecutableNodeDTO`` for correct subgraph resolution at any
    nesting depth.

    Raises:
        RuntimeError: If node system is not loaded.
    """
    from ..nodes_context import get_nodes

    node_mappings = get_nodes()
    if len(node_mappings) == 0:
        raise RuntimeError(
            "Node system not loaded. Call import_all_nodes_in_workspace() first."
        )

    sg_defs = _collect_subgraph_defs(workflow)

    dto_map = _build_dto_map(workflow, sg_defs)

    api_workflow: dict[str, dict] = {}

    for dto in dto_map.values():
        node = dto.node
        class_type = node.get('type', '')
        mode = node.get('mode', 0)

        if class_type in _VIRTUAL_NODE_TYPES:
            continue
        if _is_subgraph_type(class_type, sg_defs):
            continue
        if mode in (_MODE_NEVER, _MODE_BYPASS):
            continue

        class_def = _get_node_class(node_mappings, class_type)
        input_types = _get_input_types(class_def) if class_def is not None else None
        is_unknown = class_def is None or input_types is None

        widgets_values = node.get('widgets_values')
        _wv_consumed = 0
        if is_unknown:
            api_inputs = _map_unknown_widgets(widgets_values)
            use_class_type = None
        elif isinstance(widgets_values, list):
            api_inputs, _wv_consumed = _map_widgets(input_types, widgets_values)
            use_class_type = class_type
        elif isinstance(widgets_values, dict):
            api_inputs = _map_widgets_dict(input_types, widgets_values)
            use_class_type = class_type
        else:
            api_inputs = {}
            use_class_type = class_type

        if dto.sg_node_exec_id and not is_unknown:
            sg_dto = dto_map.get(dto.sg_node_exec_id)
            if sg_dto:
                all_inputs = {**input_types.get('required', {}),
                              **input_types.get('optional', {})}
                nid = node['id']
                for (ovr_nid, wname), val in sg_dto.proxy_overrides.items():
                    if ovr_nid != nid or val is None:
                        continue
                    if wname not in all_inputs:
                        continue
                    ts, opts = _input_type_and_opts(all_inputs[wname])
                    if _is_widget_type(ts, opts) and not opts.get('forceInput'):
                        api_inputs[wname] = _wrap_value(val)

        if not is_unknown and input_types:
            _all_input_names = (
                set(input_types.get('required', {}).keys())
                | set(input_types.get('optional', {}).keys())
            )
        else:
            _all_input_names = None

        _widget_input_names: set[str] = set()
        if input_types:
            for _wn, _we in {**input_types.get('required', {}),
                             **input_types.get('optional', {})}.items():
                _wts, _wopts = _input_type_and_opts(_we)
                if _is_widget_type(_wts, _wopts):
                    _widget_input_names.add(_wn)

        for i, inp in enumerate(node.get('inputs', [])):
            inp_name = inp.get('name')
            if inp_name is None:
                continue
            link_id = inp.get('link')
            if link_id is None or link_id not in dto.graph_links:
                continue

            skip_bw = is_unknown or inp_name not in _widget_input_names
            resolved = _resolve_dto_input(dto, i, dto_map,
                                          skip_boundary_widgets=skip_bw)
            if resolved is None:
                api_inputs.pop(inp_name, None)
            elif resolved[0] == 'value':
                if _all_input_names is None or inp_name in _all_input_names:
                    api_inputs[inp_name] = _wrap_value(resolved[1])
            else:
                api_inputs[inp_name] = [resolved[1], resolved[2]]

        if use_class_type and use_class_type in _FRONTEND_INJECTED_WIDGETS:
            injected = _FRONTEND_INJECTED_WIDGETS[use_class_type]
            for j, (widget_name, default_value) in enumerate(injected):
                if widget_name not in api_inputs:
                    wv_idx = _wv_consumed + j
                    if isinstance(widgets_values, list) and wv_idx < len(widgets_values):
                        val = widgets_values[wv_idx]
                        if val is None and isinstance(default_value, str):
                            val = default_value
                        api_inputs[widget_name] = val
                    else:
                        api_inputs[widget_name] = default_value

        entry = {
            'class_type': use_class_type,
            'inputs': api_inputs,
        }
        title = node.get('title') or use_class_type
        entry['_meta'] = {'title': title}
        api_workflow[dto.exec_id] = entry

    for entry in api_workflow.values():
        inputs = entry['inputs']
        for key in list(inputs.keys()):
            val = inputs[key]
            if (
                isinstance(val, list)
                and len(val) == 2
                and isinstance(val[0], str)
                and val[0] not in api_workflow
            ):
                del inputs[key]

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
        logger.debug(f"Failed to get INPUT_TYPES for {class_def}: {exc}")
    return None
