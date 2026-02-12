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
from typing import Optional

from .litegraph_types import LiteLink

logger = logging.getLogger(__name__)

# ── widget-type classification ────────────────────────────────────────────────
# These INPUT_TYPES type strings produce UI widgets rather than connection slots.
_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})

# Virtual node types that never appear in the API output.
_VIRTUAL_NODE_TYPES = frozenset({
    "Reroute",
    "PrimitiveNode",
    "Note",
    "MarkdownNote",
})

# ── LiteGraph node mode constants ─────────────────────────────────────────────
_MODE_ALWAYS = 0
_MODE_ON_EVENT = 1
_MODE_NEVER = 2  # muted
_MODE_ON_TRIGGER = 3
_MODE_BYPASS = 4

# Subgraph boundary node IDs used by the frontend.
_SUBGRAPH_INPUT_NODE_ID = -10
_SUBGRAPH_OUTPUT_NODE_ID = -20

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_widget_type(type_spec) -> bool:
    """Return True if *type_spec* from INPUT_TYPES represents a widget."""
    if isinstance(type_spec, list):
        return True  # COMBO – list of choices
    return isinstance(type_spec, str) and type_spec in _WIDGET_TYPES


def _input_type_and_opts(entry) -> tuple:
    """Extract ``(type_spec, options_dict)`` from an INPUT_TYPES value."""
    if isinstance(entry, (list, tuple)):
        type_spec = entry[0]
        opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        return type_spec, opts
    return entry, {}


def _extra_widgets_after(opts: dict) -> list[str | None]:
    """Return names of extra frontend widgets inserted after this input.

    Each entry is the serialized name (used as the API key) or ``None`` when
    the extra value must be consumed but *not* included in the API output
    (because the frontend marks it ``serialize: false``).
    """
    extras: list[str | None] = []
    if opts.get("control_after_generate"):
        extras.append(None)  # frontend-only, serialize=false
    if opts.get("image_upload") or opts.get("video_upload") or opts.get("audio_upload"):
        extras.append("upload")  # included in API
    return extras


def _wrap_value(val):
    """Wrap list values as ``{"__value__": val}`` for API output."""
    return {"__value__": val} if isinstance(val, list) else val


def _map_widgets(input_types: dict, widgets_values: list) -> dict[str, object]:
    """Map a ``widgets_values`` positional array to ``{input_name: value}``.

    Walks INPUT_TYPES in declaration order (required then optional), consuming
    from *widgets_values* for widget-type inputs and skipping connection types.
    Extra frontend widgets (e.g. ``control_after_generate``) are consumed and
    either included or discarded depending on their serialize flag.
    """
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})

    result: dict[str, object] = {}
    idx = 0

    for name, entry in list(required.items()) + list(optional.items()):
        type_spec, opts = _input_type_and_opts(entry)

        if not _is_widget_type(type_spec):
            continue
        # ``forceInput`` turns a widget into a pure connection slot – no widget
        # value stored.
        if opts.get("forceInput"):
            continue

        # Consume the widget value.
        if idx < len(widgets_values):
            result[name] = _wrap_value(widgets_values[idx])
            idx += 1
        else:
            break

        # Consume (and optionally store) extra frontend widgets that follow.
        for extra_name in _extra_widgets_after(opts):
            if idx < len(widgets_values):
                if extra_name is not None:
                    result[extra_name] = _wrap_value(widgets_values[idx])
                idx += 1

    return result


# ── link resolution ───────────────────────────────────────────────────────────

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
    """Find the input slot that a bypass node should route to for *slot*.

    Mirrors ``ExecutableNodeDTO._getBypassSlotIndex`` from the frontend:

    1. Any-type shortcut: if *target_type* is ``*`` or empty, prefer the
       input at the same index, else fall back to 0.
    2. If the opposite input (same index) is a valid connection for both the
       output type *and* the target type, use it.
    3. Find first input with exact *target_type* match.
    4. Find first input with a valid connection to both the output type and
       the target type.
    """
    if not target_type or target_type == "*":
        return slot if slot < len(inputs) else 0

    out_type = outputs[slot].get("type") if slot < len(outputs) else None

    # Prefer opposite slot.
    if slot < len(inputs):
        opp_type = inputs[slot].get("type")
        if _types_match(opp_type, out_type) and _types_match(opp_type, target_type):
            return slot

    # Exact match.
    for i, inp in enumerate(inputs):
        if inp.get("type") == target_type:
            return i

    # Any valid connection.
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
    """Trace a link back through bypass / reroute / primitive nodes.

    *target_type* is the type of the requesting input (used for bypass
    slot matching).

    Returns one of:

    * ``("link", node_id_str, slot_index)`` – a normal resolved connection.
    * ``("value", python_value)`` – resolved to a literal value (PrimitiveNode).
    * ``None`` – dead end (muted, disconnected, cycle).
    """
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

    # ── Reroute: transparently follow the single input ────────────────────
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

    # ── PrimitiveNode: yield its stored widget value ──────────────────────
    if node_type == "PrimitiveNode":
        wv = node.get("widgets_values", [])
        if wv:
            return ("value", wv[0])
        return None

    # ── Muted node (mode=2): dead end ─────────────────────────────────────
    if mode == _MODE_NEVER:
        return None

    # ── Bypassed node (mode=4): route through matching input ──────────────
    # Mirrors frontend ``ExecutableNodeDTO._getBypassSlotIndex``.
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

    # ── Normal / active node: this is the source ──────────────────────────
    return ("link", str(src_node_id), src_slot)


# ── subgraph / group-node expansion ──────────────────────────────────────────

def _collect_subgraph_defs(workflow: dict) -> dict[str, dict]:
    """Extract subgraph definitions keyed by UUID from the workflow."""
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
    """Parse a link from either list (outer) or dict (inner subgraph) format."""
    if isinstance(raw, dict):
        return LiteLink.from_dict(raw)
    return LiteLink.from_list(raw)


def _build_inner_links(sg_def: dict) -> dict[int, LiteLink]:
    """Build a link index for a subgraph's internal links."""
    inner_links: dict[int, LiteLink] = {}
    for raw in sg_def.get("links", []):
        link = _parse_link(raw)
        inner_links[link.link_id] = link
    return inner_links


def _build_input_boundary(
    outer_node: dict,
    sg_def: dict,
    outer_nodes_by_id: dict[int, dict],
    outer_links: dict[int, LiteLink],
) -> dict[int, tuple | None]:
    """Map subgraph input slots to resolved outer sources.

    Returns ``{slot_index: resolved}`` where *resolved* is one of:

    * ``("link", node_id_str, slot_index)``
    * ``("value", python_value)``
    * ``None`` – not connected
    """
    boundary: dict[int, tuple | None] = {}
    sg_inputs = sg_def.get("inputs", [])
    proxy_widgets = outer_node.get("properties", {}).get("proxyWidgets", [])
    outer_wv = outer_node.get("widgets_values", [])

    # 1. Populate from proxy widget values (for "-1" / interface inputs).
    for pw_idx, pw in enumerate(proxy_widgets):
        if not isinstance(pw, (list, tuple)) or len(pw) < 2:
            continue
        pw_target, pw_name = pw[0], pw[1]
        if str(pw_target) != "-1":
            continue
        if pw_idx >= len(outer_wv):
            continue
        # Find which subgraph input slot this name corresponds to.
        for slot_idx, sg_inp in enumerate(sg_inputs):
            if sg_inp.get("name") == pw_name:
                boundary[slot_idx] = ("value", outer_wv[pw_idx])
                break

    # 2. Override with resolved outer link connections.
    for outer_inp in outer_node.get("inputs", []):
        link_id = outer_inp.get("link")
        if link_id is None or link_id not in outer_links:
            continue
        inp_name = outer_inp.get("name")
        # Find corresponding subgraph input slot.
        for slot_idx, sg_inp in enumerate(sg_inputs):
            if sg_inp.get("name") == inp_name:
                outer_link = outer_links[link_id]
                resolved = _resolve_source(
                    outer_link.src_node, outer_link.src_slot,
                    outer_nodes_by_id, outer_links,
                    target_type=outer_inp.get("type"),
                )
                boundary[slot_idx] = resolved
                break

    return boundary


def _build_proxy_overrides(
    outer_node: dict,
) -> dict[tuple[int, str], object]:
    """Build ``{(inner_node_id, widget_name): value}`` from proxyWidgets."""
    overrides: dict[tuple[int, str], object] = {}
    proxy_widgets = outer_node.get("properties", {}).get("proxyWidgets", [])
    outer_wv = outer_node.get("widgets_values", [])

    for pw_idx, pw in enumerate(proxy_widgets):
        if not isinstance(pw, (list, tuple)) or len(pw) < 2:
            continue
        pw_target, pw_name = pw[0], pw[1]
        if str(pw_target) == "-1":
            continue  # interface inputs handled by input boundary
        if pw_idx >= len(outer_wv):
            continue
        try:
            inner_nid = int(pw_target)
        except (ValueError, TypeError):
            continue
        overrides[(inner_nid, pw_name)] = outer_wv[pw_idx]

    return overrides


def _convert_subgraph(
    outer_node: dict,
    sg_def: dict,
    outer_nodes_by_id: dict[int, dict],
    outer_links: dict[int, LiteLink],
    node_mappings,
) -> tuple[dict[str, dict], dict[int, tuple[str, int]]]:
    """Expand a subgraph node into API entries for its inner nodes.

    Returns:
        api_entries: ``{prefixed_id: {class_type, inputs}}``
        output_map: ``{output_slot: (prefixed_id, inner_slot)}``
    """
    outer_id = outer_node["id"]

    inner_links = _build_inner_links(sg_def)
    inner_nodes_by_id: dict[int, dict] = {}
    for n in sg_def.get("nodes", []):
        nid = n.get("id")
        if nid is not None:
            inner_nodes_by_id[nid] = n

    # Add a synthetic entry for inputNode so _resolve_source can traverse to it.
    inner_nodes_by_id[_SUBGRAPH_INPUT_NODE_ID] = {
        "type": "_SubgraphInput", "mode": 0,
    }

    input_boundary = _build_input_boundary(
        outer_node, sg_def, outer_nodes_by_id, outer_links,
    )
    proxy_overrides = _build_proxy_overrides(outer_node)

    # Build output map: output slot → (inner source node id, inner source slot).
    output_map: dict[int, tuple[str, int]] = {}
    for link in inner_links.values():
        if link.dst_node == _SUBGRAPH_OUTPUT_NODE_ID:
            # Resolve through any bypass/reroute inside the subgraph.
            resolved = _resolve_source(
                link.src_node, link.src_slot, inner_nodes_by_id, inner_links,
            )
            if resolved is not None and resolved[0] == "link":
                resolved_id = resolved[1]
                if resolved_id == str(_SUBGRAPH_INPUT_NODE_ID):
                    # Pass-through subgraph: output connects directly to input.
                    boundary = input_boundary.get(resolved[2])
                    if boundary is not None:
                        output_map[link.dst_slot] = boundary
                else:
                    prefixed = f"{outer_id}:{resolved_id}"
                    output_map[link.dst_slot] = (prefixed, resolved[2])

    # Convert each inner node.
    api_entries: dict[str, dict] = {}

    for inner_node in sg_def.get("nodes", []):
        inner_nid = inner_node.get("id")
        if inner_nid is None or inner_nid < 0:
            continue

        mode = inner_node.get("mode", 0)
        if mode in (_MODE_NEVER, _MODE_BYPASS):
            continue

        class_type = inner_node.get("type")
        if class_type is None or class_type in _VIRTUAL_NODE_TYPES:
            continue

        # Inner subgraphs (nested) – skip for now (very rare in templates).
        class_def = _get_node_class(node_mappings, class_type)
        if class_def is None:
            logger.debug("Skipping unknown inner node type: %s", class_type)
            continue

        input_types = _get_input_types(class_def)
        if input_types is None:
            continue

        # Map inner widget values.
        widgets_values = inner_node.get("widgets_values")
        if isinstance(widgets_values, list):
            api_inputs = _map_widgets(input_types, widgets_values)
        elif isinstance(widgets_values, dict):
            api_inputs = {
                k: _wrap_value(v) for k, v in widgets_values.items()
            }
        else:
            api_inputs = {}

        # Apply direct proxy widget overrides.
        for (nid, wname), val in proxy_overrides.items():
            if nid == inner_nid:
                api_inputs[wname] = _wrap_value(val)

        # Resolve inner linked inputs.
        for inp in inner_node.get("inputs", []):
            inp_name = inp.get("name")
            if inp_name is None:
                continue
            link_id = inp.get("link")
            if link_id is None or link_id not in inner_links:
                continue

            link = inner_links[link_id]

            # Resolve through bypass/reroute/primitive inside the subgraph.
            resolved = _resolve_source(
                link.src_node, link.src_slot, inner_nodes_by_id, inner_links,
                target_type=inp.get("type"),
            )

            if resolved is None:
                api_inputs.pop(inp_name, None)
                continue

            if resolved[0] == "value":
                api_inputs[inp_name] = _wrap_value(resolved[1])
                continue

            # resolved[0] == "link"
            resolved_id = resolved[1]

            if resolved_id == str(_SUBGRAPH_INPUT_NODE_ID):
                # Crossed the input boundary.
                boundary = input_boundary.get(resolved[2])
                if boundary is None:
                    api_inputs.pop(inp_name, None)
                elif boundary[0] == "value":
                    api_inputs[inp_name] = _wrap_value(boundary[1])
                elif boundary[0] == "link":
                    api_inputs[inp_name] = [boundary[1], boundary[2]]
            else:
                # Normal inner link – prefix the ID.
                api_inputs[inp_name] = [f"{outer_id}:{resolved_id}", resolved[2]]

        prefixed_id = f"{outer_id}:{inner_nid}"
        api_entries[prefixed_id] = {
            "class_type": class_type,
            "inputs": api_inputs,
        }

    return api_entries, output_map


def _is_subgraph_type(class_type: str, sg_defs: dict[str, dict]) -> bool:
    """Return True if *class_type* is a subgraph/group node UUID."""
    return class_type in sg_defs


# ── public API ────────────────────────────────────────────────────────────────

def convert_ui_to_api(workflow: dict) -> dict:
    """Convert a UI (LiteGraph) workflow dict to API format.

    Handles muted nodes (mode 2), bypassed nodes (mode 4), reroute /
    primitive / note virtual nodes, ``control_after_generate`` /
    ``image_upload`` extra widgets, converted widgets, subgraph/group
    node expansion, ``__value__`` wrapping of list widget values, and
    dangling-link cleanup.

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

    # ── collect subgraph definitions ─────────────────────────────────────
    sg_defs = _collect_subgraph_defs(workflow)

    # ── build indices ─────────────────────────────────────────────────────
    links: dict[int, LiteLink] = {}
    for raw in workflow.get("links", []):
        link = LiteLink.from_list(raw)
        links[link.link_id] = link

    nodes_by_id: dict[int, dict] = {}
    for node in workflow.get("nodes", []):
        nid = node.get("id")
        if nid is not None:
            nodes_by_id[nid] = node

    # ── expand subgraphs ─────────────────────────────────────────────────
    # output_remaps: maps (group_node_id, output_slot) to the resolved
    # inner source so outer nodes can find the correct connection.
    output_remaps: dict[tuple[int, int], tuple] = {}

    # ── convert each eligible node ────────────────────────────────────────
    api_workflow: dict[str, dict] = {}

    for node in workflow.get("nodes", []):
        node_id = node.get("id")
        if node_id is None:
            continue

        mode = node.get("mode", 0)
        if mode in (_MODE_NEVER, _MODE_BYPASS):
            continue

        class_type = node.get("type")
        if class_type is None:
            continue
        if class_type in _VIRTUAL_NODE_TYPES:
            continue

        # ── subgraph / group node ────────────────────────────────────────
        if _is_subgraph_type(class_type, sg_defs):
            entries, output_map = _convert_subgraph(
                node, sg_defs[class_type], nodes_by_id, links, node_mappings,
            )
            api_workflow.update(entries)
            for slot, target in output_map.items():
                output_remaps[(node_id, slot)] = target
            continue

        class_def = _get_node_class(node_mappings, class_type)
        if class_def is None:
            logger.debug("Skipping unknown node type: %s", class_type)
            continue

        input_types = _get_input_types(class_def)
        if input_types is None:
            continue

        # Map widget values ------------------------------------------------
        widgets_values = node.get("widgets_values")
        if isinstance(widgets_values, list):
            api_inputs = _map_widgets(input_types, widgets_values)
        elif isinstance(widgets_values, dict):
            api_inputs = {
                k: _wrap_value(v) for k, v in widgets_values.items()
            }
        else:
            api_inputs = {}

        # Resolve linked inputs (overrides widget values) ------------------
        for inp in node.get("inputs", []):
            inp_name = inp.get("name")
            if inp_name is None:
                continue
            link_id = inp.get("link")
            if link_id is None or link_id not in links:
                continue

            link = links[link_id]

            # Check if the source is a group node – use the output remap.
            remap = output_remaps.get((link.src_node, link.src_slot))
            if remap is not None:
                if isinstance(remap, tuple) and len(remap) >= 2:
                    if remap[0] == "value":
                        api_inputs[inp_name] = _wrap_value(remap[1])
                    else:
                        api_inputs[inp_name] = [remap[0], remap[1]]
                continue

            resolved = _resolve_source(
                link.src_node, link.src_slot, nodes_by_id, links,
                target_type=inp.get("type"),
            )
            if resolved is None:
                # Dead link – remove any widget value for this input.
                api_inputs.pop(inp_name, None)
            elif resolved[0] == "value":
                api_inputs[inp_name] = _wrap_value(resolved[1])
            else:
                # Check if resolved to a group node (bypass/reroute resolves).
                resolved_nid = int(resolved[1]) if resolved[1].lstrip("-").isdigit() else None
                remap2 = output_remaps.get((resolved_nid, resolved[2])) if resolved_nid is not None else None
                if remap2 is not None:
                    if isinstance(remap2, tuple) and len(remap2) >= 2:
                        if remap2[0] == "value":
                            api_inputs[inp_name] = _wrap_value(remap2[1])
                        else:
                            api_inputs[inp_name] = [remap2[0], remap2[1]]
                else:
                    api_inputs[inp_name] = [resolved[1], resolved[2]]

        api_workflow[str(node_id)] = {
            "class_type": class_type,
            "inputs": api_inputs,
        }

    # ── clean up dangling references ──────────────────────────────────────
    for entry in api_workflow.values():
        inputs = entry["inputs"]
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


# ── node-class helpers ────────────────────────────────────────────────────────

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
