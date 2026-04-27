"""Workflow parameter discovery and application.

A `Param` represents one user-facing knob on a workflow: a (node_id, widget_name)
pair with its current value, type, optional role tags, and a tier for ranking.

`discover(workflow)` runs a registry of *predicates*, each of which contributes
candidate `Param`s (or annotates existing ones identified by `(node_id, widget_name)`).
The pool of candidates is then merged and ranked.

The first predicate, `frontend_widget_pool`, mirrors the bundled ComfyUI
frontend's subgraph "Advanced Inputs" enumeration
(`src/components/rightSidePanel/parameters/TabSubgraphInputs.vue`,
`src/core/graph/subgraph/unpromotedWidgetUtils.ts`): every non-disabled widget
on every interior node is a candidate — no class-type heuristic. Other
predicates (added in later stages) annotate candidates with roles, raise tier,
or contribute candidates the frontend rule alone wouldn't capture.

`apply(workflow, param, value)` mutates the workflow in place by writing
`value` into the node's `inputs[widget_name]`.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..component_model.workflow_convert import convert_ui_to_api, is_ui_workflow
from ..component_model import prompt_utils as _pu

logger = logging.getLogger(__name__)


# (class_type, widget_name) -> role.
# Single source of truth for class-type-driven role tagging. Re-uses the
# class-type sets in prompt_utils so we don't duplicate them; stage 5 will
# fold the override-application code in prompt_utils onto these same roles.
_DIRECT_ROLES: dict[tuple[str, str], str] = {}


def _seed_direct_roles() -> None:
    if _DIRECT_ROLES:
        return
    for ct in _pu._STEPS_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "steps")] = "steps"
    for ct in _pu._CFG_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "cfg")] = "cfg"
    for ct in _pu._SAMPLER_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "sampler_name")] = "sampler"
    for ct in _pu._SCHEDULER_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "scheduler")] = "scheduler"
    for ct in _pu._DENOISE_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "denoise")] = "denoise"
    for ct in _pu._LATENT_SIZE_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "width")] = "width"
        _DIRECT_ROLES[(ct, "height")] = "height"
        _DIRECT_ROLES[(ct, "batch_size")] = "batch_size"
    for ct in _pu._CHECKPOINT_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "ckpt_name")] = "checkpoint"
    for ct in _pu._DIFFUSION_MODEL_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "unet_name")] = "unet"
    for ct, field_name in _pu._SEED_FIELDS.items():
        _DIRECT_ROLES[(ct, field_name)] = "seed"
    # Filesystem and URL loaders both expose a media-bearing widget; the role
    # is the same regardless of which form the workflow uses. Tagging only —
    # the URL-rewrite that --image performs lives at apply-time, not here.
    for ct in _pu._IMAGE_LOAD_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "image")] = "image_input"
        _DIRECT_ROLES[(ct, "value")] = "image_input"
    for ct in _pu._VIDEO_LOAD_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "video")] = "video_input"
        _DIRECT_ROLES[(ct, "value")] = "video_input"
    for ct in _pu._AUDIO_LOAD_CLASS_TYPES:
        _DIRECT_ROLES[(ct, "audio")] = "audio_input"
        _DIRECT_ROLES[(ct, "value")] = "audio_input"
    for ct, fields in _pu._TEXT_ENCODE_FIELDS.items():
        for field_name in fields:
            _DIRECT_ROLES[(ct, field_name)] = "text_encode"


_seed_direct_roles()


TIER_HEADLINE = 0
TIER_COMMON = 1
TIER_ADVANCED = 2


@dataclass
class Param:
    node_id: str
    class_type: str
    widget_name: str
    value: Any
    type: str = "ANY"
    options: list[Any] = field(default_factory=list)
    roles: set[str] = field(default_factory=set)
    tier: int = TIER_ADVANCED
    flag_name: str | None = None
    label: str | None = None
    source_predicates: list[str] = field(default_factory=list)

    @property
    def address(self) -> tuple[str, str]:
        return (self.node_id, self.widget_name)


Predicate = Callable[[dict, dict | None], Iterable[Param]]


def _is_link(value: Any) -> bool:
    """Mirror the API-format link shape: [src_node_id, src_slot_idx]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, list):
        return "COMBO"
    return "ANY"


def frontend_widget_pool(api: dict, ui: dict | None) -> list[Param]:
    """Every non-link input on every API node is a candidate Param.

    Mirrors the frontend's subgraph "Advanced Inputs" enumeration: walk
    interior nodes and yield every widget whose value isn't a connected
    link. The frontend filters by `widget.computedDisabled`; the API-format
    equivalent is "input value is not a [src_node_id, src_slot] pair".
    """
    out: list[Param] = []
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        title = ((node.get("_meta") or {}).get("title")) if isinstance(node.get("_meta"), dict) else None
        for widget_name, value in (node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    label=title,
                    source_predicates=["frontend_widget_pool"],
                )
            )
    return out


def class_type_roles(api: dict, ui: dict | None) -> list[Param]:
    """Annotate widgets whose ``(class_type, widget_name)`` is in the role table.

    Tags candidates with the matching role and lifts tier from advanced to
    common. ``text_encode`` is a generic tag covering every text-encoder widget;
    the more specific ``prompt`` / ``negative_prompt`` polarity is applied on
    top by `prompt_polarity`.
    """
    out: list[Param] = []
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        for widget_name, value in (node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            role = _DIRECT_ROLES.get((class_type, widget_name))
            if role is None:
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={role},
                    tier=TIER_COMMON,
                    source_predicates=["class_type_roles"],
                )
            )
    return out


def _slug(name: str) -> str:
    """Lowercase, ASCII-safe form of a Set_/title fragment used for role keys."""
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def set_node_pairs(api: dict, ui: dict | None) -> list[Param]:
    """SetNode titled ``Set_<Name>`` blesses its immediate upstream node.

    The Set/Get pattern is the workflow author's canonical "named variable":
    a `Set_<Name>` declares "this wire is the parameter ``<Name>``". We trace
    its single input link to the source node and tag every non-link widget on
    that source with ``role="set:<slug>"``, lifting tier to headline.

    SetNode is a UI-only virtual class (stripped by ``convert_ui_to_api``),
    so we read titles + links from the UI workflow and address widgets in
    the API workflow by source node id.
    """
    if ui is None:
        return []

    nodes_by_id = {n.get("id"): n for n in (ui.get("nodes") or [])}
    links_by_id = {link[0]: link for link in (ui.get("links") or []) if link}

    out: list[Param] = []
    for node in ui.get("nodes") or []:
        if node.get("type") != "SetNode":
            continue
        title = node.get("title") or ""
        if not title.startswith("Set_"):
            continue
        slug = _slug(title[len("Set_"):])
        if not slug:
            continue

        inputs = node.get("inputs") or []
        if not inputs:
            continue
        link_id = inputs[0].get("link")
        link = links_by_id.get(link_id) if link_id is not None else None
        if link is None:
            continue
        src_node_id = link[1]
        src_node = nodes_by_id.get(src_node_id)
        if src_node is None:
            continue

        api_node = api.get(str(src_node_id))
        if not isinstance(api_node, dict):
            continue
        class_type = api_node.get("class_type") or src_node.get("type") or ""
        title_label = title[len("Set_"):].replace("_", " ")

        for widget_name, value in (api_node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(src_node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={f"set:{slug}"},
                    tier=TIER_HEADLINE,
                    label=title_label,
                    source_predicates=["set_node_pairs"],
                )
            )
    return out


# PrimitiveNode is the legacy virtual primitive (stripped during conversion).
# The typed primitives are real nodes that survive into the API; we tag the
# primitive's own widget value with the primitive's title.
_TYPED_PRIMITIVE_CLASSES = frozenset({
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveBoolean",
})


def primitive_nodes(api: dict, ui: dict | None) -> list[Param]:
    """`PrimitiveNode` (virtual) and typed `Primitive*` are user-visible inputs.

    For ``PrimitiveNode``: it's stripped during ``convert_ui_to_api`` and its
    ``widgets_values[0]`` is propagated onto the consuming node's widget. We
    walk the UI to find each consumer's widget and tag *that* Param.

    For typed primitives (``PrimitiveInt``/``PrimitiveFloat``/``PrimitiveString``/
    ``PrimitiveStringMultiline``/``PrimitiveBoolean``): they survive as their
    own API nodes; we tag their own widget.

    Both forms get ``role="primitive:<slug>"`` at headline tier, where slug is
    derived from the node's ``title`` (or its node id if untitled).
    """
    out: list[Param] = []

    # Typed primitives via API (no UI required).
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        if class_type not in _TYPED_PRIMITIVE_CLASSES:
            continue
        title = ((node.get("_meta") or {}).get("title")) if isinstance(node.get("_meta"), dict) else None
        slug = _slug(title) if title else f"node_{node_id}"
        for widget_name, value in (node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={f"primitive:{slug}"},
                    tier=TIER_HEADLINE,
                    label=title,
                    source_predicates=["primitive_nodes"],
                )
            )

    # Legacy PrimitiveNode requires UI to find consumers.
    if ui is None:
        return out

    nodes_by_id = {n.get("id"): n for n in (ui.get("nodes") or [])}
    links_by_id = {link[0]: link for link in (ui.get("links") or []) if link}

    for node in ui.get("nodes") or []:
        if node.get("type") != "PrimitiveNode":
            continue
        title = node.get("title") or ""
        slug = _slug(title) if title else f"node_{node.get('id')}"

        outputs = node.get("outputs") or []
        for slot_idx, output in enumerate(outputs):
            for link_id in output.get("links") or []:
                link = links_by_id.get(link_id)
                if link is None:
                    continue
                dst_node_id = link[3]
                dst_slot = link[4]
                dst_node = nodes_by_id.get(dst_node_id)
                if dst_node is None:
                    continue
                dst_inputs = dst_node.get("inputs") or []
                if dst_slot >= len(dst_inputs):
                    continue
                widget = (dst_inputs[dst_slot].get("widget") or {})
                widget_name = widget.get("name")
                if not widget_name:
                    continue
                api_node = api.get(str(dst_node_id))
                if not isinstance(api_node, dict):
                    continue
                value = (api_node.get("inputs") or {}).get(widget_name)
                if value is None or _is_link(value):
                    continue
                out.append(
                    Param(
                        node_id=str(dst_node_id),
                        class_type=api_node.get("class_type") or dst_node.get("type") or "",
                        widget_name=widget_name,
                        value=value,
                        type=_infer_type(value),
                        roles={f"primitive:{slug}"},
                        tier=TIER_HEADLINE,
                        label=title or None,
                        source_predicates=["primitive_nodes"],
                    )
                )
    return out


# Map (class_type, widget_name) -> role for the easy-use convenience nodes.
# Sourced from yolain/ComfyUI-Easy-Use: easy seed has a `seed` widget,
# easy positive a `positive` widget (multiline string), easy negative a
# `negative` widget. easy showAnything is a UI debugger, not a parameter.
_EASY_PACK_ROLES: dict[tuple[str, str], str] = {
    ("easy seed", "seed"): "seed",
    ("easy positive", "positive"): "prompt",
    ("easy negative", "negative"): "negative_prompt",
}


def easy_pack_nodes(api: dict, ui: dict | None) -> list[Param]:
    """Tag widgets on comfyui-easy-use's seed/positive/negative convenience nodes.

    These are explicit "user input" nodes by convention — when the workflow
    author drops one in, they're declaring "this is the user's parameter".
    Promotes to headline tier with a role matching the standard one
    (``seed`` / ``prompt`` / ``negative_prompt``) so the same CLI flag wires
    to either the easy-pack node or a stock KSampler/CLIPTextEncode.
    """
    out: list[Param] = []
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        for widget_name, value in (node.get("inputs") or {}).items():
            role = _EASY_PACK_ROLES.get((class_type, widget_name))
            if role is None:
                continue
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=widget_name,
                    value=value,
                    type=_infer_type(value),
                    roles={role},
                    tier=TIER_HEADLINE,
                    source_predicates=["easy_pack_nodes"],
                )
            )
    return out


def prompt_polarity(api: dict, ui: dict | None) -> list[Param]:
    """Disambiguate positive vs negative text encoder among ``text_encode`` candidates.

    Mirrors the heuristic stack in ``prompt_utils.find_positive_text_encoder`` /
    ``find_negative_text_encoder`` (positive-input ref, BasicGuider conditioning,
    title keyword, sole-encoder fallback). Yields candidate Params that the
    merger overlays onto the matching ``text_encode`` Params from
    `class_type_roles`.
    """
    out: list[Param] = []
    positive_node = _pu.find_positive_text_encoder(api)
    negative_node = _pu.find_negative_text_encoder(api)

    def _emit(node_id: str | None, role: str) -> None:
        if node_id is None:
            return
        node = api.get(node_id) or {}
        class_type = node.get("class_type") or ""
        fields = _pu._TEXT_ENCODE_FIELDS.get(class_type, [])
        for field in fields:
            if field not in (node.get("inputs") or {}):
                continue
            value = node["inputs"][field]
            if _is_link(value):
                continue
            out.append(
                Param(
                    node_id=str(node_id),
                    class_type=class_type,
                    widget_name=field,
                    value=value,
                    type=_infer_type(value),
                    roles={role},
                    tier=TIER_COMMON,
                    source_predicates=["prompt_polarity"],
                )
            )

    _emit(positive_node, "prompt")
    _emit(negative_node, "negative_prompt")
    return out


_PREDICATES: list[tuple[str, Predicate]] = [
    ("frontend_widget_pool", frontend_widget_pool),
    ("class_type_roles", class_type_roles),
    ("prompt_polarity", prompt_polarity),
    ("set_node_pairs", set_node_pairs),
    ("primitive_nodes", primitive_nodes),
    ("easy_pack_nodes", easy_pack_nodes),
]


def _merge(into: dict[tuple[str, str], Param], candidate: Param) -> None:
    existing = into.get(candidate.address)
    if existing is None:
        into[candidate.address] = candidate
        return
    existing.roles |= candidate.roles
    if candidate.tier < existing.tier:
        existing.tier = candidate.tier
    for source in candidate.source_predicates:
        if source not in existing.source_predicates:
            existing.source_predicates.append(source)
    if existing.flag_name is None and candidate.flag_name is not None:
        existing.flag_name = candidate.flag_name
    if existing.label is None and candidate.label is not None:
        existing.label = candidate.label
    if existing.type == "ANY" and candidate.type != "ANY":
        existing.type = candidate.type
    if not existing.options and candidate.options:
        existing.options = candidate.options


def _to_api(workflow: dict) -> tuple[dict, dict | None]:
    if is_ui_workflow(workflow):
        return convert_ui_to_api(workflow), workflow
    return workflow, None


def _rank(params: list[Param]) -> list[Param]:
    return sorted(params, key=lambda p: (p.tier, p.node_id, p.widget_name))


def discover(workflow: dict) -> list[Param]:
    api, ui = _to_api(workflow)
    candidates: dict[tuple[str, str], Param] = {}
    for _name, predicate in _PREDICATES:
        for cand in predicate(api, ui):
            _merge(candidates, cand)
    return _rank(list(candidates.values()))


def params_by_role(params: Iterable[Param], role: str) -> list[Param]:
    return [p for p in params if role in p.roles]


def params_by_address(
    params: Iterable[Param], node_id: str, widget_name: str
) -> Param | None:
    for p in params:
        if p.node_id == str(node_id) and p.widget_name == widget_name:
            return p
    return None


def apply(workflow: dict, param: Param, value: Any) -> dict:
    """Return a copy of *workflow* (API format) with *param*'s widget set to *value*."""
    if is_ui_workflow(workflow):
        raise ValueError(
            "apply() expects an API-format workflow; call convert_ui_to_api first"
        )
    out = copy.deepcopy(workflow)
    node = out.get(param.node_id)
    if not isinstance(node, dict):
        raise KeyError(f"node {param.node_id!r} not in workflow")
    inputs = node.setdefault("inputs", {})
    inputs[param.widget_name] = value
    return out
