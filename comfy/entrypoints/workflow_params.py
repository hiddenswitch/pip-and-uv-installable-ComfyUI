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

logger = logging.getLogger(__name__)


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


_PREDICATES: list[tuple[str, Predicate]] = [
    ("frontend_widget_pool", frontend_widget_pool),
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
