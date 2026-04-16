"""Declarative rules for rewriting class_types in UI workflows."""
from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ClassTypeRewriteRule:
    match: frozenset[str]
    to: str
    default_inputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    drop_inputs: frozenset[str] = frozenset()
    reason: str = ""


def rewrite_api_node(
    node: dict,
    to: str,
    *,
    replace_inputs: Mapping[str, Any] | None = None,
    default_inputs: Mapping[str, Any] | None = None,
    drop_inputs: Iterable[str] = (),
    drop_meta: bool = False,
) -> None:
    node["class_type"] = to
    if replace_inputs is not None:
        node["inputs"] = dict(replace_inputs)
    else:
        inputs = node.setdefault("inputs", {})
        for key in drop_inputs:
            inputs.pop(key, None)
        if default_inputs:
            for key, value in default_inputs.items():
                inputs.setdefault(key, value)
    if drop_meta:
        node.pop("_meta", None)


DEFAULT_REWRITE_RULES: list[ClassTypeRewriteRule] = [
    ClassTypeRewriteRule(
        match=frozenset({"UnetLoaderGGUF", "UnetLoaderGGUFAdvanced"}),
        to="UNETLoader",
        default_inputs={"weight_dtype": "default"},
        reason=".gguf handled natively by core UNETLoader",
    ),
    ClassTypeRewriteRule(
        match=frozenset({"CLIPLoaderGGUF"}),
        to="CLIPLoader",
        default_inputs={"type": "stable_diffusion", "device": "default"},
        reason=".gguf handled natively by core CLIPLoader",
    ),
    ClassTypeRewriteRule(
        match=frozenset({"DualCLIPLoaderGGUF"}),
        to="DualCLIPLoader",
        default_inputs={"type": "flux", "device": "default"},
        reason=".gguf handled natively by core DualCLIPLoader",
    ),
]


def _find_rule(
    class_type: str | None,
    rules: Iterable[ClassTypeRewriteRule],
) -> ClassTypeRewriteRule | None:
    if not class_type:
        return None
    for rule in rules:
        if class_type in rule.match:
            return rule
    return None


def rewrite_class_type(
    class_type: str,
    rules: Iterable[ClassTypeRewriteRule] = DEFAULT_REWRITE_RULES,
) -> str:
    rule = _find_rule(class_type, rules)
    return rule.to if rule is not None else class_type


def _log_applied(counts: Mapping[str, int], rules: Iterable[ClassTypeRewriteRule]) -> None:
    for old, count in counts.items():
        rule = _find_rule(old, rules)
        if rule is None:
            continue
        logger.info("Rewrote %d node(s) %s → %s (%s)", count, old, rule.to, rule.reason)


def apply_to_ui_workflow(
    ui_workflow: dict,
    rules: Iterable[ClassTypeRewriteRule] = DEFAULT_REWRITE_RULES,
) -> dict:
    ui_workflow = copy.deepcopy(ui_workflow)
    counts: dict[str, int] = {}
    for node in ui_workflow.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        rule = _find_rule(node.get("type"), rules)
        if rule is None:
            continue
        counts[node["type"]] = counts.get(node["type"], 0) + 1
        node["type"] = rule.to
        props = node.get("properties")
        if isinstance(props, dict) and "Node name for S&R" in props:
            props["Node name for S&R"] = rule.to
    _log_applied(counts, rules)
    return ui_workflow


def apply_to_api_workflow(
    api_workflow: dict,
    rules: Iterable[ClassTypeRewriteRule] = DEFAULT_REWRITE_RULES,
) -> dict:
    api_workflow = copy.deepcopy(api_workflow)
    counts: dict[str, int] = {}
    for node in api_workflow.values():
        if not isinstance(node, dict):
            continue
        rule = _find_rule(node.get("class_type"), rules)
        if rule is None:
            continue
        counts[node["class_type"]] = counts.get(node["class_type"], 0) + 1
        rewrite_api_node(
            node,
            rule.to,
            default_inputs=rule.default_inputs,
            drop_inputs=rule.drop_inputs,
        )
    _log_applied(counts, rules)
    return api_workflow
