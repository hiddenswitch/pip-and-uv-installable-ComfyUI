from __future__ import annotations

from collections.abc import Callable
import logging
import os
from typing import Any

import torch
import torch.fx

from .weight_cast import get_materialization_spec
from .weight_cast_ops import get_registered_module

logger = logging.getLogger(__name__)
DEFAULT_PREFETCH_BUDGET_BYTES = 2 * 1024 ** 3


def _target_name(node: torch.fx.Node) -> str:
    target = node.target
    name = getattr(target, "__name__", None)
    if name is not None:
        return name
    return str(target)


def _is_resolve_weight(node: torch.fx.Node) -> bool:
    name = _target_name(node)
    return (
        node.op == "call_function"
        and "prefetched" not in name
        and ("comfy_weight.resolve_weight" in name or name == "resolve_weight")
    )


def _is_resolve_weight_bias(node: torch.fx.Node) -> bool:
    name = _target_name(node)
    return (
        node.op == "call_function"
        and "prefetched" not in name
        and ("comfy_weight.resolve_weight_bias" in name or name == "resolve_weight_bias")
    )


def _tensor_meta(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return value.meta.get("val")
    return value


def _tensor_nbytes(value: Any) -> int:
    value = _tensor_meta(value)
    if value is None or not hasattr(value, "numel") or not hasattr(value, "element_size"):
        return 0
    try:
        return int(value.numel()) * int(value.element_size())
    except Exception:
        return 0


def _module_from_arg(value: Any) -> torch.nn.Module | None:
    value = _tensor_meta(value)
    if not isinstance(value, (torch.Tensor, int)):
        return None
    try:
        return get_registered_module(value)
    except Exception:
        return None


def _resolve_nbytes(node: torch.fx.Node) -> int:
    args = tuple(node.args)
    module_key_index = 3 if _is_resolve_weight_bias(node) else 2
    module = _module_from_arg(args[module_key_index])
    if module is not None:
        spec_bytes = get_materialization_spec(module).vram_bytes
        if spec_bytes > 0:
            return spec_bytes
    if _is_resolve_weight_bias(node):
        return _tensor_nbytes(args[1]) + _tensor_nbytes(args[2])
    return _tensor_nbytes(args[1])


def _anchor_index_for_prefetch(
    resolve_sizes: list[int],
    index: int,
    *,
    lookahead: int,
    budget_bytes: int | None,
) -> int:
    earliest = max(0, index - lookahead)
    if budget_bytes is None or budget_bytes <= 0:
        return earliest

    total = 0
    anchor = index
    for candidate in range(index, earliest - 1, -1):
        candidate_size = resolve_sizes[candidate]
        if candidate != index and total + candidate_size > budget_bytes:
            break
        total += candidate_size
        anchor = candidate
    return anchor


def schedule_weight_prefetches(
    gm: torch.fx.GraphModule,
    *,
    lookahead: int = 2,
    budget_bytes: int | None = None,
) -> torch.fx.GraphModule:
    """Split graph-visible weight resolves into prefetch + wait-resolve.

    The manual-cast graph initially contains:

        resolve_weight[_bias] -> compute -> release

    That makes the copy synchronous at the point of use. This pass derives
    transfer intent from the FX graph, inserts a prefetch token earlier in
    graph order, and rewrites the resolve node to consume that token. Runtime
    callbacks still enforce stream waits, but the graph now exposes a movable
    async transfer boundary.
    """
    lookahead = max(1, int(lookahead))
    graph = gm.graph
    resolve_nodes = [node for node in graph.nodes if _is_resolve_weight(node) or _is_resolve_weight_bias(node)]
    if not resolve_nodes:
        return gm

    resolve_sizes = [_resolve_nbytes(node) for node in resolve_nodes]
    for index, node in enumerate(resolve_nodes):
        anchor_index = _anchor_index_for_prefetch(
            resolve_sizes,
            index,
            lookahead=lookahead,
            budget_bytes=budget_bytes,
        )
        anchor = resolve_nodes[anchor_index]
        args = tuple(node.args)

        with graph.inserting_before(anchor):
            if _is_resolve_weight_bias(node):
                prefetch = graph.call_function(torch.ops.comfy_weight.prefetch_weight_bias, args=args)
            else:
                prefetch = graph.call_function(torch.ops.comfy_weight.prefetch_weight, args=args)
            prefetch.meta.update(node.meta)

        with graph.inserting_before(node):
            if _is_resolve_weight_bias(node):
                new_args = args[:3] + (prefetch,) + args[3:]
                resolved = graph.call_function(torch.ops.comfy_weight.resolve_prefetched_weight_bias, args=new_args)
            else:
                new_args = args[:2] + (prefetch,) + args[2:]
                resolved = graph.call_function(torch.ops.comfy_weight.resolve_prefetched_weight, args=new_args)
            resolved.meta.update(node.meta)

        node.replace_all_uses_with(resolved)
        graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return gm


def wrap_backend_with_weight_prefetch_scheduler(
    backend: str | Callable[..., Any],
    *,
    lookahead: int = 2,
    budget_bytes: int | None = None,
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Any]:
    if isinstance(backend, str):
        from torch._dynamo.backends.registry import lookup_backend

        compile_backend = lookup_backend(backend)
    else:
        compile_backend = backend

    if budget_bytes is None:
        budget_mb = os.environ.get("COMFY_WEIGHT_PREFETCH_BUDGET_MB")
        if budget_mb:
            budget_bytes = int(float(budget_mb) * 1024 * 1024)
        else:
            budget_bytes = DEFAULT_PREFETCH_BUDGET_BYTES

    def scheduled_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor], **kwargs):
        scheduled = schedule_weight_prefetches(gm, lookahead=lookahead, budget_bytes=budget_bytes)
        kwargs.pop("options", None)
        kwargs.pop("mode", None)
        return compile_backend(scheduled, example_inputs, **kwargs)

    return scheduled_backend
