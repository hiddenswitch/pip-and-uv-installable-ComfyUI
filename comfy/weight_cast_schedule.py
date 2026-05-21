from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
import logging
import os
from typing import Any

import torch
import torch.fx

from .weight_cast import get_materialization_spec
from .weight_cast_ops import get_registered_module

logger = logging.getLogger(__name__)
DEBUG_DUMP_ENV = "COMFY_WEIGHT_PREFETCH_DUMP"


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
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        module_bytes = _tensor_nbytes(weight)
        if _is_resolve_weight_bias(node):
            module_bytes += _tensor_nbytes(bias)
        if module_bytes > 0:
            return module_bytes
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
    lookahead = max(0, int(lookahead))
    if lookahead == 0:
        return gm
    graph = gm.graph
    original_nodes = list(graph.nodes)
    original_index = {node: index for index, node in enumerate(original_nodes)}
    resolve_nodes = [node for node in graph.nodes if _is_resolve_weight(node) or _is_resolve_weight_bias(node)]
    if not resolve_nodes:
        return gm

    resolve_sizes = [_resolve_nbytes(node) for node in resolve_nodes]
    replacements: dict[torch.fx.Node, torch.fx.Node] = {}
    for index, node in enumerate(resolve_nodes):
        anchor_index = _anchor_index_for_prefetch(
            resolve_sizes,
            index,
            lookahead=lookahead,
            budget_bytes=budget_bytes,
        )
        args = tuple(node.args)
        node_index = original_index[node]
        insertion_index = original_index[resolve_nodes[anchor_index]]
        is_self_prefetch = insertion_index >= node_index
        anchor = node if is_self_prefetch else _prefetch_after_compute_anchor(replacements[resolve_nodes[anchor_index]], graph)

        with graph.inserting_before(anchor):
            if _is_resolve_weight_bias(node):
                prefetch = graph.call_function(torch.ops.comfy_weight.prefetch_weight_bias, args=args[3:])
            else:
                prefetch = graph.call_function(torch.ops.comfy_weight.prefetch_weight, args=args[2:])
            prefetch.meta.update(node.meta)
            if not is_self_prefetch:
                _attach_prefetch_to_anchor(graph, anchor, prefetch)

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
        replacements[node] = resolved

    graph.lint()
    gm.recompile()
    return gm


def _attach_prefetch_to_anchor(graph: torch.fx.Graph, anchor: torch.fx.Node, prefetch: torch.fx.Node) -> None:
    """Thread a no-op dependency from a prefetch token into its scheduling anchor.

    Inductor may legally sink an otherwise-functional prefetch custom op to its
    resolve consumer. The anchor op preserves the async-copy launch order in
    dataflow without waiting on the copy itself.
    """
    tensor_arg = _first_tensor_node_arg(anchor)
    if tensor_arg is None:
        return
    with graph.inserting_before(anchor):
        anchored = graph.call_function(torch.ops.comfy_weight.prefetch_anchor, args=(tensor_arg, prefetch))
        anchored.meta.update(tensor_arg.meta)
    anchor.replace_input_with(tensor_arg, anchored)


def _first_tensor_node_arg(node: torch.fx.Node) -> torch.fx.Node | None:
    result: torch.fx.Node | None = None

    def visit(value: torch.fx.node.Argument) -> torch.fx.node.Argument:
        nonlocal result
        if result is None and isinstance(value, torch.fx.Node):
            result = value
        return value

    torch.fx.map_arg((node.args, node.kwargs), visit)
    return result


def _prefetch_after_compute_anchor(resolved: torch.fx.Node, graph: torch.fx.Graph) -> torch.fx.Node:
    """Return the release/next node after the resolved weight's compute use.

    Future prefetches must not run before an earlier resolve has materialized
    its VBAR view and queued the consuming kernel. Anchoring on the release
    preserves overlap with subsequent queued compute without allowing a later
    VBAR fault to remap the view before its consumer has launched.
    """
    live_nodes = list(graph.nodes)
    live_set = set(live_nodes)
    order = {node: index for index, node in enumerate(live_nodes)}
    users = [user for user in resolved.users if user in live_set]
    if not users:
        return resolved
    compute = min(users, key=lambda user: order[user])
    compute_users = [user for user in compute.users if user in live_set]
    release_users = [user for user in compute_users if "release_" in _target_name(user)]
    if release_users:
        return min(release_users, key=lambda user: order[user])
    if compute_users:
        return min(compute_users, key=lambda user: order[user])
    return compute


def wrap_backend_with_weight_prefetch_scheduler(
    backend: str | Callable[..., Any],
    *,
    lookahead: int = 0,
    budget_bytes: int | None = None,
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Any]:
    if isinstance(backend, str):
        from torch._dynamo.backends.registry import lookup_backend

        compile_backend = lookup_backend(backend)
    else:
        compile_backend = backend

    def scheduled_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor], **kwargs):
        scheduled = schedule_weight_prefetches(gm, lookahead=lookahead, budget_bytes=budget_bytes)
        _dump_scheduled_graph_if_requested(scheduled)
        kwargs.pop("options", None)
        kwargs.pop("mode", None)
        try:
            return compile_backend(scheduled, example_inputs, **kwargs)
        except AssertionError:
            _log_topological_sort_blockers(scheduled.graph)
            raise

    return scheduled_backend


def _dump_scheduled_graph_if_requested(gm: torch.fx.GraphModule) -> None:
    dump_path = os.environ.get(DEBUG_DUMP_ENV)
    if not dump_path:
        return
    try:
        with open(dump_path, "w", encoding="utf-8") as handle:
            for index, node in enumerate(gm.graph.nodes):
                detail = ""
                if node.op == "call_function":
                    name = _target_name(node)
                    args = tuple(node.args)
                    if "resolve_prefetched_weight_bias" in name and len(args) >= 7:
                        detail = f" module={args[4]} invocation={args[5]}"
                    elif "resolve_prefetched_weight" in name and len(args) >= 6:
                        detail = f" module={args[3]} invocation={args[4]}"
                    elif "resolve_weight_bias" in name and len(args) >= 6:
                        detail = f" module={args[3]} invocation={args[4]}"
                    elif "resolve_weight" in name and len(args) >= 5:
                        detail = f" module={args[2]} invocation={args[3]}"
                    elif "prefetch_weight_bias" in name and len(args) >= 2:
                        detail = f" module={args[0]} invocation={args[1]}"
                    elif "prefetch_weight" in name and len(args) >= 2:
                        detail = f" module={args[0]} invocation={args[1]}"
                    elif "release_" in name and len(args) >= 3:
                        detail = f" module={args[1]} invocation={args[2]}"
                handle.write(f"{index:05d} {node.op} {_target_name(node)} {node.name}{detail}\n")
    except Exception:
        logger.exception("Failed to dump scheduled weight prefetch graph to %s", dump_path)


def _log_topological_sort_blockers(graph: torch.fx.Graph) -> None:
    pending = list(reversed(graph.nodes))
    ready = set()
    waiting: dict[torch.fx.Node, list[torch.fx.Node]] = defaultdict(list)
    while pending:
        node = pending.pop()
        waiting_for = [arg for arg in _node_args(node) if arg not in ready]
        if waiting_for:
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            pending.extend(reversed(waiting.pop(node, ())))
    if not waiting and len(ready) == len(graph.nodes):
        return
    graph_nodes = set(graph.nodes)
    for dependency, blocked in list(waiting.items())[:10]:
        logger.error(
            "Inductor topological sort blocker: dependency=%s in_graph=%s blocked=%s",
            dependency,
            dependency in graph_nodes,
            [str(node) for node in blocked[:5]],
        )


def _node_args(node: torch.fx.Node) -> list[torch.fx.node.Argument]:
    args: list[torch.fx.node.Argument] = []
    torch.fx.map_arg((node.args, node.kwargs), args.append)
    return args
