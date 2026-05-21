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
PREFETCH_BUDGET_MB_ENV = "COMFY_WEIGHT_PREFETCH_BUDGET_MB"
PREFETCH_LOOKAHEAD_ENV = "COMFY_WEIGHT_PREFETCH_LOOKAHEAD"


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


def _is_prefetched_resolve_weight(node: torch.fx.Node) -> bool:
    name = _target_name(node)
    return (
        node.op == "call_function"
        and "resolve_prefetched_weight" in name
        and "bias" not in name
    )


def _is_prefetched_resolve_weight_bias(node: torch.fx.Node) -> bool:
    name = _target_name(node)
    return node.op == "call_function" and "resolve_prefetched_weight_bias" in name


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


def _dtype_element_size(dtype_code: Any) -> int:
    try:
        code = int(dtype_code)
    except Exception:
        return 0
    if code == 1:
        return 4
    if code in {2, 3, 4, 5}:
        return 2 if code in {2, 3} else 1
    return 0


def _shape_nbytes(shape: Any, dtype_code: Any) -> int:
    shape = _tensor_meta(shape)
    if not isinstance(shape, (list, tuple)):
        return 0
    numel = 1
    try:
        for dim in shape:
            numel *= int(dim)
    except Exception:
        return 0
    return int(numel) * _dtype_element_size(dtype_code)


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
    if _is_prefetched_resolve_weight_bias(node):
        module_key_index = 4
        dtype_index = 6
        bias_dtype_index = 7
    elif _is_prefetched_resolve_weight(node):
        module_key_index = 3
        dtype_index = 5
        bias_dtype_index = None
    elif _is_resolve_weight_bias(node):
        module_key_index = 3
        dtype_index = 5
        bias_dtype_index = 6
    else:
        module_key_index = 2
        dtype_index = 4
        bias_dtype_index = None
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
    if _is_resolve_weight_bias(node) or _is_prefetched_resolve_weight_bias(node):
        return (
            _tensor_nbytes(args[1])
            + _tensor_nbytes(args[2])
            + _shape_nbytes(args[1], args[dtype_index])
            + _shape_nbytes(args[2], args[bias_dtype_index])
        )
    return _tensor_nbytes(args[1]) + _shape_nbytes(args[1], args[dtype_index])


def schedule_weight_prefetches(
    gm: torch.fx.GraphModule,
    *,
    lookahead: int = 2,
    budget_bytes: int | None = None,
) -> torch.fx.GraphModule:
    """Split graph-visible weight resolves into budgeted prefetch + wait-resolve.

    The manual-cast graph initially contains:

        resolve_weight[_bias] -> compute -> release

    That makes the copy synchronous at the point of use. This pass makes the
    memory resource visible as graph dataflow: prefetches consume a memory
    token, and releases return one. A future prefetch is schedulable as soon as
    the graph has produced enough release tokens to satisfy the configured
    budget.
    """
    lookahead = max(0, int(lookahead))
    if lookahead == 0 or budget_bytes is None or budget_bytes <= 0:
        return gm
    graph = gm.graph
    resolve_nodes = [node for node in graph.nodes if _is_resolve_weight(node) or _is_resolve_weight_bias(node)]
    if not resolve_nodes:
        return gm

    release_nodes = _release_nodes_by_invocation(graph)
    memory_seed = _insert_memory_seed(graph, resolve_nodes[0])
    resolve_sizes = [_resolve_nbytes(node) for node in resolve_nodes]
    in_flight: list[tuple[int, torch.fx.Node]] = []
    insertion_anchors: dict[torch.fx.Node, torch.fx.Node] = {}
    resident_bytes = 0

    for index, node in enumerate(resolve_nodes):
        args = tuple(node.args)
        invocation = _resolve_invocation(node)
        release = release_nodes.get(invocation)
        size = max(1, resolve_sizes[index])

        if release is None:
            continue

        if size > budget_bytes:
            release_token = _replace_release_with_memory_token(graph, release, memory_seed)
            in_flight = [(budget_bytes, release_token)]
            resident_bytes = budget_bytes
            continue

        freed_tokens: list[torch.fx.Node] = []
        while resident_bytes + size > budget_bytes and in_flight:
            freed_size, freed_token = in_flight.pop(0)
            resident_bytes -= freed_size
            freed_tokens.append(freed_token)

        if resident_bytes + size > budget_bytes:
            release_token = _replace_release_with_memory_token(graph, release, memory_seed)
            in_flight = [(budget_bytes, release_token)]
            resident_bytes = budget_bytes
            continue

        memory_token = _join_memory_tokens(graph, freed_tokens, memory_seed, node)
        prefetch = _insert_prefetch_after_memory_token(graph, node, memory_token, insertion_anchors)

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
        release_token = _replace_release_with_memory_token(graph, release, prefetch)
        in_flight.append((size, release_token))
        resident_bytes += size

    graph.lint()
    gm.recompile()
    return gm


def _resolve_invocation(node: torch.fx.Node) -> tuple[Any, Any]:
    args = tuple(node.args)
    module_key_index = 3 if _is_resolve_weight_bias(node) else 2
    return args[module_key_index], args[module_key_index + 1]


def _is_release(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and "release_" in _target_name(node)


def _release_invocation(node: torch.fx.Node) -> tuple[Any, Any] | None:
    args = tuple(node.args)
    if len(args) < 3:
        return None
    return args[-2], args[-1]


def _release_nodes_by_invocation(graph: torch.fx.Graph) -> dict[tuple[Any, Any], torch.fx.Node]:
    releases: dict[tuple[Any, Any], torch.fx.Node] = {}
    for node in graph.nodes:
        if not _is_release(node):
            continue
        invocation = _release_invocation(node)
        if invocation is not None:
            releases[invocation] = node
    return releases


def _insert_memory_seed(graph: torch.fx.Graph, first_resolve: torch.fx.Node) -> torch.fx.Node:
    exemplar = tuple(first_resolve.args)[0]
    with graph.inserting_before(first_resolve):
        seed = graph.call_function(torch.ops.comfy_weight.memory_seed, args=(exemplar,))
    seed.meta.update(getattr(exemplar, "meta", {}))
    return seed


def _join_memory_tokens(
    graph: torch.fx.Graph,
    freed_tokens: list[torch.fx.Node],
    seed: torch.fx.Node,
    anchor: torch.fx.Node,
) -> torch.fx.Node:
    token = seed
    for freed in freed_tokens:
        with graph.inserting_before(anchor):
            token = graph.call_function(torch.ops.comfy_weight.memory_join, args=(token, freed))
            token.meta.update(freed.meta)
    return token


def _insert_prefetch_after_memory_token(
    graph: torch.fx.Graph,
    resolve: torch.fx.Node,
    memory_token: torch.fx.Node,
    insertion_anchors: dict[torch.fx.Node, torch.fx.Node],
) -> torch.fx.Node:
    args = tuple(resolve.args)
    insertion_anchor = insertion_anchors.get(memory_token, memory_token)
    with graph.inserting_after(insertion_anchor):
        if _is_resolve_weight_bias(resolve):
            prefetch = graph.call_function(torch.ops.comfy_weight.prefetch_weight_bias_after, args=(memory_token, *args[3:]))
        else:
            prefetch = graph.call_function(torch.ops.comfy_weight.prefetch_weight_after, args=(memory_token, *args[2:]))
        prefetch.meta.update(resolve.meta)
    insertion_anchors[memory_token] = prefetch
    return prefetch


def _replace_release_with_memory_token(
    graph: torch.fx.Graph,
    release: torch.fx.Node,
    memory_token: torch.fx.Node,
) -> torch.fx.Node:
    if "release_memory_" in _target_name(release):
        return release
    args = tuple(release.args)
    with graph.inserting_before(release):
        released = graph.call_function(torch.ops.comfy_weight.release_memory_, args=(args[0], memory_token, args[1], args[2]))
        released.meta.update(memory_token.meta)
    release.replace_all_uses_with(released)
    graph.erase_node(release)
    return released




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
        effective_lookahead = int(os.environ.get(PREFETCH_LOOKAHEAD_ENV, lookahead))
        effective_budget = budget_bytes
        if os.environ.get(PREFETCH_BUDGET_MB_ENV):
            effective_budget = int(os.environ[PREFETCH_BUDGET_MB_ENV]) * 1024 * 1024
            if effective_lookahead == 0:
                effective_lookahead = 1_000_000
        scheduled = schedule_weight_prefetches(gm, lookahead=effective_lookahead, budget_bytes=effective_budget)
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
                        detail = f" module={args[4]} invocation={args[5]} nbytes={_resolve_nbytes(node)}"
                    elif "resolve_prefetched_weight" in name and len(args) >= 6:
                        detail = f" module={args[3]} invocation={args[4]} nbytes={_resolve_nbytes(node)}"
                    elif "resolve_weight_bias" in name and len(args) >= 6:
                        detail = f" module={args[3]} invocation={args[4]} nbytes={_resolve_nbytes(node)}"
                    elif "resolve_weight" in name and len(args) >= 5:
                        detail = f" module={args[2]} invocation={args[3]} nbytes={_resolve_nbytes(node)}"
                    elif "prefetch_weight_bias_after" in name and len(args) >= 3:
                        detail = f" token={args[0]} module={args[1]} invocation={args[2]}"
                    elif "prefetch_weight_after" in name and len(args) >= 3:
                        detail = f" token={args[0]} module={args[1]} invocation={args[2]}"
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
