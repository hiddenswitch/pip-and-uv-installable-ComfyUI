from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from typing import Any

import torch
import torch.fx

from . import model_management
from .weight_cast import get_materialization_spec
from .weight_cast_ops import get_registered_module

logger = logging.getLogger(__name__)
DEBUG_DUMP_ENV = "COMFY_WEIGHT_PREFETCH_DUMP"
PREFETCH_BUDGET_MB_ENV = "COMFY_WEIGHT_PREFETCH_BUDGET_MB"
PREFETCH_LOOKAHEAD_ENV = "COMFY_WEIGHT_PREFETCH_LOOKAHEAD"
PREFETCH_MAX_WEIGHT_MB_ENV = "COMFY_WEIGHT_PREFETCH_MAX_WEIGHT_MB"
DEFAULT_PREFETCH_MAX_WEIGHT_BYTES = 256 * 1024 * 1024
PATCH_MATERIALIZATION_RESERVATION_FACTOR = 3
AUTO_BUDGET_HEADROOM_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class _ScheduleItem:
    node: torch.fx.Node
    kind: str
    size: int
    release: torch.fx.Node | None
    prefetchable: bool


@dataclass(frozen=True)
class _ScheduleDecision:
    scheduled: bool
    dependencies: tuple[int, ...]
    reserved_size: int


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


def _is_materialize_fp8(node: torch.fx.Node) -> bool:
    name = _target_name(node)
    return (
        node.op == "call_function"
        and "materialize_per_tensor_fp8" in name
        and "after" not in name
    )


def _tensor_meta(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return value.meta.get("val", value.meta.get("example_value"))
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


def _materialization_nbytes(node: torch.fx.Node) -> int:
    args = tuple(node.args)
    if len(args) < 3:
        return 0
    return _tensor_nbytes(node) or _shape_nbytes(getattr(_tensor_meta(args[0]), "shape", None), args[2])


def _module_from_arg(value: Any) -> torch.nn.Module | None:
    value = _tensor_meta(value)
    if not isinstance(value, (torch.Tensor, int)):
        return None
    try:
        return get_registered_module(value)
    except Exception:
        return None


def _module_has_patch_materialization(module: torch.nn.Module) -> bool:
    for attr in ("weight_function", "bias_function"):
        functions = getattr(module, attr, ())
        if functions:
            return True
    for attr in ("weight_lowvram_function", "bias_lowvram_function"):
        if getattr(module, attr, None) is not None:
            return True
    return False


def _reserve_patch_materialization_scratch(module: torch.nn.Module, nbytes: int) -> int:
    if nbytes <= 0 or not _module_has_patch_materialization(module):
        return nbytes
    return nbytes * PATCH_MATERIALIZATION_RESERVATION_FACTOR


def _resolve_module(node: torch.fx.Node) -> torch.nn.Module | None:
    args = tuple(node.args)
    if _is_prefetched_resolve_weight_bias(node):
        module_key_index = 4
    elif _is_prefetched_resolve_weight(node):
        module_key_index = 3
    elif _is_resolve_weight_bias(node):
        module_key_index = 3
    else:
        module_key_index = 2
    if len(args) <= module_key_index:
        return None
    return _module_from_arg(args[module_key_index])


def _can_prefetch_resolve(node: torch.fx.Node) -> bool:
    module = _resolve_module(node)
    if module is None:
        return True
    return not _module_has_patch_materialization(module)


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
    module = _resolve_module(node)
    if module is not None:
        spec = get_materialization_spec(module)
        spec_bytes = spec.vram_bytes
        if spec_bytes > 0:
            if spec.has_python_materialization:
                return spec_bytes
            return _reserve_patch_materialization_scratch(module, spec_bytes)
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        module_bytes = _tensor_nbytes(weight)
        if _is_resolve_weight_bias(node):
            module_bytes += _tensor_nbytes(bias)
        if module_bytes > 0:
            return _reserve_patch_materialization_scratch(module, module_bytes)
    if _is_resolve_weight_bias(node) or _is_prefetched_resolve_weight_bias(node):
        return (
            _tensor_nbytes(args[1])
            + _tensor_nbytes(args[2])
            + _shape_nbytes(args[1], args[dtype_index])
            + _shape_nbytes(args[2], args[bias_dtype_index])
        )
    return _tensor_nbytes(args[1]) + _shape_nbytes(args[1], args[dtype_index])


def _prefetch_nbytes(node: torch.fx.Node) -> int:
    args = tuple(node.args)
    if len(args) < 2:
        return 0
    module = _module_from_arg(args[1])
    if module is None:
        return 0
    spec = get_materialization_spec(module)
    if spec.vram_bytes > 0:
        if spec.has_python_materialization:
            return spec.vram_bytes
        return _reserve_patch_materialization_scratch(module, spec.vram_bytes)
    module_bytes = _tensor_nbytes(getattr(module, "weight", None))
    if "bias" in _target_name(node):
        module_bytes += _tensor_nbytes(getattr(module, "bias", None))
    return _reserve_patch_materialization_scratch(module, module_bytes)


def _first_cuda_device(example_inputs: list[torch.Tensor]) -> torch.device | None:
    for value in example_inputs:
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            return value.device
    return None


def _auto_prefetch_budget_bytes(example_inputs: list[torch.Tensor], required_bytes: int) -> int | None:
    device = _first_cuda_device(example_inputs)
    if device is None or required_bytes <= 0:
        return None
    try:
        free_bytes = int(model_management.get_free_memory(device))
        reserved_bytes = int(model_management.extra_reserved_memory()) + AUTO_BUDGET_HEADROOM_BYTES
    except Exception:
        return None
    usable_bytes = max(0, free_bytes - reserved_bytes)
    if usable_bytes <= 0:
        return None
    return min(required_bytes, usable_bytes)


def _auto_prefetch_lookahead(sizes: list[int], budget_bytes: int | None) -> int:
    if not sizes or budget_bytes is None or budget_bytes <= 0:
        return 0
    # Auto mode is constrained by measured memory credits, not by a fixed
    # window. Let every traced use be considered; the budget/release-token
    # solver decides which loads can actually be in flight.
    return len(sizes)


def _module_debug_name(value: Any) -> str:
    module = _module_from_arg(value)
    if module is None:
        return ""
    seed_key = getattr(module, "seed_key", None)
    if seed_key:
        return f" name={seed_key}"
    return f" name={type(module).__name__}"


def schedule_weight_prefetches(
    gm: torch.fx.GraphModule,
    *,
    lookahead: int = 2,
    budget_bytes: int | None = None,
    max_weight_bytes: int | None = DEFAULT_PREFETCH_MAX_WEIGHT_BYTES,
) -> torch.fx.GraphModule:
    """Split graph-visible weight resolves into budgeted prefetch + wait-resolve.

    The manual-cast graph initially contains:

        resolve_weight[_bias] -> compute -> release

    That makes the copy synchronous at the point of use. This pass makes the
    memory resource visible as graph dataflow: prefetches consume a memory
    token, and releases return one. A future prefetch is schedulable as soon as
    the graph has produced enough release tokens to satisfy the budget. With a
    lookahead at least as large as the traced use count, this is the earliest
    feasible schedule for an ordered forward: initial loads start from the seed
    until capacity is exhausted, and each later load depends only on the exact
    release tokens needed to make room.
    """
    lookahead = max(0, int(lookahead))
    if lookahead == 0 or budget_bytes is None or budget_bytes <= 0:
        return gm
    graph = gm.graph
    items = _collect_schedule_items(graph)
    if not items:
        return gm

    memory_seed = _insert_memory_seed(graph, items[0].node)
    _schedule_memory_items(graph, items, memory_seed, budget_bytes, lookahead, max_weight_bytes)

    graph.lint()
    gm.recompile()
    return gm


def _collect_schedule_items(graph: torch.fx.Graph) -> list[_ScheduleItem]:
    release_nodes = _release_nodes_by_invocation(graph)
    items: list[_ScheduleItem] = []
    for node in graph.nodes:
        if _is_resolve_weight(node) or _is_resolve_weight_bias(node):
            invocation = _resolve_invocation(node)
            items.append(
                _ScheduleItem(
                    node=node,
                    kind="resolve",
                    size=max(1, _resolve_nbytes(node)),
                    release=release_nodes.get(invocation),
                    prefetchable=_can_prefetch_resolve(node),
                )
            )
        elif _is_materialize_fp8(node):
            items.append(
                _ScheduleItem(
                    node=node,
                    kind="materialize",
                    size=max(1, _materialization_nbytes(node)),
                    release=_last_user_node(graph, node),
                    prefetchable=True,
                )
            )
    return items


def _schedule_memory_items(
    graph: torch.fx.Graph,
    items: list[_ScheduleItem],
    memory_seed: torch.fx.Node,
    budget_bytes: int,
    lookahead: int,
    max_weight_bytes: int | None,
) -> None:
    insertion_anchors: dict[torch.fx.Node, torch.fx.Node] = {}
    decisions = _solve_memory_schedule(
        items,
        budget_bytes=budget_bytes,
        lookahead=lookahead,
        max_weight_bytes=max_weight_bytes,
    )
    release_tokens: dict[int, torch.fx.Node] = {}

    for index, item in enumerate(items):
        node = item.node
        if node.graph is None:
            continue
        release = item.release
        decision = decisions[index]

        if release is None:
            continue

        if not decision.scheduled:
            release_token = _reserve_unscheduled_item(graph, item, release, memory_seed)
            release_tokens[index] = release_token
            continue

        freed_tokens = [release_tokens[dependency] for dependency in decision.dependencies]

        memory_token = _join_memory_tokens(graph, freed_tokens, memory_seed, node)
        if item.kind == "resolve":
            active_token = _rewrite_resolve_with_prefetch(graph, node, memory_token, insertion_anchors)
            release_token = _replace_release_with_memory_token(graph, release, active_token)
        else:
            active_token = _replace_materialization_with_memory_token(graph, node, memory_token, insertion_anchors)
            release_token = _insert_materialization_release(graph, release, active_token, memory_token)
        release_tokens[index] = release_token


def _solve_memory_schedule(
    items: list[_ScheduleItem],
    *,
    budget_bytes: int,
    lookahead: int,
    max_weight_bytes: int | None,
) -> list[_ScheduleDecision]:
    in_flight: list[tuple[int, int]] = []
    resident_bytes = 0
    decisions: list[_ScheduleDecision] = []

    for index, item in enumerate(items):
        size = item.size
        if item.release is None:
            decisions.append(_ScheduleDecision(False, (), 0))
            continue
        can_schedule = (
            item.prefetchable
            and size <= budget_bytes
            and not (item.kind == "resolve" and max_weight_bytes is not None and size > max_weight_bytes)
        )
        if not can_schedule:
            decisions.append(_ScheduleDecision(False, (), budget_bytes))
            in_flight.append((budget_bytes, index))
            resident_bytes += budget_bytes
            continue

        freed: list[int] = []
        while (len(in_flight) >= lookahead or resident_bytes + size > budget_bytes) and in_flight:
            freed_size, freed_index = in_flight.pop(0)
            resident_bytes -= freed_size
            freed.append(freed_index)

        if resident_bytes + size > budget_bytes:
            decisions.append(_ScheduleDecision(False, (), budget_bytes))
            in_flight = [(budget_bytes, index)]
            resident_bytes = budget_bytes
            continue

        decisions.append(_ScheduleDecision(True, tuple(freed), size))
        in_flight.append((size, index))
        resident_bytes += size

    return decisions


def _reserve_unscheduled_item(
    graph: torch.fx.Graph,
    item: _ScheduleItem,
    release: torch.fx.Node,
    memory_seed: torch.fx.Node,
) -> torch.fx.Node:
    if item.kind == "resolve":
        return _replace_release_with_memory_token(graph, release, memory_seed)
    return _insert_materialization_release(graph, release, item.node, memory_seed)


def _rewrite_resolve_with_prefetch(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    memory_token: torch.fx.Node,
    insertion_anchors: dict[torch.fx.Node, torch.fx.Node],
) -> torch.fx.Node:
    args = tuple(node.args)
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
    return prefetch


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
    if not freed_tokens:
        return seed
    token = freed_tokens[0]
    for freed in freed_tokens[1:]:
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


def _replace_materialization_with_memory_token(
    graph: torch.fx.Graph,
    materialize: torch.fx.Node,
    memory_token: torch.fx.Node,
    insertion_anchors: dict[torch.fx.Node, torch.fx.Node],
) -> torch.fx.Node:
    args = tuple(materialize.args)
    insertion_anchor = insertion_anchors.get(memory_token, memory_token)
    with graph.inserting_after(insertion_anchor):
        accounted = graph.call_function(
            torch.ops.comfy_quant.materialize_per_tensor_fp8_after,
            args=(memory_token, *args),
        )
        accounted.meta.update(materialize.meta)
    insertion_anchors[memory_token] = accounted
    materialize.replace_all_uses_with(accounted)
    graph.erase_node(materialize)
    return accounted


def _last_user_node(graph: torch.fx.Graph, node: torch.fx.Node) -> torch.fx.Node | None:
    order = {candidate: index for index, candidate in enumerate(graph.nodes)}
    users = [user for user in node.users if user in order]
    if not users:
        return None
    return max(users, key=lambda user: order[user])


def _insert_materialization_release(
    graph: torch.fx.Graph,
    release_anchor: torch.fx.Node,
    materialized: torch.fx.Node,
    memory_token: torch.fx.Node,
) -> torch.fx.Node:
    with graph.inserting_after(release_anchor):
        released = graph.call_function(
            torch.ops.comfy_quant.release_materialization_,
            args=(release_anchor, materialized, memory_token),
        )
        released.meta.update(memory_token.meta)
    return released


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
        resolve_nodes = [node for node in gm.graph.nodes if _is_resolve_weight(node) or _is_resolve_weight_bias(node)]
        materialize_nodes = [node for node in gm.graph.nodes if _is_materialize_fp8(node)]
        schedule_sizes = [_resolve_nbytes(node) for node in resolve_nodes] + [_materialization_nbytes(node) for node in materialize_nodes]
        required_bytes = sum(max(1, size) for size in schedule_sizes)

        explicit_lookahead = os.environ.get(PREFETCH_LOOKAHEAD_ENV)
        effective_lookahead = int(explicit_lookahead) if explicit_lookahead is not None else int(lookahead)
        effective_budget = budget_bytes
        if os.environ.get(PREFETCH_BUDGET_MB_ENV):
            effective_budget = int(os.environ[PREFETCH_BUDGET_MB_ENV]) * 1024 * 1024
            if explicit_lookahead is None and effective_lookahead == 0:
                effective_lookahead = _auto_prefetch_lookahead(schedule_sizes, effective_budget)
        elif effective_budget is None:
            effective_budget = _auto_prefetch_budget_bytes(example_inputs, required_bytes)
            if explicit_lookahead is None and effective_lookahead == 0:
                effective_lookahead = _auto_prefetch_lookahead(schedule_sizes, effective_budget)
        effective_max_weight = DEFAULT_PREFETCH_MAX_WEIGHT_BYTES
        if effective_budget is not None and not os.environ.get(PREFETCH_MAX_WEIGHT_MB_ENV):
            effective_max_weight = None
        if os.environ.get(PREFETCH_MAX_WEIGHT_MB_ENV):
            max_weight_mb = int(os.environ[PREFETCH_MAX_WEIGHT_MB_ENV])
            effective_max_weight = None if max_weight_mb <= 0 else max_weight_mb * 1024 * 1024
        scheduled = schedule_weight_prefetches(
            gm,
            lookahead=effective_lookahead,
            budget_bytes=effective_budget,
            max_weight_bytes=effective_max_weight,
        )
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
                        detail = f" module={args[4]} invocation={args[5]} nbytes={_resolve_nbytes(node)}{_module_debug_name(args[4])}"
                    elif "resolve_prefetched_weight" in name and len(args) >= 6:
                        detail = f" module={args[3]} invocation={args[4]} nbytes={_resolve_nbytes(node)}{_module_debug_name(args[3])}"
                    elif "resolve_weight_bias" in name and len(args) >= 6:
                        detail = f" module={args[3]} invocation={args[4]} nbytes={_resolve_nbytes(node)}{_module_debug_name(args[3])}"
                    elif "resolve_weight" in name and len(args) >= 5:
                        detail = f" module={args[2]} invocation={args[3]} nbytes={_resolve_nbytes(node)}{_module_debug_name(args[2])}"
                    elif "prefetch_weight_bias_after" in name and len(args) >= 3:
                        detail = f" token={args[0]} module={args[1]} invocation={args[2]} nbytes={_prefetch_nbytes(node)}{_module_debug_name(args[1])}"
                    elif "prefetch_weight_after" in name and len(args) >= 3:
                        detail = f" token={args[0]} module={args[1]} invocation={args[2]} nbytes={_prefetch_nbytes(node)}{_module_debug_name(args[1])}"
                    elif "prefetch_weight_bias" in name and len(args) >= 2:
                        detail = f" module={args[0]} invocation={args[1]}"
                    elif "prefetch_weight" in name and len(args) >= 2:
                        detail = f" module={args[0]} invocation={args[1]}"
                    elif "release_memory_" in name and len(args) >= 4:
                        detail = f" token={args[1]} module={args[2]} invocation={args[3]}"
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
