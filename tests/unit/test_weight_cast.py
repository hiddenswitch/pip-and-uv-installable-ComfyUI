from __future__ import annotations

import torch
import pytest

from comfy.cli_args_types import Configuration
from comfy.execution_context import context_configuration


def _node_target_contains(graph: torch.fx.Graph, text: str) -> list[torch.fx.Node]:
    return [node for node in graph.nodes if text in str(node.target)]


def _node_depends_on(node: torch.fx.Node, dependency: torch.fx.Node) -> bool:
    seen: set[torch.fx.Node] = set()
    stack = [arg for arg in node.all_input_nodes]
    while stack:
        current = stack.pop()
        if current is dependency:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(current.all_input_nodes)
    return False


def test_weight_cast_backends_report_dynamic_vram_disabled():
    from comfy import weight_cast

    with context_configuration(Configuration(disable_dynamic_vram=True)):
        backends = weight_cast.list_weight_cast_backends()

    assert backends["eager"]["available"] is True
    assert backends["aimdo"]["available"] is False
    assert backends["aimdo"]["disabled"] is True
    assert backends["aimdo"]["unavailable_reason"] == "dynamic VRAM disabled"


def test_weight_cast_backends_report_aimdo_available_when_allocator_exists(monkeypatch):
    from comfy import memory_management, weight_cast

    monkeypatch.setattr(memory_management, "aimdo_allocator", object())
    with context_configuration(Configuration(disable_dynamic_vram=False)):
        backends = weight_cast.list_weight_cast_backends()

    assert backends["aimdo"]["available"] is True
    assert backends["aimdo"]["unavailable_reason"] is None
    assert backends["graph_visible"]["available"] is True


def test_manual_cast_linear_preserves_eager_output():
    from comfy import ops

    layer = ops.manual_cast.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]]))
        layer.bias.copy_(torch.tensor([0.25, -0.5]))
    x = torch.tensor([[1.0, -2.0, 0.5]])

    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

    assert torch.allclose(layer(x), expected)


def test_manual_cast_embedding_preserves_float_weight_dtype():
    from comfy import ops

    layer = ops.manual_cast.Embedding(4, 3, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ],
                dtype=torch.bfloat16,
            )
        )
    tokens = torch.tensor([[0, 2, 3]], dtype=torch.long)

    out = layer(tokens)

    assert out.dtype is torch.bfloat16
    assert torch.allclose(out, torch.nn.functional.embedding(tokens, layer.weight))


def test_dynamic_quantized_lowvram_lora_patch_is_baked(monkeypatch):
    from comfy import model_patcher
    from comfy import ops

    class DummyQuantizedTensor:
        pass

    monkeypatch.setattr(model_patcher, "QuantizedTensor", DummyQuantizedTensor)
    monkeypatch.setattr(ops, "lowvram_lora_materialization_policy", lambda: "quantized-cache")

    assert model_patcher.should_bake_lowvram_patch(object(), DummyQuantizedTensor(), set_func=lambda _: None) is True
    assert (
        model_patcher.should_bake_lowvram_patch(
            object(),
            DummyQuantizedTensor(),
            set_func=lambda _: None,
            dynamic_lowvram=True,
        )
        is True
    )

    monkeypatch.setattr(ops, "lowvram_lora_materialization_policy", lambda: "on-demand")
    assert (
        model_patcher.should_bake_lowvram_patch(
            object(),
            DummyQuantizedTensor(),
            set_func=lambda _: None,
            dynamic_lowvram=True,
        )
        is False
    )


def test_comfy_weight_custom_ops_compile_with_eager_backend():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module

    layer = ops.manual_cast.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]]))
        layer.bias.copy_(torch.tensor([0.25, -0.5]))
    key = register_module(layer)
    invocation_id = 1
    weight_shape = module_weight_shape(layer)
    bias_shape = module_bias_shape(layer)

    def fn(x):
        weight, bias = torch.ops.comfy_weight.resolve_weight_bias(
            x, weight_shape, bias_shape, key, invocation_id, 0, 0, 0, False, 0, -1
        )
        out = torch.nn.functional.linear(x, weight, bias)
        torch.ops.comfy_weight.release_(out, key, invocation_id)
        return out

    x = torch.randn(4, 3)
    compiled = torch.compile(fn, backend="eager")

    assert torch.allclose(compiled(x), fn(x))


def test_comfy_weight_custom_ops_are_present_in_fx_graph():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module

    layer = ops.manual_cast.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]]))
        layer.bias.copy_(torch.tensor([0.25, -0.5]))
    key = register_module(layer)
    invocation_id = 1
    weight_shape = module_weight_shape(layer)
    bias_shape = module_bias_shape(layer)
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def fn(x):
        weight, bias = torch.ops.comfy_weight.resolve_weight_bias(
            x, weight_shape, bias_shape, key, invocation_id, 0, 0, 0, False, 0, -1
        )
        out = torch.nn.functional.linear(x, weight, bias)
        torch.ops.comfy_weight.release_(out, key, invocation_id)
        return out

    compiled = torch.compile(fn, backend=capture_backend)

    compiled(torch.randn(4, 3))

    graph_text = graphs[0].code
    assert "comfy_weight.resolve_weight_bias" in graph_text
    assert "comfy_weight.release" in graph_text


def test_weight_prefetch_scheduler_rewrites_future_resolves_from_fx_graph():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    layer = ops.manual_cast.Linear(3, 2)
    key = register_module(layer)
    invocation_id = 1
    weight_shape = module_weight_shape(layer)
    bias_shape = module_bias_shape(layer)
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=1, budget_bytes=40))
        return graphs[-1].forward

    def fn(x):
        weight, bias = torch.ops.comfy_weight.resolve_weight_bias(
            x, weight_shape, bias_shape, key, invocation_id, 0, 0, 0, False, 0, -1
        )
        out = torch.nn.functional.linear(x, weight, bias)
        torch.ops.comfy_weight.release_(out, key, invocation_id)
        return out

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(4, 3))

    graph_text = graphs[0].code
    assert "comfy_weight.prefetch_weight_bias_after" in graph_text
    assert "comfy_weight.resolve_prefetched_weight_bias" in graph_text
    assert "comfy_weight.resolve_weight_bias" not in graph_text


def test_weight_prefetch_scheduler_lookahead_zero_leaves_demand_resolves():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    layer = ops.manual_cast.Linear(2, 2)
    args = (
        module_weight_shape(layer),
        module_bias_shape(layer),
        register_module(layer),
        1,
    )
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=0))
        return graphs[-1].forward

    def fn(x):
        weight, bias = torch.ops.comfy_weight.resolve_weight_bias(
            x, *args, 0, 0, 0, False, 0, -1
        )
        out = torch.nn.functional.linear(x, weight, bias)
        torch.ops.comfy_weight.release_(out, args[2], args[3])
        return out

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    graph_text = graphs[0].code
    assert "comfy_weight.resolve_weight_bias" in graph_text
    assert "comfy_weight.prefetch_weight_bias" not in graph_text


def test_memory_schedule_solver_emits_earliest_capacity_dependencies():
    import comfy.weight_cast_schedule as schedule

    items = [
        schedule._ScheduleItem(None, "resolve", 4, object(), True),
        schedule._ScheduleItem(None, "resolve", 2, object(), True),
        schedule._ScheduleItem(None, "resolve", 2, object(), True),
        schedule._ScheduleItem(None, "resolve", 3, object(), True),
        schedule._ScheduleItem(None, "resolve", 1, object(), True),
    ]

    decisions = schedule._solve_memory_schedule(
        items,
        budget_bytes=6,
        lookahead=len(items),
        max_weight_bytes=None,
    )

    assert [decision.scheduled for decision in decisions] == [True] * len(items)
    assert [decision.dependencies for decision in decisions] == [
        (),
        (),
        (0,),
        (1,),
        (),
    ]


def test_memory_schedule_solver_handles_unscheduled_items_as_full_capacity_barriers():
    import comfy.weight_cast_schedule as schedule

    items = [
        schedule._ScheduleItem(None, "resolve", 2, object(), True),
        schedule._ScheduleItem(None, "resolve", 2, object(), False),
        schedule._ScheduleItem(None, "resolve", 2, object(), True),
    ]

    decisions = schedule._solve_memory_schedule(
        items,
        budget_bytes=4,
        lookahead=len(items),
        max_weight_bytes=None,
    )

    assert [decision.scheduled for decision in decisions] == [True, False, True]
    assert decisions[2].dependencies == (0, 1)


def _assert_timed_schedule_feasible(items, result, *, budget_bytes, copy_engines):
    horizon = result.makespan
    assert all(start >= 0 for start in result.load_starts)
    assert all(start >= 0 for start in result.compute_starts)
    for index, item in enumerate(items):
        assert result.load_ends[index] == result.load_starts[index] + item.copy_cost
        assert result.compute_ends[index] == result.compute_starts[index] + item.compute_cost
        assert result.compute_starts[index] >= result.load_ends[index]
    for index, item in enumerate(items[:-1]):
        assert result.compute_starts[index + 1] >= result.compute_ends[index]

    for tick in range(horizon + 1):
        active_copies = sum(start <= tick < end for start, end in zip(result.load_starts, result.load_ends, strict=True))
        assert active_copies <= copy_engines
        resident_bytes = sum(
            item.size
            for item, load_start, compute_end in zip(items, result.load_starts, result.compute_ends, strict=True)
            if load_start <= tick < compute_end
        )
        assert resident_bytes <= budget_bytes


def _fifo_timed_schedule_makespan(items, *, budget_bytes, copy_engines):
    time = 0
    next_compute = 0
    resident: set[int] = set()
    ready: set[int] = set()
    copy_jobs: list[tuple[int, int]] = []
    compute_job: tuple[int, int] | None = None

    while next_compute < len(items):
        changed = True
        while changed:
            changed = False
            completed_copies = [(index, end) for index, end in copy_jobs if end <= time]
            if completed_copies:
                ready.update(index for index, _ in completed_copies)
                copy_jobs = [(index, end) for index, end in copy_jobs if end > time]
                changed = True
            if compute_job is not None and compute_job[1] <= time:
                index, _ = compute_job
                resident.remove(index)
                ready.discard(index)
                next_compute = index + 1
                compute_job = None
                changed = True
        if next_compute >= len(items):
            break

        if compute_job is None and next_compute in ready:
            compute_job = (next_compute, time + items[next_compute].compute_cost)
            continue

        used_bytes = sum(items[index].size for index in resident)
        started_copy = False
        while len(copy_jobs) < copy_engines:
            next_load = None
            for candidate in range(next_compute, len(items)):
                item = items[candidate]
                if candidate in resident:
                    continue
                if not item.prefetchable and candidate != next_compute:
                    break
                if used_bytes + item.size <= budget_bytes:
                    next_load = candidate
                    break
            if next_load is None:
                break
            item = items[next_load]
            resident.add(next_load)
            copy_jobs.append((next_load, time + item.copy_cost))
            used_bytes += item.size
            started_copy = True
            if not item.prefetchable:
                break
        if started_copy:
            continue

        next_times = [end for _, end in copy_jobs]
        if compute_job is not None:
            next_times.append(compute_job[1])
        if not next_times:
            return None
        time = min(candidate for candidate in next_times if candidate > time)

    return max([time] + [end for _, end in copy_jobs] + ([compute_job[1]] if compute_job else []))


def test_timed_memory_schedule_uses_milp_for_non_fifo_copy_order():
    import comfy.weight_cast_schedule as schedule

    items = [
        schedule._TimedScheduleItem(size=6, copy_cost=5, compute_cost=2),
        schedule._TimedScheduleItem(size=2, copy_cost=1, compute_cost=5),
        schedule._TimedScheduleItem(size=4, copy_cost=4, compute_cost=1),
        schedule._TimedScheduleItem(size=3, copy_cost=2, compute_cost=4),
        schedule._TimedScheduleItem(size=5, copy_cost=6, compute_cost=2),
        schedule._TimedScheduleItem(size=1, copy_cost=1, compute_cost=3),
        schedule._TimedScheduleItem(size=2, copy_cost=3, compute_cost=2),
    ]

    result = schedule._solve_timed_memory_schedule(items, budget_bytes=10, copy_engines=2)

    _assert_timed_schedule_feasible(items, result, budget_bytes=10, copy_engines=2)
    assert result.makespan == 26
    assert result.load_order == (0, 1, 2, 3, 4, 6, 5)
    assert result.load_starts == (0, 6, 8, 11, 13, 19, 14)
    assert result.compute_starts == (5, 7, 12, 14, 19, 21, 24)


def test_timed_memory_schedule_respects_non_prefetchable_items():
    import comfy.weight_cast_schedule as schedule

    items = [
        schedule._TimedScheduleItem(size=3, copy_cost=2, compute_cost=3),
        schedule._TimedScheduleItem(size=2, copy_cost=4, compute_cost=2, prefetchable=False),
        schedule._TimedScheduleItem(size=3, copy_cost=2, compute_cost=3),
    ]

    result = schedule._solve_timed_memory_schedule(items, budget_bytes=6, copy_engines=2)

    _assert_timed_schedule_feasible(items, result, budget_bytes=6, copy_engines=2)
    assert result.compute_starts[1] == result.load_ends[1]


def test_timed_memory_schedule_optimizes_larger_mixed_trace_beyond_fifo():
    import comfy.weight_cast_schedule as schedule

    items = [
        schedule._TimedScheduleItem(size=6, copy_cost=5, compute_cost=2),
        schedule._TimedScheduleItem(size=2, copy_cost=1, compute_cost=5),
        schedule._TimedScheduleItem(size=4, copy_cost=4, compute_cost=1),
        schedule._TimedScheduleItem(size=3, copy_cost=2, compute_cost=4),
        schedule._TimedScheduleItem(size=5, copy_cost=6, compute_cost=2),
        schedule._TimedScheduleItem(size=1, copy_cost=1, compute_cost=3),
        schedule._TimedScheduleItem(size=2, copy_cost=3, compute_cost=2),
        schedule._TimedScheduleItem(size=4, copy_cost=2, compute_cost=4, prefetchable=False),
        schedule._TimedScheduleItem(size=3, copy_cost=5, compute_cost=1),
        schedule._TimedScheduleItem(size=2, copy_cost=1, compute_cost=3),
        schedule._TimedScheduleItem(size=5, copy_cost=4, compute_cost=2),
        schedule._TimedScheduleItem(size=1, copy_cost=2, compute_cost=2),
    ]
    budget_bytes = 10
    copy_engines = 2

    result = schedule._solve_timed_memory_schedule(items, budget_bytes=budget_bytes, copy_engines=copy_engines)
    fifo_makespan = _fifo_timed_schedule_makespan(items, budget_bytes=budget_bytes, copy_engines=copy_engines)

    _assert_timed_schedule_feasible(items, result, budget_bytes=budget_bytes, copy_engines=copy_engines)
    assert fifo_makespan is not None
    assert result.makespan < fifo_makespan
    assert any(later < earlier for earlier, later in zip(result.load_order, result.load_order[1:], strict=False))
    assert result.compute_starts[7] == result.load_ends[7]


def test_weight_prefetch_backend_auto_sizes_without_env(monkeypatch):
    from comfy import ops
    import comfy.weight_cast_schedule as schedule
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module

    monkeypatch.delenv("COMFY_WEIGHT_PREFETCH_BUDGET_MB", raising=False)
    monkeypatch.delenv("COMFY_WEIGHT_PREFETCH_LOOKAHEAD", raising=False)
    monkeypatch.setattr(schedule, "_first_cuda_device", lambda example_inputs: torch.device("cuda:0"))
    monkeypatch.setattr(schedule.model_management, "get_free_memory", lambda device: 2 * 1024 * 1024 * 1024)
    monkeypatch.setattr(schedule.model_management, "extra_reserved_memory", lambda: 128 * 1024 * 1024)

    layers = [ops.manual_cast.Linear(2, 2) for _ in range(3)]
    args = [
        (
            module_weight_shape(layer),
            module_bias_shape(layer),
            register_module(layer),
            invocation,
        )
        for invocation, layer in enumerate(layers, start=1)
    ]
    graphs = []

    def inner_backend(gm, example_inputs, **kwargs):
        graphs.append(gm)
        return gm.forward

    backend = schedule.wrap_backend_with_weight_prefetch_scheduler(inner_backend)

    def fn(x):
        total = x
        for layer_args in args:
            weight, bias = torch.ops.comfy_weight.resolve_weight_bias(x, *layer_args, 0, 0, 0, False, 0, -1)
            total = total + torch.nn.functional.linear(x, weight, bias)
            torch.ops.comfy_weight.release_(total, layer_args[2], layer_args[3])
        return total

    compiled = torch.compile(fn, backend=backend)
    compiled(torch.randn(1, 2))

    graph_text = graphs[0].code
    assert "comfy_weight.prefetch_weight_bias_after" in graph_text
    assert "comfy_weight.resolve_prefetched_weight_bias" in graph_text


def test_weight_prefetch_backend_auto_solves_large_weights_from_memory(monkeypatch):
    from comfy import ops
    import comfy.weight_cast_schedule as schedule
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module

    one_gib = 1024 * 1024 * 1024
    monkeypatch.delenv("COMFY_WEIGHT_PREFETCH_BUDGET_MB", raising=False)
    monkeypatch.delenv("COMFY_WEIGHT_PREFETCH_LOOKAHEAD", raising=False)
    monkeypatch.delenv("COMFY_WEIGHT_PREFETCH_MAX_WEIGHT_MB", raising=False)
    monkeypatch.setattr(schedule, "_first_cuda_device", lambda example_inputs: torch.device("cuda:0"))
    monkeypatch.setattr(schedule.model_management, "get_free_memory", lambda device: 4 * one_gib)
    monkeypatch.setattr(schedule.model_management, "extra_reserved_memory", lambda: 0)

    layers = [ops.manual_cast.Linear(2, 2) for _ in range(6)]
    args = [
        (
            module_weight_shape(layer),
            module_bias_shape(layer),
            register_module(layer),
            invocation,
        )
        for invocation, layer in enumerate(layers, start=1)
    ]
    graphs = []

    monkeypatch.setattr(schedule, "_resolve_nbytes", lambda node: one_gib)

    def inner_backend(gm, example_inputs, **kwargs):
        graphs.append(gm)
        return gm.forward

    backend = schedule.wrap_backend_with_weight_prefetch_scheduler(inner_backend)

    def fn(x):
        total = x
        for layer_args in args:
            weight, bias = torch.ops.comfy_weight.resolve_weight_bias(x, *layer_args, 0, 0, 0, False, 0, -1)
            total = total + torch.nn.functional.linear(x, weight, bias)
            torch.ops.comfy_weight.release_(total, layer_args[2], layer_args[3])
        return total

    compiled = torch.compile(fn, backend=backend)
    compiled(torch.randn(1, 2))

    graph = graphs[0].graph
    prefetches = _node_target_contains(graph, "prefetch_weight_bias_after")
    releases = _node_target_contains(graph, "release_memory_")

    assert len(prefetches) == 6
    assert len(releases) == 6
    assert not _node_depends_on(prefetches[0], releases[0])
    assert not _node_depends_on(prefetches[1], releases[0])
    assert _node_depends_on(prefetches[2], releases[0])
    assert not _node_depends_on(prefetches[2], releases[1])
    assert _node_depends_on(prefetches[3], releases[1])
    assert not _node_depends_on(prefetches[3], releases[0])


def test_weight_prefetch_backend_auto_keeps_compile_headroom(monkeypatch):
    import comfy.weight_cast_schedule as schedule

    one_gib = 1024 * 1024 * 1024
    monkeypatch.setattr(schedule, "_first_cuda_device", lambda example_inputs: torch.device("cuda:0"))
    monkeypatch.setattr(schedule.model_management, "get_free_memory", lambda device: 6 * one_gib)
    monkeypatch.setattr(schedule.model_management, "extra_reserved_memory", lambda: 512 * 1024 * 1024)

    budget = schedule._auto_prefetch_budget_bytes(
        [torch.randn(1)],
        [256 * 1024 * 1024, one_gib],
        required_bytes=6 * one_gib,
    )

    assert budget == 2 * one_gib


def test_compiled_manual_cast_uses_graph_visible_op_even_when_resident(monkeypatch):
    from comfy import ops
    from comfy import weight_cast

    layer = ops.manual_cast.Linear(2, 2)
    x = torch.randn(1, 2)

    assert layer.weight.device == x.device
    assert layer.weight.dtype == x.dtype

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(weight_cast, "_is_device_cpu", lambda device: False)

    runtime = weight_cast.get_weight_cast_runtime(layer, x)

    assert runtime.name == weight_cast.BACKEND_GRAPH_VISIBLE


def test_compiled_cast_capable_module_uses_graph_visible_even_when_flag_false(monkeypatch):
    from comfy import ops
    from comfy import weight_cast

    layer = ops.disable_weight_init.Linear(2, 2)
    layer.comfy_cast_weights = False
    x = torch.randn(1, 2)

    monkeypatch.setattr(weight_cast, "is_torch_compiling", lambda: True)
    monkeypatch.setattr(weight_cast, "_is_device_cpu", lambda device: False)

    runtime = weight_cast.get_weight_cast_runtime(layer, x)

    assert runtime.name == weight_cast.BACKEND_GRAPH_VISIBLE


def test_compiled_runtime_selection_avoids_module_attribute_guards(monkeypatch):
    from comfy import ops
    from comfy import weight_cast

    layer = ops.disable_weight_init.Linear(2, 2)
    x = torch.randn(1, 2)

    monkeypatch.setattr(weight_cast, "is_torch_compiling", lambda: True)
    monkeypatch.setattr(weight_cast, "_is_device_cpu", lambda device: False)
    monkeypatch.setattr(
        weight_cast,
        "_module_needs_graph_visible_weight_cast",
        lambda module, input: (_ for _ in ()).throw(AssertionError("module attrs inspected")),
    )

    runtime = weight_cast.get_weight_cast_runtime(layer, x)

    assert runtime.name == weight_cast.BACKEND_GRAPH_VISIBLE


def test_mixed_precision_linear_compile_path_avoids_parameter_inspection(monkeypatch):
    from comfy import model_management
    from comfy import ops
    from comfy import weight_cast

    linear = ops.mixed_precision_ops({}, torch.bfloat16).Linear(2, 2)
    calls = []

    def fake_forward_comfy_cast_weights(input, compute_dtype=None, want_requant=False):
        calls.append((compute_dtype, want_requant))
        return input

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(weight_cast, "graph_visible_backend_unavailable_reason", lambda: None)
    monkeypatch.setattr(model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(linear, "forward_comfy_cast_weights", fake_forward_comfy_cast_weights)

    x = torch.randn(1, 2)
    out = linear(x)

    assert out is x
    assert calls == [(x.dtype, False)]


def test_auto_fp8_materialization_happens_before_device_cast(monkeypatch):
    from comfy import ops
    from comfy import model_management
    from comfy.quant_ops import QuantizedTensor

    layer = ops.manual_cast.Linear(2, 2, dtype=torch.bfloat16)
    qdata = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
    params = ops.TensorCoreFP8Layout.Params(
        scale=torch.ones((), dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=(2, 2),
    )
    layer.weight = torch.nn.Parameter(
        QuantizedTensor(qdata, "TensorCoreFP8Layout", params),
        requires_grad=False,
    )
    layer.bias = None

    calls = []
    original_cast_to = model_management.cast_to

    def capture_cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
        calls.append(weight)
        return original_cast_to(weight, dtype=dtype, device=None, non_blocking=non_blocking, copy=copy, stream=None, r=None)

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "auto")
    monkeypatch.setattr(model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(model_management, "cast_to", capture_cast_to)
    monkeypatch.setattr(model_management, "device_supports_non_blocking", lambda device: False)
    monkeypatch.setattr(model_management, "sync_stream", lambda device, stream: None)

    weight, bias = ops.cast_bias_weight(
        layer,
        torch.randn(1, 2),
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        bias_dtype=torch.bfloat16,
        offloadable=False,
        compute_dtype=torch.bfloat16,
        want_requant=False,
    )

    assert bias is None
    assert weight.dtype == torch.bfloat16
    assert calls
    assert not isinstance(calls[0], QuantizedTensor)
    assert calls[0].device.type == "cpu"
    assert calls[0].dtype == torch.bfloat16


def test_cpu_fp8_materialization_reuses_buffer():
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor

    if not ops.mixed_precision_quantization_available():
        return

    ops._CPU_MATERIALIZATION_BUFFERS.clear()
    source = torch.randn(8, 8, dtype=torch.bfloat16)
    quantized = QuantizedTensor.from_float(source, "TensorCoreFP8E4M3Layout", scale="recalculate")

    first = ops._materialize_quantized_tensor_on_cpu(quantized, torch.bfloat16)
    first_ptr = first.data_ptr()
    second = ops._materialize_quantized_tensor_on_cpu(quantized, torch.bfloat16)

    assert first.device.type == "cpu"
    assert first.dtype is torch.bfloat16
    assert second.data_ptr() == first_ptr
    assert len(ops._CPU_MATERIALIZATION_BUFFERS) == 1


def test_lowvram_patch_materializes_quantized_weight_on_cpu_before_transfer():
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor

    if not ops.mixed_precision_quantization_available():
        return

    source = torch.randn(8, 8, dtype=torch.bfloat16)
    quantized = QuantizedTensor.from_float(source, "TensorCoreFP8E4M3Layout", scale="recalculate")
    calls = []

    class Module:
        def __init__(self):
            self.layout_type = "TensorCoreFP8E4M3Layout"
            self.seed_key = "test.weight"
            self.weight = torch.nn.Parameter(quantized, requires_grad=False)
            self.weight_lowvram_function = self.patch
            self._v_signature = object()

        def patch(self, weight):
            calls.append(weight)
            return weight + 1

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            out = QuantizedTensor.from_float(weight, self.layout_type, scale="recalculate", stochastic_rounding=seed)
            if return_weight:
                return out
            self.weight = torch.nn.Parameter(out, requires_grad=False)

    module = Module()
    old_weight = module.weight
    old_storage_numel = old_weight._qdata.numel()
    patched, applied = ops._apply_lowvram_patch_on_cpu(module, "weight", module.weight, torch.bfloat16)

    assert applied is True
    assert calls
    assert calls[0].device.type == "cpu"
    assert calls[0].dtype is torch.bfloat16
    assert patched.device.type == "cpu"
    assert torch.allclose(patched, calls[0] + 1)
    assert module.weight is not patched
    assert isinstance(module.weight, QuantizedTensor)
    assert module.weight._qdata.numel() == old_storage_numel
    assert module.weight_lowvram_function is None
    assert old_weight._qdata.numel() == 0
    assert module._v_signature is None


def test_lowvram_patch_requantizes_to_original_layout():
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor, _CK_MXFP8_AVAILABLE

    if not ops.mixed_precision_quantization_available() or not _CK_MXFP8_AVAILABLE:
        return

    layout_type = "TensorCoreMXFP8Layout"
    source = torch.randn(64, 64, dtype=torch.bfloat16)
    quantized = QuantizedTensor.from_float(source, layout_type)

    class Module:
        def __init__(self):
            self.layout_type = layout_type
            self.seed_key = "test.mxfp8.weight"
            self.weight = torch.nn.Parameter(quantized, requires_grad=False)
            self.weight_lowvram_function = lambda weight: weight + 1
            self._v_signature = object()

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            out = QuantizedTensor.from_float(weight, self.layout_type, scale="recalculate", stochastic_rounding=seed)
            if return_weight:
                return out
            self.weight = torch.nn.Parameter(out, requires_grad=False)

    module = Module()
    patched, applied = ops._apply_lowvram_patch_on_cpu(module, "weight", module.weight, torch.bfloat16)

    assert applied is True
    assert patched.dtype is torch.bfloat16
    assert isinstance(module.weight, QuantizedTensor)
    assert module.weight._layout_cls == layout_type
    assert module.weight._qdata.dtype is torch.float8_e4m3fn
    assert module.weight_lowvram_function is None


def test_lowvram_patch_on_demand_keeps_quantized_source(monkeypatch):
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor

    if not ops.mixed_precision_quantization_available():
        return

    monkeypatch.setattr(ops, "lowvram_lora_materialization_policy", lambda: "on-demand")

    source = torch.randn(8, 8, dtype=torch.bfloat16)
    quantized = QuantizedTensor.from_float(source, "TensorCoreFP8E4M3Layout", scale="recalculate")
    calls = []

    class Module:
        def __init__(self):
            self.layout_type = "TensorCoreFP8E4M3Layout"
            self.seed_key = "test.on_demand.weight"
            self.weight = torch.nn.Parameter(quantized, requires_grad=False)
            self.patch_fn = self.patch
            self.weight_lowvram_function = self.patch_fn
            self._v_signature = object()

        def patch(self, weight):
            calls.append(weight)
            return weight + 1

    module = Module()
    old_weight = module.weight
    patched, applied = ops._apply_lowvram_patch_on_cpu(module, "weight", module.weight, torch.bfloat16)

    assert applied is True
    assert patched.device.type == "cpu"
    assert torch.allclose(patched, calls[0] + 1)
    assert module.weight is old_weight
    assert module.weight_lowvram_function is module.patch_fn
    assert old_weight._qdata.numel() == quantized._qdata.numel()
    assert module._v_signature is not None


def test_large_lowvram_cpu_patch_trims_allocator(monkeypatch):
    from comfy import ops

    calls = []

    class Module:
        def __init__(self):
            self.weight_lowvram_function = lambda weight: weight

    monkeypatch.setattr(ops, "_CPU_PATCH_TRIM_THRESHOLD", 1)
    monkeypatch.setattr(ops, "_CPU_PATCH_TRIM_BATCH_THRESHOLD", 1)
    monkeypatch.setattr(ops, "_CPU_PATCH_TRIM_PENDING_BYTES", 0)
    monkeypatch.setattr(ops, "_trim_cpu_allocator", lambda: calls.append("trim"))

    _, applied = ops._apply_lowvram_patch_on_cpu(
        Module(),
        "weight",
        torch.ones(2, 2, dtype=torch.bfloat16),
        torch.bfloat16,
    )

    assert applied is True
    assert calls == ["trim"]


def test_cpu_lora_bake_uses_chunked_delta(monkeypatch):
    from comfy.weight_adapter import lora as lora_adapter

    weight = torch.zeros(6, 4, dtype=torch.bfloat16)
    up = torch.arange(18, dtype=torch.bfloat16).reshape(6, 3)
    down = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    adapter = lora_adapter.LoRAAdapter(set(), (up, down, None, None, None, None))

    mm_shapes = []
    original_mm = torch.mm

    def capture_mm(a, b):
        mm_shapes.append(tuple(a.shape))
        return original_mm(a, b)

    monkeypatch.setattr(lora_adapter, "CPU_LORA_CHUNK_BYTES", weight[0].numel() * weight.element_size() * 2)
    monkeypatch.setattr(torch, "mm", capture_mm)

    out = adapter.calculate_weight(
        weight.clone(),
        "layer.weight",
        strength=1.0,
        strength_model=1.0,
        offset=None,
        function=lambda x: x,
        intermediate_dtype=torch.bfloat16,
    )

    expected = original_mm(up, down).to(torch.bfloat16)
    assert torch.allclose(out, expected)
    assert mm_shapes == [(2, 3), (2, 3), (2, 3)]


def test_cpu_lora_bake_bounds_and_restores_torch_threads(monkeypatch):
    from comfy.weight_adapter import lora as lora_adapter

    events = []

    monkeypatch.setattr(lora_adapter, "CPU_LORA_MAX_THREADS", 4)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: events.append(value))

    with lora_adapter._bounded_threaded_cpu_lora():
        events.append("body")

    assert events == [4, "body", 8]


def test_weight_prefetch_scheduler_respects_byte_budget():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    first = ops.manual_cast.Linear(2, 2)
    second = ops.manual_cast.Linear(256, 256)
    first_args = (
        module_weight_shape(first),
        module_bias_shape(first),
        register_module(first),
        1,
    )
    second_args = (
        module_weight_shape(second),
        module_bias_shape(second),
        register_module(second),
        2,
    )
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=1, budget_bytes=64))
        return graphs[-1].forward

    def fn(x_small, x_large):
        w1, b1 = torch.ops.comfy_weight.resolve_weight_bias(x_small, *first_args, 0, 0, 0, False, 0, -1)
        y1 = torch.nn.functional.linear(x_small, w1, b1)
        torch.ops.comfy_weight.release_(y1, first_args[2], first_args[3])
        w2, b2 = torch.ops.comfy_weight.resolve_weight_bias(x_large, *second_args, 0, 0, 0, False, 0, -1)
        y2 = torch.nn.functional.linear(x_large, w2, b2)
        torch.ops.comfy_weight.release_(y2, second_args[2], second_args[3])
        return y1, y2

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2), torch.randn(1, 256))

    nodes = list(graphs[0].graph.nodes)
    prefetches = [i for i, node in enumerate(nodes) if "prefetch_weight_bias" in str(node.target)]
    assert len(prefetches) == 1
    second_resolve = [i for i, node in enumerate(nodes) if "resolve_weight_bias" in str(node.target)][0]
    first_release_memory = next(i for i, node in enumerate(nodes) if "release_memory" in str(node.target))

    assert first_release_memory < second_resolve


def test_weight_prefetch_scheduler_respects_live_lookahead_window():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    layers = [ops.manual_cast.Linear(2, 2) for _ in range(4)]
    args = [
        (
            module_weight_shape(layer),
            module_bias_shape(layer),
            register_module(layer),
            invocation,
        )
        for invocation, layer in enumerate(layers, start=1)
    ]
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=2, budget_bytes=1024))
        return graphs[-1].forward

    def fn(x):
        total = x
        for layer_args in args:
            weight, bias = torch.ops.comfy_weight.resolve_weight_bias(x, *layer_args, 0, 0, 0, False, 0, -1)
            total = total + torch.nn.functional.linear(x, weight, bias)
            torch.ops.comfy_weight.release_(total, layer_args[2], layer_args[3])
        return total

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    nodes = list(graphs[0].graph.nodes)
    prefetches = [i for i, node in enumerate(nodes) if "prefetch_weight_bias_after" in str(node.target)]
    first_release_memory = next(i for i, node in enumerate(nodes) if "release_memory" in str(node.target))

    assert len(prefetches) == 4
    assert prefetches[0] < first_release_memory
    assert prefetches[1] < first_release_memory
    assert first_release_memory < prefetches[2]
    assert first_release_memory < prefetches[3]


def test_weight_prefetch_scheduler_models_reusable_memory_slots():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    layers = [ops.manual_cast.Linear(2, 2) for _ in range(4)]
    args = [
        (
            module_weight_shape(layer),
            module_bias_shape(layer),
            register_module(layer),
            invocation,
        )
        for invocation, layer in enumerate(layers, start=1)
    ]
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=2, budget_bytes=1024))
        return graphs[-1].forward

    def fn(x):
        total = x
        for layer_args in args:
            weight, bias = torch.ops.comfy_weight.resolve_weight_bias(x, *layer_args, 0, 0, 0, False, 0, -1)
            total = total + torch.nn.functional.linear(x, weight, bias)
            torch.ops.comfy_weight.release_(total, layer_args[2], layer_args[3])
        return total

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    graph = graphs[0].graph
    prefetches = _node_target_contains(graph, "prefetch_weight_bias_after")
    releases = _node_target_contains(graph, "release_memory_")

    assert len(prefetches) == 4
    assert len(releases) == 4
    assert not _node_depends_on(prefetches[0], releases[0])
    assert not _node_depends_on(prefetches[1], releases[0])
    assert _node_depends_on(prefetches[2], releases[0])
    assert not _node_depends_on(prefetches[2], releases[1])
    assert _node_depends_on(prefetches[3], releases[1])
    assert not _node_depends_on(prefetches[3], releases[0])


def test_weight_prefetch_scheduler_budgets_from_shapes_when_module_lookup_misses(monkeypatch):
    from comfy import ops
    import comfy.weight_cast_schedule as schedule
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module

    first = ops.manual_cast.Linear(2, 2)
    second = ops.manual_cast.Linear(256, 256)
    first_args = (
        module_weight_shape(first),
        module_bias_shape(first),
        register_module(first),
        1,
    )
    second_args = (
        module_weight_shape(second),
        module_bias_shape(second),
        register_module(second),
        2,
    )
    graphs = []

    monkeypatch.setattr(schedule, "get_registered_module", lambda value: (_ for _ in ()).throw(KeyError(value)))

    def capture_backend(gm, example_inputs):
        graphs.append(schedule.schedule_weight_prefetches(gm, lookahead=1, budget_bytes=128))
        return graphs[-1].forward

    def fn(x_small, x_large):
        w1, b1 = torch.ops.comfy_weight.resolve_weight_bias(x_small, *first_args, 1, 1, 1, False, 0, -1)
        y1 = torch.nn.functional.linear(x_small, w1, b1)
        torch.ops.comfy_weight.release_(y1, first_args[2], first_args[3])
        w2, b2 = torch.ops.comfy_weight.resolve_weight_bias(x_large, *second_args, 1, 1, 1, False, 0, -1)
        y2 = torch.nn.functional.linear(x_large, w2, b2)
        torch.ops.comfy_weight.release_(y2, second_args[2], second_args[3])
        return y1, y2

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2), torch.randn(1, 256))

    graph_text = graphs[0].code
    assert graph_text.count("comfy_weight.prefetch_weight_bias_after") == 1
    assert graph_text.count("comfy_weight.resolve_weight_bias") == 1


def test_weight_prefetch_scheduler_uses_materialization_spec_budget():
    from comfy import ops, weight_cast
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    first = ops.manual_cast.Linear(2, 2)
    second = ops.manual_cast.Linear(2, 2)
    weight_cast.set_materialization_param(
        second,
        "weight",
        key="second.weight",
        tensor=second.weight,
        model_dtype=second.weight.dtype,
        vram_bytes=1024,
    )
    first_args = (
        module_weight_shape(first),
        module_bias_shape(first),
        register_module(first),
        1,
    )
    second_args = (
        module_weight_shape(second),
        module_bias_shape(second),
        register_module(second),
        2,
    )
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=1, budget_bytes=64))
        return graphs[-1].forward

    def fn(x):
        w1, b1 = torch.ops.comfy_weight.resolve_weight_bias(x, *first_args, 0, 0, 0, False, 0, -1)
        y1 = torch.nn.functional.linear(x, w1, b1)
        torch.ops.comfy_weight.release_(y1, first_args[2], first_args[3])
        w2, b2 = torch.ops.comfy_weight.resolve_weight_bias(x, *second_args, 0, 0, 0, False, 0, -1)
        y2 = torch.nn.functional.linear(x, w2, b2)
        torch.ops.comfy_weight.release_(y2, second_args[2], second_args[3])
        return y1 + y2

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    nodes = list(graphs[0].graph.nodes)
    prefetches = [i for i, node in enumerate(nodes) if "prefetch_weight_bias" in str(node.target)]
    assert len(prefetches) == 1
    second_resolve = [i for i, node in enumerate(nodes) if "resolve_weight_bias" in str(node.target)][0]
    first_release_memory = next(i for i, node in enumerate(nodes) if "release_memory" in str(node.target))
    assert first_release_memory < second_resolve


def test_weight_prefetch_scheduler_respects_per_weight_prefetch_cap():
    from comfy import ops, weight_cast
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    first = ops.manual_cast.Linear(2, 2)
    second = ops.manual_cast.Linear(2, 2)
    weight_cast.set_materialization_param(
        second,
        "weight",
        key="second.weight",
        tensor=second.weight,
        model_dtype=second.weight.dtype,
        vram_bytes=1024,
    )
    first_args = (
        module_weight_shape(first),
        module_bias_shape(first),
        register_module(first),
        1,
    )
    second_args = (
        module_weight_shape(second),
        module_bias_shape(second),
        register_module(second),
        2,
    )
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=2, budget_bytes=2048, max_weight_bytes=64))
        return graphs[-1].forward

    def fn(x):
        w1, b1 = torch.ops.comfy_weight.resolve_weight_bias(x, *first_args, 0, 0, 0, False, 0, -1)
        y1 = torch.nn.functional.linear(x, w1, b1)
        torch.ops.comfy_weight.release_(y1, first_args[2], first_args[3])
        w2, b2 = torch.ops.comfy_weight.resolve_weight_bias(x, *second_args, 0, 0, 0, False, 0, -1)
        y2 = torch.nn.functional.linear(x, w2, b2)
        torch.ops.comfy_weight.release_(y2, second_args[2], second_args[3])
        return y1 + y2

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    graph_text = graphs[0].code
    assert graph_text.count("comfy_weight.prefetch_weight_bias_after") == 1
    assert graph_text.count("comfy_weight.resolve_weight_bias") == 1


def test_weight_prefetch_scheduler_keeps_live_patch_function_on_demand_path():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    first = ops.manual_cast.Linear(2, 2)
    second = ops.manual_cast.Linear(2, 2)
    second.weight_function = [lambda weight: weight]
    first_args = (
        module_weight_shape(first),
        module_bias_shape(first),
        register_module(first),
        1,
    )
    second_args = (
        module_weight_shape(second),
        module_bias_shape(second),
        register_module(second),
        2,
    )
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=1, budget_bytes=64))
        return graphs[-1].forward

    def fn(x):
        w1, b1 = torch.ops.comfy_weight.resolve_weight_bias(x, *first_args, 0, 0, 0, False, 0, -1)
        y1 = torch.nn.functional.linear(x, w1, b1)
        torch.ops.comfy_weight.release_(y1, first_args[2], first_args[3])
        w2, b2 = torch.ops.comfy_weight.resolve_weight_bias(x, *second_args, 0, 0, 0, False, 0, -1)
        y2 = torch.nn.functional.linear(x, w2, b2)
        torch.ops.comfy_weight.release_(y2, second_args[2], second_args[3])
        return y1 + y2

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    graph_text = graphs[0].code
    assert graph_text.count("comfy_weight.prefetch_weight_bias_after") == 1
    assert graph_text.count("comfy_weight.resolve_prefetched_weight_bias") == 1
    assert graph_text.count("comfy_weight.resolve_weight_bias") == 1
    assert graph_text.count("comfy_weight.release_memory_") == 2


def test_weight_prefetch_scheduler_keeps_existing_window_across_demand_resolve():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    first = ops.manual_cast.Linear(2, 2)
    second = ops.manual_cast.Linear(2, 2)
    third = ops.manual_cast.Linear(2, 2)
    second.weight_function = [lambda weight: weight]
    first_args = (
        module_weight_shape(first),
        module_bias_shape(first),
        register_module(first),
        1,
    )
    second_args = (
        module_weight_shape(second),
        module_bias_shape(second),
        register_module(second),
        2,
    )
    third_args = (
        module_weight_shape(third),
        module_bias_shape(third),
        register_module(third),
        3,
    )
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=4, budget_bytes=1024))
        return graphs[-1].forward

    def fn(x):
        w1, b1 = torch.ops.comfy_weight.resolve_weight_bias(x, *first_args, 0, 0, 0, False, 0, -1)
        y1 = torch.nn.functional.linear(x, w1, b1)
        torch.ops.comfy_weight.release_(y1, first_args[2], first_args[3])
        w2, b2 = torch.ops.comfy_weight.resolve_weight_bias(x, *second_args, 0, 0, 0, False, 0, -1)
        y2 = torch.nn.functional.linear(x, w2, b2)
        torch.ops.comfy_weight.release_(y2, second_args[2], second_args[3])
        w3, b3 = torch.ops.comfy_weight.resolve_weight_bias(x, *third_args, 0, 0, 0, False, 0, -1)
        y3 = torch.nn.functional.linear(x, w3, b3)
        torch.ops.comfy_weight.release_(y3, third_args[2], third_args[3])
        return y1 + y2 + y3

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    nodes = list(graphs[0].graph.nodes)
    prefetches = [i for i, node in enumerate(nodes) if "prefetch_weight_bias_after" in str(node.target)]
    release_memory = [i for i, node in enumerate(nodes) if "release_memory" in str(node.target)]

    assert len(prefetches) == 2
    assert prefetches[0] < release_memory[0]
    assert release_memory[1] < prefetches[1]


def test_dynamic_vbar_prefetch_uses_cast_buffer_when_aimdo_has_no_room(monkeypatch):
    from comfy import model_management, ops

    layer = ops.manual_cast.Linear(2, 2)
    layer._v = (object(), 0, 4096)
    stream = object()
    calls = []

    monkeypatch.setattr(model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(model_management, "device_supports_non_blocking", lambda device: True)
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)

    def fake_cast_modules_with_vbar(modules, dtype, device, bias_dtype, non_blocking, **kwargs):
        calls.append((modules, dtype, device, bias_dtype, non_blocking, kwargs))
        modules[0]._prefetch = {"signature": None, "resident": False}
        return stream

    monkeypatch.setattr(ops, "cast_modules_with_vbar", fake_cast_modules_with_vbar)

    state = ops._legacy_weight_cast_prefetch(
        layer,
        torch.device("cuda:0"),
        torch.float16,
        torch.float16,
        torch.float16,
        False,
    )

    assert state[0] is stream
    assert calls[0][5]["dedicated_buffer"] is True
    assert calls[0][5]["prefetch_hint"] is False


def test_dynamic_vbar_resolve_uses_demand_path_after_deferred_prefetch(monkeypatch):
    from comfy import ops

    layer = ops.manual_cast.Linear(2, 2)
    x = torch.randn(1, 2)
    calls = []

    def fake_cast_bias_weight(module, input, **kwargs):
        calls.append((module, input, kwargs))
        return "weight", "bias", "token"

    monkeypatch.setattr(ops, "cast_bias_weight", fake_cast_bias_weight)

    result = ops._legacy_weight_cast_resolve(
        layer,
        x,
        torch.float16,
        torch.float16,
        torch.float16,
        False,
        prefetch_state=None,
    )

    assert result == ("weight", "bias", "token")
    assert calls[0][0] is layer
    assert calls[0][1] is x
    assert calls[0][2]["offloadable"] is True


def test_dynamic_vbar_prefetch_fallback_release_tracks_materialized_tensors(monkeypatch):
    from comfy import ops

    layer = ops.manual_cast.Linear(2, 2)
    layer._prefetch = {"signature": None, "resident": False}
    x = torch.randn(1, 2)
    weight = torch.randn_like(layer.weight)
    bias = torch.randn_like(layer.bias)
    stream = object()

    monkeypatch.setattr(ops, "resolve_cast_module_with_vbar", lambda *args, **kwargs: (weight, bias))
    monkeypatch.setattr(ops.model_management, "sync_stream", lambda device, stream: None)

    resolved_weight, resolved_bias, release_state = ops._legacy_weight_cast_resolve(
        layer,
        x,
        torch.float16,
        torch.float16,
        torch.float16,
        False,
        prefetch_state=(stream, torch.device("cuda:0"), None),
    )

    assert resolved_weight is weight
    assert resolved_bias is bias
    assert release_state == (stream, weight, bias)
    assert layer._prefetch is None


def test_non_vbar_offload_falls_back_when_shared_cast_buffer_is_unavailable(monkeypatch):
    from comfy import memory_management, model_management, ops

    layer = ops.manual_cast.Linear(2, 2)
    layer._v = None
    stream = object()
    cast_calls = []

    monkeypatch.setattr(model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(model_management, "get_offload_stream", lambda device: stream)
    monkeypatch.setattr(model_management, "get_cast_buffer", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_management, "sync_stream", lambda device, stream: None)
    monkeypatch.setattr(
        memory_management,
        "interpret_gathered_like",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cast buffer is unavailable")),
    )

    def fake_cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
        cast_calls.append((weight, dtype, device, non_blocking, copy, stream, r))
        return torch.empty_like(weight, dtype=dtype)

    monkeypatch.setattr(model_management, "cast_to", fake_cast_to)

    weight, bias, token = ops.cast_bias_weight(
        layer,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
        bias_dtype=torch.float16,
        offloadable=True,
    )

    assert weight.dtype is torch.float16
    assert bias.dtype is torch.float16
    assert token[0] is stream
    assert cast_calls[0][-1] is None
    assert cast_calls[1][-1] is None


def test_vbar_release_defers_unpin_until_cuda_event_completes(monkeypatch):
    from comfy import model_management, ops

    class FakeEvent:
        complete = False

        def record(self, stream):
            self.stream = stream

        def query(self):
            return self.complete

        def synchronize(self):
            self.complete = True

    layer = ops.manual_cast.Linear(2, 2)
    layer._v = ("vbar", 0, 4096)
    unpinned = []
    events = []

    def fake_event_factory():
        event = FakeEvent()
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", fake_event_factory)
    monkeypatch.setattr(model_management, "current_stream", lambda device: "current-stream")
    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, "vbar_unpin", lambda alloc: unpinned.append(alloc))
    ops._DEFERRED_VBAR_UNPINS.clear()

    try:
        ops._legacy_weight_cast_release(layer, torch.empty(1), None, (None, torch.device("cuda:0"), None))

        assert unpinned == []
        assert len(ops._DEFERRED_VBAR_UNPINS) == 1
        assert events[0].stream == "current-stream"

        ops._drain_deferred_vbar_unpins(block=False)
        assert unpinned == []

        events[0].complete = True
        ops._drain_deferred_vbar_unpins(block=False)
        assert unpinned == [layer._v]
        assert ops._DEFERRED_VBAR_UNPINS == []
    finally:
        ops._DEFERRED_VBAR_UNPINS.clear()


def test_weight_prefetch_scheduler_can_cross_exemplar_dependency():
    from comfy import ops
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    first = ops.manual_cast.Linear(2, 2)
    second = ops.manual_cast.Linear(2, 2)
    first_args = (
        module_weight_shape(first),
        module_bias_shape(first),
        register_module(first),
        1,
    )
    second_args = (
        module_weight_shape(second),
        module_bias_shape(second),
        register_module(second),
        2,
    )
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=2, budget_bytes=1024))
        return graphs[-1].forward

    def fn(x):
        w1, b1 = torch.ops.comfy_weight.resolve_weight_bias(x, *first_args, 0, 0, 0, False, 0, -1)
        y1 = torch.nn.functional.linear(x, w1, b1)
        torch.ops.comfy_weight.release_(y1, first_args[2], first_args[3])
        exemplar = y1 + 1
        w2, b2 = torch.ops.comfy_weight.resolve_weight_bias(exemplar, *second_args, 0, 0, 0, False, 0, -1)
        y2 = torch.nn.functional.linear(exemplar, w2, b2)
        torch.ops.comfy_weight.release_(y2, second_args[2], second_args[3])
        return y2

    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2))

    nodes = list(graphs[0].graph.nodes)
    second_prefetch = [i for i, node in enumerate(nodes) if "prefetch_weight_bias" in str(node.target)][1]
    exemplar = next(i for i, node in enumerate(nodes) if node.name == "exemplar")

    assert second_prefetch < exemplar


def test_comfy_weight_custom_ops_track_overlapping_invocations():
    from comfy import ops
    from comfy import weight_cast_ops

    layer = ops.manual_cast.Linear(2, 2)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    key = weight_cast_ops.register_module(layer)
    weight_shape = weight_cast_ops.module_weight_shape(layer)
    bias_shape = weight_cast_ops.module_bias_shape(layer)
    first_invocation = 1
    second_invocation = 2
    released: list[tuple[float, str]] = []

    def fake_resolve(module, exemplar, dtype, bias_dtype, compute_dtype, want_requant):
        value = float(len(released) + len(weight_cast_ops._ACTIVE) + 1)
        weight = torch.full_like(module.weight, value, dtype=exemplar.dtype)
        bias = torch.zeros_like(module.bias, dtype=exemplar.dtype)
        return weight, bias, f"state-{value}"

    def fake_release(module, weight, bias, state):
        released.append((float(weight[0, 0]), state))

    previous_prefetch = weight_cast_ops._PREFETCH
    previous_resolve = weight_cast_ops._RESOLVE
    previous_release = weight_cast_ops._RELEASE
    weight_cast_ops.set_callbacks(fake_resolve, fake_release)
    try:
        x = torch.ones(1, 2)
        torch.ops.comfy_weight.resolve_weight_bias(x, weight_shape, bias_shape, key, first_invocation, 0, 0, 0, False, 0, -1)
        torch.ops.comfy_weight.resolve_weight_bias(x, weight_shape, bias_shape, key, second_invocation, 0, 0, 0, False, 0, -1)

        torch.ops.comfy_weight.release_(x, key, first_invocation)
        torch.ops.comfy_weight.release_(x, key, second_invocation)
    finally:
        weight_cast_ops._ACTIVE.clear()
        weight_cast_ops._PREFETCHED.clear()
        weight_cast_ops.set_callbacks(previous_resolve, previous_release, previous_prefetch)

    assert released == [(1.0, "state-1.0"), (2.0, "state-2.0")]


def test_comfy_weight_prefetch_token_is_consumed_by_prefetched_resolve():
    from comfy import ops
    from comfy import weight_cast_ops

    layer = ops.manual_cast.Linear(2, 2)
    key = weight_cast_ops.register_module(layer)
    weight_shape = weight_cast_ops.module_weight_shape(layer)
    bias_shape = weight_cast_ops.module_bias_shape(layer)
    invocation = 7
    events: list[tuple[str, object]] = []

    def fake_prefetch(module, device, dtype, bias_dtype, compute_dtype, want_requant):
        events.append(("prefetch", (module, device)))
        return "prefetched-state"

    def fake_resolve(module, exemplar, dtype, bias_dtype, compute_dtype, want_requant, prefetch_state=None):
        events.append(("resolve", prefetch_state))
        return torch.ones_like(module.weight), torch.zeros_like(module.bias), "active-state"

    def fake_release(module, weight, bias, state):
        events.append(("release", state))

    previous_prefetch = weight_cast_ops._PREFETCH
    previous_resolve = weight_cast_ops._RESOLVE
    previous_release = weight_cast_ops._RELEASE
    weight_cast_ops.set_callbacks(fake_resolve, fake_release, fake_prefetch)
    try:
        x = torch.ones(1, 2)
        token = torch.ops.comfy_weight.prefetch_weight_bias(
            key, invocation, 0, 0, 0, False, 0, -1
        )
        weight, bias = torch.ops.comfy_weight.resolve_prefetched_weight_bias(
            x, weight_shape, bias_shape, token, key, invocation, 0, 0, 0, False, 0, -1
        )
        torch.ops.comfy_weight.release_(torch.nn.functional.linear(x, weight, bias), key, invocation)
    finally:
        weight_cast_ops._ACTIVE.clear()
        weight_cast_ops._PREFETCHED.clear()
        weight_cast_ops.set_callbacks(previous_resolve, previous_release, previous_prefetch)

    assert events == [("prefetch", (layer, torch.device("cpu"))), ("resolve", "prefetched-state"), ("release", "active-state")]


def test_weight_invocation_ids_can_be_reset_between_compiled_calls():
    from comfy import weight_cast_ops

    weight_cast_ops.reset_invocation_ids()
    assert [weight_cast_ops.next_invocation_id(), weight_cast_ops.next_invocation_id()] == [1, 2]

    weight_cast_ops.reset_invocation_ids()

    assert weight_cast_ops.next_invocation_id() == 1


def test_graph_visible_runtime_uses_distinct_invocations_for_repeated_module(monkeypatch):
    from comfy import ops, weight_cast, weight_cast_ops
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    monkeypatch.setattr(weight_cast, "_is_device_cpu", lambda device: False)
    layer = ops.manual_cast.Linear(2, 1, dtype=torch.float16)
    weight_cast_ops.register_module(layer)
    layer._comfy_weight_cast_weight_shape = weight_cast_ops.module_weight_shape(layer)
    layer._comfy_weight_cast_bias_shape = weight_cast_ops.module_bias_shape(layer)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    events: list[tuple[str, object]] = []

    def fake_prefetch(module, device, dtype, bias_dtype, compute_dtype, want_requant):
        state = f"prefetch-{len([event for event in events if event[0] == 'prefetch']) + 1}"
        events.append(("prefetch", state))
        return state

    def fake_resolve(module, exemplar, dtype, bias_dtype, compute_dtype, want_requant, prefetch_state=None):
        events.append(("resolve", prefetch_state))
        value = 1.0 if prefetch_state == "prefetch-1" else 2.0
        if prefetch_state is None:
            value = -100.0
        weight = torch.full_like(module.weight, value, dtype=exemplar.dtype)
        bias = torch.zeros_like(module.bias, dtype=exemplar.dtype)
        return weight, bias, prefetch_state

    def fake_release(module, weight, bias, state):
        events.append(("release", state))

    previous_prefetch = weight_cast_ops._PREFETCH
    previous_resolve = weight_cast_ops._RESOLVE
    previous_release = weight_cast_ops._RELEASE
    weight_cast_ops.set_callbacks(fake_resolve, fake_release, fake_prefetch)
    graphs = []
    try:
        def capture_backend(gm, example_inputs):
            schedule_weight_prefetches(gm, lookahead=2, budget_bytes=1024)
            graphs.append(gm)
            return gm.forward

        def fn(x):
            return layer(x) + layer(x)

        compiled = torch.compile(fn, backend=capture_backend)
        out = compiled(torch.ones(1, 2))
    finally:
        weight_cast_ops._ACTIVE.clear()
        weight_cast_ops._PREFETCHED.clear()
        weight_cast_ops.set_callbacks(previous_resolve, previous_release, previous_prefetch)

    assert out.item() == 6.0
    assert [event[0] for event in events].count("prefetch") == 2
    assert [event[0] for event in events].count("resolve") == 2
    assert [event[0] for event in events].count("release") == 2
    resolved_states = [state for event, state in events if event == "resolve"]
    released_states = [state for event, state in events if event == "release"]
    assert sorted(resolved_states) == ["prefetch-1", "prefetch-2"]
    assert sorted(released_states) == ["prefetch-1", "prefetch-2"]
    assert any("comfy_weight.resolve_prefetched_weight_bias" in graph.code for graph in graphs)
    assert any("torch._C._nn.linear" in graph.code for graph in graphs)


def test_manual_cast_compile_uses_graph_visible_weight_resolution(monkeypatch):
    from comfy import ops, weight_cast

    monkeypatch.setattr(weight_cast, "_is_device_cpu", lambda device: False)
    layer = ops.manual_cast.Linear(3, 2, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]]))
        layer.bias.copy_(torch.tensor([0.25, -0.5]))
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def fn(x):
        return layer(x)

    x = torch.tensor([[1.0, -2.0, 0.5]])
    compiled = torch.compile(fn, backend=capture_backend)

    out = compiled(x)

    assert torch.allclose(out, torch.nn.functional.linear(x, layer.weight.to(x.dtype), layer.bias.to(x.dtype)))
    assert any("comfy_weight.resolve_weight_bias" in graph.code for graph in graphs)
    assert any("comfy_weight.release" in graph.code for graph in graphs)


def test_manual_cast_compile_tracks_replaced_parameters(monkeypatch):
    from comfy import ops, weight_cast

    monkeypatch.setattr(weight_cast, "_is_device_cpu", lambda device: False)
    layer = ops.manual_cast.Linear(3, 2, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]]))
        layer.bias.copy_(torch.tensor([0.25, -0.5]))
    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def fn(x):
        return layer(x)

    x = torch.tensor([[1.0, -2.0, 0.5]])
    compiled = torch.compile(fn, backend=capture_backend)

    first = compiled(x)
    with torch.no_grad():
        layer.weight = torch.nn.Parameter(
            layer.weight.detach().clone() + 1.0, requires_grad=False
        )
        layer.bias = torch.nn.Parameter(
            layer.bias.detach().clone() - 0.25, requires_grad=False
        )
    second = compiled(x)

    assert torch.allclose(first, torch.tensor([[-1.25, 4.0]]))
    assert torch.allclose(
        second,
        torch.nn.functional.linear(x, layer.weight.to(x.dtype), layer.bias.to(x.dtype)),
    )
    assert any("comfy_weight.resolve_weight_bias" in graph.code for graph in graphs)
    assert any("comfy_weight.release" in graph.code for graph in graphs)


def test_model_patcher_dynamic_records_weight_materialization_spec(monkeypatch):
    from comfy import ops, weight_cast
    from comfy.model_patcher import ModelPatcherDynamic

    model = torch.nn.Sequential(ops.manual_cast.Linear(3, 2))
    monkeypatch.setattr(ModelPatcherDynamic, "_vbar_get", lambda self, create=False: None)
    patcher = ModelPatcherDynamic(model, torch.device("cuda:0"), torch.device("cpu"))

    patcher.load(device_to=torch.device("cuda:0"))

    spec = weight_cast.get_materialization_spec(model[0])
    assert spec.weight_key == "0.weight"
    assert spec.bias_key == "0.bias"
    assert spec.weight_shape == tuple(model[0].weight.shape)
    assert spec.bias_shape == tuple(model[0].bias.shape)
    assert spec.weight_vram_bytes > 0
    assert spec.bias_vram_bytes > 0
    assert spec.force_loaded is False


def test_lowvram_materialization_vram_bytes_reserves_patch_scratch():
    from comfy import memory_management
    from comfy.model_patcher import LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR, lowvram_materialization_vram_bytes

    geometry = memory_management.TensorGeometry(shape=(4, 4), dtype=torch.float32)
    final_bytes = memory_management.vram_aligned_size(geometry)

    assert lowvram_materialization_vram_bytes(geometry) == final_bytes
    assert (
        lowvram_materialization_vram_bytes(geometry, function_count=1)
        == final_bytes * (1 + LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR)
    )
    assert (
        lowvram_materialization_vram_bytes(geometry, has_lowvram_patch=True)
        == final_bytes * (1 + LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR)
    )
    assert (
        lowvram_materialization_vram_bytes(geometry, has_lowvram_patch=True, cpu_lowvram_patch=True)
        == final_bytes
    )


def test_materialization_keys_are_stable_across_module_instances():
    from comfy import weight_cast

    first = torch.nn.Linear(3, 2)
    second = torch.nn.Linear(3, 2)

    first_spec = weight_cast.set_materialization_param(
        first,
        "weight",
        key="diffusion_model.block.weight",
        tensor=first.weight,
        model_dtype=torch.float32,
        vram_bytes=first.weight.numel() * first.weight.element_size(),
    )
    second_spec = weight_cast.set_materialization_param(
        second,
        "weight",
        key="diffusion_model.block.weight",
        tensor=second.weight,
        model_dtype=torch.float32,
        vram_bytes=second.weight.numel() * second.weight.element_size(),
    )

    assert first_spec.module_key == second_spec.module_key
    assert first._comfy_weight_cast_key == second._comfy_weight_cast_key


def test_graph_visible_runtime_uses_recorded_materialization_shape(monkeypatch):
    from comfy import ops, weight_cast

    monkeypatch.setattr(weight_cast, "_is_device_cpu", lambda device: False)
    layer = ops.manual_cast.Linear(3, 2)
    weight_cast.set_materialization_param(
        layer,
        "weight",
        key="layer.weight",
        tensor=layer.weight,
        model_dtype=torch.float32,
        vram_bytes=24,
    )
    layer.weight = None

    shape = weight_cast._materialization_shape(layer, "weight")

    assert shape is not None
    assert tuple(shape) == (2, 3)


def test_graph_visible_runtime_uses_cached_module_shape_before_parameter_shape():
    from comfy import ops, weight_cast

    layer = ops.manual_cast.Linear(3, 4)
    layer._comfy_weight_cast_weight_shape = (8, 7)

    assert weight_cast._materialization_shape(layer, "weight") == [8, 7]


def test_cast_to_gathered_can_materialize_target_dtype():
    from comfy import memory_management, model_management

    source = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    target_geometry = [memory_management.TensorGeometry(shape=source.shape, dtype=torch.float32)]
    gathered = torch.empty((memory_management.vram_aligned_size(target_geometry),), dtype=torch.uint8)

    model_management.cast_to_gathered([source], gathered, target_geometries=target_geometry)
    (materialized,) = memory_management.interpret_gathered_like(target_geometry, gathered)

    assert materialized.dtype == torch.float32
    assert torch.allclose(materialized, source.float())


def test_cast_to_gathered_can_materialize_quantized_tensor_as_target_dtype():
    from comfy import memory_management, model_management, ops
    from comfy.quant_ops import QuantizedTensor

    if not ops.mixed_precision_quantization_available():
        return

    source_float = torch.randn(4, 5, dtype=torch.bfloat16)
    source = QuantizedTensor.from_float(source_float, "TensorCoreFP8E4M3Layout", scale="recalculate")
    target_geometry = [model_management.tensor_materialization_geometry(source, dtype=torch.bfloat16)]
    gathered = torch.empty((memory_management.vram_aligned_size(target_geometry),), dtype=torch.uint8)

    model_management.cast_to_gathered([source], gathered, target_geometries=target_geometry)
    (materialized,) = memory_management.interpret_gathered_like(target_geometry, gathered)

    assert not isinstance(materialized, QuantizedTensor)
    assert materialized.dtype == torch.bfloat16
    assert torch.isfinite(materialized).all()


def test_fp8_dequant_custom_op_is_present_in_fx_graph():
    import comfy.quant_ops  # noqa: F401

    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def fn(qdata, scale):
        return torch.ops.comfy_quant.dequantize_per_tensor_fp8(qdata, scale, 2)

    qdata = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    scale = torch.ones((), dtype=torch.float32)
    compiled = torch.compile(fn, backend=capture_backend)

    out = compiled(qdata, scale)

    assert out.dtype is torch.bfloat16
    assert "comfy_quant.dequantize_per_tensor_fp8" in graphs[0].code


def test_fp8_materialization_custom_op_is_present_in_fx_graph(monkeypatch):
    from comfy import quant_ops

    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def fn(qdata, scale):
        return quant_ops.materialize_per_tensor_fp8(qdata, scale, torch.bfloat16)

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "torch")
    qdata = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    scale = torch.ones((), dtype=torch.float32)
    compiled = torch.compile(fn, backend=capture_backend)

    out = compiled(qdata, scale)

    assert out.dtype is torch.bfloat16
    assert "comfy_quant.materialize_per_tensor_fp8" in graphs[0].code


def test_fp8_materialization_scheduler_threads_memory_credits(monkeypatch):
    from comfy import quant_ops
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=1, budget_bytes=40))
        return graphs[-1].forward

    def fn(qdata_1, qdata_2, scale):
        first = quant_ops.materialize_per_tensor_fp8(qdata_1, scale, torch.bfloat16)
        y1 = first.float().sum()
        second = quant_ops.materialize_per_tensor_fp8(qdata_2, scale, torch.bfloat16)
        y2 = second.float().sum()
        return y1 + y2

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "torch")
    qdata_1 = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    qdata_2 = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    scale = torch.ones((), dtype=torch.float32)
    compiled = torch.compile(fn, backend=capture_backend)

    compiled(qdata_1, qdata_2, scale)

    graph_text = graphs[0].code
    assert "comfy_quant.materialize_per_tensor_fp8_after" in graph_text
    assert "comfy_quant.release_materialization_" in graph_text


def test_fp8_materialization_after_uses_memory_token_device_in_fake_mode():
    import comfy.quant_ops  # noqa: F401

    with torch._subclasses.fake_tensor.FakeTensorMode():
        token = torch.empty((), device="cuda")
        qdata = torch.empty(4, 4, dtype=torch.float8_e4m3fn, device="cpu")
        scale = torch.ones((), dtype=torch.float32, device="cpu")
        out = torch.ops.comfy_quant.materialize_per_tensor_fp8_after(token, qdata, scale, 2, 1)

    assert out.device.type == "cuda"
    assert out.dtype is torch.bfloat16


def test_fp8_materialization_after_uses_cpu_materialization(monkeypatch):
    from comfy import quant_ops

    if quant_ops.ck is not None:
        monkeypatch.setattr(
            quant_ops.ck,
            "dequantize_per_tensor_fp8",
            lambda *args, **kwargs: pytest.fail("materialize_per_tensor_fp8_after should not use CUDA/backend dequantization"),
        )

    token = torch.empty((), dtype=torch.int64)
    source = torch.randn(8, 8, dtype=torch.bfloat16)
    scale = torch.ones((), dtype=torch.float32)
    qdata = source.to(torch.float8_e4m3fn)

    out = torch.ops.comfy_quant.materialize_per_tensor_fp8_after(token, qdata, scale, 2, 2)

    assert out.device.type == "cpu"
    assert out.dtype is torch.bfloat16
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_materialization_after_compiles_cpu_storage_to_cuda_output():
    import comfy.quant_ops  # noqa: F401

    def fn(token, qdata, scale):
        return torch.ops.comfy_quant.materialize_per_tensor_fp8_after(token, qdata, scale, 2, 0)

    qdata = torch.randn(8, 8, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale = torch.ones((), dtype=torch.float32)
    token = torch.empty((), device="cuda", dtype=torch.int64)
    compiled = torch.compile(fn, backend="inductor", mode="max-autotune")

    out = compiled(token, qdata, scale)

    assert out.device.type == "cuda"
    assert out.dtype is torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_fp8_materialization_scheduler_models_reusable_memory_slots(monkeypatch):
    from comfy import quant_ops
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=2, budget_bytes=1024))
        return graphs[-1].forward

    def fn(qdata_1, qdata_2, qdata_3, qdata_4, scale):
        first = quant_ops.materialize_per_tensor_fp8(qdata_1, scale, torch.bfloat16)
        y1 = first.float().sum()
        second = quant_ops.materialize_per_tensor_fp8(qdata_2, scale, torch.bfloat16)
        y2 = second.float().sum()
        third = quant_ops.materialize_per_tensor_fp8(qdata_3, scale, torch.bfloat16)
        y3 = third.float().sum()
        fourth = quant_ops.materialize_per_tensor_fp8(qdata_4, scale, torch.bfloat16)
        y4 = fourth.float().sum()
        return y1 + y2 + y3 + y4

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "torch")
    qdatas = [torch.zeros(4, 4, dtype=torch.float8_e4m3fn) for _ in range(4)]
    scale = torch.ones((), dtype=torch.float32)
    compiled = torch.compile(fn, backend=capture_backend)

    compiled(*qdatas, scale)

    graph = graphs[0].graph
    materializes = _node_target_contains(graph, "materialize_per_tensor_fp8_after")
    releases = _node_target_contains(graph, "release_materialization_")

    assert len(materializes) == 4
    assert len(releases) == 4
    assert not _node_depends_on(materializes[0], releases[0])
    assert not _node_depends_on(materializes[1], releases[0])
    assert _node_depends_on(materializes[2], releases[0])
    assert not _node_depends_on(materializes[2], releases[1])
    assert _node_depends_on(materializes[3], releases[1])
    assert not _node_depends_on(materializes[3], releases[0])


def test_weight_prefetch_scheduler_solves_unified_flux_like_memory_graph(monkeypatch):
    from comfy import ops, quant_ops
    import comfy.weight_cast_schedule as schedule
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module

    one_gib = 1024 * 1024 * 1024
    layers = [ops.manual_cast.Linear(2, 2) for _ in range(3)]
    layer_args = [
        (
            module_weight_shape(layer),
            module_bias_shape(layer),
            register_module(layer),
            invocation,
        )
        for invocation, layer in enumerate(layers, start=1)
    ]
    graphs = []

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "torch")
    monkeypatch.setattr(schedule, "_resolve_nbytes", lambda node: one_gib)
    monkeypatch.setattr(schedule, "_materialization_nbytes", lambda node: one_gib)

    def capture_backend(gm, example_inputs):
        graphs.append(
            schedule.schedule_weight_prefetches(
                gm,
                lookahead=128,
                budget_bytes=2 * one_gib,
                max_weight_bytes=None,
            )
        )
        return graphs[-1].forward

    def fn(x, qdata_1, qdata_2, qdata_3, scale):
        total = x
        for args, qdata in zip(layer_args, (qdata_1, qdata_2, qdata_3), strict=True):
            weight, bias = torch.ops.comfy_weight.resolve_weight_bias(x, *args, 0, 0, 0, False, 0, -1)
            total = total + torch.nn.functional.linear(x, weight, bias)
            torch.ops.comfy_weight.release_(total, args[2], args[3])
            materialized = quant_ops.materialize_per_tensor_fp8(qdata, scale, torch.bfloat16)
            total = total + materialized.float().sum().to(total.dtype)
        return total

    qdatas = [torch.zeros(2, 2, dtype=torch.float8_e4m3fn) for _ in range(3)]
    scale = torch.ones((), dtype=torch.float32)
    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2), *qdatas, scale)

    graph = graphs[0].graph
    memory_ops = [
        node
        for node in graph.nodes
        if "prefetch_weight_bias_after" in str(node.target) or "materialize_per_tensor_fp8_after" in str(node.target)
    ]
    releases = [
        node
        for node in graph.nodes
        if "release_memory_" in str(node.target) or "release_materialization_" in str(node.target)
    ]

    assert len(memory_ops) == 6
    assert len(releases) == 6
    assert not _node_depends_on(memory_ops[0], releases[0])
    assert not _node_depends_on(memory_ops[1], releases[0])
    assert _node_depends_on(memory_ops[2], releases[0])
    assert not _node_depends_on(memory_ops[2], releases[1])
    assert _node_depends_on(memory_ops[3], releases[1])
    assert not _node_depends_on(memory_ops[3], releases[0])


def test_weight_prefetch_scheduler_rewrites_larger_interleaved_fx_graph(monkeypatch):
    from comfy import ops, quant_ops
    import comfy.weight_cast_schedule as schedule
    from comfy.weight_cast_ops import module_bias_shape, module_weight_shape, register_module

    layers = [ops.manual_cast.Linear(2, 2) for _ in range(10)]
    layer_args = [
        (
            module_weight_shape(layer),
            module_bias_shape(layer),
            register_module(layer),
            invocation,
        )
        for invocation, layer in enumerate(layers, start=1)
    ]
    resolve_sizes = [5, 2, 4, 1, 3, 6, 2, 5, 1, 4]
    materialize_sizes = [2, 4, 1, 5, 2, 3, 6, 1, 4, 2]
    graphs = []
    resolve_iter = iter(resolve_sizes)
    materialize_iter = iter(materialize_sizes)

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "torch")
    monkeypatch.setattr(schedule, "_resolve_nbytes", lambda node: next(resolve_iter))
    monkeypatch.setattr(schedule, "_materialization_nbytes", lambda node: next(materialize_iter))

    def capture_backend(gm, example_inputs):
        graphs.append(
            schedule.schedule_weight_prefetches(
                gm,
                lookahead=64,
                budget_bytes=10,
                max_weight_bytes=None,
            )
        )
        return graphs[-1].forward

    def fn(x, scale, *qdatas):
        total = x
        for index, (args, qdata) in enumerate(zip(layer_args, qdatas, strict=True)):
            weight, bias = torch.ops.comfy_weight.resolve_weight_bias(x, *args, 0, 0, 0, False, 0, -1)
            total = total + torch.nn.functional.linear(x, weight, bias)
            torch.ops.comfy_weight.release_(total, args[2], args[3])
            if index % 3 == 1:
                total = total.sin()
            materialized = quant_ops.materialize_per_tensor_fp8(qdata, scale, torch.bfloat16)
            total = total + materialized.float().sum().to(total.dtype)
            if index % 4 == 2:
                total = total + x.cos()
        return total

    qdatas = [torch.zeros(2, 2, dtype=torch.float8_e4m3fn) for _ in layers]
    scale = torch.ones((), dtype=torch.float32)
    compiled = torch.compile(fn, backend=capture_backend)
    compiled(torch.randn(1, 2), scale, *qdatas)

    graph = graphs[0].graph
    graph_text = graphs[0].code
    memory_ops = [
        node
        for node in graph.nodes
        if "prefetch_weight_bias_after" in str(node.target) or "materialize_per_tensor_fp8_after" in str(node.target)
    ]
    releases = [
        node
        for node in graph.nodes
        if "release_memory_" in str(node.target) or "release_materialization_" in str(node.target)
    ]

    assert len(memory_ops) == 20
    assert len(releases) == 20
    assert "resolve_prefetched_weight_bias" in graph_text
    assert "comfy_weight.resolve_weight_bias" not in graph_text
    assert "comfy_quant.materialize_per_tensor_fp8_after" in graph_text
    assert graph_text.count("memory_join") >= 5


def test_fp8_materialization_scheduler_respects_budget(monkeypatch):
    from comfy import quant_ops
    from comfy.weight_cast_schedule import schedule_weight_prefetches

    graphs = []

    def capture_backend(gm, example_inputs):
        graphs.append(schedule_weight_prefetches(gm, lookahead=1, budget_bytes=16))
        return graphs[-1].forward

    def fn(qdata_1, qdata_2, scale):
        first = quant_ops.materialize_per_tensor_fp8(qdata_1, scale, torch.bfloat16)
        y1 = first.float().sum()
        second = quant_ops.materialize_per_tensor_fp8(qdata_2, scale, torch.bfloat16)
        y2 = second.float().sum()
        return y1 + y2

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "torch")
    qdata_1 = torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
    qdata_2 = torch.zeros(64, 64, dtype=torch.float8_e4m3fn)
    scale = torch.ones((), dtype=torch.float32)
    compiled = torch.compile(fn, backend=capture_backend)

    compiled(qdata_1, qdata_2, scale)

    graph_text = graphs[0].code
    assert graph_text.count("comfy_quant.materialize_per_tensor_fp8_after") == 1
    assert graph_text.count("comfy_quant.materialize_per_tensor_fp8") >= 1


def test_fp8_materialization_rejects_unknown_mode(monkeypatch):
    from comfy import quant_ops

    monkeypatch.setenv("COMFYUI_FP8_MATERIALIZATION", "sideways")

    with pytest.raises(ValueError, match="Unsupported FP8 materialization mode"):
        quant_ops.materialize_per_tensor_fp8(
            torch.zeros(1, dtype=torch.float8_e4m3fn),
            torch.ones((), dtype=torch.float32),
            torch.bfloat16,
        )


def test_direct_materialize_prefetch_allows_dtype_changing_weight(monkeypatch):
    from comfy import ops

    class Module:
        pass

    module = Module()
    module._v = object()
    module.weight = torch.empty((2, 2), dtype=torch.float8_e4m3fn)
    module.bias = None

    monkeypatch.setenv("COMFY_DIRECT_MATERIALIZE_PINNING", "1")
    monkeypatch.setattr(ops.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(ops.model_management, "is_device_cpu", lambda device: False)
    calls = []

    def fake_cast_modules_with_vbar(modules, dtype, device, bias_dtype, non_blocking, **kwargs):
        calls.append((modules, dtype, device, bias_dtype, non_blocking, kwargs))
        modules[0]._prefetch = {"signature": None, "resident": False}
        return None

    monkeypatch.setattr(ops, "cast_modules_with_vbar", fake_cast_modules_with_vbar)

    assert ops._legacy_weight_cast_prefetch(
        module,
        torch.device("cuda", 0),
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        False,
    ) is not None
    assert calls[0][1] is torch.bfloat16
    assert calls[0][5]["prefetch_hint"] is False
