from __future__ import annotations

import torch

from comfy.cli_args_types import Configuration
from comfy.execution_context import context_configuration


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


def test_dynamic_quantized_lowvram_lora_patch_is_deferred(monkeypatch):
    from comfy import model_patcher

    class DummyQuantizedTensor:
        pass

    monkeypatch.setattr(model_patcher, "QuantizedTensor", DummyQuantizedTensor)

    assert model_patcher.should_bake_lowvram_patch(object(), DummyQuantizedTensor(), set_func=lambda _: None) is False


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
        graphs.append(schedule_weight_prefetches(gm, lookahead=1))
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
    assert "comfy_weight.prefetch_weight_bias" in graph_text
    assert "comfy_weight.resolve_prefetched_weight_bias" in graph_text
    assert "comfy_weight.resolve_weight_bias" not in graph_text


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
    assert len(prefetches) == 2
    second_prefetch = prefetches[1]
    first_resolve = next(i for i, node in enumerate(nodes) if "resolve_prefetched_weight_bias" in str(node.target))

    assert second_prefetch > first_resolve


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
    second_prefetch = [i for i, node in enumerate(nodes) if "prefetch_weight_bias" in str(node.target)][1]
    first_resolve = next(i for i, node in enumerate(nodes) if "resolve_prefetched_weight_bias" in str(node.target))
    assert second_prefetch > first_resolve


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
        graphs.append(schedule_weight_prefetches(gm, lookahead=2))
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
    anchors = [i for i, node in enumerate(nodes) if "prefetch_anchor" in str(node.target)]
    exemplar = next(i for i, node in enumerate(nodes) if node.name == "exemplar")

    assert second_prefetch < exemplar
    assert anchors
    assert second_prefetch < anchors[0] < exemplar


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
            schedule_weight_prefetches(gm, lookahead=2)
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
    assert events == [
        ("prefetch", "prefetch-1"),
        ("resolve", "prefetch-1"),
        ("release", "prefetch-1"),
        ("prefetch", "prefetch-2"),
        ("resolve", "prefetch-2"),
        ("release", "prefetch-2"),
    ]
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


def test_direct_materialize_prefetch_declines_dtype_changing_weight(monkeypatch):
    from comfy import ops

    class Module:
        pass

    module = Module()
    module._v = object()
    module.weight = torch.empty((2, 2), dtype=torch.float8_e4m3fn)
    module.bias = None

    monkeypatch.setenv("COMFY_DIRECT_MATERIALIZE_PINNING", "1")
    monkeypatch.setattr(ops.model_management, "device_supports_non_blocking", lambda device: True)

    assert ops._legacy_weight_cast_prefetch(
        module,
        torch.device("cuda", 0),
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        False,
    ) is None
