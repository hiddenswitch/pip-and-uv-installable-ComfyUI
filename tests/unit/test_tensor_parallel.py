import contextlib
import threading
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import comfy.ops as comfy_ops
from comfy import model_management
from comfy.ldm.minimax.model import Attention, MLP
from comfy.model_base import BaseModel, MiniMaxH3
from comfy.model_patcher import ModelPatcherDynamic
from comfy.tensor_parallel import (
    AbstractBaseTensorParallelOperations,
    TensorParallelConfig,
    TorchDistributedTensorParallelOperations,
    shard_minimax_h3_state_dict,
    tensor_parallel_operations,
)
from comfy.tensor_parallel.distributed import (
    RemoteModelParallelRankModel,
    TorchDistributedTensorParallelExecutor,
    _ObjectCoordinator,
    _broadcast_tensors,
    _without_execution_caches,
)


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute_method(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return args[0] * 2


class RecordingCoordinator:
    def broadcast_command(self, command):
        self.command = command

    def receive_object(self, rank):
        return {"kind": "done"}


class RecordingRootModel:
    def forward(self, value):
        self.value = value
        return value


class FailingRootModel:
    def forward(self):
        raise RuntimeError("rank zero failed")


class PreprocessingDiffusion(torch.nn.Module):
    dtype = torch.float32

    def preprocess_text_embeds(self, value):
        raise AssertionError("TP preprocessing must run through the rank executor")


class CompletedBroadcast:
    def __init__(self):
        self.waited = False
        self.stream_blocked = False

    def wait(self):
        self.waited = True

    def block_current_stream(self):
        self.stream_blocked = True


class ThreadCollective:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.values = {}

    def sum(self, index, tensor):
        with self.condition:
            values = self.values.setdefault(index, [])
            values.append(tensor)
            if len(values) == self.size:
                total = sum(value.clone() for value in values)
                for value in values:
                    value.copy_(total)
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: len(values) == self.size)
        return tensor


class ThreadTensorParallelOperations(AbstractBaseTensorParallelOperations):
    def __init__(self, collective, rank):
        self.collective = collective
        self._rank = rank
        self.index = 0

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self.collective.size

    def sum(self, tensor):
        index = self.index
        self.index += 1
        return self.collective.sum(index, tensor)


def _attention(query, key, value, heads, mask=None, **kwargs):
    del heads, kwargs
    scores = torch.matmul(query, key.transpose(-2, -1)) * query.shape[-1] ** -0.5
    if mask is not None:
        scores = scores + mask
    return torch.matmul(scores.softmax(dim=-1), value).transpose(1, 2).flatten(2)


def _run_parallel(modules, value):
    outputs = [None] * len(modules)
    threads = [
        threading.Thread(target=lambda index=index: outputs.__setitem__(index, modules[index](value)))
        for index in range(len(modules))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outputs


def test_minimax_checkpoint_shards_megatron_projection_dimensions():
    state = {
        "blocks.0.attn.qkv_proj.weight": torch.arange(24).reshape(12, 2),
        "blocks.0.attn.qkv_proj.weight_scale": torch.arange(12).reshape(12, 1),
        "blocks.0.attn.out_proj.weight": torch.arange(32).reshape(4, 8),
        "blocks.0.mlp.fc1.weight": torch.arange(32).reshape(16, 2),
        "blocks.0.mlp.fc1.weight_scale": torch.arange(16).reshape(16, 1),
        "blocks.0.mlp.fc2.weight": torch.arange(32).reshape(4, 8),
        "blocks.0.norm.weight": torch.ones(4),
    }

    first = shard_minimax_h3_state_dict(state, 0, 2)
    second = shard_minimax_h3_state_dict(state, 1, 2)

    assert first["blocks.0.attn.qkv_proj.weight"][:, 0].tolist() == [0, 2, 8, 10, 16, 18]
    assert second["blocks.0.attn.qkv_proj.weight"][:, 0].tolist() == [4, 6, 12, 14, 20, 22]
    assert first["blocks.0.mlp.fc1.weight"][:, 0].tolist() == [0, 2, 4, 6, 16, 18, 20, 22]
    assert second["blocks.0.mlp.fc1.weight"][:, 0].tolist() == [8, 10, 12, 14, 24, 26, 28, 30]
    assert torch.equal(first["blocks.0.attn.out_proj.weight"], state["blocks.0.attn.out_proj.weight"][:, :4])
    assert torch.equal(second["blocks.0.mlp.fc2.weight"], state["blocks.0.mlp.fc2.weight"][:, 4:])
    assert first["blocks.0.norm.weight"] is state["blocks.0.norm.weight"]


def test_minimax_conditioning_uses_tensor_parallel_executor():
    model = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(model)
    model.diffusion_model = PreprocessingDiffusion()
    model.pipeline_executor = RecordingExecutor()
    model.concat_keys = ()
    model.manual_cast_dtype = None

    cross_attn = torch.randn(1, 3, 4)
    conditions = model.extra_conds(cross_attn=cross_attn, device=torch.device("cpu"))

    assert model.pipeline_executor.calls[0][0] == "preprocess_text_embeds"
    torch.testing.assert_close(conditions["c_crossattn"].cond, cross_attn * 2)


def test_tensor_parallel_cpu_inputs_use_control_group(monkeypatch):
    calls = []

    def broadcast(tensor, src, group, async_op):
        completion = CompletedBroadcast()
        calls.append((tensor, src, group, async_op, completion))
        return completion

    monkeypatch.setattr("comfy.tensor_parallel.distributed.dist.broadcast", broadcast)
    operations = type("Operations", (), {
        "control_process_group": "gloo",
        "process_group": "nccl",
    })()

    result = _broadcast_tensors(operations, tensors={"seed": torch.tensor(42)})

    assert result["seed"].device.type == "cpu"
    assert calls[0][2] == "gloo"
    assert calls[0][4].waited


def test_tensor_parallel_gpu_inputs_move_to_rank_device_before_collective(monkeypatch):
    calls = []

    def broadcast(tensor, src, group, async_op):
        completion = CompletedBroadcast()
        calls.append((tensor, src, group, async_op, completion))
        return completion

    monkeypatch.setattr("comfy.tensor_parallel.distributed.dist.broadcast", broadcast)
    operations = type("Operations", (), {
        "device": torch.device("cuda", 1),
        "control_process_group": "gloo",
        "process_group": "nccl",
    })()

    with FakeTensorMode():
        result = _broadcast_tensors(
            operations,
            tensors={"latent": torch.empty(2, device="cuda:0")},
        )

    assert result["latent"].device == torch.device("cuda", 1)
    assert calls[0][0].device == torch.device("cuda", 1)
    assert calls[0][2] == "nccl"
    assert calls[0][4].stream_blocked


def test_tensor_parallel_sum_reduces_owned_output_in_place(monkeypatch):
    calls = []

    def all_reduce(tensor, group, async_op):
        completion = CompletedBroadcast()
        calls.append((tensor, group, async_op, completion))
        return completion

    monkeypatch.setattr("torch.distributed.all_reduce", all_reduce)
    operations = TorchDistributedTensorParallelOperations(
        0, 2, torch.device("cpu"), "nccl"
    )
    tensor = torch.randn(3, 4)

    result = operations.sum(tensor)

    assert result is tensor
    assert calls[0][:3] == (tensor, "nccl", True)
    assert calls[0][3].stream_blocked


def test_tensor_parallel_root_uses_collective_prepared_inputs(monkeypatch):
    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed._broadcast_tensors",
        lambda operations, tensors: {name: tensor + 1 for name, tensor in tensors.items()},
    )
    root = RecordingRootModel()
    executor = TorchDistributedTensorParallelExecutor(
        root,
        object(),
        RecordingCoordinator(),
        (),
        (),
        (torch.device("cpu"), torch.device("cpu")),
        (0, 0),
    )

    output = executor.execute(torch.tensor(2))

    assert output.item() == 3
    assert root.value.item() == 3


def test_model_parallel_root_failure_aborts_device_group(monkeypatch):
    aborted = []
    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed.dist.distributed_c10d._abort_process_group",
        aborted.append,
    )
    operations = type("Operations", (), {"process_group": "nccl"})()
    executor = TorchDistributedTensorParallelExecutor(
        FailingRootModel(), operations, RecordingCoordinator(), (), (),
        (torch.device("cpu"),), (0,),
    )

    with pytest.raises(RuntimeError, match="rank zero failed"):
        executor.execute()

    assert aborted == ["nccl"]
    assert executor._device_group_active is False


def test_model_parallel_transport_removes_only_rank_local_compile_wrapper():
    compile_wrapper = lambda executor, *args, **kwargs: executor(*args, **kwargs)
    extension_wrapper = object()
    value = {
        "transformer_options": {
            "wrappers": {
                "apply_model": {
                    "torch.compile": [compile_wrapper],
                    "extension": [extension_wrapper],
                },
                "diffusion_model": {"extension": [extension_wrapper]},
            },
            "layout": object(),
            "seed": 42,
        }
    }

    cleaned = _without_execution_caches(value)

    options = cleaned["transformer_options"]
    assert "layout" not in options
    assert options["wrappers"]["apply_model"] == {
        "extension": [extension_wrapper]
    }
    assert options["wrappers"]["diffusion_model"] == {
        "extension": [extension_wrapper]
    }
    assert options["seed"] == 42


def test_tensor_parallel_finish_execution_releases_all_ranks(monkeypatch):
    coordinator = RecordingCoordinator()
    finished = []
    monkeypatch.setattr(
        "comfy.model_prefetch.finish_model_execution",
        lambda: finished.append("root"),
    )
    executor = TorchDistributedTensorParallelExecutor(
        RecordingRootModel(), object(), coordinator, (), (),
        (torch.device("cpu"), torch.device("cpu")), (0, 0),
    )

    executor.finish_execution()

    assert coordinator.command["kind"] == "finish_execution"
    assert finished == ["root"]


def test_xdit_memory_estimate_uses_rank_local_sequence_share(monkeypatch):
    monkeypatch.setattr(
        "comfy.model_management.pytorch_attention_flash_attention",
        lambda: True,
    )
    monkeypatch.setattr("comfy.model_management.xformers_enabled", lambda: False)

    def estimate(parallel_size):
        diffusion_model = SimpleNamespace()
        if parallel_size is not None:
            diffusion_model.xdit_sequence_parallel = SimpleNamespace(
                size=parallel_size,
            )
        model = SimpleNamespace(
            diffusion_model=diffusion_model,
            memory_usage_factor_conds=(),
            memory_usage_shape_process={},
            memory_usage_factor=4.0,
            get_dtype_inference=lambda: torch.bfloat16,
        )
        return BaseModel.memory_required(model, [2, 16, 4, 8, 8])

    assert estimate(2) == estimate(None) / 2


def test_remote_rank_receives_rank_local_activation_reserve(monkeypatch):
    class FakeExecutor:
        def __init__(self):
            self.command = None

        def remote_rank_command(self, rank, command):
            assert rank == 1
            self.command = command
            return {"kind": "rank_model", "loaded_size": 100}

    executor = FakeExecutor()
    remote = RemoteModelParallelRankModel(
        executor,
        rank=1,
        device=torch.device("cuda:1"),
        size=100,
        dtype=torch.bfloat16,
        dynamic=True,
    )
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    monkeypatch.setattr(model_management, "cleanup_models_gc", lambda: None)
    monkeypatch.setattr(
        model_management,
        "prepare_device_model_loads",
        lambda required, **_kwargs: [],
    )

    model_management._load_models_gpu(
        [remote],
        memory_required=400,
        minimum_memory_required=200,
    )

    assert executor.command["memory_required"] == 400
    assert executor.command["minimum_memory_required"] == 200


def test_dynamic_rank_load_flushes_stale_allocator_reservations(monkeypatch):
    emptied = []
    monkeypatch.setattr(model_management, "free_memory", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(model_management, "get_free_memory", lambda _device: 10_000)
    monkeypatch.setattr(
        "comfy.memory_management.aimdo_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        model_management,
        "_soft_empty_cache",
        lambda force=False: emptied.append(force),
    )

    model_management.prepare_device_model_loads(
        {torch.device("cuda:0"): 100},
        extra_mem=20,
        minimum_memory_required=10,
        free_for_dynamic=True,
    )

    assert emptied == [True]


def test_dynamic_patcher_applies_model_manager_weight_budget():
    class FakeVBAR:
        def __init__(self):
            self.watermarks = []

        def set_watermark(self, value):
            self.watermarks.append(value)

    vbar = FakeVBAR()
    patcher = SimpleNamespace(
        model=SimpleNamespace(current_weight_patches_uuid=None),
        patches_uuid=None,
        offload_device=torch.device("cpu"),
        loaded_size=lambda: 100,
        model_size=lambda: 1_000,
        use_ejected=lambda **_kwargs: contextlib.nullcontext(),
        unpatch_model=lambda *_args, **_kwargs: None,
        patch_model=lambda *_args, **_kwargs: None,
        load=lambda *_args, **_kwargs: None,
        detach=lambda: None,
        _vbar_get=lambda: vbar,
    )

    ModelPatcherDynamic.partially_load(
        patcher,
        torch.device("cuda:0"),
        extra_memory=400,
    )

    assert vbar.watermarks == [500]


def test_tensor_parallel_commands_propagate_trace_context(monkeypatch):
    values_seen = []

    def inject(carrier, context):
        del context
        carrier["traceparent"] = "00-test"

    def broadcast_object_list(values, src):
        del src
        values_seen.extend(values)

    monkeypatch.setattr("comfy.distributed.tracing.propagate.inject", inject)
    monkeypatch.setattr(
        "comfy.tensor_parallel.distributed.dist.broadcast_object_list",
        broadcast_object_list,
    )

    _ObjectCoordinator().broadcast_command({"kind": "execute"})

    assert values_seen == [{
        "kind": "execute",
        "trace_context": {"traceparent": "00-test"},
    }]


def test_minimax_tensor_parallel_attention_and_mlp_match_full_model(monkeypatch):
    monkeypatch.setattr("comfy.ldm.minimax.model.optimized_attention", _attention)
    torch.manual_seed(5)
    full_attention = Attention(8, 4, 2, 1e-5, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init)
    full_mlp = MLP(8, 12, dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init)
    for module in (full_attention, full_mlp):
        for parameter in module.parameters():
            torch.nn.init.uniform_(parameter, -0.1, 0.1)

    collective = ThreadCollective(2)
    attentions = []
    mlps = []
    for rank in range(2):
        config = TensorParallelConfig(ThreadTensorParallelOperations(collective, rank))
        operations = tensor_parallel_operations(comfy_ops.disable_weight_init, config)
        attention = Attention(8, 4, 2, 1e-5, dtype=torch.float32, device="cpu",
                              operations=operations)
        mlp = MLP(8, 12, dtype=torch.float32, device="cpu",
                  operations=operations)
        qkv = full_attention.qkv_proj.weight.reshape(3, 8, 8)
        attention.qkv_proj.weight.detach().copy_(qkv[:, rank * 4:(rank + 1) * 4].reshape(12, 8))
        attention.out_proj.weight.detach().copy_(full_attention.out_proj.weight[:, rank * 4:(rank + 1) * 4])
        attention.q_norm.weight.detach().copy_(full_attention.q_norm.weight)
        attention.k_norm.weight.detach().copy_(full_attention.k_norm.weight)
        fc1 = full_mlp.fc1.weight.reshape(2, 12, 8)
        mlp.fc1.weight.detach().copy_(fc1[:, rank * 6:(rank + 1) * 6].reshape(12, 8))
        mlp.fc2.weight.detach().copy_(full_mlp.fc2.weight[:, rank * 6:(rank + 1) * 6])
        attentions.append(attention)
        mlps.append(mlp)

    value = torch.randn(5, 8)
    expected_attention = full_attention(value)
    expected_mlp = full_mlp(value)
    attention_outputs = _run_parallel(attentions, value)
    mlp_outputs = _run_parallel(mlps, value)

    for output in attention_outputs:
        torch.testing.assert_close(output, expected_attention)
    for output in mlp_outputs:
        torch.testing.assert_close(output, expected_mlp)
