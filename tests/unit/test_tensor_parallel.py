import threading

import torch

import comfy.ops as comfy_ops
from comfy.ldm.minimax.model import Attention, MLP
from comfy.model_base import MiniMaxH3
from comfy.tensor_parallel import (
    AbstractBaseTensorParallelOperations,
    TensorParallelConfig,
    shard_minimax_h3_state_dict,
    tensor_parallel_operations,
)
from comfy.tensor_parallel.distributed import _broadcast_tensors


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute_method(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return args[0] * 2


class PreprocessingDiffusion(torch.nn.Module):
    dtype = torch.float32

    def preprocess_text_embeds(self, value):
        raise AssertionError("TP preprocessing must run through the rank executor")


class CompletedBroadcast:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


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
