"""Run output-row W4A8 sharding through the real CUDA kernels and NCCL."""
import json
import importlib
from importlib.metadata import distribution
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from comfy import ops
from comfy.tensor_parallel.operations import gathered_output_operations
from comfy.tensor_parallel.runtime import TorchDistributedTensorParallelOperations
from comfy.tensor_parallel.types import TensorParallelConfig
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy_compatibility.vanilla import prepare_vanilla_environment


def _w4a8_rank(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        entry = next(ep for ep in distribution("comfyui-hunyuanimage3").entry_points if ep.group == "comfyui.custom_nodes").load()
        root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "ComfyUI-HunyuanImage3"
        prepare_vanilla_environment()
        assert _vanilla_load_custom_nodes_1(str(root)).NODE_CLASS_MAPPINGS
        sliced = importlib.import_module(root.name + ".hunyuan_image_3.ops").expert_linear_sliced
        device = torch.device("cuda", rank)
        base = ops.mixed_precision_ops({}, torch.bfloat16)
        parallel = TensorParallelConfig(TorchDistributedTensorParallelOperations(rank, 2, device, dist.group.WORLD))
        sharded = gathered_output_operations(base, parallel)
        for bank in (False, True):
            shape = (2, 256, 256) if bank else (256, 256)
            state = {
                "weight": torch.randint(-128, 127, shape, dtype=torch.int8),
                "weight_s_rel": torch.ones((*shape[:-1], 32)).to(torch.float8_e4m3fn),
                "weight_s_channel": torch.full(shape[:-1], 0.01),
                "weight_codebook": torch.linspace(-1, 1, 16),
                "comfy_quant": torch.tensor(list(json.dumps({"format": "asym_w4a8_int8", "group_size": 16,
                                                             "convrot_groupsize": 256}).encode()), dtype=torch.uint8),
            }
            reference = base.MoEExperts(2, 512, 256, bias=False) if bank else base.Linear(512, 256, bias=False)
            target = sharded.MoEExperts(2, 512, 256, bias=False) if bank else sharded.Linear(512, 256, bias=False)
            reference.load_state_dict(dict(state))
            axis = 1 if bank else 0
            owned = {k: v.narrow(axis, rank * 128, 128).clone() if k in ("weight", "weight_s_rel", "weight_s_channel")
                     else v.clone() for k, v in state.items()}
            target.load_state_dict(owned)
            for tokens in (1, 5):
                value = torch.randn(tokens, 512).to(device=device, dtype=torch.bfloat16)
                expected = reference.expert_linear(value, 1) if bank else reference(value)
                actual = target.expert_linear(value, 1) if bank else target(value)
                assert expected.shape == (tokens, 256), (bank, tokens, expected.shape, reference.weight.shape)
                assert actual.shape == expected.shape, (bank, tokens, actual.shape, expected.shape)
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
                if bank:
                    torch.testing.assert_close(sliced(target, value, 1), sliced(reference, value, 1), rtol=0.01, atol=0.002)
            if bank:
                # Weight patches take the dense fallback, whose flattened bank
                # must still be indexed by expert rather than by output row.
                reference.weight_function = target.weight_function = [lambda weight: weight * 1.1]
                torch.testing.assert_close(target.expert_linear(value, 1), reference.expert_linear(value, 1), rtol=0.01, atol=0.002)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_w4a8_tp2_matches_unsharded_cuda(tmp_path):
    mp.spawn(_w4a8_rank, args=((tmp_path / "nccl-init").as_uri(),), nprocs=2, join=True)
