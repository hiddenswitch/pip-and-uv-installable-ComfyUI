from __future__ import annotations

from collections.abc import Mapping

import torch


def _section_rows(tensor: torch.Tensor, sections: int, rank: int, size: int) -> torch.Tensor:
    section_size = tensor.shape[0] // sections
    rank_size = section_size // size
    return torch.cat([
        tensor.narrow(0, section * section_size + rank * rank_size, rank_size)
        for section in range(sections)
    ]).clone()


def _columns(tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    rank_size = tensor.shape[1] // size
    return tensor.narrow(1, rank * rank_size, rank_size).clone()


def shard_minimax_h3_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    rank: int,
    size: int,
) -> dict[str, torch.Tensor]:
    """Return one Megatron-style MiniMax shard without retaining full-weight views."""
    if size < 2:
        return dict(state_dict)

    sharded = {}
    for key, tensor in state_dict.items():
        if key.endswith(".attn.qkv_proj.weight") or key.endswith(".attn.qkv_proj.weight_scale"):
            sharded[key] = _section_rows(tensor, 3, rank, size)
        elif key.endswith(".mlp.fc1.weight") or key.endswith(".mlp.fc1.weight_scale"):
            sharded[key] = _section_rows(tensor, 2, rank, size)
        elif key.endswith(".attn.out_proj.weight") or key.endswith(".mlp.fc2.weight"):
            sharded[key] = _columns(tensor, rank, size)
        else:
            sharded[key] = tensor
    return sharded
