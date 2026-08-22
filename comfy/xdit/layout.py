from __future__ import annotations

import torch

from .types import XDiTSequenceParallelConfig


def split_sequence(
    tensor: torch.Tensor,
    parallel: XDiTSequenceParallelConfig,
    dim: int,
    pad_value=0,
) -> tuple[torch.Tensor, int]:
    """Pad and return this rank's equal contiguous sequence shard."""

    dim %= tensor.ndim
    length = tensor.shape[dim]
    padding = (-length) % parallel.size
    if padding:
        shape = list(tensor.shape)
        shape[dim] = padding
        pad = torch.full(
            shape,
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, pad), dim=dim)
    return torch.chunk(tensor, parallel.size, dim=dim)[parallel.rank].contiguous(), padding


def gather_sequence(
    tensor: torch.Tensor,
    parallel: XDiTSequenceParallelConfig,
    dim: int,
    padding: int = 0,
) -> torch.Tensor:
    """Gather equal sequence shards and remove padding from the final shard."""

    output = parallel.operations.all_gather(tensor, dim)
    if padding:
        output = output.narrow(dim, 0, output.shape[dim] - padding)
    return output


def local_padding_mask(
    length: int,
    padding: int,
    parallel: XDiTSequenceParallelConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return an additive key mask for this rank's contiguous padded shard."""

    shard_length = (length + padding) // parallel.size
    mask = torch.zeros((1, 1, shard_length), dtype=dtype, device=device)
    if padding and parallel.rank == parallel.size - 1:
        mask[..., -padding:] = torch.finfo(dtype).min
    return mask


def combine_local_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    present = [mask for mask in masks if mask is not None]
    if not present:
        return None
    return torch.cat(present, dim=-1)


def attention_mask_pad_value(dtype: torch.dtype):
    if dtype == torch.bool:
        return False
    if dtype.is_floating_point:
        return torch.finfo(dtype).min
    return 0


def localize_segments(
    segments,
    rank: int,
    size: int,
    padded_length: int,
):
    """Translate global half-open segment ranges into one sequence shard."""

    shard_length = padded_length // size
    shard_start = rank * shard_length
    shard_end = shard_start + shard_length
    localized = []
    for start, end, row in segments:
        local_start = max(start, shard_start)
        local_end = min(end, shard_end)
        if local_start < local_end:
            local_row = row
            if torch.is_tensor(row) and row.ndim > 0:
                local_row = row[local_start - start:local_end - start]
            localized.append(
                (local_start - shard_start, local_end - shard_start, local_row)
            )
    return localized
