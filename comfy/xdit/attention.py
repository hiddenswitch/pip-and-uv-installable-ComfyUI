from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import XDiTSequenceParallelConfig


def _combined_qkv_input_all_to_all(
    parallel: XDiTSequenceParallelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, sequence, head_dim = query.shape
    if heads % parallel.size:
        raise ValueError(
            f"attention heads {heads} must divide Ulysses degree {parallel.size}"
        )
    qkv = torch.stack((query, key, value))
    qkv = qkv.view(
        3,
        batch,
        parallel.size,
        heads // parallel.size,
        sequence,
        head_dim,
    )
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()
    qkv = parallel.operations.all_to_all(qkv)
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).reshape(
        3,
        batch,
        heads // parallel.size,
        sequence * parallel.size,
        head_dim,
    )
    return tuple(torch.unbind(qkv, dim=0))


def _output_all_to_all(
    parallel: XDiTSequenceParallelConfig,
    tensor: torch.Tensor,
) -> torch.Tensor:
    batch, heads, sequence, head_dim = tensor.shape
    if sequence % parallel.size:
        raise ValueError(
            f"attention sequence {sequence} must divide Ulysses degree {parallel.size}"
        )
    tensor = tensor.permute(2, 0, 1, 3).contiguous()
    tensor = parallel.operations.all_to_all(tensor)
    tensor = tensor.reshape(
        parallel.size,
        sequence // parallel.size,
        batch,
        heads,
        head_dim,
    )
    return tensor.permute(2, 0, 3, 1, 4).reshape(
        batch,
        parallel.size * heads,
        sequence // parallel.size,
        head_dim,
    )


def ulysses_attention(
    parallel: XDiTSequenceParallelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_function,
    mask=None,
    sequence_padding: int = 0,
):
    """Run native Comfy attention between xDiT Ulysses all-to-alls.

    Inputs use Comfy's ``[batch, heads, local_sequence, head_dim]`` layout.
    ``attention_function`` receives full-sequence Q/K/V with only this rank's
    heads and returns Comfy's flattened ``[batch, sequence, channels]`` output.
    """

    query, key, value = _combined_qkv_input_all_to_all(
        parallel,
        query,
        key,
        value,
    )
    padded_sequence = query.shape[2]
    if sequence_padding:
        if mask is not None:
            raise ValueError(
                "Ulysses suffix trimming cannot be combined with an attention mask"
            )
        real_sequence = padded_sequence - sequence_padding
        query = query[:, :, :real_sequence]
        key = key[:, :, :real_sequence]
        value = value[:, :, :real_sequence]
    if mask is not None:
        mask = parallel.operations.all_gather(mask.contiguous(), dim=-1)
    local_heads = query.shape[1]
    output = attention_function(query, key, value, local_heads, mask)
    output = output.view(
        output.shape[0],
        output.shape[1],
        local_heads,
        query.shape[-1],
    ).permute(0, 2, 1, 3).contiguous()
    if sequence_padding:
        output = torch.nn.functional.pad(
            output,
            (0, 0, 0, sequence_padding),
        )
    output = _output_all_to_all(parallel, output)
    return output.transpose(1, 2).flatten(2)


def ring_attention(
    parallel: XDiTSequenceParallelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_function,
    mask=None,
    sequence_padding: int = 0,
):
    """Run exact non-causal Ring attention over contiguous sequence shards."""

    local_sequence = key.shape[2]
    gathered_mask = None
    if mask is not None:
        gathered_mask = parallel.operations.all_gather(mask.contiguous(), dim=-1)
    step = 0

    def local_attention(local_query, local_key, local_value, is_causal=False):
        nonlocal step
        if is_causal:
            raise ValueError("xDiT Ring diffusion attention must be non-causal")
        source_rank = (parallel.rank - step) % parallel.size
        local_mask = None
        if gathered_mask is not None:
            start = source_rank * local_sequence
            local_mask = gathered_mask[..., start : start + local_sequence]
        if sequence_padding and source_rank == parallel.size - 1:
            real_length = local_sequence - sequence_padding
            local_key = local_key[:, :, :real_length]
            local_value = local_value[:, :, :real_length]
            if local_mask is not None:
                local_mask = local_mask[..., :real_length]
        step += 1
        return attention_function(
            local_query,
            local_key,
            local_value,
            local_mask,
        )

    output, _logsumexp = parallel.operations.ring_attention(
        local_attention,
        query,
        key,
        value,
    )
    return output.transpose(1, 2).flatten(2)


def _reshape_to_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    batch, sequence, channels = tensor.shape
    if channels % heads:
        raise ValueError(
            f"attention channels {channels} must divide attention heads {heads}"
        )
    return tensor.view(batch, sequence, heads, channels // heads).transpose(1, 2)


def _combine_attention_masks(mask, padding_mask):
    if padding_mask is None:
        return mask
    if mask is None:
        return padding_mask
    if mask.dtype == torch.bool:
        return mask & padding_mask.to(torch.bool)
    return mask + padding_mask.to(mask.dtype)


@dataclass(frozen=True, eq=False)
class UlyssesAttentionOverride:
    """Comfy attention-dispatch override backed by Ulysses collectives.

    Comfy's attention wrapper supplies the selected native kernel as ``native``.
    This keeps SageAttention, PyTorch SDPA, Flash Attention, and future attention
    providers underneath the distributed layout operation instead of teaching
    every model attention class about xDiT.
    """

    parallel: XDiTSequenceParallelConfig
    padding_mask: torch.Tensor | None = None
    sequence_padding: int = 0

    def __call__(
        self,
        native,
        query,
        key,
        value,
        heads,
        mask=None,
        skip_reshape=False,
        skip_output_reshape=False,
        transformer_options=None,
        **kwargs,
    ):
        if not skip_reshape:
            query = _reshape_to_heads(query, heads)
            key = _reshape_to_heads(key, heads)
            value = _reshape_to_heads(value, heads)
        options = dict(transformer_options or {})
        options.pop("optimized_attention_override", None)
        kwargs.pop("_inside_attn_wrapper", None)
        mask = _combine_attention_masks(mask, self.padding_mask)
        output = ulysses_attention(
            self.parallel,
            query,
            key,
            value,
            lambda local_query, local_key, local_value, local_heads, local_mask: native(
                local_query,
                local_key,
                local_value,
                local_heads,
                mask=local_mask,
                skip_reshape=True,
                skip_output_reshape=False,
                transformer_options=options,
                **kwargs,
            ),
            mask=mask,
            sequence_padding=self.sequence_padding,
        )
        if skip_output_reshape:
            return _reshape_to_heads(output, heads)
        return output


@dataclass(frozen=True, eq=False)
class RingAttentionOverride:
    """Comfy attention-dispatch override backed by exact Ring attention."""

    parallel: XDiTSequenceParallelConfig
    padding_mask: torch.Tensor | None = None
    sequence_padding: int = 0

    def __call__(
        self,
        native,
        query,
        key,
        value,
        heads,
        mask=None,
        skip_reshape=False,
        skip_output_reshape=False,
        transformer_options=None,
        **kwargs,
    ):
        if not skip_reshape:
            query = _reshape_to_heads(query, heads)
            key = _reshape_to_heads(key, heads)
            value = _reshape_to_heads(value, heads)
        options = dict(transformer_options or {})
        options.pop("optimized_attention_override", None)
        kwargs.pop("_inside_attn_wrapper", None)
        mask = _combine_attention_masks(mask, self.padding_mask)

        def attention_with_lse(local_query, local_key, local_value, local_mask):
            result = native(
                local_query,
                local_key,
                local_value,
                heads,
                mask=local_mask,
                skip_reshape=True,
                skip_output_reshape=True,
                transformer_options=options,
                return_lse=True,
                **kwargs,
            )
            if not isinstance(result, tuple) or len(result) < 2:
                raise RuntimeError(
                    "The selected attention backend cannot return logsumexp "
                    "required by xDiT Ring attention"
                )
            return result

        output = ring_attention(
            self.parallel,
            query,
            key,
            value,
            attention_with_lse,
            mask=mask,
            sequence_padding=self.sequence_padding,
        )
        if skip_output_reshape:
            return _reshape_to_heads(output, heads)
        return output


def install_ulysses_attention_override(
    transformer_options: dict,
    parallel: XDiTSequenceParallelConfig,
    padding_mask: torch.Tensor | None = None,
    sequence_padding: int = 0,
) -> None:
    transformer_options["optimized_attention_override"] = (
        UlyssesAttentionOverride(parallel, padding_mask, sequence_padding)
    )


def install_sequence_parallel_attention_override(
    transformer_options: dict,
    parallel: XDiTSequenceParallelConfig,
    padding_mask: torch.Tensor | None = None,
    sequence_padding: int = 0,
) -> None:
    if parallel.strategy == "ring":
        transformer_options["optimized_attention_override"] = RingAttentionOverride(
            parallel,
            padding_mask,
            sequence_padding,
        )
        return
    install_ulysses_attention_override(
        transformer_options,
        parallel,
        padding_mask,
        sequence_padding,
    )
