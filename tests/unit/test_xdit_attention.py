from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import wrap_attn
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.xdit import (
    RingAttentionOverride,
    UlyssesAttentionOverride,
    XDiTSequenceParallelConfig,
    local_padding_mask,
    split_sequence,
)
from comfy.xdit.attention import install_ulysses_attention_override


class _ThreadedCollectives:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.rounds = {}

    def exchange(self, key, rank, tensor, build_results):
        with self.condition:
            round_state = self.rounds.setdefault(
                key,
                {"inputs": {}, "results": None},
            )
            round_state["inputs"][rank] = tensor
            if len(round_state["inputs"]) == self.size:
                round_state["results"] = build_results(round_state["inputs"])
                self.condition.notify_all()
            else:
                self.condition.wait_for(lambda: round_state["results"] is not None)
            return round_state["results"][rank]


class _SimulatedUlyssesOperations:
    def __init__(self, collectives, rank):
        self.collectives = collectives
        self.rank = rank
        self.world_size = collectives.size
        self.all_to_all_round = 0
        self.all_gather_round = 0

    def all_to_all(self, tensor):
        round_index = self.all_to_all_round
        self.all_to_all_round += 1

        def build(inputs):
            return {
                destination: torch.cat(
                    [
                        torch.chunk(inputs[source], self.world_size, dim=0)[
                            destination
                        ]
                        for source in range(self.world_size)
                    ],
                    dim=0,
                )
                for destination in range(self.world_size)
            }

        return self.collectives.exchange(
            ("all_to_all", round_index),
            self.rank,
            tensor,
            build,
        )

    def all_gather(self, tensor, dim):
        round_index = self.all_gather_round
        self.all_gather_round += 1

        def build(inputs):
            gathered = torch.cat(
                [inputs[source] for source in range(self.world_size)],
                dim=dim,
            )
            return {rank: gathered for rank in range(self.world_size)}

        return self.collectives.exchange(
            ("all_gather", round_index),
            self.rank,
            tensor,
            build,
        )

    def ring_attention(self, operation, query, key, value):
        round_index = getattr(self, "ring_round", 0)
        self.ring_round = round_index + 1

        def build(inputs):
            return {
                rank: tuple(
                    inputs[(rank - step) % self.world_size]
                    for step in range(self.world_size)
                )
                for rank in range(self.world_size)
            }

        shards = self.collectives.exchange(
            ("ring", round_index),
            self.rank,
            (key, value),
            build,
        )
        output = None
        logsumexp = None
        for source_key, source_value in shards:
            block_output, block_logsumexp = operation(
                query,
                source_key,
                source_value,
                is_causal=False,
            )
            if output is None:
                output = block_output
                logsumexp = block_logsumexp
                continue
            weight = torch.sigmoid(block_logsumexp - logsumexp).unsqueeze(-1)
            output = output - weight * (output - block_output)
            logsumexp = torch.logaddexp(logsumexp, block_logsumexp)
        return output, logsumexp


@wrap_attn
def _native_attention(
    query,
    key,
    value,
    heads,
    mask=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    assert skip_reshape
    assert heads == query.shape[1]
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
    )
    if kwargs.get("return_lse", False):
        scale = kwargs.get("scale", query.shape[-1] ** -0.5)
        scores = torch.matmul(
            query.to(torch.float32),
            key.to(torch.float32).transpose(-2, -1),
        ) * scale
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores + mask.to(scores.dtype)
        return output, torch.logsumexp(scores, dim=-1)
    if skip_output_reshape:
        return output
    return output.transpose(1, 2).flatten(2)


def _run_rank(rank, collectives, query, key, value, padding_mask=None):
    parallel = XDiTSequenceParallelConfig(
        _SimulatedUlyssesOperations(collectives, rank)
    )
    options = {
        "optimized_attention_override": UlyssesAttentionOverride(
            parallel,
            padding_mask,
        )
    }
    return _native_attention(
        query,
        key,
        value,
        query.shape[1],
        skip_reshape=True,
        transformer_options=options,
    )


def _run_ring_rank(
    rank,
    collectives,
    query,
    key,
    value,
    padding_mask=None,
    sequence_padding=0,
):
    parallel = XDiTSequenceParallelConfig(
        _SimulatedUlyssesOperations(collectives, rank),
        "ring",
    )
    options = {
        "optimized_attention_override": RingAttentionOverride(
            parallel,
            padding_mask,
            sequence_padding,
        )
    }
    return _native_attention(
        query,
        key,
        value,
        query.shape[1],
        skip_reshape=True,
        transformer_options=options,
    )


def test_ulysses_attention_matches_full_sequence_attention():
    torch.manual_seed(7)
    size = 2
    query = torch.randn(1, 4, 6, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = _native_attention(
        query,
        key,
        value,
        query.shape[1],
        skip_reshape=True,
    )
    collectives = _ThreadedCollectives(size)

    with ThreadPoolExecutor(max_workers=size) as executor:
        futures = [
            executor.submit(
                _run_rank,
                rank,
                collectives,
                torch.chunk(query, size, dim=2)[rank],
                torch.chunk(key, size, dim=2)[rank],
                torch.chunk(value, size, dim=2)[rank],
            )
            for rank in range(size)
        ]
        actual = torch.cat([future.result() for future in futures], dim=1)

    torch.testing.assert_close(actual, expected)


def test_ring_attention_matches_full_sequence_attention():
    torch.manual_seed(17)
    size = 2
    query = torch.randn(1, 4, 6, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = _native_attention(
        query,
        key,
        value,
        query.shape[1],
        skip_reshape=True,
    )
    collectives = _ThreadedCollectives(size)

    with ThreadPoolExecutor(max_workers=size) as executor:
        futures = [
            executor.submit(
                _run_ring_rank,
                rank,
                collectives,
                torch.chunk(query, size, dim=2)[rank],
                torch.chunk(key, size, dim=2)[rank],
                torch.chunk(value, size, dim=2)[rank],
            )
            for rank in range(size)
        ]
        actual = torch.cat([future.result() for future in futures], dim=1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_ring_attention_trims_odd_sequence_padding():
    torch.manual_seed(19)
    size = 2
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = _native_attention(
        query,
        key,
        value,
        query.shape[1],
        skip_reshape=True,
    )
    collectives = _ThreadedCollectives(size)

    def run(rank):
        parallel = XDiTSequenceParallelConfig(
            _SimulatedUlyssesOperations(collectives, rank),
            "ring",
        )
        local_query, padding = split_sequence(query, parallel, 2)
        local_key, _ = split_sequence(key, parallel, 2)
        local_value, _ = split_sequence(value, parallel, 2)
        return _run_ring_rank(
            rank,
            collectives,
            local_query,
            local_key,
            local_value,
            sequence_padding=padding,
        )

    with ThreadPoolExecutor(max_workers=size) as executor:
        actual = torch.cat(list(executor.map(run, range(size))), dim=1)
    actual = actual[:, : query.shape[2]]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_minimax_text_preprocessing_uses_sequence_parallel_layout():
    class IdentityRefiner(torch.nn.Module):
        def forward(self, tensor, transformer_options=None):
            assert "optimized_attention_override" in transformer_options
            return tensor

    torch.manual_seed(23)
    size = 2
    text = torch.randn(1, 5, 8)
    collectives = _ThreadedCollectives(size)

    def run(rank):
        model = MiniMaxH3Model.__new__(MiniMaxH3Model)
        torch.nn.Module.__init__(model)
        model.hidden_size = 8
        model.condition_proj = torch.nn.Identity()
        model.token_refiner = IdentityRefiner()
        model.xdit_sequence_parallel = XDiTSequenceParallelConfig(
            _SimulatedUlyssesOperations(collectives, rank),
            "ring",
        )
        return model.preprocess_text_embeds(text)

    with ThreadPoolExecutor(max_workers=size) as executor:
        outputs = list(executor.map(run, range(size)))

    for output in outputs:
        torch.testing.assert_close(output, text)


def test_ulysses_attention_masks_sequence_padding():
    torch.manual_seed(11)
    size = 2
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = _native_attention(
        query,
        key,
        value,
        query.shape[1],
        skip_reshape=True,
    )
    collectives = _ThreadedCollectives(size)

    def rank_inputs(rank):
        operations = _SimulatedUlyssesOperations(collectives, rank)
        parallel = XDiTSequenceParallelConfig(operations)
        local_query, padding = split_sequence(query, parallel, 2)
        local_key, _ = split_sequence(key, parallel, 2)
        local_value, _ = split_sequence(value, parallel, 2)
        mask = local_padding_mask(
            query.shape[2],
            padding,
            parallel,
            query.dtype,
            query.device,
        )
        options = {
            "optimized_attention_override": UlyssesAttentionOverride(
                parallel,
                mask,
            )
        }
        return _native_attention(
            local_query,
            local_key,
            local_value,
            query.shape[1],
            skip_reshape=True,
            transformer_options=options,
        )

    with ThreadPoolExecutor(max_workers=size) as executor:
        actual = torch.cat(
            list(executor.map(rank_inputs, range(size))),
            dim=1,
        )[:, : query.shape[2]]

    torch.testing.assert_close(actual, expected)


def test_ulysses_attention_trims_suffix_padding_before_native_attention():
    torch.manual_seed(13)
    size = 2
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = _native_attention(
        query,
        key,
        value,
        query.shape[1],
        skip_reshape=True,
    )
    collectives = _ThreadedCollectives(size)
    native_lengths = []

    def rank_inputs(rank):
        operations = _SimulatedUlyssesOperations(collectives, rank)
        parallel = XDiTSequenceParallelConfig(operations)
        local_query, padding = split_sequence(query, parallel, 2)
        local_key, _ = split_sequence(key, parallel, 2)
        local_value, _ = split_sequence(value, parallel, 2)

        @wrap_attn
        def recording_attention(
            local_query,
            local_key,
            local_value,
            heads,
            mask=None,
            skip_reshape=False,
            skip_output_reshape=False,
            **_kwargs,
        ):
            assert mask is None
            assert skip_reshape
            native_lengths.append(local_query.shape[2])
            output = F.scaled_dot_product_attention(
                local_query,
                local_key,
                local_value,
            )
            if skip_output_reshape:
                return output
            return output.transpose(1, 2).flatten(2)

        return recording_attention(
            local_query,
            local_key,
            local_value,
            query.shape[1],
            skip_reshape=True,
            transformer_options={
                "optimized_attention_override": UlyssesAttentionOverride(
                    parallel,
                    sequence_padding=padding,
                )
            },
        )

    with ThreadPoolExecutor(max_workers=size) as executor:
        actual = torch.cat(
            list(executor.map(rank_inputs, range(size))),
            dim=1,
        )[:, : query.shape[2]]

    assert native_lengths == [5, 5]
    torch.testing.assert_close(actual, expected)


def test_ulysses_override_is_traceable_by_torch_compile():
    class SingleRankOperations:
        rank = 0
        world_size = 1

        @staticmethod
        def all_to_all(tensor):
            return tensor

        @staticmethod
        def all_gather(tensor, dim):
            del dim
            return tensor

    parallel = XDiTSequenceParallelConfig(SingleRankOperations())

    def attention_with_override(query, key, value):
        options = {}
        install_ulysses_attention_override(options, parallel)
        return _native_attention(
            query,
            key,
            value,
            query.shape[1],
            skip_reshape=True,
            transformer_options=options,
        )

    query = torch.randn(1, 2, 4, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = attention_with_override(query, key, value)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(
            attention_with_override,
            backend="eager",
            fullgraph=True,
        )
        actual = compiled(query, key, value)
    finally:
        torch._dynamo.reset()

    torch.testing.assert_close(actual, expected)
