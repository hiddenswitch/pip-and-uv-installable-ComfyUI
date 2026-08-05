from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.distributed._functional_collectives as functional_collectives


class AbstractBaseXDiTSequenceParallelOperations(ABC):
    """Injected collectives used by xDiT-style sequence parallelism."""

    @property
    @abstractmethod
    def rank(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def world_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def ring_attention(
        self,
        operation,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run an injected local attention operation across rotated KV shards."""

        raise NotImplementedError


class TorchDistributedUlyssesOperations(
    AbstractBaseXDiTSequenceParallelOperations
):
    """Ulysses collectives backed by an injected torch.distributed group."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        process_group,
        control_process_group=None,
    ):
        self._rank = rank
        self._world_size = world_size
        self.device = device
        self.process_group = process_group
        self.control_process_group = control_process_group

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @staticmethod
    def _wait(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(
            tensor,
            functional_collectives.AsyncCollectiveTensor,
        ):
            return tensor.wait()
        return tensor

    def all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        output = functional_collectives.all_to_all_single(
            tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group=self.process_group,
        )
        return self._wait(output)

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        output = functional_collectives.all_gather_tensor(
            tensor.contiguous(),
            gather_dim=dim,
            group=self.process_group,
        )
        return self._wait(output)

    def ring_attention(
        self,
        operation,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch.distributed.tensor.experimental._attention import (
            _cp_options,
            _templated_ring_attention,
        )

        # Diffusion attention is non-causal, so the causal round-robin load
        # balancer does not apply. PyTorch otherwise owns transport selection
        # (all-gather or peer rotation) for the injected process group.
        _cp_options.enable_load_balance = False
        output, logsumexp, *_ = _templated_ring_attention(
            self.process_group,
            2,
            operation,
            query,
            key,
            value,
            is_causal=False,
        )
        return output, logsumexp


# The runtime now supplies both Ulysses and Ring operations. Preserve the old
# public name for extensions that imported it before Ring support was added.
TorchDistributedXDiTSequenceParallelOperations = TorchDistributedUlyssesOperations
