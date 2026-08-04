from __future__ import annotations

from abc import ABC, abstractmethod

import torch


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
    def _complete(completion, tensor: torch.Tensor) -> None:
        if tensor.device.type == "cuda":
            completion.block_current_stream()
        else:
            completion.wait()

    def all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        output = torch.empty_like(tensor)
        completion = dist.all_to_all_single(
            output,
            tensor,
            group=self.process_group,
            async_op=True,
        )
        self._complete(completion, output)
        return output

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        import torch.distributed as dist

        outputs = [torch.empty_like(tensor) for _ in range(self.world_size)]
        completion = dist.all_gather(
            outputs,
            tensor.contiguous(),
            group=self.process_group,
            async_op=True,
        )
        self._complete(completion, tensor)
        return torch.cat(outputs, dim=dim)
