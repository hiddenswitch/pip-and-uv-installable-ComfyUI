from __future__ import annotations

from dataclasses import dataclass

from .runtime import AbstractBaseTensorParallelOperations


@dataclass(frozen=True)
class TensorParallelConfig:
    operations: AbstractBaseTensorParallelOperations

    @property
    def rank(self) -> int:
        return self.operations.rank

    @property
    def size(self) -> int:
        return self.operations.world_size
