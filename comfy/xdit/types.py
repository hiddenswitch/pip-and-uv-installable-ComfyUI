from __future__ import annotations

from dataclasses import dataclass

from .runtime import AbstractBaseXDiTSequenceParallelOperations


@dataclass(frozen=True)
class XDiTSequenceParallelConfig:
    operations: AbstractBaseXDiTSequenceParallelOperations

    @property
    def rank(self) -> int:
        return self.operations.rank

    @property
    def size(self) -> int:
        return self.operations.world_size
