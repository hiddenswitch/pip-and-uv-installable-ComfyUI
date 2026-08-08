from __future__ import annotations

from dataclasses import dataclass

from .runtime import AbstractBaseXDiTSequenceParallelOperations


@dataclass(frozen=True)
class XDiTSequenceParallelConfig:
    operations: AbstractBaseXDiTSequenceParallelOperations
    strategy: str = "ulysses"

    def __post_init__(self) -> None:
        if self.strategy not in ("ulysses", "ring"):
            raise ValueError(f"Unknown xDiT sequence-parallel strategy {self.strategy!r}")

    @property
    def rank(self) -> int:
        return self.operations.rank

    @property
    def size(self) -> int:
        return self.operations.world_size
