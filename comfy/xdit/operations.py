from __future__ import annotations

from .types import XDiTSequenceParallelConfig


def xdit_sequence_parallel_operations(
    base_operations,
    parallel: XDiTSequenceParallelConfig,
):
    """Decorate Comfy operations with an injected xDiT sequence runtime."""

    class XDiTSequenceParallelOperations(base_operations):
        xdit_sequence_parallel = parallel

    XDiTSequenceParallelOperations.__name__ = (
        f"XDiTSequenceParallel{base_operations.__name__}"
    )
    return XDiTSequenceParallelOperations
