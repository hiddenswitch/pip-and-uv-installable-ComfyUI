from __future__ import annotations

from collections.abc import Sequence

import torch

from .cuda import CudaPeerPipelineOperations
from .distributed import TorchDistributedPipelineOperations
from .runtime import AbstractBasePipelineOperations


class PipelineOperationsMux:
    def __init__(self, providers: Sequence[AbstractBasePipelineOperations]):
        self.providers = tuple(providers)

    def select(self, devices: Sequence[torch.device]) -> AbstractBasePipelineOperations:
        for provider in self.providers:
            if provider.supports(devices):
                return provider
        raise RuntimeError(f"No registered pipeline operations support devices {devices}")


def select_pipeline_operations(devices, operations=None):
    if operations is not None:
        if not operations.supports(devices):
            raise RuntimeError(f"The selected pipeline operations do not support devices {devices}")
        return operations
    return PipelineOperationsMux((
        CudaPeerPipelineOperations(),
        TorchDistributedPipelineOperations(),
    )).select(devices)
