from __future__ import annotations

from collections.abc import Sequence

import torch

from comfy.distributed.config import resolve_distributed_configuration
from comfy.execution_context import current_execution_context

from .cuda import CudaPeerPipelineOperations
from .distributed import TorchDistributedPipelineOperations
from .external import (
    ExternalTorchDistributedPipelineOperations,
    get_external_pipeline_runtime,
)
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
    distributed = resolve_distributed_configuration(
        current_execution_context().configuration
    )
    if distributed.externally_launched:
        runtime = get_external_pipeline_runtime()
        if runtime is None:
            raise RuntimeError("External-launcher pipeline runtime has not been initialized")
        selected = ExternalTorchDistributedPipelineOperations(runtime)
        if not selected.supports(devices):
            raise RuntimeError(
                f"External-launcher pipeline operations do not support devices {devices}"
            )
        return selected
    if distributed.executor_backend == "peer":
        selected = CudaPeerPipelineOperations()
        if not selected.supports(devices):
            raise RuntimeError(f"CUDA peer pipeline operations do not support devices {devices}")
        return selected
    if distributed.executor_backend == "mp":
        selected = TorchDistributedPipelineOperations()
        if not selected.supports(devices):
            raise RuntimeError(f"Multiprocess pipeline operations do not support devices {devices}")
        return selected
    if distributed.executor_backend == "external_launcher":
        raise RuntimeError("External launcher requires WORLD_SIZE greater than one")
    return PipelineOperationsMux((
        CudaPeerPipelineOperations(),
        TorchDistributedPipelineOperations(),
    )).select(devices)
