from __future__ import annotations

from abc import ABC, abstractmethod

from .config import DistributedConfiguration


class AbstractBaseDistributedRuntimeBootstrap(ABC):
    @abstractmethod
    def select_device(self, configuration: DistributedConfiguration) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, configuration: DistributedConfiguration):
        raise NotImplementedError


class CudaDistributedRuntimeBootstrap(AbstractBaseDistributedRuntimeBootstrap):
    """Select the rank device before pipeline modules initialize device state."""

    def select_device(self, configuration: DistributedConfiguration) -> None:
        if not configuration.externally_launched:
            return

        import torch

        torch.cuda.set_device(configuration.local_rank)

    def initialize(self, configuration: DistributedConfiguration):
        if not configuration.externally_launched:
            return None

        from ..pipeline_parallel.external import initialize_external_pipeline_runtime

        return initialize_external_pipeline_runtime(configuration)


def initialize_distributed_runtime(
    configuration: DistributedConfiguration,
    bootstrap: AbstractBaseDistributedRuntimeBootstrap | None = None,
):
    return (bootstrap or CudaDistributedRuntimeBootstrap()).initialize(configuration)


def select_distributed_device(
    configuration: DistributedConfiguration,
    bootstrap: AbstractBaseDistributedRuntimeBootstrap | None = None,
) -> None:
    (bootstrap or CudaDistributedRuntimeBootstrap()).select_device(configuration)
