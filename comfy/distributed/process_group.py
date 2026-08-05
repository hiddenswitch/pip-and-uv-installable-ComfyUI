from __future__ import annotations

from abc import ABC, abstractmethod
import os
import threading

import torch
import torch.distributed as dist


NCCL_PROTOCOLS = ("auto", "simple", "ll", "ll128")
_NCCL_ENVIRONMENT_LOCK = threading.Lock()


class AbstractBaseDeviceProcessGroupFactory(ABC):
    @abstractmethod
    def create(self, ranks, device: torch.device, nccl_proto: str):
        raise NotImplementedError


class TorchDistributedCudaProcessGroupFactory(
    AbstractBaseDeviceProcessGroupFactory
):
    """Create an eagerly initialized NCCL group with injected tuning."""

    def create(self, ranks, device: torch.device, nccl_proto: str):
        if device.type != "cuda":
            raise ValueError(f"NCCL process groups require a CUDA device, got {device}")
        normalized = nccl_proto.lower()
        if normalized not in NCCL_PROTOCOLS:
            raise ValueError(
                f"Unknown NCCL protocol {nccl_proto!r}; expected one of "
                f"{', '.join(NCCL_PROTOCOLS)}"
            )

        # NCCL protocol selection is not exposed by ncclConfig_t or PyTorch's
        # NCCLConfig. device_id makes new_group initialize the communicator
        # eagerly, so the process environment can remain an implementation
        # detail rather than application configuration.
        with _NCCL_ENVIRONMENT_LOCK:
            previous = os.environ.get("NCCL_PROTO")
            try:
                if normalized == "auto":
                    os.environ.pop("NCCL_PROTO", None)
                else:
                    os.environ["NCCL_PROTO"] = normalized.upper()
                return dist.new_group(
                    ranks=list(ranks),
                    backend="nccl",
                    device_id=device,
                )
            finally:
                if previous is None:
                    os.environ.pop("NCCL_PROTO", None)
                else:
                    os.environ["NCCL_PROTO"] = previous


def create_device_process_group(
    ranks,
    device: torch.device,
    configuration=None,
    factory: AbstractBaseDeviceProcessGroupFactory | None = None,
):
    if configuration is None:
        from ..execution_context import current_execution_context

        configuration = current_execution_context().configuration
    return (factory or TorchDistributedCudaProcessGroupFactory()).create(
        ranks,
        device,
        configuration.nccl_proto,
    )
