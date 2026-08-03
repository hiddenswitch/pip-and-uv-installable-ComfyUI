from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
import gc
import logging
import traceback

import torch
import torch.distributed as dist

from ..distributed.config import DistributedConfiguration

from .distributed import (
    TorchDistributedPipelineExecutor,
    TorchDistributedProcessGroupCoordinator,
    _load_worker_stage,
    _run_worker_commands,
)
from .runtime import AbstractBasePipelineOperations


logger = logging.getLogger(__name__)
_runtime: ExternalTorchDistributedRuntime | None = None


class AbstractBaseExternalPipelineRuntime(ABC):
    @property
    @abstractmethod
    def configuration(self) -> DistributedConfiguration:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_driver(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run_worker_service(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_executor(self, plan, first_stage, worker_load_specs):
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ExternalTorchDistributedRuntime(AbstractBaseExternalPipelineRuntime):
    """Long-lived process groups and rank service created by torchrun et al."""

    def __init__(self, configuration: DistributedConfiguration):
        if not configuration.externally_launched:
            raise ValueError("External runtime requires a launcher-provided process identity")
        if configuration.local_world_size != configuration.world_size:
            raise NotImplementedError(
                "External pipeline launching currently supports one node; "
                "LOCAL_WORLD_SIZE must equal WORLD_SIZE"
            )
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise RuntimeError("External pipeline launching requires CUDA and NCCL")

        self._configuration = configuration
        self._device = torch.device("cuda", configuration.local_rank)
        torch.cuda.set_device(self.device)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                rank=configuration.rank,
                world_size=configuration.world_size,
                timeout=timedelta(minutes=5),
            )
        elif (
            dist.get_rank() != configuration.rank
            or dist.get_world_size() != configuration.world_size
        ):
            raise RuntimeError(
                "Existing torch.distributed process group does not match canonical configuration"
            )
        self.device_group = dist.new_group(
            ranks=list(range(configuration.world_size)), backend="nccl"
        )
        self.coordinator = TorchDistributedProcessGroupCoordinator(
            configuration.rank,
            self.device,
            self.device_group,
        )
        self._closed = False

        from .. import model_management

        model_management.set_torch_device(self.device)
        logger.info(
            "Initialized external pipeline rank %d/%d on %s",
            configuration.rank,
            configuration.world_size,
            self.device,
        )

    @property
    def configuration(self) -> DistributedConfiguration:
        return self._configuration

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_driver(self) -> bool:
        return self.configuration.is_first_pipeline_stage

    def run_worker_service(self) -> None:
        if self.is_driver:
            raise RuntimeError("Pipeline rank zero is the ComfyUI driver, not a worker")
        rank = self.configuration.pipeline_rank
        while True:
            command = self.coordinator.broadcast_command()
            if command["kind"] == "close":
                break
            if command["kind"] != "load_pipeline":
                raise RuntimeError(
                    f"External pipeline rank {rank} expected a load command, got {command!r}"
                )
            try:
                load_specs = command["worker_load_specs"]
                load_spec = load_specs[rank - 1]
                if load_spec.stage_index != rank:
                    raise RuntimeError(
                        f"External pipeline rank {rank} received stage {load_spec.stage_index}"
                    )
                patcher, geometry = _load_worker_stage(load_spec, rank)
                self.coordinator.send_object(
                    {"kind": "ready", "geometry": geometry}, 0
                )
                result = _run_worker_commands(
                    self.coordinator,
                    rank,
                    self.configuration.world_size,
                    self.device,
                    patcher,
                )
                del patcher
                gc.collect()
                if result == "close":
                    break
            except Exception:
                self.coordinator.send_object(
                    {"kind": "error", "traceback": traceback.format_exc()}, 0
                )
                raise
        self.close()

    def load_executor(self, plan, first_stage, worker_load_specs):
        if not self.is_driver:
            raise RuntimeError("Only pipeline rank zero can create the driver executor")
        if plan.size != self.configuration.pipeline_parallel_size:
            raise ValueError(
                f"Plan size {plan.size} does not match launched pipeline size "
                f"{self.configuration.pipeline_parallel_size}"
            )
        self.coordinator.broadcast_command(
            {
                "kind": "load_pipeline",
                "worker_load_specs": tuple(worker_load_specs),
            }
        )
        worker_geometries = []
        for rank in range(1, plan.size):
            response = self.coordinator.receive_object(rank)
            if response.get("kind") != "ready":
                raise RuntimeError(
                    f"External pipeline rank {rank} failed to load:\n{response['traceback']}"
                )
            worker_geometries.append(response["geometry"])
        return TorchDistributedPipelineExecutor(
            plan,
            first_stage,
            self.coordinator,
            (),
            (),
            self.device_group,
            worker_geometries,
            close_command="unload_pipeline",
            destroy_process_group=False,
        )

    def close(self) -> None:
        global _runtime
        if self._closed:
            return
        self._closed = True
        if self.is_driver and dist.is_initialized():
            try:
                self.coordinator.broadcast_command({"kind": "close"})
            except Exception:
                logger.debug("Could not stop external pipeline ranks", exc_info=True)
        if dist.is_initialized():
            try:
                dist.destroy_process_group(self.device_group)
            finally:
                dist.destroy_process_group()
        if _runtime is self:
            _runtime = None


class ExternalTorchDistributedPipelineOperations(AbstractBasePipelineOperations):
    def __init__(self, runtime: AbstractBaseExternalPipelineRuntime):
        self.runtime = runtime

    @property
    def uses_worker_processes(self) -> bool:
        return True

    def supports(self, devices) -> bool:
        return (
            self.runtime.is_driver
            and len(devices) == self.runtime.configuration.pipeline_parallel_size
            and all(device.type == "cuda" for device in devices)
        )

    def create_transport(self):
        raise RuntimeError("External pipeline transport is owned by its rank runtime")

    def create_stage_runner(self):
        raise RuntimeError("External pipeline stages run in launcher processes")

    def create_executor(self, plan, stages, worker_load_specs=()):
        if len(stages) != 1:
            raise ValueError("External pipeline driver loads only its first stage")
        return self.runtime.load_executor(plan, stages[0], worker_load_specs)


def initialize_external_pipeline_runtime(
    configuration: DistributedConfiguration,
) -> ExternalTorchDistributedRuntime | None:
    global _runtime
    if not configuration.externally_launched:
        return None
    if _runtime is None:
        _runtime = ExternalTorchDistributedRuntime(configuration)
    elif _runtime.configuration != configuration:
        raise RuntimeError("External pipeline runtime was already initialized differently")
    return _runtime


def get_external_pipeline_runtime() -> ExternalTorchDistributedRuntime | None:
    return _runtime
