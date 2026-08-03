from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import torch

from .types import PipelineIntermediateSchema, PipelineIntermediateTensors, PipelinePartitionPlan, pack_pipeline_value, unpack_pipeline_value


class AbstractBaseDeviceRuntime(ABC):
    @abstractmethod
    def supports(self, device: torch.device) -> bool:
        raise NotImplementedError

    @abstractmethod
    def can_transfer(self, source: torch.device, destination: torch.device) -> bool:
        raise NotImplementedError

    @abstractmethod
    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def copy(self, source: torch.Tensor, destination: torch.Tensor) -> object:
        raise NotImplementedError

    @abstractmethod
    def wait(self, completion: object, device: torch.device) -> None:
        raise NotImplementedError

    @abstractmethod
    def memory_available(self, device: torch.device) -> int:
        raise NotImplementedError


class AbstractBasePipelineBufferPool(ABC):
    @abstractmethod
    def acquire(self, schema: PipelineIntermediateSchema, device: torch.device) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class AbstractBasePipelineTransport(ABC):
    @abstractmethod
    def transfer(
        self,
        tensors: Mapping[str, torch.Tensor],
        schema: PipelineIntermediateSchema,
        destination: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def transfer_metadata(self, metadata: Mapping[str, object], destination: torch.device) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class AbstractBasePipelineExecutor(ABC):
    @property
    @abstractmethod
    def plan(self) -> PipelinePartitionPlan:
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class AbstractBasePipelineStageRunner(ABC):
    @abstractmethod
    def run(self, stage, device: torch.device, *args, **kwargs):
        raise NotImplementedError


class AbstractBasePipelineOperations(ABC):
    @property
    @abstractmethod
    def uses_worker_processes(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports(self, devices: Sequence[torch.device]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_transport(self) -> AbstractBasePipelineTransport:
        raise NotImplementedError

    @abstractmethod
    def create_stage_runner(self) -> AbstractBasePipelineStageRunner:
        raise NotImplementedError

    @abstractmethod
    def create_executor(self, plan, stages, worker_load_specs=()) -> AbstractBasePipelineExecutor:
        raise NotImplementedError


class DirectPipelineStageRunner(AbstractBasePipelineStageRunner):
    def run(self, stage, device: torch.device, *args, **kwargs):
        return stage(*args, **kwargs)


class DevicePipelineBufferPool(AbstractBasePipelineBufferPool):
    def __init__(self, runtime: AbstractBaseDeviceRuntime):
        self.runtime = runtime
        self._buffers: dict[tuple, dict[str, torch.Tensor]] = {}

    def acquire(self, schema: PipelineIntermediateSchema, device: torch.device) -> Mapping[str, torch.Tensor]:
        key = (
            device,
            tuple((name, descriptor.shape, descriptor.dtype) for name, descriptor in schema.tensors.items()),
        )
        buffers = self._buffers.get(key)
        if buffers is None:
            buffers = {
                name: self.runtime.allocate(descriptor.shape, descriptor.dtype, device)
                for name, descriptor in schema.tensors.items()
            }
            self._buffers[key] = buffers
        return buffers

    def clear(self) -> None:
        self._buffers.clear()


class SingleProcessPipelineExecutor(AbstractBasePipelineExecutor):
    def __init__(
        self,
        plan: PipelinePartitionPlan,
        stages: Sequence[object],
        transport: AbstractBasePipelineTransport,
        stage_runner: AbstractBasePipelineStageRunner | None = None,
    ):
        if len(stages) != plan.size:
            raise ValueError("Pipeline executor requires one stage implementation per stage plan")
        self._plan = plan
        self.stages = tuple(stages)
        self.transport = transport
        self.stage_runner = stage_runner or DirectPipelineStageRunner()

    @property
    def plan(self) -> PipelinePartitionPlan:
        return self._plan

    def execute(self, *args, **kwargs):
        output = self.stage_runner.run(self.stages[0], self.plan.stages[0].device, *args, **kwargs)
        for stage_plan, stage in zip(self.plan.stages[1:], self.stages[1:], strict=True):
            if not isinstance(output, PipelineIntermediateTensors):
                raise TypeError("A non-final pipeline stage must return PipelineIntermediateTensors")
            schema = output.schema()
            schema.validate(output.tensors, output.metadata)
            tensors = self.transport.transfer(output.tensors, schema, stage_plan.device)
            metadata = self.transport.transfer_metadata(output.metadata, stage_plan.device)
            output = self.stage_runner.run(
                stage,
                stage_plan.device,
                PipelineIntermediateTensors(dict(tensors), metadata),
            )
        if isinstance(output, PipelineIntermediateTensors):
            raise TypeError("The final pipeline stage returned intermediate tensors")
        tensors = {}
        structure = pack_pipeline_value(output, tensors, "pipeline_output")
        if not tensors:
            return output
        schema = PipelineIntermediateTensors(tensors, {}).schema()
        tensors = self.transport.transfer(tensors, schema, self.plan.stages[0].device)
        output_metadata = self.transport.transfer_metadata(
            {"structure": structure},
            self.plan.stages[0].device,
        )
        return unpack_pipeline_value(output_metadata["structure"], tensors)

    def close(self) -> None:
        self.transport.close()
