from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from comfy import model_management

from .runtime import AbstractBaseDeviceRuntime, AbstractBasePendingPipelineIntermediate, AbstractBasePipelineBufferPool, AbstractBasePipelineOperations, AbstractBasePipelineStageRunner, AbstractBasePipelineTransport, DevicePipelineBufferPool, SingleProcessPipelineExecutor
from .types import PipelineIntermediateSchema, PipelineIntermediateTensors, deserialize_pipeline_metadata, serialize_pipeline_metadata


@dataclass
class _CudaCopyCompletion:
    event: torch.cuda.Event
    source: torch.Tensor


class CudaPendingPipelineIntermediate(AbstractBasePendingPipelineIntermediate):
    def __init__(self, runtime, tensors, metadata, completions, destination):
        self.runtime = runtime
        self.tensors = tensors
        self.metadata = metadata
        self.completions = completions
        self.destination = destination

    def wait(self) -> PipelineIntermediateTensors:
        for completion in self.completions:
            self.runtime.wait(completion, self.destination)
        result = PipelineIntermediateTensors(dict(self.tensors), self.metadata)
        self.completions = ()
        return result


class CudaDeviceRuntime(AbstractBaseDeviceRuntime):
    def __init__(self):
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}

    def supports(self, device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.is_available()

    def can_transfer(self, source: torch.device, destination: torch.device) -> bool:
        if not self.supports(source) or not self.supports(destination):
            return False
        if source == destination:
            return True
        return torch.cuda.can_device_access_peer(source.index, destination.index)

    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.supports(device):
            raise ValueError(f"CudaDeviceRuntime does not support {device}")
        return torch.empty(shape, dtype=dtype, device=device)

    def copy(self, source: torch.Tensor, destination: torch.Tensor) -> _CudaCopyCompletion:
        if not self.can_transfer(source.device, destination.device):
            raise RuntimeError(f"Direct CUDA peer transfer is unavailable from {source.device} to {destination.device}")
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("CUDA peer transfer source and destination must have matching shape and dtype")

        source_ready = torch.cuda.Event()
        source_ready.record(torch.cuda.current_stream(source.device))
        stream = self._copy_streams.get(destination.device)
        if stream is None:
            stream = torch.cuda.Stream(device=destination.device)
            self._copy_streams[destination.device] = stream
        with torch.cuda.device(destination.device), torch.cuda.stream(stream):
            stream.wait_event(source_ready)
            destination.copy_(source, non_blocking=True)
            complete = stream.record_event()
        return _CudaCopyCompletion(complete, source)

    def wait(self, completion: _CudaCopyCompletion, device: torch.device) -> None:
        torch.cuda.current_stream(device).wait_event(completion.event)

    def memory_available(self, device: torch.device) -> int:
        return int(model_management.get_free_memory(device))


class CudaPeerPipelineTransport(AbstractBasePipelineTransport):
    def __init__(self, runtime: CudaDeviceRuntime, buffers: AbstractBasePipelineBufferPool):
        self.runtime = runtime
        self.buffers = buffers

    def transfer(
        self,
        tensors: Mapping[str, torch.Tensor],
        schema: PipelineIntermediateSchema,
        destination: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        output, completions = self._begin_tensor_transfer(tensors, schema, destination)
        for completion in completions:
            self.runtime.wait(completion, destination)
        return output

    def _begin_tensor_transfer(self, tensors, schema, destination):
        output = self.buffers.acquire(schema, destination)
        completions = tuple(
            self.runtime.copy(tensor, output[name])
            for name, tensor in tensors.items()
        )
        return output, completions

    def begin_transfer(self, intermediate, destination):
        schema = intermediate.schema()
        schema.validate(intermediate.tensors, intermediate.metadata)
        output, completions = self._begin_tensor_transfer(
            intermediate.tensors,
            schema,
            destination,
        )
        metadata = deserialize_pipeline_metadata(
            serialize_pipeline_metadata(intermediate.metadata)
        )
        return CudaPendingPipelineIntermediate(
            self.runtime,
            output,
            metadata,
            completions,
            destination,
        )

    def close(self) -> None:
        self.buffers.clear()

    def transfer_metadata(self, metadata, destination: torch.device) -> dict[str, object]:
        del destination
        return deserialize_pipeline_metadata(serialize_pipeline_metadata(metadata))


class CudaPipelineStageRunner(AbstractBasePipelineStageRunner):
    def run(self, stage, device: torch.device, *args, **kwargs):
        with torch.cuda.device(device):
            return stage(*args, **kwargs)


class CudaPeerPipelineOperations(AbstractBasePipelineOperations):
    @property
    def uses_worker_processes(self) -> bool:
        return False

    def supports(self, devices) -> bool:
        runtime = CudaDeviceRuntime()
        return all(runtime.supports(device) for device in devices) and all(
            runtime.can_transfer(source, destination)
            for source, destination in zip(devices, devices[1:])
        )

    def create_transport(self) -> AbstractBasePipelineTransport:
        runtime = CudaDeviceRuntime()
        return CudaPeerPipelineTransport(runtime, DevicePipelineBufferPool(runtime))

    def create_stage_runner(self) -> AbstractBasePipelineStageRunner:
        return CudaPipelineStageRunner()

    def create_executor(self, plan, stages, worker_load_specs=()):
        del worker_load_specs
        return SingleProcessPipelineExecutor(
            plan,
            stages,
            self.create_transport(),
            self.create_stage_runner(),
        )
