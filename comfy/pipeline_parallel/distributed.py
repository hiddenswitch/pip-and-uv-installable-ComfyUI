from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
import logging
import multiprocessing.connection
import os
import secrets
import socket
import subprocess
import sys
import traceback
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

from comfy.model_management_types import ModelManageableStub

from .runtime import AbstractBasePipelineExecutor, AbstractBasePipelineOperations
from .types import (
    PipelineIntermediateTensors,
    PipelinePartitionPlan,
    PipelineStageMemoryGeometry,
    PipelineWorkerLoadSpec,
    TensorDescriptor,
    deserialize_pipeline_metadata,
    pack_pipeline_value,
    serialize_pipeline_metadata,
    unpack_pipeline_value,
)

logger = logging.getLogger(__name__)


class _PipelineAbort(Exception):
    pass


class AbstractBaseProcessGroupCoordinator(ABC):
    """Rank-local control and tensor transport used by a pipeline executor."""

    @abstractmethod
    def broadcast_command(self, command) -> object:
        raise NotImplementedError

    @abstractmethod
    def send_object(self, value, destination_rank: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def receive_object(self, source_rank: int):
        raise NotImplementedError

    @abstractmethod
    def send_tensors(self, tensors: Mapping[str, torch.Tensor], destination_rank: int) -> Sequence[object]:
        raise NotImplementedError

    @abstractmethod
    def receive_tensors(self, descriptors: Mapping[str, TensorDescriptor], source_rank: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def wait(self, work: Sequence[object]) -> None:
        raise NotImplementedError


class TorchDistributedProcessGroupCoordinator(AbstractBaseProcessGroupCoordinator):
    """Gloo control plane plus an injected torch.distributed device group."""

    def __init__(self, rank: int, device: torch.device, device_group):
        self.rank = rank
        self.device = device
        self.device_group = device_group
        self._buffers: dict[tuple, dict[str, torch.Tensor]] = {}

    def broadcast_command(self, command=None):
        values = [command]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def send_object(self, value, destination_rank: int) -> None:
        dist.send_object_list([value], dst=destination_rank)

    def receive_object(self, source_rank: int):
        values = [None]
        dist.recv_object_list(values, src=source_rank)
        return values[0]

    def send_tensors(self, tensors: Mapping[str, torch.Tensor], destination_rank: int) -> Sequence[object]:
        operations = [
            dist.P2POp(dist.isend, tensor.contiguous(), destination_rank, self.device_group)
            for tensor in tensors.values()
        ]
        return tuple(
            dist.batch_isend_irecv(operations)
            if operations else ()
        )

    def receive_tensors(self, descriptors: Mapping[str, TensorDescriptor], source_rank: int) -> dict[str, torch.Tensor]:
        key = tuple((name, descriptor.shape, descriptor.dtype) for name, descriptor in descriptors.items())
        buffers = self._buffers.get(key)
        if buffers is None:
            buffers = {
                name: torch.empty(descriptor.shape, dtype=descriptor.dtype, device=self.device)
                for name, descriptor in descriptors.items()
            }
            self._buffers[key] = buffers
        operations = [
            dist.P2POp(dist.irecv, tensor, source_rank, self.device_group)
            for tensor in buffers.values()
        ]
        work = tuple(dist.batch_isend_irecv(operations) if operations else ())
        self.wait(work)
        return buffers

    def wait(self, work: Sequence[object]) -> None:
        for item in work:
            item.wait()


def _descriptors(tensors: Mapping[str, torch.Tensor]) -> dict[str, TensorDescriptor]:
    return {
        name: TensorDescriptor(tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size())
        for name, tensor in tensors.items()
    }


def _send_tensor_packet(coordinator, tensors, payload, destination_rank):
    coordinator.send_object(
        {"kind": "tensors", "descriptors": _descriptors(tensors), "payload": payload},
        destination_rank,
    )
    work = coordinator.send_tensors(tensors, destination_rank)
    return tuple(work), tuple(tensors.values())


def _receive_tensor_packet(coordinator, source_rank):
    header = coordinator.receive_object(source_rank)
    if header.get("kind") == "abort":
        raise _PipelineAbort
    if header.get("kind") == "error":
        raise RuntimeError(f"Pipeline rank {source_rank} failed:\n{header['traceback']}")
    if header.get("kind") != "tensors":
        raise RuntimeError(f"Invalid pipeline tensor packet from rank {source_rank}: {header!r}")
    tensors = coordinator.receive_tensors(header["descriptors"], source_rank)
    return tensors, header["payload"]


class TorchDistributedPipelineExecutor(AbstractBasePipelineExecutor):
    def __init__(self, plan, first_stage, coordinator, processes, ready_connections, device_group, worker_geometries):
        self._plan = plan
        self.first_stage = first_stage
        self.coordinator = coordinator
        self.processes = tuple(processes)
        self.ready_connections = tuple(ready_connections)
        self.device_group = device_group
        self.worker_geometries = tuple(worker_geometries)
        self._closed = False

    def remote_stage_command(self, rank, command):
        self.coordinator.broadcast_command({"kind": "stage_model", "rank": rank, "command": command})
        response = self.coordinator.receive_object(rank)
        if response.get("kind") == "error":
            raise RuntimeError(f"Pipeline rank {rank} failed to update model residency:\n{response['traceback']}")
        return response

    @property
    def plan(self) -> PipelinePartitionPlan:
        return self._plan

    def execute(self, *args, **kwargs):
        self.coordinator.broadcast_command({"kind": "execute"})
        try:
            output = self.first_stage(*args, **kwargs)
        except Exception:
            self.coordinator.send_object({"kind": "abort"}, 1)
            raise
        if not isinstance(output, PipelineIntermediateTensors):
            raise TypeError("A non-final pipeline stage must return PipelineIntermediateTensors")
        schema = output.schema()
        schema.validate(output.tensors, output.metadata)
        send_work, send_refs = _send_tensor_packet(
            self.coordinator,
            output.tensors,
            serialize_pipeline_metadata(output.metadata),
            1,
        )
        tensors, structure = _receive_tensor_packet(self.coordinator, self.plan.size - 1)
        self.coordinator.wait(send_work)
        del send_refs
        return unpack_pipeline_value(structure, tensors)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.coordinator.broadcast_command({
            "kind": "add_patches",
            "patches": patches,
            "strength_patch": strength_patch,
            "strength_model": strength_model,
        })
        applied = set()
        for rank in range(1, self.plan.size):
            response = self.coordinator.receive_object(rank)
            if response.get("kind") == "error":
                raise RuntimeError(f"Pipeline rank {rank} failed to apply patches:\n{response['traceback']}")
            applied.update(response["applied"])
        return list(applied)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if dist.is_initialized():
                self.coordinator.broadcast_command({"kind": "close"})
        except Exception:
            logger.debug("Could not send pipeline worker shutdown", exc_info=True)
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if dist.is_initialized():
            try:
                dist.destroy_process_group(self.device_group)
            finally:
                dist.destroy_process_group()

    def __del__(self):
        self.close()


class RemotePipelineStageModel(ModelManageableStub):
    """Model-manager view of a stage whose weights live in another rank."""

    def __init__(self, executor, rank, device, size, dtype, dynamic, ckpt_name=None):
        self.executor = executor
        self.rank = rank
        self.load_device = device
        self.offload_device = torch.device("cpu")
        self.model = torch.nn.Module()
        self.size = int(size)
        self.dtype = dtype
        self.dynamic = dynamic
        self.ckpt_name = ckpt_name
        self._loaded_size = 0
        self._current_device = self.offload_device

    def model_size(self):
        return self.size

    def model_dtype(self):
        return self.dtype

    def model_mmap_residency(self, free=False):
        del free
        return self.size, self.size

    def current_loaded_device(self):
        return self._current_device

    @property
    def current_device(self):
        return self._current_device

    def loaded_size(self):
        return self._loaded_size

    def reclaimable_non_vbar_memory(self):
        return self._loaded_size

    def loaded_ram_size(self):
        return self.size

    def is_dynamic(self):
        return self.dynamic

    def reset_dynamic_buffers(self):
        self.executor.remote_stage_command(self.rank, {"kind": "reset_dynamic_buffers"})

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        del patches, strength_patch, strength_model
        return []

    def model_state_dict(self, filter_prefix=None):
        del filter_prefix
        return {}

    def get_key_patches(self, filter_prefix=None):
        del filter_prefix
        return {}

    def get_additional_models(self):
        return []

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        del load_weights
        self.partially_load(device_to or self.load_device, lowvram_model_memory, force_patch_weights)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        del unpatch_weights
        self.partially_unload(device_to or self.offload_device, self._loaded_size)
        return self.model

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        response = self.executor.remote_stage_command(self.rank, {
            "kind": "load",
            "extra_memory": extra_memory,
            "force_patch_weights": force_patch_weights,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        self._current_device = device_to if self._loaded_size else self.offload_device
        return self._loaded_size - previous

    def partially_unload(self, device_to, memory_to_free=0):
        response = self.executor.remote_stage_command(self.rank, {
            "kind": "unload",
            "memory_to_free": memory_to_free,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        self._current_device = self.load_device if self._loaded_size else device_to
        return previous - self._loaded_size

    def __str__(self):
        return f"<RemotePipelineStage rank={self.rank} device={self.load_device}>"


def _worker_main(
    rank: int,
    world_size: int,
    init_method: str,
    load_spec: PipelineWorkerLoadSpec,
    ready: multiprocessing.connection.Connection,
):
    device = load_spec.plan.stages[rank].device
    try:
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5),
        )
        device_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
        coordinator = TorchDistributedProcessGroupCoordinator(rank, device, device_group)

        os.environ["COMFY_AIMDO_DEVICE_INDICES"] = str(device.index)
        from comfy import model_management

        model_management.set_torch_device(device)
        import comfy.aimdo_integration  # noqa: F401
        from .loader import load_pipeline_worker_stage
        from .memory import ComfyDynamicVRAMStageMemoryEstimator
        from .stages import get_pipeline_stage_spec

        patcher = load_pipeline_worker_stage(load_spec)
        stage = patcher.model.diffusion_model.forward_pipeline_stage
        stage_spec = get_pipeline_stage_spec(load_spec.plan.model_family)
        geometry = ComfyDynamicVRAMStageMemoryEstimator().estimate_stage(
            stage_spec,
            load_spec.plan.stages[rank],
            patcher,
        )
        ready.send({"kind": "ready", "geometry": geometry})

        while True:
            command = coordinator.broadcast_command()
            if command["kind"] == "close":
                break
            if command["kind"] == "stage_model":
                if command["rank"] != rank:
                    continue
                try:
                    stage_command = command["command"]
                    if stage_command["kind"] == "load":
                        patcher.partially_load(
                            device,
                            stage_command["extra_memory"],
                            force_patch_weights=stage_command["force_patch_weights"],
                        )
                    elif stage_command["kind"] == "unload":
                        patcher.partially_unload(
                            patcher.offload_device,
                            stage_command["memory_to_free"],
                        )
                    elif stage_command["kind"] == "reset_dynamic_buffers":
                        patcher.reset_dynamic_buffers()
                    else:
                        raise RuntimeError(f"Unknown pipeline stage model command: {stage_command!r}")
                    coordinator.send_object({"kind": "stage_model", "loaded_size": patcher.loaded_size()}, 0)
                except Exception:
                    coordinator.send_object({"kind": "error", "traceback": traceback.format_exc()}, 0)
                continue
            if command["kind"] == "add_patches":
                try:
                    applied = patcher.add_patches(
                        command["patches"],
                        command["strength_patch"],
                        command["strength_model"],
                    )
                    coordinator.send_object({"kind": "patches", "applied": applied}, 0)
                except Exception:
                    coordinator.send_object({"kind": "error", "traceback": traceback.format_exc()}, 0)
                continue
            if command["kind"] != "execute":
                raise RuntimeError(f"Unknown pipeline worker command: {command!r}")

            try:
                tensors, metadata = _receive_tensor_packet(coordinator, rank - 1)
                intermediate = PipelineIntermediateTensors(
                    dict(tensors),
                    deserialize_pipeline_metadata(metadata),
                )
                output = stage(intermediate)
                if rank < world_size - 1:
                    if not isinstance(output, PipelineIntermediateTensors):
                        raise TypeError("A non-final pipeline stage must return PipelineIntermediateTensors")
                    work, refs = _send_tensor_packet(
                        coordinator,
                        output.tensors,
                        serialize_pipeline_metadata(output.metadata),
                        rank + 1,
                    )
                else:
                    tensors = {}
                    structure = pack_pipeline_value(output, tensors, "pipeline_output")
                    work, refs = _send_tensor_packet(coordinator, tensors, structure, 0)
                coordinator.wait(work)
                del refs
            except _PipelineAbort:
                if rank < world_size - 1:
                    coordinator.send_object({"kind": "abort"}, rank + 1)
                continue
            except Exception:
                destination_rank = rank + 1 if rank < world_size - 1 else 0
                coordinator.send_object({"kind": "error", "traceback": traceback.format_exc()}, destination_rank)
    except Exception:
        try:
            ready.send({"kind": "error", "traceback": traceback.format_exc()})
        except Exception:
            pass
        raise
    finally:
        ready.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_tcp_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


class TorchDistributedPipelineOperations(AbstractBasePipelineOperations):
    @property
    def uses_worker_processes(self) -> bool:
        return True

    def supports(self, devices) -> bool:
        return (
            len(devices) >= 2
            and all(device.type == "cuda" for device in devices)
            and dist.is_available()
            and dist.is_nccl_available()
            and not dist.is_initialized()
        )

    def create_transport(self):
        raise RuntimeError("Distributed pipeline transport is owned by its rank coordinator")

    def create_stage_runner(self):
        raise RuntimeError("Distributed pipeline stages run in rank workers")

    def create_executor(self, plan, stages, worker_load_specs=()):
        if len(worker_load_specs) != plan.size - 1:
            raise ValueError("Distributed pipeline execution requires one load specification per worker rank")
        init_method = _free_tcp_init_method()
        listener_authkey = secrets.token_bytes(32)
        listener = multiprocessing.connection.Listener(("127.0.0.1", 0), authkey=listener_authkey)
        listener_host, listener_port = listener.address
        processes = []
        ready_connections = [None] * (plan.size - 1)
        for spec in worker_load_specs:
            process = subprocess.Popen(
                (
                    sys.executable,
                    "-m",
                    "comfy.pipeline_parallel.worker",
                    str(listener_host),
                    str(listener_port),
                    listener_authkey.hex(),
                    str(spec.stage_index),
                ),
                cwd=os.getcwd(),
            )
            processes.append(process)

        try:
            for _ in worker_load_specs:
                connection = listener.accept()
                rank = connection.recv()
                ready_connections[rank - 1] = connection
                connection.send((plan.size, init_method, worker_load_specs[rank - 1]))
            listener.close()
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=0,
                world_size=plan.size,
                timeout=timedelta(minutes=5),
            )
            device_group = dist.new_group(ranks=list(range(plan.size)), backend="nccl")
            coordinator = TorchDistributedProcessGroupCoordinator(0, plan.stages[0].device, device_group)
            worker_geometries: list[PipelineStageMemoryGeometry] = []
            for rank, connection in enumerate(ready_connections, start=1):
                if not connection.poll(300):
                    raise TimeoutError(f"Pipeline rank {rank} did not finish loading within five minutes")
                response = connection.recv()
                if response.get("kind") != "ready":
                    raise RuntimeError(f"Pipeline rank {rank} failed to start:\n{response['traceback']}")
                worker_geometries.append(response["geometry"])
            return TorchDistributedPipelineExecutor(
                plan,
                stages[0],
                coordinator,
                processes,
                ready_connections,
                device_group,
                worker_geometries,
            )
        except Exception:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if dist.is_initialized():
                dist.destroy_process_group()
            raise
        finally:
            listener.close()
