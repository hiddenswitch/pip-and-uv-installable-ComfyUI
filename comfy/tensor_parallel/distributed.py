from __future__ import annotations

import concurrent.futures
from datetime import timedelta
import gc
import multiprocessing.connection
import os
import secrets
import socket
import traceback

import torch
import torch.distributed as dist

from ..distributed.config import DistributedConfiguration, resolve_distributed_configuration
from ..distributed.device import accelerator_device_provider
from ..distributed.executors import ContextVarProcessPoolExecutor
from ..distributed.tracing import distributed_command_span, inject_trace_context
from ..model_management_types import ModelManageableStub
from ..pipeline_parallel.types import TensorDescriptor, pack_pipeline_value, unpack_pipeline_value
from .runtime import TorchDistributedTensorParallelOperations


def _free_tcp_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _without_execution_caches(value):
    if isinstance(value, dict):
        return {
            key: _without_execution_caches(item)
            for key, item in value.items()
            if key != "layout"
        }
    if isinstance(value, list):
        return [_without_execution_caches(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_execution_caches(item) for item in value)
    return value


def _broadcast_tensors(operations, tensors=None, descriptors=None):
    if descriptors is None:
        prepared = {
            name: tensor.to(
                device=tensor.device if tensor.device.type == "cpu" else operations.device,
                non_blocking=True,
            ).contiguous()
            for name, tensor in tensors.items()
        }
    else:
        prepared = {
            name: torch.empty(
                descriptor.shape,
                dtype=descriptor.dtype,
                device="cpu" if descriptor.device_type == "cpu" else operations.device,
            )
            for name, descriptor in descriptors.items()
        }
    completions = [
        (tensor, dist.broadcast(
            tensor,
            src=0,
            group=(
                operations.control_process_group
                if tensor.device.type == "cpu"
                else operations.process_group
            ),
            async_op=True,
        ))
        for tensor in prepared.values()
    ]
    for tensor, completion in completions:
        if tensor.device.type == "cpu":
            completion.wait()
        else:
            completion.block_current_stream()
    return prepared


class TorchDistributedTensorParallelExecutor:
    def __init__(self, root_model, operations, coordinator, workers, connections, devices, rank_sizes):
        self.root_model = root_model
        self.operations = operations
        self.coordinator = coordinator
        self.workers = tuple(workers)
        self.connections = tuple(connections)
        self.devices = tuple(devices)
        self.rank_sizes = tuple(rank_sizes)
        self._closed = False

    def execute(self, *args, **kwargs):
        return self.execute_method("forward", *args, **kwargs)

    def execute_method(self, method, *args, **kwargs):
        tensors = {}
        structure = pack_pipeline_value(
            _without_execution_caches((args, kwargs)), tensors, "tensor_parallel_input"
        )
        descriptors = {
            name: TensorDescriptor(
                tuple(tensor.shape), tensor.dtype, tensor.nbytes, tensor.device.type
            )
            for name, tensor in tensors.items()
        }
        self.coordinator.broadcast_command({
            "kind": "execute",
            "method": method,
            "structure": structure,
            "descriptors": descriptors,
        })
        tensors = _broadcast_tensors(self.operations, tensors=tensors)
        args, kwargs = unpack_pipeline_value(structure, tensors)
        output = getattr(self.root_model, method)(*args, **kwargs)
        for rank in range(1, len(self.devices)):
            response = self.coordinator.receive_object(rank)
            if response["kind"] == "error":
                raise RuntimeError(f"Tensor-parallel rank {rank} failed:\n{response['traceback']}")
        return output

    def remote_rank_command(self, rank, command):
        self.coordinator.broadcast_command({"kind": "rank_model", "rank": rank, "command": command})
        return self.coordinator.receive_object(rank)

    def finish_execution(self):
        self.coordinator.broadcast_command({"kind": "finish_execution"})
        from .. import model_prefetch

        model_prefetch.finish_model_execution()
        for rank in range(1, len(self.devices)):
            response = self.coordinator.receive_object(rank)
            if response["kind"] == "error":
                raise RuntimeError(
                    f"Tensor-parallel rank {rank} failed to finish execution:\n"
                    f"{response['traceback']}"
                )

    def close(self):
        if self._closed:
            return
        self._closed = True
        if dist.is_initialized():
            self.coordinator.broadcast_command({"kind": "close"})
        for worker_pool, worker_future in self.workers:
            try:
                worker_future.result(timeout=10)
            except concurrent.futures.TimeoutError:
                worker_pool.shutdown(wait=False)
            except Exception:
                worker_pool.shutdown(wait=True)
            else:
                worker_pool.shutdown(wait=True)
        if dist.is_initialized():
            try:
                dist.destroy_process_group(self.operations.process_group)
            finally:
                dist.destroy_process_group()

    def __del__(self):
        self.close()


class RemoteTensorParallelRankModel(ModelManageableStub):
    def __init__(self, executor, rank, device, size, dtype, dynamic, ckpt_name=None):
        self.executor = executor
        self.rank = rank
        self.load_device = device
        self.offload_device = torch.device("cpu")
        self.model = torch.nn.Module()
        self.size = size
        self.dtype = dtype
        self.dynamic = dynamic
        self.ckpt_name = ckpt_name
        self._loaded_size = 0

    def model_size(self):
        return self.size

    def model_dtype(self):
        return self.dtype

    def model_mmap_residency(self, free=False):
        del free
        return self.size, self.size

    def current_loaded_device(self):
        return self.load_device if self._loaded_size else self.offload_device

    @property
    def current_device(self):
        return self.current_loaded_device()

    def loaded_size(self):
        return self._loaded_size

    def reclaimable_non_vbar_memory(self):
        return self._loaded_size

    def loaded_ram_size(self):
        return self.size

    def is_dynamic(self):
        return self.dynamic

    def get_additional_models(self):
        return []

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        del patches, strength_patch, strength_model
        raise NotImplementedError("Tensor-parallel LoRA loading is not implemented")

    def model_state_dict(self, filter_prefix=None):
        del filter_prefix
        return {}

    def get_key_patches(self, filter_prefix=None):
        del filter_prefix
        return {}

    def reset_dynamic_buffers(self):
        self.executor.remote_rank_command(self.rank, {"kind": "reset_dynamic_buffers"})

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        del load_weights
        self.partially_load(device_to or self.load_device, lowvram_model_memory, force_patch_weights)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        del unpatch_weights
        self.partially_unload(device_to or self.offload_device, self._loaded_size)
        return self.model

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        response = self.executor.remote_rank_command(self.rank, {
            "kind": "load",
            "extra_memory": extra_memory,
            "force_patch_weights": force_patch_weights,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        return self._loaded_size - previous

    def partially_unload(self, device_to, memory_to_free=0):
        response = self.executor.remote_rank_command(self.rank, {
            "kind": "unload",
            "memory_to_free": memory_to_free,
        })
        previous = self._loaded_size
        self._loaded_size = int(response["loaded_size"])
        return previous - self._loaded_size


class _ObjectCoordinator:
    def broadcast_command(self, command=None):
        if command is not None:
            command = inject_trace_context(command)
        values = [command]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def send_object(self, value, destination_rank):
        dist.send_object_list([value], dst=destination_rank)

    def receive_object(self, source_rank):
        values = [None]
        dist.recv_object_list(values, src=source_rank)
        return values[0]


def _run_worker(operations, coordinator, patcher):
    while True:
        command = coordinator.broadcast_command()
        with distributed_command_span(
            command,
            "Tensor Parallel Rank Command",
            {
                "distributed.rank": operations.rank,
                "distributed.world_size": operations.world_size,
                "comfy.command.kind": command["kind"],
            },
        ):
                if command["kind"] == "close":
                    return
                if command["kind"] == "rank_model":
                    if command["rank"] != operations.rank:
                        continue
                    try:
                        model_command = command["command"]
                        if model_command["kind"] == "load":
                            patcher.partially_load(
                                operations.device,
                                model_command["extra_memory"],
                                force_patch_weights=model_command["force_patch_weights"],
                            )
                        elif model_command["kind"] == "unload":
                            patcher.partially_unload(patcher.offload_device, model_command["memory_to_free"])
                        elif model_command["kind"] == "reset_dynamic_buffers":
                            patcher.reset_dynamic_buffers()
                        coordinator.send_object({"kind": "rank_model", "loaded_size": patcher.loaded_size()}, 0)
                    except Exception:
                        coordinator.send_object({"kind": "error", "traceback": traceback.format_exc()}, 0)
                    continue
                if command["kind"] == "finish_execution":
                    try:
                        from .. import model_prefetch

                        model_prefetch.finish_model_execution()
                        coordinator.send_object({"kind": "done"}, 0)
                    except Exception:
                        coordinator.send_object({"kind": "error", "traceback": traceback.format_exc()}, 0)
                    continue
                try:
                    tensors = _broadcast_tensors(operations, descriptors=command["descriptors"])
                    args, kwargs = unpack_pipeline_value(command["structure"], tensors)
                    getattr(patcher.model.diffusion_model, command["method"])(*args, **kwargs)
                    coordinator.send_object({"kind": "done"}, 0)
                except Exception:
                    coordinator.send_object({"kind": "error", "traceback": traceback.format_exc()}, 0)


def worker_main(host, port, authkey):
    connection = multiprocessing.connection.Client((host, int(port)), authkey=bytes.fromhex(authkey))
    distributed = resolve_distributed_configuration(environment=os.environ)
    rank = distributed.rank
    connection.send(rank)
    world_size, init_method, load_spec, device_identity = connection.recv()
    device_provider = accelerator_device_provider(device_identity.device_type)
    device = device_provider.resolve(device_identity)
    device_provider.select(device)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=5))
    device_group = dist.new_group(
        ranks=list(range(world_size)), backend="nccl", device_id=device
    )
    operations = TorchDistributedTensorParallelOperations(
        rank, world_size, device, device_group, dist.group.WORLD
    )
    os.environ["COMFY_AIMDO_DEVICE_INDICES"] = str(device.index)
    from .. import model_management
    from .. import aimdo_integration  # noqa: F401
    from .loader import load_tensor_parallel_rank

    model_management.set_torch_device(device)
    try:
        patcher = load_tensor_parallel_rank(load_spec, rank, device, operations)
        connection.send({"kind": "ready", "size": patcher.model_size()})
        _run_worker(operations, _ObjectCoordinator(), patcher)
        del patcher
        gc.collect()
    except Exception:
        connection.send({"kind": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        connection.close()
        if dist.is_initialized():
            dist.destroy_process_group(device_group)
            dist.destroy_process_group()


def launch_tensor_parallel(load_spec, devices, load_root):
    size = len(devices)
    device_provider = accelerator_device_provider(devices[0].type)
    device_identities = tuple(device_provider.identify(device) for device in devices)
    init_method = _free_tcp_init_method()
    authkey = secrets.token_bytes(32)
    listener = multiprocessing.connection.Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    master_host, master_port = init_method.removeprefix("tcp://").rsplit(":", 1)
    workers = []
    connections = [None] * (size - 1)
    for rank in range(1, size):
        environment = DistributedConfiguration(
            rank=rank,
            world_size=size,
            local_rank=rank,
            local_world_size=size,
            master_addr=master_host,
            master_port=int(master_port),
            pipeline_parallel_size=1,
            tensor_parallel_size=size,
            executor_backend="mp",
        ).canonical_environment()
        worker_pool = ContextVarProcessPoolExecutor(max_workers=1)
        worker_future = worker_pool.submit_with_environment(
            environment,
            worker_main,
            str(host),
            str(port),
            authkey.hex(),
        )
        workers.append((worker_pool, worker_future))
    try:
        for _ in range(size - 1):
            connection = listener.accept()
            rank = connection.recv()
            connections[rank - 1] = connection
            connection.send((size, init_method, load_spec, device_identities[rank]))
        device_provider.select(devices[0])
        dist.init_process_group("gloo", init_method=init_method, rank=0, world_size=size,
                                timeout=timedelta(minutes=5))
        device_group = dist.new_group(
            ranks=list(range(size)), backend="nccl", device_id=devices[0]
        )
        operations = TorchDistributedTensorParallelOperations(
            0, size, devices[0], device_group, dist.group.WORLD
        )
        root = load_root(operations)
        rank_sizes = [root.model_size()]
        for rank, connection in enumerate(connections, start=1):
            if not connection.poll(300):
                raise TimeoutError(f"Tensor-parallel rank {rank} did not load within five minutes")
            response = connection.recv()
            if response["kind"] != "ready":
                raise RuntimeError(f"Tensor-parallel rank {rank} failed to load:\n{response['traceback']}")
            rank_sizes.append(response["size"])
        return root, TorchDistributedTensorParallelExecutor(
            root.model.diffusion_model, operations, _ObjectCoordinator(), workers,
            connections, devices, rank_sizes,
        )
    except Exception:
        for worker_pool, _worker_future in workers:
            worker_pool.shutdown(wait=False)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        listener.close()
