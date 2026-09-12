"""Observed accelerator topology: device identity, peer links, and the
model-parallel sizes this machine can host.

Used by ``comfyui env check``; nothing here runs during normal startup.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from ..component_model.guess_settings import default_tensor_parallel_size

DEFAULT_TRANSFER_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class DeviceLink:
    """One measured direction of a device-to-device copy."""

    source: torch.device
    destination: torch.device
    peer_access: bool
    bandwidth_bytes_per_second: float


def cuda_devices() -> list[torch.device]:
    if not torch.cuda.is_available():
        return []
    return [torch.device("cuda", index) for index in range(torch.cuda.device_count())]


def device_names(devices: Sequence[torch.device]) -> list[str]:
    return [torch.cuda.get_device_name(device) for device in devices]


def nccl_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_nccl_available()


def measure_device_link(
    source: torch.device,
    destination: torch.device,
    size_bytes: int = DEFAULT_TRANSFER_BYTES,
    iterations: int = 5,
) -> DeviceLink:
    """Time ``iterations`` copies of ``size_bytes`` from ``source`` to ``destination``.

    The copy goes through peer access when the driver allows it and through
    host memory otherwise; both are what a pipeline stage or tensor-parallel
    rank would see, so the measured figure is the usable bandwidth either way.
    """
    if source == destination:
        raise ValueError("a device link needs two different devices")
    peer_access = bool(torch.cuda.can_device_access_peer(source.index, destination.index))
    payload = torch.empty(size_bytes, dtype=torch.uint8, device=source)
    landing = torch.empty(size_bytes, dtype=torch.uint8, device=destination)
    landing.copy_(payload)
    torch.cuda.synchronize(source)
    torch.cuda.synchronize(destination)
    started = time.perf_counter()
    for _ in range(iterations):
        landing.copy_(payload, non_blocking=True)
    torch.cuda.synchronize(source)
    torch.cuda.synchronize(destination)
    elapsed = time.perf_counter() - started
    del payload, landing
    return DeviceLink(
        source=source,
        destination=destination,
        peer_access=peer_access,
        bandwidth_bytes_per_second=(size_bytes * iterations) / elapsed if elapsed > 0 else float("inf"),
    )


def measure_device_links(
    devices: Sequence[torch.device],
    size_bytes: int = DEFAULT_TRANSFER_BYTES,
    iterations: int = 5,
) -> list[DeviceLink]:
    """Measure every ordered pair of ``devices``."""
    return [
        measure_device_link(source, destination, size_bytes=size_bytes, iterations=iterations)
        for source in devices
        for destination in devices
        if source != destination
    ]


def links_by_pair(links: Sequence[DeviceLink]) -> dict[tuple[torch.device, torch.device], DeviceLink]:
    return {(link.source, link.destination): link for link in links}


def max_tensor_parallel_size(names: Sequence[str], nccl_available: bool, windows: bool) -> int:
    """Largest tensor-parallel size the visible devices support.

    Ranks exchange activations over NCCL, which Windows builds of torch do not
    ship, and the rank shards assume every rank has the same product.
    """
    if windows or not nccl_available:
        return 1
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return 1
    largest = max(counts.values())
    return default_tensor_parallel_size([""] * largest)


def max_pipeline_parallel_size(devices: Sequence[torch.device], links: Sequence[DeviceLink]) -> tuple[int, str]:
    """Largest pipeline-parallel size and the transport that carries it.

    Every visible device can host a stage. Consecutive stages move activations
    over peer access in one process when the whole chain allows it, and
    through worker processes (``mp``) otherwise.
    """
    if len(devices) < 2:
        return len(devices), "none"
    table = links_by_pair(links)
    chain = all(
        (link := table.get((source, destination))) is not None and link.peer_access
        for source, destination in zip(devices, devices[1:])
    )
    return len(devices), "peer" if chain else "mp"


def is_windows() -> bool:
    return os.name == "nt"
