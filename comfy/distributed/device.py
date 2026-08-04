from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AcceleratorDeviceIdentity:
    device_type: str
    uuid: str


class AbstractBaseAcceleratorDeviceProvider(ABC):
    @abstractmethod
    def identify(self, device: torch.device) -> AcceleratorDeviceIdentity:
        raise NotImplementedError

    @abstractmethod
    def resolve(self, identity: AcceleratorDeviceIdentity) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def select(self, device: torch.device) -> None:
        raise NotImplementedError


class CudaAcceleratorDeviceProvider(AbstractBaseAcceleratorDeviceProvider):
    def identify(self, device: torch.device) -> AcceleratorDeviceIdentity:
        if device.type != "cuda":
            raise ValueError(f"CUDA device provider cannot identify {device}")
        return AcceleratorDeviceIdentity("cuda", str(torch.cuda.get_device_properties(device).uuid))

    def resolve(self, identity: AcceleratorDeviceIdentity) -> torch.device:
        if identity.device_type != "cuda":
            raise ValueError(f"CUDA device provider cannot resolve {identity.device_type}")
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            if str(torch.cuda.get_device_properties(device).uuid) == identity.uuid:
                return device
        raise RuntimeError(f"CUDA device {identity.uuid} is not visible in this process")

    def select(self, device: torch.device) -> None:
        torch.accelerator.set_device_index(device)

def accelerator_device_provider(device_type: str) -> AbstractBaseAcceleratorDeviceProvider:
    if device_type == "cuda":
        return CudaAcceleratorDeviceProvider()
    raise ValueError(f"Distributed accelerator device type {device_type!r} is not supported")
