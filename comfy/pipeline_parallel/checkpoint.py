from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import struct
from typing import Collection, Mapping

import torch

from .. import utils

from .types import TensorDescriptor


class AbstractBaseCheckpointReader(ABC):
    @property
    @abstractmethod
    def metadata(self) -> Mapping[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def tensors(self) -> Mapping[str, TensorDescriptor]:
        raise NotImplementedError

    @abstractmethod
    def detection_state_dict(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_keys(self, keys: Collection[str]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class SafetensorsCheckpointReader(AbstractBaseCheckpointReader):
    _VALUE_KEYS = frozenset(("scaled_fp8",))

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.suffix.lower() not in (".safetensors", ".sft"):
            raise ValueError("Pipeline parallel loading currently requires a safetensors checkpoint")
        self.path = path.resolve(strict=True)
        with self.path.open("rb") as checkpoint:
            header_size = struct.unpack("<Q", checkpoint.read(8))[0]
            header = json.loads(checkpoint.read(header_size))
        self._metadata = header.pop("__metadata__", {})
        self._tensors = {
            name: TensorDescriptor(
                shape=tuple(info["shape"]),
                dtype=utils._TYPES[info["dtype"]],
                nbytes=info["data_offsets"][1] - info["data_offsets"][0],
            )
            for name, info in header.items()
        }

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    @property
    def tensors(self) -> Mapping[str, TensorDescriptor]:
        return self._tensors

    def detection_state_dict(self) -> dict[str, torch.Tensor]:
        value_keys = {
            key for key in self._tensors
            if key.endswith(".comfy_quant") or key in self._VALUE_KEYS
        }
        values = self.load_keys(value_keys)
        return {
            key: values.get(key, torch.empty(descriptor.shape, dtype=descriptor.dtype, device="meta"))
            for key, descriptor in self._tensors.items()
        }

    def load_keys(self, keys: Collection[str]) -> dict[str, torch.Tensor]:
        unknown = set(keys).difference(self._tensors)
        if unknown:
            raise KeyError(f"Checkpoint does not contain pipeline keys: {sorted(unknown)[:5]}")
        return utils.load_torch_file(str(self.path), include_keys=frozenset(keys))
