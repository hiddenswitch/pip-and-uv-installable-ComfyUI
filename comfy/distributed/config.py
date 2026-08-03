from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import os


DISTRIBUTED_EXECUTOR_BACKENDS = ("auto", "peer", "mp", "external_launcher")


def _first_int(environment: Mapping[str, str], names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = environment.get(name)
        if value not in (None, ""):
            try:
                # Slurm may encode a homogeneous per-node count as ``2(x4)``.
                return int(value.split("(", 1)[0])
            except ValueError as error:
                raise ValueError(f"{name} must be an integer, got {value!r}") from error
    return default


@dataclass(frozen=True)
class DistributedConfiguration:
    """Canonical process and pipeline identity for one ComfyUI process.

    The process fields follow the environment contract used by ``torchrun``.
    ComfyUI currently has one model-parallel dimension, so a launched process'
    pipeline rank is its global rank and its pipeline size defaults to world size.
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    pipeline_parallel_size: int = 1
    executor_backend: str = "auto"
    externally_launched: bool = False

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world size must be at least one")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is outside world size {self.world_size}")
        if self.local_world_size < 1:
            raise ValueError("local world size must be at least one")
        if not 0 <= self.local_rank < self.local_world_size:
            raise ValueError(
                f"local rank {self.local_rank} is outside local world size {self.local_world_size}"
            )
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline parallel size must be at least one")
        if self.externally_launched and self.pipeline_parallel_size != self.world_size:
            raise ValueError(
                "external launch currently requires pipeline parallel size to equal world size"
            )
        if self.executor_backend not in DISTRIBUTED_EXECUTOR_BACKENDS:
            raise ValueError(
                f"distributed executor backend must be one of {DISTRIBUTED_EXECUTOR_BACKENDS}"
            )
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master port must be between 1 and 65535")

    @property
    def pipeline_rank(self) -> int:
        if self.externally_launched:
            return self.rank
        return 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_first_pipeline_stage(self) -> bool:
        return self.pipeline_rank == 0

    @property
    def is_last_pipeline_stage(self) -> bool:
        return self.pipeline_rank == self.pipeline_parallel_size - 1

    def canonical_environment(self) -> dict[str, str]:
        return {
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }


class AbstractBaseDistributedConfigurationProvider(ABC):
    @abstractmethod
    def resolve(
        self,
        configuration,
        environment: Mapping[str, str] | None = None,
    ) -> DistributedConfiguration:
        raise NotImplementedError


class CanonicalDistributedConfigurationProvider(
    AbstractBaseDistributedConfigurationProvider
):
    """Resolve torchrun variables, with launcher aliases used by Accelerate.

    Explicit ``Configuration`` values win. PyTorch/TorchElastic names are next,
    followed by PMI, Open MPI, MVAPICH, and Slurm compatibility variables.
    The result is immutable and is the only rank/topology object consumers use.
    """

    _RANK = ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID")
    _WORLD_SIZE = ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS")
    _LOCAL_RANK = ("LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID")
    _LOCAL_WORLD_SIZE = ("LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE")

    def resolve(
        self,
        configuration,
        environment: Mapping[str, str] | None = None,
    ) -> DistributedConfiguration:
        environment = os.environ if environment is None else environment

        world_size = self._configured_int(configuration, "world_size")
        if world_size is None:
            world_size = _first_int(environment, self._WORLD_SIZE, 1)
        rank = self._configured_int(configuration, "rank")
        if rank is None:
            rank = _first_int(environment, self._RANK, 0)
        local_world_size = self._configured_int(configuration, "local_world_size")
        if local_world_size is None:
            local_world_size = _first_int(environment, self._LOCAL_WORLD_SIZE, world_size)
        local_rank = self._configured_int(configuration, "local_rank")
        if local_rank is None:
            local_rank = _first_int(environment, self._LOCAL_RANK, rank)

        pipeline_size = self._configured_int(configuration, "pipeline_parallel_size")
        if pipeline_size is None:
            pipeline_size = world_size
        backend = self._configured_value(configuration, "distributed_executor_backend") or "auto"
        master_addr = self._configured_value(configuration, "master_addr") or environment.get("MASTER_ADDR") or "127.0.0.1"
        master_port = self._configured_int(configuration, "master_port")
        if master_port is None:
            master_port = _first_int(environment, ("MASTER_PORT",), 29500)

        launch_variables_present = any(
            name in environment
            for name in (*self._RANK, *self._WORLD_SIZE, *self._LOCAL_RANK)
        )
        explicitly_configured = any(
            self._configured_value(configuration, name) is not None
            for name in ("rank", "world_size", "local_rank", "local_world_size")
        )
        externally_launched = world_size > 1 and (
            backend == "external_launcher"
            or launch_variables_present
            or explicitly_configured
        )

        return DistributedConfiguration(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            master_addr=str(master_addr),
            master_port=master_port,
            pipeline_parallel_size=pipeline_size,
            executor_backend=str(backend),
            externally_launched=externally_launched,
        )

    @staticmethod
    def _configured_value(configuration, name: str):
        if configuration is None:
            return None
        return configuration.get(name)

    @classmethod
    def _configured_int(cls, configuration, name: str) -> int | None:
        value = cls._configured_value(configuration, name)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an integer, got {value!r}") from error


def resolve_distributed_configuration(
    configuration=None,
    environment: Mapping[str, str] | None = None,
    provider: AbstractBaseDistributedConfigurationProvider | None = None,
) -> DistributedConfiguration:
    return (provider or CanonicalDistributedConfigurationProvider()).resolve(
        configuration,
        environment,
    )
