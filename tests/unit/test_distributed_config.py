import os

import pytest

from comfy.cli_args_types import Configuration
from comfy.distributed.config import (
    CanonicalDistributedConfigurationProvider,
    DistributedConfiguration,
)
from comfy.component_model.setup import prepare_distributed_environment


def test_torchrun_environment_is_the_canonical_process_identity():
    environment = {
        "RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "worker-0",
        "MASTER_PORT": "23456",
    }

    distributed = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(), environment
    )

    assert distributed.rank == 1
    assert distributed.world_size == 2
    assert distributed.local_rank == 1
    assert distributed.local_world_size == 2
    assert distributed.pipeline_rank == 1
    assert distributed.pipeline_parallel_size == 2
    assert not distributed.is_first_pipeline_stage
    assert distributed.is_last_pipeline_stage
    assert distributed.master_addr == "worker-0"
    assert distributed.master_port == 23456


def test_configuration_values_override_launcher_environment():
    configuration = Configuration(
        rank=0,
        world_size=4,
        local_rank=0,
        local_world_size=2,
        master_addr="configured",
        master_port=12345,
        pipeline_parallel_size=4,
        distributed_executor_backend="external_launcher",
    )

    distributed = CanonicalDistributedConfigurationProvider().resolve(
        configuration,
        {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"},
    )

    assert distributed == DistributedConfiguration(
        rank=0,
        world_size=4,
        local_rank=0,
        local_world_size=2,
        master_addr="configured",
        master_port=12345,
        pipeline_parallel_size=4,
        executor_backend="external_launcher",
        externally_launched=True,
    )


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {
                "OMPI_COMM_WORLD_RANK": "2",
                "OMPI_COMM_WORLD_SIZE": "4",
                "OMPI_COMM_WORLD_LOCAL_RANK": "0",
                "OMPI_COMM_WORLD_LOCAL_SIZE": "2",
            },
            (2, 4, 0, 2),
        ),
        (
            {
                "SLURM_PROCID": "3",
                "SLURM_NTASKS": "4",
                "SLURM_LOCALID": "1",
                "SLURM_NTASKS_PER_NODE": "2",
            },
            (3, 4, 1, 2),
        ),
    ],
)
def test_launcher_aliases_are_normalized(environment, expected):
    distributed = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(), environment
    )

    assert (
        distributed.rank,
        distributed.world_size,
        distributed.local_rank,
        distributed.local_world_size,
    ) == expected


def test_canonical_environment_round_trips_worker_identity():
    original = DistributedConfiguration(
        rank=1,
        world_size=2,
        local_rank=1,
        local_world_size=2,
        master_addr="127.0.0.1",
        master_port=29501,
        pipeline_parallel_size=2,
        executor_backend="mp",
    )

    resolved = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(distributed_executor_backend="mp"),
        original.canonical_environment(),
    )

    assert resolved.rank == original.rank
    assert resolved.world_size == original.world_size
    assert resolved.local_rank == original.local_rank
    assert resolved.local_world_size == original.local_world_size
    assert resolved.pipeline_rank == 1


def test_invalid_rank_fails_before_process_group_initialization():
    with pytest.raises(ValueError, match="outside world size"):
        CanonicalDistributedConfigurationProvider().resolve(
            Configuration(), {"RANK": "2", "WORLD_SIZE": "2"}
        )


def test_slurm_compressed_local_world_size_is_normalized():
    distributed = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(),
        {
            "SLURM_PROCID": "1",
            "SLURM_NTASKS": "8",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS_PER_NODE": "2(x4)",
        },
    )

    assert distributed.local_world_size == 2


@pytest.mark.parametrize(
    ("rank", "expected_devices", "custom_nodes_disabled"),
    [(0, "0,1", False), (1, "1", True)],
)
def test_external_rank_prepares_aimdo_and_custom_node_scope(
    monkeypatch,
    rank,
    expected_devices,
    custom_nodes_disabled,
):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    configuration = Configuration()

    prepare_distributed_environment(configuration)

    assert os.environ["COMFY_AIMDO_DEVICE_INDICES"] == expected_devices
    assert configuration.disable_all_custom_nodes is custom_nodes_disabled
