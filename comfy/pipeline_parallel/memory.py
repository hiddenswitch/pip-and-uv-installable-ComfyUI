from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from .. import model_management

from .types import PipelineDeviceMemoryBudget, PipelineModelMemoryGeometry, PipelinePartitionPlan, PipelineStageMemoryGeometry


class AbstractBasePipelineMemoryCoordinator(ABC):
    @abstractmethod
    def budgets(self, devices: Sequence[torch.device]) -> tuple[PipelineDeviceMemoryBudget, ...]:
        raise NotImplementedError


class ComfyPipelineMemoryCoordinator(AbstractBasePipelineMemoryCoordinator):
    """Use DynamicVRAM's own per-device reclaimable-memory accounting."""

    def budgets(self, devices: Sequence[torch.device]) -> tuple[PipelineDeviceMemoryBudget, ...]:
        projected = model_management.projected_dynamic_vram_available_memory(devices)
        return tuple(
            PipelineDeviceMemoryBudget(
                device=device,
                available_weight_bytes=max(1, projected[device]),
            )
            for device in devices
        )


class ExternalPipelineMemoryCoordinator(AbstractBasePipelineMemoryCoordinator):
    """Ask every external launcher rank for its current DynamicVRAM budget."""

    def __init__(self, runtime):
        self.runtime = runtime

    def budgets(self, devices: Sequence[torch.device]) -> tuple[PipelineDeviceMemoryBudget, ...]:
        return self.runtime.probe_memory_budgets(devices)


class AbstractBasePipelineStageMemoryEstimator(ABC):
    @abstractmethod
    def estimate(self, stage_spec, plan: PipelinePartitionPlan, patchers: Sequence[object]) -> PipelineModelMemoryGeometry:
        raise NotImplementedError

    @abstractmethod
    def estimate_stage(self, stage_spec, stage, patcher) -> PipelineStageMemoryGeometry:
        raise NotImplementedError


class ComfyDynamicVRAMStageMemoryEstimator(AbstractBasePipelineStageMemoryEstimator):
    """Measure stage demand with the geometry already used by DynamicVRAM."""

    def estimate(self, stage_spec, plan: PipelinePartitionPlan, patchers: Sequence[object]) -> PipelineModelMemoryGeometry:
        if len(patchers) != plan.size:
            raise ValueError("Pipeline memory estimation requires one model patcher per stage")

        block_bytes = [0] * stage_spec.block_count
        non_block_bytes = [0] * plan.size
        for stage, patcher in zip(plan.stages, patchers, strict=True):
            geometry = self.estimate_stage(stage_spec, stage, patcher)
            for index, size in geometry.block_bytes.items():
                block_bytes[index] += size
            non_block_bytes[stage.index] += geometry.non_block_bytes

        return PipelineModelMemoryGeometry(tuple(block_bytes), tuple(non_block_bytes))

    def estimate_stage(self, stage_spec, stage, patcher) -> PipelineStageMemoryGeometry:
        block_bytes = {}
        non_block_bytes = 0
        stored_bytes = 0
        for item in patcher._load_list(for_dynamic=True):
            stored_bytes += item.module_size
            demand = max(item.module_size, item.module_offload_mem - item.module_size)
            name = item.name.removeprefix("diffusion_model.")
            block_index = stage_spec.block_index(name)
            if block_index is None:
                non_block_bytes += demand
            else:
                block_bytes[block_index] = block_bytes.get(block_index, 0) + demand
        non_block_bytes += max(0, patcher.model_size() - stored_bytes)
        return PipelineStageMemoryGeometry(block_bytes, non_block_bytes)
