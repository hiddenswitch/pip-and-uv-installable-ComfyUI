from __future__ import annotations

import logging
import os

import torch

from .. import model_detection, model_management, utils
from ..distributed.config import resolve_distributed_configuration
from ..execution_context import current_execution_context
from ..model_patcher import get_model_patcher_class

from .checkpoint import SafetensorsCheckpointReader
from .distributed import RemotePipelineStageModel
from .memory import (
    AbstractBasePipelineMemoryCoordinator,
    AbstractBasePipelineStageMemoryEstimator,
    ComfyDynamicVRAMStageMemoryEstimator,
    ComfyPipelineMemoryCoordinator,
)
from .patcher import get_pipeline_model_patcher_class
from .operations import select_pipeline_operations
from .runtime import AbstractBasePipelineOperations
from .stages import get_pipeline_stage_spec
from .types import PipelineModelMemoryGeometry, PipelineParallelConfig, PipelineStageConfig, PipelineWorkerLoadSpec

logger = logging.getLogger(__name__)


def _normalize_detection_state(reader: SafetensorsCheckpointReader):
    state_dict = reader.detection_state_dict()
    metadata = dict(reader.metadata)
    state_dict, metadata = utils.convert_old_quants(state_dict, "", metadata=metadata)
    prefix = model_detection.unet_prefix_from_state_dict(state_dict)
    normalized = utils.state_dict_prefix_replace(state_dict, {prefix: ""}, filter_keys=True)
    if not normalized:
        return state_dict, metadata, ""
    normalized, metadata = utils.convert_old_quants(normalized, "", metadata=metadata)
    return normalized, metadata, prefix


def _normalized_descriptors(reader: SafetensorsCheckpointReader, prefix: str):
    descriptors = {}
    original_keys = {}
    for key, descriptor in reader.tensors.items():
        if prefix:
            if not key.startswith(prefix):
                continue
            normalized = key[len(prefix):]
        else:
            normalized = key
        descriptors[normalized] = descriptor
        original_keys[normalized] = key
    return descriptors, original_keys


def _load_pipeline_stage(
    reader,
    model_config,
    metadata,
    prefix,
    original_keys,
    unet_path,
    stage_plan,
    stage,
    model_options,
    disable_dynamic,
    dtype,
    offload_device,
):
    stage_model_config = type(model_config)(model_config.unet_config)
    stage_model_config.quant_config = utils.deepcopy_list_dict(model_config.quant_config) if model_config.quant_config is not None else None
    stage_model_config.custom_operations = model_config.custom_operations
    stage_model_config.optimizations = model_config.optimizations.copy()
    manual_cast_dtype = model_management.unet_manual_cast(
        None if stage_model_config.quant_config is not None else dtype,
        stage.device,
        stage_model_config.supported_inference_dtypes,
    )
    stage_model_config.set_inference_dtype(dtype, manual_cast_dtype, device=stage.device)
    if model_options.get("custom_operations") is not None:
        stage_model_config.custom_operations = model_options["custom_operations"]
    if model_options.get("fp8_optimizations", False):
        stage_model_config.optimizations["fp8"] = True
    stage_model_config.unet_config["pipeline_stage"] = PipelineStageConfig(
        stage.index, stage_plan.size, stage.start_layer, stage.end_layer
    )

    raw_keys = {original_keys[key] for key in stage.owned_keys}
    stage_state = reader.load_keys(raw_keys)
    if prefix:
        stage_state = utils.state_dict_prefix_replace(stage_state, {prefix: ""}, filter_keys=True)
    if model_options.get("custom_operations") is None:
        stage_state, _ = utils.convert_old_quants(stage_state, "", metadata=dict(metadata))
    stage_spec = get_pipeline_stage_spec(model_config.unet_config.get("image_model"))
    stage_state = {
        key: value for key, value in stage_state.items()
        if stage_spec.owns_key(key, stage, stage_plan.size)
    }

    model = stage_model_config.get_model(stage_state, "", device=offload_device)
    patcher_class = get_pipeline_model_patcher_class(disable_dynamic) if stage.index == 0 else get_model_patcher_class(disable_dynamic)
    patcher = patcher_class(
        model,
        load_device=stage.device,
        offload_device=offload_device,
        ckpt_name=os.path.basename(unet_path),
    )
    model.load_model_weights(stage_state, "", assign=patcher.is_dynamic())
    return patcher


def load_pipeline_worker_stage(load_spec: PipelineWorkerLoadSpec):
    reader = SafetensorsCheckpointReader(load_spec.checkpoint_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    if model_config is None:
        raise RuntimeError(f"Pipeline worker could not detect {load_spec.checkpoint_path}")
    _descriptors, original_keys = _normalized_descriptors(reader, prefix)
    return _load_pipeline_stage(
        reader,
        model_config,
        metadata,
        prefix,
        original_keys,
        load_spec.checkpoint_path,
        load_spec.plan,
        load_spec.plan.stages[load_spec.stage_index],
        load_spec.model_options,
        load_spec.disable_dynamic,
        load_spec.dtype,
        torch.device("cpu"),
    )


def load_diffusion_model_pipeline(
    unet_path,
    pipeline_config: PipelineParallelConfig,
    model_options=None,
    disable_dynamic=False,
    pipeline_operations: AbstractBasePipelineOperations | None = None,
    memory_coordinator: AbstractBasePipelineMemoryCoordinator | None = None,
    stage_memory_estimator: AbstractBasePipelineStageMemoryEstimator | None = None,
):
    model_options = dict(model_options or {})
    reader = SafetensorsCheckpointReader(unet_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    if model_config is None:
        raise RuntimeError(f"Pipeline parallel loading could not detect the model type of {unet_path}")

    image_model = model_config.unet_config.get("image_model")
    stage_spec = get_pipeline_stage_spec(image_model)
    descriptors, original_keys = _normalized_descriptors(reader, prefix)
    memory_coordinator = memory_coordinator or ComfyPipelineMemoryCoordinator()
    stage_memory_estimator = stage_memory_estimator or ComfyDynamicVRAMStageMemoryEstimator()
    budgets = memory_coordinator.budgets(pipeline_config.devices)
    plan = stage_spec.plan(descriptors, pipeline_config, memory_budgets=budgets)
    pipeline_operations = select_pipeline_operations(pipeline_config.devices, pipeline_operations)

    parameters = utils.calculate_parameters(detection_state)
    weight_dtype = utils.weight_dtype(detection_state)
    if model_config.quant_config is not None:
        weight_dtype = None
    supported_dtypes = list(model_config.supported_inference_dtypes)
    dtype = model_options.get("dtype")
    if dtype is None:
        dtype = model_management.unet_dtype(
            device=plan.stages[0].device,
            model_params=parameters,
            supported_dtypes=supported_dtypes,
            weight_dtype=weight_dtype,
        )

    offload_device = model_options.get("offload_device", torch.device("cpu"))

    def load_stages(stage_plan):
        return [
            _load_pipeline_stage(
                reader,
                model_config,
                metadata,
                prefix,
                original_keys,
                unet_path,
                stage_plan,
                stage,
                model_options,
                disable_dynamic,
                dtype,
                offload_device,
            )
            for stage in stage_plan.stages
        ]

    def load_root(stage_plan):
        return _load_pipeline_stage(
            reader,
            model_config,
            metadata,
            prefix,
            original_keys,
            unet_path,
            stage_plan,
            stage_plan.stages[0],
            model_options,
            disable_dynamic,
            dtype,
            offload_device,
        )

    def worker_load_specs(stage_plan):
        return tuple(
            PipelineWorkerLoadSpec(
                checkpoint_path=os.fspath(unet_path),
                plan=stage_plan,
                stage_index=stage.index,
                model_options=model_options,
                disable_dynamic=disable_dynamic,
                dtype=dtype,
            )
            for stage in stage_plan.stages[1:]
        )

    def worker_geometry(stage_plan, root_patcher, worker_geometries):
        geometries = (
            stage_memory_estimator.estimate_stage(stage_spec, stage_plan.stages[0], root_patcher),
            *worker_geometries,
        )
        block_bytes = [0] * stage_spec.block_count
        non_block_bytes = [0] * stage_plan.size
        for stage, geometry in zip(stage_plan.stages, geometries, strict=True):
            for index, size in geometry.block_bytes.items():
                block_bytes[index] += size
            non_block_bytes[stage.index] += geometry.non_block_bytes
        return PipelineModelMemoryGeometry(tuple(block_bytes), tuple(non_block_bytes))

    executor = None

    if pipeline_operations.uses_worker_processes:
        root = load_root(plan)
        executor = pipeline_operations.create_executor(
            plan,
            [root.model.diffusion_model],
            worker_load_specs(plan),
        )
        model_memory_geometry = worker_geometry(plan, root, executor.worker_geometries)
        measured_plan = stage_spec.plan(
            descriptors,
            pipeline_config,
            memory_budgets=budgets,
            model_memory_geometry=model_memory_geometry,
        )
        initial_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in plan.stages)
        measured_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in measured_plan.stages)
        if initial_boundaries != measured_boundaries:
            logger.info(
                "Repartitioning %s process ranks from checkpoint estimate %s to DynamicVRAM geometry %s",
                image_model,
                initial_boundaries,
                measured_boundaries,
            )
            executor.close()
            del root
            plan = measured_plan
            root = load_root(plan)
            executor = pipeline_operations.create_executor(
                plan,
                [root.model.diffusion_model],
                worker_load_specs(plan),
            )
        else:
            plan = measured_plan
        stage_patchers = [root]
    else:
        stage_patchers = load_stages(plan)
        model_memory_geometry = stage_memory_estimator.estimate(stage_spec, plan, stage_patchers)
        measured_plan = stage_spec.plan(
            descriptors,
            pipeline_config,
            memory_budgets=budgets,
            model_memory_geometry=model_memory_geometry,
        )
        initial_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in plan.stages)
        measured_boundaries = tuple((stage.start_layer, stage.end_layer) for stage in measured_plan.stages)
        if initial_boundaries != measured_boundaries:
            logger.info(
                "Repartitioning %s from checkpoint estimate %s to DynamicVRAM geometry %s",
                image_model,
                initial_boundaries,
                measured_boundaries,
            )
            del stage_patchers
            stage_patchers = load_stages(measured_plan)
        plan = measured_plan

    root = stage_patchers[0]
    if executor is None:
        stages = [root.model.diffusion_model]
        stages.extend(patcher.model.diffusion_model.forward_pipeline_stage for patcher in stage_patchers[1:])
        executor = pipeline_operations.create_executor(plan, stages)
    if pipeline_operations.uses_worker_processes:
        remote_stages = [
            RemotePipelineStageModel(
                executor,
                stage.index,
                stage.device,
                model_memory_geometry.stage_bytes(stage),
                dtype,
                not disable_dynamic,
                os.path.basename(unet_path),
            )
            for stage in plan.stages[1:]
        ]
        root.set_additional_models(root.pipeline_additional_models_key, remote_stages)
        del stage_patchers[1:]
    else:
        root.set_additional_models(root.pipeline_additional_models_key, stage_patchers[1:])
    root.model.pipeline_executor = executor
    root.set_attachments("pipeline_parallel_plan", plan)
    root.set_attachments("pipeline_parallel_executor", executor)
    root.cached_patcher_init = (
        load_diffusion_model_pipeline,
        (unet_path, pipeline_config, model_options, disable_dynamic),
    )
    logger.info(
        "Loaded %s with pipeline stages %s",
        image_model,
        ", ".join(
            f"{stage.device}:{stage.start_layer}-{stage.end_layer} "
            f"({model_memory_geometry.stage_bytes(stage) / (1024 ** 3):.2f} GiB DynamicVRAM geometry, "
            f"{budgets[stage.index].available_weight_bytes / (1024 ** 3):.2f} GiB available)"
            for stage in plan.stages
        ),
    )
    return root


def try_load_diffusion_model_pipeline(
    unet_path,
    model_options=None,
    disable_dynamic=False,
    pipeline_operations: AbstractBasePipelineOperations | None = None,
    memory_coordinator: AbstractBasePipelineMemoryCoordinator | None = None,
    stage_memory_estimator: AbstractBasePipelineStageMemoryEstimator | None = None,
):
    """Load a supported checkpoint transparently when multiple devices are selected."""
    if os.path.splitext(os.fspath(unet_path))[1].lower() not in (".safetensors", ".sft"):
        return None

    current = model_management.get_torch_device()
    available = model_management.get_all_torch_devices()
    devices = tuple([current] + [device for device in available if device != current])
    runtime_configuration = current_execution_context().configuration
    distributed = resolve_distributed_configuration(runtime_configuration)
    requested_size = distributed.pipeline_parallel_size
    if requested_size > 1:
        if requested_size > len(devices):
            raise RuntimeError(
                f"Pipeline parallel size {requested_size} exceeds the {len(devices)} available devices"
            )
        devices = devices[:requested_size]
    elif (
        runtime_configuration is not None
        and runtime_configuration.pipeline_parallel_size == 1
    ):
        return None
    if len(devices) < 2:
        return None

    reader = SafetensorsCheckpointReader(unet_path)
    detection_state, metadata, _prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    if model_config is None or model_config.unet_config.get("image_model") not in ("qwen_image", "minimax_h3"):
        return None

    return load_diffusion_model_pipeline(
        unet_path,
        PipelineParallelConfig(devices),
        model_options=model_options,
        disable_dynamic=disable_dynamic,
        pipeline_operations=pipeline_operations,
        memory_coordinator=memory_coordinator,
        stage_memory_estimator=stage_memory_estimator,
    )
