from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import torch

from .. import model_detection, model_management, ops, utils
from ..execution_context import current_execution_context
from ..model_patcher import get_model_patcher_class
from ..pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from ..pipeline_parallel.loader import _normalize_detection_state, _normalized_descriptors
from .checkpoint import shard_minimax_h3_state_dict
from .distributed import RemoteTensorParallelRankModel, launch_tensor_parallel
from .types import TensorParallelConfig
from .operations import tensor_parallel_operations


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorParallelWorkerLoadSpec:
    checkpoint_path: str
    model_options: dict
    disable_dynamic: bool
    dtype: torch.dtype


def _load_rank(
    reader,
    model_config,
    metadata,
    prefix,
    original_keys,
    checkpoint_path,
    model_options,
    disable_dynamic,
    dtype,
    device,
    parallel_config,
    operations_decorator=tensor_parallel_operations,
    state_transform=shard_minimax_h3_state_dict,
):
    rank_config = type(model_config)(model_config.unet_config)
    rank_config.quant_config = utils.deepcopy_list_dict(model_config.quant_config) if model_config.quant_config is not None else None
    rank_config.custom_operations = model_config.custom_operations
    rank_config.optimizations = model_config.optimizations.copy()
    manual_cast_dtype = model_management.unet_manual_cast(
        None if rank_config.quant_config is not None else dtype,
        device,
        rank_config.supported_inference_dtypes,
    )
    rank_config.set_inference_dtype(dtype, manual_cast_dtype, device=device)
    if model_options.get("custom_operations") is not None:
        rank_config.custom_operations = model_options["custom_operations"]
    if model_options.get("fp8_optimizations", False):
        rank_config.optimizations["fp8"] = True
    base_operations = rank_config.custom_operations
    if base_operations is None:
        base_operations = ops.pick_operations(
            rank_config.unet_config.get("dtype"),
            rank_config.manual_cast_dtype,
            fp8_optimizations=rank_config.optimizations.get("fp8", False),
            model_config=rank_config,
        )
    rank_config.custom_operations = operations_decorator(
        base_operations,
        parallel_config,
    )

    state = reader.load_keys(original_keys.values())
    if prefix:
        state = utils.state_dict_prefix_replace(state, {prefix: ""}, filter_keys=True)
    state = state_transform(state, parallel_config.rank, parallel_config.size)
    if model_options.get("custom_operations") is None:
        state, _ = utils.convert_old_quants(state, "", metadata=dict(metadata))

    model = rank_config.get_model(state, "", device=torch.device("cpu"))
    patcher = get_model_patcher_class(disable_dynamic)(
        model,
        load_device=device,
        offload_device=torch.device("cpu"),
        ckpt_name=os.path.basename(checkpoint_path),
    )
    model.load_model_weights(state, "", assign=patcher.is_dynamic())
    return patcher


def load_tensor_parallel_rank(load_spec, rank, device, tensor_operations):
    del rank
    reader = SafetensorsCheckpointReader(load_spec.checkpoint_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    if model_config is None or model_config.unet_config.get("image_model") != "minimax_h3":
        raise RuntimeError(f"Tensor-parallel worker could not detect MiniMax H3 in {load_spec.checkpoint_path}")
    _descriptors, original_keys = _normalized_descriptors(reader, prefix)
    return _load_rank(
        reader, model_config, metadata, prefix, original_keys,
        load_spec.checkpoint_path, load_spec.model_options, load_spec.disable_dynamic,
        load_spec.dtype, device, TensorParallelConfig(tensor_operations),
    )


def load_diffusion_model_tensor_parallel(unet_path, devices, model_options=None, disable_dynamic=False):
    model_options = dict(model_options or {})
    reader = SafetensorsCheckpointReader(unet_path)
    detection_state, metadata, prefix = _normalize_detection_state(reader)
    model_config = model_detection.model_config_from_unet(detection_state, "", metadata=metadata)
    if model_config is None or model_config.unet_config.get("image_model") != "minimax_h3":
        raise ValueError("Tensor parallelism currently supports MiniMax H3 checkpoints")
    _descriptors, original_keys = _normalized_descriptors(reader, prefix)

    parameters = utils.calculate_parameters(detection_state)
    weight_dtype = None if model_config.quant_config is not None else utils.weight_dtype(detection_state)
    dtype = model_options.get("dtype") or model_management.unet_dtype(
        device=devices[0], model_params=parameters,
        supported_dtypes=list(model_config.supported_inference_dtypes),
        weight_dtype=weight_dtype,
    )
    load_spec = TensorParallelWorkerLoadSpec(
        os.fspath(unet_path), model_options, disable_dynamic, dtype
    )

    def load_root(tensor_operations):
        return _load_rank(
            reader, model_config, metadata, prefix, original_keys, unet_path,
            model_options,
            disable_dynamic,
            dtype,
            devices[0],
            TensorParallelConfig(tensor_operations),
        )

    root, executor = launch_tensor_parallel(load_spec, devices, load_root)
    remotes = [
        RemoteTensorParallelRankModel(
            executor, rank, devices[rank], executor.rank_sizes[rank], dtype,
            not disable_dynamic, os.path.basename(unet_path),
        )
        for rank in range(1, len(devices))
    ]
    root.set_additional_models("tensor_parallel", remotes)
    root.model.pipeline_executor = executor
    root.set_attachments("tensor_parallel_executor", executor)
    root.cached_patcher_init = (
        load_diffusion_model_tensor_parallel,
        (unet_path, devices, model_options, disable_dynamic),
    )
    logger.info(
        "Loaded minimax_h3 with tensor-parallel ranks %s",
        ", ".join(
            f"{device}:{executor.rank_sizes[index] / (1024 ** 3):.2f} GiB"
            for index, device in enumerate(devices)
        ),
    )
    return root


def try_load_diffusion_model_tensor_parallel(unet_path, model_options=None, disable_dynamic=False):
    configuration = current_execution_context().configuration
    size = int(configuration.tensor_parallel_size or 1)
    if size == 1:
        return None
    if os.path.splitext(os.fspath(unet_path))[1].lower() not in (".safetensors", ".sft"):
        raise ValueError("Tensor parallel loading requires a safetensors checkpoint")
    current = model_management.get_torch_device()
    available = model_management.get_all_torch_devices()
    devices = tuple([current] + [device for device in available if device != current])
    if size > len(devices):
        raise ValueError(f"Tensor parallel size {size} exceeds {len(devices)} available devices")
    return load_diffusion_model_tensor_parallel(
        unet_path, devices[:size], model_options=model_options, disable_dynamic=disable_dynamic
    )
