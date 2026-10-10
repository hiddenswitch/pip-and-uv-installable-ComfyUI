from __future__ import annotations

import importlib
import json
import logging
import os
from functools import partial, wraps
from types import MethodType

import torch
from tokenizers import Tokenizer

from .. import model_management, ops, utils
from ..execution_context import current_execution_context
from ..model_patcher import get_model_patcher_class
from ..pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from .distributed import RemoteTensorParallelRankModel, launch_tensor_parallel
from .custom_node_state import CachedStateExecutor, install_state_transport
from .operations import gathered_output_operations
from .types import TensorParallelConfig


def _modules(extension):
    return tuple(importlib.import_module(extension + ".hunyuan_image_3." + name) for name in ("loader", "model", "model_base"))


def _shard_model(model, base_operations, parallel):
    operations = gathered_output_operations(base_operations, parallel)
    shards = {}
    for name, module in list(model.named_modules()):
        if not name.startswith("model.layers.") or name.endswith(".gate.wg"):
            continue
        if isinstance(module, base_operations.MoEExperts):
            replacement = operations.MoEExperts(module.num_experts, module.in_features, module.out_features, bias=module.bias is not None, device="cpu")
            axis = 1
        elif isinstance(module, base_operations.Linear):
            replacement = operations.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="cpu")
            axis = 0
        else:
            continue
        model.set_submodule(name, replacement)
        shards[name] = axis
    return shards


def _load_state(reader, shards, parallel, head_prefixes):
    selections = {}
    row_parameters = {"weight", "bias", "weight_scale", "weight_s_rel", "weight_s_channel"}
    for key, descriptor in reader.tensors.items():
        if key.startswith(head_prefixes):
            continue
        module, _, parameter = key.rpartition(".")
        axis = shards.get(module)
        selection = None
        if axis is not None and parameter in row_parameters and len(descriptor.shape) > axis:
            width, remainder = divmod(descriptor.shape[axis], parallel.size)
            if remainder:
                raise ValueError(f"{key} cannot be split across {parallel.size} ranks")
            selection = tuple(slice(parallel.rank * width, (parallel.rank + 1) * width) if i == axis else slice(None) for i in range(len(descriptor.shape)))
        selections[key] = selection
    return reader.load_slices(selections)


def _text_forward(self, embeddings, frequencies, mask, cache):
    hidden = self.model(embeddings, frequencies, mask, cache)
    return hidden, cache


class _TextTransformer:
    def __init__(self, model, executor):
        self.wte = model.wte
        self.executor = executor

    def __call__(self, embeddings, frequencies, mask, cache):
        hidden, updated = self.executor.execute_method(
            "tensor_parallel_text_forward", embeddings, frequencies, mask, cache,
        )
        cache[:] = updated
        return hidden


class _TextModel:
    def __init__(self, model, executor):
        self._model = model
        self.model = _TextTransformer(model.model, executor)

    def __getattr__(self, name):
        return getattr(self._model, name)


def load_rank(load_spec, device, tensor_operations):
    extension = os.path.basename(load_spec.custom_node_path)
    upstream_loader, upstream_model, upstream_base = _modules(extension)
    reader = SafetensorsCheckpointReader(load_spec.checkpoint_path)
    detection = reader.detection_state_dict()
    detection.update(reader.load_keys([upstream_loader.FINGERPRINT_KEY]))
    model_type = upstream_loader.detect_model_type(detection, load_spec.checkpoint_path)
    with open(upstream_loader.CONFIG_PATH) as source:
        config = json.load(source)
    config.update(upstream_loader.MODEL_TYPES[model_type])
    config["model_type"] = model_type
    quantization = utils.detect_layer_quantization(detection, "")
    if quantization is None:
        raise ValueError("HunyuanImage3 tensor parallelism requires a quantized checkpoint")
    operations = ops.pick_operations(load_spec.dtype, load_spec.dtype, load_device=device, model_config=upstream_loader._OpsConfig(quantization))
    config = upstream_base.HunyuanImage3ModelConfig(
        upstream_model.params_from_config(config), upstream_loader.HunyuanImage3(),
        {"shift": upstream_loader.SHIFT}, quant_config=quantization,
        dtype=load_spec.dtype, custom_operations=operations,
    )
    model = upstream_base.HunyuanImage3Model(config, device=torch.device("cpu"))
    parallel = TensorParallelConfig(tensor_operations)
    shards = _shard_model(model.diffusion_model, operations, parallel)
    state = _load_state(reader, shards, parallel, upstream_model.HEAD_PREFIXES)
    upstream_loader.shared_codebooks(state)
    patcher = get_model_patcher_class(load_spec.disable_dynamic)(
        model, load_device=device, offload_device=torch.device("cpu"),
        ckpt_name=os.path.basename(load_spec.checkpoint_path),
    )
    loaded = model.diffusion_model.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError(f"HunyuanImage3 checkpoint mismatch: {loaded}")
    model.tokenizer = Tokenizer.from_file(upstream_loader.TOKENIZER_PATH)
    model.diffusion_model.tensor_parallel_text_forward = MethodType(_text_forward, model.diffusion_model)
    spectrum = importlib.import_module(extension + ".hunyuan_image_3.spectrum")
    install_state_transport(model.diffusion_model, "hy3_spectrum", (
        spectrum.SpectrumState, spectrum.SpectrumForecaster, spectrum.ChebyshevForecaster,
    ))
    return patcher


def load_model(checkpoint_path, extension_path, disable_dynamic=False):
    from .loader import TensorParallelWorkerLoadSpec

    configuration = current_execution_context().configuration
    size = int(configuration.tensor_parallel_size or 1)
    current = model_management.get_torch_device()
    devices = tuple([current] + [device for device in model_management.get_all_torch_devices() if device != current])
    if size > len(devices):
        raise ValueError(f"Tensor parallel size {size} exceeds {len(devices)} available devices")
    devices = devices[:size]
    dtype = model_management.unet_dtype(supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])
    spec = TensorParallelWorkerLoadSpec(os.fspath(checkpoint_path), "hunyuan_image3", {}, disable_dynamic, dtype, extension_path)
    root, executor = launch_tensor_parallel(spec, devices, lambda operations: load_rank(spec, devices[0], operations))
    root.set_additional_models("tensor_parallel", [
        RemoteTensorParallelRankModel(executor, rank, devices[rank], executor.rank_sizes[rank], dtype, not disable_dynamic, os.path.basename(checkpoint_path))
        for rank in range(1, size)
    ])
    root.model.pipeline_executor = CachedStateExecutor(executor)
    root.model.diffusion_model._comfy_tensor_parallel_executor = executor
    root.set_attachments("tensor_parallel_executor", executor)
    root.cached_patcher_init = (partial(load_model, disable_dynamic=disable_dynamic), (checkpoint_path, extension_path))
    logging.info("Loaded hunyuan_image3 with tensor-parallel ranks %s", ", ".join(
        f"{device}:{executor.rank_sizes[rank] / (1024 ** 3):.2f} GiB" for rank, device in enumerate(devices)
    ))
    return root


def install(extension):
    upstream_loader, _, _ = _modules(extension.__name__)
    if getattr(upstream_loader.load_hunyuan_image_3, "_comfy_tensor_parallel", False):
        return
    original = upstream_loader.load_hunyuan_image_3
    extension_path = os.path.dirname(extension.__file__)

    @wraps(original)
    def load(checkpoint_path, disable_dynamic=False):
        if int(current_execution_context().configuration.tensor_parallel_size or 1) == 1:
            return original(checkpoint_path, disable_dynamic=disable_dynamic)
        patcher = load_model(checkpoint_path, extension_path, disable_dynamic)
        return patcher, patcher.model.tokenizer

    load._comfy_tensor_parallel = True
    upstream_loader.load_hunyuan_image_3 = load
    importlib.import_module(extension.__name__ + ".nodes").load_hunyuan_image_3 = load

    # The decode fast path fetches just the selected expert. Gather after its
    # matmul without forcing an entire offloaded bank onto each GPU per token.
    upstream_ops = importlib.import_module(extension.__name__ + ".hunyuan_image_3.ops")
    matmul = upstream_ops._matmul

    @wraps(matmul)
    def expert_matmul(module, *args, **kwargs):
        output = matmul(module, *args, **kwargs)
        if getattr(module, "tensor_parallel_output", False):
            return module.tensor_parallel.operations.gather(output)
        return output

    upstream_ops._matmul = expert_matmul
    rewrite = importlib.import_module(extension.__name__ + ".hunyuan_image_3.rewrite")
    generate = rewrite.generate_text

    @wraps(generate)
    def generate_text(model, *args, **kwargs):
        executor = getattr(model, "_comfy_tensor_parallel_executor", None)
        if executor is not None:
            model = _TextModel(model, executor)
        return generate(model, *args, **kwargs)

    rewrite.generate_text = generate_text
