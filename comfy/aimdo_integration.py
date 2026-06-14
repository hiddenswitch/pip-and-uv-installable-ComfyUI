import logging
import importlib

logger = logging.getLogger(__name__)

from . import memory_management
from . import model_management

import comfy_aimdo.control
import comfy_aimdo.host_buffer
import comfy_aimdo.model_vbar
import comfy_aimdo.torch
import comfy_aimdo.vram_buffer
import torch

from .cli_args import args, dynamic_vram_requested, dynamic_vram_supported, enables_dynamic_vram

if dynamic_vram_requested() and not dynamic_vram_supported():
    logger.warning("Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later. Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
    memory_management.aimdo_allocator = None
elif enables_dynamic_vram() and model_management.get_torch_device().type == "cuda":
    torch_device = model_management.get_torch_device()
    torch.cuda.init()
    device_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()

    if comfy_aimdo.control.init():
        importlib.reload(comfy_aimdo.host_buffer)
        importlib.reload(comfy_aimdo.model_vbar)
        importlib.reload(comfy_aimdo.vram_buffer)

    if comfy_aimdo.control.lib is not None and comfy_aimdo.control.init_device(device_index):
        if args.verbose == 'DEBUG':
            comfy_aimdo.control.set_log_debug()
        elif args.verbose == 'CRITICAL':
            comfy_aimdo.control.set_log_critical()
        elif args.verbose == 'ERROR':
            comfy_aimdo.control.set_log_error()
        elif args.verbose == 'WARNING':
            comfy_aimdo.control.set_log_warning()
        else: #INFO
            comfy_aimdo.control.set_log_info()

        memory_management.aimdo_allocator = comfy_aimdo.torch.get_torch_allocator()
        logger.info("DynamicVRAM support detected and enabled")
    else:
        logger.info("No working comfy-aimdo install detected. DynamicVRAM support disabled. Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
        memory_management.aimdo_allocator = None
else:
    # Dynamic VRAM (comfy-aimdo) is CUDA-only. On CPU-only environments (the
    # headless serve-pip facade, CPU inference) leave the legacy ModelPatcher in
    # place; init_device(None) would otherwise crash startup.
    memory_management.aimdo_allocator = None
