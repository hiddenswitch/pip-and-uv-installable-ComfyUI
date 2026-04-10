import torch
import comfy_aimdo.host_buffer
import comfy_aimdo.torch
import psutil

from . import memory_management
from . import model_management
from .cli_args import args

def get_pin(module):
    return getattr(module, "_pin", None)

def pin_memory(module):
    if module.pin_failed or args.disable_pinned_memory or get_pin(module) is not None:
        return
    #FIXME: This is a RAM cache trigger event
    ram_headroom = memory_management.RAM_CACHE_HEADROOM
    #we split the difference and assume half the RAM cache headroom is for us
    if ram_headroom > 0 and psutil.virtual_memory().available < (ram_headroom * 0.5):
        memory_management.extra_ram_release(ram_headroom)

    size = memory_management.vram_aligned_size([ module.weight, module.bias ])

    if model_management.MAX_PINNED_MEMORY <= 0 or (model_management.TOTAL_PINNED_MEMORY + size) > model_management.MAX_PINNED_MEMORY:
        module.pin_failed = True
        return False

    try:
        hostbuf = comfy_aimdo.host_buffer.HostBuffer(size)
    except RuntimeError:
        module.pin_failed = True
        return False

    module._pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)
    module._pin_hostbuf = hostbuf
    model_management.TOTAL_PINNED_MEMORY += size
    return True

def unpin_memory(module):
    if get_pin(module) is None:
        return 0
    size = module._pin.numel() * module._pin.element_size()

    # todo: needs merge, this is per process or per machine or...? should this be migrated to the execution context?
    model_management.TOTAL_PINNED_MEMORY -= size
    if model_management.TOTAL_PINNED_MEMORY < 0:
        model_management.TOTAL_PINNED_MEMORY = 0

    del module._pin
    del module._pin_hostbuf
    return size
