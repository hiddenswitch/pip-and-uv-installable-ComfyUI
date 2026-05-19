import comfy_aimdo.host_buffer
import comfy_aimdo.torch
import torch

from . import memory_management
from . import model_management
from .cli_args import args

def _geometry_key(geometry):
    if geometry is None:
        return None
    key = []
    for item in geometry:
        if item is None:
            key.append(None)
        elif hasattr(item, "shape") and hasattr(item, "dtype"):
            key.append((tuple(item.shape), item.dtype))
        else:
            key.append((tuple(item.shape), item.dtype))
    return tuple(key)


def get_pin(module, geometry=None):
    key = _geometry_key(geometry)
    if key is None:
        return getattr(module, "_pin", None)
    return getattr(module, "_direct_pins", {}).get(key)


def _event_key(geometry=None):
    key = _geometry_key(geometry)
    return "__default__" if key is None else key


def wait_pin_ready(module, geometry=None):
    events = getattr(module, "_pin_ready_events", None)
    if events is None:
        return
    event = events.get(_event_key(geometry))
    if event is not None:
        event.synchronize()


def record_pin_use(module, geometry=None, stream=None, device=None):
    if stream is None or device is None or getattr(device, "type", None) != "cuda":
        return
    event = torch.cuda.Event()
    event.record(stream)
    events = getattr(module, "_pin_ready_events", None)
    if events is None:
        events = {}
        module._pin_ready_events = events
    events[_event_key(geometry)] = event


def pin_memory(module, geometry=None):
    if module.pin_failed or args.disable_pinned_memory or get_pin(module, geometry) is not None:
        return

    size = memory_management.vram_aligned_size(geometry if geometry is not None else [ module.weight, module.bias ])

    if model_management.MAX_PINNED_MEMORY <= 0 or (model_management.TOTAL_PINNED_MEMORY + size) > model_management.MAX_PINNED_MEMORY:
        module.pin_failed = True
        return False

    try:
        hostbuf = comfy_aimdo.host_buffer.HostBuffer(size)
    except RuntimeError:
        module.pin_failed = True
        return False

    pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)
    key = _geometry_key(geometry)
    if key is None:
        module._pin = pin
        module._pin_hostbuf = hostbuf
    else:
        direct_pins = getattr(module, "_direct_pins", None)
        direct_pin_hostbufs = getattr(module, "_direct_pin_hostbufs", None)
        if direct_pins is None:
            direct_pins = {}
            direct_pin_hostbufs = {}
            module._direct_pins = direct_pins
            module._direct_pin_hostbufs = direct_pin_hostbufs
        direct_pins[key] = pin
        direct_pin_hostbufs[key] = hostbuf
    model_management.TOTAL_PINNED_MEMORY += size
    return True

def pin_size(module):
    total = 0
    pin = get_pin(module)
    if pin is not None:
        total += pin.numel() * pin.element_size()
    for pin in getattr(module, "_direct_pins", {}).values():
        total += pin.numel() * pin.element_size()
    return total

def unpin_memory(module):
    size = pin_size(module)
    if size == 0:
        return 0

    # todo: needs merge, this is per process or per machine or...? should this be migrated to the execution context?
    model_management.TOTAL_PINNED_MEMORY -= size
    if model_management.TOTAL_PINNED_MEMORY < 0:
        model_management.TOTAL_PINNED_MEMORY = 0

    for attr in ("_pin", "_pin_hostbuf", "_direct_pins", "_direct_pin_hostbufs", "_pin_ready_events"):
        if hasattr(module, attr):
            delattr(module, attr)
    return size
