import comfy_aimdo.model_vbar
from . import memory_management, model_management, ops

PREFETCH_QUEUES = []


def _dynamic_vbar_modules(module):
    return [s for s in module.modules() if getattr(s, "_v", None) is not None]

def cleanup_prefetched_modules(comfy_modules):
    for s in comfy_modules:
        prefetch = getattr(s, "_prefetch", None)
        if prefetch is None:
            continue
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                lowvram_fn.clear_prepared()
        if prefetch["signature"] is not None:
            comfy_aimdo.model_vbar.vbar_unpin(s._v)
        delattr(s, "_prefetch")

def cleanup_prefetch_queues():
    global PREFETCH_QUEUES

    for queue in PREFETCH_QUEUES:
        for entry in queue:
            if entry is None or not isinstance(entry, tuple):
                continue
            _, prefetch_state = entry
            comfy_modules = prefetch_state[1]
            if comfy_modules is not None:
                cleanup_prefetched_modules(comfy_modules)
    PREFETCH_QUEUES = []


def finish_model_execution():
    """Release asynchronous weight state at an outer model boundary."""
    cleanup_prefetch_queues()
    ops.finish_weight_cast_execution()

def prefetch_queue_pop(queue, device, module):
    if queue is None:
        return

    consumed = queue.pop(0)
    if consumed is not None:
        offload_stream, prefetch_state = consumed
        if offload_stream is not None:
            offload_stream.wait_stream(model_management.current_stream(device))
        _, comfy_modules = prefetch_state
        if comfy_modules is not None:
            cleanup_prefetched_modules(comfy_modules)

    prefetch = queue[0]
    if prefetch is not None:
        comfy_modules = _dynamic_vbar_modules(prefetch)

        registerable_size = 0
        for s in comfy_modules:
            registerable_size += memory_management.vram_aligned_size([s.weight, s.bias])
            for param_key in ("weight", "bias"):
                lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
                if lowvram_fn is not None:
                    registerable_size += lowvram_fn.memory_required()

        offload_stream = ops.cast_modules_with_vbar(comfy_modules, None, device, None, True)
        if not model_management.args.fast_disk:
            model_management.ensure_pin_registerable(registerable_size)
        model_management.sync_stream(device, offload_stream)
        queue[0] = (offload_stream, (prefetch, comfy_modules))

def make_prefetch_queue(queue, device, transformer_options):
    if (not transformer_options.get("prefetch_dynamic_vbars", False)
        or model_management.NUM_STREAMS == 0
        or model_management.is_device_cpu(device)
        or not model_management.device_supports_non_blocking(device)):
        return None

    queue = [None] + queue + [None]
    PREFETCH_QUEUES.append(queue)
    return queue
