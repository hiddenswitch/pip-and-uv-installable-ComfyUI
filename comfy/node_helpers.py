import hashlib
import logging

from PIL import ImageFile, UnidentifiedImageError

from comfy_api.latest import io
from .component_model.files import get_package_as_path
from .execution_context import current_execution_context

logger = logging.getLogger(__name__)


def conditioning_set_values(conditioning, values: dict = None, append=False) -> io.Conditioning.CondList:
    if values is None:
        values = {}
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c


def conditioning_set_values_with_timestep_range(conditioning, values={}, start_percent=0.0, end_percent=1.0):
    """
    Apply values to conditioning only during [start_percent, end_percent], keeping the
    original conditioning active outside that range. Respects existing per-entry ranges.
    """
    if start_percent > end_percent:
        logger.warning(f"start_percent ({start_percent}) must be <= end_percent ({end_percent})")
        return conditioning

    EPS = 1e-5  # the sampler gates entries with strict > / <, shift boundaries slightly to ensure only one conditioning is active per timestep
    c = []
    for t in conditioning:
        cond_start = t[1].get("start_percent", 0.0)
        cond_end = t[1].get("end_percent", 1.0)
        intersect_start = max(start_percent, cond_start)
        intersect_end = min(end_percent, cond_end)

        if intersect_start >= intersect_end:  # no overlap: emit unchanged
            c.append(t)
            continue

        if intersect_start > cond_start:  # part before the requested range
            c.extend(conditioning_set_values([t], {"start_percent": cond_start, "end_percent": intersect_start - EPS}))

        c.extend(conditioning_set_values([t], {**values, "start_percent": intersect_start, "end_percent": intersect_end}))

        if intersect_end < cond_end:  # part after the requested range
            c.extend(conditioning_set_values([t], {"start_percent": intersect_end + EPS, "end_percent": cond_end}))
    return c


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):  # PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def hasher():
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    args = current_execution_context().configuration
    return hashfuncs[args.default_hashing_function]


def export_custom_nodes():
    """
    Finds all non-abstract classes in the current module that extend CustomNode and creates
    a NODE_CLASS_MAPPINGS dictionary mapping class names to class objects.
    Must be called from within the module where the CustomNode classes are defined.
    """
    import inspect
    from .nodes.package_typing import CustomNode

    # Get the calling module
    frame = inspect.currentframe()
    try:
        module = inspect.getmodule(frame.f_back)

        custom_nodes = {}
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                    CustomNode in obj.__mro__ and
                    obj != CustomNode and
                    not inspect.isabstract(obj)):
                custom_nodes[name] = obj
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            node_class_mappings: dict = getattr(module, 'NODE_CLASS_MAPPINGS')
            node_class_mappings.update(custom_nodes)
        else:
            setattr(module, 'NODE_CLASS_MAPPINGS', custom_nodes)

    finally:
        # Clean up circular reference
        del frame

    return custom_nodes


def export_package_as_web_directory(package: str):
    import inspect

    # Get the calling module
    frame = inspect.currentframe()
    try:
        module = inspect.getmodule(frame.f_back)
        setattr(module, 'WEB_DIRECTORY', get_package_as_path(package))

    finally:
        # Clean up circular reference
        del frame


def string_to_torch_dtype(string):
    import torch
    if string == "fp32":
        return torch.float32
    if string == "fp16":
        return torch.float16
    if string == "bf16":
        return torch.bfloat16


def image_alpha_fix(destination, source):
    import torch
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., :destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        destination = torch.nn.functional.pad(destination, (0, 1))
        destination[..., -1] = 1.0
    return destination, source
