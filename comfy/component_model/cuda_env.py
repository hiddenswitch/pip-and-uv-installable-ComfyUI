from __future__ import annotations

import os
from collections.abc import MutableMapping


_DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"
_XPU_ENV_KEYS = (
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
    "ZE_AFFINITY_MASK",
)


def should_skip_cuda_alloc_conf_for_xpu(
    env: MutableMapping[str, str] | None = None,
) -> bool:
    env = os.environ if env is None else env

    for key in _XPU_ENV_KEYS:
        value = env.get(key)
        if value and value.strip():
            return True

    if env.get("CUDA_VISIBLE_DEVICES") or env.get("HIP_VISIBLE_DEVICES"):
        return False

    dev_dir = "/dev/dri"
    if not os.path.isdir(dev_dir):
        return False

    has_render_node = any(name.startswith("renderD") for name in os.listdir(dev_dir))
    has_cuda_node = any(name.startswith("nvidia") for name in os.listdir("/dev") if os.path.exists("/dev"))
    has_rocm_node = os.path.exists("/dev/kfd")
    return has_render_node and not has_cuda_node and not has_rocm_node


def ensure_pytorch_cuda_alloc_conf(
    env: MutableMapping[str, str] | None = None,
    *,
    skip_for_xpu: bool = False,
) -> str:
    env = os.environ if env is None else env
    current = env.get("PYTORCH_CUDA_ALLOC_CONF")
    if (current is None or not current.strip()) and not skip_for_xpu:
        env["PYTORCH_CUDA_ALLOC_CONF"] = _DEFAULT_CUDA_ALLOC_CONF
    return env.get("PYTORCH_CUDA_ALLOC_CONF", "")
