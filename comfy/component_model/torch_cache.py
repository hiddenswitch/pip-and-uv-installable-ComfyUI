from __future__ import annotations

import os

from ..vendor.appdirs import user_cache_dir


def setup_torch_compile_cache_dirs() -> None:
    """Set persistent torch compile cache directories unless the user did."""
    cache_root = os.path.join(user_cache_dir(appname="comfyui"), "torch_compile")
    cache_dirs = {
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_root, "inductor"),
        "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
        "CUDA_CACHE_PATH": os.path.join(cache_root, "cuda"),
    }
    for env_name, path in cache_dirs.items():
        if env_name not in os.environ:
            os.makedirs(path, exist_ok=True)
            os.environ[env_name] = path
