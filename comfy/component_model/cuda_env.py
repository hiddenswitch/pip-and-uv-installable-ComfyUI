from __future__ import annotations

import os
from collections.abc import MutableMapping


_DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


def ensure_pytorch_cuda_alloc_conf(
    env: MutableMapping[str, str] | None = None,
) -> str:
    env = os.environ if env is None else env
    current = env.get("PYTORCH_CUDA_ALLOC_CONF")
    if current is None or not current.strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = _DEFAULT_CUDA_ALLOC_CONF
    return env["PYTORCH_CUDA_ALLOC_CONF"]
