from comfy.component_model.cuda_env import ensure_pytorch_cuda_alloc_conf, should_skip_cuda_alloc_conf_for_xpu


def test_sets_expandable_segments_when_missing():
    env = {}

    assert ensure_pytorch_cuda_alloc_conf(env) == "expandable_segments:True"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_sets_expandable_segments_when_blank():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "  "}

    assert ensure_pytorch_cuda_alloc_conf(env) == "expandable_segments:True"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_preserves_existing_cuda_alloc_conf():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"}

    assert ensure_pytorch_cuda_alloc_conf(env) == "backend:cudaMallocAsync"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"


def test_skips_expandable_segments_for_xpu():
    env = {}

    assert ensure_pytorch_cuda_alloc_conf(env, skip_for_xpu=True) == ""
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env


def test_detects_oneapi_selector_as_xpu():
    env = {"ONEAPI_DEVICE_SELECTOR": "level_zero:0"}

    assert should_skip_cuda_alloc_conf_for_xpu(env)
