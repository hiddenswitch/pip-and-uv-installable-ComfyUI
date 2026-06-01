from comfy.component_model.cuda_env import ensure_pytorch_cuda_alloc_conf


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
