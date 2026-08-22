from comfy.cli_args_types import Configuration
from comfy.component_model import setup


def test_cuda_device_all_preserves_visibility_and_default_order(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,1")
    config = Configuration(cuda_device="all", default_device=1)

    setup.setup_cuda_devices(config)

    assert setup.os.environ["CUDA_VISIBLE_DEVICES"] == "3,1"
    assert setup.os.environ["HIP_VISIBLE_DEVICES"] == "3,1"


def test_windows_multiple_gpus_default_to_gpu_zero(monkeypatch):
    from comfy.app import logger as app_logger
    from comfy.cmd import cuda_malloc

    warnings = []
    monkeypatch.setattr(setup.os, "name", "nt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cuda_malloc, "get_gpu_names", lambda: ["NVIDIA RTX A5000", "NVIDIA RTX A5000"])
    monkeypatch.setattr(app_logger, "log_startup_warning", warnings.append)

    setup.setup_windows_multi_gpu_defaults(Configuration(guess_settings=False))

    assert setup.os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert len(warnings) == 1
    assert "use GPU 0 only" in warnings[0]
    assert "--cuda-device all --disable-pinned-memory" in warnings[0]


def test_windows_cuda_device_all_warns_when_pinned_memory_is_enabled(monkeypatch):
    from comfy.app import logger as app_logger
    from comfy.cmd import cuda_malloc

    warnings = []
    monkeypatch.setattr(setup.os, "name", "nt")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cuda_malloc, "get_gpu_names", lambda: ["NVIDIA RTX A5000", "NVIDIA RTX A5000"])
    monkeypatch.setattr(app_logger, "log_startup_warning", warnings.append)

    setup.setup_windows_multi_gpu_defaults(Configuration(cuda_device="all", guess_settings=False))

    assert "CUDA_VISIBLE_DEVICES" not in setup.os.environ
    assert len(warnings) == 1
    assert "pinned memory enabled" in warnings[0]
