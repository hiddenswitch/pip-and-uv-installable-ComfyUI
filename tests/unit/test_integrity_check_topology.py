import torch
from rich.console import Console

from comfy.cmd import integrity_check
from comfy.distributed import topology


def _render(section):
    console = Console(record=True, width=200, force_terminal=False, color_system=None)
    section(console)
    return console.export_text()


def test_interconnect_section_reports_links_and_parallel_sizes(monkeypatch):
    devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    monkeypatch.setattr(topology, "cuda_devices", lambda: devices)
    monkeypatch.setattr(topology, "device_names", lambda devices: ["NVIDIA RTX A5000", "NVIDIA RTX A5000"])
    monkeypatch.setattr(topology, "nccl_available", lambda: True)
    monkeypatch.setattr(topology, "is_windows", lambda: False)
    monkeypatch.setattr(
        topology,
        "measure_device_links",
        lambda devices, **kwargs: [
            topology.DeviceLink(devices[0], devices[1], True, 48.5e9),
            topology.DeviceLink(devices[1], devices[0], True, 47.9e9),
        ],
    )
    monkeypatch.setattr(integrity_check.model_management, "get_total_memory", lambda device: 24 * 1024 ** 3)

    text = _render(integrity_check._section_interconnect)

    assert "NVIDIA RTX A5000" in text
    assert "48.5 GB/s" in text
    assert "47.9 GB/s" in text
    assert "peer" in text
    assert "Max tensor parallel size" in text and "2" in text
    assert "Max pipeline parallel size" in text


def test_interconnect_section_single_device(monkeypatch):
    monkeypatch.setattr(topology, "cuda_devices", lambda: [torch.device("cuda", 0)])
    monkeypatch.setattr(topology, "device_names", lambda devices: ["NVIDIA RTX A5000"])
    monkeypatch.setattr(topology, "nccl_available", lambda: True)
    monkeypatch.setattr(topology, "is_windows", lambda: False)
    monkeypatch.setattr(topology, "measure_device_links", lambda devices, **kwargs: [])
    monkeypatch.setattr(integrity_check.model_management, "get_total_memory", lambda device: 24 * 1024 ** 3)

    text = _render(integrity_check._section_interconnect)

    assert "Max tensor parallel size" in text
    assert "Max pipeline parallel size" in text
    assert "one CUDA device" in text


def test_interconnect_section_without_cuda(monkeypatch):
    monkeypatch.setattr(topology, "cuda_devices", lambda: [])

    text = _render(integrity_check._section_interconnect)

    assert "No CUDA devices" in text


def test_guess_settings_section_reports_tensor_parallel_decision(monkeypatch):
    from comfy.component_model import guess_settings

    monkeypatch.setattr(guess_settings, "_has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(guess_settings, "_has_amd_gpu", lambda: False)
    monkeypatch.setattr(guess_settings, "_total_ram_gb", lambda: 128.0)
    monkeypatch.setattr(guess_settings, "_competing_gpu_processes", lambda: [])
    monkeypatch.setattr(guess_settings, "_nvidia_compute_caps", lambda: [(8, 6), (8, 6)])
    monkeypatch.setattr(guess_settings, "_nvidia_gpu_names", lambda: ["NVIDIA RTX A5000", "NVIDIA RTX A5000"])
    monkeypatch.setattr(guess_settings, "_has_package", lambda name: False)
    monkeypatch.setattr(guess_settings.os, "name", "posix")
    monkeypatch.setattr(guess_settings.sys, "platform", "linux")
    for name in ("CUDA_VISIBLE_DEVICES", "RANK", "WORLD_SIZE", "COMFYUI_TENSOR_PARALLEL_SIZE", "COMFYUI_PIPELINE_PARALLEL_SIZE"):
        monkeypatch.delenv(name, raising=False)

    text = _render(integrity_check._section_guess_settings)

    assert "NVIDIA GPUs" in text
    assert "tensor_parallel_size" in text
    assert "2 identical NVIDIA GPUs" in text
