import collections
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from comfy.cli_args_types import Configuration, PerformanceFeature
from comfy.component_model.guess_settings import (
    apply_guess_settings,
    _total_ram_gb,
    _has_nvidia_gpu,
    _has_amd_gpu,
    _competing_gpu_processes,
    _has_package,
)


def _config(**overrides) -> Configuration:
    return Configuration(**overrides)


# ---------------------------------------------------------------------------
# _total_ram_gb
# ---------------------------------------------------------------------------

class TestTotalRamGb:
    def test_psutil(self):
        mem = collections.namedtuple("svmem", ["total"])(total=16 * 1024 ** 3)
        with patch("psutil.virtual_memory", return_value=mem):
            assert abs(_total_ram_gb() - 16.0) < 0.1

    def test_fallback_proc_meminfo(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:       16384000 kB\n")
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.open", side_effect=lambda *a, **kw: open(str(meminfo), **kw) if "/proc/meminfo" in str(a) else open(*a, **kw)):
                # The ImportError path returns 0 when psutil is missing and
                # /proc/meminfo can't be opened.
                pass

    def test_returns_zero_on_failure(self):
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("builtins.open", side_effect=OSError):
            result = _total_ram_gb()
            assert result == 0.0


# ---------------------------------------------------------------------------
# _has_nvidia_gpu / _has_amd_gpu
# ---------------------------------------------------------------------------

class TestGpuDetection:
    def test_nvidia_present(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            assert _has_nvidia_gpu() is True

    def test_nvidia_absent(self):
        with patch("shutil.which", return_value=None):
            assert _has_nvidia_gpu() is False

    def test_amd_rocm_smi(self):
        def _which(name):
            return "/usr/bin/rocm-smi" if name == "rocm-smi" else None
        with patch("shutil.which", side_effect=_which), \
             patch("os.path.exists", return_value=False):
            assert _has_amd_gpu() is True

    def test_amd_dev_kfd(self):
        with patch("shutil.which", return_value=None), \
             patch("os.path.exists", return_value=True):
            assert _has_amd_gpu() is True

    def test_amd_absent(self):
        with patch("shutil.which", return_value=None), \
             patch("os.path.exists", return_value=False):
            assert _has_amd_gpu() is False


# ---------------------------------------------------------------------------
# _competing_gpu_processes
# ---------------------------------------------------------------------------

class TestCompetingGpuProcesses:
    def test_no_nvidia_smi(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _competing_gpu_processes() == []

    def test_no_processes(self):
        result = MagicMock(returncode=0, stdout="\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_only_python(self):
        result = MagicMock(returncode=0, stdout="python3\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_only_nvidia(self):
        result = MagicMock(returncode=0, stdout="nvtop\nnvidia-smi\n")
        with patch("subprocess.run", return_value=result):
            assert _competing_gpu_processes() == []

    def test_competing_processes(self):
        result = MagicMock(returncode=0, stdout="python3\nDiscord\nfirefox\n")
        with patch("subprocess.run", return_value=result):
            procs = _competing_gpu_processes()
            assert "Discord" in procs
            assert "firefox" in procs
            assert "python3" not in procs

    def test_windows_paths(self):
        result = MagicMock(returncode=0, stdout="C:\\Program Files\\Discord\\Discord.exe\nC:\\Python312\\python.exe\n")
        with patch("subprocess.run", return_value=result):
            procs = _competing_gpu_processes()
            assert "Discord.exe" in procs
            assert "python.exe" not in procs

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            assert _competing_gpu_processes() == []


# ---------------------------------------------------------------------------
# apply_guess_settings: NVIDIA
# ---------------------------------------------------------------------------

class TestGuessSettingsNvidia:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_cublas_ops_enabled(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert PerformanceFeature.CublasOps in cfg.fast

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_cublas_ops_added_to_existing_fast(self, *_mocks):
        cfg = _config()
        cfg.fast = [PerformanceFeature.Fp16Accumulation]
        apply_guess_settings(cfg)
        assert PerformanceFeature.CublasOps in cfg.fast
        assert PerformanceFeature.Fp16Accumulation in cfg.fast

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=["Discord", "firefox"])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_novram_when_competing_processes(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.novram is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=["Discord"])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_novram_not_set_when_user_set_highvram(self, *_mocks):
        cfg = _config(highvram=True)
        apply_guess_settings(cfg)
        assert cfg.novram is False
        assert cfg.highvram is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_no_novram_when_no_competing_processes(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.novram is False


# ---------------------------------------------------------------------------
# apply_guess_settings: AMD
# ---------------------------------------------------------------------------

class TestGuessSettingsAmd:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_fp32_vae_enabled(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.fp32_vae is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_fp32_vae_not_set_when_user_set_fp16(self, *_mocks):
        cfg = _config(fp16_vae=True)
        apply_guess_settings(cfg)
        assert cfg.fp32_vae is False
        assert cfg.fp16_vae is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=True)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_no_cublas_on_amd(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert PerformanceFeature.CublasOps not in cfg.fast


# ---------------------------------------------------------------------------
# apply_guess_settings: RAM
# ---------------------------------------------------------------------------

class TestGuessSettingsRam:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=16.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_low_ram_disables_pinned(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_high_ram_keeps_pinned(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is False

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=31.9)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_boundary_ram(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=32.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_exactly_32gb(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.disable_pinned_memory is False


# ---------------------------------------------------------------------------
# apply_guess_settings: attention backend
# ---------------------------------------------------------------------------

class TestGuessSettingsAttention:
    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    def test_sage_attention_preferred(self, *_mocks):
        with patch("comfy.component_model.guess_settings._has_package", side_effect=lambda n: n == "sageattention"):
            cfg = _config()
            apply_guess_settings(cfg)
            assert cfg.use_sage_attention is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    def test_xformers_fallback(self, *_mocks):
        with patch("comfy.component_model.guess_settings._has_package", side_effect=lambda n: n == "xformers"):
            cfg = _config()
            apply_guess_settings(cfg)
            assert cfg.use_sage_attention is False
            assert cfg.disable_xformers is False

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    @patch("comfy.component_model.guess_settings._has_package", return_value=False)
    def test_pytorch_attention_fallback(self, *_mocks):
        cfg = _config()
        apply_guess_settings(cfg)
        assert cfg.use_pytorch_cross_attention is True

    @patch("comfy.component_model.guess_settings._has_nvidia_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._has_amd_gpu", return_value=False)
    @patch("comfy.component_model.guess_settings._total_ram_gb", return_value=64.0)
    @patch("comfy.component_model.guess_settings._competing_gpu_processes", return_value=[])
    def test_user_set_attention_not_overridden(self, *_mocks):
        with patch("comfy.component_model.guess_settings._has_package", side_effect=lambda n: n == "sageattention"):
            cfg = _config(use_flash_attention=True)
            apply_guess_settings(cfg)
            assert cfg.use_sage_attention is False
            assert cfg.use_flash_attention is True


# ---------------------------------------------------------------------------
# CLI parsing integration
# ---------------------------------------------------------------------------

class TestGuessSettingsCliArg:
    def test_default_is_false(self):
        from tests.unit.test_cli_args import _parse_test_args
        cfg = _parse_test_args([])
        assert cfg.guess_settings is False

    def test_flag_is_true(self):
        from tests.unit.test_cli_args import _parse_test_args
        cfg = _parse_test_args(["--guess-settings"])
        assert cfg.guess_settings is True
