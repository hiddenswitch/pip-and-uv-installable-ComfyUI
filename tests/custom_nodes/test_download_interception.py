from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

from .conftest import (
    install_custom_node_from_spec,
    make_base_dirs,
    build_config,
)
from .node_registry import get_spec

logger = logging.getLogger(__name__)


class TestInterceptionMechanics:

    def test_hf_hub_download_is_intercepted(self):
        import huggingface_hub
        from comfy.nodes.download_interception import patch_hf_hub_download

        original = huggingface_hub.hf_hub_download
        with patch_hf_hub_download():
            assert huggingface_hub.hf_hub_download is not original
        assert huggingface_hub.hf_hub_download is original

    def test_snapshot_download_is_intercepted(self):
        import huggingface_hub
        from comfy.nodes.download_interception import patch_snapshot_download

        original = huggingface_hub.snapshot_download
        with patch_snapshot_download():
            assert huggingface_hub.snapshot_download is not original
        assert huggingface_hub.snapshot_download is original

    def test_folder_paths_functions_are_intercepted(self):
        from comfy.cmd import folder_paths
        from comfy import model_downloader
        from comfy.nodes.download_interception import patch_folder_paths_functions

        original_gfp = folder_paths.get_full_path
        with patch_folder_paths_functions():
            assert folder_paths.get_full_path is model_downloader.get_full_path
            assert folder_paths.get_full_path_or_raise is model_downloader.get_full_path_or_raise
            assert folder_paths.get_filename_list is model_downloader.get_filename_list
        assert folder_paths.get_full_path is original_gfp

    def test_folder_names_setitem_intercepted(self):
        from comfy.nodes.download_interception import patch_folder_names_dict
        from comfy.component_model.folder_path_types import FolderNames

        original_setitem = FolderNames.__setitem__
        with patch_folder_names_dict():
            assert FolderNames.__setitem__ is not original_setitem
            with patch("comfy.cmd.folder_paths.add_model_folder_path") as mock_amfp:
                fn = FolderNames()
                fn["unet_gguf"] = (["/some/path"], {".gguf"})
                assert mock_amfp.called
                call_args = mock_amfp.call_args
                assert call_args[0][0] == "unet_gguf"
                assert call_args[0][1] == "/some/path"
        assert FolderNames.__setitem__ is original_setitem

    def test_torch_hub_download_is_intercepted(self):
        import torch.hub
        from comfy.nodes.download_interception import patch_torch_downloads

        original = torch.hub.download_url_to_file
        with patch_torch_downloads():
            assert torch.hub.download_url_to_file is not original
        assert torch.hub.download_url_to_file is original

    def test_snapshot_download_routes_through_model_downloader(self):
        import huggingface_hub
        from comfy.nodes.download_interception import patch_snapshot_download

        with patch_snapshot_download():
            with patch("comfy.model_downloader.get_or_download_huggingface_repo") as mock_dl:
                mock_dl.return_value = "/fake/path/to/repo"
                result = huggingface_hub.snapshot_download(
                    repo_id="Kijai/sam2-safetensors",
                    local_dir="/should/be/ignored",
                    allow_patterns=["*.safetensors"],
                )
                assert result == "/fake/path/to/repo"
                mock_dl.assert_called_once_with(
                    "Kijai/sam2-safetensors",
                    allow_patterns=["*.safetensors"],
                    ignore_patterns=None,
                )

    def test_hf_hub_download_routes_through_model_downloader(self):
        import huggingface_hub
        from comfy.nodes.download_interception import patch_hf_hub_download

        with patch_hf_hub_download():
            with patch("comfy.model_downloader.get_or_download") as mock_dl:
                mock_dl.return_value = "/fake/path/to/model.pth"
                result = huggingface_hub.hf_hub_download(
                    repo_id="lllyasviel/Annotators",
                    filename="body_pose_model.pth",
                    local_dir="/should/be/ignored",
                )
                assert result == "/fake/path/to/model.pth"
                assert mock_dl.called
                args = mock_dl.call_args
                assert args[0][0] == "huggingface"

    def test_hf_hub_download_with_subfolder(self):
        import huggingface_hub
        from comfy.nodes.download_interception import patch_hf_hub_download

        with patch_hf_hub_download():
            with patch("comfy.model_downloader.get_or_download") as mock_dl:
                mock_dl.return_value = "/fake/path"
                huggingface_hub.hf_hub_download(
                    repo_id="org/repo",
                    filename="model.bin",
                    subfolder="weights",
                )
                known_files = mock_dl.call_args[1]["known_files"]
                assert known_files[0].filename == "weights/model.bin"


class TestPipInterception:

    def test_subprocess_run_with_s_flag(self):
        import subprocess
        import sys
        from comfy_compatibility.vanilla import patch_pip_install_subprocess_run

        with patch_pip_install_subprocess_run():
            result = subprocess.run(
                [sys.executable, "-s", "-m", "pip", "install", "fake-pkg"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0

    def test_subprocess_run_without_s_flag(self):
        import subprocess
        import sys
        from comfy_compatibility.vanilla import patch_pip_install_subprocess_run

        with patch_pip_install_subprocess_run():
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "fake-pkg"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0

    def test_subprocess_check_call(self):
        import subprocess
        import sys
        from comfy_compatibility.vanilla import patch_pip_install_subprocess_run

        with patch_pip_install_subprocess_run():
            ret = subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "fake-pkg"],
            )
            assert ret == 0

    def test_subprocess_popen_uv_pip(self):
        import subprocess
        import sys
        from comfy_compatibility.vanilla import patch_pip_install_popen

        with patch_pip_install_popen():
            proc = subprocess.Popen(
                [sys.executable, "-m", "uv", "pip", "install", "fake-pkg"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert proc.stdout == []

    def test_non_pip_subprocess_not_blocked(self):
        import subprocess
        import sys
        from comfy_compatibility.vanilla import patch_pip_install_subprocess_run

        with patch_pip_install_subprocess_run():
            result = subprocess.run(
                [sys.executable, "-c", "print('hello')"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert "hello" in result.stdout


@pytest.mark.slow
@pytest.mark.git_clone
class TestWanVideoWrapperDownloadInterception:

    def _setup_base(self, tmp_path):
        base_dir = tmp_path / "base"
        make_base_dirs(base_dir)

        spec = get_spec("ComfyUI-WanVideoWrapper")
        install_custom_node_from_spec(spec, base_dir)
        for dep_id in spec.depends_on:
            install_custom_node_from_spec(get_spec(dep_id), base_dir)

        config = build_config(base_dir, torch_device="cuda:1")
        return config, base_dir

    @pytest.mark.asyncio
    async def test_wanvideo_wrapper_loaders(self, tmp_path):
        from comfy.client.embedded_comfy_client import Comfy

        config, base_dir = self._setup_base(tmp_path)

        api_prompt = {
            "1": {
                "class_type": "WanVideoModelLoader",
                "inputs": {
                    "model": "wan2.1_t2v_1.3B_bf16.safetensors",
                    "base_precision": "bf16",
                    "quantization": "disabled",
                    "load_device": "offload_device",
                },
            },
            "2": {
                "class_type": "WanVideoVAELoader",
                "inputs": {
                    "model_name": "wan_2.1_vae.safetensors",
                    "precision": "bf16",
                },
            },
            "3": {
                "class_type": "WanVideoTextEncodeCached",
                "inputs": {
                    "model_name": "umt5_xxl_fp16.safetensors",
                    "precision": "bf16",
                    "positive_prompt": "A cat walks on the grass",
                    "negative_prompt": "bad quality",
                    "quantization": "disabled",
                    "use_disk_cache": False,
                    "device": "gpu",
                },
            },
            "4": {
                "class_type": "WanVideoEmptyEmbeds",
                "inputs": {
                    "width": 480,
                    "height": 320,
                    "num_frames": 17,
                },
            },
            "5": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": ["1", 0],
                    "text_embeds": ["3", 0],
                    "image_embeds": ["4", 0],
                    "steps": 2,
                    "cfg": 1.0,
                    "shift": 5.0,
                    "seed": 42,
                    "force_offload": True,
                    "scheduler": "unipc",
                    "riflex_freq_index": 0,
                },
            },
            "6": {
                "class_type": "WanVideoDecode",
                "inputs": {
                    "vae": ["2", 0],
                    "samples": ["5", 0],
                    "enable_vae_tiling": False,
                    "tile_x": 272,
                    "tile_y": 272,
                    "tile_stride_x": 144,
                    "tile_stride_y": 128,
                },
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "wanvideo_test",
                },
            },
        }

        async with Comfy(configuration=config) as client:
            outputs = await client.queue_prompt(api_prompt)
            logger.info("Execution outputs: %s", list(outputs.keys()))

        models_dir = base_dir / "models"
        symlink_count = 0
        for folder in ("diffusion_models", "vae", "text_encoders"):
            folder_path = models_dir / folder
            files = list(folder_path.rglob("*")) if folder_path.exists() else []
            real_files = [f for f in files if f.is_file() or f.is_symlink()]
            for f in real_files:
                if f.is_symlink():
                    target = os.readlink(f)
                    logger.info("  %s/%s -> %s (symlink)", folder, f.relative_to(folder_path), target)
                    symlink_count += 1
                else:
                    logger.info("  %s/%s (regular file)", folder, f.relative_to(folder_path))

        assert symlink_count > 0, "Expected models to be symlinked from HF cache"
