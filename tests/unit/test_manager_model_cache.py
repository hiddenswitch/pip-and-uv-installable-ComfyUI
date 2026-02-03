"""
Unit tests for manager model cache functionality.
"""
import pytest

from comfy.manager_model_cache import (
    parse_huggingface_url,
    ManagerModelEntry,
    entry_to_downloadable,
    _resolve_folder,
    FOLDER_TO_MANAGER_TYPES,
)
from comfy.model_downloader_types import HuggingFile, UrlFile


class TestParseHuggingfaceUrl:
    def test_standard_resolve_url(self):
        url = "https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/sd3.5_large_controlnet_blur.safetensors"
        result = parse_huggingface_url(url, "sd3.5_large_controlnet_blur.safetensors")
        assert result is not None
        assert result.repo_id == "stabilityai/stable-diffusion-3.5-controlnets"
        assert result.filename == "sd3.5_large_controlnet_blur.safetensors"
        assert result.revision is None  # main is default
        assert result.show_in_ui is False

    def test_url_with_revision(self):
        url = "https://huggingface.co/org/repo/resolve/v1.0/model.safetensors"
        result = parse_huggingface_url(url, "model.safetensors")
        assert result is not None
        assert result.revision == "v1.0"

    def test_url_with_subpath(self):
        url = "https://huggingface.co/org/repo/resolve/main/models/diffusion/model.safetensors"
        result = parse_huggingface_url(url, "model.safetensors")
        assert result is not None
        assert result.filename == "models/diffusion/model.safetensors"
        assert result.save_with_filename == "model.safetensors"

    def test_non_huggingface_url(self):
        url = "https://civitai.com/api/download/models/12345"
        result = parse_huggingface_url(url, "model.safetensors")
        assert result is None

    def test_blob_url_not_matched(self):
        """blob URLs should not be matched, only resolve URLs"""
        url = "https://huggingface.co/org/repo/blob/main/model.safetensors"
        result = parse_huggingface_url(url, "model.safetensors")
        assert result is None


class TestResolveFolder:
    def test_resolve_folder_from_type_checkpoint(self):
        assert _resolve_folder("default", "checkpoint") == "checkpoints"

    def test_resolve_folder_from_type_lora(self):
        assert _resolve_folder("default", "lora") == "loras"

    def test_resolve_folder_from_type_controlnet(self):
        assert _resolve_folder("default", "controlnet") == "controlnet"

    def test_resolve_folder_from_type_unet(self):
        assert _resolve_folder("default", "unet") == "diffusion_models"

    def test_resolve_folder_from_save_path(self):
        assert _resolve_folder("checkpoints/SDXL", "checkpoint") == "checkpoints"

    def test_resolve_folder_from_nested_save_path(self):
        assert _resolve_folder("loras/HyperSD/SDXL", "lora") == "loras"

    def test_resolve_folder_from_type_vae(self):
        assert _resolve_folder("default", "vae") == "vae"

    def test_resolve_folder_from_type_diffusion_model(self):
        assert _resolve_folder("default", "diffusion_model") == "diffusion_models"

    def test_resolve_folder_from_type_embedding(self):
        assert _resolve_folder("default", "embedding") == "embeddings"

    def test_resolve_folder_unknown_type_uses_type_as_folder(self):
        assert _resolve_folder("default", "some_unknown_type") == "some_unknown_type"


class TestFolderToManagerTypes:
    def test_all_common_folders_have_mappings(self):
        required_folders = ["checkpoints", "loras", "vae", "controlnet", "diffusion_models"]
        for folder in required_folders:
            assert folder in FOLDER_TO_MANAGER_TYPES, f"Missing mapping for {folder}"

    def test_checkpoint_types(self):
        assert "checkpoint" in FOLDER_TO_MANAGER_TYPES["checkpoints"]
        assert "checkpoints" in FOLDER_TO_MANAGER_TYPES["checkpoints"]

    def test_unet_maps_to_diffusion_models(self):
        assert "unet" in FOLDER_TO_MANAGER_TYPES["diffusion_models"]
        assert "diffusion_model" in FOLDER_TO_MANAGER_TYPES["diffusion_models"]

    def test_text_encoders_mapping(self):
        assert "clip" in FOLDER_TO_MANAGER_TYPES["text_encoders"]
        assert "text_encoders" in FOLDER_TO_MANAGER_TYPES["text_encoders"]

    def test_embeddings_mapping(self):
        assert "embedding" in FOLDER_TO_MANAGER_TYPES["embeddings"]
        assert "embeddings" in FOLDER_TO_MANAGER_TYPES["embeddings"]


class TestEntryToDownloadable:
    def test_huggingface_url_converts_to_huggingfile(self):
        entry = ManagerModelEntry(
            name="Test Model",
            type="checkpoint",
            base="SDXL",
            save_path="default",
            filename="model.safetensors",
            url="https://huggingface.co/org/repo/resolve/main/model.safetensors"
        )
        result = entry_to_downloadable(entry)
        assert isinstance(result, HuggingFile)
        assert result.repo_id == "org/repo"
        assert result.filename == "model.safetensors"
        assert result.show_in_ui is False

    def test_non_huggingface_url_converts_to_urlfile(self):
        entry = ManagerModelEntry(
            name="Test Model",
            type="checkpoint",
            base="SD1.5",
            save_path="default",
            filename="model.safetensors",
            url="https://example.com/model.safetensors"
        )
        result = entry_to_downloadable(entry)
        assert isinstance(result, UrlFile)
        assert result.url == "https://example.com/model.safetensors"
        assert result.show_in_ui is False

    def test_empty_url_returns_none(self):
        entry = ManagerModelEntry(
            name="Test Model",
            type="checkpoint",
            base="SD1.5",
            save_path="default",
            filename="model.safetensors",
            url=""
        )
        result = entry_to_downloadable(entry)
        assert result is None
