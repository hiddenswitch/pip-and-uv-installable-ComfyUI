"""
Tests for URI handling in model_downloader.

Tests that hf:// URIs are properly detected and downloaded to local cache,
returning local file paths that work with downstream code.
"""
import json
import os

import pytest
from unittest.mock import patch

from comfy.cmd import folder_paths
from comfy.component_model.folder_path_types import FolderPathsTuple
from comfy.component_model.uris import is_hf_uri, is_uri
from comfy.model_downloader import parse_hf_uri, get_or_download
from comfy.model_downloader_types import HuggingFile


class TestIsHfUri:
    """Tests for the is_hf_uri helper function."""

    @pytest.mark.parametrize("uri", [
        "hf://Comfy-Org/flux1-schnell/flux1-schnell-fp8.safetensors",
        "hf://gpt2/config.json",
        "hf://datasets/squad/train.json",
        "hf://spaces/gradio/hello_world/app.py",
    ])
    def test_valid_hf_uris(self, uri):
        """Test that valid hf:// URIs are detected."""
        assert is_hf_uri(uri) is True

    @pytest.mark.parametrize("path", [
        "https://example.com/model.safetensors",
        "s3://bucket/path/model.safetensors",
        "model.safetensors",
        "/path/to/model.safetensors",
        "relative/path/model.safetensors",
        "",
    ])
    def test_non_hf_uris(self, path):
        """Test that non-hf:// paths are not detected as HF URIs."""
        assert is_hf_uri(path) is False


class TestIsUri:
    """Tests for the is_uri helper function."""

    @pytest.mark.parametrize("uri", [
        "hf://gpt2/config.json",
        "https://example.com/model.safetensors",
        "http://example.com/model.safetensors",
        "s3://bucket/path/model.safetensors",
        "file:///path/to/file",
    ])
    def test_valid_uris(self, uri):
        """Test that valid URIs are detected."""
        assert is_uri(uri) is True

    @pytest.mark.parametrize("path", [
        "model.safetensors",
        "/path/to/model.safetensors",
        "relative/path/model.safetensors",
        "://missing-scheme",
        "",
    ])
    def test_non_uris(self, path):
        """Test that non-URIs are not detected."""
        assert is_uri(path) is False


class TestParseHfUri:
    """Tests for the parse_hf_uri function."""

    def test_parses_simple_repo(self):
        """Test parsing a simple repo/file URI."""
        uri = "hf://gpt2/config.json"
        hf_file = parse_hf_uri(uri)

        assert isinstance(hf_file, HuggingFile)
        assert hf_file.repo_id == "gpt2"
        assert hf_file.filename == "config.json"
        assert hf_file.repo_type == "model"

    def test_parses_org_repo(self):
        """Test parsing an org/repo/file URI."""
        uri = "hf://Comfy-Org/flux1-schnell/flux1-schnell-fp8.safetensors"
        hf_file = parse_hf_uri(uri)

        assert isinstance(hf_file, HuggingFile)
        assert hf_file.repo_id == "Comfy-Org/flux1-schnell"
        assert hf_file.filename == "flux1-schnell-fp8.safetensors"
        assert hf_file.repo_type == "model"

    def test_parses_nested_path(self):
        """Test parsing a URI with nested path."""
        uri = "hf://Comfy-Org/flux1-schnell/split_files/diffusion_models/flux1-schnell-fp8.safetensors"
        hf_file = parse_hf_uri(uri)

        assert hf_file.repo_id == "Comfy-Org/flux1-schnell"
        assert hf_file.filename == "split_files/diffusion_models/flux1-schnell-fp8.safetensors"

    def test_parses_dataset_uri(self):
        """Test parsing a datasets URI."""
        uri = "hf://datasets/squad/data/train.json"
        hf_file = parse_hf_uri(uri)

        assert hf_file.repo_id == "squad/data"
        assert hf_file.filename == "train.json"
        assert hf_file.repo_type == "datasets"


class TestGetOrDownloadWithHfUri:
    """Tests for get_or_download handling of hf:// URIs."""

    def test_hf_uri_returns_local_path(self):
        """Test that hf:// URIs return a local cached path string."""
        uri = "hf://gpt2/config.json"
        result = get_or_download("checkpoints", uri)

        assert isinstance(result, str)
        assert os.path.isfile(result)

    def test_hf_uri_file_is_usable(self):
        """Test that the returned path can be used with standard file operations."""
        uri = "hf://gpt2/config.json"
        result = get_or_download("checkpoints", uri)

        with open(result, "r") as f:
            content = f.read()

        assert len(content) > 0
        data = json.loads(content)
        assert "model_type" in data

    @patch("comfy.model_downloader.folder_paths.get_full_path")
    def test_local_filename_uses_normal_lookup(self, mock_get_full_path):
        """Test that local filenames use normal path lookup."""
        import comfy.model_downloader as md
        mock_get_full_path.return_value = "/path/to/model.safetensors"

        saved = getattr(md, '_original_get_full_path', None)
        try:
            if hasattr(md, '_original_get_full_path'):
                delattr(md, '_original_get_full_path')
            result = get_or_download("checkpoints", "model.safetensors")
        finally:
            if saved is not None:
                md._original_get_full_path = saved

        assert result == "/path/to/model.safetensors"
        mock_get_full_path.assert_called_once()

    def test_existing_ideogram_plain_filename_is_found_before_download(self, tmp_path):
        models_dir = tmp_path / "diffusion_models"
        models_dir.mkdir()
        expected = models_dir / "ideogram4_fp8_scaled.safetensors"
        expected.write_bytes(b"already here")

        previous = folder_paths.folder_names_and_paths.get("diffusion_models")
        folder_paths.folder_names_and_paths["diffusion_models"] = FolderPathsTuple(
            "diffusion_models",
            [str(models_dir)],
            {".safetensors"},
        )
        known_file = HuggingFile(
            "Comfy-Org/Ideogram-4",
            "diffusion_models/ideogram4_fp8_scaled.safetensors",
            save_with_filename="ideogram4_fp8_scaled.safetensors",
            alternate_filenames=("ideogram4_fp8_scaled.safetensors",),
        )
        try:
            with patch("comfy.model_downloader.hf_hub_download") as hf_hub_download:
                result = get_or_download(
                    "diffusion_models",
                    "ideogram4_fp8_scaled.safetensors",
                    known_files=[known_file],
                )
        finally:
            if previous is None:
                del folder_paths.folder_names_and_paths["diffusion_models"]
            else:
                folder_paths.folder_names_and_paths["diffusion_models"] = previous

        assert result == str(expected)
        hf_hub_download.assert_not_called()

    def test_offline_uses_hf_cache_without_linking(self, tmp_path):
        models_dir = tmp_path / "checkpoints"
        models_dir.mkdir()
        cached = tmp_path / "hf-cache" / "ideogram4_fp8_scaled.safetensors"
        cached.parent.mkdir()
        cached.write_bytes(b"cached")

        previous = folder_paths.folder_names_and_paths.get("checkpoints")
        folder_paths.folder_names_and_paths["checkpoints"] = FolderPathsTuple(
            "checkpoints",
            [str(models_dir)],
            {".safetensors"},
        )
        known_file = HuggingFile(
            "Comfy-Org/Ideogram-4",
            "diffusion_models/ideogram4_fp8_scaled.safetensors",
            save_with_filename="ideogram4_fp8_scaled.safetensors",
            alternate_filenames=("ideogram4_fp8_scaled.safetensors",),
        )
        try:
            with patch("comfy.model_downloader.hf_hub_download", return_value=str(cached)) as hf_hub_download:
                result = get_or_download(
                    "checkpoints",
                    "ideogram4_fp8_scaled.safetensors",
                    known_files=[known_file],
                    offline=True,
                )
        finally:
            if previous is None:
                del folder_paths.folder_names_and_paths["checkpoints"]
            else:
                folder_paths.folder_names_and_paths["checkpoints"] = previous

        assert result == str(cached)
        assert not (models_dir / "ideogram4_fp8_scaled.safetensors").exists()
        hf_hub_download.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_is_not_hf_uri(self):
        """Test that empty string is not detected as HF URI."""
        assert is_hf_uri("") is False

    def test_hf_prefix_only(self):
        """Test that 'hf://' alone is detected."""
        assert is_hf_uri("hf://") is True
