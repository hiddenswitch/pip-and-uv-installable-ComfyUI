"""
Test DownloadableFileList integration with manager model cache.
"""
import pytest
from unittest.mock import patch, MagicMock

from comfy.model_downloader_types import DownloadableFileList, HuggingFile


class TestDownloadableFileListWithManager:
    def test_view_for_validation_includes_manager_models(self):
        """Test that view_for_validation includes manager models when available."""
        existing = ["local_model.safetensors"]
        downloadable = [HuggingFile("org/repo", "known_model.safetensors")]

        dfl = DownloadableFileList(existing, downloadable, folder_name="checkpoints")

        mock_filenames = frozenset([
            "manager_model_1.safetensors",
            "manager_model_2.safetensors",
        ])

        with patch('comfy.manager_model_cache.get_filenames_for_folder', return_value=mock_filenames):
            validation_view = dfl.view_for_validation()

        # Should include local, known, and manager models
        assert "local_model.safetensors" in validation_view
        assert "known_model.safetensors" in validation_view
        assert "manager_model_1.safetensors" in validation_view
        assert "manager_model_2.safetensors" in validation_view

    def test_ui_list_excludes_manager_models(self):
        """Test that the list itself (for UI) excludes manager models."""
        existing = ["local_model.safetensors"]
        downloadable = [HuggingFile("org/repo", "known_model.safetensors")]

        dfl = DownloadableFileList(existing, downloadable, folder_name="checkpoints")

        # The list itself should only have existing + downloadable
        assert "local_model.safetensors" in dfl
        assert "known_model.safetensors" in dfl
        assert len(dfl) == 2

    def test_manager_cache_unavailable_graceful_fallback(self):
        """Test graceful fallback when manager cache is unavailable."""
        existing = ["local_model.safetensors"]
        dfl = DownloadableFileList(existing, [], folder_name="checkpoints")

        with patch('comfy.manager_model_cache.get_filenames_for_folder', return_value=frozenset()):
            validation_view = dfl.view_for_validation()

        # Should still work with just local models
        assert "local_model.safetensors" in validation_view
        assert len(validation_view) == 1

    def test_manager_models_only_loaded_once(self):
        """Test that manager models are only loaded on first validation call."""
        existing = ["local_model.safetensors"]
        dfl = DownloadableFileList(existing, [], folder_name="checkpoints")

        mock_get_filenames = MagicMock(return_value=frozenset(["manager_model.safetensors"]))

        with patch('comfy.manager_model_cache.get_filenames_for_folder', mock_get_filenames):
            # Call twice
            dfl.view_for_validation()
            dfl.view_for_validation()

        # Should only be called once (lazy load)
        mock_get_filenames.assert_called_once()

    def test_no_folder_name_skips_manager_loading(self):
        """Test that when folder_name is None, manager models are not loaded."""
        existing = ["local_model.safetensors"]
        dfl = DownloadableFileList(existing, [])  # No folder_name

        mock_get_filenames = MagicMock()

        with patch('comfy.manager_model_cache.get_filenames_for_folder', mock_get_filenames):
            validation_view = dfl.view_for_validation()

        # Should not attempt to load manager models
        mock_get_filenames.assert_not_called()
        assert "local_model.safetensors" in validation_view

    def test_downloadable_files_with_show_in_ui_false(self):
        """Test that downloadable files with show_in_ui=False are still in validation view but not in list."""
        existing = ["local_model.safetensors"]
        downloadable = [HuggingFile("org/repo", "hidden_model.safetensors", show_in_ui=False)]

        dfl = DownloadableFileList(existing, downloadable, folder_name="checkpoints")

        # Hidden model should not be in UI list
        assert "hidden_model.safetensors" not in dfl
        assert len(dfl) == 1

        # But should be in validation view
        with patch('comfy.manager_model_cache.get_filenames_for_folder', return_value=frozenset()):
            validation_view = dfl.view_for_validation()

        assert "hidden_model.safetensors" in validation_view
