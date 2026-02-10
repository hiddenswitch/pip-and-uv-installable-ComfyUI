"""Tests for HuggingFace token handling in model_downloader."""

import pytest

from comfy.model_downloader import _get_hf_token


class TestGetHfToken:
    def test_returns_none_when_no_token(self, monkeypatch):
        """When no HF token is configured, _get_hf_token should return None."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        # Clear any cached token file by patching the token lookup
        monkeypatch.setattr(
            "huggingface_hub.utils._headers.get_token_to_send",
            lambda token: None,
        )
        result = _get_hf_token()
        assert result is None

    def test_returns_token_when_set(self, monkeypatch):
        """When HF_TOKEN is set, _get_hf_token should return it."""
        monkeypatch.setattr(
            "huggingface_hub.utils._headers.get_token_to_send",
            lambda token: "hf_test_fake_token_123",
        )
        result = _get_hf_token()
        assert result == "hf_test_fake_token_123"

    def test_does_not_raise_on_missing_token(self, monkeypatch):
        """_get_hf_token must never raise LocalTokenNotFoundError."""
        from huggingface_hub.utils import LocalTokenNotFoundError

        def raise_token_error(token):
            raise LocalTokenNotFoundError("no token found")

        monkeypatch.setattr(
            "huggingface_hub.utils._headers.get_token_to_send",
            raise_token_error,
        )
        result = _get_hf_token()
        assert result is None


class TestDownloadWithoutToken:
    def test_hf_hub_download_kwargs_use_none_token(self, monkeypatch):
        """The download kwargs should use token=None when no token is configured."""
        monkeypatch.setattr(
            "comfy.model_downloader._get_hf_token",
            lambda: None,
        )

        # Verify that get_or_download for a public file works without a token
        # by downloading a known small file (gpt2 config)
        from comfy.model_downloader import get_or_download
        import os
        result = get_or_download("checkpoints", "hf://gpt2/config.json")
        assert result is not None
        assert os.path.isfile(result)
