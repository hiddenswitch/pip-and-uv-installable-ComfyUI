"""Civitai author-prefetch: pull a workflow author's other uploads into
the model index so community checkpoints/loras resolve at lookup time."""
from __future__ import annotations

from unittest.mock import patch, Mock

import pytest

from comfy import civitai_model_cache as c


def setup_function(_):
    c._enabled = True
    c._index = {}
    c._prefetched_users.clear()
    c._live_miss_cache.clear()


def teardown_function(_):
    c._enabled = False
    c._index = {}
    c._prefetched_users.clear()
    c._live_miss_cache.clear()


def _stub_response(payload: dict) -> Mock:
    r = Mock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def test_prefetch_user_indexes_files():
    payload = {
        "items": [
            {
                "id": 999, "type": "Checkpoint",
                "modelVersions": [{
                    "id": 1, "files": [
                        {"name": "myCheckpoint_v10.safetensors",
                         "downloadUrl": "https://civitai.com/api/download/models/1"},
                    ],
                }],
            },
        ],
        "metadata": {},
    }
    with patch.object(c.requests, "get", return_value=_stub_response(payload)) as mock_get, \
         patch.object(c, "_save_disk_cache"), \
         patch.object(c, "_api_token", return_value="dummy"):
        added = c.prefetch_civitai_models_for_user("alice")
    assert added == 1
    assert any("Checkpoint" in k for k in c._index)
    # Calling twice is a no-op (idempotent per username).
    with patch.object(c.requests, "get", return_value=_stub_response(payload)) as mock_get:
        added2 = c.prefetch_civitai_models_for_user("alice")
    assert added2 == 0
    assert mock_get.call_count == 0


def test_prefetch_user_disabled_when_no_token():
    with patch.object(c, "_api_token", return_value=None):
        added = c.prefetch_civitai_models_for_user("anyone")
    assert added == 0


def test_prefetch_for_workflow_uri_resolves_author(monkeypatch):
    """``civitai://m/<id>`` → look up creator.username → prefetch."""
    model_payload = {"creator": {"username": "bob"}}
    user_payload = {
        "items": [{
            "id": 1, "type": "LORA",
            "modelVersions": [{"id": 1, "files": [
                {"name": "bobsLora_v1.safetensors",
                 "downloadUrl": "https://civitai.com/api/download/models/1"}
            ]}],
        }],
        "metadata": {},
    }

    def fake_get(url, **kwargs):
        if "/models/123" in url:
            return _stub_response(model_payload)
        return _stub_response(user_payload)

    with patch.object(c.requests, "get", side_effect=fake_get), \
         patch.object(c, "_save_disk_cache"), \
         patch.object(c, "_api_token", return_value="dummy"):
        added = c.prefetch_civitai_models_for_workflow_uri("civitai://m/123")
    assert added == 1
    # The basename should be findable via get_model_entry.
    e = c.get_model_entry("loras", "bobsLora_v1.safetensors")
    assert e is not None and e[0] == "loras"


def test_prefetch_for_non_civitai_uri_is_noop():
    added = c.prefetch_civitai_models_for_workflow_uri("https://example.com/workflow.json")
    assert added == 0


def test_basename_fallback_in_get_model_entry():
    from comfy.component_model.files import canonicalize_path
    bn = "myCheckpoint_v10.safetensors"
    c._index[canonicalize_path(bn)] = {
        "folder": "checkpoints",
        "url": "https://civitai.com/api/download/models/1",
        "name": bn,
    }
    # Workflow value with a Windows-style subdir prefix should hit basename.
    e = c.get_model_entry("checkpoints", f"SD1.5\\{bn}")
    assert e is not None and e[0] == "checkpoints"
