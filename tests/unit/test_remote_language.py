import os
from unittest.mock import patch, Mock, AsyncMock, MagicMock

import pytest
import torch

from comfy.language.language_types import ProcessorResult
from comfy.language.remote_model import RemoteLanguageModel


@pytest.fixture
def model():
    return RemoteLanguageModel("openai:gpt-4o")


def test_remote_language_model_tokenize(model):
    result = model.tokenize("Hello world", None)
    assert result["inputs"] == ["Hello world"]
    assert result["attention_mask"] is not None
    assert result["images"] is None


def test_remote_language_model_tokenize_with_language_prompt(model):
    prompt = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
    ]
    result = model.tokenize(prompt, None)
    assert result["inputs"] is prompt
    assert result["images"] is None


def test_remote_language_model_tokenize_with_images(model):
    images = torch.rand((2, 64, 64, 3))
    result = model.tokenize("Describe this", images)
    assert result["inputs"] == ["Describe this"]
    assert result["images"] is images


def test_remote_language_model_repo_id(model):
    assert model.repo_id == "openai:gpt-4o"


def test_remote_language_model_from_pretrained():
    m = RemoteLanguageModel.from_pretrained("anthropic:claude-sonnet-4-5-20250514")
    assert m.repo_id == "anthropic:claude-sonnet-4-5-20250514"


def _make_mock_agent(chunks):
    """Create a mock Agent that streams the given text chunks."""
    mock_agent_instance = MagicMock()

    async def fake_stream_text(delta=False):
        for chunk in chunks:
            yield chunk

    mock_stream_result = MagicMock()
    mock_stream_result.stream_text = fake_stream_text

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_stream_result)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_agent_instance.run_stream = MagicMock(return_value=mock_ctx)
    return mock_agent_instance


@patch("comfy.language.remote_model.Agent")
def test_remote_language_model_generate(MockAgent):
    mock_agent = _make_mock_agent(["Hello", " ", "world"])
    MockAgent.return_value = mock_agent

    model = RemoteLanguageModel("openai:gpt-4o")
    tokens: ProcessorResult = {"inputs": ["What is AI?"]}
    result = model.generate(tokens, max_new_tokens=100, seed=42)

    assert result == "Hello world"
    MockAgent.assert_called_once()
    assert MockAgent.call_args.kwargs["model"] == "openai:gpt-4o"


@patch("comfy.language.remote_model.Agent")
def test_remote_language_model_generate_with_images(MockAgent):
    mock_agent = _make_mock_agent(["A", " cat"])
    MockAgent.return_value = mock_agent

    model = RemoteLanguageModel("openai:gpt-4o")
    images = torch.rand((1, 32, 32, 3))
    tokens: ProcessorResult = {"inputs": ["Describe this image"], "images": images}
    result = model.generate(tokens, max_new_tokens=50, seed=0)

    assert result == "A cat"
    # Verify user_prompt includes BinaryContent
    call_kwargs = mock_agent.run_stream.call_args.kwargs
    user_parts = call_kwargs["user_prompt"]
    assert len(user_parts) == 2  # text + image


@patch("comfy.language.remote_model.Agent")
def test_remote_language_model_generate_with_sampler(MockAgent):
    mock_agent = _make_mock_agent(["OK"])
    MockAgent.return_value = mock_agent

    model = RemoteLanguageModel("openai:gpt-4o")
    tokens: ProcessorResult = {"inputs": ["Hi"]}
    sampler = {"temperature": 0.5, "top_p": 0.9}
    result = model.generate(tokens, max_new_tokens=10, seed=0, sampler=sampler)

    assert result == "OK"
    call_kwargs = mock_agent.run_stream.call_args.kwargs
    settings = call_kwargs["model_settings"]
    assert settings["temperature"] == 0.5
    assert settings["top_p"] == 0.9
    assert settings["max_tokens"] == 10


@patch("comfy.language.remote_model.Agent")
def test_remote_language_model_generate_no_sampler_omits_temperature(MockAgent):
    """When no sampler is provided, temperature/top_p should not be sent."""
    mock_agent = _make_mock_agent(["OK"])
    MockAgent.return_value = mock_agent

    model = RemoteLanguageModel("openai:gpt-4o")
    tokens: ProcessorResult = {"inputs": ["Hi"]}
    result = model.generate(tokens, max_new_tokens=10, seed=0)

    assert result == "OK"
    call_kwargs = mock_agent.run_stream.call_args.kwargs
    settings = call_kwargs["model_settings"]
    assert "temperature" not in settings
    assert "top_p" not in settings
    assert settings["max_tokens"] == 10


@patch("comfy.language.remote_model.Agent")
def test_remote_language_model_generate_with_language_prompt(MockAgent):
    mock_agent = _make_mock_agent(["Response"])
    MockAgent.return_value = mock_agent

    model = RemoteLanguageModel("anthropic:claude-sonnet-4-5-20250514")
    prompt = [
        {"role": "system", "content": "You are a poet."},
        {"role": "user", "content": [{"type": "text", "text": "Write a haiku"}]},
    ]
    tokens: ProcessorResult = {"inputs": prompt}
    result = model.generate(tokens, max_new_tokens=50, seed=0)

    assert result == "Response"
    # System prompt should be passed as instructions
    assert MockAgent.call_args.kwargs["instructions"] == "You are a poet."
    # User prompt should contain the text
    call_kwargs = mock_agent.run_stream.call_args.kwargs
    assert "Write a haiku" in call_kwargs["user_prompt"]


# --- RemoteLanguageLoader tests ---

@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
def test_remote_language_loader_execute():
    from comfy_extras.nodes.nodes_remote_language import RemoteLanguageLoader
    loader = RemoteLanguageLoader()
    model, = loader.execute("openai:gpt-4o")
    assert isinstance(model, RemoteLanguageModel)
    assert model.repo_id == "openai:gpt-4o"


def test_remote_language_loader_custom_model():
    from comfy_extras.nodes.nodes_remote_language import RemoteLanguageLoader
    loader = RemoteLanguageLoader()
    model, = loader.execute("openai:gpt-4o", custom_model="anthropic:claude-sonnet-4-5-20250514")
    assert isinstance(model, RemoteLanguageModel)
    assert model.repo_id == "anthropic:claude-sonnet-4-5-20250514"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
def test_remote_language_loader_validate_inputs_with_key():
    from comfy_extras.nodes.nodes_remote_language import RemoteLanguageLoader
    result = RemoteLanguageLoader.VALIDATE_INPUTS(model="openai:gpt-4o")
    assert result is True


@patch.dict(os.environ, {}, clear=True)
def test_remote_language_loader_validate_inputs_without_key():
    from comfy_extras.nodes.nodes_remote_language import RemoteLanguageLoader
    with patch("comfy_extras.nodes.nodes_remote_language.args") as mock_args:
        mock_args.openai_api_key = None
        result = RemoteLanguageLoader.VALIDATE_INPUTS(model="openai:gpt-4o")
        assert isinstance(result, str)
        assert "OPENAI_API_KEY" in result


def test_get_available_models():
    from comfy_extras.nodes.nodes_remote_language import get_available_models
    models = get_available_models()
    assert "openai:gpt-4o" in models
    assert "anthropic:claude-sonnet-4-5-20250514" in models
    assert "google-gla:gemini-2.0-flash" in models
    # No suffixes — clean model IDs for stable serialization
    assert all("[" not in m for m in models)
