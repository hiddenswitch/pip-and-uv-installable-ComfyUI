"""Focused coverage for partner-node behavior changed by upstream merges."""

import pytest

from comfy_api_nodes import nodes_anthropic, nodes_bytedance, nodes_openrouter, nodes_runway
from comfy_api_nodes.apis.anthropic import AnthropicMessagesResponse
from comfy_api_nodes.apis.openrouter import OpenRouterChatResponse, OpenRouterUsage


def _schema_input(node, input_id):
    return next(schema_input for schema_input in node.define_schema().inputs if schema_input.id == input_id)


def test_new_claude_models_expose_expected_thinking_controls():
    assert nodes_anthropic.CLAUDE_MODELS["Opus 4.8"] == "claude-opus-4-8"
    assert nodes_anthropic.CLAUDE_MODELS["Fable 5"] == "claude-fable-5"
    assert nodes_anthropic.CLAUDE_MODELS["Sonnet 5"] == "claude-sonnet-5"

    sonnet_inputs = {
        schema_input.id: schema_input
        for schema_input in nodes_anthropic._claude_model_inputs("Sonnet 5")
    }
    assert "temperature" not in sonnet_inputs
    assert sonnet_inputs["reasoning_effort"].options == ["off", "low", "medium", "high"]

    fable_inputs = {
        schema_input.id: schema_input
        for schema_input in nodes_anthropic._claude_model_inputs("Fable 5")
    }
    assert "temperature" not in fable_inputs
    assert fable_inputs["reasoning_effort"].options == ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_claude_refusal_is_reported_as_an_actionable_error(monkeypatch):
    async def refuse(*_args, **_kwargs):
        return AnthropicMessagesResponse(stop_reason="refusal")

    monkeypatch.setattr(nodes_anthropic, "sync_op", refuse)

    with pytest.raises(ValueError, match="declined to answer"):
        await nodes_anthropic.ClaudeNode.execute(
            prompt="test prompt",
            model={"model": "Sonnet 5", "reasoning_effort": "off"},
            seed=0,
        )


def test_seed_audio_schema_defaults_to_multilingual_model():
    model_input = _schema_input(nodes_bytedance.ByteDanceSeedAudioNode, "model")

    assert model_input.optional is True
    assert model_input.default == "seed-audio-1.0-multilingual"
    assert model_input.options == ["seed-audio-1.0-multilingual", "seed-audio-1.0"]


@pytest.mark.asyncio
async def test_seed_audio_selected_model_is_sent_in_request(monkeypatch):
    captured = {}

    async def capture_request(*_args, **kwargs):
        captured["request"] = kwargs["data"]
        return nodes_bytedance.SeedAudioResponse(code=0, message="test", audio=None)

    monkeypatch.setattr(nodes_bytedance, "sync_op", capture_request)

    with pytest.raises(Exception, match="returned no audio"):
        await nodes_bytedance.ByteDanceSeedAudioNode.execute(
            text_prompt="test prompt",
            reference_mode={"reference_mode": nodes_bytedance.MODE_TEXT},
            sample_rate="24000",
            speech_rate=0,
            loudness_rate=0,
            pitch_rate=0,
            seed=42,
            model="seed-audio-1.0-multilingual",
        )

    assert captured["request"].model == "seed-audio-1.0-multilingual"


def test_openrouter_new_models_and_billing_multiplier():
    slugs = {spec.slug for spec in nodes_openrouter.MODELS}

    assert "openai/gpt-5.6-sol-pro" in slugs
    assert "google/gemini-3.5-flash" in slugs
    assert "deepseek/deepseek-v4-pro" in slugs
    response = OpenRouterChatResponse(usage=OpenRouterUsage(cost=2.0))
    assert nodes_openrouter._calculate_price(response) == pytest.approx(2.86)


@pytest.mark.parametrize(
    "node",
    [
        nodes_runway.RunwayImageToVideoNodeGen3a,
        nodes_runway.RunwayFirstLastFrameNode,
    ],
)
def test_runway_gen3a_nodes_are_deprecated(node):
    assert node.define_schema().is_deprecated is True
