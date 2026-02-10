import importlib.resources
import json

import pytest

from comfy.component_model.prompt_utils import (
    find_positive_text_encoder,
    replace_prompt_text,
    _TEXT_ENCODE_FIELDS,
)
from tests.inference import workflows


def _all_workflow_files():
    return {
        f.name: f
        for f in importlib.resources.files(workflows).iterdir()
        if f.is_file() and f.name.endswith(".json")
    }


def _load_workflow(workflow_file) -> dict:
    return json.loads(workflow_file.read_text(encoding="utf8"))


class TestFindPositiveTextEncoder:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_finds_positive_in_all_workflows(self, workflow_name, workflow_file):
        """Every bundled workflow should have a detectable positive text encoding node."""
        prompt = _load_workflow(workflow_file)
        # some workflows may not have any text encode nodes at all (e.g. pure video/audio loaders)
        has_text_encode = any(
            node.get("class_type", "") in _TEXT_ENCODE_FIELDS
            for node in prompt.values()
        )
        if not has_text_encode:
            pytest.skip(f"{workflow_name} has no text encoding nodes")

        node_id = find_positive_text_encoder(prompt)
        assert node_id is not None, f"Could not find positive text encoder in {workflow_name}"
        assert node_id in prompt
        class_type = prompt[node_id]["class_type"]
        assert class_type in _TEXT_ENCODE_FIELDS


class TestReplacePromptText:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_in_all_workflows(self, workflow_name, workflow_file):
        """replace_prompt_text should work on every workflow that has text encoding nodes."""
        prompt = _load_workflow(workflow_file)
        has_text_encode = any(
            node.get("class_type", "") in _TEXT_ENCODE_FIELDS
            for node in prompt.values()
        )
        if not has_text_encode:
            pytest.skip(f"{workflow_name} has no text encoding nodes")

        replacement = "a test prompt for unit testing"
        result = replace_prompt_text(prompt, replacement)

        # find the node that was replaced
        node_id = find_positive_text_encoder(prompt)
        class_type = result[node_id]["class_type"]
        fields = _TEXT_ENCODE_FIELDS[class_type]

        for field in fields:
            if field in result[node_id]["inputs"]:
                assert result[node_id]["inputs"][field] == replacement

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_does_not_mutate_original(self, workflow_name, workflow_file):
        """replace_prompt_text must not modify the original prompt dict."""
        prompt = _load_workflow(workflow_file)
        has_text_encode = any(
            node.get("class_type", "") in _TEXT_ENCODE_FIELDS
            for node in prompt.values()
        )
        if not has_text_encode:
            pytest.skip(f"{workflow_name} has no text encoding nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_prompt_text(prompt, "mutated?")
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplacePromptTextSpecific:
    def test_simple_two_node_positive_negative(self):
        """KSampler with positive/negative CLIPTextEncode nodes."""
        prompt = {
            "1": {
                "inputs": {
                    "seed": 123,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"text": "original positive", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
            },
            "3": {
                "inputs": {"text": "original negative", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
            },
        }
        result = replace_prompt_text(prompt, "new positive")
        assert result["2"]["inputs"]["text"] == "new positive"
        assert result["3"]["inputs"]["text"] == "original negative"

    def test_single_clip_text_encode(self):
        """When there's only one CLIPTextEncode, it should be replaced."""
        prompt = {
            "1": {
                "inputs": {"text": "sole prompt", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_prompt_text(prompt, "replaced")
        assert result["1"]["inputs"]["text"] == "replaced"

    def test_sd3_multi_prompt(self):
        """CLIPTextEncodeSD3 should replace all three text fields."""
        prompt = {
            "1": {
                "inputs": {
                    "seed": 1,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {
                    "clip_l": "original l",
                    "clip_g": "original g",
                    "t5xxl": "original t5",
                    "empty_padding": "none",
                    "clip": ["4", 0],
                },
                "class_type": "CLIPTextEncodeSD3",
            },
            "3": {
                "inputs": {"text": "negative", "clip": ["4", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_prompt_text(prompt, "new prompt")
        assert result["2"]["inputs"]["clip_l"] == "new prompt"
        assert result["2"]["inputs"]["clip_g"] == "new prompt"
        assert result["2"]["inputs"]["t5xxl"] == "new prompt"
        assert result["3"]["inputs"]["text"] == "negative"

    def test_guider_chain(self):
        """SamplerCustomAdvanced → BasicGuider → FluxGuidance → CLIPTextEncode."""
        prompt = {
            "1": {
                "inputs": {"guider": ["2", 0]},
                "class_type": "SamplerCustomAdvanced",
            },
            "2": {
                "inputs": {"conditioning": ["3", 0], "model": ["5", 0]},
                "class_type": "BasicGuider",
            },
            "3": {
                "inputs": {"guidance": 3, "conditioning": ["4", 0]},
                "class_type": "FluxGuidance",
            },
            "4": {
                "inputs": {"text": "flux prompt", "clip": ["6", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
            },
        }
        result = replace_prompt_text(prompt, "replaced flux")
        assert result["4"]["inputs"]["text"] == "replaced flux"

    def test_qwen_image_edit(self):
        """TextEncodeQwenImageEdit uses 'prompt' field, not 'text'."""
        prompt = {
            "1": {
                "inputs": {
                    "seed": 1,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {
                    "prompt": "edit instruction",
                    "clip": ["4", 0],
                    "vae": ["5", 0],
                    "image": ["6", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
            "3": {
                "inputs": {
                    "prompt": "",
                    "clip": ["4", 0],
                    "vae": ["5", 0],
                    "image": ["6", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
        }
        result = replace_prompt_text(prompt, "new edit")
        assert result["2"]["inputs"]["prompt"] == "new edit"
        assert result["3"]["inputs"]["prompt"] == ""

    def test_raises_on_no_text_encoder(self):
        """Should raise ValueError when no text encoder exists."""
        prompt = {
            "1": {
                "inputs": {"width": 512, "height": 512},
                "class_type": "EmptyLatentImage",
            },
        }
        with pytest.raises(ValueError, match="Could not find"):
            replace_prompt_text(prompt, "test")
