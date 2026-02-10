import importlib.resources
import json

import pytest

from comfy.component_model.prompt_utils import (
    find_positive_text_encoder,
    replace_prompt_text,
    find_steps_nodes,
    replace_steps,
    find_image_load_nodes,
    replace_images,
    _TEXT_ENCODE_FIELDS,
    _STEPS_CLASS_TYPES,
    _IMAGE_LOAD_CLASS_TYPES,
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


# ---------------------------------------------------------------------------
# --prompt: find positive text encoder
# ---------------------------------------------------------------------------

class TestFindPositiveTextEncoder:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_finds_positive_in_all_workflows(self, workflow_name, workflow_file):
        """Every bundled workflow should have a detectable positive text encoding node."""
        prompt = _load_workflow(workflow_file)
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


# ---------------------------------------------------------------------------
# --prompt: replace_prompt_text
# ---------------------------------------------------------------------------

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

    def test_oneshot_instruct_tokenize(self):
        """OneShotInstructTokenize uses 'prompt' field."""
        prompt = {
            "1": {
                "inputs": {"prompt": "instruct me", "model": ["2", 0]},
                "class_type": "OneShotInstructTokenize",
            },
        }
        result = replace_prompt_text(prompt, "new instruction")
        assert result["1"]["inputs"]["prompt"] == "new instruction"

    def test_transformers_translation_tokenize(self):
        """TransformersTranslationTokenize uses 'prompt' field."""
        prompt = {
            "1": {
                "inputs": {"prompt": "translate this", "model": ["2", 0]},
                "class_type": "TransformersTranslationTokenize",
            },
        }
        result = replace_prompt_text(prompt, "translate that")
        assert result["1"]["inputs"]["prompt"] == "translate that"

    def test_transformers_tokenize(self):
        """TransformersTokenize uses 'prompt' field."""
        prompt = {
            "1": {
                "inputs": {"prompt": "tokenize this", "model": ["2", 0]},
                "class_type": "TransformersTokenize",
            },
        }
        result = replace_prompt_text(prompt, "tokenize that")
        assert result["1"]["inputs"]["prompt"] == "tokenize that"

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


# ---------------------------------------------------------------------------
# --steps: find and replace steps
# ---------------------------------------------------------------------------

class TestReplaceSteps:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_steps_in_all_workflows(self, workflow_name, workflow_file):
        """replace_steps should work on every workflow that has step-bearing nodes."""
        prompt = _load_workflow(workflow_file)
        has_steps = any(
            node.get("class_type", "") in _STEPS_CLASS_TYPES
            and "steps" in node.get("inputs", {})
            for node in prompt.values()
        )
        if not has_steps:
            pytest.skip(f"{workflow_name} has no step-bearing nodes")

        result = replace_steps(prompt, 42)
        for nid in find_steps_nodes(prompt):
            assert result[nid]["inputs"]["steps"] == 42

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_steps_does_not_mutate_original(self, workflow_name, workflow_file):
        """replace_steps must not modify the original prompt dict."""
        prompt = _load_workflow(workflow_file)
        has_steps = any(
            node.get("class_type", "") in _STEPS_CLASS_TYPES
            and "steps" in node.get("inputs", {})
            for node in prompt.values()
        )
        if not has_steps:
            pytest.skip(f"{workflow_name} has no step-bearing nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_steps(prompt, 99)
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceStepsSpecific:
    def test_ksampler_steps(self):
        """KSampler steps should be replaced."""
        prompt = {
            "1": {
                "inputs": {"seed": 1, "steps": 20, "cfg": 7.0},
                "class_type": "KSampler",
            },
        }
        result = replace_steps(prompt, 50)
        assert result["1"]["inputs"]["steps"] == 50

    def test_multiple_step_nodes(self):
        """All step-bearing nodes should be updated."""
        prompt = {
            "1": {
                "inputs": {"seed": 1, "steps": 20, "cfg": 7.0},
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"steps": 30, "scheduler": "normal", "model": ["3", 0]},
                "class_type": "BasicScheduler",
            },
        }
        result = replace_steps(prompt, 10)
        assert result["1"]["inputs"]["steps"] == 10
        assert result["2"]["inputs"]["steps"] == 10

    def test_no_step_nodes_returns_unchanged(self):
        """When no step-bearing nodes exist, return the original prompt."""
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_steps(prompt, 50)
        assert result is prompt  # same object, no copy needed

    def test_flux2_scheduler(self):
        """Flux2Scheduler steps should be replaced."""
        prompt = {
            "1": {
                "inputs": {"steps": 25, "shift": 1.0, "model": ["2", 0]},
                "class_type": "Flux2Scheduler",
            },
        }
        result = replace_steps(prompt, 15)
        assert result["1"]["inputs"]["steps"] == 15

    def test_ksampler_advanced_steps(self):
        """KSamplerAdvanced steps should be replaced."""
        prompt = {
            "1": {
                "inputs": {"steps": 30, "cfg": 8.0, "start_at_step": 0, "end_at_step": 30},
                "class_type": "KSamplerAdvanced",
            },
        }
        result = replace_steps(prompt, 40)
        assert result["1"]["inputs"]["steps"] == 40


# ---------------------------------------------------------------------------
# --image: find and replace images
# ---------------------------------------------------------------------------

class TestReplaceImages:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_images_in_all_workflows(self, workflow_name, workflow_file):
        """replace_images should work on every workflow that has image load nodes."""
        prompt = _load_workflow(workflow_file)
        has_images = any(
            node.get("class_type", "") in _IMAGE_LOAD_CLASS_TYPES
            for node in prompt.values()
        )
        if not has_images:
            pytest.skip(f"{workflow_name} has no image load nodes")

        test_uri = "https://example.com/test.png"
        node_ids = find_image_load_nodes(prompt)
        result = replace_images(prompt, [test_uri] * len(node_ids))

        for nid in node_ids:
            node = result[nid]
            assert node["class_type"] in ("LoadImageFromURL", "ImageRequestParameter")
            assert node["inputs"]["value"] == test_uri

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_images_does_not_mutate_original(self, workflow_name, workflow_file):
        """replace_images must not modify the original prompt dict."""
        prompt = _load_workflow(workflow_file)
        has_images = any(
            node.get("class_type", "") in _IMAGE_LOAD_CLASS_TYPES
            for node in prompt.values()
        )
        if not has_images:
            pytest.skip(f"{workflow_name} has no image load nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_images(prompt, ["https://example.com/test.png"])
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceImagesSpecific:
    def test_load_image_converted_to_url(self):
        """LoadImage should be converted to LoadImageFromURL."""
        prompt = {
            "1": {
                "inputs": {"image": "example.png", "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"},
            },
        }
        result = replace_images(prompt, ["https://example.com/photo.png"])
        assert result["1"]["class_type"] == "LoadImageFromURL"
        assert result["1"]["inputs"] == {"value": "https://example.com/photo.png"}
        assert "_meta" not in result["1"]

    def test_load_image_from_url_updated(self):
        """LoadImageFromURL should just get its value updated."""
        prompt = {
            "1": {
                "inputs": {"value": "https://old.com/img.png"},
                "class_type": "LoadImageFromURL",
            },
        }
        result = replace_images(prompt, ["https://new.com/img.png"])
        assert result["1"]["class_type"] == "LoadImageFromURL"
        assert result["1"]["inputs"]["value"] == "https://new.com/img.png"

    def test_image_request_parameter_updated(self):
        """ImageRequestParameter should get its value updated."""
        prompt = {
            "1": {
                "inputs": {"value": "old_path.png"},
                "class_type": "ImageRequestParameter",
            },
        }
        result = replace_images(prompt, ["s3://bucket/new.png"])
        assert result["1"]["inputs"]["value"] == "s3://bucket/new.png"

    def test_multiple_images_assigned_in_order(self):
        """Multiple images are assigned to nodes in order."""
        prompt = {
            "1": {
                "inputs": {"image": "a.png", "upload": "image"},
                "class_type": "LoadImage",
            },
            "2": {
                "inputs": {"image": "b.png", "upload": "image"},
                "class_type": "LoadImage",
            },
        }
        result = replace_images(prompt, ["https://x.com/1.png", "https://x.com/2.png"])
        assert result["1"]["inputs"]["value"] == "https://x.com/1.png"
        assert result["2"]["inputs"]["value"] == "https://x.com/2.png"

    def test_fewer_images_than_nodes(self):
        """Extra image nodes should be left unchanged when fewer images are provided."""
        prompt = {
            "1": {
                "inputs": {"image": "a.png", "upload": "image"},
                "class_type": "LoadImage",
            },
            "2": {
                "inputs": {"image": "b.png", "upload": "image"},
                "class_type": "LoadImage",
            },
        }
        result = replace_images(prompt, ["https://x.com/only.png"])
        assert result["1"]["class_type"] == "LoadImageFromURL"
        assert result["1"]["inputs"]["value"] == "https://x.com/only.png"
        # second node stays as LoadImage
        assert result["2"]["class_type"] == "LoadImage"
        assert result["2"]["inputs"]["image"] == "b.png"

    def test_no_image_nodes_returns_unchanged(self):
        """When no image load nodes exist, return the original prompt."""
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_images(prompt, ["https://x.com/img.png"])
        assert result is prompt

    def test_empty_images_list_returns_unchanged(self):
        """Empty images list should return the original prompt."""
        prompt = {
            "1": {
                "inputs": {"image": "a.png", "upload": "image"},
                "class_type": "LoadImage",
            },
        }
        result = replace_images(prompt, [])
        assert result is prompt
