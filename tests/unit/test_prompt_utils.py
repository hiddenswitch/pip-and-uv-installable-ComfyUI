import importlib.resources
import json

import pytest

from comfy.component_model.prompt_utils import (
    find_positive_text_encoder,
    find_negative_text_encoder,
    replace_prompt_text,
    replace_negative_prompt_text,
    find_steps_nodes,
    replace_steps,
    find_seed_nodes,
    replace_seed,
    find_image_load_nodes,
    replace_images,
    find_video_load_nodes,
    replace_videos,
    find_audio_load_nodes,
    replace_audios,
    _is_node_ref,
    _TEXT_ENCODE_FIELDS,
    _SAMPLER_CLASS_TYPES,
    _STEPS_CLASS_TYPES,
    _SEED_FIELDS,
    _IMAGE_LOAD_CLASS_TYPES,
    _VIDEO_LOAD_CLASS_TYPES,
    _AUDIO_LOAD_CLASS_TYPES,
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


class TestReplacePromptText:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_in_all_workflows(self, workflow_name, workflow_file):
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
        prompt = {
            "1": {
                "inputs": {"text": "sole prompt", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_prompt_text(prompt, "replaced")
        assert result["1"]["inputs"]["text"] == "replaced"

    def test_sd3_multi_prompt(self):
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
        prompt = {
            "1": {
                "inputs": {"prompt": "instruct me", "model": ["2", 0]},
                "class_type": "OneShotInstructTokenize",
            },
        }
        result = replace_prompt_text(prompt, "new instruction")
        assert result["1"]["inputs"]["prompt"] == "new instruction"

    def test_transformers_translation_tokenize(self):
        prompt = {
            "1": {
                "inputs": {"prompt": "translate this", "model": ["2", 0]},
                "class_type": "TransformersTranslationTokenize",
            },
        }
        result = replace_prompt_text(prompt, "translate that")
        assert result["1"]["inputs"]["prompt"] == "translate that"

    def test_transformers_tokenize(self):
        prompt = {
            "1": {
                "inputs": {"prompt": "tokenize this", "model": ["2", 0]},
                "class_type": "TransformersTokenize",
            },
        }
        result = replace_prompt_text(prompt, "tokenize that")
        assert result["1"]["inputs"]["prompt"] == "tokenize that"

    def test_raises_on_no_text_encoder(self):
        prompt = {
            "1": {
                "inputs": {"width": 512, "height": 512},
                "class_type": "EmptyLatentImage",
            },
        }
        with pytest.raises(ValueError, match="Could not find"):
            replace_prompt_text(prompt, "test")


class TestReplaceSteps:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_steps_in_all_workflows(self, workflow_name, workflow_file):
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
        prompt = {
            "1": {
                "inputs": {"seed": 1, "steps": 20, "cfg": 7.0},
                "class_type": "KSampler",
            },
        }
        result = replace_steps(prompt, 50)
        assert result["1"]["inputs"]["steps"] == 50

    def test_multiple_step_nodes(self):
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
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_steps(prompt, 50)
        assert result is prompt  # same object, no copy needed

    def test_flux2_scheduler(self):
        prompt = {
            "1": {
                "inputs": {"steps": 25, "shift": 1.0, "model": ["2", 0]},
                "class_type": "Flux2Scheduler",
            },
        }
        result = replace_steps(prompt, 15)
        assert result["1"]["inputs"]["steps"] == 15

    def test_ksampler_advanced_steps(self):
        prompt = {
            "1": {
                "inputs": {"steps": 30, "cfg": 8.0, "start_at_step": 0, "end_at_step": 30},
                "class_type": "KSamplerAdvanced",
            },
        }
        result = replace_steps(prompt, 40)
        assert result["1"]["inputs"]["steps"] == 40


class TestReplaceSeed:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_seed_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        pairs = find_seed_nodes(prompt)
        if not pairs:
            pytest.skip(f"{workflow_name} has no seed-bearing nodes")

        result = replace_seed(prompt, 42)
        for nid, field in pairs:
            assert result[nid]["inputs"][field] == 42

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_seed_does_not_mutate_original(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        if not find_seed_nodes(prompt):
            pytest.skip(f"{workflow_name} has no seed-bearing nodes")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_seed(prompt, 99)
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceSeedSpecific:
    def test_ksampler_seed(self):
        prompt = {
            "1": {
                "inputs": {"seed": 12345, "steps": 20, "cfg": 7.0},
                "class_type": "KSampler",
            },
        }
        result = replace_seed(prompt, 99999)
        assert result["1"]["inputs"]["seed"] == 99999

    def test_random_noise_seed(self):
        prompt = {
            "1": {
                "inputs": {"noise_seed": 1038979},
                "class_type": "RandomNoise",
            },
        }
        result = replace_seed(prompt, 42)
        assert result["1"]["inputs"]["noise_seed"] == 42

    def test_sampler_custom_seed(self):
        prompt = {
            "1": {
                "inputs": {"noise_seed": 555, "cfg": 8.0},
                "class_type": "SamplerCustom",
            },
        }
        result = replace_seed(prompt, 111)
        assert result["1"]["inputs"]["noise_seed"] == 111

    def test_transformers_generate_seed(self):
        prompt = {
            "1": {
                "inputs": {"seed": 2013744903, "max_tokens": 512},
                "class_type": "TransformersGenerate",
            },
        }
        result = replace_seed(prompt, 0)
        assert result["1"]["inputs"]["seed"] == 0

    def test_multiple_seed_nodes(self):
        prompt = {
            "1": {
                "inputs": {"seed": 100, "steps": 20},
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"noise_seed": 200},
                "class_type": "RandomNoise",
            },
        }
        result = replace_seed(prompt, 777)
        assert result["1"]["inputs"]["seed"] == 777
        assert result["2"]["inputs"]["noise_seed"] == 777

    def test_no_seed_nodes_returns_unchanged(self):
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_seed(prompt, 42)
        assert result is prompt


class TestReplaceImages:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_images_in_all_workflows(self, workflow_name, workflow_file):
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
        prompt = {
            "1": {
                "inputs": {"value": "old_path.png"},
                "class_type": "ImageRequestParameter",
            },
        }
        result = replace_images(prompt, ["s3://bucket/new.png"])
        assert result["1"]["inputs"]["value"] == "s3://bucket/new.png"

    def test_multiple_images_assigned_in_order(self):
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
        prompt = {
            "1": {
                "inputs": {"text": "hello", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        result = replace_images(prompt, ["https://x.com/img.png"])
        assert result is prompt

    def test_empty_images_list_returns_unchanged(self):
        prompt = {
            "1": {
                "inputs": {"image": "a.png", "upload": "image"},
                "class_type": "LoadImage",
            },
        }
        result = replace_images(prompt, [])
        assert result is prompt


class TestFindNegativeTextEncoder:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_finds_negative_in_workflows_with_sampler(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        has_sampler_with_negative = any(
            node.get("class_type", "") in _SAMPLER_CLASS_TYPES
            and _is_node_ref(node.get("inputs", {}).get("negative"))
            for node in prompt.values()
        )
        if not has_sampler_with_negative:
            pytest.skip(f"{workflow_name} has no sampler with negative conditioning input")

        node_id = find_negative_text_encoder(prompt)
        assert node_id is not None, f"Could not find negative text encoder in {workflow_name}"
        assert node_id in prompt
        class_type = prompt[node_id]["class_type"]
        assert class_type in _TEXT_ENCODE_FIELDS


class TestReplaceNegativePromptText:
    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_negative_in_all_workflows(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        if find_negative_text_encoder(prompt) is None:
            pytest.skip(f"{workflow_name} has no negative text encoder")

        replacement = "a test negative prompt"
        result = replace_negative_prompt_text(prompt, replacement)

        node_id = find_negative_text_encoder(prompt)
        class_type = result[node_id]["class_type"]
        fields = _TEXT_ENCODE_FIELDS[class_type]

        for field in fields:
            if field in result[node_id]["inputs"]:
                assert result[node_id]["inputs"][field] == replacement

    @pytest.mark.parametrize("workflow_name, workflow_file", _all_workflow_files().items())
    def test_replace_negative_does_not_mutate_original(self, workflow_name, workflow_file):
        prompt = _load_workflow(workflow_file)
        if find_negative_text_encoder(prompt) is None:
            pytest.skip(f"{workflow_name} has no negative text encoder")

        original_json = json.dumps(prompt, sort_keys=True)
        replace_negative_prompt_text(prompt, "mutated?")
        assert json.dumps(prompt, sort_keys=True) == original_json


class TestReplaceNegativePromptTextSpecific:
    def test_ksampler_positive_negative(self):
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
        result = replace_negative_prompt_text(prompt, "new negative")
        assert result["3"]["inputs"]["text"] == "new negative"
        assert result["2"]["inputs"]["text"] == "original positive"

    def test_negative_via_title(self):
        prompt = {
            "1": {
                "inputs": {"text": "positive prompt", "clip": ["3", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive"},
            },
            "2": {
                "inputs": {"text": "negative prompt", "clip": ["3", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative"},
            },
        }
        result = replace_negative_prompt_text(prompt, "replaced negative")
        assert result["2"]["inputs"]["text"] == "replaced negative"
        assert result["1"]["inputs"]["text"] == "positive prompt"

    def test_qwen_image_edit_negative(self):
        prompt = {
            "3": {
                "inputs": {
                    "seed": 1,
                    "positive": ["76", 0],
                    "negative": ["77", 0],
                },
                "class_type": "KSampler",
            },
            "76": {
                "inputs": {
                    "prompt": "edit instruction",
                    "clip": ["38", 0],
                    "vae": ["39", 0],
                    "image": ["93", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
            "77": {
                "inputs": {
                    "prompt": "",
                    "clip": ["38", 0],
                    "vae": ["39", 0],
                    "image": ["93", 0],
                },
                "class_type": "TextEncodeQwenImageEdit",
            },
        }
        result = replace_negative_prompt_text(prompt, "ugly, bad quality")
        assert result["77"]["inputs"]["prompt"] == "ugly, bad quality"
        assert result["76"]["inputs"]["prompt"] == "edit instruction"

    def test_raises_on_no_negative_encoder(self):
        prompt = {
            "1": {
                "inputs": {"text": "sole prompt", "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        with pytest.raises(ValueError, match="Could not find"):
            replace_negative_prompt_text(prompt, "test")

    def test_negative_through_passthrough(self):
        prompt = {
            "1": {
                "inputs": {
                    "seed": 1,
                    "positive": ["2", 0],
                    "negative": ["4", 0],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"text": "positive", "clip": ["5", 0]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": "negative", "clip": ["5", 0]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"conditioning": ["3", 0]},
                "class_type": "ConditioningZeroOut",
            },
        }
        result = replace_negative_prompt_text(prompt, "replaced negative")
        assert result["3"]["inputs"]["text"] == "replaced negative"
        assert result["2"]["inputs"]["text"] == "positive"


class TestReplaceVideosSpecific:
    def test_load_video_converted_to_url(self):
        prompt = {
            "1": {
                "inputs": {"file": "clip.mp4"},
                "class_type": "LoadVideo",
                "_meta": {"title": "Load Video"},
            }
        }
        result = replace_videos(prompt, ["https://example.com/clip.mp4"])
        assert result["1"]["class_type"] == "LoadVideoFromURL"
        assert result["1"]["inputs"]["value"] == "https://example.com/clip.mp4"
        assert "_meta" not in result["1"]

    def test_load_video_from_url_value_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.mp4"},
                "class_type": "LoadVideoFromURL",
            }
        }
        result = replace_videos(prompt, ["https://example.com/new.mp4"])
        assert result["1"]["class_type"] == "LoadVideoFromURL"
        assert result["1"]["inputs"]["value"] == "https://example.com/new.mp4"

    def test_video_request_parameter_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.mp4", "frame_load_cap": 100},
                "class_type": "VideoRequestParameter",
            }
        }
        result = replace_videos(prompt, ["https://example.com/new.mp4"])
        assert result["1"]["inputs"]["value"] == "https://example.com/new.mp4"

    def test_multiple_video_nodes(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
            "2": {"inputs": {"value": "b.mp4"}, "class_type": "LoadVideoFromURL"},
        }
        result = replace_videos(prompt, ["url1.mp4", "url2.mp4"])
        assert result["1"]["inputs"]["value"] == "url1.mp4"
        assert result["2"]["inputs"]["value"] == "url2.mp4"

    def test_more_videos_than_nodes(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
        }
        result = replace_videos(prompt, ["url1.mp4", "url2.mp4", "url3.mp4"])
        assert result["1"]["inputs"]["value"] == "url1.mp4"

    def test_no_video_nodes_unchanged(self):
        prompt = {
            "1": {"inputs": {"text": "hello"}, "class_type": "CLIPTextEncode"},
        }
        result = replace_videos(prompt, ["url1.mp4"])
        assert result is prompt  # no copy when nothing to replace

    def test_empty_videos_unchanged(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
        }
        result = replace_videos(prompt, [])
        assert result is prompt

    def test_does_not_mutate_original(self):
        prompt = {
            "1": {"inputs": {"file": "a.mp4"}, "class_type": "LoadVideo"},
        }
        result = replace_videos(prompt, ["url1.mp4"])
        assert prompt["1"]["class_type"] == "LoadVideo"
        assert result["1"]["class_type"] == "LoadVideoFromURL"


class TestReplaceAudiosSpecific:
    def test_load_audio_converted_to_url(self):
        prompt = {
            "1": {
                "inputs": {"audio": "track.mp3"},
                "class_type": "LoadAudio",
                "_meta": {"title": "Load Audio"},
            }
        }
        result = replace_audios(prompt, ["https://example.com/track.mp3"])
        assert result["1"]["class_type"] == "LoadAudioFromURL"
        assert result["1"]["inputs"]["value"] == "https://example.com/track.mp3"
        assert "_meta" not in result["1"]

    def test_load_audio_from_url_value_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.wav"},
                "class_type": "LoadAudioFromURL",
            }
        }
        result = replace_audios(prompt, ["https://example.com/new.wav"])
        assert result["1"]["inputs"]["value"] == "https://example.com/new.wav"

    def test_audio_request_parameter_updated(self):
        prompt = {
            "1": {
                "inputs": {"value": "old.wav"},
                "class_type": "AudioRequestParameter",
            }
        }
        result = replace_audios(prompt, ["https://example.com/new.wav"])
        assert result["1"]["inputs"]["value"] == "https://example.com/new.wav"

    def test_no_audio_nodes_unchanged(self):
        prompt = {
            "1": {"inputs": {"text": "hello"}, "class_type": "CLIPTextEncode"},
        }
        result = replace_audios(prompt, ["url.wav"])
        assert result is prompt

    def test_does_not_mutate_original(self):
        prompt = {
            "1": {"inputs": {"audio": "a.mp3"}, "class_type": "LoadAudio"},
        }
        result = replace_audios(prompt, ["url.mp3"])
        assert prompt["1"]["class_type"] == "LoadAudio"
        assert result["1"]["class_type"] == "LoadAudioFromURL"


class TestFindMediaNodes:
    def test_find_video_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadVideo"},
            "2": {"inputs": {}, "class_type": "LoadVideoFromURL"},
            "3": {"inputs": {}, "class_type": "CLIPTextEncode"},
        }
        assert set(find_video_load_nodes(prompt)) == {"1", "2"}

    def test_find_audio_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadAudio"},
            "2": {"inputs": {}, "class_type": "AudioRequestParameter"},
            "3": {"inputs": {}, "class_type": "LoadImage"},
        }
        assert set(find_audio_load_nodes(prompt)) == {"1", "2"}

    def test_find_no_video_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadImage"},
        }
        assert find_video_load_nodes(prompt) == []

    def test_find_no_audio_nodes(self):
        prompt = {
            "1": {"inputs": {}, "class_type": "LoadImage"},
        }
        assert find_audio_load_nodes(prompt) == []
