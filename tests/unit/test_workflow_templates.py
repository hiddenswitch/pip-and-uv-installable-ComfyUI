import json

import pytest

from comfy.cmd.workflow_templates import (
    TemplateInfo,
    _collect_class_types,
    _detect_supported_params,
    _build_example_invocation,
    _populate_supported_params,
)


def _api_workflow(*class_types):
    return {
        str(i): {"class_type": ct, "inputs": {}}
        for i, ct in enumerate(class_types)
    }


def _ui_workflow(*node_types):
    return {
        "nodes": [{"id": i, "type": nt} for i, nt in enumerate(node_types)]
    }


def _ui_subgraph_workflow(*node_types):
    return {
        "nodes": [{"id": 0, "type": "some-uuid-1234"}],
        "definitions": {
            "subgraphs": [
                {
                    "id": "some-uuid-1234",
                    "nodes": [{"id": i, "type": nt} for i, nt in enumerate(node_types)],
                }
            ]
        },
    }


def _extra_prompt_workflow(*class_types):
    return {
        "extra": {
            "prompt": {
                str(i): {"class_type": ct, "inputs": {}}
                for i, ct in enumerate(class_types)
            }
        }
    }


class TestCollectClassTypes:
    def test_api_format(self):
        wf = _api_workflow("KSampler", "CLIPTextEncode")
        assert _collect_class_types(wf) == {"KSampler", "CLIPTextEncode"}

    def test_ui_format(self):
        wf = _ui_workflow("KSampler", "CLIPTextEncode")
        assert _collect_class_types(wf) == {"KSampler", "CLIPTextEncode"}

    def test_subgraph_format(self):
        wf = _ui_subgraph_workflow("KSampler", "CLIPTextEncode", "LoadImage")
        types = _collect_class_types(wf)
        assert "KSampler" in types
        assert "CLIPTextEncode" in types
        assert "LoadImage" in types
        assert "some-uuid-1234" in types

    def test_extra_prompt_format(self):
        wf = _extra_prompt_workflow("BasicScheduler", "RandomNoise")
        types = _collect_class_types(wf)
        assert "BasicScheduler" in types
        assert "RandomNoise" in types

    def test_empty_workflow(self):
        assert _collect_class_types({}) == set()


class TestDetectSupportedParams:
    def test_prompt_only(self):
        wf = _ui_workflow("CLIPTextEncode", "CheckpointLoaderSimple")
        params = _detect_supported_params(wf)
        assert "prompt" in params
        assert "negative-prompt" not in params

    def test_prompt_and_negative(self):
        wf = _ui_workflow("CLIPTextEncode", "KSampler")
        params = _detect_supported_params(wf)
        assert "prompt" in params
        assert "negative-prompt" in params

    def test_negative_with_cfg_guider(self):
        wf = _api_workflow("CLIPTextEncode", "CFGGuider")
        params = _detect_supported_params(wf)
        assert "negative-prompt" in params

    def test_steps(self):
        wf = _ui_workflow("BasicScheduler")
        params = _detect_supported_params(wf)
        assert "steps" in params

    def test_seed(self):
        wf = _api_workflow("RandomNoise")
        params = _detect_supported_params(wf)
        assert "seed" in params

    def test_image(self):
        wf = _ui_workflow("LoadImage")
        params = _detect_supported_params(wf)
        assert "image" in params

    def test_video(self):
        wf = _ui_workflow("LoadVideo")
        params = _detect_supported_params(wf)
        assert "video" in params

    def test_audio(self):
        wf = _ui_workflow("LoadAudio")
        params = _detect_supported_params(wf)
        assert "audio" in params

    def test_full_workflow(self):
        wf = _ui_workflow("CLIPTextEncode", "KSampler", "LoadImage")
        params = _detect_supported_params(wf)
        assert params == ["prompt", "negative-prompt", "steps", "seed", "image"]

    def test_no_params(self):
        wf = _ui_workflow("CheckpointLoaderSimple", "VAEDecode")
        assert _detect_supported_params(wf) == []

    def test_subgraph_detection(self):
        wf = _ui_subgraph_workflow("CLIPTextEncode", "KSampler", "LoadImage")
        params = _detect_supported_params(wf)
        assert "prompt" in params
        assert "seed" in params
        assert "image" in params


class TestBuildExampleInvocation:
    def test_with_template_id(self):
        tmpl = TemplateInfo(
            name="Test", source="package",
            template_id="my_template",
            supported_params=["prompt", "seed"],
        )
        result = _build_example_invocation(tmpl)
        assert result == 'comfyui post-workflow my_template --prompt "your text here" --seed 42'

    def test_fallback_to_name(self):
        tmpl = TemplateInfo(
            name="My Workflow", source="dir:/tmp",
            supported_params=["prompt"],
        )
        result = _build_example_invocation(tmpl)
        assert result == 'comfyui post-workflow My Workflow --prompt "your text here"'

    def test_no_params(self):
        tmpl = TemplateInfo(name="bare", source="package", template_id="bare")
        result = _build_example_invocation(tmpl)
        assert result == "comfyui post-workflow bare"

    def test_all_params(self):
        tmpl = TemplateInfo(
            name="full", source="package", template_id="full",
            supported_params=["prompt", "negative-prompt", "steps", "seed", "image", "video", "audio"],
        )
        result = _build_example_invocation(tmpl)
        assert "--prompt" in result
        assert "--negative-prompt" in result
        assert "--steps 20" in result
        assert "--seed 42" in result
        assert "--image" in result
        assert "--video" in result
        assert "--audio" in result


class TestPopulateSupportedParams:
    def test_populates_from_file(self, tmp_path):
        wf = _ui_workflow("CLIPTextEncode", "KSampler")
        wf_path = tmp_path / "test.json"
        wf_path.write_text(json.dumps(wf))

        tmpl = TemplateInfo(name="test", source="dir", path=str(wf_path))
        _populate_supported_params([tmpl])
        assert "prompt" in tmpl.supported_params
        assert "seed" in tmpl.supported_params

    def test_missing_file(self):
        tmpl = TemplateInfo(name="test", source="dir", path="/nonexistent/path.json")
        _populate_supported_params([tmpl])
        assert tmpl.supported_params == []

    def test_no_path(self):
        tmpl = TemplateInfo(name="test", source="custom_node")
        _populate_supported_params([tmpl])
        assert tmpl.supported_params == []


class TestFiltering:
    def test_api_tag_excluded_by_default(self):
        templates = [
            TemplateInfo(name="local", source="package", tags=[]),
            TemplateInfo(name="api_one", source="package", tags=["API", "image"]),
            TemplateInfo(name="custom", source="custom_node"),
        ]
        filtered = [t for t in templates if "API" not in t.tags]
        assert len(filtered) == 2
        assert all(t.name != "api_one" for t in filtered)

    def test_show_all_includes_api(self):
        templates = [
            TemplateInfo(name="local", source="package", tags=[]),
            TemplateInfo(name="api_one", source="package", tags=["API"]),
        ]
        assert len(templates) == 2
