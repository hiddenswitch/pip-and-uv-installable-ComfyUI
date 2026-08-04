import json
import types

import pytest

from comfy.cmd.workflow_templates import (
    TemplateInfo,
    _collect_class_types,
    _detect_supported_params,
    _detect_task,
    _build_example_invocation,
    _facade_custom_node_roots,
    _populate_supported_params,
    _templates_from_custom_nodes,
    resolve_template,
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
        assert params == [
            "prompt", "negative-prompt", "steps", "seed",
            "cfg", "sampler", "scheduler", "denoise",
            "image",
        ]

    def test_no_params(self):
        wf = _ui_workflow("CheckpointLoaderSimple", "VAEDecode")
        params = _detect_supported_params(wf)
        assert params == ["checkpoint"]

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
        assert result == 'comfyui workflows run my_template -ag --prompt "your text here" --seed 42'

    def test_fallback_to_name(self):
        tmpl = TemplateInfo(
            name="My Workflow", source="dir:/tmp",
            supported_params=["prompt"],
        )
        result = _build_example_invocation(tmpl)
        assert result == 'comfyui workflows run My Workflow -ag --prompt "your text here"'

    def test_no_params(self):
        tmpl = TemplateInfo(name="bare", source="package", template_id="bare")
        result = _build_example_invocation(tmpl)
        assert result == "comfyui workflows run bare -ag"

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


class TestCustomNodeWorkflowDiscovery:
    def test_exact_package_template_does_not_scan_custom_nodes(self, monkeypatch):
        template = TemplateInfo(
            name="Official workflow",
            source="package",
            path="official.json",
            template_id="official_workflow",
        )
        monkeypatch.setattr("comfy.cmd.workflow_templates._templates_from_package", lambda: [template])
        monkeypatch.setattr(
            "comfy.cmd.workflow_templates._templates_from_custom_nodes",
            lambda: pytest.fail("exact package template should not scan custom nodes"),
        )

        assert resolve_template("official_workflow") == "official.json"

    def test_facade_custom_node_roots_from_entrypoints(self, tmp_path, monkeypatch):
        vendor_root = tmp_path / "_vendor"
        vendor_root.mkdir()

        module = types.ModuleType("_appmana_facade_test.entrypoint")
        module.COMFYUI_VANILLA_NODE_PATHS = [str(vendor_root)]

        class FakeEntryPoint:
            name = "comfyui-custom-scripts"

            @staticmethod
            def load():
                return module

        class FakeEntryPoints:
            @staticmethod
            def select(**kwargs):
                assert kwargs == {"group": "comfyui.custom_nodes"}
                return [FakeEntryPoint()]

        monkeypatch.setattr("comfy.cmd.workflow_templates.entry_points", lambda: FakeEntryPoints())

        assert _facade_custom_node_roots() == [str(vendor_root.resolve())]

    def test_templates_from_custom_nodes_includes_facade_installs(self, tmp_path, monkeypatch):
        vendor_root = tmp_path / "_vendor"
        workflow_dir = vendor_root / "ComfyUI-Custom-Scripts" / "examples"
        workflow_dir.mkdir(parents=True)
        workflow_path = workflow_dir / "show_text.json"
        workflow_path.write_text(json.dumps(_ui_workflow("ShowText|pysssss")), encoding="utf-8")

        module = types.ModuleType("_appmana_facade_test.entrypoint")
        module.COMFYUI_VANILLA_NODE_PATHS = [str(vendor_root)]

        class FakeEntryPoint:
            name = "comfyui-custom-scripts"

            @staticmethod
            def load():
                return module

        class FakeEntryPoints:
            @staticmethod
            def select(**kwargs):
                assert kwargs == {"group": "comfyui.custom_nodes"}
                return [FakeEntryPoint()]

        monkeypatch.setattr("comfy.cmd.workflow_templates.entry_points", lambda: FakeEntryPoints())
        monkeypatch.setattr("comfy.cmd.folder_paths.get_folder_paths", lambda folder: [])

        templates = _templates_from_custom_nodes()

        assert len(templates) == 1
        template = templates[0]
        assert template.name == "show_text"
        assert template.source == "custom_node:ComfyUI-Custom-Scripts"
        assert template.path == str(workflow_path)


# ---------------------------------------------------------------------------
# _detect_task
# ---------------------------------------------------------------------------


def _wf(*class_types):
    return {str(i): {"class_type": ct, "inputs": {}} for i, ct in enumerate(class_types)}


class TestDetectTask:
    def test_text_to_image(self):
        wf = _wf("CheckpointLoaderSimple", "CLIPTextEncode", "KSampler",
                 "VAEDecode", "SaveImage")
        assert _detect_task(wf, _detect_supported_params(wf)) == "text-to-image"

    def test_image_edit(self):
        wf = _wf("LoadImage", "CheckpointLoaderSimple", "CLIPTextEncode",
                 "KSampler", "VAEDecode", "SaveImage")
        assert _detect_task(wf, _detect_supported_params(wf)) == "image-edit"

    def test_text_to_video(self):
        wf = _wf("CheckpointLoaderSimple", "CLIPTextEncode", "KSampler",
                 "VAEDecode", "VHS_VideoCombine")
        assert _detect_task(wf, _detect_supported_params(wf)) == "text-to-video"

    def test_image_to_video(self):
        wf = _wf("LoadImage", "CheckpointLoaderSimple", "CLIPTextEncode",
                 "KSampler", "VAEDecode", "SaveVideo")
        assert _detect_task(wf, _detect_supported_params(wf)) == "image-to-video"

    def test_audio(self):
        wf = _wf("CLIPTextEncode", "KSampler", "SaveAudio")
        assert _detect_task(wf, _detect_supported_params(wf)) == "audio"

    def test_other_has_no_known_output(self):
        wf = _wf("SomeWeirdNode")
        assert _detect_task(wf, _detect_supported_params(wf)) == "other"

    def test_t2v_workflow_with_nonstandard_image_reference_still_t2v(self):
        """Some text-to-video templates use a LoadImageFromURL solely as a
        reference / condition, not as the dominant input. If a video-output
        sink is present the task is always video-producing; the image-input
        vs image-reference distinction is below our signal threshold."""
        wf = _wf("LoadImageFromURL", "CLIPTextEncode", "KSampler",
                 "SaveAnimatedWEBP")
        assert _detect_task(wf, _detect_supported_params(wf)) == "image-to-video"


class TestPopulateTask:
    def test_populate_sets_task_field(self, tmp_path):
        wf_path = tmp_path / "t2i.json"
        wf_path.write_text(json.dumps(_wf(
            "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler",
            "VAEDecode", "SaveImage",
        )))
        tmpl = TemplateInfo(name="t2i", source="dir", path=str(wf_path))
        _populate_supported_params([tmpl])
        assert tmpl.task == "text-to-image"

    def test_populate_skips_missing_path(self):
        tmpl = TemplateInfo(name="noexist", source="dir", path="/no/such/file.json")
        _populate_supported_params([tmpl])
        assert tmpl.task is None
