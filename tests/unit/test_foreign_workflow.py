"""Format detection + A1111/Fooocus → ComfyUI translation."""
from __future__ import annotations

import json

import pytest

from comfy.component_model.foreign_workflow import (
    UnsupportedWorkflowFormatError,
    detect_workflow_format,
    parse_a1111_dump,
    parse_fooocus_preset,
    synthesize_comfyui_workflow,
    translate_foreign_workflow,
)


A1111_SAMPLE = (
    "masterpiece, 1girl, beautiful, <lora:add_detail:0.8>, <lora:more_detail:1.2>\n"
    "Negative prompt: low quality, worst quality, blurry\n"
    "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7.5, Seed: 42, "
    'Size: 768x1024, Model hash: abc123, Model: dreamshaper_8, '
    'Lora hashes: "add_detail: deadbeef, more_detail: cafebabe"'
)


def test_detect_comfyui_ui():
    assert detect_workflow_format({"nodes": [], "links": []}) == "comfyui-ui"


def test_detect_comfyui_api():
    assert detect_workflow_format({"1": {"class_type": "KSampler", "inputs": {}}}) == "comfyui-api"


def test_detect_a1111_text():
    assert detect_workflow_format(A1111_SAMPLE) == "a1111"


def test_detect_a1111_bytes():
    assert detect_workflow_format(A1111_SAMPLE.encode("utf-8")) == "a1111"


def test_detect_fooocus():
    parsed = {"prompt": "a", "negative_prompt": "b", "base_model_name": "sd.safetensors"}
    assert detect_workflow_format(parsed) == "fooocus"


def test_detect_swarmui():
    assert detect_workflow_format({"rawInput": {"model": "x"}}) == "swarmui"


def test_detect_invokeai():
    assert detect_workflow_format({"graph": {"nodes": {"a": {"type": "main_model"}}}}) == "invokeai"


def test_detect_krita_ai():
    parsed = {"version": "1.0", "kind": "txt2img", "checkpoint": "sd.safetensors"}
    assert detect_workflow_format(parsed) == "krita-ai"


def test_detect_unknown():
    assert detect_workflow_format({"hello": "world"}) == "unknown"
    assert detect_workflow_format("just a sentence") == "unknown"


def test_parse_a1111_dump():
    p = parse_a1111_dump(A1111_SAMPLE)
    assert "masterpiece" in p["positive"]
    assert p["negative"] == "low quality, worst quality, blurry"
    assert p["steps"] == 30
    assert p["sampler"] == "DPM++ 2M Karras"
    assert p["cfg_scale"] == 7.5
    assert p["seed"] == 42
    assert p["width"] == 768
    assert p["height"] == 1024
    assert p["model"] == "dreamshaper_8"
    # Lora hashes line gives names; weights default to 1.0 there
    names = [n for n, _ in p["loras"]]
    assert "add_detail" in names and "more_detail" in names


def test_synthesize_workflow_shape():
    p = parse_a1111_dump(A1111_SAMPLE)
    wf = synthesize_comfyui_workflow(p)
    classes = sorted({n["class_type"] for n in wf.values()})
    assert "CheckpointLoaderSimple" in classes
    assert "KSampler" in classes
    assert "VAEDecode" in classes
    assert "SaveImage" in classes
    # Two LoRAs from the prompt + two from "Lora hashes" — but dedup by name.
    n_lora = sum(1 for n in wf.values() if n["class_type"] == "LoraLoader")
    assert n_lora == 2


def test_synthesize_uses_inline_lora_weights():
    p = parse_a1111_dump(A1111_SAMPLE)
    wf = synthesize_comfyui_workflow(p)
    weights = sorted(n["inputs"]["strength_model"] for n in wf.values() if n["class_type"] == "LoraLoader")
    # Inline <lora:...:weight> values should win over hashes-line default of 1.0
    assert 0.8 in weights
    assert 1.2 in weights


def test_synthesize_maps_dpmpp_2m_karras():
    p = parse_a1111_dump(A1111_SAMPLE)
    wf = synthesize_comfyui_workflow(p)
    sampler_node = next(n for n in wf.values() if n["class_type"] == "KSampler")
    assert sampler_node["inputs"]["sampler_name"] == "dpmpp_2m"
    assert sampler_node["inputs"]["scheduler"] == "karras"


def test_translate_a1111_returns_workflow():
    wf = translate_foreign_workflow(A1111_SAMPLE, source="test.txt")
    assert any(n["class_type"] == "KSampler" for n in wf.values())


def test_translate_fooocus_list_loras():
    parsed = {
        "prompt": "a forest at sunrise",
        "negative_prompt": "blurry",
        "performance_selection_to_steps": 30,
        "guidance_scale": 7.5,
        "seed": 99,
        "aspect_ratios_selection_width": 1024,
        "aspect_ratios_selection_height": 1024,
        "base_model_name": "sd_xl_base_1.0.safetensors",
        "default_loras": [["true", "more_details.safetensors", 0.7, 0.7]],
    }
    p = parse_fooocus_preset(parsed)
    wf = synthesize_comfyui_workflow(p)
    assert any(n["class_type"] == "LoraLoader" for n in wf.values())


def test_translate_unsupported_raises():
    with pytest.raises(UnsupportedWorkflowFormatError) as exc:
        translate_foreign_workflow({"rawInput": {"model": "x"}}, source="weird.json")
    assert exc.value.kind == "swarmui"
    assert "weird.json" in str(exc.value)


def test_unsupported_is_apivalueerror():
    from comfy.api.exceptions import ApiValueError
    err = UnsupportedWorkflowFormatError("invokeai", source="x")
    assert isinstance(err, ApiValueError)
    assert isinstance(err, ValueError)
