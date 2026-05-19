from __future__ import annotations

import itertools

from comfy.cli_args_types import Configuration
from comfy.component_model.workflow_convert import apply_ui_seed_quantity, convert_ui_to_api
from comfy.entrypoints import workflow as workflow_entrypoint
from comfy.entrypoints.workflow import expand_workflow_quantity
from comfy.execution_context import context_add_custom_nodes
from comfy.nodes.package_typing import ExportedNodes


class _KSamplerLike:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**64, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler", "dpmpp"], {}),
                "scheduler": (["normal", "karras"], {}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }


def _api_workflow(seed=10, noise_seed=20):
    return {
        "1": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20}},
        "2": {"class_type": "RandomNoise", "inputs": {"noise_seed": noise_seed}},
    }


def _ui_workflow(seed=42, control="increment"):
    return {
        "nodes": [
            {
                "id": 1,
                "type": "KSampler",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [seed, control, 20, 8, "euler", "normal", 1],
            },
        ],
        "links": [],
    }


def _with_ksampler_nodes():
    exported = ExportedNodes()
    exported.NODE_CLASS_MAPPINGS["KSampler"] = _KSamplerLike
    return context_add_custom_nodes(exported)


def test_api_quantity_with_seed_override_increments_all_seed_fields():
    expanded = expand_workflow_quantity(_api_workflow(), Configuration(quantity=3, seed=100))

    assert [wf["1"]["inputs"]["seed"] for wf in expanded] == [100, 101, 102]
    assert [wf["2"]["inputs"]["noise_seed"] for wf in expanded] == [100, 101, 102]


def test_api_quantity_without_seed_override_uses_random_bases(monkeypatch):
    monkeypatch.setattr(workflow_entrypoint, "_random_seed", lambda: 1000)

    expanded = expand_workflow_quantity(_api_workflow(), Configuration(quantity=3))

    assert [wf["1"]["inputs"]["seed"] for wf in expanded] == [1000, 1001, 1002]
    assert [wf["2"]["inputs"]["noise_seed"] for wf in expanded] == [1000, 1001, 1002]


def test_api_quantity_one_without_seed_override_preserves_existing_seed():
    expanded = expand_workflow_quantity(_api_workflow(seed=12, noise_seed=34), Configuration(quantity=1))

    assert len(expanded) == 1
    assert expanded[0]["1"]["inputs"]["seed"] == 12
    assert expanded[0]["2"]["inputs"]["noise_seed"] == 34


def test_ui_quantity_respects_increment_control_after_generate():
    with _with_ksampler_nodes():
        expanded = expand_workflow_quantity(_ui_workflow(seed=42, control="increment"), Configuration(quantity=3))

    assert [wf["1"]["inputs"]["seed"] for wf in expanded] == [42, 43, 44]


def test_ui_quantity_respects_decrement_with_seed_override():
    with _with_ksampler_nodes():
        expanded = expand_workflow_quantity(_ui_workflow(seed=42, control="decrement"), Configuration(quantity=3, seed=100))

    assert [wf["1"]["inputs"]["seed"] for wf in expanded] == [100, 99, 98]


def test_ui_quantity_respects_fixed_control_after_generate():
    with _with_ksampler_nodes():
        expanded = expand_workflow_quantity(_ui_workflow(seed=42, control="fixed"), Configuration(quantity=3))

    assert [wf["1"]["inputs"]["seed"] for wf in expanded] == [42, 42, 42]


def test_ui_quantity_respects_randomize_control_after_generate():
    seeds = itertools.count(700)
    with _with_ksampler_nodes():
        ui_jobs = [
            apply_ui_seed_quantity(_ui_workflow(seed=42, control="randomize"), i, random_seed=lambda: next(seeds))
            for i in range(3)
        ]
        expanded = [convert_ui_to_api(job) for job in ui_jobs]

    assert [wf["1"]["inputs"]["seed"] for wf in expanded] == [700, 701, 702]
