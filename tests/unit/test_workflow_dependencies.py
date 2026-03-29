from __future__ import annotations

import json

import pytest

from comfy.component_model.workflow_dependencies import (
    VIRTUAL_NODE_TYPES,
    extract_class_types_from_workflow,
    resolve_workflow_packages_versioned,
)


def _package_names(workflow, builtin=frozenset()):
    return [name for name, _ in resolve_workflow_packages_versioned(workflow, builtin_class_types=builtin)]


def test_primitive_node_excluded_from_packages():
    workflow = {
        "nodes": [
            {"id": 1, "type": "KSampler"},
            {"id": 2, "type": "PrimitiveNode"},
            {"id": 3, "type": "Note"},
            {"id": 4, "type": "MarkdownNote"},
            {"id": 5, "type": "Reroute"},
        ]
    }
    class_types = extract_class_types_from_workflow(workflow)
    assert "PrimitiveNode" in class_types
    assert "Note" in class_types
    assert "MarkdownNote" in class_types
    assert "Reroute" in class_types

    packages = _package_names(workflow, builtin=frozenset({"KSampler"}))
    assert packages == []


def test_virtual_node_types_are_complete():
    for vt in ("PrimitiveNode", "Note", "MarkdownNote", "Reroute"):
        assert vt in VIRTUAL_NODE_TYPES


def test_uuid_subgraph_nodes_excluded():
    workflow = {
        "nodes": [
            {"id": 1, "type": "KSampler"},
            {"id": 2, "type": "e805dea3-d9c7-4e74-924c-8b2d21f5e623"},
        ]
    }
    packages = _package_names(workflow, builtin=frozenset({"KSampler"}))
    assert packages == []


def test_api_format_extraction():
    workflow = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {}},
    }
    class_types = extract_class_types_from_workflow(workflow)
    assert class_types == {"KSampler", "CLIPTextEncode"}


def test_resolve_with_custom_nodes():
    workflow = {
        "nodes": [
            {"id": 1, "type": "KSampler"},
            {"id": 2, "type": "VHS_LoadVideo"},
            {"id": 3, "type": "PrimitiveNode"},
        ]
    }
    packages = _package_names(workflow, builtin=frozenset({"KSampler"}))
    assert "comfyui-videohelpersuite" in packages
    assert len(packages) >= 1


def _res4lyf_installed() -> bool:
    try:
        import importlib.metadata
        importlib.metadata.distribution("res4lyf")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _res4lyf_workflow_path() -> str:
    import importlib.metadata
    dist = importlib.metadata.distribution("res4lyf")
    return str(dist._path.parent / "_appmana_facade_res4lyf/_vendor/RES4LYF/example_workflows/chroma txt2img.json")


# Patch community model filenames to known downloadable names.
_RES4LYF_MODEL_PATCHES: dict[str, str] = {
    "ae.sft": "ae.safetensors",
    "chroma-unlocked-v37-detail-calibrated.safetensors": "chroma-unlocked-v37.safetensors",
}


def _patch_model_names(workflow: dict) -> dict:
    import copy
    workflow = copy.deepcopy(workflow)
    for node in workflow.get("nodes", []):
        wv = node.get("widgets_values")
        if isinstance(wv, list):
            for i, val in enumerate(wv):
                if isinstance(val, str) and val in _RES4LYF_MODEL_PATCHES:
                    wv[i] = _RES4LYF_MODEL_PATCHES[val]
        elif isinstance(wv, dict):
            for k, val in wv.items():
                if isinstance(val, str) and val in _RES4LYF_MODEL_PATCHES:
                    wv[k] = _RES4LYF_MODEL_PATCHES[val]
    return workflow


@pytest.mark.skipif(not _res4lyf_installed(), reason="res4lyf not installed")
def test_res4lyf_workflow_converts():
    """Verify a RES4LYF example workflow converts when the package is installed via serve-pip."""
    from comfy.nodes.package import import_all_nodes_in_workspace
    from comfy.execution_context import context_add_custom_nodes
    from comfy.component_model.workflow_convert import convert_ui_to_api

    workflow = json.loads(open(_res4lyf_workflow_path(), encoding="utf-8").read())

    nodes = import_all_nodes_in_workspace()
    with context_add_custom_nodes(nodes):
        result = convert_ui_to_api(workflow)

    assert isinstance(result, dict)
    assert len(result) > 0
    class_types = {v["class_type"] for v in result.values() if isinstance(v, dict)}
    assert "ClownsharKSampler_Beta" in class_types or "ClownModelLoader" in class_types


@pytest.mark.skipif(not _res4lyf_installed(), reason="res4lyf not installed")
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_res4lyf_workflow_executes():
    """Execute the patched chroma txt2img workflow end-to-end."""
    from comfy.client.embedded_comfy_client import Comfy
    from comfy.cli_args import default_configuration
    from tests.custom_nodes.test_custom_node_execution import _apply_cost_reduction

    workflow = json.loads(open(_res4lyf_workflow_path(), encoding="utf-8").read())
    workflow = _patch_model_names(workflow)
    workflow = _apply_cost_reduction(workflow)

    config = default_configuration()
    config.database_url = "sqlite:///:memory:"

    async with Comfy(configuration=config) as client:
        outputs = await client.queue_prompt(workflow)

    assert outputs is not None
    assert len(outputs) > 0
