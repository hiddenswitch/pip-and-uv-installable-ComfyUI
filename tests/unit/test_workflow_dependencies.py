from __future__ import annotations

import json

import pytest

from comfy.component_model.workflow_dependencies import (
    VIRTUAL_NODE_TYPES,
    extract_class_types_from_workflow,
    resolve_workflow_packages,
)


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

    builtin = frozenset({"KSampler"})
    packages = resolve_workflow_packages(workflow, builtin_class_types=builtin)
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
    builtin = frozenset({"KSampler"})
    packages = resolve_workflow_packages(workflow, builtin_class_types=builtin)
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
    builtin = frozenset({"KSampler"})
    packages = resolve_workflow_packages(workflow, builtin_class_types=builtin)
    assert "comfyui-videohelpersuite" in packages
    assert len(packages) >= 1


def _res4lyf_installed() -> bool:
    try:
        import importlib.metadata
        importlib.metadata.distribution("res4lyf")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


@pytest.mark.skipif(not _res4lyf_installed(), reason="res4lyf not installed")
def test_res4lyf_workflow_converts():
    """Verify a RES4LYF example workflow converts when the package is installed via serve-pip."""
    import importlib.metadata
    from comfy.nodes.package import import_all_nodes_in_workspace
    from comfy.execution_context import context_add_custom_nodes
    from comfy.component_model.workflow_convert import convert_ui_to_api

    dist = importlib.metadata.distribution("res4lyf")
    workflow_path = dist._path.parent / "_appmana_facade_res4lyf/_vendor/RES4LYF/example_workflows/chroma txt2img.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    nodes = import_all_nodes_in_workspace()
    with context_add_custom_nodes(nodes):
        result = convert_ui_to_api(workflow)

    assert isinstance(result, dict)
    assert len(result) > 0
    class_types = {v["class_type"] for v in result.values() if isinstance(v, dict)}
    assert "ClownsharKSampler_Beta" in class_types or "ClownModelLoader" in class_types
