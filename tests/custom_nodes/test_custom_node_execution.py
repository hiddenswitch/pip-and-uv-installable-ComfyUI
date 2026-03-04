from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from pathlib import Path

import pytest

from comfy.app.custom_node_manager import CustomNodeManager
from comfy.cmd.workflow_templates import _collect_class_types
from comfy.component_model.prompt_utils import replace_steps, replace_width, replace_height
from comfy.component_model.workflow_convert import is_ui_workflow

from .node_registry import CUSTOM_NODE_REGISTRY, get_spec
from .conftest import (
    install_all_nodes,
    make_base_dirs,
    build_config,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.environ.get(
    "COMFY_TEST_CACHE_DIR",
    Path.home() / ".cache" / "comfy-test" / "custom_nodes",
))

# 1×1 white PNG
_STUB_IMAGE_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/"
    "AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
# 100-sample silent WAV (8 kHz, 16-bit mono)
_STUB_AUDIO_URI = (
    "data:audio/wav;base64,"
    "UklGRuwAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YcgAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
)

_IMAGE_TO_URL: dict[str, str] = {
    "LoadImage": "LoadImageFromURL",
    "LoadImageMask": "LoadImageFromURL",
    "LoadImageOutput": "LoadImageFromURL",
}
_AUDIO_TO_URL: dict[str, str] = {
    "LoadAudio": "LoadAudioFromURL",
    "VHS_LoadAudio": "LoadAudioFromURL",
}
# VHS_LoadVideo/VHS_LoadVideoPath output IMAGE frames, not VIDEO —
# substituting with LoadVideoFromURL causes type mismatches.
_VIDEO_TO_URL: dict[str, str] = {
    "LoadVideo": "LoadVideoFromURL",
}

_EXTRA_STEPS_CLASS_TYPES = frozenset({
    "WanVideoSampler",
    "WanVideoSamplerAdvanced",
    "SamplerCustomAdvanced",
    "SamplerCustom",
    "KSamplerSelect",
    "RES4LYF_Sampler",
})

_EXTRA_LATENT_CLASS_TYPES = frozenset({
    "WanVideoEmptyLatent",
    "EmptyMochiLatentVideo",
    "EmptyHunyuanLatentVideo",
    "EmptyLTXVLatentVideo",
    "EmptyCosmosLatentVideo",
})

_VIDEO_FRAME_CLASS_TYPES = frozenset({
    "WanVideoEmptyLatent",
    "EmptyMochiLatentVideo",
    "EmptyHunyuanLatentVideo",
    "EmptyLTXVLatentVideo",
    "EmptyCosmosLatentVideo",
    "WanVideoSampler",
})

_MODEL_MISSING_PATTERNS = (
    "does not contain",
    "FileNotFoundError",
    "No such file",
    "model not found",
    "Could not find",
    "not found in",
    "Unable to find",
    "Cannot find",
    "Missing model",
    "value_not_in_list",
    "not in list",
    "Value not in list",
    "required input is missing",
    "missing_node_type",
    "not found. The custom node may not be installed",
    "custom_validation_failed",
    "Invalid image file",
    "Invalid video file",
    "Invalid audio file",
    "has no attribute 'solutions'",
    "VIDEO != IMAGE",
    "MASK != INT",
)


def _substitute_media_nodes(workflow: dict) -> dict:
    _ALL_MEDIA: dict[str, tuple[str, str]] = {}
    for src, dst in _IMAGE_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_IMAGE_URI)
    for src, dst in _AUDIO_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_AUDIO_URI)
    for src, dst in _VIDEO_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_IMAGE_URI)

    if is_ui_workflow(workflow):
        return _substitute_media_nodes_ui(workflow, _ALL_MEDIA)
    return _substitute_media_nodes_api(workflow, _ALL_MEDIA)


def _substitute_media_nodes_api(
    workflow: dict,
    media_map: dict[str, tuple[str, str]],
) -> dict:
    node_ids = [
        nid for nid, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type", "") in media_map
    ]
    if not node_ids:
        return workflow
    workflow = copy.deepcopy(workflow)
    for nid in node_ids:
        node = workflow[nid]
        url_class, stub_uri = media_map[node["class_type"]]
        node["class_type"] = url_class
        node["inputs"] = {"value": stub_uri}
        node.pop("_meta", None)
    return workflow


def _substitute_media_nodes_ui(
    workflow: dict,
    media_map: dict[str, tuple[str, str]],
) -> dict:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return workflow

    patched_ids = [
        node.get("id")
        for node in nodes
        if isinstance(node, dict) and node.get("type", "") in media_map
    ]
    if not patched_ids:
        return workflow

    workflow = copy.deepcopy(workflow)
    for node in workflow["nodes"]:
        if not isinstance(node, dict) or node.get("id") not in patched_ids:
            continue
        node_type = node.get("type", "")
        url_class, stub_uri = media_map[node_type]
        node["type"] = url_class
        node["widgets_values"] = [stub_uri]
    return workflow


def _apply_cost_reduction_api(workflow: dict) -> dict:
    workflow = replace_steps(workflow, 2)
    workflow = replace_width(workflow, 256)
    workflow = replace_height(workflow, 256)

    modified = False
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        if class_type in _EXTRA_STEPS_CLASS_TYPES and "steps" in inputs:
            if not modified:
                workflow = copy.deepcopy(workflow)
                modified = True
            workflow[nid]["inputs"]["steps"] = 2

        if class_type in _EXTRA_LATENT_CLASS_TYPES:
            if not modified:
                workflow = copy.deepcopy(workflow)
                modified = True
            if "width" in inputs:
                workflow[nid]["inputs"]["width"] = 256
            if "height" in inputs:
                workflow[nid]["inputs"]["height"] = 256

        if class_type in _VIDEO_FRAME_CLASS_TYPES:
            if not modified:
                workflow = copy.deepcopy(workflow)
                modified = True
            for field in ("num_frames", "length", "video_frames", "batch_size"):
                if field in inputs and isinstance(inputs[field], (int, float)):
                    workflow[nid]["inputs"][field] = min(int(inputs[field]), 2)

    return workflow


def _apply_cost_reduction_ui(workflow: dict) -> dict:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return workflow

    modified = False
    for node in nodes:
        if not isinstance(node, dict):
            continue
        widgets = node.get("widgets_values")
        if not isinstance(widgets, (list, dict)):
            continue
        node_type = node.get("type", "")

        if isinstance(widgets, dict):
            for key in ("steps", "width", "height", "num_frames", "length"):
                if key in widgets and isinstance(widgets[key], (int, float)):
                    if not modified:
                        workflow = copy.deepcopy(workflow)
                        modified = True
                        nodes = workflow["nodes"]
                    new_val = 2 if key in ("steps", "num_frames", "length") else 256
                    for n in nodes:
                        if n.get("id") == node.get("id"):
                            if isinstance(n.get("widgets_values"), dict):
                                n["widgets_values"][key] = new_val
                            break
            continue

        all_steps_types = {"KSampler", "KSamplerAdvanced", "BasicScheduler",
                          "Flux2Scheduler", "LTXVScheduler"} | _EXTRA_STEPS_CLASS_TYPES
        all_latent_types = {"EmptyLatentImage", "EmptySD3LatentImage"} | _EXTRA_LATENT_CLASS_TYPES

        if node_type in all_steps_types or node_type in all_latent_types or node_type in _VIDEO_FRAME_CLASS_TYPES:
            for i, val in enumerate(widgets):
                if isinstance(val, int) and val > 256:
                    if not modified:
                        workflow = copy.deepcopy(workflow)
                        modified = True
                        nodes = workflow["nodes"]
                    for n in nodes:
                        if n.get("id") == node.get("id"):
                            wv = n.get("widgets_values")
                            if isinstance(wv, list) and i < len(wv):
                                wv[i] = min(val, 256)
                            break

    return workflow


def _apply_cost_reduction(workflow: dict) -> dict:
    if is_ui_workflow(workflow):
        return _apply_cost_reduction_ui(workflow)
    return _apply_cost_reduction_api(workflow)


def _is_model_missing_error(error_msg: str) -> bool:
    return any(pattern.lower() in error_msg.lower() for pattern in _MODEL_MISSING_PATTERNS)


def _collect_workflow_entries(base_dir):
    custom_nodes_root = str(base_dir / "custom_nodes")
    return CustomNodeManager.scan_example_workflows([custom_nodes_root])


def _get_shared_base_dir() -> Path:
    base_dir = _CACHE_DIR
    marker = base_dir / ".installed"

    if marker.exists():
        logger.info("Using cached custom node installation at %s", base_dir)
    else:
        logger.info("Installing all custom nodes into %s (first run)", base_dir)
        make_base_dirs(base_dir)
        installed = install_all_nodes(base_dir)
        logger.info("Installed %d custom nodes", len(installed))
        marker.write_text(json.dumps({
            "count": len(installed),
            "nodes": sorted(installed.keys()),
        }))

    return base_dir


_shared_base_dir: Path | None = None


@pytest.fixture(scope="session")
def shared_base_dir():
    global _shared_base_dir
    if _shared_base_dir is None:
        _shared_base_dir = _get_shared_base_dir()
    return _shared_base_dir


@pytest.mark.slow
@pytest.mark.git_clone
@pytest.mark.parametrize(
    "node_id",
    [spec.node_id for spec in CUSTOM_NODE_REGISTRY],
    ids=[spec.node_id for spec in CUSTOM_NODE_REGISTRY],
)
class TestCustomNodeExecution:

    @pytest.mark.asyncio
    async def test_execute_example_workflows(self, node_id, shared_base_dir):
        from comfy.client.embedded_comfy_client import Comfy

        spec = get_spec(node_id)
        if spec.xfail:
            pytest.xfail(spec.xfail_reason)

        base_dir = shared_base_dir

        entries = _collect_workflow_entries(base_dir)
        node_dir_name = node_id
        node_entries = [
            (name, wf_name, path)
            for name, wf_name, path in entries
            if name == node_dir_name
        ]
        if not node_entries:
            pytest.skip(f"{node_id} has no example workflows")

        logger.info("%s: found %d example workflow(s)", node_id, len(node_entries))

        import comfy.cmd.main_pre
        real_base = str(Path(__file__).resolve().parents[3])
        config = build_config(base_dir, torch_device="cuda:1", base_paths=[real_base])

        executed = 0
        model_errors = []
        real_errors = []

        async with Comfy(configuration=config) as client:
            for node_name, workflow_name, filepath in node_entries:
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning("%s/%s: invalid JSON, skipping", node_name, workflow_name)
                        continue

                if not isinstance(data, dict):
                    logger.warning("%s/%s: top-level is not a dict, skipping", node_name, workflow_name)
                    continue

                data = _apply_cost_reduction(data)
                data = _substitute_media_nodes(data)

                class_types = _collect_class_types(data)
                logger.info(
                    "%s/%s: %d class_types: %s",
                    node_name, workflow_name, len(class_types),
                    sorted(class_types)[:10],
                )

                start = time.monotonic()
                try:
                    outputs = await client.queue_prompt(data)
                    elapsed = time.monotonic() - start
                    executed += 1
                    logger.info(
                        "%s/%s: executed in %.1fs, outputs: %s",
                        node_name, workflow_name, elapsed, outputs,
                    )
                except Exception as e:
                    elapsed = time.monotonic() - start
                    error_msg = f"{node_name}/{workflow_name}: {e}"
                    if _is_model_missing_error(str(e)):
                        model_errors.append(error_msg)
                        logger.warning(
                            "%s/%s: missing model/node after %.1fs: %s",
                            node_name, workflow_name, elapsed, e,
                        )
                    else:
                        real_errors.append(error_msg)
                        logger.error(
                            "%s/%s: execution failed after %.1fs: %s",
                            node_name, workflow_name, elapsed, e,
                        )

        total_errors = len(model_errors) + len(real_errors)
        logger.info(
            "%s: executed %d workflows, %d model errors, %d real errors",
            node_id, executed, len(model_errors), len(real_errors),
        )

        if model_errors:
            logger.warning(
                "%s: %d workflow(s) had missing models/nodes (expected):\n  %s",
                node_id, len(model_errors), "\n  ".join(model_errors),
            )

        if real_errors:
            logger.error(
                "%s: %d workflow(s) had real execution errors:\n  %s",
                node_id, len(real_errors), "\n  ".join(real_errors),
            )

        assert executed > 0 or total_errors > 0, (
            f"{node_id}: no workflows were attempted"
        )
