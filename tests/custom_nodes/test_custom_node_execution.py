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

from comfy.component_model.node_registry import CUSTOM_NODE_REGISTRY, get_spec
from .conftest import (
    add_node_site_to_path,
    install_all_nodes,
    make_base_dirs,
    build_config,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.environ.get(
    "COMFY_TEST_CACHE_DIR",
    Path.home() / ".cache" / "comfy-test" / "custom_nodes",
))

_STUB_IMAGE_URI = "pkg://tests.custom_nodes.test_data/president_official_portrait_hires2-1-1024x1024.jpg"
_STUB_AUDIO_URI = "pkg://tests.custom_nodes.test_data/test_audio.wav"
_STUB_VIDEO_URI = "pkg://tests.custom_nodes.test_data/test_video.mp4"

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
# Instead, we patch their "video" input field in _patch_vhs_video_inputs.
_VIDEO_TO_URL: dict[str, str] = {
    "LoadVideo": "LoadVideoFromURL",
}

_VHS_VIDEO_CLASS_TYPES = frozenset({"VHS_LoadVideo", "VHS_LoadVideoPath"})

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

_MODEL_MISSING_PATTERNS: tuple[str, ...] = tuple()

# Segformer local paths → proper HuggingFace repo IDs.
_SEGFORMER_REPO_MAP: dict[str, str] = {
    "segformer_b3_clothes": "mattmdjaga/segformer_b3_clothes",
    "segformer_b2_clothes": "mattmdjaga/segformer_b2_clothes",
    "segformer_b3_fashion": "mattmdjaga/segformer_b3_fashion",
}


def _patch_vhs_video_inputs(workflow: dict) -> dict:
    """Replace hardcoded video filenames in VHS_LoadVideo nodes with test video."""
    if is_ui_workflow(workflow):
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list):
            return workflow
        need_patch = any(
            isinstance(n, dict) and n.get("type", "") in _VHS_VIDEO_CLASS_TYPES
            for n in nodes
        )
        if not need_patch:
            return workflow
        workflow = copy.deepcopy(workflow)
        for node in workflow["nodes"]:
            if isinstance(node, dict) and node.get("type", "") in _VHS_VIDEO_CLASS_TYPES:
                wv = node.get("widgets_values")
                if isinstance(wv, list) and wv:
                    wv[0] = _STUB_VIDEO_URI
                elif isinstance(wv, dict) and "video" in wv:
                    wv["video"] = _STUB_VIDEO_URI
    else:
        need_patch = any(
            isinstance(n, dict) and n.get("class_type", "") in _VHS_VIDEO_CLASS_TYPES
            for n in workflow.values()
        )
        if not need_patch:
            return workflow
        workflow = copy.deepcopy(workflow)
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type", "") in _VHS_VIDEO_CLASS_TYPES:
                inputs = node.get("inputs", {})
                if "video" in inputs:
                    inputs["video"] = _STUB_VIDEO_URI
    return workflow


def _substitute_media_nodes(workflow: dict) -> dict:
    _ALL_MEDIA: dict[str, tuple[str, str]] = {}
    for src, dst in _IMAGE_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_IMAGE_URI)
    for src, dst in _AUDIO_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_AUDIO_URI)
    for src, dst in _VIDEO_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_VIDEO_URI)

    if is_ui_workflow(workflow):
        workflow = _substitute_media_nodes_ui(workflow, _ALL_MEDIA)
    else:
        workflow = _substitute_media_nodes_api(workflow, _ALL_MEDIA)
    return _patch_vhs_video_inputs(workflow)


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



# Node types to bypass (mode=4) in UI workflows before conversion.
_BYPASS_NODE_TYPES: frozenset[str] = frozenset({
    "WanVideoTorchCompileSettings",
})


def _bypass_nodes(workflow: dict) -> dict:
    """Set mode=4 (bypass) on specific node types in a UI workflow."""
    if not is_ui_workflow(workflow):
        return workflow
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return workflow
    need_patch = any(
        isinstance(n, dict) and n.get("type", "") in _BYPASS_NODE_TYPES
        for n in nodes
    )
    if not need_patch:
        return workflow
    workflow = copy.deepcopy(workflow)
    for node in workflow["nodes"]:
        if isinstance(node, dict) and node.get("type", "") in _BYPASS_NODE_TYPES:
            node["mode"] = 4
    return workflow


def _install_segformer_monkeypatch():
    """Monkeypatch segformer_ultra.py to use HuggingFace repo IDs instead of local paths."""
    try:
        import sys
        for mod_name, mod in list(sys.modules.items()):
            if "segformer_ultra" in mod_name and hasattr(mod, "get_segmentation"):
                _orig_get_seg = mod.get_segmentation

                def _patched_get_seg(tensor_image, model_name='segformer_b2_clothes', _orig=_orig_get_seg):
                    import os
                    import folder_paths
                    # If the local model folder doesn't exist, rewrite the path to a HF repo ID
                    model_folder_path = os.path.join(folder_paths.models_dir, model_name)
                    if not os.path.isdir(model_folder_path) and model_name in _SEGFORMER_REPO_MAP:
                        # Create the directory and download via HF hub
                        from huggingface_hub import snapshot_download
                        snapshot_download(
                            _SEGFORMER_REPO_MAP[model_name],
                            local_dir=model_folder_path,
                        )
                    return _orig(tensor_image, model_name)

                mod.get_segmentation = _patched_get_seg
                logger.info("Monkeypatched segformer_ultra.get_segmentation for HF repo IDs")
                break
    except Exception as e:
        logger.warning("Failed to monkeypatch segformer_ultra: %s", e)


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

    add_node_site_to_path(base_dir)
    return base_dir


# ---------------------------------------------------------------------------
# Collect (node_id, workflow_name, filepath) at import time for parametrize.
# Only scans the filesystem — no heavy imports or node loading.
# ---------------------------------------------------------------------------
def _collect_all_workflow_params() -> list[tuple[str, str, str]]:
    """Scan the cache dir for example workflow JSON files.

    Returns (node_id, workflow_name, filepath) triples.
    If the cache dir doesn't exist yet, returns an empty list (the session
    fixture will install nodes on first run).
    """
    custom_nodes_root = _CACHE_DIR / "custom_nodes"
    if not custom_nodes_root.is_dir():
        return []
    results = []
    for folder_name in CustomNodeManager.EXAMPLE_WORKFLOW_FOLDER_NAMES:
        for filepath in sorted(custom_nodes_root.glob(f"*/{folder_name}/*.json")):
            node_id = filepath.parent.parent.name
            workflow_name = filepath.stem
            results.append((node_id, workflow_name, str(filepath)))
    return results


_ALL_WORKFLOW_PARAMS = _collect_all_workflow_params()

# Build the pytest parameter list: id string is "node_id/workflow_name"
_PARAM_IDS = [f"{node_id}/{wf}" for node_id, wf, _ in _ALL_WORKFLOW_PARAMS]

_shared_base_dir: Path | None = None


@pytest.fixture(scope="session")
def shared_base_dir():
    global _shared_base_dir
    if _shared_base_dir is None:
        _shared_base_dir = _get_shared_base_dir()
    return _shared_base_dir


class TestCustomNodeExecution:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "node_id,workflow_name,workflow_path",
        _ALL_WORKFLOW_PARAMS,
        ids=_PARAM_IDS,
    )
    async def test_execute_workflow(self, node_id, workflow_name, workflow_path, shared_base_dir):
        from comfy.client.embedded_comfy_client import Comfy

        spec = get_spec(node_id)
        if spec is not None and spec.xfail:
            pytest.xfail(spec.xfail_reason)

        base_dir = shared_base_dir

        with open(workflow_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pytest.skip(f"{node_id}/{workflow_name}: invalid JSON")

        if not isinstance(data, dict):
            pytest.skip(f"{node_id}/{workflow_name}: top-level is not a dict")

        data = _apply_cost_reduction(data)
        data = _substitute_media_nodes(data)
        data = _bypass_nodes(data)

        class_types = _collect_class_types(data)
        logger.info(
            "%s/%s: %d class_types: %s",
            node_id, workflow_name, len(class_types),
            sorted(class_types)[:10],
        )

        import comfy.cmd.main_pre
        real_base = str(Path(__file__).resolve().parents[3])
        config = build_config(base_dir, base_paths=[real_base])

        async with Comfy(configuration=config) as client:
            _install_segformer_monkeypatch()
            start = time.monotonic()
            outputs = await client.queue_prompt(data)
            elapsed = time.monotonic() - start
            logger.info(
                "%s/%s: executed in %.1fs, outputs: %s",
                node_id, workflow_name, elapsed, outputs,
            )
