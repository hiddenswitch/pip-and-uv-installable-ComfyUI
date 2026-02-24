from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from comfy.app.custom_node_manager import CustomNodeManager

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = frozenset({
    ".safetensors", ".gguf", ".ckpt", ".pth", ".pt", ".bin", ".onnx",
})

LOADER_FOLDER_MAP: dict[str, str] = {
    "CheckpointLoaderSimple": "checkpoints",
    "CheckpointLoader": "checkpoints",
    "UNETLoader": "diffusion_models",
    "WanVideoModelLoader": "diffusion_models",
    "VAELoader": "vae",
    "WanVideoVAELoader": "vae",
    "LoraLoader": "loras",
    "LoraLoaderModelOnly": "loras",
    "WanVideoLoraSelect": "loras",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "TripleCLIPLoader": "text_encoders",
    "LoadWanVideoT5TextEncoder": "text_encoders",
    "CLIPVisionLoader": "clip_vision",
    "ControlNetLoader": "controlnet",
    "DiffControlNetLoader": "diff_controlnet",
    "UpscaleModelLoader": "upscale_models",
    "StyleModelLoader": "style_models",
    "GLIGENLoader": "gligen",
    "DownloadAndLoadSAM2Model": "sams",
    "Sam2Segmentation": "sams",
    "DownloadAndLoadDepthAnythingV2Model": "depthanything",
    "DownloadAndLoadFlorence2Model": "LLM",
    "IPAdapterModelLoader": "ipadapter",
    "IPAdapterInsightFaceLoader": "insightface",
    "UltralyticsDetectorProvider": "ultralytics",
    "SAMLoader": "sams",
    "VHS_LoadVideo": "__skip__",
    "LoadImage": "__skip__",
    "LoadVideo": "__skip__",
}


def _is_model_filename(value: str) -> bool:
    if not isinstance(value, str):
        return False
    _, ext = os.path.splitext(value)
    if ext.lower() not in MODEL_EXTENSIONS:
        return False
    if value.startswith("http://") or value.startswith("https://"):
        return False
    if len(value) > 500:
        return False
    return True


def _extract_from_api_format(workflow: dict) -> list[tuple[str, str]]:
    results = []
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if _is_model_filename(value):
                results.append((class_type, value))
    return results


def _extract_from_ui_format(workflow: dict) -> list[tuple[str, str]]:
    results = []
    nodes = workflow.get("nodes", [])
    if not isinstance(nodes, list):
        return results
    for node in nodes:
        if not isinstance(node, dict):
            continue
        class_type = node.get("type", "")
        widgets = node.get("widgets_values", [])
        if isinstance(widgets, list):
            for value in widgets:
                if _is_model_filename(value):
                    results.append((class_type, value))
        elif isinstance(widgets, dict):
            for value in widgets.values():
                if _is_model_filename(value):
                    results.append((class_type, value))
    return results


def extract_model_references(workflow: dict) -> list[dict]:
    from comfy.component_model.workflow_convert import is_ui_workflow

    if is_ui_workflow(workflow):
        raw = _extract_from_ui_format(workflow)
    else:
        raw = _extract_from_api_format(workflow)

    results = []
    for class_type, filename in raw:
        folder = LOADER_FOLDER_MAP.get(class_type, "unknown")
        if folder == "__skip__":
            continue
        results.append({
            "class_type": class_type,
            "filename": filename,
            "folder_name": folder,
        })
    return results


def scan_all_workflows(custom_nodes_root: str) -> dict:
    entries = CustomNodeManager.scan_example_workflows([custom_nodes_root])
    logger.info("Found %d example workflows total", len(entries))

    by_node: dict[str, list[dict]] = defaultdict(list)

    for node_name, workflow_name, filepath in entries:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("%s/%s: could not read workflow", node_name, workflow_name)
            continue

        if not isinstance(data, dict):
            continue

        refs = extract_model_references(data)
        for ref in refs:
            ref["workflow"] = workflow_name
            ref["source_node"] = node_name
        by_node[node_name].extend(refs)

    result = {}
    for node_id, refs in by_node.items():
        seen = set()
        unique = []
        for ref in refs:
            key = (ref["class_type"], ref["filename"], ref["folder_name"])
            if key not in seen:
                seen.add(key)
                unique.append(ref)
        result[node_id] = unique

    return result


def generate_report(model_refs: dict) -> str:
    lines = []
    for node_id in sorted(model_refs.keys()):
        refs = model_refs[node_id]
        if not refs:
            continue
        lines.append(f"\n## {node_id} ({len(refs)} model references)")
        by_folder: dict[str, list] = defaultdict(list)
        for ref in refs:
            by_folder[ref["folder_name"]].append(ref)
        for folder in sorted(by_folder.keys()):
            folder_refs = by_folder[folder]
            lines.append(f"  [{folder}]")
            for ref in folder_refs:
                lines.append(f"    {ref['class_type']}: {ref['filename']}")
    return "\n".join(lines)


def test_extract_model_references(tmp_path):
    from .conftest import install_all_nodes, make_base_dirs

    base_dir = tmp_path / "base"
    make_base_dirs(base_dir)
    installed = install_all_nodes(base_dir)
    logger.info("Installed %d nodes", len(installed))

    custom_nodes_root = str(base_dir / "custom_nodes")
    model_refs = scan_all_workflows(custom_nodes_root)

    output_path = Path(__file__).parent / "model_references.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_refs, f, indent=2)
    logger.info("Wrote model references to %s", output_path)

    report = generate_report(model_refs)
    logger.info("Model reference report:\n%s", report)

    total = sum(len(refs) for refs in model_refs.values())
    logger.info("Total: %d unique model references across %d nodes", total, len(model_refs))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if len(sys.argv) > 1:
        custom_nodes_root = sys.argv[1]
    else:
        custom_nodes_root = str(Path(__file__).resolve().parents[3] / "custom_nodes")

    if not Path(custom_nodes_root).is_dir():
        print(f"Custom nodes directory not found: {custom_nodes_root}")
        sys.exit(1)

    model_refs = scan_all_workflows(custom_nodes_root)
    report = generate_report(model_refs)
    print(report)

    output_path = Path(__file__).parent / "model_references.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_refs, f, indent=2)
    print(f"\nWrote {output_path}")
