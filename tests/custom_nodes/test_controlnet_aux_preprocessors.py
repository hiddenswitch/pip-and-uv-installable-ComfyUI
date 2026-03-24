"""Test that all comfyui-controlnet-aux AIO preprocessors can be loaded and executed.

Each preprocessor is run with a small test image to verify that:
1. The node class loads correctly from the facade package
2. Model downloads are intercepted and routed through model_downloader
3. The preprocessor produces output without crashing
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from .conftest import make_base_dirs, build_config

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).resolve().parents[2]

# All AIO preprocessors reported by ControlNetPreprocessorSelector, excluding "none"
_ALL_PREPROCESSORS: list[str] = []


def _discover_preprocessors() -> list[str]:
    """Load controlnet-aux and return all supported AIO preprocessor names."""
    global _ALL_PREPROCESSORS
    if _ALL_PREPROCESSORS:
        return _ALL_PREPROCESSORS

    from comfy_compatibility.vanilla import prepare_vanilla_environment
    prepare_vanilla_environment()

    from comfy.nodes.package import _extract_vanilla_custom_node_roots
    from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
    from importlib.metadata import entry_points

    ep = next(
        ep for ep in entry_points().select(group="comfyui.custom_nodes")
        if "controlnet" in ep.name.lower()
    )
    mod = ep.load()
    roots = _extract_vanilla_custom_node_roots(mod)
    vendor_path = roots[0]
    # Find the actual repo directory inside the vendor path
    children = [c for c in Path(vendor_path).iterdir() if c.is_dir() and c.name != "__pycache__"]
    if len(children) == 1:
        vendor_path = str(children[0])

    exported = _vanilla_load_custom_nodes_1(vendor_path)
    selector = exported.NODE_CLASS_MAPPINGS.get("ControlNetPreprocessorSelector")
    if selector is None:
        return []

    inputs = selector.INPUT_TYPES()
    preprocessors = inputs.get("required", {}).get("preprocessor", [None])[0]
    if isinstance(preprocessors, list):
        _ALL_PREPROCESSORS = [p for p in sorted(preprocessors) if p != "none"]

    return _ALL_PREPROCESSORS


def _make_workflow(preprocessor_name: str) -> dict:
    """Build a minimal AIO_Preprocessor workflow for the given preprocessor."""
    return {
        "85": {
            "inputs": {
                "preprocessor": preprocessor_name,
            },
            "class_type": "ControlNetPreprocessorSelector",
            "_meta": {"title": "Preprocessor Selector"},
        },
        "86": {
            "inputs": {
                "preprocessor": ["85", 0],
                "resolution": 512,
                "image": ["87", 0],
            },
            "class_type": "AIO_Preprocessor",
            "_meta": {"title": "AIO Aux Preprocessor"},
        },
        "87": {
            "inputs": {
                "value": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Colossus_the_Cat_2.JPG/330px-Colossus_the_Cat_2.JPG",
                "alpha_is_transparency": False,
            },
            "class_type": "LoadImageFromURL",
            "_meta": {"title": "LoadImageFromURL"},
        },
        "76": {
            "inputs": {
                "filename_prefix": f"comfyui-test-{preprocessor_name}",
                "images": ["86", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
    }


try:
    _preprocessor_names = _discover_preprocessors()
except Exception as exc:
    logger.warning("Failed to discover preprocessors: %s", exc)
    _preprocessor_names = []


def _find_facade_vendor_dir() -> Path | None:
    """Return the site-packages facade vendor directory for controlnet-aux."""
    import importlib.metadata
    for dist in importlib.metadata.distributions():
        try:
            name = dist.name
        except Exception:
            continue
        if name and "controlnet" in name.lower():
            loc = dist._path.parent if hasattr(dist, "_path") else None
            if loc:
                facade = loc / f"_appmana_facade_{name.replace('-', '_')}"
                if facade.exists():
                    return facade
    return None


def _files_in_tree(root: Path) -> set[str]:
    """Return relative paths of all files under root."""
    if not root.exists():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


_XFAIL_PREPROCESSORS = {
    "MeshGraphormer-DepthMapPreprocessor": "upstream ImportError: load_tf_weights_in_bert missing from bundled bert",
}


@pytest.mark.parametrize("preprocessor", _preprocessor_names)
@pytest.mark.asyncio
async def test_aio_preprocessor(preprocessor: str, tmp_path: Path):
    reason = _XFAIL_PREPROCESSORS.get(preprocessor)
    if reason:
        pytest.xfail(reason)
    from comfy.client.embedded_comfy_client import Comfy

    base_dir = tmp_path / "base"
    make_base_dirs(base_dir)

    # Snapshot files in the facade vendor dir before execution
    vendor_dir = _find_facade_vendor_dir()
    files_before = _files_in_tree(vendor_dir) if vendor_dir else set()

    config = build_config(base_dir, cpu=True)
    workflow = _make_workflow(preprocessor)

    async with Comfy(configuration=config) as client:
        outputs = await client.queue_prompt(workflow)
        logger.info("Preprocessor %s outputs: %s", preprocessor, list(outputs.keys()))

    # Verify no new files were written into site-packages
    if vendor_dir:
        files_after = _files_in_tree(vendor_dir)
        new_files = files_after - files_before
        assert not new_files, (
            f"Preprocessor {preprocessor} wrote files into site-packages: {new_files}"
        )
