from __future__ import annotations

import io
import types
import zipfile
from pathlib import Path

from comfy.custom_node_facade.builder import FacadeWheelBuilder
from comfy.custom_node_facade.registry import FacadeProject, FacadeVersion
from comfy.nodes.package import _extract_vanilla_custom_node_roots


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return buffer.getvalue()


def test_extract_vanilla_custom_node_roots(tmp_path: Path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_c = tmp_path / "c"
    root_a.mkdir()
    root_b.mkdir()
    root_c.mkdir()

    module = types.ModuleType("facade_marker")
    module.COMFYUI_VANILLA_NODE_PATH = str(root_a)
    module.COMFYUI_VANILLA_NODE_PATHS = [str(root_a), str(root_b)]
    module.get_vanilla_custom_node_paths = lambda: [str(root_c)]

    assert _extract_vanilla_custom_node_roots(module) == [
        str(root_a.resolve()),
        str(root_b.resolve()),
        str(root_c.resolve()),
    ]


def test_build_wheel_from_archive_injects_entrypoint_and_metadata(tmp_path: Path):
    project = FacadeProject(
        canonical_name="comfyui-wanvideowrapper",
        display_name="ComfyUI-WanVideoWrapper",
        node_id="comfyui-wanvideowrapper",
        repo_url="https://github.com/kijai/ComfyUI-WanVideoWrapper",
        repo_name="ComfyUI-WanVideoWrapper",
        description="Wrapper nodes",
        aliases=("comfyui-wanvideowrapper",),
        extra_requirements=("mediapipe<=0.10.21",),
        skip_requirements=frozenset({"torch", "comfyui"}),
        depends_on=("ComfyUI-KJNodes",),
        latest_version="3.3.3",
    )
    version = FacadeVersion(
        version="3.3.3",
        download_url="https://example.invalid/node.zip",
        dependencies=("requests>=2", "torch"),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes(
        {
            "ComfyUI-WanVideoWrapper/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n",
            "ComfyUI-WanVideoWrapper/helper.py": b"value = 1\n",
        }
    )

    builder = FacadeWheelBuilder(session=None, registry=None, cache_prefix=str(tmp_path))  # type: ignore[arg-type]
    wheel_path = str(tmp_path / "comfyui-wanvideowrapper" / "comfyui_wanvideowrapper-3.3.3-py3-none-any.whl")

    built_path = builder._build_wheel_from_archive(
        project,
        version,
        archive_bytes,
        wheel_path,
        ["comfyui-kjnodes"],
    )

    assert built_path.cache_path == wheel_path
    assert built_path.local_path == wheel_path
    assert Path(wheel_path).exists()

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        assert "_appmana_facade_comfyui_wanvideowrapper/entrypoint.py" in names
        assert "_appmana_facade_comfyui_wanvideowrapper/_vendor/ComfyUI-WanVideoWrapper/__init__.py" in names

        metadata_name = "comfyui_wanvideowrapper-3.3.3.dist-info/METADATA"
        entry_points_name = "comfyui_wanvideowrapper-3.3.3.dist-info/entry_points.txt"

        metadata = wheel.read(metadata_name).decode("utf-8")
        assert "Name: comfyui-wanvideowrapper" in metadata
        assert "Version: 3.3.3" in metadata
        assert "Requires-Dist: requests>=2" in metadata
        assert "Requires-Dist: mediapipe<=0.10.21" in metadata
        assert "Requires-Dist: comfyui-kjnodes" in metadata
        assert "Requires-Dist: torch" not in metadata

        entry_points = wheel.read(entry_points_name).decode("utf-8")
        assert "[comfyui.custom_nodes]" in entry_points
        assert "comfyui-wanvideowrapper = _appmana_facade_comfyui_wanvideowrapper.entrypoint" in entry_points

        entrypoint_module = wheel.read("_appmana_facade_comfyui_wanvideowrapper/entrypoint.py").decode("utf-8")
        assert "COMFYUI_VANILLA_NODE_PATHS" in entrypoint_module


def test_build_wheel_to_memory_fs_prefix(tmp_path: Path):
    project = FacadeProject(
        canonical_name="comfyui-custom-scripts",
        display_name="ComfyUI-Custom-Scripts",
        node_id="comfyui-custom-scripts",
        repo_url="https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        repo_name="ComfyUI-Custom-Scripts",
        description="Scripts",
        aliases=("comfyui-custom-scripts",),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.2.5",
    )
    version = FacadeVersion(
        version="1.2.5",
        download_url="https://example.invalid/node.zip",
        dependencies=(),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes(
        {"ComfyUI-Custom-Scripts/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"}
    )

    builder = FacadeWheelBuilder(session=None, registry=None, cache_prefix="memory://facade-cache")  # type: ignore[arg-type]
    wheel = builder._build_wheel_from_archive(
        project,
        version,
        archive_bytes,
        "facade-cache/comfyui-custom-scripts/comfyui_custom_scripts-1.2.5-py3-none-any.whl",
        [],
    )

    assert wheel.local_path is None
    payload = builder._cache.read_bytes(wheel.cache_path)  # type: ignore[attr-defined]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert "_appmana_facade_comfyui_custom_scripts/entrypoint.py" in archive.namelist()
