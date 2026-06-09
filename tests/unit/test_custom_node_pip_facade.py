from __future__ import annotations

import io
import os
import types
import zipfile
from pathlib import Path

from comfy.custom_node_facade import flash_attention_wheels
from comfy.custom_node_facade.builder import FacadeWheelBuilder
from comfy.custom_node_facade.builder import FlashAttentionProxySpec
from comfy.custom_node_facade.builder import PYPI_PROXY_INDEX
from comfy.custom_node_facade.registry import FacadeProject, FacadeVersion
from comfy.cmd.node_info import node_info
from comfy.nodes.package import _extract_vanilla_custom_node_roots
from comfy.nodes.vanilla_node_importing import _stamp_relative_python_modules


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return buffer.getvalue()


from comfy.custom_node_facade.builder import _strip_url_dependency


class _NoNetworkSession:
    def get(self, url: str):
        raise AssertionError(f"Flash Attention proxy should not fetch at request time: {url}")


_FLASH_ATTENTION_WHEEL_URLS = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.8.2/flash_attn-2.8.3+cu126torch2.8-cp39-abi3-linux_x86_64.whl",
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.8.2/flash_attn-2.8.3+cu128torch2.8-cp39-abi3-linux_x86_64.whl",
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.8.2/flash_attn_3-3.0.0+cu128torch2.8-cp39-abi3-linux_x86_64.whl",
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.29/flash_attn_3-3.0.0+cu124torch2.5-cp39-abi3-linux_x86_64.whl",
)


def test_strip_url_dependency_rewrites_sam2():
    assert _strip_url_dependency("sam2 @ git+https://github.com/facebookresearch/sam2") == "sam2"


def test_strip_url_dependency_preserves_plain_requirement():
    assert _strip_url_dependency("requests>=2.28") == "requests>=2.28"


def test_strip_url_dependency_preserves_extras_and_markers():
    result = _strip_url_dependency("pkg[extra1] @ https://example.com/pkg.tar.gz ; python_version >= '3.10'")
    assert result == 'pkg[extra1]; python_version >= "3.10"'


async def test_flash_attention_proxy_filters_packages_by_cuda(monkeypatch):
    monkeypatch.setattr(flash_attention_wheels, "FLASH_ATTENTION_WHEEL_URLS", _FLASH_ATTENTION_WHEEL_URLS)
    session = _NoNetworkSession()
    proxy = FlashAttentionProxySpec(name="flash-attn", wheel_project_prefix="flash_attn")

    html = await proxy.render_index(session, "cu128")  # type: ignore[arg-type]

    assert "flash_attn-2.8.3+cu128torch2.8" in html
    assert "flash_attn-2.8.3+cu126torch2.8" not in html
    assert "flash_attn_3-3.0.0+cu128torch2.8" not in html
    assert "github.com/mjun0812/flash-attention-prebuild-wheels/releases/download" in html


async def test_flash_attention_3_proxy_filters_packages_by_distribution(monkeypatch):
    monkeypatch.setattr(flash_attention_wheels, "FLASH_ATTENTION_WHEEL_URLS", _FLASH_ATTENTION_WHEEL_URLS)
    session = _NoNetworkSession()
    proxy = FlashAttentionProxySpec(
        name="flash-attn-3",
        wheel_project_prefix="flash_attn_3",
        cuda_variants=("cu124", "cu126", "cu128", "cu129", "cu130", "cu132"),
    )

    html = await proxy.render_index(session, "cu128")  # type: ignore[arg-type]

    assert "flash_attn_3-3.0.0+cu128torch2.8" in html
    assert "flash_attn-2.8.3+cu128torch2.8" not in html
    assert "flash_attn_3-3.0.0+cu124torch2.5" not in html


async def test_flash_attention_3_proxy_rejects_unsupported_cuda(monkeypatch):
    monkeypatch.setattr(flash_attention_wheels, "FLASH_ATTENTION_WHEEL_URLS", _FLASH_ATTENTION_WHEEL_URLS)
    session = _NoNetworkSession()
    proxy = FlashAttentionProxySpec(
        name="flash-attn-3",
        wheel_project_prefix="flash_attn_3",
        cuda_variants=("cu124", "cu126", "cu128", "cu129", "cu130", "cu132"),
    )

    html = await proxy.render_index(session, "cu121")  # type: ignore[arg-type]

    assert ".whl" not in html


def test_cuda_specific_proxy_support_matrix():
    expected = {
        "flash-attn": {"cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu131", "cu132"},
        "flash-attn-3": {"cu124", "cu126", "cu128", "cu129", "cu130", "cu132"},
        "sageattention": {"cu128", "cu130"},
        "nunchaku": {"cu128", "cu130"},
    }
    checked_cuda_variants = {"cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu131", "cu132"}

    for project_name, supported_cuda_variants in expected.items():
        proxy = PYPI_PROXY_INDEX[project_name]
        assert {
            cuda
            for cuda in checked_cuda_variants
            if proxy.supports_cuda(cuda)
        } == supported_cuda_variants

    assert all(PYPI_PROXY_INDEX["insightface"].supports_cuda(cuda) for cuda in checked_cuda_variants)


def test_url_dependency_stripped_from_wheel_metadata(tmp_path: Path):
    project = FacadeProject(
        canonical_name="comfyui-impact-pack",
        display_name="ComfyUI-Impact-Pack",
        node_id="comfyui-impact-pack",
        repo_url="https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        repo_name="ComfyUI-Impact-Pack",
        description="Impact pack",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch", "comfyui"}),
        depends_on=(),
        latest_version="8.28.2",
    )
    version = FacadeVersion(
        version="8.28.2",
        download_url="https://example.invalid/node.zip",
        dependencies=(
            "segment-anything>=1.0",
            "sam2 @ git+https://github.com/facebookresearch/sam2",
            "ultralytics>=8.0",
        ),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes(
        {"ComfyUI-Impact-Pack/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"}
    )

    builder = FacadeWheelBuilder(session=None, registry=None, cache_prefix=str(tmp_path))  # type: ignore[arg-type]
    wheel_path = str(tmp_path / "comfyui-impact-pack" / "comfyui_impact_pack-8.28.2-py3-none-any.whl")

    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("comfyui_impact_pack-8.28.2.dist-info/METADATA").decode("utf-8")
        assert "Requires-Dist: sam2" in metadata
        assert "git+https" not in metadata
        assert "Requires-Dist: segment-anything>=1.0" in metadata
        assert "Requires-Dist: ultralytics>=8.0" in metadata


def test_version_constraints_stripped_from_pinned_dependencies(tmp_path: Path):
    project = FacadeProject(
        canonical_name="test-node",
        display_name="Test",
        node_id="test-node",
        repo_url="https://github.com/test/test",
        repo_name="test",
        description="",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.0.0",
    )
    version = FacadeVersion(
        version="1.0.0",
        download_url="https://example.invalid/node.zip",
        dependencies=(
            "numpy<2",
            "jax>=0.4,<0.8",
            "jaxlib>=0.4,<0.8",
            "opencv-python-headless==4.7.0.72",
            "scipy>=1.0",
        ),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes({"test/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"})
    builder = FacadeWheelBuilder(session=None, registry=None, cache_prefix=str(tmp_path))  # type: ignore[arg-type]
    wheel_path = str(tmp_path / "test-node" / "test_node-1.0.0-py3-none-any.whl")
    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("test_node-1.0.0.dist-info/METADATA").decode("utf-8")
        assert "Requires-Dist: numpy\n" in metadata
        assert "numpy<" not in metadata
        assert "Requires-Dist: jax\n" in metadata
        assert "jax>" not in metadata
        assert "Requires-Dist: jaxlib\n" in metadata
        assert "Requires-Dist: opencv-contrib-python-headless\n" in metadata
        assert "opencv-python-headless" not in metadata
        assert "Requires-Dist: scipy>=1.0" in metadata


def test_onnxruntime_expanded_to_platform_variants(tmp_path: Path):
    project = FacadeProject(
        canonical_name="test-onnx-node",
        display_name="Test",
        node_id="test-onnx-node",
        repo_url="https://github.com/test/test",
        repo_name="test",
        description="",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.0.0",
    )
    version = FacadeVersion(
        version="1.0.0",
        download_url="https://example.invalid/node.zip",
        dependencies=("onnxruntime", "pillow"),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes({"test/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"})
    builder = FacadeWheelBuilder(session=None, registry=None, cache_prefix=str(tmp_path))  # type: ignore[arg-type]
    wheel_path = str(tmp_path / "test-onnx-node" / "test_onnx_node-1.0.0-py3-none-any.whl")
    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("test_onnx_node-1.0.0.dist-info/METADATA").decode("utf-8")
        assert 'sys_platform == "darwin"' in metadata
        assert "onnxruntime-gpu" in metadata
        assert 'sys_platform == "win32"' in metadata
        assert "Requires-Dist: pillow" in metadata
        bare = [l for l in metadata.splitlines() if l == "Requires-Dist: onnxruntime"]
        assert bare == [], f"bare onnxruntime without marker: {bare}"


def test_pynvml_rewritten_to_nvidia_ml_py(tmp_path: Path):
    """Any custom node that lists pynvml as a requirement should have its
    facade wheel METADATA rewritten to nvidia-ml-py. The two packages ship
    an identical ``pynvml.py`` module, but pynvml adds a .pth hook that
    warns on every import — silencing that is the whole point.
    """
    project = FacadeProject(
        canonical_name="test-pynvml-node",
        display_name="Test",
        node_id="test-pynvml-node",
        repo_url="https://github.com/test/test",
        repo_name="test",
        description="",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.0.0",
    )
    version = FacadeVersion(
        version="1.0.0",
        download_url="https://example.invalid/node.zip",
        dependencies=("pynvml", "pillow"),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes({"test/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"})
    builder = FacadeWheelBuilder(session=None, registry=None, cache_prefix=str(tmp_path))  # type: ignore[arg-type]
    wheel_path = str(tmp_path / "test-pynvml-node" / "test_pynvml_node-1.0.0-py3-none-any.whl")
    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("test_pynvml_node-1.0.0.dist-info/METADATA").decode("utf-8")
        assert "Requires-Dist: nvidia-ml-py" in metadata
        assert "Requires-Dist: pillow" in metadata
        bare = [l for l in metadata.splitlines() if l == "Requires-Dist: pynvml"]
        assert bare == [], f"pynvml should have been rewritten, got: {bare}"


def test_injected_project_has_github_archive_version():
    from comfy.custom_node_facade.registry import FacadeRegistry
    from comfy.component_model.node_registry import CustomNodeSpec

    spec = CustomNodeSpec(
        node_id="test-injected-node",
        repo_url="https://github.com/TestOrg/TestRepo",
        display_name="Test Injected Node",
        inject_version="0.1.0",
        git_ref="main",
    )
    registry = FacadeRegistry.__new__(FacadeRegistry)
    registry._versions_cache = {}
    registry._overlay_index = None

    project = registry._build_injected_project(spec)

    assert project.canonical_name == "test-injected-node"
    assert project.latest_version == "0.1.0"
    assert project.repo_url == "https://github.com/TestOrg/TestRepo"

    versions = registry._versions_cache["test-injected-node"]
    assert len(versions) == 1
    assert versions[0].version == "0.1.0"
    assert versions[0].download_url == "https://api.github.com/repos/TestOrg/TestRepo/zipball/main"


def test_github_archive_url_uses_zipball_for_default_branch():
    from comfy.custom_node_facade.registry import FacadeRegistry

    # No ref: uses zipball endpoint that resolves to the repo's default branch
    url = FacadeRegistry._github_archive_url("https://github.com/Lightricks/ComfyUI-LTXVideo")
    assert url == "https://api.github.com/repos/Lightricks/ComfyUI-LTXVideo/zipball"

    # Explicit ref: uses zipball with that ref
    url = FacadeRegistry._github_archive_url("https://github.com/Lightricks/ComfyUI-LTXVideo", "master")
    assert url == "https://api.github.com/repos/Lightricks/ComfyUI-LTXVideo/zipball/master"

    # Strips trailing .git
    url = FacadeRegistry._github_archive_url("https://github.com/Lightricks/ComfyUI-LTXVideo.git")
    assert url == "https://api.github.com/repos/Lightricks/ComfyUI-LTXVideo/zipball"


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
        dependencies=("requests>=2", "torch", "numpy<2", "opencv-python-headless==4.7.0.72"),
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
    assert os.path.normpath(built_path.local_path or "") == os.path.normpath(wheel_path)
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
        assert "Requires-Dist: numpy<2" not in metadata
        assert "Requires-Dist: numpy\n" in metadata
        assert "Requires-Dist: opencv-python-headless==4.7.0.72" not in metadata
        assert "Requires-Dist: opencv-contrib-python-headless\n" in metadata

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


def test_facade_cache_store_handles_windows_local_parent(tmp_path: Path):
    builder = FacadeWheelBuilder(session=None, registry=None, cache_prefix=str(tmp_path))  # type: ignore[arg-type]
    cache = builder._cache  # type: ignore[attr-defined]

    windows_style_path = str(tmp_path / "facade" / "wheel.whl").replace("/", "\\")

    assert cache._parent(windows_style_path).endswith("\\facade")


def test_stamp_relative_python_modules_uses_class_module():
    class ExampleNode:
        __module__ = "vendor.custom_scripts.nodes"
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        CATEGORY = "tests"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    _stamp_relative_python_modules({"ExampleNode": ExampleNode})

    assert getattr(ExampleNode, "RELATIVE_PYTHON_MODULE", None) == "vendor.custom_scripts.nodes"


def test_node_info_python_module_falls_back_to_class_module():
    class ExampleNode:
        __module__ = "vendor.custom_scripts.nodes"
        RELATIVE_PYTHON_MODULE = None
        ESSENTIALS_CATEGORY = "Tests"
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        CATEGORY = "tests"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    info = node_info("ExampleNode", {"ExampleNode": ExampleNode}, {})

    assert info["python_module"] == "vendor.custom_scripts.nodes"
    assert info["essentials_category"] == "Tests"
