from __future__ import annotations

from ..cmd.main_pre import tracer

import asyncio
import base64
import csv
import io
import ntpath
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import fsspec

from .registry import FacadeProject, FacadeRegistryProtocol, FacadeVersion, canonicalize_project_name

_WHEEL_NAME_RE = re.compile(r"[^A-Za-z0-9.]+")
_FACADE_BUILD_REVISION = 4
_FACADE_ALWAYS_SKIPPED_DEPENDENCIES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PyPIRewriteSpec:
    """A PyPI package to re-serve with patched dependency metadata."""
    name: str
    version: str
    wheel_url: str
    dependencies: tuple[str, ...]


PYPI_REWRITE_PACKAGES: list[PyPIRewriteSpec] = [
    PyPIRewriteSpec(
        name="image-reward",
        version="1.5",
        wheel_url="https://files.pythonhosted.org/packages/ea/df/b2e66a6f93494ac43d2cfb77475e0321e062b765ec35a368a07decf22a1d/image_reward-1.5-py3-none-any.whl",
        dependencies=(
            "timm",
            "transformers>=4.27.4",
            "fairscale",
            "huggingface-hub>=0.13.4",
            "diffusers>=0.16.0",
            "accelerate>=0.16.0",
            "datasets>=2.11.0",
        ),
    ),
]

_PYPI_REWRITE_INDEX: dict[str, PyPIRewriteSpec] = {
    canonicalize_project_name(spec.name): spec for spec in PYPI_REWRITE_PACKAGES
}


DEFAULT_CUDA_VARIANT = "cu130"
SUPPORTED_CUDA_VARIANTS = ("cu128", "cu130")


@dataclass(frozen=True)
class PyPIProxySpec:
    """A package whose simple index page is proxied from an upstream URL template.

    ``upstream_index_url_template`` contains a ``{cuda}`` placeholder that is
    replaced at request time with the selected CUDA variant (e.g. ``cu130``).
    """
    name: str
    upstream_index_url_template: str

    def upstream_index_url(self, cuda: str = DEFAULT_CUDA_VARIANT) -> str:
        return self.upstream_index_url_template.format(cuda=cuda)

PYPI_PROXY_PACKAGES: list[PyPIProxySpec] = [
    PyPIProxySpec(
        name="sageattention",
        upstream_index_url_template="https://appmana.github.io/forks-sageattention-stable-abi/{cuda}/sageattention/",
    ),
    PyPIProxySpec(
        name="nunchaku",
        upstream_index_url_template="https://appmana.github.io/forks-nunchaku-stable-abi/{cuda}/nunchaku/",
    ),
]

PYPI_PROXY_INDEX: dict[str, PyPIProxySpec] = {
    canonicalize_project_name(spec.name): spec for spec in PYPI_PROXY_PACKAGES
}

_FACADE_STRIP_VERSION_DEPENDENCIES = frozenset({
    "image-reward",
    "jax",
    "jaxlib",
    "numpy",
    "protobuf",
    "timm",
})

_OPENCV_HEADLESS = ["opencv-python-headless"]

# Dependencies that should be expanded to platform-specific variants.
# Each entry maps a canonical name to the list of requirements that replace it.
_ONNXRUNTIME_DEPS: list[str] = [
    'onnxruntime; sys_platform == "darwin"',
    'onnxruntime; sys_platform == "linux" and platform_machine == "aarch64"',
    'onnxruntime-gpu; sys_platform == "linux" and platform_machine == "x86_64"',
    'onnxruntime-gpu; sys_platform == "win32"',
]

_FACADE_EXPANDED_DEPENDENCIES: dict[str, list[str]] = {
    "onnxruntime": _ONNXRUNTIME_DEPS,
    "onnxruntime-gpu": _ONNXRUNTIME_DEPS,
    "onnxruntime-directml": _ONNXRUNTIME_DEPS,
    "onnxruntime-rocm": _ONNXRUNTIME_DEPS,
    "onnxruntime-openvino": _ONNXRUNTIME_DEPS,
    "onnxruntime-silicon": _ONNXRUNTIME_DEPS,
    "opencv-python": _OPENCV_HEADLESS,
    "opencv-python-headless": _OPENCV_HEADLESS,
    "opencv-contrib-python": _OPENCV_HEADLESS,
    "opencv-contrib-python-headless": _OPENCV_HEADLESS,
    # The pynvml package is a deprecation shim: it installs a .pth file that
    # emits a FutureWarning on every ``import pynvml`` (even indirect ones
    # from torch). nvidia-ml-py ships an identical ``pynvml.py`` module
    # under a non-deprecated package, so rewriting the dependency here is
    # transparent to any node code that does ``import pynvml``.
    "pynvml": ["nvidia-ml-py"],
}


def _wheel_distribution_name(name: str) -> str:
    return _WHEEL_NAME_RE.sub("_", name)


def _stub_package_name(name: str) -> str:
    identifier = _WHEEL_NAME_RE.sub("_", name).strip("_").lower()
    return f"_appmana_facade_{identifier}"


def _build_record_from_tree(tree: Path) -> list[tuple[str, str, str]]:
    if os.name != "nt" and shutil.which("find") is not None:
        return _build_record_from_tree_unix(tree)
    return _build_record_from_tree_python(tree)


def _build_record_from_tree_unix(tree: Path) -> list[tuple[str, str, str]]:
    result = subprocess.run(
        ["find", ".", "-type", "f", "-print0"],
        cwd=str(tree),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"find failed: {result.stderr.decode(errors='replace')}")
    files = sorted(f for f in result.stdout.decode().split("\0") if f)

    result = subprocess.run(
        ["xargs", "-0", "sha256sum"],
        cwd=str(tree),
        input=result.stdout,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sha256sum failed: {result.stderr.decode(errors='replace')}")

    hash_map: dict[str, str] = {}
    for line in result.stdout.decode().splitlines():
        hex_digest, path = line.split(maxsplit=1)
        if path.startswith("./"):
            path = path[2:]
        raw = bytes.fromhex(hex_digest)
        hash_map[path] = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    records: list[tuple[str, str, str]] = []
    for f in files:
        rel = f[2:] if f.startswith("./") else f
        full = tree / rel
        size = full.stat().st_size
        digest = hash_map.get(rel, "")
        records.append((rel, f"sha256={digest}", str(size)))
    return records


def _build_record_from_tree_python(tree: Path) -> list[tuple[str, str, str]]:
    import hashlib
    records: list[tuple[str, str, str]] = []
    for full in sorted(tree.rglob("*")):
        if not full.is_file():
            continue
        rel = full.relative_to(tree).as_posix()
        size = full.stat().st_size
        raw = hashlib.sha256(full.read_bytes()).digest()
        digest = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        records.append((rel, f"sha256={digest}", str(size)))
    return records


def _strip_url_dependency(requirement: str) -> str:
    """Strip the URL from a PEP 440 URL dependency, keeping name/extras/specifier/marker."""
    from packaging.requirements import Requirement, InvalidRequirement
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return requirement
    if parsed.url is None:
        return requirement
    parts = [parsed.name]
    if parsed.extras:
        parts.append("[" + ",".join(sorted(parsed.extras)) + "]")
    if parsed.specifier:
        parts.append(str(parsed.specifier))
    if parsed.marker:
        parts.append("; " + str(parsed.marker))
    return "".join(parts)


def _parse_requirement_name(requirement: str) -> str:
    token = requirement.split(";", 1)[0].strip()
    if " @ " in token:
        token = token.split(" @ ", 1)[0].strip()
    for separator in ("==", ">=", "<=", "!=", "~=", ">", "<", "[", " "):
        token = token.split(separator, 1)[0]
    return canonicalize_project_name(token)


def _render_metadata(
    project: FacadeProject,
    version: FacadeVersion,
    dependencies: list[str],
) -> bytes:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {project.canonical_name}",
        f"Version: {version.version}",
        f"Summary: {project.description or project.display_name}",
        f"Home-page: {project.repo_url}",
        "Requires-Python: >=3.10",
    ]
    for dependency in dependencies:
        lines.append(f"Requires-Dist: {dependency}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _render_entrypoint_module(project: FacadeProject) -> bytes:
    return """from __future__ import annotations

from pathlib import Path

COMFYUI_VANILLA_NODE_PATH = str(Path(__file__).resolve().parent / "_vendor")
COMFYUI_VANILLA_NODE_PATHS = [COMFYUI_VANILLA_NODE_PATH]


def get_vanilla_custom_node_paths() -> list[str]:
    return list(COMFYUI_VANILLA_NODE_PATHS)
""".encode("utf-8")


def _render_entry_points(dist_name: str, module_name: str) -> bytes:
    return f"""[comfyui.custom_nodes]
{dist_name} = {module_name}.entrypoint
""".encode("utf-8")


def _render_wheel() -> bytes:
    return b"Wheel-Version: 1.0\nGenerator: appmana-comfyui serve-pip\nRoot-Is-Purelib: true\nTag: py3-none-any\n"


def _read_requirements_file(repo_root: Path) -> list[str]:
    requirements_path = repo_root / "requirements.txt"
    if not requirements_path.exists():
        return []

    requirements: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-r"):
            continue
        requirements.append(stripped)
    return requirements


@dataclass(frozen=True)
class CachedWheel:
    cache_path: str
    local_path: str | None = None


class FacadeCacheStore:
    def __init__(self, prefix: str | os.PathLike[str]) -> None:
        self._prefix = str(prefix)
        self._fs, self._root = fsspec.core.url_to_fs(self._prefix)

    def wheel_path(self, project: FacadeProject, filename: str, revision: int = _FACADE_BUILD_REVISION) -> str:
        return self._join(f"v{revision}", canonicalize_project_name(project.canonical_name), filename)

    def exists(self, path: str) -> bool:
        return bool(self._fs.exists(path))

    def write_bytes(self, path: str, data: bytes) -> None:
        parent = self._parent(path)
        if parent:
            self._fs.makedirs(parent, exist_ok=True)
        with self._fs.open(path, "wb") as handle:
            handle.write(data)

    def copy_from(self, source_path: str, dest_path: str) -> None:
        parent = self._parent(dest_path)
        if parent:
            self._fs.makedirs(parent, exist_ok=True)
        local = self._local_path(dest_path)
        if local is not None:
            shutil.copy2(source_path, local)
        else:
            with open(source_path, "rb") as src, self._fs.open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    def read_bytes(self, path: str) -> bytes:
        with self._fs.open(path, "rb") as handle:
            return handle.read()

    def cached_wheel(self, path: str) -> CachedWheel:
        return CachedWheel(cache_path=path, local_path=self._local_path(path))

    def _join(self, *parts: str) -> str:
        clean_parts = [part.strip("/") for part in parts if part]
        if not self._root:
            return posixpath.join(*clean_parts)
        root = self._root.rstrip("/")
        if not clean_parts:
            return root
        return posixpath.join(root, *clean_parts)

    def _parent(self, path: str) -> str:
        if isinstance(self._fs, fsspec.implementations.local.LocalFileSystem):
            return ntpath.dirname(path)
        return posixpath.dirname(path)

    def _local_path(self, path: str) -> str | None:
        try:
            return os.fspath(fsspec.open_local(path, mode="rb"))
        except (AttributeError, FileNotFoundError, OSError, ValueError):
            return None


class FacadeWheelBuilder:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        registry: FacadeRegistryProtocol,
        *,
        cache_prefix: str | os.PathLike[str],
        cache_revision: int | None = None,
    ) -> None:
        self._session = session
        self._registry = registry
        self._cache = FacadeCacheStore(cache_prefix)
        self._cache_revision = cache_revision or _FACADE_BUILD_REVISION
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def build_wheel(
        self,
        project: FacadeProject | str,
        version: FacadeVersion | str,
    ) -> CachedWheel:
        resolved_project = await self._registry.get_project(project) if isinstance(project, str) else project
        if resolved_project is None:
            raise KeyError(f"Unknown facade project: {project}")

        resolved_version = await self._registry.get_version(resolved_project, version) if isinstance(version, str) else version
        if resolved_version is None:
            raise KeyError(f"Unknown facade version: {resolved_project.canonical_name}@{version}")

        with tracer.start_as_current_span("Build Facade Wheel") as span:
            span.set_attribute("facade.project_name", resolved_project.canonical_name)
            span.set_attribute("facade.node_id", resolved_project.node_id)
            span.set_attribute("facade.version", resolved_version.version)
            wheel_name = self.wheel_filename(resolved_project, resolved_version.version)
            wheel_path = self._cache.wheel_path(resolved_project, wheel_name, self._cache_revision)
            span.set_attribute("facade.wheel_path", wheel_path)
            if self._cache.exists(wheel_path):
                span.set_attribute("facade.cache_hit", True)
                return self._cache.cached_wheel(wheel_path)

            span.set_attribute("facade.cache_hit", False)
            lock = self._locks.setdefault((resolved_project.canonical_name, resolved_version.version), asyncio.Lock())
            async with lock:
                if self._cache.exists(wheel_path):
                    span.set_attribute("facade.cache_hit_after_lock", True)
                    return self._cache.cached_wheel(wheel_path)
                rewrite = _PYPI_REWRITE_INDEX.get(resolved_project.canonical_name)
                if rewrite is not None:
                    return await self._build_rewrite_wheel(rewrite, resolved_project, resolved_version, wheel_path)
                with tracer.start_as_current_span("Download Facade Source Archive") as download_span:
                    download_span.set_attribute("facade.download_url", resolved_version.download_url)
                    async with self._session.get(resolved_version.download_url) as response:
                        response.raise_for_status()
                        archive_bytes = await response.read()
                    download_span.set_attribute("facade.archive_bytes", len(archive_bytes))
                dependency_package_names = [
                    await self._registry.dependency_project_name(dependency_id)
                    for dependency_id in resolved_project.depends_on
                ]
                built = await asyncio.to_thread(
                    self._build_wheel_from_archive,
                    resolved_project,
                    resolved_version,
                    archive_bytes,
                    wheel_path,
                    dependency_package_names,
                )
                return built

    def _build_wheel_from_archive(
        self,
        project: FacadeProject,
        version: FacadeVersion,
        archive_bytes: bytes,
        wheel_path: str,
        dependency_package_names: list[str],
    ) -> CachedWheel:
        with tracer.start_as_current_span("Assemble Facade Wheel") as span:
            span.set_attribute("facade.project_name", project.canonical_name)
            span.set_attribute("facade.version", version.version)
            with tempfile.TemporaryDirectory(prefix="comfyui_facade_") as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                source_root = temp_dir / "source"
                source_root.mkdir(parents=True, exist_ok=True)
                self._extract_archive(archive_bytes, version.download_url, source_root)
                repo_root = self._select_repo_root(source_root)
                span.set_attribute("facade.repo_root", str(repo_root))

                requirements = self._collect_requirements(project, version, repo_root, dependency_package_names)
                span.set_attribute("facade.requirement_count", len(requirements))
                self._write_wheel(project, version, repo_root, requirements, wheel_path)
            return self._cache.cached_wheel(wheel_path)

    def _collect_requirements(
        self,
        project: FacadeProject,
        version: FacadeVersion,
        repo_root: Path,
        dependency_package_names: list[str],
    ) -> list[str]:
        dependencies: list[str] = list(version.dependencies) or _read_requirements_file(repo_root)
        dependencies.extend(project.extra_requirements)

        dependencies.extend(dependency_package_names)

        filtered: list[str] = []
        seen: set[str] = set()
        skipped = {canonicalize_project_name(item) for item in project.skip_requirements}
        skipped.update(_FACADE_ALWAYS_SKIPPED_DEPENDENCIES)
        for dependency in dependencies:
            stripped = dependency.strip()
            if not stripped:
                continue
            stripped = _strip_url_dependency(stripped)
            dep_name = _parse_requirement_name(stripped)
            if dep_name in skipped or dep_name == canonicalize_project_name(project.canonical_name):
                continue
            if dep_name in _FACADE_EXPANDED_DEPENDENCIES:
                for expanded in _FACADE_EXPANDED_DEPENDENCIES[dep_name]:
                    if expanded not in seen:
                        seen.add(expanded)
                        filtered.append(expanded)
                continue
            if dep_name in _FACADE_STRIP_VERSION_DEPENDENCIES:
                stripped = dep_name
            if stripped in seen:
                continue
            seen.add(stripped)
            filtered.append(stripped)
        return filtered

    def _write_wheel(
        self,
        project: FacadeProject,
        version: FacadeVersion,
        repo_root: Path,
        requirements: list[str],
        wheel_path: str,
    ) -> None:
        module_name = _stub_package_name(project.canonical_name)
        dist_name = _wheel_distribution_name(project.canonical_name)
        dist_info = f"{dist_name}-{version.version}.dist-info"

        with tempfile.TemporaryDirectory(prefix="comfyui_facade_wheel_") as staging_dir:
            staging = Path(staging_dir)
            tree = staging / "tree"

            # Lay out the full wheel directory structure on disk
            (tree / module_name).mkdir(parents=True)
            (tree / module_name / "__init__.py").write_bytes(b"")
            (tree / module_name / "entrypoint.py").write_bytes(_render_entrypoint_module(project))
            (tree / dist_info).mkdir()
            (tree / dist_info / "METADATA").write_bytes(_render_metadata(project, version, requirements))
            (tree / dist_info / "WHEEL").write_bytes(_render_wheel())
            (tree / dist_info / "entry_points.txt").write_bytes(_render_entry_points(project.canonical_name, module_name))

            # Symlink vendor tree (zip follows symlinks by default on Linux)
            vendor_dir = tree / module_name / "_vendor" / project.repo_name
            vendor_dir.parent.mkdir(parents=True, exist_ok=True)
            vendor_dir.symlink_to(repo_root)

            # Build RECORD using find + sha256sum (no Python memory)
            records = _build_record_from_tree(tree)
            record_buf = io.StringIO()
            writer = csv.writer(record_buf, lineterminator="\n")
            for row in records:
                writer.writerow(row)
            writer.writerow((f"{dist_info}/RECORD", "", ""))
            (tree / dist_info / "RECORD").write_bytes(record_buf.getvalue().encode("utf-8"))

            # Create the zip with system zip (constant memory, follows symlinks)
            # Falls back to Python zipfile on Windows where zip is unavailable
            temp_wheel = staging / "wheel.whl"
            if shutil.which("zip") is not None and os.name != "nt":
                result = subprocess.run(
                    ["zip", "-q", "-r", "-0", str(temp_wheel), "."],
                    cwd=str(tree),
                    capture_output=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"zip failed: {result.stderr.decode(errors='replace')}")
            else:
                with zipfile.ZipFile(temp_wheel, "w", compression=zipfile.ZIP_STORED) as zf:
                    for root, _dirs, files in os.walk(tree, followlinks=True):
                        for fname in sorted(files):
                            full = Path(root) / fname
                            zf.write(full, full.relative_to(tree))

            self._cache.copy_from(str(temp_wheel), wheel_path)

    @staticmethod
    def wheel_filename(project: FacadeProject, version: str) -> str:
        dist_name = _wheel_distribution_name(project.canonical_name)
        return f"{dist_name}-{version}-py3-none-any.whl"

    async def _build_rewrite_wheel(
        self,
        rewrite: PyPIRewriteSpec,
        project: FacadeProject,
        version: FacadeVersion,
        wheel_path: str,
    ) -> CachedWheel:
        """Download a PyPI wheel and re-pack it with patched METADATA."""
        with tracer.start_as_current_span("Build Rewrite Wheel") as span:
            span.set_attribute("facade.rewrite_url", rewrite.wheel_url)
            async with self._session.get(rewrite.wheel_url) as response:
                response.raise_for_status()
                wheel_bytes = await response.read()
            span.set_attribute("facade.wheel_bytes", len(wheel_bytes))
            await asyncio.to_thread(
                self._patch_wheel_metadata, wheel_bytes, project, version, list(rewrite.dependencies), wheel_path,
            )
            return self._cache.cached_wheel(wheel_path)

    def _patch_wheel_metadata(
        self,
        wheel_bytes: bytes,
        project: FacadeProject,
        version: FacadeVersion,
        dependencies: list[str],
        wheel_path: str,
    ) -> None:
        """Rewrite METADATA inside a wheel zip and cache the result."""
        new_metadata = _render_metadata(project, version, dependencies)
        with tempfile.NamedTemporaryFile(prefix="comfyui_rewrite_", suffix=".whl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with zipfile.ZipFile(io.BytesIO(wheel_bytes), "r") as src, \
                 zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                metadata_suffix = ".dist-info/METADATA"
                record_suffix = ".dist-info/RECORD"
                for item in src.infolist():
                    data = src.read(item.filename)
                    if item.filename.endswith(metadata_suffix):
                        data = new_metadata
                    elif item.filename.endswith(record_suffix):
                        # RECORD will be invalid after patching; pip/uv don't
                        # validate it for remote installs, so write an empty one.
                        data = b""
                    dst.writestr(item, data)
            self._cache.copy_from(tmp_path, wheel_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def read_cached_wheel(self, wheel: CachedWheel) -> bytes:
        with tracer.start_as_current_span("Read Cached Facade Wheel") as span:
            span.set_attribute("facade.wheel_path", wheel.cache_path)
            return await asyncio.to_thread(self._cache.read_bytes, wheel.cache_path)

    @staticmethod
    def _extract_archive(archive_bytes: bytes, download_url: str, destination: Path) -> None:
        archive = io.BytesIO(archive_bytes)
        lowered = download_url.lower()
        if lowered.endswith(".zip") or archive_bytes[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(archive) as zip_file:
                zip_file.extractall(destination)
            return
        if lowered.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(fileobj=archive, mode="r:*") as tar_file:
                tar_file.extractall(destination)
            return
        raise ValueError(f"Unsupported archive type for facade package: {download_url}")

    @staticmethod
    def _select_repo_root(source_root: Path) -> Path:
        children = [child for child in source_root.iterdir() if child.name not in ("__MACOSX",)]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return source_root
