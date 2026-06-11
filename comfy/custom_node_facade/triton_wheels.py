"""Serve ``triton`` and ``triton-windows`` through the pip facade.

Behaviour:

* ``triton`` on Linux is a straight redirect to the PyPI ``triton`` manylinux
  wheels.
* ``triton`` on Windows, and ``triton-windows`` everywhere, serve the
  woct0rdho ``triton-windows`` ``win_amd64`` wheels (published to PyPI). For the
  ``triton`` project those wheels are renamed/rewritten so ``pip install triton``
  resolves to them on Windows.
* For CUDA 13 variants (``/simple/cu130/...`` etc.) the bundled CUDA 12.x
  compiler toolchain inside the ``triton-windows`` wheel
  (``triton/backends/nvidia/bin/ptxas.exe`` + ``include/cuda.h`` +
  ``lib/x64/cuda.lib``) is swapped for the matching CUDA 13.x redistributable
  binaries. woct0rdho's wheels bundle 12.x regardless of target, which breaks
  ``ptxas`` for sm_120+/CUDA-13 toolchains; this patches that on the fly.

The patched/renamed wheel is built on demand by :class:`TritonWheelBuilder` and
served from ``/packages/triton/{cuda}/{filename}``.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import html
import io
import re
import zipfile

import aiohttp

PYPI_SIMPLE = "https://pypi.org/simple"
PYPI_JSON = "https://pypi.org/pypi"
NVIDIA_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist"

# Files inside the wheel that carry the bundled CUDA compiler toolchain.
_NVIDIA_PTXAS = "triton/backends/nvidia/bin/ptxas.exe"
_NVIDIA_CUDA_H = "triton/backends/nvidia/include/cuda.h"
_NVIDIA_CUDA_LIB = "triton/backends/nvidia/lib/x64/cuda.lib"

# Where each patched file is sourced from inside the NVIDIA redist archives.
#   wheel member -> (redist package, member path inside the archive's top dir)
_CUDA_PATCH_SOURCES: dict[str, tuple[str, str]] = {
    _NVIDIA_PTXAS: ("cuda_nvcc", "bin/ptxas.exe"),
    _NVIDIA_CUDA_H: ("cuda_cudart", "include/cuda.h"),
    _NVIDIA_CUDA_LIB: ("cuda_cudart", "lib/x64/cuda.lib"),
}

_CU_RE = re.compile(r"^cu(\d+)(?:torch[0-9.]+)?$")


def cuda_major(cuda: str) -> int | None:
    """Return the CUDA major version for a ``cuXYZ`` variant token (13 for cu130)."""
    match = _CU_RE.match(cuda)
    if match is None:
        return None
    digits = match.group(1)
    # cu130 -> 13, cu128 -> 12, cu90 -> 9
    return int(digits[:-1]) if len(digits) >= 2 else int(digits)


def cuda_redist_label(cuda: str) -> str | None:
    """Map a CUDA-13 variant token to its redist minor series, e.g. cu131 -> '13.1'."""
    match = _CU_RE.match(cuda)
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) < 2:
        return None
    return f"{digits[:-1]}.{digits[-1]}"


def needs_cuda13_patch(cuda: str) -> bool:
    return cuda_major(cuda) == 13


@dataclasses.dataclass(frozen=True)
class TritonProxySpec:
    """Proxy ``triton`` / ``triton-windows`` PyPI pages, routing Windows wheels
    through our patched-wheel endpoint."""
    name: str
    rename_to_triton: bool

    def supports_cuda(self, cuda: str) -> bool:
        # Plain CUDA variants only (no cuXXXtorchY.Z): triton wheels are not
        # torch-ABI tagged.
        return _CU_RE.match(cuda) is not None and "torch" not in cuda

    async def render_index(self, session: aiohttp.ClientSession, cuda: str) -> str:
        entries: list[tuple[str, str]] = []

        # Linux/manylinux part: only the `triton` project serves the PyPI
        # `triton` wheels directly (straight redirect).
        if self.rename_to_triton:
            for filename, url in await _pypi_wheels(session, "triton"):
                if "win_amd64" in filename:
                    continue
                entries.append((filename, url))

        # Windows part: triton-windows wheels routed through our patcher so
        # CUDA-13 variants get the swapped toolchain (and `triton` gets renamed).
        for filename, _ in await _pypi_wheels(session, "triton-windows"):
            if "win_amd64" not in filename:
                continue
            served = _rename_filename(filename, "triton") if self.rename_to_triton else filename
            entries.append((served, f"/packages/triton/{cuda}/{served}"))

        body = "\n".join(
            f'<a href="{html.escape(url, quote=True)}">{html.escape(name)}</a><br/>'
            for name, url in sorted(set(entries))
        )
        title = f"Simple Index for {self.name}"
        return f"<!DOCTYPE html><html><head><title>{html.escape(title)}</title></head><body>{body}</body></html>"


async def _pypi_wheels(session: aiohttp.ClientSession, project: str) -> list[tuple[str, str]]:
    async with session.get(f"{PYPI_SIMPLE}/{project}/") as resp:
        resp.raise_for_status()
        text = await resp.text()
    out: list[tuple[str, str]] = []
    for href in re.findall(r'<a[^>]+href="([^"]+)"', text):
        url = html.unescape(href)
        filename = url.rsplit("/", 1)[-1].split("#", 1)[0]
        if filename.endswith(".whl"):
            out.append((filename, url))
    return out


def _rename_filename(filename: str, project: str) -> str:
    # triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl -> triton-3.7.0.post26-...
    parts = filename.split("-")
    parts[0] = project
    return "-".join(parts)


def _record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"{path},sha256={digest},{len(data)}"


class TritonWheelBuilder:
    """Downloads a ``triton-windows`` wheel, optionally swaps its bundled CUDA
    toolchain for CUDA 13.x, optionally renames it to ``triton``, and returns the
    rebuilt wheel bytes. Network fetches are injected for testability."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def fetch_url(self, url: str) -> bytes:
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def source_wheel_url(self, served_filename: str) -> str:
        """Resolve the served (possibly renamed) filename back to the upstream
        ``triton-windows`` wheel URL on PyPI."""
        original = _rename_filename(served_filename, "triton_windows") if served_filename.startswith("triton-") else served_filename
        async with self._session.get(f"{PYPI_JSON}/triton-windows/json") as resp:
            resp.raise_for_status()
            data = await resp.json()
        for release in data.get("releases", {}).values():
            for file in release:
                if file.get("filename") == original:
                    return file["url"]
        raise KeyError(f"no upstream triton-windows wheel for {served_filename!r}")

    async def cuda13_binaries(self, cuda: str) -> dict[str, bytes]:
        """Fetch the CUDA 13.x redist binaries to patch into the wheel, keyed by
        their destination path inside the wheel."""
        label = cuda_redist_label(cuda)
        if label is None:
            raise ValueError(f"not a CUDA-13 variant: {cuda}")
        index_url = await self._resolve_redist_index(label)
        index = await self._fetch_json(index_url)
        archives: dict[str, bytes] = {}
        out: dict[str, bytes] = {}
        for dest, (pkg, member) in _CUDA_PATCH_SOURCES.items():
            rel = index[pkg]["windows-x86_64"]["relative_path"]
            if rel not in archives:
                archives[rel] = await self.fetch_url(f"{NVIDIA_REDIST}/{rel}")
            with zipfile.ZipFile(io.BytesIO(archives[rel])) as zf:
                top = zf.namelist()[0].split("/", 1)[0]
                out[dest] = zf.read(f"{top}/{member}")
        return out

    async def _resolve_redist_index(self, label: str) -> str:
        listing = (await self.fetch_url(f"{NVIDIA_REDIST}/")).decode("utf-8", "ignore")
        candidates = sorted(
            set(re.findall(rf"redistrib_{re.escape(label)}\.[0-9.]+\.json", listing)),
            key=lambda name: [int(part) for part in re.findall(r"\d+", name)],
        )
        if not candidates:
            raise KeyError(f"no CUDA redist for {label}")
        return f"{NVIDIA_REDIST}/{candidates[-1]}"

    async def _fetch_json(self, url: str) -> dict:
        import json

        return json.loads(await self.fetch_url(url))

    async def build(self, *, served_filename: str, cuda: str) -> bytes:
        rename = served_filename.startswith("triton-")
        source_url = await self.source_wheel_url(served_filename)
        wheel_bytes = await self.fetch_url(source_url)
        replacements: dict[str, bytes] = {}
        if needs_cuda13_patch(cuda):
            replacements = await self.cuda13_binaries(cuda)
        return _rebuild_wheel(
            wheel_bytes,
            served_filename=served_filename,
            replacements=replacements,
            rename_to_triton=rename,
        )


def _rebuild_wheel(
    wheel_bytes: bytes,
    *,
    served_filename: str,
    replacements: dict[str, bytes],
    rename_to_triton: bool,
) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(wheel_bytes))
    members = [info.filename for info in src.infolist()]

    old_dist = next(n for n in members if n.endswith(".dist-info/RECORD")).split("/", 1)[0]
    dist_stem = old_dist[: -len(".dist-info")]  # triton_windows-3.7.0.post26
    _old_name, _, version = dist_stem.rpartition("-")  # triton_windows, 3.7.0.post26
    new_dist = old_dist
    if rename_to_triton:
        new_dist = f"triton-{version}.dist-info"

    def remap(path: str) -> str:
        if rename_to_triton and path.startswith(old_dist + "/"):
            return new_dist + path[len(old_dist):]
        return path

    out = io.BytesIO()
    record_lines: list[str] = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            name = info.filename
            if name.endswith(".dist-info/RECORD"):
                continue  # rebuilt last
            data = replacements.get(name)
            if data is None:
                data = src.read(name)
            if name.endswith(".dist-info/METADATA") and rename_to_triton:
                data = re.sub(rb"(?m)^Name:\s*triton-windows\s*$", b"Name: triton", data)
            new_path = remap(name)
            dst.writestr(new_path, data)
            record_lines.append(_record_line(new_path, data))
        record_path = f"{new_dist}/RECORD"
        record_lines.append(f"{record_path},,")
        dst.writestr(record_path, "\n".join(record_lines) + "\n")
    del served_filename
    return out.getvalue()
