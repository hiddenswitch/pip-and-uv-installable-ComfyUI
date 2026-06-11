from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile

import pytest

from comfy.custom_node_facade import triton_wheels as tw


# --------------------------------------------------------------------------
# Fake aiohttp session
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, *, text=None, data=None, obj=None):
        self._text = text
        self._data = data
        self._obj = obj

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._text is None and self._data is None and self._obj is None:
            raise RuntimeError("404")

    async def text(self):
        return self._text

    async def read(self):
        return self._data

    async def json(self):
        return self._obj


class _FakeSession:
    def __init__(self, routes: dict):
        self.routes = routes

    def get(self, url):
        for key, resp in self.routes.items():
            if url == key:
                return resp
        return _FakeResp()


def _make_triton_windows_wheel() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("triton/__init__.py", "# triton\n")
        z.writestr("triton/backends/nvidia/bin/ptxas.exe", b"OLD_PTXAS_12")
        z.writestr("triton/backends/nvidia/include/cuda.h", b"// cuda 12 header")
        z.writestr("triton/backends/nvidia/lib/x64/cuda.lib", b"OLD_LIB_12")
        z.writestr(
            "triton_windows-3.7.0.post26.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: triton-windows\nVersion: 3.7.0.post26\n",
        )
        z.writestr("triton_windows-3.7.0.post26.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        z.writestr("triton_windows-3.7.0.post26.dist-info/RECORD", "triton/__init__.py,,\n")
    return buf.getvalue()


def _make_redist_archive(top: str, members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, data in members.items():
            z.writestr(f"{top}/{path}", data)
    return buf.getvalue()


def _record_valid(wheel_bytes: bytes, dist_info: str) -> bool:
    z = zipfile.ZipFile(io.BytesIO(wheel_bytes))
    record = z.read(f"{dist_info}/RECORD").decode()
    for line in record.splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(",", 2)
        if len(parts) == 3 and parts[1].startswith("sha256="):
            got = base64.urlsafe_b64encode(hashlib.sha256(z.read(parts[0])).digest()).rstrip(b"=").decode()
            if got != parts[1][7:]:
                return False
    return True


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def test_cuda_variant_helpers():
    assert tw.cuda_major("cu130") == 13
    assert tw.cuda_major("cu128") == 12
    assert tw.cuda_major("cu130torch2.12") == 13
    assert tw.cuda_major("nonsense") is None
    assert tw.cuda_redist_label("cu131") == "13.1"
    assert tw.cuda_redist_label("cu130") == "13.0"
    assert tw.needs_cuda13_patch("cu130")
    assert not tw.needs_cuda13_patch("cu128")


def test_triton_proxy_supports_only_plain_cuda():
    proxy = tw.TritonProxySpec(name="triton", rename_to_triton=True)
    assert proxy.supports_cuda("cu130")
    assert proxy.supports_cuda("cu128")
    assert not proxy.supports_cuda("cu130torch2.12")
    assert not proxy.supports_cuda("flash-attn")


# --------------------------------------------------------------------------
# Proxy index rendering
# --------------------------------------------------------------------------
async def test_triton_index_lists_linux_pypi_and_windows_patched():
    session = _FakeSession({
        "https://pypi.org/simple/triton/": _FakeResp(text=(
            '<a href="https://files/triton-3.7.0-cp312-cp312-manylinux_2_28_x86_64.whl">a</a>'
            '<a href="https://files/triton-3.7.0-cp312-cp312-win_amd64.whl">b</a>'
        )),
        "https://pypi.org/simple/triton-windows/": _FakeResp(text=(
            '<a href="https://files/triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl">c</a>'
        )),
    })
    proxy = tw.TritonProxySpec(name="triton", rename_to_triton=True)
    body = await proxy.render_index(session, "cu130")
    # Linux manylinux wheel served straight from PyPI
    assert "https://files/triton-3.7.0-cp312-cp312-manylinux_2_28_x86_64.whl" in body
    # No upstream PyPI win_amd64 `triton` wheel (there isn't one) leaked through
    assert "files/triton-3.7.0-cp312-cp312-win_amd64.whl" not in body
    # Windows wheel renamed to triton and routed through our patcher
    assert "/packages/triton/cu130/triton-3.7.0.post26-cp312-cp312-win_amd64.whl" in body


async def test_triton_windows_index_keeps_name_and_routes_through_patcher():
    session = _FakeSession({
        "https://pypi.org/simple/triton-windows/": _FakeResp(text=(
            '<a href="https://files/triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl">c</a>'
        )),
    })
    proxy = tw.TritonProxySpec(name="triton-windows", rename_to_triton=False)
    body = await proxy.render_index(session, "cu131")
    assert "/packages/triton/cu131/triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl" in body
    # triton-windows does not serve linux pypi triton wheels
    assert "manylinux" not in body


# --------------------------------------------------------------------------
# Builder: download + patch + rename
# --------------------------------------------------------------------------
def _build_session(wheel_bytes: bytes) -> _FakeSession:
    nvcc = _make_redist_archive("cuda_nvcc-windows-x86_64-13.0.1-archive", {"bin/ptxas.exe": b"NEW_PTXAS_13"})
    cudart = _make_redist_archive(
        "cuda_cudart-windows-x86_64-13.0.1-archive",
        {"include/cuda.h": b"// cuda 13 header", "lib/x64/cuda.lib": b"NEW_LIB_13"},
    )
    redist_index = {
        "cuda_nvcc": {"windows-x86_64": {"relative_path": "cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-13.0.1-archive.zip"}},
        "cuda_cudart": {"windows-x86_64": {"relative_path": "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.0.1-archive.zip"}},
    }
    return _FakeSession({
        "https://pypi.org/pypi/triton-windows/json": _FakeResp(obj={
            "releases": {"3.7.0.post26": [{
                "filename": "triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl",
                "url": "https://files/tw.whl",
            }]},
        }),
        "https://files/tw.whl": _FakeResp(data=wheel_bytes),
        "https://developer.download.nvidia.com/compute/cuda/redist/": _FakeResp(
            data=b'redistrib_13.0.0.json redistrib_13.0.1.json redistrib_13.1.0.json'),
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.1.json":
            _FakeResp(data=json.dumps(redist_index).encode()),
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-13.0.1-archive.zip":
            _FakeResp(data=nvcc),
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.0.1-archive.zip":
            _FakeResp(data=cudart),
    })


async def test_build_triton_renamed_and_cuda13_patched():
    session = _build_session(_make_triton_windows_wheel())
    builder = tw.TritonWheelBuilder(session)
    out = await builder.build(
        served_filename="triton-3.7.0.post26-cp312-cp312-win_amd64.whl", cuda="cu130")
    z = zipfile.ZipFile(io.BytesIO(out))
    names = z.namelist()
    # renamed dist-info, METADATA Name rewritten
    assert "triton-3.7.0.post26.dist-info/METADATA" in names
    assert not any("triton_windows" in n for n in names)
    assert "Name: triton\n" in z.read("triton-3.7.0.post26.dist-info/METADATA").decode()
    # CUDA-13 binaries swapped in
    assert z.read("triton/backends/nvidia/bin/ptxas.exe") == b"NEW_PTXAS_13"
    assert z.read("triton/backends/nvidia/lib/x64/cuda.lib") == b"NEW_LIB_13"
    assert _record_valid(out, "triton-3.7.0.post26.dist-info")


async def test_build_triton_windows_cu128_is_not_patched():
    session = _build_session(_make_triton_windows_wheel())
    builder = tw.TritonWheelBuilder(session)
    out = await builder.build(
        served_filename="triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl", cuda="cu128")
    z = zipfile.ZipFile(io.BytesIO(out))
    # cu128 (CUDA 12) keeps the original bundled toolchain and the real name
    assert z.read("triton/backends/nvidia/bin/ptxas.exe") == b"OLD_PTXAS_12"
    assert "triton_windows-3.7.0.post26.dist-info/METADATA" in z.namelist()
    assert _record_valid(out, "triton_windows-3.7.0.post26.dist-info")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
