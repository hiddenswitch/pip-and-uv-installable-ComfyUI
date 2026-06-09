from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from comfy.component_model.configuration import Configuration
from comfy.custom_node_facade import server as facade_server


class _FakeRegistry:
    calls = 0

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def list_projects(self):
        type(self).calls += 1
        return []


class _FakeSnapshotRegistry(_FakeRegistry):
    snapshot_uri = None

    def __init__(self, *args, snapshot_uri=None, **kwargs):
        del args, kwargs
        type(self).snapshot_uri = snapshot_uri


class _FakeBuilder:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class _FakeProxy:
    def __init__(self, *cuda_variants: str):
        self.cuda_variants = set(cuda_variants)
        self.render_calls: list[str] = []

    def supports_cuda(self, cuda: str) -> bool:
        return cuda in self.cuda_variants

    async def render_index(self, session, cuda: str) -> str:
        del session
        self.render_calls.append(cuda)
        return f"<html><body>{cuda}</body></html>"


async def test_serve_pip_warms_registry_before_reporting_ready(monkeypatch):
    _FakeRegistry.calls = 0
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)

    app = facade_server.create_facade_app(configuration=Configuration())
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        assert _FakeRegistry.calls == 1

        live = await client.get("/livez")
        assert live.status == 200
        assert await live.json() == {"ok": True, "live": True, "ready": True}

        ready = await client.get("/readyz")
        assert ready.status == 200
        assert await ready.json() == {"ok": True, "live": True, "ready": True}

        health = await client.get("/healthz")
        assert health.status == 200
        assert await health.json() == {"ok": True, "live": True, "ready": True}
    finally:
        await client.close()


async def test_cuda_simple_index_filters_proxy_projects_by_supported_cuda(monkeypatch):
    _FakeRegistry.calls = 0
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(
        facade_server,
        "PYPI_PROXY_INDEX",
        {
            "flash-attn": _FakeProxy("cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu131", "cu132"),
            "flash-attn-3": _FakeProxy("cu124", "cu126", "cu128", "cu129", "cu130", "cu132"),
            "sageattention": _FakeProxy("cu128", "cu130"),
            "nunchaku": _FakeProxy("cu128", "cu130"),
        },
    )

    app = facade_server.create_facade_app(configuration=Configuration())
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        cu126 = await client.get("/simple/cu126/")
        assert cu126.status == 200
        cu126_html = await cu126.text()
        assert "/simple/cu126/flash-attn/" in cu126_html
        assert "/simple/cu126/flash-attn-3/" in cu126_html
        assert "sageattention" not in cu126_html
        assert "nunchaku" not in cu126_html

        cu131 = await client.get("/simple/cu131/")
        assert cu131.status == 200
        cu131_html = await cu131.text()
        assert "/simple/cu131/flash-attn/" in cu131_html
        assert "flash-attn-3" not in cu131_html
        assert "sageattention" not in cu131_html
        assert "nunchaku" not in cu131_html

        cu130 = await client.get("/simple/cu130/")
        assert cu130.status == 200
        cu130_html = await cu130.text()
        assert "/simple/flash-attn/" in cu130_html
        assert "/simple/flash-attn-3/" in cu130_html
        assert "/simple/sageattention/" in cu130_html
        assert "/simple/nunchaku/" in cu130_html
    finally:
        await client.close()


async def test_cuda_project_pages_reject_unsupported_proxy_cuda(monkeypatch):
    _FakeRegistry.calls = 0
    flash_attn = _FakeProxy("cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu131", "cu132")
    flash_attn_3 = _FakeProxy("cu124", "cu126", "cu128", "cu129", "cu130", "cu132")
    sageattention = _FakeProxy("cu128", "cu130")
    nunchaku = _FakeProxy("cu128", "cu130")
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(
        facade_server,
        "PYPI_PROXY_INDEX",
        {
            "flash-attn": flash_attn,
            "flash-attn-3": flash_attn_3,
            "sageattention": sageattention,
            "nunchaku": nunchaku,
        },
    )

    app = facade_server.create_facade_app(configuration=Configuration())
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        assert (await client.get("/simple/cu126/flash-attn/")).status == 200
        assert (await client.get("/simple/cu126/flash-attn-3/")).status == 200
        assert (await client.get("/simple/cu126/sageattention/")).status == 404
        assert (await client.get("/simple/cu126/nunchaku/")).status == 404
        assert (await client.get("/simple/cu131/flash-attn-3/")).status == 404
        assert (await client.get("/simple/cu130/sageattention/")).status == 200
        assert (await client.get("/simple/cu130/nunchaku/")).status == 200

        assert flash_attn.render_calls == ["cu126"]
        assert flash_attn_3.render_calls == ["cu126"]
        assert sageattention.render_calls == ["cu130"]
        assert nunchaku.render_calls == ["cu130"]
    finally:
        await client.close()


async def test_serve_pip_uses_snapshot_registry_when_configured(monkeypatch):
    _FakeRegistry.calls = 0
    _FakeSnapshotRegistry.calls = 0
    _FakeSnapshotRegistry.snapshot_uri = None
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "SnapshotFacadeRegistry", _FakeSnapshotRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)

    configuration = Configuration()
    configuration.pip_facade_snapshot_uri = "file:///tmp/registry.sqlite.xz"

    app = facade_server.create_facade_app(configuration=configuration)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        assert _FakeRegistry.calls == 0
        assert _FakeSnapshotRegistry.calls == 1
        assert _FakeSnapshotRegistry.snapshot_uri == configuration.pip_facade_snapshot_uri
    finally:
        await client.close()
