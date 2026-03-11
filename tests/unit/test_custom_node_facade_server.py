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
