"""Regression coverage for PR #68: custom-node routes queued before startup."""

import asyncio
from unittest.mock import Mock

import pytest
from aiohttp import web

from comfy.app.assets.manager import AssetManager
from comfy.cli_args_types import Configuration
from comfy.cmd.server import PromptServer
from comfy.component_model.plugins import prompt_server_instance_routes
from comfy.execution_context import context_configuration


@pytest.mark.parametrize("method", ["get", "post"])
async def test_deferred_custom_node_route(aiohttp_client, monkeypatch, tmp_path, method):
    monkeypatch.setattr(PromptServer, "instance", None)
    monkeypatch.setattr(prompt_server_instance_routes, "routes", [])

    @getattr(prompt_server_instance_routes, method)("/custom-node/probe")
    async def handler(request):
        return web.json_response({"method": request.method})

    assert len(prompt_server_instance_routes.routes) == 1
    configuration = Configuration(
        front_end_root=str(tmp_path),
        user_directory=str(tmp_path / "user"),
    )
    with context_configuration(configuration):
        server = PromptServer(
            asyncio.get_running_loop(),
            asset_manager=Mock(spec=AssetManager, enabled=False),
        )
        server.add_routes()

        assert prompt_server_instance_routes.routes == []
        client = await aiohttp_client(server.app)
        for path in ("/custom-node/probe", "/api/custom-node/probe"):
            response = await client.request(method.upper(), path)
            assert response.status == 200
            assert await response.json() == {"method": method.upper()}
