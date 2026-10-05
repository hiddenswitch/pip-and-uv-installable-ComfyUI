"""Tests for --offline, --disable-partner-nodes and the deprecated --disable-api-nodes"""

from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import default_configuration
from comfy.cmd import cli
from comfy.cmd import server
from comfy.execution_context import context_configuration


async def pong(request):
    return web.Response(text="pong")


def parse_args(*flags):
    config = cli._build_config({flag.removeprefix("--").replace("-", "_"): True for flag in flags})
    return [str(config.offline), str(config.disable_partner_nodes), str(config.disable_api_nodes)]


@pytest.mark.parametrize("argv,expected", [
    ([], ["False", "False", "False"]),
    (["--disable-partner-nodes"], ["False", "True", "False"]),
    (["--offline"], ["True", "True", "False"]),
    (["--disable-api-nodes"], ["True", "True", "True"]),
    (["--disable-partner-nodes", "--offline"], ["True", "True", "False"]),
])
def test_arg_parsing(argv, expected):
    assert parse_args(*argv) == expected


def test_disable_partner_nodes_skips_partner_nodes():
    from comfy.nodes.package import import_all_nodes_in_workspace

    config = default_configuration()
    config.disable_all_custom_nodes = True
    config.disable_partner_nodes = True
    try:
        with context_configuration(config):
            exported = import_all_nodes_in_workspace()
            assert "IdeogramTextToImageApi" not in exported.NODE_CLASS_MAPPINGS
            assert "KSampler" in exported.NODE_CLASS_MAPPINGS
    finally:
        # the node registry is shared; later tests expect the default set
        import_all_nodes_in_workspace()


@pytest.mark.asyncio
@pytest.mark.parametrize("offline,expect_csp", [
    (False, False),
    (True, True),
])
async def test_csp_header(offline, expect_csp):
    config = default_configuration()
    config.offline = offline
    with context_configuration(config):
        prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.app.router.add_get("/ping", pong)
    async with TestClient(TestServer(prompt_server.app)) as client:
        resp = await client.get("/ping")
        assert resp.status == 200
        csp = resp.headers.get("Content-Security-Policy")
    if expect_csp:
        assert "connect-src 'self' data:" in csp
    else:
        assert csp is None
