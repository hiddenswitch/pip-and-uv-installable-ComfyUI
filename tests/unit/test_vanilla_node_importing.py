from __future__ import annotations

from comfy.nodes.vanilla_node_importing import _PromptServerStub


def test_prompt_server_stub_has_app_router_shape():
    stub = _PromptServerStub()

    assert stub.app.router.frozen is True
    stub.app.add_routes([])

